"""GMA-55 slice 2: licensed OCR path (LightOnOCR-3-1B grounding on MLX, PaddleOCR-VL-1.6 fallback),
page images instead of text, escalate policy and --force-inspect."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from packet.backends import (
    EscalatingExtractor,
    FallbackExtractor,
    LightOnExtractor,
    OcrUnavailable,
    PaddleExtractor,
    default_extractors,
)
from packet.extract import NativeTextExtractor, Router
from packet.ocr_parse import parse_lighton_grounding, parse_paddle_blocks
from packet.pipeline import Pipeline
from packet.source import PdfPageSource
from packet.types import BlockKind, PageClass, PageRef, SourceKind

SCAN = PageRef(packet_id="p", index=3, width=612.0, height=792.0, page_class=PageClass.SCAN)
GROUNDING = """![header](50,20,950,40)
ACME Corp — Invoice
![title](80,70,920,140)
# Invoice 1001
![title](80,150,920,170)
## Line items
![text](80,175,920,225)
Revenue rose 20%.
![text+](80,230,920,260)
continued here.
![page_number](900,960,950,980)
Page 1 of 2
![footer](50,970,500,990)
Confidential
"""


# ---------- grounding / paddle output -> Block ----------

def test_parse_lighton_grounding_maps_labels_bbox_and_levels():
    blocks = parse_lighton_grounding(GROUNDING, SCAN)
    kinds = [b.kind for b in blocks]
    assert kinds == [BlockKind.HEADER, BlockKind.TITLE, BlockKind.HEADING, BlockKind.PARAGRAPH,
                     BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER, BlockKind.FOOTER]
    title = blocks[1]
    assert title.text == "Invoice 1001" and title.heading_level == 1
    assert blocks[2].text == "Line items" and blocks[2].heading_level == 2
    assert blocks[4].extra["continuation"] is True
    assert all(b.source == SourceKind.VLM and b.page == 3 for b in blocks)
    bb = title.bbox
    assert (round(bb.x0, 2), round(bb.y0, 2), round(bb.x1, 2)) == (48.96, 55.44, 563.04)
    assert [b.reading_order for b in blocks] == list(range(7))
    assert blocks[0].extra["label"] == "header"


def test_parse_lighton_plain_markdown_still_returns_blocks():
    blocks = parse_lighton_grounding("# Title\n\nSome body text.", SCAN)
    assert [b.kind for b in blocks] == [BlockKind.TITLE, BlockKind.PARAGRAPH]


def test_parse_lighton_empty_is_empty():
    assert parse_lighton_grounding("  \n", SCAN) == []


def test_parse_paddle_blocks_scales_pixels_to_points():
    raw = [{"label": "doc_title", "bbox": [100, 50, 500, 90], "text": "Lease"},
           {"label": "number", "bbox": [1000, 1600, 1100, 1630], "text": "3"},
           {"label": "mystery", "bbox": None, "text": "x"}]
    blocks = parse_paddle_blocks(raw, SCAN, dpi=200)
    assert [b.kind for b in blocks] == [BlockKind.TITLE, BlockKind.PAGE_NUMBER, BlockKind.UNKNOWN]
    assert blocks[0].bbox.x0 == pytest.approx(36.0) and blocks[2].bbox is None
    assert blocks[0].source == SourceKind.VLM


# ---------- page images ----------

def _image_only_pdf(tmp_path: Path, pages: int = 1) -> Path:
    img = Image.new("RGB", (850, 1100), "white")
    out = tmp_path / "scan.pdf"
    img.save(out, "PDF", save_all=True, append_images=[img] * (pages - 1), resolution=100)
    return out


def test_source_renders_page_images_for_ocr(tmp_path):
    pdf = _image_only_pdf(tmp_path)
    with PdfPageSource(pdf) as src:
        img = src.render(0)
    assert max(img.size) == 1540


class _SpyOcr:
    needs_image = True

    def __init__(self, blocks=None, exc=None):
        self.calls, self.blocks, self.exc = [], blocks, exc

    def supports(self, page_class):
        return True

    def extract(self, ref, text, image=None):
        self.calls.append((ref.index, image() if callable(image) else image))
        if self.exc:
            raise self.exc
        return list(self.blocks or [])


def test_router_passes_lazy_image_only_to_image_extractors():
    rendered = []
    spy = _SpyOcr(blocks=["b"])
    out = Router([spy]).extract(SCAN, "", image=lambda: rendered.append(1) or "IMG")
    assert out == ["b"] and spy.calls == [(3, "IMG")] and rendered == [1]
    # native extractor keeps the old 2-arg signature and never triggers a render
    rendered.clear()
    text_ref = SCAN.model_copy(update={"page_class": PageClass.TEXT})
    Router([NativeTextExtractor()]).extract(text_ref, "hello", image=lambda: rendered.append(1))
    assert rendered == []


# ---------- fallback ----------

def test_fallback_uses_paddle_when_lighton_unavailable():
    p, f = _SpyOcr(exc=OcrUnavailable("no mlx")), _SpyOcr(blocks=["paddle"])
    ext = FallbackExtractor(p, f)
    assert ext.extract(SCAN, "", image=lambda: "IMG") == ["paddle"]
    assert ext.last_used is f


def test_fallback_used_when_primary_returns_nothing():
    p, f = _SpyOcr(blocks=[]), _SpyOcr(blocks=["paddle"])
    assert FallbackExtractor(p, f).extract(SCAN, "", image=lambda: "IMG") == ["paddle"]


def test_both_engines_unavailable_raises_ocr_unavailable():
    ext = FallbackExtractor(_SpyOcr(exc=OcrUnavailable("a")), _SpyOcr(exc=OcrUnavailable("b")))
    with pytest.raises(OcrUnavailable):
        ext.extract(SCAN, "", image=lambda: "IMG")


def test_default_engines_are_lighton_grounding_mlx_then_paddle():
    ext = default_extractors()[0]
    assert isinstance(ext, EscalatingExtractor)
    ocr = ext.ocr
    assert isinstance(ocr.primary, LightOnExtractor) and isinstance(ocr.fallback, PaddleExtractor)
    assert ocr.primary.mode == "grounding" and ocr.primary.backend == "mlx"
    assert ocr.primary.model == "lightonai/LightOnOCR-3-1B"


def test_lighton_raises_unavailable_without_mlx(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("mlx"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(OcrUnavailable):
        LightOnExtractor().extract(SCAN, "", image=lambda: Image.new("RGB", (10, 10)))


# ---------- escalate policy ----------

def _esc(ocr_blocks, force=False):
    ocr = _SpyOcr(blocks=ocr_blocks)
    return EscalatingExtractor(native=NativeTextExtractor(), ocr=ocr, force_inspect=force), ocr


def test_text_page_never_goes_to_ocr():
    esc, ocr = _esc(["ocr"])
    ref = SCAN.model_copy(update={"page_class": PageClass.TEXT})
    out = esc.extract(ref, "A trustworthy text layer with plenty of characters in it.", image=lambda: "IMG")
    assert ocr.calls == [] and out and out[0].source == SourceKind.NATIVE


@pytest.mark.parametrize("pc", [PageClass.SCAN, PageClass.SUSPECT_ENCODING])
def test_scan_pages_go_to_ocr(pc):
    esc, _ocr = _esc(["ocr"])
    assert esc.extract(SCAN.model_copy(update={"page_class": pc}), "", image=lambda: "IMG") == ["ocr"]


def test_mixed_page_escalates_to_ocr():
    esc, _ocr = _esc(["ocr"])
    ref = SCAN.model_copy(update={"page_class": PageClass.MIXED})
    assert esc.extract(ref, "p. 3", image=lambda: "IMG") == ["ocr"]


def test_force_inspect_sends_text_pages_to_ocr():
    esc, _ocr = _esc(["ocr"], force=True)
    ref = SCAN.model_copy(update={"page_class": PageClass.TEXT})
    assert esc.extract(ref, "plenty of text " * 10, image=lambda: "IMG") == ["ocr"]


def test_ocr_unavailable_on_mixed_keeps_native_text():
    esc = EscalatingExtractor(native=NativeTextExtractor(), ocr=_SpyOcr(exc=OcrUnavailable("x")))
    ref = SCAN.model_copy(update={"page_class": PageClass.MIXED})
    out = esc.extract(ref, "p. 3", image=lambda: "IMG")
    assert [b.text for b in out] == ["p. 3"]
    assert esc.unavailable  # surfaced, not silent


def test_pipeline_routes_scanned_pdf_through_ocr_with_page_image(tmp_path):
    pdf = _image_only_pdf(tmp_path, pages=2)
    from packet.types import Block
    seen = []

    class _Fake:
        needs_image = True

        def supports(self, pc):
            return True

        def extract(self, ref, text, image=None):
            img = image()
            seen.append(img.size)
            return [Block(id=f"x{ref.index}", page=ref.index, kind=BlockKind.PARAGRAPH, text="scanned words",
                          source=SourceKind.VLM)]

    pipe = Pipeline(extractors=[EscalatingExtractor(native=NativeTextExtractor(), ocr=_Fake())])
    packet = pipe.parse(pdf)
    assert len(seen) == 2 and all(max(s) == 1540 for s in seen)
    texts = [b.text for d in packet.documents for s in d.sections for b in s.blocks]
    assert "scanned words" in " ".join(texts)


def test_cli_force_inspect_flag(monkeypatch, tmp_path):
    from packet import cli
    captured = {}
    real = cli.default_extractors

    def spy(force_inspect=False):
        captured["force"] = force_inspect
        return real(force_inspect=force_inspect)

    monkeypatch.setattr(cli, "default_extractors", spy)
    pdf = _image_only_pdf(tmp_path)
    monkeypatch.setenv("PACKET_OCR", "off")
    assert cli.main(["parse", str(pdf), "-o", str(tmp_path / "o.json"), "--force-inspect"]) == 0
    assert captured["force"] is True


# ---------- real model (opt-in) ----------

SCANNED = os.environ.get("PACKET_REAL_SCAN_PDF")


@pytest.mark.skipif(not SCANNED, reason="set PACKET_REAL_SCAN_PDF to run the real LightOnOCR-3 proof")
def test_real_scanned_page_is_not_empty():
    packet = Pipeline(extractors=default_extractors()).parse(SCANNED)
    words = " ".join(b.text for d in packet.documents for s in d.sections for b in s.blocks)
    assert len(words.split()) > 30
