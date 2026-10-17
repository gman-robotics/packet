"""GMA-55 slice 2: licensed OCR path (LightOnOCR-3-1B grounding on MLX, PaddleOCR-VL-1.6 fallback),
page images instead of text, escalate policy and --force-inspect."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image

pytestmark = pytest.mark.needs_ocr_extras

from packet.backends import (
    EscalatingExtractor,
    FallbackExtractor,
    LightOnExtractor,
    OCRConfig,
    OcrConfigError,
    OcrUnavailable,
    PaddleExtractor,
    build_ocr_extractor,
    default_extractors,
    default_ocr_extractor,
)
from packet.eval.run import run_fixture
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


def test_untagged_inline_trailing_page_label_splits():
    blocks = parse_lighton_grounding("Northwind Invoice Amount due: 12.00 Page 1 of 2", SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER]
    assert blocks[0].text == "Northwind Invoice Amount due: 12.00"
    assert blocks[1].text == "Page 1 of 2"


def test_mid_sentence_page_keyword_stays_body():
    blocks = parse_lighton_grounding("see page 1 of 2 for terms", SCAN)
    assert len(blocks) == 1 and blocks[0].kind == BlockKind.PARAGRAPH


def test_no_footer_leaves_lighton_block_unchanged():
    blocks = parse_lighton_grounding("Just body text without a page label.", SCAN)
    assert len(blocks) == 1 and blocks[0].kind == BlockKind.PARAGRAPH


def test_mid_sentence_page_number_stays_body():
    blocks = parse_lighton_grounding("See Page 3 for the appendix details.", SCAN)
    assert len(blocks) == 1 and "Page 3" in blocks[0].text


def test_tagged_page_number_and_footer_kinds():
    raw = """![text](80,175,920,225)
Body line
![page_number](800,960,950,980)
Page 1 of 2
![footer](50,970,500,990)
Confidential"""
    blocks = parse_lighton_grounding(raw, SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER, BlockKind.FOOTER]


def test_footer_tag_with_page_label_becomes_page_number():
    raw = """![footer](800,960,950,980)
Page 1 of 2"""
    blocks = parse_lighton_grounding(raw, SCAN)
    assert len(blocks) == 1 and blocks[0].kind == BlockKind.PAGE_NUMBER


@pytest.mark.parametrize(
    "text",
    ["Total\n12", "Invoice\n12"],
)
def test_untagged_total_or_invoice_with_digit_stays_body(text):
    blocks = parse_lighton_grounding(text, SCAN)
    assert len(blocks) == 1 and blocks[0].kind == BlockKind.PARAGRAPH


@pytest.mark.parametrize(
    "text",
    [
        "Section 1 of 3",
        "Table 1 of 2",
        "Chapter 2 of 5",
        "Step 1 of 3",
        "Qty 3/4",
        "Cell 12.00 1/2",
        "Figure 1/2",
        "2S 1/2",
        "see pp. 1 of 2",
        "pages 1 of 2",
    ],
)
def test_untagged_text_does_not_invent_page_numbers(text):
    blocks = parse_lighton_grounding(text, SCAN)
    assert len(blocks) == 1
    assert all(b.kind != BlockKind.PAGE_NUMBER for b in blocks)


def test_packet_ocr_model_rejected_for_paddle(monkeypatch):
    monkeypatch.setenv("PACKET_OCR", "paddle")
    monkeypatch.setenv("PACKET_OCR_MODEL", "other/model")
    with pytest.raises(OcrConfigError, match="PACKET_OCR_MODEL"):
        default_extractors()


def test_packet_ocr_model_rejected_on_legacy_chain(monkeypatch):
    monkeypatch.delenv("PACKET_OCR", raising=False)
    monkeypatch.setenv("PACKET_OCR_MODEL", "other/model")
    with pytest.raises(OcrConfigError, match="needs PACKET_OCR=lighton or auto"):
        default_extractors()


@pytest.mark.parametrize(
    "text",
    [
        "Steps\n1\nDo the thing",
        "Amount due: 12.00\n1",
        "Total\n12",
        "Effective\n2024",
    ],
)
def test_lighton_does_not_split_bare_digit_lines(text):
    blocks = parse_lighton_grounding(text, SCAN)
    assert all(b.kind != BlockKind.PAGE_NUMBER for b in blocks)


def test_untagged_multiline_page_label_on_own_line_splits_once():
    blocks = parse_lighton_grounding("Invoice\n12\nPage 1 of 2", SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER]
    assert blocks[0].text == "Invoice\n12"
    assert blocks[1].text == "Page 1 of 2"


def test_heading_keeps_kind_and_level_with_tags():
    raw = """![title](80,70,920,140)
# Northwind Invoice
![page_number](800,960,950,980)
Page 1 of 2"""
    blocks = parse_lighton_grounding(raw, SCAN)
    assert blocks[0].kind == BlockKind.TITLE and blocks[0].heading_level == 1
    assert blocks[0].text == "Northwind Invoice"
    assert blocks[1].kind == BlockKind.PAGE_NUMBER


def test_untagged_whole_page_label_becomes_page_number():
    blocks = parse_lighton_grounding("Page 1 of 2", SCAN)
    assert len(blocks) == 1 and blocks[0].kind == BlockKind.PAGE_NUMBER


def test_whole_page_label_never_becomes_document_title():
    from packet.assemble import Assembler

    blocks = parse_lighton_grounding("Page 1 of 2\n\nNorthwind Invoice Amount due: 12.00", SCAN)
    assert blocks[0].kind == BlockKind.PAGE_NUMBER
    assert blocks[1].kind == BlockKind.PARAGRAPH
    doc = Assembler().build_document("d", blocks)
    assert doc.title == "Northwind Invoice Amount due: 12.00"


def test_tagged_title_whole_page_label_is_page_number_not_title():
    raw = """![title](80,70,920,140)
# Page 1 of 2
![text](80,100,920,200)
Northwind Invoice Amount due: 12.00"""
    blocks = parse_lighton_grounding(raw, SCAN)
    assert blocks[0].kind == BlockKind.PAGE_NUMBER
    assert blocks[1].text == "Northwind Invoice Amount due: 12.00"


def test_markdown_title_whole_page_label_is_page_number():
    blocks = parse_lighton_grounding("# Page 1 of 2\n\nNorthwind Invoice Amount due: 12.00", SCAN)
    assert blocks[0].kind == BlockKind.PAGE_NUMBER
    assert blocks[1].text == "Northwind Invoice Amount due: 12.00"


def test_page_label_block_does_not_emit_title_top_boundary_signal():
    from packet.boundary import BoundaryDetector

    def _ref(i: int) -> PageRef:
        return PageRef(packet_id="p", index=i, width=612.0, height=792.0)

    det = BoundaryDetector()
    det.observe(
        _ref(0),
        parse_lighton_grounding(_lighton_real_page_text(1), _ref(0)),
    )
    sig = det.observe(
        _ref(1),
        parse_lighton_grounding("Page 1 of 2\n\nNorthwind Invoice Amount due: 12.00", _ref(1)),
    )
    assert sig is None or "title_top" not in sig.reason


def test_pre_tag_text_expanded_only_once():
    raw = """Intro Page 1 of 2 Page 2 of 2
![text](80,175,920,225)
More body"""
    blocks = parse_lighton_grounding(raw, SCAN)
    pn = [b for b in blocks if b.kind == BlockKind.PAGE_NUMBER]
    assert len(pn) == 1 and pn[0].text == "Page 2 of 2"


@pytest.mark.parametrize(
    "text",
    ["1. Pay rent Page 1 of 2", "- Item Page 1 of 2"],
)
def test_list_item_trailing_page_label_splits(text):
    blocks = parse_lighton_grounding(text, SCAN)
    assert [b.kind for b in blocks] == [BlockKind.LIST_ITEM, BlockKind.PAGE_NUMBER]


def test_tagged_list_does_not_split_trailing_page_label():
    raw = """![list](80,175,920,225)
1. Pay rent Page 1 of 2"""
    blocks = parse_lighton_grounding(raw, SCAN)
    assert len(blocks) == 1 and blocks[0].kind == BlockKind.LIST_ITEM


def test_cli_model_load_error_is_one_stderr_line(monkeypatch, capsys, tmp_path):
    from packet import cli
    from packet.backends import LightOnExtractor, OcrConfigError

    monkeypatch.setenv("PACKET_OCR", "lighton")
    monkeypatch.setenv("PACKET_OCR_MODEL", "missing/repo")
    monkeypatch.setattr("packet.backends._mlx_vlm_available", lambda: True)

    def boom(self):
        raise OcrConfigError("PACKET_OCR_MODEL='missing/repo': failed to load (404)")

    monkeypatch.setattr(LightOnExtractor, "_load_impl", boom)
    pdf = _image_only_pdf(tmp_path)
    with pytest.raises(SystemExit) as exited:
        cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json")])
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [ln for ln in captured.err.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("packet: PACKET_OCR_MODEL=")
    assert "ocr-mlx" not in lines[0]


def _tagged_lighton_page(body: str, page_label: str) -> str:
    return f"""![text](80,175,920,400)
{body}
![page_number](800,960,950,980)
{page_label}"""


_TAGGED_LIGHTON_PAGES = [
    _tagged_lighton_page("Northwind Invoice Amount due: 12.00", "Page 1 of 2"),
    _tagged_lighton_page("Thank you for your business.", "Page 2 of 2"),
    _tagged_lighton_page("Contoso Lease Premises: 1 Main St", "Page 1 of 2"),
    _tagged_lighton_page("Signatures follow.", "Page 2 of 2"),
]

_LIGHTON_REAL_DIR = Path(__file__).resolve().parent / "fixtures" / "lighton_real"
# SHA-256 of on-disk raw-pageN.txt (see tests/fixtures/lighton_real/SOURCE.txt).
_LIGHTON_REAL_SHA256 = [
    "662de4bef67cba1bc0e03890c6d9a003a08fb15e0420741279bdcaed2067b246",
    "109e3457dd2769de1fa4493813f71b5c2acaaf57c4cea9f2cfc22b68bb059143",
    "8e0109237b9b4285526136c85f1693c28e779f2a98ec4663677a75c3f379cab5",
    "a85e7b88a18b5de7a06700983ce15e38fd61b37ddd3de839bb6ac4e690bbeaa4",
]


def _load_lighton_real_fixture_bytes(page: int) -> bytes:
    data = (_LIGHTON_REAL_DIR / f"raw-page{page}.txt").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == _LIGHTON_REAL_SHA256[page]
    return data


def _lighton_real_page_text(page: int, *, keep_eof_lf: bool = True) -> str:
    data = _load_lighton_real_fixture_bytes(page)
    if not keep_eof_lf and data.endswith(b"\n"):
        data = data[:-1]
    return data.decode("ascii")


class _TaggedLightonPages:
    needs_image = True

    def supports(self, page_class):
        return True

    def extract(self, ref, text, image=None):
        return parse_lighton_grounding(_TAGGED_LIGHTON_PAGES[ref.index], ref)


class _BoRealLightonPages:
    needs_image = True

    def supports(self, page_class):
        return True

    def extract(self, ref, text, image=None):
        return parse_lighton_grounding(_lighton_real_page_text(ref.index), ref)


def test_lighton_real_fixture_files_match_expected_sha256():
    for page in range(4):
        _load_lighton_real_fixture_bytes(page)


def test_page1_of_2_trailing_split_on_text_block():
    blocks = parse_lighton_grounding("Amount due: 12.00 page1 of 2", SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER]
    assert blocks[1].text == "page1 of 2"


def test_bo_real_lighton_format_splits_body_and_page_number():
    from packet.boundary import keyword_page_label_fraction

    blocks = parse_lighton_grounding(_lighton_real_page_text(0), SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER]
    assert blocks[0].text == "Northwind Invoice Amount due: 12.00"
    assert blocks[1].text == "Page 1 of 2"
    assert keyword_page_label_fraction(blocks[1].text) == (1, 2)

    blocks2 = parse_lighton_grounding(_lighton_real_page_text(2), SCAN)
    assert blocks2[0].text == "Contoso Lease Premises: 1 Main St"
    assert blocks2[1].text == "Page 1 of 2"


@pytest.mark.parametrize("keep_eof_lf", [True, False])
def test_bo_real_page0_parse_with_and_without_trailing_lf(keep_eof_lf):
    blocks = parse_lighton_grounding(_lighton_real_page_text(0, keep_eof_lf=keep_eof_lf), SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER]
    assert blocks[0].text == "Northwind Invoice Amount due: 12.00"
    assert blocks[1].text == "Page 1 of 2"


@pytest.mark.parametrize("keep_eof_lf", [True, False])
def test_bo_real_all_pages_parse_with_and_without_trailing_lf(keep_eof_lf):
    from packet.boundary import keyword_page_label_fraction

    expected_fracs = [(1, 2), (2, 2), (1, 2), (2, 2)]
    for page, frac in enumerate(expected_fracs):
        blocks = parse_lighton_grounding(_lighton_real_page_text(page, keep_eof_lf=keep_eof_lf), SCAN)
        assert blocks[-1].kind == BlockKind.PAGE_NUMBER
        assert keyword_page_label_fraction(blocks[-1].text) == frac


def test_bo_real_page0_label_on_own_line_after_body():
    raw = "![text](75,49,246,110) Northwind Invoice Amount due: 12.00\nPage 1 of 2"
    blocks = parse_lighton_grounding(raw, SCAN)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PAGE_NUMBER]
    assert blocks[0].text == "Northwind Invoice Amount due: 12.00"
    assert blocks[1].text == "Page 1 of 2"


def test_bo_real_lighton_pages_split_multi_doc_fixture(tmp_path):
    from packet.boundary import keyword_page_label_fraction
    from packet.eval.fixtures import build_builtin

    fx = next(f for f in build_builtin(tmp_path) if f.name == "multi-doc")
    pipe = Pipeline(
        extractors=[
            EscalatingExtractor(
                native=NativeTextExtractor(),
                ocr=_BoRealLightonPages(),
                force_inspect=True,
            )
        ]
    )
    packet = pipe.parse(fx.pdf)
    assert packet.document_count() == 2 and packet.origin.page_count == 4
    spans = [(d.page_start, d.page_end) for d in packet.documents]
    assert spans == [(0, 1), (2, 3)]
    pn_blocks = [
        b for d in packet.documents for b in d.furniture if b.kind == BlockKind.PAGE_NUMBER
    ]
    assert len(pn_blocks) == 4
    assert [keyword_page_label_fraction(b.text) for b in pn_blocks] == [
        (1, 2),
        (2, 2),
        (1, 2),
        (2, 2),
    ]


def test_tagged_lighton_pages_split_multi_doc_fixture(tmp_path):
    from packet.eval.fixtures import build_builtin

    fx = next(f for f in build_builtin(tmp_path) if f.name == "multi-doc")
    pipe = Pipeline(
        extractors=[
            EscalatingExtractor(
                native=NativeTextExtractor(),
                ocr=_TaggedLightonPages(),
                force_inspect=True,
            )
        ]
    )
    packet = pipe.parse(fx.pdf)
    assert packet.document_count() == 2 and packet.origin.page_count == 4
    spans = [(d.page_start, d.page_end) for d in packet.documents]
    assert spans == [(0, 1), (2, 3)]


def test_cli_ocr_config_error_is_one_line_without_traceback(monkeypatch, capsys, tmp_path):
    from packet import cli
    from packet.eval.pdfutil import write_text_pdf

    monkeypatch.setenv("PACKET_OCR", "lighton")
    monkeypatch.setattr("packet.backends._mlx_vlm_available", lambda: False)
    pdf = write_text_pdf(tmp_path / "one.pdf", [["Hello world with enough text here."]])
    with pytest.raises(SystemExit) as exited:
        cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json")])
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [ln for ln in captured.err.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0] == (
        "packet: PACKET_OCR=lighton: mlx-vlm is not installed; "
        "install with pip install 'packet-dl[ocr-mlx]' on Apple silicon"
    )


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


def test_empty_primary_result_is_final_blank_page_does_not_load_paddle():
    p, f = _SpyOcr(blocks=[]), _SpyOcr(blocks=["paddle"])
    assert FallbackExtractor(p, f).extract(SCAN, "", image=lambda: "IMG") == []
    assert f.calls == []


def test_fallback_used_when_primary_crashes():
    p, f = _SpyOcr(exc=RuntimeError("oom")), _SpyOcr(blocks=["paddle"])
    assert FallbackExtractor(p, f).extract(SCAN, "", image=lambda: "IMG") == ["paddle"]


@pytest.mark.parametrize("a,b", [(RuntimeError("crash"), OcrUnavailable("none")),
                                 (OcrUnavailable("none"), RuntimeError("crash"))])
def test_no_engine_result_raises_never_silent_empty(a, b):
    ext = FallbackExtractor(_SpyOcr(exc=a), _SpyOcr(exc=b))
    with pytest.raises(OcrUnavailable, match="crash"):
        ext.extract(SCAN, "", image=lambda: "IMG")


def test_failed_load_is_cached_not_retried_per_page(monkeypatch):
    calls = []

    def boom(self):
        calls.append(1)
        raise OcrUnavailable("offline")

    monkeypatch.setattr(LightOnExtractor, "_load_impl", boom)
    ext = LightOnExtractor()
    for _ in range(3):
        with pytest.raises(OcrUnavailable):
            ext.extract(SCAN, "", image=lambda: "IMG")
    assert calls == [1]


def test_lighton_extract_with_mocked_model(monkeypatch):
    ext = LightOnExtractor()
    seen = {}

    def gen(model, proc, prompt, image, **kw):
        seen["prompt"], seen["image"] = prompt, image
        return GROUNDING

    monkeypatch.setattr(LightOnExtractor, "_load_impl", lambda self: ("M", "P", (gen, "PROMPT")))
    blocks = ext.extract(SCAN, "", image=lambda: "IMG")
    assert seen == {"prompt": "PROMPT", "image": ["IMG"]} and len(blocks) == 7


def test_paddle_extract_with_mocked_pipeline(monkeypatch):
    class _Res:
        json = {"res": {"parsing_res_list": [  # noqa: RUF012
            {"block_label": "doc_title", "block_bbox": [0, 0, 100, 20], "block_content": "Lease"}]}}

    class _Pipe:
        def predict(self, path):
            assert Path(path).exists()
            return [_Res()]

    monkeypatch.setattr(PaddleExtractor, "_load_impl", lambda self: _Pipe())
    img = Image.new("RGB", (1190, 1540), "white")  # 612pt wide -> ~140 dpi
    blocks = PaddleExtractor().extract(SCAN, "", image=lambda: img)
    assert [b.text for b in blocks] == ["Lease"] and blocks[0].kind == BlockKind.TITLE
    assert blocks[0].bbox.x1 == pytest.approx(100 * 612 / 1190)


def test_mixed_page_with_empty_ocr_keeps_native_text():
    esc, _ocr = _esc([])
    ref = SCAN.model_copy(update={"page_class": PageClass.MIXED})
    assert [b.text for b in esc.extract(ref, "p. 3", image=lambda: "IMG")] == ["p. 3"]


def test_parse_keeps_text_before_first_tag_and_clamps_bbox():
    blocks = parse_lighton_grounding("Stray preamble\n![text](-5,0,1200,50)\nBody", SCAN)
    assert [b.text for b in blocks] == ["Stray preamble", "Body"]
    assert blocks[1].bbox.x0 == 0 and blocks[1].bbox.x1 == pytest.approx(612.0)


def test_both_engines_unavailable_raises_ocr_unavailable():
    ext = FallbackExtractor(_SpyOcr(exc=OcrUnavailable("a")), _SpyOcr(exc=OcrUnavailable("b")))
    with pytest.raises(OcrUnavailable):
        ext.extract(SCAN, "", image=lambda: "IMG")


def test_pipeline_default_extractor_is_native_text_only():
    extractors = Pipeline().router.extractors
    assert len(extractors) == 1 and type(extractors[0]) is NativeTextExtractor


def test_run_fixture_still_uses_default_pipeline():
    import inspect

    assert "pipeline or Pipeline()" in inspect.getsource(run_fixture)


def test_auto_resolve_uses_paddle_without_importing_mlx(monkeypatch):
    import builtins
    import importlib.util

    real_find = importlib.util.find_spec
    real_import = builtins.__import__

    def find_spec(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx"):
            return None
        return real_find(name, *args, **kwargs)

    def fake_import(name, *args, **kwargs):
        if name in {"vllm", "transformers"} or name.startswith(("vllm.", "transformers.")):
            raise ImportError(name)
        if name == "mlx_vlm" or name.startswith("mlx"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("packet.backends._paddle_available", lambda: True)
    ext = build_ocr_extractor(OCRConfig(engine="auto"))
    assert isinstance(ext, PaddleExtractor)
    assert "mlx_vlm" not in sys.modules


def test_explicit_lighton_missing_mlx_raises(monkeypatch):
    monkeypatch.setattr("packet.backends._mlx_vlm_available", lambda: False)
    with pytest.raises(OcrConfigError, match="mlx-vlm"):
        build_ocr_extractor(OCRConfig(engine="lighton"))


def test_default_ocr_extractor_without_config_is_unchanged_chain():
    ext = default_ocr_extractor()
    assert isinstance(ext, FallbackExtractor)
    assert isinstance(ext.primary, LightOnExtractor) and isinstance(ext.fallback, PaddleExtractor)
    assert ext.primary.model == "lightonai/LightOnOCR-3-1B"


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


@pytest.mark.needs_models
@pytest.mark.skipif(not SCANNED, reason="set PACKET_REAL_SCAN_PDF to run the real LightOnOCR-3 proof")
def test_real_scanned_page_is_not_empty():
    packet = Pipeline(extractors=default_extractors()).parse(SCANNED)
    words = " ".join(b.text for d in packet.documents for s in d.sections for b in s.blocks)
    assert len(words.split()) > 30
