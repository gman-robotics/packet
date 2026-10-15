"""GMA-218 (a): keep OCR models loaded across pages, extractors and files."""

import json

import pytest

from packet import cli
from packet.backends import (
    EscalatingExtractor,
    LightOnExtractor,
    OcrUnavailable,
    PaddleExtractor,
    clear_engine_cache,
)
from packet.eval.pdfutil import write_text_pdf
from packet.types import PageClass, PageRef

SCAN = PageRef(packet_id="p", index=0, page_class=PageClass.SCAN, width=612, height=792)


def _count_loads(monkeypatch, cls, value=("M", "P", (lambda *a, **k: "", "PROMPT"))):
    calls = []

    def load(self):
        calls.append(1)
        return value

    monkeypatch.setattr(cls, "_load_impl", load)
    return calls


def test_model_is_loaded_once_per_process_across_extractor_instances(monkeypatch):
    calls = _count_loads(monkeypatch, LightOnExtractor)
    for _ in range(3):
        LightOnExtractor().extract(SCAN, "", image=lambda: "IMG")
    assert calls == [1]


def test_cache_is_keyed_by_engine_and_model(monkeypatch):
    light = _count_loads(monkeypatch, LightOnExtractor)
    paddle = _count_loads(monkeypatch, PaddleExtractor, value=object())
    LightOnExtractor()._load()
    PaddleExtractor()._load()
    other = LightOnExtractor()
    other.revision = "other-rev"
    other._load()
    assert light == [1, 1] and paddle == [1]


def test_failed_load_is_shared_and_clear_resets(monkeypatch):
    calls = []

    def boom(self):
        calls.append(1)
        raise OcrUnavailable("offline")

    monkeypatch.setattr(LightOnExtractor, "_load_impl", boom)
    for _ in range(2):
        with pytest.raises(OcrUnavailable):
            LightOnExtractor()._load()
    assert calls == [1]
    clear_engine_cache()
    with pytest.raises(OcrUnavailable):
        LightOnExtractor()._load()
    assert calls == [1, 1]


def test_ocr_page_timings_are_recorded():
    class _Ocr:
        needs_image = True

        def supports(self, pc):
            return True

        def extract(self, ref, text, image=None):
            return []

    esc = EscalatingExtractor(ocr=_Ocr())
    esc.extract(SCAN, "", image=lambda: "IMG")
    esc.extract(SCAN.model_copy(update={"index": 1}), "", image=lambda: "IMG")
    assert [t["page"] for t in esc.ocr_timings] == [0, 1]
    assert all(t["seconds"] >= 0 for t in esc.ocr_timings)


def test_parse_many_files_into_out_dir_with_one_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("PACKET_OCR", "off")
    a = write_text_pdf(tmp_path / "a.pdf", [["Alpha body.", "Page 1 of 1"]])
    b = write_text_pdf(tmp_path / "b.pdf", [["Beta body.", "Page 1 of 2"], ["More.", "Page 2 of 2"]])
    out = tmp_path / "out"
    timings = tmp_path / "t.jsonl"
    assert cli.main(["parse", str(a), str(b), "--out-dir", str(out), "--timings", str(timings)]) == 0
    assert json.loads((out / "a.json").read_text())["origin"]["page_count"] == 1
    assert json.loads((out / "b.json").read_text())["origin"]["page_count"] == 2
    rows = [json.loads(x) for x in timings.read_text().splitlines()]
    assert [r["file"] for r in rows] == ["a.pdf", "b.pdf"]
    assert rows[1]["pages"] == 2 and rows[1]["ok"] and "ocr_page_seconds" in rows[1]


def test_parse_many_files_needs_out_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PACKET_OCR", "off")
    a = write_text_pdf(tmp_path / "a.pdf", [["A."]])
    assert cli.main(["parse", str(a), str(a)]) == 2
    assert "--out-dir" in capsys.readouterr().err


def test_parse_many_keeps_going_after_a_bad_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PACKET_OCR", "off")
    good = write_text_pdf(tmp_path / "good.pdf", [["Fine."]])
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    timings = tmp_path / "t.jsonl"
    rc = cli.main(["parse", str(bad), str(good), "--out-dir", str(tmp_path / "o"), "--timings", str(timings)])
    assert rc == 1
    assert (tmp_path / "o" / "good.json").exists()
    rows = [json.loads(x) for x in timings.read_text().splitlines()]
    assert [r["ok"] for r in rows] == [False, True] and rows[0]["error"]
