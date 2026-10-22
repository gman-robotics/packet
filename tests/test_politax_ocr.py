"""GMA-275: optional PoliTax OCR cache (behavior, fake recognizer, no OCR extras)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from packet.eval import politax
from packet.eval import politax_ocr as po
from packet.eval.pdfutil import write_blank_scan_pdf, write_text_pdf

ENGINE = po.EngineId("lighton-mlx", "m/x", "rev1", "grounding", 4096, 1540)


def _doc(path: Path, subject: str = "trump") -> politax.PoliTaxDoc:
    from packet.source import PdfPageSource

    with PdfPageSource(path) as src:
        n = src.page_count
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return politax.PoliTaxDoc(subject, path.name, path.name, "https://x.invalid", n, sha, "t")


@pytest.fixture()
def corpus(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    scan = write_blank_scan_pdf(data / "scan.pdf", 3)
    mixed = write_text_pdf(
        data / "mixed.pdf",
        [["x" * 60], ["short"], []],  # page 0 native text, pages 1-2 need OCR
    )
    return data, [_doc(scan), _doc(mixed)]


class FakeRecognizer:
    def __init__(self, tokens: int = 10, fail_at: int | None = None):
        self.calls: list[tuple[str, int]] = []
        self.tokens = tokens
        self.fail_at = fail_at

    def __call__(self, plan: po.PagePlan):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("engine boom")
        self.calls.append((plan.pdf_sha256, plan.index))
        return f"<raw {plan.file_name} {plan.index}>", self.tokens


def _page_files(data: Path) -> list[Path]:
    return sorted(p for p in (data / "ocr").rglob("p*.json"))


def test_native_chars_strips_whitespace_like_the_census():
    assert po.native_chars("   " + "a" * 30 + " " * 20 + "\n") == 30
    assert po.native_chars(None) == 0
    assert po.OCR_MIN_NATIVE_CHARS == 40


def test_plan_selects_only_pages_under_40_native_chars(corpus):
    data, docs = corpus
    plans, native = po.plan_pages(docs, data)
    picked = {(p.file_name, p.index) for p in plans}
    assert picked == {("scan.pdf", 0), ("scan.pdf", 1), ("scan.pdf", 2),
                      ("mixed.pdf", 1), ("mixed.pdf", 2)}
    assert native == 1
    assert all(p.native_chars < 40 for p in plans)


def test_plan_keys_by_manifest_sha_and_filters(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data, file="mixed.pdf")
    assert {p.pdf_sha256 for p in plans} == {docs[1].sha256}
    assert po.plan_pages(docs, data, subject="obama")[0] == []


def test_run_writes_one_record_per_page_and_rerun_is_a_noop(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    fake = FakeRecognizer()
    s1 = po.run(plans, fake, data, ENGINE, versions={"mlx_vlm": "t"})
    assert s1.ocr_done == 5 and len(fake.calls) == 5
    files = _page_files(data)
    assert len(files) == 5
    rec = json.loads(files[0].read_text())
    assert rec["schema"] == 1 and rec["engine"]["revision"] == "rev1"
    assert rec["raw"].startswith("<raw") and rec["truncated"] is False
    mtimes = {f: f.stat().st_mtime_ns for f in files}

    runs_dir = data / "ocr" / "_runs"
    n_logs = len(list(runs_dir.glob("*.json")))
    fake2 = FakeRecognizer()
    s2 = po.run(plans, fake2, data, ENGINE)
    assert len(list(runs_dir.glob("*.json"))) == n_logs + 1
    assert fake2.calls == [] and s2.ocr_done == 0 and s2.skipped_cached == 5
    assert {f: f.stat().st_mtime_ns for f in _page_files(data)} == mtimes


def test_record_path_is_sha_and_zero_based_page(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data, file="scan.pdf")
    po.run(plans, FakeRecognizer(), data, ENGINE)
    assert (data / "ocr" / docs[0].sha256 / "p00000.json").is_file()
    assert (data / "ocr" / docs[0].sha256 / "p00002.json").is_file()


@pytest.mark.parametrize(
    "field,value", [("revision", "rev2"), ("max_tokens", 2048), ("render_px", 1024),
                    ("model", "m/y"), ("mode", "plain"), ("name", "other")],
)
def test_strict_stale_rule_redoes_pages(corpus, field, value):
    import dataclasses

    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    po.run(plans, FakeRecognizer(), data, ENGINE)
    changed = dataclasses.replace(ENGINE, **{field: value})
    assert po.classify(plans, data, changed)["stale"] == 5
    fake = FakeRecognizer()
    s = po.run(plans, fake, data, changed)
    assert len(fake.calls) == 5 and s.redone == 5


def test_corrupt_record_is_redone(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    po.run(plans, FakeRecognizer(), data, ENGINE)
    _page_files(data)[0].write_text("{not json")
    assert po.classify(plans, data, ENGINE)["corrupt"] == 1
    fake = FakeRecognizer()
    s = po.run(plans, fake, data, ENGINE)
    assert len(fake.calls) == 1 and s.redone == 1


def test_crash_between_temp_write_and_replace_leaves_no_partial_record(corpus, monkeypatch):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    real_replace = po._replace
    seen = {"n": 0}

    def flaky(src, dst):
        seen["n"] += 1
        if seen["n"] == 3:
            assert Path(src).exists()  # the temp file was written
            raise KeyboardInterrupt
        real_replace(src, dst)

    monkeypatch.setattr(po, "_replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        po.run(plans, FakeRecognizer(), data, ENGINE)
    assert len(_page_files(data)) == 2
    for f in _page_files(data):
        json.loads(f.read_text())
    logs = list((data / "ocr" / "_runs").glob("*.json"))
    assert len(logs) == 1 and json.loads(logs[0].read_text())["interrupted"] is True

    monkeypatch.setattr(po, "_replace", real_replace)
    fake = FakeRecognizer()
    po.run(plans, fake, data, ENGINE)
    assert len(fake.calls) == 3
    assert len(_page_files(data)) == 5
    assert not list((data / "ocr").rglob("*.tmp"))


def test_engine_error_stops_and_resume_finishes(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    with pytest.raises(RuntimeError):
        po.run(plans, FakeRecognizer(fail_at=2), data, ENGINE)
    assert len(_page_files(data)) == 2
    fake = FakeRecognizer()
    po.run(plans, fake, data, ENGINE)
    assert len(fake.calls) == 3


def test_truncated_flag_when_tokens_reach_max(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data, file="scan.pdf")
    po.run(plans, FakeRecognizer(tokens=4096), data, ENGINE)
    assert all(json.loads(f.read_text())["truncated"] for f in _page_files(data))
    assert po.report(data, ENGINE)["truncated"] == 3


def test_limit_counts_recognize_calls_and_is_not_interrupted(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    fake = FakeRecognizer()
    s = po.run(plans, fake, data, ENGINE, limit=2)
    assert len(fake.calls) == 2 and s.interrupted is False
    fake2 = FakeRecognizer()
    po.run(plans, fake2, data, ENGINE, limit=2)
    assert len(fake2.calls) == 2  # cached pages do not count against the limit


def test_report_sums_wall_time_only_over_runs_that_recognized(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    import itertools

    ticks = itertools.count()
    po.run(plans, FakeRecognizer(), data, ENGINE, clock=lambda: float(next(ticks)))
    po.run(plans, FakeRecognizer(), data, ENGINE)  # no-op resume
    rep = po.report(data, ENGINE)
    assert rep["pages_cached"] == 5 and rep["pdfs"] == 2
    assert rep["runs"] == 1
    assert rep["wall_s_total"] == pytest.approx(11.0)  # start + 2 per page + end
    assert rep["engine"]["revision"] == "rev1"
    assert "pages_cached 5" in po.format_report(rep)


def test_leftover_temp_files_are_removed_at_start(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    d = data / "ocr" / docs[0].sha256
    d.mkdir(parents=True)
    (d / ".p00000.json.123.tmp").write_text("x")
    po.run(plans, FakeRecognizer(), data, ENGINE, limit=0)
    assert not list(d.glob("*.tmp"))


def test_lighton_recognize_returns_raw_and_tokens_and_extract_is_unchanged():
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor
    from packet.ocr_parse import parse_lighton_grounding
    from packet.types import PageRef

    raw = "Hello world"
    calls = []

    def fake_generate(model, processor, prompt, image, max_tokens, temperature, verbose):
        calls.append(max_tokens)
        return SimpleNamespace(text=raw, generation_tokens=17)

    ext = LightOnExtractor(max_tokens=123)
    ext._load = lambda: (None, None, (fake_generate, "prompt"))
    assert ext.recognize(object()) == (raw, 17)
    ref = PageRef(packet_id="p", index=0, width=612.0, height=792.0, rotation=0,
                  has_text_ops=False)
    assert ext.extract(ref, "", image=lambda: object()) == parse_lighton_grounding(raw, ref)
    assert calls == [123, 123]


def test_failed_first_engine_call_still_counts_in_wall_time(corpus):
    import itertools

    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    ticks = itertools.count()
    with pytest.raises(RuntimeError):
        po.run(plans, FakeRecognizer(fail_at=0), data, ENGINE, clock=lambda: float(next(ticks)))
    log = json.loads(next((data / "ocr" / "_runs").glob("*.json")).read_text())
    assert log["engine_calls"] == 1 and log["ocr_done"] == 0
    rep = po.report(data, ENGINE)
    assert rep["runs"] == 1 and rep["wall_s_total"] > 0
