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
    def __init__(self, tokens: int = 10, fail_at: int | None = None, *,
                 stop: str = "eos", prompt_tokens: int = 0,
                 prefill_s: float = 0.0, decode_s: float = 0.0, render_s: float = 0.0):
        self.calls: list[tuple[str, int]] = []
        self.tokens = tokens
        self.fail_at = fail_at
        self.stop = stop
        self.prompt_tokens = prompt_tokens
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.render_s = render_s

    def __call__(self, plan: po.PagePlan):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("engine boom")
        self.calls.append((plan.pdf_sha256, plan.index))
        return po.PageResult(
            raw=f"<raw {plan.file_name} {plan.index}>",
            tokens=self.tokens,
            prompt_tokens=self.prompt_tokens,
            prefill_s=self.prefill_s,
            decode_s=self.decode_s,
            stop=self.stop,
            render_s=self.render_s,
        )


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
    po.run(plans, FakeRecognizer(tokens=4096, stop="max_tokens"), data, ENGINE)
    recs = [json.loads(f.read_text()) for f in _page_files(data)]
    assert all(r["truncated"] for r in recs)
    assert all(r["stop"] == "max_tokens" for r in recs)
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

    from packet.backends import LightOnExtractor, Recognition
    from packet.ocr_parse import parse_lighton_grounding
    from packet.types import PageRef

    raw = "Hello world"
    calls = []

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        calls.append(max_tokens)
        yield SimpleNamespace(text=raw, generation_tokens=17, prompt_tokens=40,
                              prompt_tps=20.0, generation_tps=10.0, finish_reason="stop")

    ext = LightOnExtractor(max_tokens=123)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object())
    assert isinstance(rec, Recognition)
    assert rec.raw == raw and rec.tokens == 17 and rec.prompt_tokens == 40
    assert rec.prefill_s == pytest.approx(2.0) and rec.decode_s == pytest.approx(1.7)
    assert rec.stop == "eos"
    ref = PageRef(packet_id="p", index=0, width=612.0, height=792.0, rotation=0,
                  has_text_ops=False)
    assert ext.extract(ref, "", image=lambda: object()) == parse_lighton_grounding(raw, ref)
    assert calls == [123, 123]


@pytest.mark.parametrize(
    "finish,tokens,max_tokens,want",
    [("length", 10, 4096, "max_tokens"),
     ("stop", 4096, 4096, "max_tokens"),
     ("stop", 10, 4096, "eos"),
     (None, 5, 4096, "eos")],
)
def test_lighton_recognize_maps_stop_reason(finish, tokens, max_tokens, want):
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        yield SimpleNamespace(text="x", generation_tokens=tokens, prompt_tokens=0,
                              prompt_tps=0.0, generation_tps=0.0, finish_reason=finish)

    ext = LightOnExtractor(max_tokens=max_tokens)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object())
    assert rec.stop == want
    assert rec.prefill_s == 0.0 and rec.decode_s == 0.0


def test_lighton_recognize_uses_last_generation_result_for_stats():
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        yield SimpleNamespace(text="Hel", generation_tokens=1, prompt_tokens=8,
                              prompt_tps=8.0, generation_tps=1.0, finish_reason=None)
        yield SimpleNamespace(text="lo", generation_tokens=5, prompt_tokens=8,
                              prompt_tps=16.0, generation_tps=5.0, finish_reason="stop")

    ext = LightOnExtractor(max_tokens=4096)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object())
    assert rec.raw == "Hello"
    assert rec.tokens == 5 and rec.prompt_tokens == 8
    assert rec.prefill_s == pytest.approx(0.5) and rec.decode_s == pytest.approx(1.0)
    assert rec.stop == "eos"


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


def test_page_json_records_stage_fields_and_summary_accumulates(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data, file="scan.pdf")
    fake = FakeRecognizer(tokens=8, prompt_tokens=12, prefill_s=0.5, decode_s=1.25,
                          stop="eos", render_s=0.04)
    s = po.run(plans, fake, data, ENGINE, setup_s=1.5)
    rec = json.loads((data / "ocr" / docs[0].sha256 / "p00000.json").read_text())
    assert rec["schema"] == 1
    assert rec["render_s"] == 0.04
    assert rec["prefill_s"] == 0.5
    assert rec["decode_s"] == 1.25
    assert rec["prompt_tokens"] == 12
    assert rec["stop"] == "eos"
    assert rec["generation_tokens"] == 8
    assert rec["truncated"] is False
    assert "seconds" in rec
    assert s.stops == {"eos": 3}
    assert s.setup_s == pytest.approx(1.5)
    assert s.render_s == pytest.approx(0.12)
    assert s.prefill_s == pytest.approx(1.5)
    assert s.decode_s == pytest.approx(3.75)
    assert s.write_s >= 0
    log = json.loads(next((data / "ocr" / "_runs").glob("*.json")).read_text())
    assert log["setup_s"] == pytest.approx(1.5)
    assert log["stops"] == {"eos": 3}


def test_old_records_without_stage_keys_stay_cached(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data, file="scan.pdf")
    po.run(plans, FakeRecognizer(), data, ENGINE)
    for path in _page_files(data):
        rec = json.loads(path.read_text())
        for k in ("render_s", "prefill_s", "decode_s", "prompt_tokens", "stop"):
            rec.pop(k, None)
        path.write_text(json.dumps(rec))
    assert po.classify(plans, data, ENGINE)["cached"] == 3
    fake = FakeRecognizer()
    s = po.run(plans, fake, data, ENGINE)
    assert fake.calls == [] and s.skipped_cached == 3


def test_report_sums_stage_times_and_stop_counts_missing_keys_are_zero(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data)
    po.run(plans, FakeRecognizer(prompt_tokens=2, prefill_s=0.2, decode_s=0.3,
                                 render_s=0.1, stop="eos"), data, ENGINE, setup_s=1.0)
    po.run(plans, FakeRecognizer(), data, ENGINE)  # no-op resume
    old = data / "ocr" / "_runs" / "old.json"
    old.write_text(json.dumps({"engine_calls": 1, "wall_s": 0.0, "gen_s": 0.0}))
    rep = po.report(data, ENGINE)
    assert rep["setup_s"] == pytest.approx(1.0)
    assert rep["render_s"] == pytest.approx(0.5)
    assert rep["prefill_s"] == pytest.approx(1.0)
    assert rep["decode_s"] == pytest.approx(1.5)
    assert rep["write_s"] >= 0
    assert rep["stops"]["eos"] == 5
    assert rep["stops"]["max_tokens"] == 0
    text = po.format_report(rep)
    assert "pages_cached 5" in text
    assert "render_s" in text and "stops" in text
    assert "eos=5" in text


def _prose(n: int = 1500) -> str:
    parts: list[str] = []
    i = 0
    while sum(map(len, parts)) < n:
        parts.append(
            f"Schedule {i} names taxpayer {i * 17 + 3} of precinct {1000 + i}, "
            f"ballot {chr(65 + (i % 26))}{i}-{i * 13 % 97}, "
            f"and a note about levy {i * 29 % 300} distinct from the prior line. "
        )
        i += 1
    return "".join(parts)[:n]


def test_loop_fired_on_short_repeat_not_on_prose():
    from packet.backends import LoopRule, _loop_fired

    rule = LoopRule()
    assert _loop_fired("a" * 1500, rule) is True
    assert _loop_fired("ab" * 800, rule) is True
    assert _loop_fired("a" * 1499, rule) is False
    assert _loop_fired(_prose(1500), rule) is False
    assert _loop_fired(_prose(1800), rule) is False


def test_lighton_recognize_loop_rule_stops_before_max_tokens():
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor, LoopRule

    max_tokens = 2000
    yielded = {"n": 0}

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        assert temperature == 0.0 and verbose is False and max_tokens == 2000
        for i in range(1, max_tokens + 1):
            yielded["n"] += 1
            yield SimpleNamespace(
                text="a",
                generation_tokens=i,
                prompt_tokens=8,
                prompt_tps=16.0,
                generation_tps=10.0,
                finish_reason="length" if i == max_tokens else None,
            )

    ext = LightOnExtractor(max_tokens=max_tokens)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object(), loop_rule=LoopRule())
    assert rec.stop == "loop"
    assert rec.tokens == 1536 and rec.tokens < max_tokens
    assert rec.raw == "a" * 1536
    assert yielded["n"] == 1536
    assert rec.prompt_tokens == 8
    assert rec.prefill_s == pytest.approx(0.5)
    assert rec.decode_s == pytest.approx(153.6)


def test_lighton_recognize_without_loop_rule_drains_repetitive_stream():
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor

    max_tokens = 2000
    yielded = {"n": 0}

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        for i in range(1, max_tokens + 1):
            yielded["n"] += 1
            yield SimpleNamespace(
                text="a",
                generation_tokens=i,
                prompt_tokens=8,
                prompt_tps=16.0,
                generation_tps=10.0,
                finish_reason="length" if i == max_tokens else None,
            )

    ext = LightOnExtractor(max_tokens=max_tokens)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object())
    assert rec.stop == "max_tokens"
    assert rec.tokens == max_tokens and rec.raw == "a" * max_tokens
    assert yielded["n"] == max_tokens


def test_lighton_recognize_loop_rule_leaves_short_text_on_eos():
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor, LoopRule

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        yield SimpleNamespace(text="Hel", generation_tokens=1, prompt_tokens=8,
                              prompt_tps=8.0, generation_tps=1.0, finish_reason=None)
        yield SimpleNamespace(text="lo", generation_tokens=5, prompt_tokens=8,
                              prompt_tps=16.0, generation_tps=5.0, finish_reason="stop")

    ext = LightOnExtractor(max_tokens=4096)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object(), loop_rule=LoopRule())
    assert rec.raw == "Hello" and rec.tokens == 5 and rec.stop == "eos"
    assert rec.prefill_s == pytest.approx(0.5) and rec.decode_s == pytest.approx(1.0)


def test_loop_rule_counts_yields_when_generation_tokens_absent():
    from types import SimpleNamespace

    from packet.backends import LightOnExtractor, LoopRule

    yielded = {"n": 0}

    def fake_stream(model, processor, prompt, image, max_tokens, temperature, verbose):
        for _ in range(500):
            yielded["n"] += 1
            yield SimpleNamespace(text="a", prompt_tokens=2, prompt_tps=2.0,
                                  generation_tps=1.0, finish_reason=None)

    rule = LoopRule(window_chars=80, max_zlib_ratio=0.25, check_every_tokens=8)
    ext = LightOnExtractor(max_tokens=4096)
    ext._load = lambda: (None, None, (fake_stream, "prompt"))
    rec = ext.recognize(object(), loop_rule=rule)
    assert rec.stop == "loop" and rec.tokens == 80 and rec.raw == "a" * 80
    assert yielded["n"] == 80
    assert rec.tokens < ext.max_tokens


def test_loop_stop_page_is_not_truncated(corpus):
    data, docs = corpus
    plans, _ = po.plan_pages(docs, data, file="scan.pdf")
    s = po.run(plans, FakeRecognizer(tokens=100, stop="loop"), data, ENGINE)
    recs = [json.loads(f.read_text()) for f in _page_files(data)]
    assert len(recs) == 3
    assert all(r["stop"] == "loop" and r["truncated"] is False for r in recs)
    assert all(r["generation_tokens"] == 100 for r in recs)
    assert all(r["schema"] == 1 for r in recs)
    assert s.stops == {"loop": 3} and s.truncated == 0
    rep = po.report(data, ENGINE)
    assert rep["stops"]["loop"] == 3 and rep["stops"]["eos"] == 0
    assert rep["truncated"] == 0
    assert "loop=3" in po.format_report(rep)


def test_engine_id_has_no_loop_fields():
    import dataclasses

    assert [f.name for f in dataclasses.fields(po.EngineId)] == [
        "name", "model", "revision", "mode", "max_tokens", "render_px",
    ]
    assert po.SCHEMA == 1


def test_politax_ocr_loop_stop_flag_reaches_recognize(monkeypatch, corpus):
    import importlib.util
    import shutil

    from packet import cli
    from packet.backends import LightOnExtractor, LoopRule, Recognition
    from packet.source import PdfPageSource

    data, docs = corpus

    class Manifest:
        pass

    manifest = Manifest()
    manifest.docs = docs
    monkeypatch.setattr(politax, "load_manifest", lambda: manifest)
    monkeypatch.setattr(politax, "fetch", lambda *a, **k: None)
    monkeypatch.setattr(PdfPageSource, "render", lambda self, index, longest_px=1540: "img")
    seen: list = []

    def fake_recognize(self, img, loop_rule=None):
        seen.append(loop_rule)
        return Recognition("ok", 3, 1, 0.0, 0.0, "eos")

    monkeypatch.setattr(LightOnExtractor, "recognize", fake_recognize)
    real_find = importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name == "mlx_vlm":
            return object()
        return real_find(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    def run(flag: str | None):
        shutil.rmtree(data / "ocr", ignore_errors=True)
        seen.clear()
        argv = ["politax-ocr", "--data-dir", str(data), "--limit-pages", "1"]
        if flag:
            argv.append(flag)
        assert cli.main(argv) == 0
        return list(seen)

    assert run(None) == [LoopRule()]
    assert run("--loop-stop") == [LoopRule()]
    assert run("--no-loop-stop") == [None]


def test_replay_script_flags_only_non_truncated_loops(tmp_path):
    import os
    import subprocess
    import sys

    root = tmp_path / "ocr" / "abc"
    root.mkdir(parents=True)
    prose = _prose(1600)

    def write(name: str, raw: str, truncated: bool) -> None:
        (root / name).write_text(json.dumps({"raw": raw, "truncated": truncated}))

    write("p00000.json", "a" * 1600, False)
    write("p00001.json", "a" * 1600, True)
    write("p00002.json", prose, False)
    import ast

    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_ocr_loop_stop.py"
    mods: list[str] = []
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Import):
            mods.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module.split(".")[0])
    assert "mlx" not in mods and "mlx_vlm" not in mods
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    bad = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path / "ocr")],
        check=False, capture_output=True, text=True, env=env,
    )
    assert bad.returncode == 1
    assert "fired 1" in bad.stdout
    assert "truncated_would_fire 1" in bad.stdout
    assert "p00000.json" in bad.stderr

    (root / "p00000.json").unlink()
    before = sorted(p.name for p in root.iterdir())
    ok = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path / "ocr")],
        check=False, capture_output=True, text=True, env=env,
    )
    assert ok.returncode == 0
    assert "fired 0" in ok.stdout
    assert "truncated_would_fire 1" in ok.stdout
    assert sorted(p.name for p in root.iterdir()) == before

    missing = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path / "nope")],
        check=False, capture_output=True, text=True, env=env,
    )
    assert missing.returncode == 0
    assert "not found" in missing.stdout
    assert not (tmp_path / "nope").exists()
