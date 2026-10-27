"""GMA-274: opt-in ``packet eval --set politax --ocr-cache`` reads cached OCR text."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from packet import cli
from packet.eval import politax
from packet.eval.pdfutil import write_blank_scan_pdf
from packet.types import PageClass, PageRef


def _page(title: str, k: int, n: int) -> str:
    return (f"![title](80,70,920,140)\n# {title}\n![text](80,175,920,225)\n"
            f"Schedule {title} line items for the tax year.\n"
            f"![page_number](900,960,950,980)\nPage {k} of {n}\n")


SPECS = [("t1.pdf", "trump", 2), ("o1.pdf", "obama", 4)]


def _corpus(tmp: Path, specs=SPECS) -> tuple[Path, Path, dict[str, str]]:
    data = tmp / "politax"
    data.mkdir()
    docs, ann, shas = [], [], {}
    for name, subject, pages in specs:
        write_blank_scan_pdf(data / name, pages)
        sha = hashlib.sha256(Path(data / name).read_bytes()).hexdigest()
        shas[name] = sha
        docs.append({"subject": subject, "file_name": name, "file": name,
                     "url": f"https://example.invalid/{name}", "pages": pages, "sha256": sha,
                     "license": "test", "gold_cuts": 0})
        half = pages // 2
        ann.append({"file_name": name, "page_count": pages, "sub_documents": [
            {"type": "f", "start_page": 1, "end_page": half},
            {"type": "f", "start_page": half + 1, "end_page": pages}]})
    ap = data / "annotations.jsonl"
    ap.write_text("\n".join(json.dumps(a) for a in ann) + "\n")
    mp = tmp / "manifest.json"
    mp.write_text(json.dumps({
        "name": "t", "version": 1, "license_note": "CC0 test",
        "annotations": {"url": "https://example.invalid/a", "file": ap.name,
                        "sha256": hashlib.sha256(ap.read_bytes()).hexdigest(), "license": "CC0-1.0"},
        "docs": docs}))
    return mp, data, shas


def _write_cache(data: Path, shas: dict[str, str], pages: dict[str, int]) -> None:
    for name, sha in shas.items():
        n = pages[name]
        half = n // 2
        root = data / "ocr" / sha
        root.mkdir(parents=True)
        for i in range(n):
            title, k, m = ("Form 1040", i + 1, half) if i < half else ("Schedule K-1", i - half + 1, n - half)
            (root / f"p{i:05d}.json").write_text(json.dumps({
                "schema": 1, "pdf_sha256": sha, "page": i, "raw": _page(title, k, m),
                "truncated": i == 0, "engine": {"name": "lighton-mlx", "model": "m", "revision": "r"}}))


def _eval(monkeypatch, mp: Path, data: Path, *extra: str) -> tuple[int, dict]:
    monkeypatch.setattr(politax, "MANIFEST", mp)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["eval", "--set", "politax", "--data-dir", str(data), "--json", *extra])
    return rc, json.loads(buf.getvalue())


PAGES = {"t1.pdf": 2, "o1.pdf": 4}


def test_default_off_ignores_cache_and_never_imports_module(tmp_path, monkeypatch):
    mp, data, shas = _corpus(tmp_path)
    sys.modules.pop("packet.eval.politax_ocr_eval", None)
    rc0, before = _eval(monkeypatch, mp, data)
    _write_cache(data, shas, PAGES)
    rc1, after = _eval(monkeypatch, mp, data)
    assert rc0 == rc1 == 0
    assert "ocr_cache" not in after
    assert after == before
    assert "packet.eval.politax_ocr_eval" not in sys.modules


def test_ocr_cache_stdout_holdout_block_stays_native_baseline(tmp_path, monkeypatch, capsys):
    mp, data, shas = _corpus(tmp_path)
    _write_cache(data, shas, PAGES)
    monkeypatch.setattr(politax, "MANIFEST", mp)
    rc = cli.main(["eval", "--set", "politax", "--data-dir", str(data), "--ocr-cache"])
    out = capsys.readouterr().out
    assert rc == 0
    start = out.index("holdout heuristic baseline")
    block = out[start : out.find("macro holdout F1", start)]
    assert "0 predicted boundaries" in block
    assert "macro F1 0.0000" in block


def test_opt_in_feeds_cached_text_and_reports_stats(tmp_path, monkeypatch):
    mp, data, shas = _corpus(tmp_path)
    _, off = _eval(monkeypatch, mp, data)
    _write_cache(data, shas, PAGES)
    rc, on = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert rc == 0
    assert on["ocr_cache"] == {"pages_used": 6, "pages_missing": 0, "pages_truncated": 2,
                               "engines": {"lighton-mlx:m@r": 6}}
    hold = on["holdout_heuristic"]
    assert hold["holdout_files"] == ["o1.pdf"]
    assert on["holdout_baseline"] == hold
    # JSON holdout baseline stays native-text; the OCR lift shows up in the gated macro.
    assert hold["macro_f1"] == off["holdout_heuristic"]["macro_f1"]
    assert on["gate"]["macro_f1"] > off["gate"]["macro_f1"]


def test_opt_in_missing_records_count_and_yield_no_blocks(tmp_path, monkeypatch):
    mp, data, _ = _corpus(tmp_path)
    rc, on = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert rc == 1  # no cached text -> holdout macro 0.0 < Q7 floor 0.41
    assert on["ocr_cache"]["pages_used"] == 0 and on["ocr_cache"]["pages_missing"] == 6


def test_cached_extractor_rejects_bad_records(tmp_path):
    from packet.eval.politax_ocr_eval import CachedOcrExtractor, CacheStats

    root = tmp_path / "ocr" / "abc"
    root.mkdir(parents=True)
    (root / "p00000.json").write_text("{garbage")
    (root / "p00001.json").write_text(json.dumps({"schema": 1, "pdf_sha256": "other", "page": 1, "raw": "x"}))
    (root / "p00002.json").write_text(json.dumps({"schema": 1, "pdf_sha256": "abc", "page": 9, "raw": "x"}))
    for i, eng in ((3, "lighton"), (4, ["x"]), (5, None)):  # non-dict engine: missing, no crash
        (root / f"p{i:05d}.json").write_text(json.dumps(
            {"schema": 1, "pdf_sha256": "abc", "page": i, "raw": "x", "engine": eng}))
    stats = CacheStats()
    ext = CachedOcrExtractor(tmp_path, "abc", stats)
    assert ext.supports(PageClass.SCAN) and not ext.supports(PageClass.TEXT)
    for i in range(7):
        assert ext.extract(PageRef(packet_id="p", index=i, page_class=PageClass.SCAN), "") == []
    assert stats.used == 0 and stats.missing == 7


@pytest.mark.parametrize("extra", [["--set", "builtin"], ["--boundary-weights", "w.json"]])
def test_opt_in_rejects_other_sets_and_weights(extra, capsys):
    args = ["eval", "--ocr-cache", *extra]
    if "--set" not in extra:
        args += ["--set", "politax"]
    assert cli.main(args) == 2
    assert "--ocr-cache" in capsys.readouterr().err


def test_q7_floor_applies_only_to_ocr_cache_runs(tmp_path, monkeypatch):
    """GMA-274 Q7: --ocr-cache gates holdout macro at 0.41; plain eval stays (0, 0)."""
    from packet.eval import run as ev

    assert ev.SET_FLOORS["politax"] == (0.0, 0.0)
    assert ev.OCR_CACHE_FLOORS["politax"] == (0.0, 0.41)
    mp, data, shas = _corpus(tmp_path)
    rc_plain, plain = _eval(monkeypatch, mp, data)
    assert rc_plain == 0
    assert plain["gate"]["min_f1"] == 0.0 and plain["gate"]["min_macro_f1"] == 0.0
    assert plain["gate"]["macro_scope"] == "all"
    # No cache: holdout macro is 0.0, below the locked floor, so the ocr-cache gate fails.
    rc_miss, miss = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert rc_miss == 1 and miss["gate"]["passed"] is False
    assert miss["gate"]["min_macro_f1"] == 0.41 and miss["gate"]["min_f1"] == 0.0
    assert miss["gate"]["macro_scope"] == "holdout"
    assert any("macro (holdout) boundary F1" in f for f in miss["gate"]["failures"])
    # Cached text finds the holdout cut: macro is the holdout macro and clears 0.41.
    _write_cache(data, shas, PAGES)
    rc_hit, hit = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert rc_hit == 0 and hit["gate"]["passed"] is True
    assert hit["gate"]["macro_f1"] >= 0.41
    assert hit["holdout_heuristic"]["macro_f1"] == 0.0
    # Explicit flags still override the locked floor.
    rc_ovr, ovr = _eval(monkeypatch, mp, data, "--ocr-cache", "--min-macro-f1", "0")
    assert rc_ovr == 0 and ovr["gate"]["min_macro_f1"] == 0.0


def test_macro_results_scopes_only_the_macro_floor():
    from packet.eval import run as ev
    from packet.eval.metrics import BoundaryScore

    def row(name, f1):
        return ev.EvalResult(name, "c", 1, 1, 1, BoundaryScore(f1, f1, f1, [], [], 0), [])

    rows = [row("a", 0.0), row("b", 0.5)]
    g = ev.evaluate_gate(rows, min_f1=0.0, min_macro_f1=0.41, macro_results=rows[1:])
    assert g.passed and g.macro_f1 == 0.5
    assert not ev.evaluate_gate(rows, min_f1=0.0, min_macro_f1=0.41).passed


# Three train PDFs + one holdout: train and holdout score differently, so the gate
# outcome depends on whether the macro averages the holdout or all fixtures.
# Distinct page counts keep each blank PDF's sha256 (and OCR cache dir) unique.
SCOPE_SPECS = [("t1.pdf", "trump", 2), ("t2.pdf", "trump", 6), ("t3.pdf", "trump", 8),
               ("o1.pdf", "obama", 4)]
SCOPE_PAGES = {name: pages for name, _, pages in SCOPE_SPECS}


def _all_macro(out: dict) -> float:
    rows = out["fixtures"]
    return sum(r["boundary_f1"] for r in rows) / len(rows)


def test_q7_gate_passes_on_holdout_macro_where_all_macro_would_fail(tmp_path, monkeypatch):
    mp, data, shas = _corpus(tmp_path, SCOPE_SPECS)
    _write_cache(data, {"o1.pdf": shas["o1.pdf"]}, SCOPE_PAGES)  # holdout only
    rc, out = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert out["gate"]["macro_scope"] == "holdout"
    assert out["gate"]["macro_f1"] >= 0.41
    assert _all_macro(out) < 0.41  # an all-30-style macro would fail the floor
    assert rc == 0 and out["gate"]["passed"] is True


def test_q7_gate_fails_on_holdout_macro_where_all_macro_would_pass(tmp_path, monkeypatch):
    mp, data, shas = _corpus(tmp_path, SCOPE_SPECS)
    _write_cache(data, {n: shas[n] for n in ("t1.pdf", "t2.pdf", "t3.pdf")}, SCOPE_PAGES)
    rc, out = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert out["gate"]["macro_f1"] < 0.41
    assert _all_macro(out) >= 0.41  # an all-30-style macro would pass the floor
    assert rc == 1 and out["gate"]["passed"] is False
    assert any("macro (holdout) boundary F1" in f for f in out["gate"]["failures"])
