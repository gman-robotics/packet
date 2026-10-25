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


def _corpus(tmp: Path) -> tuple[Path, Path, dict[str, str]]:
    data = tmp / "politax"
    data.mkdir()
    specs = [("t1.pdf", "trump", 2), ("o1.pdf", "obama", 4)]
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
    # Cached text adds page-number restarts, so the holdout cut is found; text-free is not.
    assert hold["macro_f1"] > off["holdout_heuristic"]["macro_f1"]


def test_opt_in_missing_records_count_and_yield_no_blocks(tmp_path, monkeypatch):
    mp, data, _ = _corpus(tmp_path)
    rc, on = _eval(monkeypatch, mp, data, "--ocr-cache")
    assert rc == 0
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
