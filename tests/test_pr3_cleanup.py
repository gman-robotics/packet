"""GMA-217: #3 review should-fixes."""
import logging
import re
from pathlib import Path

import pytest

from packet.backends import FallbackExtractor, OcrUnavailable
from packet.types import PageClass, PageRef

REF = PageRef(packet_id="p", index=3, page_class=PageClass.SCAN)
BENCH = Path(__file__).resolve().parents[1] / "bench" / "ocr_compare"


class _Eng:
    def __init__(self, classes=(PageClass.SCAN,), out=None, exc=None):
        self.classes, self.out, self.exc, self.calls = set(classes), ["x"] if out is None else out, exc, 0

    def supports(self, pc):
        return pc in self.classes

    def extract(self, ref, text):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.out


def test_primary_crash_is_logged_and_recorded_not_swallowed(caplog):
    ext = FallbackExtractor(_Eng(exc=ValueError("bad tensor")), _Eng(out=["fb"]))
    with caplog.at_level(logging.WARNING, logger="packet.backends"):
        assert ext.extract(REF, "") == ["fb"]
    assert any("bad tensor" in r.getMessage() and "page 3" in r.getMessage() for r in caplog.records)
    assert ext.last_errors == ["_Eng: ValueError: bad tensor"]


def test_last_errors_reset_on_clean_page():
    p = _Eng(exc=ValueError("x"))
    ext = FallbackExtractor(p, _Eng())
    ext.extract(REF, "")
    p.exc = None
    ext.extract(REF, "")
    assert ext.last_errors == [] and ext.last_used is p


def test_both_fail_raises_with_both_reasons_and_types():
    ext = FallbackExtractor(_Eng(exc=OcrUnavailable("no mlx")), _Eng(exc=RuntimeError("oom")))
    with pytest.raises(OcrUnavailable, match=r"no mlx.*RuntimeError: oom"):
        ext.extract(REF, "")


def test_supports_is_union_and_skips_engine_that_does_not_support_page():
    p, f = _Eng(classes=[PageClass.SCAN]), _Eng(classes=[PageClass.MIXED], out=["fb"])
    ext = FallbackExtractor(p, f)
    assert ext.supports(PageClass.SCAN) and ext.supports(PageClass.MIXED)
    assert not ext.supports(PageClass.TEXT)
    mixed = PageRef(packet_id="p", index=0, page_class=PageClass.MIXED)
    assert ext.extract(mixed, "") == ["fb"] and p.calls == 0


def test_bench_scripts_have_no_hardcoded_sandbox_paths_and_call_real_files():
    for sh in BENCH.glob("*.sh"):
        body = sh.read_text()
        assert "/Users/" not in body, f"{sh.name} hardcodes a machine path"
        for rel in re.findall(r"\$BENCH/(\w+\.(?:py|sh))", body):
            assert (BENCH / rel).exists(), f"{sh.name} calls missing {rel}"
        assert "bench/run_" not in body, f"{sh.name} calls bench/run_* (files live in bench/ocr_compare)"


def test_force_inspect_runs_real_engines_on_text_pages():
    # Real engines only claim OCR page classes; --force-inspect must still OCR TEXT pages.
    from packet.backends import EscalatingExtractor

    text_ref = PageRef(packet_id="p", index=0, page_class=PageClass.TEXT)
    p = _Eng(classes=[PageClass.SCAN], out=["ocr"])
    esc = EscalatingExtractor(ocr=FallbackExtractor(p, _Eng(classes=[PageClass.SCAN])),
                              force_inspect=True)
    assert esc.extract(text_ref, "some text layer", image=lambda: "IMG") == ["ocr"]
    assert p.calls == 1


def test_empty_primary_result_is_final_and_records_no_error():
    p, f = _Eng(out=[]), _Eng(out=["fb"])
    p.out = []
    ext = FallbackExtractor(p, f)
    assert ext.extract(REF, "") == [] and ext.last_errors == [] and f.calls == 0


def test_bench_report_has_no_stale_paths():
    report = (BENCH / "REPORT.md").read_text()
    assert "/Users/" not in report
    assert not re.search(r"(?<![\w/])bench/(?!ocr_compare)\w", report)
