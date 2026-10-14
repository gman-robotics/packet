"""GMA-55 slice 4: `packet eval` is a real gate. It fails on boundary F1 below a
threshold and on any false-OCR page."""
from __future__ import annotations

import json

import pytest

from packet import cli
from packet.eval.fixtures import build_builtin
from packet.eval.metrics import BoundaryScore
from packet.eval.run import EvalResult, evaluate_gate, run_fixture
from packet.pipeline import Pipeline
from packet.types import Block, BlockKind, SourceKind


def _res(name="fx", f1=1.0, false_ocr=()):
    return EvalResult(name=name, condition="born-digital", pages=3, gold_docs=1, pred_docs=1,
                      boundary=BoundaryScore(f1, f1, f1, [], [], 0), false_ocr_pages=list(false_ocr))


def test_gate_passes_when_all_good():
    g = evaluate_gate([_res(), _res("b")], min_f1=0.9)
    assert g.passed and g.failures == []


def test_gate_fails_on_low_boundary_f1_per_fixture():
    g = evaluate_gate([_res(), _res("weak", f1=0.5)], min_f1=0.9)
    assert not g.passed
    assert any("weak" in f and "F1" in f for f in g.failures)


def test_gate_fails_on_low_macro_f1_even_if_threshold_per_fixture_is_lower():
    g = evaluate_gate([_res(f1=0.8), _res("b", f1=0.8)], min_f1=0.75, min_macro_f1=0.9)
    assert not g.passed and any("macro" in f for f in g.failures)


def test_gate_fails_on_any_false_ocr_page_regardless_of_f1():
    g = evaluate_gate([_res(false_ocr=[2])], min_f1=0.0)
    assert not g.passed and any("false-OCR" in f for f in g.failures)


@pytest.mark.parametrize("results,code", [([_res()], 0), ([_res(f1=0.2)], 1), ([_res(false_ocr=[0])], 1)])
def test_cli_exit_code_follows_gate(monkeypatch, capsys, results, code):
    monkeypatch.setattr("packet.eval.run.run_named_fixtures", lambda root=None, pipeline=None: results)
    assert cli.main(["eval", "--min-f1", "0.9"]) == code
    out = capsys.readouterr().out
    assert ("GATE PASS" if code == 0 else "GATE FAIL") in out


def test_cli_json_reports_gate(monkeypatch, capsys):
    monkeypatch.setattr("packet.eval.run.run_named_fixtures", lambda root=None, pipeline=None: [_res(f1=0.2)])
    assert cli.main(["eval", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate"]["passed"] is False and payload["gate"]["failures"]
    assert payload["fixtures"][0]["boundary_f1"] == 0.2


def test_false_ocr_uses_real_text_layer_not_condition(tmp_path):
    """A VLM block on a page that has a >=40-char text layer is a false-OCR hard-fail,
    whatever the fixture's condition label says."""
    fx = next(f for f in build_builtin(tmp_path) if f.name == "text-layer")
    fx.condition = "clean-scan"  # mislabelled on purpose

    class _Vlm:
        def supports(self, pc):
            return True

        def extract(self, ref, text):
            return [Block(id=f"v{ref.index}", page=ref.index, kind=BlockKind.PARAGRAPH,
                          text="x", source=SourceKind.VLM)]

    r = run_fixture(fx, Pipeline(extractors=[_Vlm()]))
    assert r.false_ocr_pages == [0, 1]  # page 2 has < 40 chars of text layer
    assert not evaluate_gate([r], min_f1=0.0).passed


def test_short_text_page_is_not_false_ocr(tmp_path):
    from packet.eval.fixtures import NamedFixture, _packet
    from packet.eval.pdfutil import write_text_pdf

    pdf = write_text_pdf(tmp_path / "short.pdf", [["p. 3"]])
    fx = NamedFixture("short", pdf, _packet("short", pdf, [(0, 0)], 1), "born-digital")

    class _Vlm:
        def supports(self, pc):
            return True

        def extract(self, ref, text):
            return [Block(id="v", page=0, kind=BlockKind.PARAGRAPH, text="x", source=SourceKind.VLM)]

    assert run_fixture(fx, Pipeline(extractors=[_Vlm()])).false_ocr_pages == []


def test_builtin_fixtures_pass_default_gate(tmp_path):
    assert cli.main(["eval", "--dir", str(tmp_path)]) == 0
