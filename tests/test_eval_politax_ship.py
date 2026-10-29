"""GMA-291: PoliTax default-on ship gate (outcome b when holdout does not improve)."""

from pathlib import Path

from packet.eval.politax_ship import (
    DLNET_LOGGED_HEURISTIC_MACRO,
    HOLDOUT_LOGGED_MAX_MODEL_P,
    _item2_holdout_strictly_up,
)
from packet.eval.politax_train import HoldoutScorecard
from packet.pipeline import PipelineConfig


def test_item2_fails_when_scorer_matches_heuristic():
    zero = HoldoutScorecard(0.0, 0.0, 0.0, 0.0, {"a.pdf": 0.0})
    ok, detail = _item2_holdout_strictly_up(zero, zero)
    assert not ok
    assert "macro_up=False" in detail


def test_item2_requires_strict_improvement():
    heur = HoldoutScorecard(0.1, 0.0, 0.0, 0.0, {"a.pdf": 0.1})
    scorer = HoldoutScorecard(0.2, 0.1, 0.1, 0.15, {"a.pdf": 0.2})
    ok, _ = _item2_holdout_strictly_up(heur, scorer)
    assert ok


def test_pipeline_scorer_default_stays_none():
    assert PipelineConfig().scorer is None


def test_eval_subcommand_does_not_read_decision_api_key():
    text = Path("src/packet/cli.py").read_text()
    block = text.split("def _cmd_eval")[1].split("def _cmd_eval_plan")[0]
    assert "PACKET_DECISION_API_KEY" not in block


def test_dlnet_logged_heuristic_floor_constant():
    assert DLNET_LOGGED_HEURISTIC_MACRO == 0.98462


def test_holdout_logged_max_model_p_constant():
    assert HOLDOUT_LOGGED_MAX_MODEL_P == 0.4989


# GMA-383 should-fixes


def _r(name, false_ocr=()):
    from packet.eval.metrics import BoundaryScore
    from packet.eval.run import EvalResult

    return EvalResult(name=name, condition="born-digital", pages=1, gold_docs=1, pred_docs=1,
                      boundary=BoundaryScore(1.0, 1.0, 1.0, [], [], 0),
                      false_ocr_pages=list(false_ocr))


def test_item4_passes_at_exact_floor_and_fails_just_below():
    from packet.eval.politax_ship import _item4_dlnet_floor

    assert _item4_dlnet_floor([], DLNET_LOGGED_HEURISTIC_MACRO)
    assert not _item4_dlnet_floor([], DLNET_LOGGED_HEURISTIC_MACRO - 1e-9)
    assert not _item4_dlnet_floor(["dlw-a"], DLNET_LOGGED_HEURISTIC_MACRO)


def test_item5_counts_builtin_and_dlnet_false_ocr():
    from packet.eval.politax_ship import _item5_false_ocr

    clean = [_r("p")]
    assert _item5_false_ocr(politax=clean, builtin=clean, dlnet=clean) == (True, [])
    ok, bad = _item5_false_ocr(politax=clean, builtin=[_r("bi", [0])], dlnet=[_r("dlw", [3])])
    assert not ok
    assert bad == ["bi", "dlw"]


def test_item5_gate_passes_builtin_and_dlnet_results():
    import inspect

    from packet.eval import politax_ship

    src = inspect.getsource(politax_ship.evaluate_politax_ship)
    assert "builtin=builtin_scorer" in src
    assert "dlnet=dlnet_results" in src


def test_ship_check_exits_2_on_bad_weights(tmp_path, capsys):
    from packet.cli import main

    bad = tmp_path / "w.json"
    bad.write_text("{not json")
    assert main(["politax-ship-check", "--weights", str(bad)]) == 2
    assert "boundary weights" in capsys.readouterr().err
    assert main(["politax-ship-check", "--weights", str(tmp_path / "missing.json")]) == 2


def _stub_report(ship):
    from packet.eval.politax_ship import ShipItem, ShipReport

    return ShipReport(ship=ship, outcome="a" if ship else "b",
                      items=[ShipItem(1, "t", ship, "d")], holdout_heuristic={},
                      holdout_scorer={}, max_holdout_model_p=0.0,
                      dlnet_macro_scorer_on=0.0, builtin_f1={})


def test_ship_check_exit_code_follows_gate(monkeypatch, tmp_path):
    import packet.eval.politax_ship as ps
    from packet import cli

    monkeypatch.setattr("packet.boundary_model.load_logistic_scorer", lambda p: None)
    for ship, code in ((False, 1), (True, 0)):
        monkeypatch.setattr(ps, "evaluate_politax_ship", lambda *a, _s=ship, **k: _stub_report(_s))
        assert cli.main(["politax-ship-check", "--weights", str(tmp_path / "w.json")]) == code
