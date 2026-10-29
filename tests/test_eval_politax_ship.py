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


# GMA-476 follow-ups


def test_item5_counts_politax_only_false_ocr():
    from packet.eval.politax_ship import _item5_false_ocr

    assert _item5_false_ocr(politax=[_r("pt", [1])], builtin=[], dlnet=[]) == (False, ["pt"])


def _stub_gate(monkeypatch, tmp_path, *, politax=(), builtin=(), dlnet=(), dlnet_f1=1.0):
    import types

    import packet.eval.politax_ship as ps
    from packet.eval.politax_train import HoldoutScorecard

    m = types.SimpleNamespace(license_note="CC0 public domain")
    fx = types.SimpleNamespace(notes="a.pdf", pdf=Path("a.pdf"))
    pt = iter([_r("pt", politax), _r("pt", ())])
    monkeypatch.setattr(ps, "prepare_politax", lambda cache: (m, [fx]))
    monkeypatch.setattr(ps, "collect_trump_training_rows", lambda m, by: [])
    monkeypatch.setattr(ps, "assert_frozen_split", lambda m: None)
    monkeypatch.setattr(ps, "assert_training_set_clean", lambda rows, m: None)
    monkeypatch.setattr(ps, "load_logistic_scorer", lambda p: None)
    monkeypatch.setattr(ps, "run_fixture", lambda fx, pipe: next(pt))
    monkeypatch.setattr(ps, "holdout_scorecard",
                        lambda m, f, r: HoldoutScorecard(0.0, 0.0, 0.0, 0.0, {}))
    monkeypatch.setattr(ps, "run_named_fixtures", lambda root, pipeline: [_r("bi", builtin)])
    dl = _r("dlw", dlnet)
    dl.boundary = dl.boundary.__class__(dlnet_f1, dlnet_f1, dlnet_f1, [], [], 0)
    monkeypatch.setattr(ps, "run_dlnet_whole", lambda cache, root, pipeline: [dl])
    monkeypatch.setattr(ps, "max_holdout_model_probability", lambda w, c: 0.0)
    monkeypatch.setattr(ps, "holdout_max_native_text_blocks", lambda m, f: 0)
    w = tmp_path / "w.json"
    w.write_text('{"schema_version": 1, "training_packets": []}')
    return ps, w


def _item(report, n):
    return next(i for i in report.items if i.number == n)


def test_gate_item5_wiring_runs_through_stubbed_runs(monkeypatch, tmp_path):
    for kw, name in (({"politax": [0]}, "pt"), ({"builtin": [0]}, "bi"), ({"dlnet": [2]}, "dlw")):
        ps, w = _stub_gate(monkeypatch, tmp_path, **kw)
        item5 = _item(ps.evaluate_politax_ship(w, tmp_path, tmp_path), 5)
        assert not item5.passed
        assert name in item5.detail
    ps, w = _stub_gate(monkeypatch, tmp_path)
    assert _item(ps.evaluate_politax_ship(w, tmp_path, tmp_path), 5).passed


def test_gate_calls_item4_dlnet_floor(monkeypatch, tmp_path):
    ps, w = _stub_gate(monkeypatch, tmp_path, dlnet_f1=DLNET_LOGGED_HEURISTIC_MACRO)
    calls = []
    real = ps._item4_dlnet_floor
    monkeypatch.setattr(ps, "_item4_dlnet_floor", lambda t, m: calls.append((t, m)) or real(t, m))
    assert _item(ps.evaluate_politax_ship(w, tmp_path, tmp_path), 4).passed
    assert calls == [([], DLNET_LOGGED_HEURISTIC_MACRO)]


def test_ship_check_exit_code_follows_gate_without_weights(monkeypatch, tmp_path):
    import packet.eval.politax_ship as ps
    from packet import cli

    out = str(tmp_path / "t.json")
    for ship, code in ((False, 1), (True, 0)):
        monkeypatch.setattr(ps, "train_and_evaluate_ship",
                            lambda *a, _s=ship, **k: _stub_report(_s))
        assert cli.main(["politax-ship-check", "--train-output", out]) == code


def test_ship_check_missing_weights_reports_on_stderr(tmp_path, capsys):
    from packet.cli import main

    missing = tmp_path / "missing.json"
    assert main(["politax-ship-check", "--weights", str(missing)]) == 2
    err = capsys.readouterr().err
    assert "boundary weights not found" in err
    assert str(missing) in err


def test_ship_check_exits_2_on_wrong_schema_version(tmp_path, capsys):
    from packet.cli import main

    w = tmp_path / "w.json"
    w.write_text('{"schema_version": 999}')
    assert main(["politax-ship-check", "--weights", str(w)]) == 2
    assert "schema_version" in capsys.readouterr().err


def test_ship_check_weights_help_mentions_exit_2(capsys):
    import pytest

    from packet.cli import main

    with pytest.raises(SystemExit):
        main(["politax-ship-check", "--help"])
    assert "load errors exit 2" in " ".join(capsys.readouterr().out.split())
