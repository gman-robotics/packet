"""GMA-291: PoliTax default-on ship gate (outcome b when holdout does not improve)."""

from pathlib import Path

from packet.eval.politax_ship import (
    DLNET_LOGGED_HEURISTIC_MACRO,
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
