"""Eval harness: metrics, built-in fixtures, optional DocLayNet packet synth."""

from packet.eval.metrics import (
    BoundaryScore,
    boundary_score,
    citation_integrity,
    false_ocr_hard_fail,
)
from packet.eval.run import EvalResult, run_fixture, run_named_fixtures

__all__ = [
    "BoundaryScore",
    "EvalResult",
    "boundary_score",
    "citation_integrity",
    "false_ocr_hard_fail",
    "run_fixture",
    "run_named_fixtures",
]
