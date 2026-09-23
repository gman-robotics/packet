from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packet.eval.fixtures import NamedFixture, build_builtin
from packet.eval.metrics import BoundaryScore, boundary_score, false_ocr_hard_fail
from packet.pipeline import Pipeline
from packet.types import Packet


@dataclass
class EvalResult:
    name: str
    condition: str
    pages: int
    gold_docs: int
    pred_docs: int
    boundary: BoundaryScore
    false_ocr_pages: list[int]
    notes: str = ""

    def passed_gate(self) -> bool:
        return not self.false_ocr_pages


def run_fixture(fixture: NamedFixture, pipeline: Pipeline | None = None) -> EvalResult:
    pipe = pipeline or Pipeline()
    predicted: Packet = pipe.parse(fixture.pdf)
    score = boundary_score(fixture.gold, predicted, tolerance=0)
    text_pages = set()
    if fixture.condition == "born-digital":
        text_pages = set(range(fixture.gold.origin.page_count))
    fails = false_ocr_hard_fail(predicted, text_pages)
    return EvalResult(
        name=fixture.name,
        condition=fixture.condition,
        pages=fixture.gold.origin.page_count,
        gold_docs=fixture.gold.document_count(),
        pred_docs=predicted.document_count(),
        boundary=score,
        false_ocr_pages=fails,
        notes=fixture.notes,
    )


def run_named_fixtures(root: Path | None = None, pipeline: Pipeline | None = None) -> list[EvalResult]:
    return [run_fixture(fx, pipeline) for fx in build_builtin(root)]


def results_as_dict(results: list[EvalResult]) -> list[dict]:
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.name,
                "condition": r.condition,
                "pages": r.pages,
                "gold_docs": r.gold_docs,
                "pred_docs": r.pred_docs,
                "boundary_p": r.boundary.precision,
                "boundary_r": r.boundary.recall,
                "boundary_f1": r.boundary.f1,
                "gold_boundaries": r.boundary.gold,
                "pred_boundaries": r.boundary.predicted,
                "false_ocr_pages": r.false_ocr_pages,
                "gate_pass": r.passed_gate(),
                "notes": r.notes,
            }
        )
    return rows
