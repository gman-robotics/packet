from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packet.classify import HeuristicClassifier
from packet.eval.fixtures import NamedFixture, build_builtin
from packet.eval.metrics import BoundaryScore, boundary_score, false_ocr_hard_fail
from packet.pipeline import Pipeline
from packet.source import PdfPageSource
from packet.types import Packet

DEFAULT_MIN_F1 = 1.0


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

    def passed_gate(self, min_f1: float = DEFAULT_MIN_F1) -> bool:
        return not self.false_ocr_pages and self.boundary.f1 >= min_f1


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    macro_f1: float
    min_f1: float
    min_macro_f1: float


def evaluate_gate(
    results: list[EvalResult],
    min_f1: float = DEFAULT_MIN_F1,
    min_macro_f1: float | None = None,
) -> GateResult:
    """EVAL.md gate: every fixture's boundary F1 (tolerance 0) >= ``min_f1``, the
    macro-average F1 >= ``min_macro_f1`` (defaults to ``min_f1``), and zero
    false-OCR pages anywhere (absolute, section 5)."""
    floor = min_f1 if min_macro_f1 is None else min_macro_f1
    failures: list[str] = []
    for r in results:
        if r.false_ocr_pages:
            failures.append(f"{r.name}: false-OCR on text-layer pages {r.false_ocr_pages}")
        if r.boundary.f1 < min_f1:
            failures.append(f"{r.name}: boundary F1 {r.boundary.f1:.2f} < {min_f1:.2f}")
    macro = sum(r.boundary.f1 for r in results) / len(results) if results else 0.0
    if not results:
        failures.append("no fixtures ran")
    elif macro < floor:
        failures.append(f"macro boundary F1 {macro:.2f} < {floor:.2f}")
    return GateResult(not failures, failures, macro, min_f1, floor)


def text_layer_pages(pdf: Path, min_chars: int | None = None) -> set[int]:
    """Pages whose native text sample is at least ``min_chars`` (EVAL.md section 5)."""
    threshold = HeuristicClassifier().min_chars if min_chars is None else min_chars
    with PdfPageSource(pdf) as src:
        return {
            i for i in range(src.page_count) if len((src.text_for(i) or "").strip()) >= threshold
        }


def run_fixture(fixture: NamedFixture, pipeline: Pipeline | None = None) -> EvalResult:
    pipe = pipeline or Pipeline()
    predicted: Packet = pipe.parse(fixture.pdf)
    score = boundary_score(fixture.gold, predicted, tolerance=0)
    text_pages = text_layer_pages(fixture.pdf)
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


def results_as_dict(results: list[EvalResult], min_f1: float = DEFAULT_MIN_F1) -> list[dict]:
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
                "gate_pass": r.passed_gate(min_f1),
                "notes": r.notes,
            }
        )
    return rows
