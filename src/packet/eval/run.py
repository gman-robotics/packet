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
# (per-fixture, macro) boundary-F1 floors when the CLI flags are not given.
SET_FLOORS = {
    "builtin": (DEFAULT_MIN_F1, None),
    "dlnet-whole": (0.0, 0.6),
    # Plain (native text) PoliTax eval stays report-only.
    "politax": (0.0, 0.0),
}
# GMA-274 Q7 (Tom, 2026-10-24): floors for ``packet eval --set politax --ocr-cache``.
# The macro floor applies to the frozen holdout packets (the population it was measured on:
# heuristic holdout macro 0.4159 truncated to 0.41); the per-fixture floor applies to all.
OCR_CACHE_FLOORS = {"politax": (0.0, 0.41)}
DEFAULT_WHOLEDOC_CACHE = Path.home() / ".cache" / "packet" / "dlnet-whole"
DEFAULT_POLITAX_CACHE = Path.home() / ".cache" / "packet" / "politax"


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

    def failures(self, min_f1: float = DEFAULT_MIN_F1) -> list[str]:
        """Per-fixture gate failures (the single source for row marks and the gate)."""
        out: list[str] = []
        if self.false_ocr_pages:
            out.append(f"{self.name}: false-OCR on text-layer pages {self.false_ocr_pages}")
        if not (self.boundary.f1 >= min_f1):  # also catches NaN
            out.append(f"{self.name}: boundary F1 {self.boundary.f1:.2f} < {min_f1:.2f}")
        return out

    def passed_gate(self, min_f1: float = DEFAULT_MIN_F1) -> bool:
        return not self.failures(min_f1)


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
    macro_results: list[EvalResult] | None = None,
) -> GateResult:
    """EVAL.md gate: every fixture's boundary F1 (tolerance 0) >= ``min_f1``, the
    macro-average F1 >= ``min_macro_f1`` (defaults to ``min_f1``), and zero
    false-OCR pages anywhere (absolute, section 5). ``macro_results`` (default
    ``results``) is the population the macro floor is averaged over."""
    floor = min_f1 if min_macro_f1 is None else min_macro_f1
    failures: list[str] = []
    for r in results:
        failures.extend(r.failures(min_f1))
    pool = results if macro_results is None else macro_results
    macro = sum(r.boundary.f1 for r in pool) / len(pool) if pool else float("nan")
    if not results or not pool:
        failures.append("no fixtures ran")
    elif not (macro >= floor):
        failures.append(f"macro boundary F1 {macro:.2f} < {floor:.2f}")
    return GateResult(not failures, failures, macro, min_f1, floor)


def text_layer_pages(pdf: Path, min_chars: int | None = None) -> set[int]:
    """Pages whose native text sample is at least ``min_chars`` (EVAL.md section 5).
    ``min_chars`` > 0 implies the page has text operators (``has_text_ops``)."""
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


def run_dlnet_whole(
    data_dir: Path, root: Path | None = None, pipeline: Pipeline | None = None
) -> list[EvalResult]:
    """``packet-dlnet-whole``: download (sha256-pinned) and join whole docs, then score."""
    from packet.eval import wholedoc

    m = wholedoc.load_manifest()
    wholedoc.fetch(m, cache=data_dir)
    out = (root or Path(".packet-eval")) / "dlnet-whole"
    return [run_fixture(fx, pipeline) for fx in wholedoc.build_fixtures(m, data_dir, out)]


def prepare_politax(data_dir: Path):
    from packet.eval import politax

    m = politax.load_manifest()
    ann_path = politax.fetch(m, cache=data_dir)
    annotations = politax.load_annotations(ann_path)
    fixtures = politax.build_fixtures(m, data_dir, annotations)
    return m, fixtures


def run_politax(data_dir: Path, pipeline: Pipeline | None = None) -> list[EvalResult]:
    """``packet-politax``: download PoliTax-Split PDFs and score one fixture per packet."""
    _, fixtures = prepare_politax(data_dir)
    return [run_fixture(fx, pipeline) for fx in fixtures]


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
