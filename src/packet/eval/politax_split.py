"""Frozen PoliTax train/holdout split (GMA-274).

Train on ``trump`` whole PDFs only. Hold out ``obama``, ``harris``, and ``biden`` as whole
files (never split inside a PDF). Q7 (locked 2026-10-24): ``OCR_CACHE_FLOORS["politax"]`` gates ``--ocr-cache`` runs at holdout
macro 0.41; plain ``SET_FLOORS["politax"]`` stays report-only. This module computes the
proposed macro floor from the holdout heuristic baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from packet.eval.fixtures import NamedFixture
from packet.eval.metrics import micro_boundary_prf
from packet.eval.politax import Manifest, PoliTaxDoc
from packet.eval.run import EvalResult

POLITAX_TRAIN_SUBJECT = "trump"
POLITAX_HOLDOUT_SUBJECTS = frozenset({"obama", "harris", "biden"})


class SplitError(ValueError):
    pass


def truncate_two_decimal_places(value: float) -> float:
    """Truncate toward zero at two decimal places (plan Q7; not rounding)."""
    return math.trunc(value * 100) / 100


def partition_docs(m: Manifest) -> tuple[list[PoliTaxDoc], list[PoliTaxDoc]]:
    """Partition manifest docs into train and holdout whole PDFs."""
    train: list[PoliTaxDoc] = []
    holdout: list[PoliTaxDoc] = []
    for d in m.docs:
        if d.subject == POLITAX_TRAIN_SUBJECT:
            train.append(d)
        elif d.subject in POLITAX_HOLDOUT_SUBJECTS:
            holdout.append(d)
        else:
            raise SplitError(f"unknown PoliTax subject {d.subject!r} on {d.file_name}")
    return train, holdout


def assert_frozen_split(m: Manifest) -> None:
    """Train and holdout are disjoint, cover every PDF, and assign one subject per file."""
    train, holdout = partition_docs(m)
    if len(train) + len(holdout) != len(m.docs):
        raise SplitError("train and holdout do not cover every manifest PDF")
    train_files = {d.file_name for d in train}
    holdout_files = {d.file_name for d in holdout}
    if train_files & holdout_files:
        raise SplitError("train and holdout share PDF file names")
    if len(train_files) != len(train) or len(holdout_files) != len(holdout):
        raise SplitError("duplicate file name in train or holdout")


def holdout_file_names(m: Manifest) -> frozenset[str]:
    _, holdout = partition_docs(m)
    return frozenset(d.file_name for d in holdout)


def holdout_results(
    m: Manifest, fixtures: list[NamedFixture], results: list[EvalResult]
) -> list[EvalResult]:
    """Eval rows for holdout PDFs only (one whole file per fixture)."""
    holdout = holdout_file_names(m)
    by_fixture = {fx.name: r for fx, r in zip(fixtures, results)}
    out: list[EvalResult] = []
    for fx in fixtures:
        file_name = fx.notes or fx.pdf.name
        if file_name in holdout:
            out.append(by_fixture[fx.name])
    if len(out) != len(holdout):
        raise SplitError(f"expected {len(holdout)} holdout results, got {len(out)}")
    return out


@dataclass
class HoldoutBaseline:
    holdout_files: list[str]
    per_subject_cuts: dict[str, int]
    macro_f1: float
    micro_p: float
    micro_r: float
    micro_f1: float
    proposed_macro_floor: float


def compute_holdout_baseline(
    m: Manifest, fixtures: list[NamedFixture], results: list[EvalResult]
) -> HoldoutBaseline:
    """Heuristic baseline (``scorer is None``) macro/micro on the frozen holdout packets."""
    assert_frozen_split(m)
    holdout_rows = holdout_results(m, fixtures, results)
    macro = sum(r.boundary.f1 for r in holdout_rows) / len(holdout_rows)
    micro_p, micro_r, micro_f1 = micro_boundary_prf(holdout_rows)
    subject_by_file = {d.file_name: d.subject for d in m.docs}
    per_subject: dict[str, int] = {}
    holdout = holdout_file_names(m)
    for fx in fixtures:
        file_name = fx.notes or fx.pdf.name
        if file_name not in holdout:
            continue
        subj = subject_by_file[file_name]
        per_subject[subj] = per_subject.get(subj, 0) + fx.gold.document_count()
    per_subject = dict(sorted(per_subject.items()))
    files = sorted(holdout_file_names(m))
    return HoldoutBaseline(
        holdout_files=files,
        per_subject_cuts=per_subject,
        macro_f1=macro,
        micro_p=micro_p,
        micro_r=micro_r,
        micro_f1=micro_f1,
        proposed_macro_floor=truncate_two_decimal_places(macro),
    )


def format_holdout_baseline(baseline: HoldoutBaseline) -> str:
    lines = [
        "holdout heuristic baseline (frozen split; scorer None):",
        f"  holdout PDFs ({len(baseline.holdout_files)}):",
    ]
    for name in baseline.holdout_files:
        lines.append(f"    - {name}")
    lines.append("  gold cuts per holdout subject:")
    for subj, cuts in baseline.per_subject_cuts.items():
        lines.append(f"    {subj}: {cuts}")
    lines.append(
        f"  macro F1 {baseline.macro_f1:.4f}  "
        f"micro P {baseline.micro_p:.4f} R {baseline.micro_r:.4f} F1 {baseline.micro_f1:.4f}"
    )
    lines.append(
        f"  proposed macro floor (truncated to 2 decimals; Q7 locks 0.41 for --ocr-cache): "
        f"{baseline.proposed_macro_floor:.2f}  "
        "(plain eval floors stay 0.0)"
    )
    return "\n".join(lines)


def holdout_baseline_as_dict(baseline: HoldoutBaseline) -> dict:
    return {
        "holdout_files": baseline.holdout_files,
        "per_subject_gold_cuts": baseline.per_subject_cuts,
        "macro_f1": baseline.macro_f1,
        "micro_boundary_p": baseline.micro_p,
        "micro_boundary_r": baseline.micro_r,
        "micro_boundary_f1": baseline.micro_f1,
        "proposed_macro_floor_truncated": baseline.proposed_macro_floor,
        "train_subject": POLITAX_TRAIN_SUBJECT,
        "holdout_subjects": sorted(POLITAX_HOLDOUT_SUBJECTS),
    }
