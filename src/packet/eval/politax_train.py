"""Train logistic boundary weights on PoliTax ``trump`` PDFs only (GMA-288)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packet.boundary_model import Weights, export_weights_json, featurize_from_facts, train_logistic
from packet.eval.fixtures import NamedFixture
from packet.eval.metrics import document_boundaries, micro_boundary_prf
from packet.eval.page_facts import page_facts
from packet.eval.politax import Manifest
from packet.eval.politax_split import (
    POLITAX_TRAIN_SUBJECT,
    holdout_file_names,
    partition_docs,
)
from packet.eval.run import EvalResult, prepare_politax


class TrainingSetError(ValueError):
    pass


@dataclass
class TrainingRows:
    X: list[list[float]]
    y: list[int]
    training_packets: list[str]


def collect_trump_training_rows(m: Manifest, fixtures_by_file: dict) -> TrainingRows:
    """Feature rows from ``trump`` whole PDFs only (page index >= 1)."""
    train_docs, holdout_docs = partition_docs(m)
    holdout_files = {d.file_name for d in holdout_docs}
    dlnet_prefixes = ("dlw-", "packet-dlnet")
    X: list[list[float]] = []
    y: list[int] = []
    packets: list[str] = []
    for d in train_docs:
        if d.subject != POLITAX_TRAIN_SUBJECT:
            raise TrainingSetError(f"non-trump doc in train list: {d.file_name}")
        if d.file_name in holdout_files:
            raise TrainingSetError(f"holdout PDF in train list: {d.file_name}")
        if any(d.file_name.startswith(p) for p in dlnet_prefixes):
            raise TrainingSetError(f"dlnet-whole name in train list: {d.file_name}")
        fx = fixtures_by_file[d.file_name]
        facts = page_facts(fx.pdf)
        gold = document_boundaries(fx.gold.documents)
        packets.append(d.file_name)
        for i in range(1, len(facts)):
            X.append(featurize_from_facts(facts, i))
            y.append(int(i - 1 in gold))
    return TrainingRows(X, y, packets)


def assert_training_set_clean(rows: TrainingRows, m: Manifest) -> None:
    holdout = holdout_file_names(m)
    if holdout & set(rows.training_packets):
        raise TrainingSetError("training set includes a holdout PDF")
    if any(name.startswith("dlw-") for name in rows.training_packets):
        raise TrainingSetError("training set includes a dlnet-whole packet id")


def train_trump_logistic(data_dir: Path, export: Path | None = None) -> tuple[Weights, TrainingRows]:
    m, fixtures = prepare_politax(data_dir)
    by_file = {fx.notes or fx.pdf.name: fx for fx in fixtures}
    rows = collect_trump_training_rows(m, by_file)
    assert_training_set_clean(rows, m)
    w = train_logistic(rows.X, rows.y)
    if export is not None:
        export_weights_json(export, w, training_packets=rows.training_packets)
    return w, rows


@dataclass
class HoldoutScorecard:
    macro_f1: float
    micro_p: float
    micro_r: float
    micro_f1: float
    per_packet_f1: dict[str, float]


def holdout_scorecard(
    m: Manifest, fixtures: list[NamedFixture], results: list[EvalResult]
) -> HoldoutScorecard:
    from packet.eval.politax_split import holdout_results

    holdout_rows = holdout_results(m, fixtures, results)
    macro = sum(r.boundary.f1 for r in holdout_rows) / len(holdout_rows)
    mp, mr, mf = micro_boundary_prf(holdout_rows)
    holdout_files = holdout_file_names(m)
    by_file = {fx.notes or fx.pdf.name: r for fx, r in zip(fixtures, results)}
    per = {name: by_file[name].boundary.f1 for name in sorted(holdout_files) if name in by_file}
    return HoldoutScorecard(macro, mp, mr, mf, per)
