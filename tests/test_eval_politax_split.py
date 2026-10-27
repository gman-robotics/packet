"""GMA-274: frozen PoliTax train/holdout split and holdout baseline floor proposal."""

from pathlib import Path

from test_eval_politax import _fake_manifest, _write_ann

from packet.eval import politax
from packet.eval.metrics import BoundaryScore
from packet.eval.politax_split import (
    POLITAX_HOLDOUT_SUBJECTS,
    POLITAX_TRAIN_SUBJECT,
    HoldoutBaseline,
    assert_frozen_split,
    compute_holdout_baseline,
    format_holdout_baseline,
    partition_docs,
    truncate_two_decimal_places,
)
from packet.eval.run import EvalResult


def test_frozen_split_covers_manifest_disjoint_whole_pdfs():
    m = politax.load_manifest()
    assert_frozen_split(m)
    train, holdout = partition_docs(m)
    train_files = {d.file_name for d in train}
    holdout_files = {d.file_name for d in holdout}
    assert not train_files & holdout_files
    assert train_files | holdout_files == {d.file_name for d in m.docs}
    assert all(d.subject == POLITAX_TRAIN_SUBJECT for d in train)
    assert all(d.subject in POLITAX_HOLDOUT_SUBJECTS for d in holdout)
    assert len(train) == 23
    assert len(holdout) == 7


def test_truncate_two_decimal_places_truncates_not_rounds():
    assert truncate_two_decimal_places(0.98462) == 0.98
    assert truncate_two_decimal_places(0.999) == 0.99
    assert truncate_two_decimal_places(0.001) == 0.0


def test_q7_note_states_zero_predicted_boundaries():
    baseline = HoldoutBaseline(
        holdout_files=["obama_2009_complete_return.pdf"],
        per_subject_cuts={"obama": 1},
        macro_f1=0.0,
        micro_p=0.0,
        micro_r=0.0,
        micro_f1=0.0,
        proposed_macro_floor=0.0,
    )
    assert "0 predicted boundaries" in format_holdout_baseline(baseline)


def test_q7_note_omitted_when_not_native_baseline():
    baseline = HoldoutBaseline(
        holdout_files=["obama_2009_complete_return.pdf"],
        per_subject_cuts={"obama": 1},
        macro_f1=0.42,
        micro_p=0.4,
        micro_r=0.4,
        micro_f1=0.4,
        proposed_macro_floor=0.41,
    )
    text = format_holdout_baseline(baseline, include_q7_native_note=False)
    assert "0 predicted boundaries" not in text
    assert "macro F1 0.4200" in text


def test_holdout_baseline_excludes_train_rows(tmp_path: Path):
    m = _fake_manifest(tmp_path, 2, subjects=["trump", "obama"])
    ann_path = _write_ann(tmp_path, m)
    cache = tmp_path / "cache"
    cache.mkdir()
    for d in m.docs:
        (cache / d.file).write_bytes((tmp_path / d.file).read_bytes())
    annotations = politax.load_annotations(ann_path)
    fixtures = politax.build_fixtures(m, cache, annotations)
    train_name = next(d.file_name for d in m.docs if d.subject == POLITAX_TRAIN_SUBJECT)
    holdout_name = next(d.file_name for d in m.docs if d.subject in POLITAX_HOLDOUT_SUBJECTS)

    def row(file_name: str, f1: float) -> EvalResult:
        fx = next(f for f in fixtures if (f.notes or f.pdf.name) == file_name)
        return EvalResult(
            fx.name,
            fx.condition,
            fx.gold.origin.page_count,
            fx.gold.document_count(),
            1,
            BoundaryScore(0.0, 0.0, f1, [], [], 0),
            [],
            notes=file_name,
        )

    results = [row(train_name, 0.99), row(holdout_name, 0.0)]
    baseline = compute_holdout_baseline(m, fixtures, results)
    assert train_name not in baseline.holdout_files
    assert POLITAX_TRAIN_SUBJECT not in baseline.per_subject_cuts
    assert baseline.macro_f1 == 0.0
