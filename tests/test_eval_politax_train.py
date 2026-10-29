"""GMA-288: train logistic weights on PoliTax trump PDFs only."""

import pytest

from packet.boundary_model import FEATURE_NAMES, LogisticBoundaryScorer, train_logistic
from packet.cli import _pipeline_for_boundary_weights
from packet.eval import politax
from packet.eval.politax_split import partition_docs
from packet.eval.politax_train import (
    TrainingRows,
    TrainingSetError,
    assert_training_set_clean,
    collect_trump_training_rows,
)
from packet.eval.run import run_named_fixtures
from tests.test_eval_politax import _fake_manifest, _write_ann


def test_assert_training_set_clean_rejects_dlnet_whole_packet_id():
    m = politax.load_manifest()
    rows = TrainingRows([], [], ["dlw-00"])
    with pytest.raises(TrainingSetError, match="dlnet-whole"):
        assert_training_set_clean(rows, m)


def test_collect_trump_rows_rejects_dlnet_whole_filename(tmp_path):
    m = _fake_manifest(tmp_path, 1, subjects=["trump"])
    orig_pdf = tmp_path / m.docs[0].file
    d0 = m.docs[0]
    m.docs[0] = politax.PoliTaxDoc(
        d0.subject,
        "dlw-00.pdf",
        "dlw-00.pdf",
        d0.url,
        d0.pages,
        d0.sha256,
        d0.license,
        d0.gold_cuts,
    )
    ann_path = _write_ann(tmp_path, m)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dlw-00.pdf").write_bytes(orig_pdf.read_bytes())
    annotations = politax.load_annotations(ann_path)
    fixtures = politax.build_fixtures(m, cache, annotations)
    by_file = {fx.notes or fx.pdf.name: fx for fx in fixtures}
    with pytest.raises(TrainingSetError, match="dlnet-whole"):
        collect_trump_training_rows(m, by_file)


def test_training_manifest_excludes_holdout_and_dlnet_names():
    m = politax.load_manifest()
    train_docs, holdout_docs = partition_docs(m)
    holdout_files = {d.file_name for d in holdout_docs}
    assert len(train_docs) == 23
    assert len(holdout_docs) == 7
    assert not holdout_files.intersection(d.file_name for d in train_docs)
    assert not any(d.file_name.startswith("dlw-") for d in train_docs)
    rows = TrainingRows([], [], [d.file_name for d in train_docs])
    assert_training_set_clean(rows, m)


def test_train_logistic_handles_empty_and_one_class_all_zero_features():
    w_empty = train_logistic([], [])
    assert len(w_empty[0]) == len(FEATURE_NAMES)
    zeros = [[0.0] * len(FEATURE_NAMES) for _ in range(5)]
    w_one = train_logistic(zeros, [0, 0, 0, 0, 0])
    assert all(c == 0.0 for c in w_one[2])


def test_eval_bad_boundary_weights_exits_2(tmp_path, capsys):
    from packet import cli

    bad = tmp_path / "bad.json"
    bad.write_text("{")
    rc = cli.main(["eval", "--boundary-weights", str(bad)])
    assert rc == 2
    assert capsys.readouterr().err.startswith("packet:")


def test_builtin_fixtures_f1_one_with_loaded_weights(tmp_path):
    from packet.boundary_model import export_weights_json
    from packet.pipeline import Pipeline

    w = train_logistic([[0.0] * len(FEATURE_NAMES)], [0])
    path = tmp_path / "neutral.json"
    export_weights_json(path, w, training_packets=["trump-only-test"])
    pipe = _pipeline_for_boundary_weights(str(path))
    assert pipe is not None
    scorer = pipe.config.scorer
    assert isinstance(scorer, LogisticBoundaryScorer)
    assert scorer.bias == pytest.approx(w[3])
    assert Pipeline().config.scorer is None
    results = run_named_fixtures(tmp_path, pipeline=pipe)
    assert len(results) == 5
    assert all(r.boundary.f1 == 1.0 for r in results)


def test_collect_trump_rows_only_from_trump_pdfs(tmp_path):
    m = _fake_manifest(tmp_path, 2, subjects=["trump", "obama"])
    ann_path = _write_ann(tmp_path, m)
    cache = tmp_path / "cache"
    cache.mkdir()
    for d in m.docs:
        (cache / d.file).write_bytes((tmp_path / d.file).read_bytes())
    annotations = politax.load_annotations(ann_path)
    all_fx = politax.build_fixtures(m, cache, annotations)
    by_file = {fx.notes: fx for fx in all_fx}
    rows = collect_trump_training_rows(m, by_file)
    assert rows.training_packets == [m.docs[0].file_name]
    assert len(rows.X) >= 1
    assert_training_set_clean(rows, m)
