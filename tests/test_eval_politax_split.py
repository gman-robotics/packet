"""GMA-274: frozen PoliTax train/holdout split and holdout baseline floor proposal."""

from packet.eval import politax
from packet.eval.politax_split import (
    POLITAX_HOLDOUT_SUBJECTS,
    POLITAX_TRAIN_SUBJECT,
    assert_frozen_split,
    partition_docs,
    truncate_two_decimal_places,
)


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
