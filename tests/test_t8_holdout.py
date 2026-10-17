"""T8 (GMA-256): whole-packet holdout boundary model, decision 2C.

Pure logic only: no PDFs, no network. The real run is ``python bench/t8_holdout.py``
and its note is ``bench/t8_holdout.md``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import t8_holdout as t8

F1 = {"dlw-00": 1.0, "dlw-01": 1.0, "dlw-02": 1.0, "dlw-03": 1.0, "dlw-04": 0.8, "dlw-05": 1.0, "dlw-12": 0.8}


def test_split_scored_and_train_are_disjoint_whole_packets_with_every_miss_scored():
    s = t8.split_packets(F1)
    assert s.miss == ["dlw-04", "dlw-12"]
    assert s.saturated == ["dlw-00", "dlw-01"]  # the two lowest-id packets at 1.0
    assert s.scored == ["dlw-00", "dlw-01", "dlw-04", "dlw-12"]
    assert s.train == ["dlw-02", "dlw-03", "dlw-05"]
    assert not set(s.scored) & set(s.train)
    assert set(s.scored) | set(s.train) == set(F1)
    for m in s.miss:
        assert m in s.scored and m not in s.train  # never train on a packet still under 1.0


def test_split_never_puts_a_saturated_packet_in_the_miss_slot():
    s = t8.split_packets({"dlw-00": 1.0, "dlw-01": 1.0, "dlw-02": 1.0})
    assert s.miss == [] and s.saturated == ["dlw-00", "dlw-01"]


def test_split_rejects_a_doc_shared_by_a_scored_and_a_train_packet():
    with pytest.raises(ValueError, match="1001.1"):
        t8.check_doc_disjoint(t8.split_packets(F1), {"dlw-12": ["1001.1"], "dlw-02": ["1001.1"]})


def test_residual_shape_is_a_gold_cut_with_no_page_number_and_no_signal():
    rows = [
        {"packet": "dlw-02", "page": 3, "gold_cut": True, "page_number": None, "signal": None},
        {"packet": "dlw-02", "page": 4, "gold_cut": True, "page_number": 1, "signal": None},
        {"packet": "dlw-02", "page": 5, "gold_cut": True, "page_number": None, "signal": [4, 0.5, "title_top"]},
        {"packet": "dlw-02", "page": 6, "gold_cut": False, "page_number": None, "signal": None},
        {"packet": "dlw-12", "page": 25, "gold_cut": True, "page_number": None, "signal": None},
    ]
    assert t8.residual_positives(rows, train=["dlw-02"]) == [("dlw-02", 3)]


def test_features_exclude_page_index_file_name_and_heuristic_confidence():
    for name in t8.FEATURES:
        assert not any(bad in name for bad in ("index", "file", "name", "confidence", "conf"))


def test_logistic_is_deterministic_and_learns_a_separable_toy_set():
    X = [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.2]]
    y = [0, 0, 1, 1]
    w1, w2 = t8.train_logistic(X, y), t8.train_logistic(X, y)
    assert w1 == w2
    assert t8.predict(w1, [1.0, 0.0]) > 0.5 > t8.predict(w1, [0.0, 1.0])


def test_model_score_is_capped_below_immediate_commit_and_never_lowers_the_heuristic():
    assert t8.combined_confidence(None, 0.99) == pytest.approx(0.74)
    assert t8.combined_confidence(0.9, 0.1) == pytest.approx(0.9)
    assert t8.combined_confidence(None, 0.2) is None  # below the propose level: no signal


SAT, MISS = ["dlw-00", "dlw-01"], ["dlw-12"]
HEUR = {"dlw-00": 1.0, "dlw-01": 1.0, "dlw-12": 0.8}


@pytest.mark.parametrize("model, positives, ship", [
    ({"dlw-00": 1.0, "dlw-01": 1.0, "dlw-12": 1.0}, 14, True),
    ({"dlw-00": 1.0, "dlw-01": 1.0, "dlw-12": 1.0}, 0, False),  # decision 10A: zero positives
    ({"dlw-00": 0.9, "dlw-01": 1.0, "dlw-12": 1.0}, 14, False),  # a saturated packet dropped
    ({"dlw-00": 1.0, "dlw-01": 1.0, "dlw-12": 0.8}, 14, False),  # the miss did not go strictly up
])
def test_ship_rule_2c(model, positives, ship):
    out = t8.ship_decision(HEUR, model, saturated=SAT, miss=MISS, residual_positives=positives,
                           heur_micro=(12, 13, 13), model_micro=(13, 13, 13) if model["dlw-12"] > 0.8 else (12, 13, 13))
    assert out["ship"] is ship


def test_ship_rule_requires_micro_to_rise_too():
    out = t8.ship_decision(HEUR, {"dlw-00": 1.0, "dlw-01": 1.0, "dlw-12": 1.0}, saturated=SAT, miss=MISS,
                           residual_positives=14, heur_micro=(12, 13, 13), model_micro=(12, 13, 13))
    assert out["ship"] is False


def test_training_rows_are_real_only():
    rows = [{"packet": "dlw-02", "page": 1, "synthetic": False}, {"packet": "dlw-02", "page": 2, "synthetic": True}]
    with pytest.raises(ValueError, match="synthetic"):
        t8.assert_real_only(rows)
