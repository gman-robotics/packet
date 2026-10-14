"""GMA-55 slice 5: look-ahead boundary splitting. Mid-confidence boundaries wait for up
to `window` following pages before committing; high-confidence ones commit at once."""
from __future__ import annotations

from pathlib import Path

from packet.boundary import BoundarySignal
from packet.eval.pdfutil import write_text_pdf
from packet.lookahead import LookaheadSplitter, PageObs
from packet.pipeline import Pipeline, PipelineConfig


def _obs(i, conf=None, num=None, total=None):
    sig = BoundarySignal(after_page=i - 1, confidence=conf, reason="t") if conf else None
    return PageObs(index=i, signal=sig, page_num=num, page_total=total)


def _run(obs, **kw):
    sp = LookaheadSplitter(**kw)
    out = []
    for o in obs:
        out += sp.push(o)
    return out + sp.flush()


def _splits(decisions):
    return [d.page for d in decisions if d.split]


def test_every_page_gets_exactly_one_decision_in_order():
    d = _run([_obs(0, num=1), _obs(1, num=2), _obs(2, 0.6, num=1), _obs(3, num=2)])
    assert [x.page for x in d] == [0, 1, 2, 3]


def test_high_confidence_commits_without_waiting():
    sp = LookaheadSplitter(window=2)
    sp.push(_obs(0, num=1))
    got = sp.push(_obs(1, 0.9, num=1))
    assert [(x.page, x.split) for x in got] == [(1, True)]


def test_mid_confidence_waits_then_confirms_on_new_numbering():
    sp = LookaheadSplitter(window=1)
    assert [x.page for x in sp.push(_obs(0, num=2))] == [0]
    assert sp.push(_obs(1, 0.6, num=1)) == []  # pending
    got = sp.push(_obs(2, num=2))
    assert next((x.page, x.split, x.reason) for x in got) == (1, True, "lookahead_numbering")


def test_mid_confidence_rejected_when_old_sequence_continues():
    d = _run([_obs(0, num=1, total=5), _obs(1, num=2, total=5), _obs(2, 0.7, num=None),
              _obs(3, num=4, total=5)])
    assert _splits(d) == []
    assert next(x for x in d if x.page == 2).reason == "old_sequence_continues"


def test_weak_signal_before_a_strong_one_falls_back_to_threshold():
    d = _run([_obs(0), _obs(1, 0.5), _obs(2, 0.9), _obs(3)])
    assert _splits(d) == [2]


def test_no_evidence_falls_back_to_eager_threshold():
    assert _splits(_run([_obs(0), _obs(1, 0.6)], fallback=0.55)) == [1]
    assert _splits(_run([_obs(0), _obs(1, 0.5)], fallback=0.55)) == []


def test_pipeline_does_not_split_on_reset_when_old_numbering_continues(tmp_path: Path):
    pdf = write_text_pdf(tmp_path / "a.pdf", [
        ["Report body starts here.", "Page 1 of 4"],
        ["Discussion continues.", "Page 2 of 4"],
        ["Inserted exhibit sheet.", "Page 1 of 1"],
        ["Report resumes.", "Page 3 of 4"],
    ])
    assert Pipeline().parse(pdf).document_count() == 1
    eager = Pipeline(config=PipelineConfig(lookahead_window=0)).parse(pdf)
    assert eager.document_count() > 1  # the old eager-commit behaviour


def test_pipeline_splits_when_new_numbering_continues(tmp_path: Path):
    pdf = write_text_pdf(tmp_path / "b.pdf", [
        ["Invoice body.", "Page 1 of 2"], ["Invoice end.", "Page 2 of 2"],
        ["Lease body.", "Page 1 of 3"], ["Lease more.", "Page 2 of 3"], ["Lease end.", "Page 3 of 3"],
    ])
    p = Pipeline().parse(pdf)
    assert [(d.page_start, d.page_end) for d in p.documents] == [(0, 1), (2, 4)]


def test_stream_events_stay_ordered_and_complete(tmp_path: Path):
    pdf = write_text_pdf(tmp_path / "c.pdf", [
        ["A body.", "Page 1 of 2"], ["A end.", "Page 2 of 2"], ["B body.", "Page 1 of 1"]])
    kinds = [e.kind.value for e in Pipeline().stream(pdf)]
    assert kinds[-1] == "packet_closed"
    assert kinds.count("document_opened") == kinds.count("document_closed") == 2
    assert kinds.count("page_classified") == 3


def test_builtin_eval_still_perfect(tmp_path):
    from packet.eval.run import run_named_fixtures

    assert all(r.boundary.f1 == 1.0 for r in run_named_fixtures(tmp_path))


# ---- round-1 review cases ----

def test_window_zero_is_exactly_eager_for_any_thresholds():
    for conf in (0.4, 0.5, 0.55, 0.6, 0.8, 0.95):
        for fallback in (0.55, 0.9):
            d = _run([_obs(0), _obs(1, conf)], window=0, commit=0.75, fallback=fallback)
            assert _splits(d) == ([1] if conf >= fallback else []), (conf, fallback)


def test_adjacent_one_page_doc_is_not_swallowed_by_stronger_neighbour():
    # page 1: one-page doc "1 of 1" (mid 0.6); page 2: new doc with stronger signal
    d = _run([_obs(0, num=1, total=1), _obs(1, 0.6, num=1, total=1), _obs(2, 0.9, num=1, total=3),
              _obs(3, num=2, total=3)])
    assert _splits(d) == [1, 2]


def test_consecutive_mid_signals_use_last_real_page_number():
    # report 1..2 of 4, exhibit (no number, mid), resume page "3 of 4" (mid), 4 of 4
    d = _run([_obs(0, num=1, total=4), _obs(1, num=2, total=4), _obs(2, 0.5),
              _obs(3, 0.6, num=3, total=4), _obs(4, num=4, total=4)])
    assert _splits(d) == []


def test_new_doc_with_coincident_numbers_but_different_total_still_splits():
    # old doc ends "1 of 1"; new doc cover (no number, mid 0.6); next "2 of 5"
    d = _run([_obs(0, num=1, total=1), _obs(1, 0.6), _obs(2, num=2, total=5)])
    assert _splits(d) == [1]


def test_mid_signal_on_last_page_uses_fallback():
    assert _splits(_run([_obs(0), _obs(1), _obs(2, 0.6)], window=2)) == [2]


def test_missing_numbers_everywhere_uses_fallback():
    assert _splits(_run([_obs(0), _obs(1, 0.5), _obs(2), _obs(3)])) == []


def test_splitter_memory_is_bounded():
    sp = LookaheadSplitter(window=2)
    for i in range(500):
        sp.push(_obs(i, num=i + 1))
    assert len(sp._obs) <= 3


def test_blank_pages_do_not_reopen_the_same_document(tmp_path):
    from packet.eval.pdfutil import write_blank_scan_pdf

    pdf = write_blank_scan_pdf(tmp_path / "s.pdf", page_count=3)
    kinds = [e.kind.value for e in Pipeline().stream(pdf)]
    assert kinds.count("document_opened") == 1 and kinds.count("document_closed") == 1
