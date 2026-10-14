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


def test_stronger_signal_in_window_supersedes():
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
