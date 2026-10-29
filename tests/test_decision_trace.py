"""GMA-456 (LangExtract T1): one ``boundary_decided`` event per page and ``--trace`` JSONL."""
from __future__ import annotations

import json

import pytest

from packet import cli
from packet.boundary import BoundarySignal
from packet.decision import DecisionBackendError
from packet.eval.fixtures import build_builtin
from packet.events import EventKind
from packet.lookahead import LookaheadSplitter, PageObs
from packet.pipeline import Pipeline


def _fixture(tmp_path, name):
    return next(fx for fx in build_builtin(tmp_path) if fx.name == name)


def _decided(tmp_path, name):
    fx = _fixture(tmp_path, name)
    events = []
    packet = Pipeline().parse(fx.pdf, on_event=events.append)
    decided = [e.payload for e in events if e.kind == EventKind.BOUNDARY_DECIDED]
    return packet, decided


def _assert_invariant(packet, decided):
    assert [d["page"] for d in decided] == list(range(packet.origin.page_count))
    starts = {doc.page_start for doc in packet.documents}
    assert starts == {0} | {d["page"] for d in decided if d["split"]}


@pytest.mark.parametrize("name", ["multi-doc", "shared-header"])
def test_numbered_fixtures_one_decision_per_page(tmp_path, name):
    packet, decided = _decided(tmp_path, name)
    _assert_invariant(packet, decided)
    assert {doc.page_start for doc in packet.documents} == {0, 2}
    assert decided[0]["reason"] == "no_signal"
    assert decided[0]["split"] is False


def test_unnumbered_cover_page0_implied_page_one(tmp_path):
    packet, decided = _decided(tmp_path, "unnumbered-cover")
    _assert_invariant(packet, decided)
    assert decided[0]["reason"] == "implied_page_one"
    assert decided[0]["split"] is True
    assert [doc.page_start for doc in packet.documents] == [0]


def test_boundary_decided_payload_carries_signal_and_numbers(tmp_path):
    _, decided = _decided(tmp_path, "multi-doc")
    d2 = decided[2]
    assert d2["split"] is True
    assert d2["signal"] is not None
    assert d2["confidence"] == d2["signal"]["confidence"]
    assert (d2["page_num"], d2["page_total"]) == (1, 2)
    assert decided[0]["signal"] is None and decided[0]["confidence"] == 0.0


def _obs(i, conf=None, num=None, total=None):
    sig = BoundarySignal(after_page=i - 1, confidence=conf, reason="t") if conf else None
    return PageObs(index=i, signal=sig, page_num=num, page_total=total)


def _run(obs, **kw):
    sp = LookaheadSplitter(**kw)
    out = []
    for o in obs:
        out += sp.push(o)
    return out + sp.flush()


CASES = {
    "eager": ([_obs(0, 0.6)], {"window": 0}, 0),
    "below_threshold": ([_obs(0, 0.5)], {"window": 0}, 0),
    "no_signal": ([_obs(0, num=1, total=2)], {}, 0),
    "implied_page_one": ([_obs(0), _obs(1, num=2, total=3)], {}, 0),
    "high_confidence": ([_obs(0, num=1, total=2), _obs(1, 0.9)], {}, 1),
    "continues_sequence": ([_obs(0, num=1, total=3), _obs(1, 0.5, 2, 3)], {}, 1),
    "lookahead_numbering": (
        [_obs(0, num=1, total=2), _obs(1, num=2, total=2), _obs(2, 0.5, 1, 2), _obs(3, num=2, total=2)],
        {}, 2,
    ),
    "old_sequence_continues": ([_obs(0, num=1, total=3), _obs(1, 0.5), _obs(2, num=2, total=3)], {}, 1),
    "fallback_eager": ([_obs(0, num=1, total=3), _obs(1, 0.6)], {}, 1),
    "fallback_below": ([_obs(0, num=1, total=3), _obs(1, 0.45)], {}, 1),
}


@pytest.mark.parametrize("reason", sorted(CASES))
def test_every_reason_and_confidence(reason):
    obs, kw, page = CASES[reason]
    decisions = _run(obs, **kw)
    assert [d.page for d in decisions] == [o.index for o in obs]
    d = decisions[page]
    assert d.reason == reason
    expected = obs[page].signal.confidence if obs[page].signal else 0.0
    assert d.confidence == expected


def test_parse_trace_writes_jsonl(tmp_path, capsys):
    fx = _fixture(tmp_path, "multi-doc")
    out, trace = tmp_path / "out.json", tmp_path / "trace.jsonl"
    assert cli.main(["parse", str(fx.pdf), "-o", str(out), "--trace", str(trace)]) == 0
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    kinds = [row["kind"] for row in lines]
    assert kinds.count("boundary_decided") == 4
    assert kinds[-1] == "packet_closed"
    decided = [row["payload"] for row in lines if row["kind"] == "boundary_decided"]
    d2 = decided[2]
    assert d2["split"] is True
    assert d2["page_num"] == 1
    assert d2["confidence"] == d2["signal"]["confidence"]
    assert d2["reason"] == "lookahead_numbering"


def test_parse_trace_missing_parent_exits_2(tmp_path, capsys):
    fx = _fixture(tmp_path, "multi-doc")
    trace = tmp_path / "no-such-dir" / "trace.jsonl"
    code = cli.main(
        ["parse", str(fx.pdf), "-o", str(tmp_path / "out.json"), "--trace", str(trace)]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert f"packet parse: cannot write trace {trace}: No such file or directory" in err
    assert "Traceback" not in err
    assert not trace.exists()
    assert not trace.parent.exists()


def test_parse_trace_directory_exits_2(tmp_path, capsys):
    fx = _fixture(tmp_path, "multi-doc")
    trace = tmp_path / "trace-dir"
    trace.mkdir()
    code = cli.main(
        ["parse", str(fx.pdf), "-o", str(tmp_path / "out.json"), "--trace", str(trace)]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert f"packet parse: cannot write trace {trace}: Is a directory" in err
    assert "Traceback" not in err
    assert trace.is_dir()
    assert not trace.with_name(trace.name + ".partial").exists()


def test_parse_trace_keeps_target_on_failure_and_replaces_on_success(tmp_path, monkeypatch, capsys):
    fx = _fixture(tmp_path, "multi-doc")
    out = tmp_path / "out.json"
    trace = tmp_path / "trace.jsonl"
    partial = trace.with_name(trace.name + ".partial")
    trace.write_text("PREVIOUS\n")
    original = cli.Pipeline.parse

    def fail_closed(self, path, on_event=None):
        raise DecisionBackendError("stopped")

    monkeypatch.setattr(cli.Pipeline, "parse", fail_closed)
    code = cli.main(["parse", str(fx.pdf), "-o", str(out), "--trace", str(trace)])
    err = capsys.readouterr().err
    assert code == 2
    assert f"packet parse: run failed; partial trace kept at {partial}" in err
    assert "Traceback" not in err
    assert trace.read_text() == "PREVIOUS\n"
    assert partial.is_file()

    def boom(self, path, on_event=None):
        raise RuntimeError("parse failed")

    monkeypatch.setattr(cli.Pipeline, "parse", boom)
    with pytest.raises(RuntimeError, match="parse failed"):
        cli.main(["parse", str(fx.pdf), "-o", str(out), "--trace", str(trace)])
    err = capsys.readouterr().err
    assert f"packet parse: run failed; partial trace kept at {partial}" in err
    assert "Traceback" not in err
    assert trace.read_text() == "PREVIOUS\n"
    assert partial.is_file()

    monkeypatch.setattr(cli.Pipeline, "parse", original)
    assert cli.main(["parse", str(fx.pdf), "-o", str(out), "--trace", str(trace)]) == 0
    ok = capsys.readouterr()
    assert "run failed" not in ok.err
    assert "cannot write trace" not in ok.err
    assert not partial.exists()
    text = trace.read_text()
    assert not text.startswith("PREVIOUS")
    rows = [json.loads(line) for line in text.splitlines()]
    assert any(row["kind"] == "boundary_decided" for row in rows)


def test_parse_without_trace_unchanged(tmp_path, capsys):
    fx = _fixture(tmp_path, "multi-doc")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert cli.main(["parse", str(fx.pdf), "-o", str(a)]) == 0
    assert cli.main(["parse", str(fx.pdf), "-o", str(b), "--trace", str(tmp_path / "t.jsonl")]) == 0
    strip = lambda p: {k: v for k, v in json.loads(p.read_text()).items() if k != "id"}
    assert strip(a) == strip(b)
