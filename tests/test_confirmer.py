"""GMA-461 (LangExtract T6): mid-band confirmer seam, off by default."""
from __future__ import annotations

import pytest

from packet.boundary import BoundarySignal
from packet.eval.fixtures import build_builtin
from packet.eval.pdfutil import write_text_pdf
from packet.events import EventKind
from packet.lookahead import LookaheadSplitter, PageObs
from packet.pipeline import Pipeline, PipelineConfig

FALLBACK = {"fallback_eager", "fallback_below"}


class StubConfirmer:
    def __init__(self, verdict: bool | None):
        self.verdict = verdict
        self.calls: list[tuple[PageObs | None, PageObs, list[PageObs]]] = []

    def confirm(self, before, page, after):
        self.calls.append((before, page, list(after)))
        return self.verdict


class _FixedScorer:
    def __init__(self, probability: float):
        self._probability = probability

    def probability(self, features):
        return self._probability


def _obs(i, conf=None, num=None, total=None):
    sig = BoundarySignal(after_page=i - 1, confidence=conf, reason="t") if conf else None
    return PageObs(index=i, signal=sig, page_num=num, page_total=total)


def _run(obs, **kw):
    splitter = LookaheadSplitter(**kw)
    out = []
    for page in obs:
        out += splitter.push(page)
    return out + splitter.flush()


def _decided(pdf, config=None):
    events = []
    Pipeline(config=config).parse(pdf, on_event=events.append)
    return [event.payload for event in events if event.kind == EventKind.BOUNDARY_DECIDED]


def _streams():
    """Synthetic PageObs streams. Indexes are contiguous from 0."""
    return [
        ([_obs(0, 0.6), _obs(1), _obs(2)], {}),
        ([_obs(0, 0.45), _obs(1), _obs(2)], {}),
        ([_obs(0, 0.4), _obs(1), _obs(2)], {}),
        ([_obs(0, 0.74), _obs(1), _obs(2)], {}),
        ([_obs(0, 0.6), _obs(1), _obs(2), _obs(3)], {"window": 1}),
        ([_obs(0, num=1, total=3), _obs(1, 0.6)], {}),
        ([_obs(0, num=1, total=3), _obs(1, 0.45)], {}),
        ([_obs(0, num=1, total=3), _obs(1, 0.6), _obs(2), _obs(3)], {}),
        ([_obs(0, 0.6)], {"window": 0}),
        ([_obs(0, 0.5)], {"window": 0}),
        ([_obs(0, num=1, total=2)], {}),
        ([_obs(0), _obs(1, num=2, total=3)], {}),
        ([_obs(0, num=1, total=2), _obs(1, 0.9)], {}),
        ([_obs(0, 0.75)], {}),
        ([_obs(0, num=1, total=3), _obs(1, 0.5, 2, 3)], {}),
        (
            [
                _obs(0, num=1, total=2),
                _obs(1, num=2, total=2),
                _obs(2, 0.5, 1, 2),
                _obs(3, num=2, total=2),
            ],
            {},
        ),
        ([_obs(0, num=1, total=3), _obs(1, 0.5), _obs(2, num=2, total=3)], {}),
    ]


def _assert_decisions(base, got, verdict):
    """Flipping a fallback split updates the numbering anchor in ``LookaheadSplitter._record``.

    A later numbered page can then decide differently. Rows before the first
    fallback still match the base. The confirmer rewrites only fallback rows,
    so no other row may carry a ``fallback_*`` or ``confirmed_*`` reason.
    """
    if verdict is None:
        assert got == base
        return
    assert len(got) == len(base)
    seen_fallback = False
    confirmed = "confirmed_split" if verdict else "confirmed_continue"
    for before, after in zip(base, got):
        assert after.page == before.page
        assert after.confidence == before.confidence
        if before.reason in FALLBACK:
            assert after.split is verdict
            assert after.reason == confirmed
            seen_fallback = True
            continue
        if not seen_fallback:
            assert after == before
            continue
        assert not after.reason.startswith("fallback_")
        assert not after.reason.startswith("confirmed_")


def _assert_payloads(base, got, verdict):
    if verdict is None:
        assert got == base
        return
    assert len(got) == len(base)
    for before, after in zip(base, got):
        assert after["page"] == before["page"]
        assert after["confidence"] == before["confidence"]
        assert after["signal"] == before["signal"]
        assert after["page_num"] == before["page_num"]
        assert after["page_total"] == before["page_total"]
        if before["reason"] in FALLBACK:
            assert after["split"] is verdict
            assert after["reason"] == ("confirmed_split" if verdict else "confirmed_continue")
        else:
            assert after == before


def _assert_calls(stub, obs, base, window):
    fallback_pages = [decision.page for decision in base if decision.reason in FALLBACK]
    assert [page.index for _before, page, _after in stub.calls] == fallback_pages
    for before, page, after in stub.calls:
        assert page is obs[page.index]
        if page.index == 0:
            assert before is None
        else:
            assert before is obs[page.index - 1]
        expected = obs[page.index + 1:page.index + 1 + window]
        assert len(after) == len(expected) <= window
        assert all(item is want for item, want in zip(after, expected, strict=True))


def test_pipeline_config_confirmer_defaults_to_none():
    assert PipelineConfig().confirmer is None


def test_builtin_fixtures_default_matches_explicit_none_confirmer(tmp_path):
    for fixture in build_builtin(tmp_path):
        assert _decided(fixture.pdf) == _decided(fixture.pdf, PipelineConfig(confirmer=None))


def test_synthetic_streams_hit_both_midband_fallbacks():
    reasons = {decision.reason for obs, kw in _streams() for decision in _run(obs, **kw)}
    assert {"fallback_eager", "fallback_below"} <= reasons
    eager = _run([_obs(0, 0.6), _obs(1), _obs(2)])
    below = _run([_obs(0, 0.45), _obs(1), _obs(2)])
    assert eager[0].reason == "fallback_eager"
    assert below[0].reason == "fallback_below"


@pytest.mark.parametrize("verdict", [True, False, None])
def test_stub_changes_only_fallback_on_synthetic_streams(verdict):
    saw_fallback = set()
    for obs, kw in _streams():
        window = kw.get("window", 2)
        base = _run(obs, **kw)
        stub = StubConfirmer(verdict)
        got = _run(obs, confirmer=stub, **kw)
        saw_fallback.update(decision.reason for decision in base if decision.reason in FALLBACK)
        _assert_decisions(base, got, verdict)
        _assert_calls(stub, obs, base, window)
    assert saw_fallback == FALLBACK


@pytest.mark.parametrize("verdict", [True, False, None])
def test_stub_changes_only_fallback_on_builtin_fixtures(tmp_path, verdict):
    for fixture in build_builtin(tmp_path):
        base = _decided(fixture.pdf)
        stub = StubConfirmer(verdict)
        got = _decided(fixture.pdf, PipelineConfig(confirmer=stub))
        _assert_payloads(base, got, verdict)
        assert stub.calls == []


@pytest.mark.parametrize("verdict", [True, False, None])
def test_fixed_scorer_builtin_replay_calls_confirmer(tmp_path, verdict):
    fixture = next(item for item in build_builtin(tmp_path) if item.name == "scan")
    scorer = _FixedScorer(0.6)
    base = _decided(fixture.pdf, PipelineConfig(scorer=scorer))
    assert any(row["reason"] in FALLBACK for row in base)
    stub = StubConfirmer(verdict)
    got = _decided(fixture.pdf, PipelineConfig(scorer=scorer, confirmer=stub))
    assert stub.calls
    _assert_payloads(base, got, verdict)


def test_pipeline_passes_confirmer_only_into_midband_fallback(tmp_path):
    pdf = write_text_pdf(
        tmp_path / "mid.pdf",
        [
            ["Alpha report body continues here with several words."],
            ["Beta discussion continues here with several words."],
            ["Gamma closing continues here with several words."],
        ],
    )
    for probability, verdict in ((0.6, True), (0.45, False), (0.6, None)):
        base = _decided(pdf, PipelineConfig(scorer=_FixedScorer(probability)))
        assert any(row["reason"] in FALLBACK for row in base)
        stub = StubConfirmer(verdict)
        got = _decided(pdf, PipelineConfig(scorer=_FixedScorer(probability), confirmer=stub))
        _assert_payloads(base, got, verdict)
        call_shape = [
            (page.index, None if before is None else before.index, [item.index for item in after])
            for before, page, after in stub.calls
        ]
        assert call_shape == [(1, 0, [2]), (2, 1, [])]
        assert all(len(after) <= 2 for _before, _page, after in stub.calls)
        assert all(page.index != 0 for _before, page, _after in stub.calls)


def test_window_zero_pipeline_does_not_call_confirmer(tmp_path):
    pdf = write_text_pdf(
        tmp_path / "eager.pdf",
        [
            ["Alpha report body continues here with several words."],
            ["Beta discussion continues here with several words."],
        ],
    )
    stub = StubConfirmer(True)
    decided = _decided(
        pdf,
        PipelineConfig(lookahead_window=0, scorer=_FixedScorer(0.6), confirmer=stub),
    )
    assert stub.calls == []
    assert [row["reason"] for row in decided] == ["below_threshold", "eager"]
