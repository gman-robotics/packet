"""GMA-462: BackendConfirmer and the pipeline observe hook. Stubs only."""

from __future__ import annotations

import json

import pytest

from packet.boundary import BoundarySignal
from packet.boundary_model import featurize_from_facts
from packet.confirm import BackendConfirmer, load_confirm_config
from packet.decision import DecisionBackendError, DecisionCapabilities, DecisionResult
from packet.decision_mlx import MlxVlmDecisionBackend
from packet.eval.fixtures import build_builtin
from packet.eval.pdfutil import write_text_pdf
from packet.events import EventKind
from packet.lookahead import PageObs
from packet.pipeline import Pipeline, PipelineConfig
from packet.types import Block, BlockKind, PageClass, PageRef

FALLBACK = {"fallback_eager", "fallback_below"}
_LINES = [
    ["Alpha report body continues here with several words."],
    ["Beta discussion continues here with several words."],
    ["Gamma closing continues here with several words."],
    ["Delta appendix continues here with several words."],
    ["Epsilon endnote continues here with several words."],
]


class _Rec:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.images: list = []

    def generate(self, prompt: str, image=None) -> str:
        self.prompts.append(prompt)
        self.images.append(image)
        return self.responses.pop(0)


class _FixedScorer:
    def __init__(self, probability: float):
        self._probability = probability

    def probability(self, features):
        del features
        return self._probability


class _Spy:
    def __init__(self) -> None:
        self.observed: list[int] = []

    def observe_page(self, classified, blocks, render, page_number, page_total) -> None:
        del blocks, render, page_number, page_total
        self.observed.append(classified.index)

    def confirm(self, before, page, after):
        del before, page, after


def _ref(index: int) -> PageRef:
    return PageRef(
        packet_id="p",
        index=index,
        width=612,
        height=792,
        rotation=0,
        page_class=PageClass.TEXT,
        class_confidence=0.7,
    )


def _blocks(index: int, text: str) -> list[Block]:
    return [
        Block(
            id=f"b{index}",
            page=index,
            kind=BlockKind.PARAGRAPH,
            text=text,
            confidence=0.9,
        )
    ]


def _page(index: int, confidence: float | None = 0.5) -> PageObs:
    signal = None
    if confidence is not None:
        signal = BoundarySignal(after_page=max(index - 1, 0), confidence=confidence, reason="t")
    return PageObs(index=index, signal=signal, page_num=None, page_total=None)


def _backend(responses: list[str], *, images: bool = True):
    made: list[_Rec] = []

    def factory(_model: str, _revision: str) -> _Rec:
        runner = _Rec(responses)
        made.append(runner)
        return runner

    backend = MlxVlmDecisionBackend(
        model="m",
        revision="r",
        send_images=images,
        capabilities=DecisionCapabilities(images=images),
        runner_factory=factory,
    )
    return backend, made


def _confirmer(backend, low: float = 0.35, high: float = 0.74) -> BackendConfirmer:
    return BackendConfirmer(backend=backend, low=low, high=high, task="confirm_boundary", window=2)


def _observe(conf: BackendConfirmer, texts: list[str], renders: list) -> None:
    for index, text in enumerate(texts):
        conf.observe_page(_ref(index), _blocks(index, text), renders[index], None, None)


def _decided(pdf, config=None):
    events = []
    Pipeline(config=config).parse(pdf, on_event=events.append)
    return [event.payload for event in events if event.kind == EventKind.BOUNDARY_DECIDED]


def _events(pdf, config=None):
    events = []
    Pipeline(config=config).parse(pdf, on_event=events.append)
    return [event.model_dump(mode="json") for event in events]


def test_confirm_config_defaults_bounds_and_secrets(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"decision": {"backend": "mlx_vlm", "model": "m", "revision": "r"}, "extra": 1}))
    conf = load_confirm_config(path)
    assert (conf.low, conf.high, conf.task, conf.window) == (0.35, 0.74, "confirm_boundary", 2)
    assert isinstance(conf.backend, MlxVlmDecisionBackend)
    path.write_text(json.dumps({
        "low": True,
        "high": 0.74,
        "decision": {"backend": "mlx_vlm", "model": "m", "revision": "r"},
    }))
    with pytest.raises(DecisionBackendError, match="'low' must be a number"):
        load_confirm_config(path)
    path.write_text(json.dumps({
        "low": 0.8,
        "high": 0.2,
        "decision": {"backend": "mlx_vlm", "model": "m", "revision": "r"},
    }))
    with pytest.raises(DecisionBackendError, match="low must be <= high"):
        load_confirm_config(path)
    path.write_text(json.dumps({"low": 0.1, "high": 0.2, "decision": "mlx"}))
    with pytest.raises(DecisionBackendError, match="decision must be an object"):
        load_confirm_config(path)
    path.write_text(json.dumps({
        "token": "sek",
        "decision": {"backend": "mlx_vlm", "model": "m", "revision": "r"},
    }))
    with pytest.raises(DecisionBackendError, match="credential"):
        load_confirm_config(path)
    path.write_text("{")
    with pytest.raises(DecisionBackendError, match="malformed confirm config"):
        load_confirm_config(path)


def test_page_zero_in_band_does_not_call_backend():
    backend, made = _backend(["0.9"])
    conf = _confirmer(backend)
    assert conf.confirm(None, _page(0, 0.5), [_page(1)]) is None
    assert made == []
    assert backend.last_request is None


@pytest.mark.parametrize("confidence", [None, 0.34, 0.75])
def test_out_of_band_does_not_call_backend(confidence):
    backend, made = _backend(["0.9"])
    conf = _confirmer(backend)
    _observe(conf, ["alpha", "beta"], [lambda: None, lambda: None])
    assert conf.confirm(_page(0), _page(1, confidence), []) is None
    assert made == []


@pytest.mark.parametrize("confidence", [0.35, 0.74])
def test_inclusive_band_edges_call_backend(confidence):
    backend, made = _backend(["0.5"])
    conf = _confirmer(backend)
    _observe(conf, ["alpha", "beta"], [lambda: None, lambda: None])
    assert conf.confirm(_page(0), _page(1, confidence), []) is True
    assert len(made[0].prompts) == 1


@pytest.mark.parametrize(("text", "expected"), [("0.5", True), ("0.49", False), ("nope", None)])
def test_probability_maps_at_one_half(text, expected):
    backend, _made = _backend([text])
    conf = _confirmer(backend)
    _observe(conf, ["alpha", "beta"], [lambda: None, lambda: None])
    assert conf.confirm(_page(0), _page(1, 0.5), []) is expected
    if text == "nope":
        assert conf.disabled is False
        assert conf.degraded_pages == []
        assert backend.degraded_pages == [1]


def test_legal_request_at_middle_and_last_page():
    image_mod = pytest.importorskip("PIL.Image")
    image = image_mod.new("RGB", (2, 2), "white")
    seen: list[str] = []

    def render_middle():
        seen.append("middle")
        return image

    def render_other(label):
        def thunk():
            seen.append(label)
            return label
        return thunk

    backend, made = _backend(["0.7", "0.2"])
    conf = _confirmer(backend)
    texts = ["before line\r\nmore", "MIDDLE page body", "Z" * 700]
    renders = [render_other("before"), render_middle, render_other("after")]
    _observe(conf, texts, renders)
    assert conf.confirm(_page(0), _page(1, 0.5), [_page(2, 0.5)]) is True
    request = backend.last_request
    assert request is not None
    assert request.task == "confirm_boundary"
    assert request.page_index == 1
    assert request.input_mode == "default"
    assert len(request.facts) == 3
    assert request.ref is not None and request.ref.index == 1
    assert len(featurize_from_facts(request.facts, request.page_index)) == 16
    prompt = made[0].prompts[0]
    assert prompt == backend.build_message(request)
    assert "before_head=before line more\n" in prompt
    assert f"after_head_1={'Z' * 600}\n" in prompt
    assert "Z" * 601 not in prompt
    assert made[0].images == [image]
    assert seen == ["middle"]

    assert conf.confirm(_page(1), _page(2, 0.5), []) is False
    last = backend.last_request
    assert last is not None and last.page_index == 2
    assert len(last.facts) == 3
    featurize_from_facts(last.facts, last.page_index)
    last_prompt = made[0].prompts[1]
    assert last_prompt == backend.build_message(last)
    assert "before_head=" in last_prompt
    assert "after_head_" not in last_prompt


def test_third_strike_disables_file_and_resets_on_next_index_zero():
    backend, made = _backend(["nope", "nope", "nope", "nope", "0.8"])
    conf = _confirmer(backend)
    _observe(conf, ["a one", "b two", "c three", "d four"], [lambda: "img"] * 4)
    assert conf.confirm(_page(0), _page(1, 0.5), [_page(2)]) is None
    assert conf.confirm(_page(1), _page(2, 0.5), [_page(3)]) is None
    assert conf.disabled is False
    assert conf.confirm(_page(2), _page(3, 0.5), []) is None
    assert conf.disabled is True
    assert conf.degraded_pages == [3]
    assert backend.degraded_pages == [1, 2]
    assert len(backend.strikes) == 2
    calls = len(made[0].prompts)
    assert conf.confirm(_page(1), _page(2, 0.5), []) is None
    assert len(made[0].prompts) == calls

    conf.observe_page(_ref(0), _blocks(0, "file two"), lambda: "img", None, None)
    assert conf.disabled is False
    assert conf.degraded_pages == []
    assert backend.strikes == []
    conf.observe_page(_ref(1), _blocks(1, "next page"), lambda: "img", None, None)
    assert conf.confirm(_page(0), _page(1, 0.5), []) is None
    assert backend.strikes == [{"page": 1, "reason": "unparseable"}]
    assert conf.disabled is False


def test_load_error_propagates_and_does_not_disable():
    class _Boom:
        def generate(self, prompt, image=None):
            del prompt, image
            raise DecisionBackendError("mlx_vlm backend: failed to load")

    backend = MlxVlmDecisionBackend(
        model="m",
        revision="r",
        runner_factory=lambda _model, _revision: _Boom(),
    )
    conf = _confirmer(backend)
    _observe(conf, ["alpha", "beta"], [lambda: None, lambda: None])
    with pytest.raises(DecisionBackendError, match="failed to load"):
        conf.confirm(_page(0), _page(1, 0.5), [])
    assert conf.disabled is False
    assert conf.degraded_pages == []


def test_release_drops_confirmed_page_and_stale_renders():
    backend, _made = _backend(["0.6"], images=False)
    conf = _confirmer(backend)
    for index in range(5):
        conf.observe_page(_ref(index), _blocks(index, "alpha beta"), lambda i=index: i, None, None)
    assert 0 not in conf._stored
    assert 1 in conf._stored and 4 in conf._stored
    assert conf.confirm(_page(0), _page(1, 0.5), [_page(2), _page(3)]) is True
    assert 1 not in conf._stored
    assert 2 in conf._stored


def test_confirmer_none_events_match_default_pipeline(tmp_path):
    for fixture in build_builtin(tmp_path):
        assert _events(fixture.pdf) == _events(fixture.pdf, PipelineConfig(confirmer=None))


def test_pipeline_observes_each_page_once_in_order(tmp_path):
    pdf = write_text_pdf(tmp_path / "three.pdf", _LINES[:3])
    spy = _Spy()
    Pipeline(config=PipelineConfig(confirmer=spy)).parse(pdf)
    assert spy.observed == [0, 1, 2]


def test_backend_confirmer_changes_only_fallback_pages(tmp_path):
    pdf = write_text_pdf(tmp_path / "five.pdf", _LINES)
    backend, made = _backend(["0.9", "0.1", "0.9", "0.9"], images=False)
    conf = _confirmer(backend)
    base = _decided(pdf, PipelineConfig(scorer=_FixedScorer(0.6)))
    got = _decided(pdf, PipelineConfig(scorer=_FixedScorer(0.6), confirmer=conf))
    assert any(row["reason"] in FALLBACK for row in base)
    assert got[0] == base[0]
    for before, after in zip(base, got, strict=True):
        assert after["page"] == before["page"]
        if before["page"] == 0 or before["reason"] not in FALLBACK:
            assert after == before
            continue
        assert after["reason"] in {"confirmed_split", "confirmed_continue"}
        assert after["confidence"] == before["confidence"]
    fallback_pages = [row["page"] for row in base if row["reason"] in FALLBACK and row["page"] != 0]
    assert len(made[0].prompts) == len(fallback_pages)
    assert 0 not in [backend.last_request.page_index]


def test_pipeline_third_strike_then_next_file_resets(tmp_path):
    pdf = write_text_pdf(tmp_path / "five.pdf", _LINES)
    other = write_text_pdf(tmp_path / "other.pdf", _LINES)
    backend, made = _backend(["nope"] * 8, images=False)
    conf = _confirmer(backend)
    pipe = Pipeline(config=PipelineConfig(scorer=_FixedScorer(0.6), confirmer=conf))
    pipe.parse(pdf)
    assert len(made[0].prompts) == 3
    assert conf.disabled is True
    assert conf.degraded_pages == [3]
    assert backend.last_request is not None and backend.last_request.page_index == 3
    pipe.parse(other)
    assert len(made[0].prompts) == 6
    assert conf.degraded_pages == [3]
    assert [row["page"] for row in backend.strikes] == [1, 2]
    assert conf.disabled is True


def test_counting_backend_is_not_used_for_mapping():
    """A probability of None from the backend stays a fallback, not a bool."""

    class _NoneBackend:
        capabilities = DecisionCapabilities()
        send_images = False
        degraded_pages: list[int] = []  # noqa: RUF012

        def reset_parse_budget(self) -> None:
            self.degraded_pages.clear()

        def decide(self, request, *, render=None):
            del request, render
            return DecisionResult(probability=None)

        @property
        def backend_label(self) -> str:
            return "none"

    conf = _confirmer(_NoneBackend())
    _observe(conf, ["alpha", "beta"], [lambda: None, lambda: None])
    assert conf.confirm(_page(0), _page(1, 0.5), []) is None


class _SliceBackend:
    capabilities = DecisionCapabilities()
    send_images = False

    def __init__(self) -> None:
        self.requests = []
        self.degraded_pages: list[int] = []

    def decide(self, request, *, render=None):
        self.requests.append(request)
        return DecisionResult(probability=0.9)

    def reset_parse_budget(self) -> None:
        pass


def test_fact_slice_reaches_the_last_after_page():
    backend = _SliceBackend()
    conf = _confirmer(backend)
    _observe(conf, ["a", "b", "c", "d"], [None] * 4)
    assert conf.confirm(_page(0), _page(1, 0.5), [_page(2), _page(3)]) is True
    assert [row["text"] for row in backend.requests[0].facts] == ["a", "b", "c", "d"]
    assert conf.confirm(_page(1), _page(2, 0.5), []) is True
    assert [row["text"] for row in backend.requests[1].facts] == ["a", "b", "c"]


def test_unobserved_page_returns_none_without_a_call():
    backend = _SliceBackend()
    conf = _confirmer(backend)
    _observe(conf, ["a", "b"], [None] * 2)
    assert conf.confirm(_page(1), _page(2, 0.5), []) is None
    assert backend.requests == []


def test_confirm_config_task_and_equal_band(tmp_path):
    path = tmp_path / "c.json"
    decision = {"backend": "mlx_vlm", "model": "m", "revision": "r"}
    path.write_text(json.dumps({"task": "   ", "decision": decision}))
    with pytest.raises(DecisionBackendError, match="task"):
        load_confirm_config(path)
    path.write_text(json.dumps({"low": 0.5, "high": 0.5, "decision": decision}))
    conf = load_confirm_config(path)
    assert conf.low == conf.high == 0.5
