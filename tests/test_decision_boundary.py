"""DB-4 (GMA-263): DecisionBackend wired into boundary observe."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from packet.boundary import CAP_WITHOUT_RESET, BoundaryDetector
from packet.decision import DecisionCapabilities, DecisionRequest, DecisionResult
from packet.decision_http import HttpDecisionBackend
from packet.lookahead import Decision, LookaheadSplitter, PageObs
from packet.pipeline import Pipeline, PipelineConfig
from packet.types import Block, BlockKind, PageClass, PageRef, SourceKind

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "decision"


def _ref(i: int, **kwargs: Any) -> PageRef:
    return PageRef(packet_id="p", index=i, width=612, height=792, **kwargs)


def _block(page: int, text: str, kind: BlockKind = BlockKind.PARAGRAPH) -> Block:
    return Block(
        id=f"b{page}-{text[:12]}",
        page=page,
        kind=kind,
        text=text,
        source=SourceKind.NATIVE,
        confidence=0.8,
    )


@dataclass
class FakeDecisionBackend:
    """In-process backend: per-page probability or unparseable (``None``)."""

    probabilities: dict[int, float | None]
    capabilities: DecisionCapabilities = field(default_factory=DecisionCapabilities)
    send_images: bool = False
    degraded_pages: list[int] = field(default_factory=list)
    decide_calls: list[tuple[int, bool]] = field(default_factory=list)
    captured_bodies: list[dict[str, Any]] = field(default_factory=list)
    input_mode: str = "default"
    model: str = "local-model"

    @property
    def backend_label(self) -> str:
        return "fake"

    def reset_parse_budget(self) -> None:
        self.degraded_pages.clear()

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        self.decide_calls.append((request.page_index, render is not None))
        if self.captured_bodies is not None:
            helper = HttpDecisionBackend(
                base_url="http://127.0.0.1:1/v1",
                model=self.model,
                capabilities=self.capabilities,
                send_images=self.send_images,
                input_mode=self.input_mode,
                max_calls=100,
            )
            self.captured_bodies.append(
                helper._build_request_body(request, render=render)
            )
        prob = self.probabilities.get(request.page_index)
        if prob is None and request.page_index in self.probabilities:
            self.degraded_pages.append(request.page_index)
            return DecisionResult(probability=None, scores=None, label=None)
        if prob is None:
            return DecisionResult(probability=None, scores=None, label=None)
        return DecisionResult(probability=prob, scores=None, label=None)


def _run_silent_sequence(backend: FakeDecisionBackend, page2_text: str) -> list[Decision]:
    det = BoundaryDetector()
    splitter = LookaheadSplitter(window=2, commit=0.75, fallback=0.55)
    pages = [
        (_ref(0), [_block(0, "introductory material without a page label.")]),
        (_ref(1), [_block(1, "continuation body without numbering yet.")]),
        (_ref(2), [_block(2, page2_text, kind=BlockKind.PAGE_NUMBER)]),
    ]
    decisions: list[Decision] = []
    for ref, blocks in pages:
        sig = det.observe(ref, blocks, decision_backend=backend)
        obs = PageObs(ref.index, sig, det.last_page_number, det.last_page_total)
        decisions.extend(splitter.push(obs))
    decisions.extend(splitter.flush())
    return decisions


def test_cap_expected_fixture():
    expected = json.loads((FIXTURES / "cap_expected.json").read_text())
    backend = FakeDecisionBackend({1: 1.0})
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "Chapter one body text here.")])
    sig = det.observe(
        _ref(1),
        [_block(1, "Chapter two starts without a page label.")],
        decision_backend=backend,
    )
    assert sig is not None
    assert sig.confidence == expected["confidence"]
    assert sig.reason == expected["reason"]
    assert sig.confidence <= CAP_WITHOUT_RESET


def test_floor_expected_fixture():
    expected = json.loads((FIXTURES / "floor_expected.json").read_text())
    backend = FakeDecisionBackend({1: 0.2})
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "Chapter one body text here.")])
    sig = det.observe(
        _ref(1),
        [_block(1, "Chapter two starts without a page label.")],
        decision_backend=backend,
    )
    assert sig == expected["signal"]


def test_degraded_expected_fixture():
    expected = json.loads((FIXTURES / "degraded_expected.json").read_text())
    backend = FakeDecisionBackend({1: None})
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "5", kind=BlockKind.PAGE_NUMBER)])
    sig = det.observe(
        _ref(1),
        [_block(1, "1", kind=BlockKind.PAGE_NUMBER)],
        decision_backend=backend,
    )
    assert sig is not None
    assert sig.confidence == expected["confidence"]
    assert sig.reason == expected["reason"]
    assert "+model" not in sig.reason
    assert backend.degraded_pages == expected["decision_degraded_pages"]


def test_silent_045_expected_fixture():
    expected = json.loads((FIXTURES / "silent_045_expected.json").read_text())
    backend = FakeDecisionBackend({1: 0.45, 2: 0.10})
    decisions = _run_silent_sequence(backend, "1")
    got = [{"page": d.page, "split": d.split, "reason": d.reason} for d in decisions]
    assert got == expected["decisions"]
    assert backend.decide_calls == [(1, False), (2, False)]


def test_silent_055_expected_fixture():
    expected = json.loads((FIXTURES / "silent_055_expected.json").read_text())
    backend = FakeDecisionBackend({1: 0.55, 2: 0.10})
    decisions = _run_silent_sequence(backend, "1")
    got = [{"page": d.page, "split": d.split, "reason": d.reason} for d in decisions]
    assert got == expected["decisions"]
    assert backend.decide_calls == [(1, False), (2, False)]


def test_default_send_images_false_withholds_render_thunk_from_decide(tmp_path):
    from packet.eval.pdfutil import write_text_pdf
    from packet.extract import NativeTextExtractor, Router

    pdf = write_text_pdf(tmp_path / "two.pdf", [["Page one text."], ["Page two text."]])
    backend = FakeDecisionBackend({1: 0.5}, send_images=False)
    images_passed: list[bool] = []
    pipe = Pipeline(
        config=PipelineConfig(decision_backend=backend, split_on_boundary=False),
        extractors=[NativeTextExtractor()],
    )
    router = pipe.router

    def spy_extract(ref, text, image=None, layout=None):  # type: ignore[no-untyped-def]
        images_passed.append(image is not None)
        return Router.extract(router, ref, text, image=image, layout=layout)

    pipe.router.extract = spy_extract  # type: ignore[method-assign]
    list(pipe.stream(pdf))
    assert images_passed == [True, True]
    assert backend.decide_calls == [(1, False)]


def test_send_images_true_images_capability_false_matches_images_off_fixture():
    expected = json.loads((FIXTURES / "http_request_images_off.json").read_text())
    ref = PageRef(packet_id="p", index=1, page_class=PageClass.MIXED, class_confidence=0.45)
    block = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
    )
    backend = FakeDecisionBackend(
        {1: 0.5},
        send_images=True,
        capabilities=DecisionCapabilities(images=False),
        input_mode="default",
    )
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "")])
    with patch("packet.boundary._extract_page_number", return_value=(1, None)):
        det.observe(
            ref,
            [block],
            decision_backend=backend,
            render=lambda: (FIXTURES / "one_pixel.png").read_bytes(),
        )
    assert backend.decide_calls == [(1, False)]
    assert backend.captured_bodies[-1] == expected


def test_send_images_true_images_capability_true_passes_render_thunk():
    backend = FakeDecisionBackend(
        {1: 0.5},
        send_images=True,
        capabilities=DecisionCapabilities(images=True),
    )
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "warmup")])
    det.observe(
        _ref(1),
        [_block(1, "body")],
        decision_backend=backend,
        render=lambda: b"png-bytes",
    )
    assert backend.decide_calls == [(1, True)]


def test_backend_input_mode_features_when_request_default():
    expected = json.loads((FIXTURES / "http_request_features.json").read_text())
    backend = FakeDecisionBackend({1: 0.5}, input_mode="features")
    ref1 = PageRef(packet_id="p", index=1, page_class=PageClass.MIXED, class_confidence=0.45)
    title = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
    )
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "")])
    with patch("packet.boundary._extract_page_number", return_value=(1, None)):
        det.observe(ref1, [title], decision_backend=backend)
    assert backend.captured_bodies[-1] == expected
