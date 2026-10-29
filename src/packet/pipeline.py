from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from packet.assemble import Assembler
from packet.boundary import BoundaryDetector
from packet.boundary_model import BoundaryScorer
from packet.citations import block_ids, unresolved_citation_ids
from packet.classify import Classifier, HeuristicClassifier
from packet.decision import DecisionBackend, decision_backend_passes_images
from packet.domain import DomainAdapter, apply_domains
from packet.events import Event
from packet.extract import Extractor, NativeTextExtractor, Router
from packet.furniture import FurnitureTagger
from packet.lookahead import Decision, LookaheadSplitter, PageObs
from packet.source import PdfPageSource
from packet.types import PACKET_FLAGS_KEY, Block, Flag, LogicalDocument, Packet, PageRef


@dataclass
class PipelineConfig:
    split_on_boundary: bool = True
    boundary_min_confidence: float = 0.55
    # Look-ahead commit: signals >= boundary_commit_confidence split at once; weaker ones
    # wait up to lookahead_window pages for page-number evidence. 0 = v0 eager commit.
    boundary_commit_confidence: float = 0.75
    lookahead_window: int = 2
    domain_adapters: list[DomainAdapter] = field(default_factory=list)
    scorer: BoundaryScorer | None = None
    decision_backend: DecisionBackend | None = None
    # GMA-443 experiment hook: page embedder for the low-weight ``embed_shift`` signal. Off by default.
    embedder: Callable[[PageRef, list[Block]], Any] | None = None
    embed_shift_max: float | None = None


class Pipeline:
    def __init__(
        self,
        classifier: Classifier | None = None,
        extractors: list[Extractor] | None = None,
        config: PipelineConfig | None = None,
    ):
        self.classifier = classifier or HeuristicClassifier()
        self.router = Router(extractors or [NativeTextExtractor()])
        self.assembler = Assembler()
        self.config = config or PipelineConfig()

    def parse(
        self, path: str | Path, on_event: Callable[[Event], None] | None = None
    ) -> Packet:
        for event in self.stream(path):
            if on_event is not None:
                on_event(event)
            if event.kind.value == "packet_closed":
                break
        assert self._packet is not None
        return self._packet

    def stream(self, path: str | Path) -> Iterator[Event]:
        self._packet = None
        detector = BoundaryDetector(embedder=self.config.embedder)
        if self.config.embed_shift_max is not None:
            detector.embed_shift_max = self.config.embed_shift_max
        splitter = LookaheadSplitter(
            window=self.config.lookahead_window,
            commit=self.config.boundary_commit_confidence,
            fallback=self.config.boundary_min_confidence,
        )
        state = _DocState()
        pending: dict[int, tuple[list[Block], PageObs]] = {}
        tagger = FurnitureTagger()

        with PdfPageSource(path) as source:
            packet = Packet(id=source.packet_id, origin=source.origin())
            self._packet = packet

            for ref in source.pages():
                text = source.text_for(ref.index)
                classified = self.classifier.classify(ref, text)
                yield Event.page_classified(classified)

                lazy_render = _lazy_render(source, classified.index)
                blocks = tagger.tag_page(
                    classified,
                    self.router.extract(
                        classified,
                        text,
                        image=lazy_render,
                        layout=_lazy_lines(source, classified.index),
                    ),
                )
                for block in blocks:
                    yield Event.block_emitted(block)

                backend = self.config.decision_backend
                decide_render = (
                    lazy_render
                    if backend is not None and decision_backend_passes_images(backend)
                    else None
                )
                signal = detector.observe(
                    classified,
                    blocks,
                    scorer=self.config.scorer,
                    decision_backend=backend,
                    render=decide_render,
                )
                if signal is not None:
                    yield Event.boundary_proposed(
                        signal.after_page, signal.confidence, signal.reason
                    )
                obs = PageObs(classified.index, signal, detector.last_page_number,
                              detector.last_page_total)
                pending[classified.index] = (blocks, obs)
                for decision in splitter.push(obs):
                    yield from self._apply(packet, state, decision, *pending.pop(decision.page), tagger)

            for decision in splitter.flush():
                yield from self._apply(packet, state, decision, *pending.pop(decision.page), tagger)

            if state.blocks or not packet.documents:
                last_page = max(source.page_count - 1, 0)
                doc = self._close_document(
                    packet, state.index, tagger.finalize(state.blocks), state.start, last_page,
                    state.pages, state.opening, eof_eligible=bool(packet.documents),
                )
                yield Event.document_closed(doc)

            yield Event.packet_closed(packet.id, packet.document_count())

    def _apply(
        self,
        packet: Packet,
        state: _DocState,
        decision: Decision,
        blocks: list[Block],
        obs: PageObs,
        tagger: FurnitureTagger,
    ) -> Iterator[Event]:
        yield Event.boundary_decided(decision, obs.signal, obs.page_num, obs.page_total)
        if self.config.split_on_boundary and decision.split and state.blocks:
            doc = self._close_document(
                packet, state.index, tagger.finalize(state.blocks), state.start, decision.page - 1,
                state.pages, state.opening, eof_eligible=False,
            )
            yield Event.document_closed(doc)
            state.index += 1
            state.blocks = []
            state.pages = []
            state.start = decision.page
            state.opened = False
            state.opening = (decision, obs)
        if not state.opened:
            yield Event.document_opened(_doc_id(packet.id, state.index), decision.page)
            state.opened = True
        state.blocks.extend(blocks)
        state.pages.append((decision, obs))

    def _close_document(
        self,
        packet: Packet,
        doc_index: int,
        blocks: list[Block],
        page_start: int,
        page_end: int,
        pages: list[tuple[Decision, PageObs]],
        opening: tuple[Decision, PageObs] | None,
        *,
        eof_eligible: bool,
    ) -> LogicalDocument:
        doc = self.assembler.build_document(_doc_id(packet.id, doc_index), blocks)
        doc.page_start = page_start
        doc.page_end = page_end
        doc = apply_domains(doc, self.config.domain_adapters)
        known: set[str] = set()
        for earlier in packet.documents:
            known.update(block_ids(earlier))
        known.update(block_ids(doc))
        fresh = flags_for_document(
            pages, opening=opening, eof_eligible=eof_eligible, doc=doc, known_ids=known,
        )
        prior = doc.extras.get(PACKET_FLAGS_KEY)
        prior_dicts = list(prior) if isinstance(prior, list) else []
        doc.extras[PACKET_FLAGS_KEY] = prior_dicts + [flag.model_dump() for flag in fresh]
        packet.documents.append(doc)
        return doc


def _lazy_lines(source: PdfPageSource, index: int):
    return lambda: source.lines_for(index)


def _lazy_render(source: PdfPageSource, index: int):
    cache: list = []

    def thunk():
        if not cache:
            cache.append(source.render(index))
        return cache[0]

    return thunk


def flags_for_document(
    pages: list[tuple[Decision, PageObs]],
    *,
    opening: tuple[Decision, PageObs] | None,
    eof_eligible: bool,
    doc: LogicalDocument,
    known_ids: set[str],
    citation_keys: set[str] | None = None,
) -> list[Flag]:
    """Uncertainty flags for one closed document. Does not change the tree.

    ``opening`` is the split that started this document after a previous one
    closed. Page 0 is not an opening split: nothing was closed before it.
    ``eof_eligible`` is true only for the final document of a multi-document packet.
    """
    flags: list[Flag] = []
    if opening is not None and _is_weak_boundary(*opening):
        decision, obs = opening
        flags.append(Flag(
            kind="weak_boundary",
            page=decision.page,
            detail={"reason": decision.reason, "signal": _signal_reason(obs)},
        ))
    for decision, obs in pages:
        if decision.reason == "fallback_below" and obs.signal is not None:
            flags.append(Flag(
                kind="fallback_decision",
                page=decision.page,
                detail={
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "signal": _signal_reason(obs),
                },
            ))
    if eof_eligible and not _has_page_number_support([obs for _, obs in pages]):
        page = pages[0][0].page if pages else doc.page_start
        flags.append(Flag(kind="eof_flush", page=page))
    for cid in unresolved_citation_ids(doc, known_ids, citation_keys):
        flags.append(Flag(kind="unresolved_citation", ref=cid))
    return flags


def _is_weak_boundary(decision: Decision, obs: PageObs) -> bool:
    """A split with no page-number evidence.

    ``fallback_eager``, or ``high_confidence`` whose signal reasons do not
    include ``page_number_reset``. Confidence is not the test: without a reset
    it is capped at 0.74, and two independent signals are lifted to 0.6.
    """
    if not decision.split:
        return False
    if decision.reason == "fallback_eager":
        return True
    if decision.reason != "high_confidence":
        return False
    return "page_number_reset" not in _signal_reasons(obs)


def _signal_reason(obs: PageObs) -> str:
    return obs.signal.reason if obs.signal is not None else ""


def _signal_reasons(obs: PageObs) -> list[str]:
    return _signal_reason(obs).split("+") if _signal_reason(obs) else []


def _has_page_number_support(pages: list[PageObs]) -> bool:
    """True when a numbered page continues a sequence started in this document.

    Every earlier numbered page is an anchor, and so is the unnumbered opening
    page. A later number that continues any of those anchors is support, so a
    numbered insert does not hide the earlier run (``1 of 3``, ``1 of 1``,
    ``2 of 3``).

    A one-page document is supported only when ``page_num`` is 1, with or
    without a ``page_number_reset``. A reset reason with no page number is not.
    """
    if not pages:
        return False
    if len(pages) == 1:
        return _starts_sequence(pages[0])
    for index, obs in enumerate(pages[1:], start=1):
        if obs.page_num is None:
            continue
        for anchor_index, anchor in enumerate(pages[:index]):
            if anchor.page_num is None and anchor_index != 0:
                continue
            if _continues_page(anchor, obs):
                return True
    return False


def _starts_sequence(obs: PageObs) -> bool:
    return obs.page_num == 1


def _continues_page(anchor: PageObs, obs: PageObs) -> bool:
    """``obs`` continues the numbering anchored on an earlier page of this document."""
    num, total = obs.page_num, obs.page_total
    if num is None or not _same_total(anchor.page_total, total):
        return False
    if anchor.page_total is not None and num > anchor.page_total:
        return False
    gap = obs.index - anchor.index
    if gap < 1:
        return False
    if anchor.page_num is None:
        return 1 <= num <= gap + 1
    return anchor.page_num < num <= anchor.page_num + gap


def _same_total(a: int | None, b: int | None) -> bool:
    return a is None or b is None or a == b


@dataclass
class _DocState:
    index: int = 0
    start: int = 0
    opened: bool = False
    blocks: list[Block] = field(default_factory=list)
    pages: list[tuple[Decision, PageObs]] = field(default_factory=list)
    opening: tuple[Decision, PageObs] | None = None


def _doc_id(packet_id: str, index: int) -> str:
    return f"{packet_id}-doc-{index}"


def new_id() -> str:
    return uuid4().hex[:12]
