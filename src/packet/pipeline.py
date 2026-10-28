from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from packet.assemble import Assembler
from packet.boundary import BoundaryDetector
from packet.boundary_model import BoundaryScorer
from packet.classify import Classifier, HeuristicClassifier
from packet.decision import DecisionBackend, decision_backend_passes_images
from packet.domain import DomainAdapter, apply_domains
from packet.events import Event
from packet.extract import Extractor, NativeTextExtractor, Router
from packet.furniture import FurnitureTagger
from packet.lookahead import Decision, LookaheadSplitter, PageObs
from packet.source import PdfPageSource
from packet.types import Block, LogicalDocument, Packet, PageRef


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

    def parse(self, path: str | Path) -> Packet:
        for event in self.stream(path):
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
        pending: dict[int, list[Block]] = {}
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
                pending[classified.index] = blocks
                obs = PageObs(classified.index, signal, detector.last_page_number,
                              detector.last_page_total)
                for decision in splitter.push(obs):
                    yield from self._apply(packet, state, decision, pending.pop(decision.page), tagger)

            for decision in splitter.flush():
                yield from self._apply(packet, state, decision, pending.pop(decision.page), tagger)

            if state.blocks or not packet.documents:
                last_page = max(source.page_count - 1, 0)
                doc = self._close_document(
                    packet, state.index, tagger.finalize(state.blocks), state.start, last_page
                )
                yield Event.document_closed(doc)

            yield Event.packet_closed(packet.id, packet.document_count())

    def _apply(
        self,
        packet: Packet,
        state: _DocState,
        decision: Decision,
        blocks: list[Block],
        tagger: FurnitureTagger,
    ) -> Iterator[Event]:
        if self.config.split_on_boundary and decision.split and state.blocks:
            doc = self._close_document(
                packet, state.index, tagger.finalize(state.blocks), state.start, decision.page - 1
            )
            yield Event.document_closed(doc)
            state.index += 1
            state.blocks = []
            state.start = decision.page
            state.opened = False
        if not state.opened:
            yield Event.document_opened(_doc_id(packet.id, state.index), decision.page)
            state.opened = True
        state.blocks.extend(blocks)

    def _close_document(
        self,
        packet: Packet,
        doc_index: int,
        blocks: list[Block],
        page_start: int,
        page_end: int,
    ) -> LogicalDocument:
        doc = self.assembler.build_document(_doc_id(packet.id, doc_index), blocks)
        doc.page_start = page_start
        doc.page_end = page_end
        doc = apply_domains(doc, self.config.domain_adapters)
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


@dataclass
class _DocState:
    index: int = 0
    start: int = 0
    opened: bool = False
    blocks: list[Block] = field(default_factory=list)


def _doc_id(packet_id: str, index: int) -> str:
    return f"{packet_id}-doc-{index}"


def new_id() -> str:
    return uuid4().hex[:12]
