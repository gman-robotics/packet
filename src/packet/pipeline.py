from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from packet.assemble import Assembler
from packet.boundary import BoundaryDetector
from packet.classify import Classifier, HeuristicClassifier
from packet.domain import DomainAdapter, apply_domains
from packet.events import Event
from packet.extract import Extractor, NativeTextExtractor, Router
from packet.source import PdfPageSource
from packet.types import Block, LogicalDocument, Packet


@dataclass
class PipelineConfig:
    split_on_boundary: bool = True
    boundary_min_confidence: float = 0.55
    domain_adapters: list[DomainAdapter] = field(default_factory=list)


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
        detector = BoundaryDetector()
        current_blocks: list[Block] = []
        current_start = 0
        doc_index = 0

        with PdfPageSource(path) as source:
            packet = Packet(id=source.packet_id, origin=source.origin())
            self._packet = packet

            for ref in source.pages():
                text = source.text_for(ref.index)
                classified = self.classifier.classify(ref, text)
                yield Event.page_classified(classified)

                blocks = self.router.extract(classified, text)
                for block in blocks:
                    yield Event.block_emitted(block)

                signal = detector.observe(classified, blocks)
                should_split = (
                    self.config.split_on_boundary
                    and signal is not None
                    and signal.confidence >= self.config.boundary_min_confidence
                    and current_blocks
                )
                if signal is not None:
                    yield Event.boundary_proposed(
                        signal.after_page, signal.confidence, signal.reason
                    )

                if should_split:
                    doc = self._close_document(
                        packet, doc_index, current_blocks, current_start, classified.index - 1
                    )
                    yield Event.document_closed(doc)
                    doc_index += 1
                    current_blocks = []
                    current_start = classified.index

                if not current_blocks:
                    yield Event.document_opened(_doc_id(packet.id, doc_index), classified.index)
                current_blocks.extend(blocks)

            if current_blocks or not packet.documents:
                last_page = max(source.page_count - 1, 0)
                doc = self._close_document(
                    packet, doc_index, current_blocks, current_start, last_page
                )
                yield Event.document_closed(doc)

            yield Event.packet_closed(packet.id, packet.document_count())

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


def _doc_id(packet_id: str, index: int) -> str:
    return f"{packet_id}-doc-{index}"


def new_id() -> str:
    return uuid4().hex[:12]
