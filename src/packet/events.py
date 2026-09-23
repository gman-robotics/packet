from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from packet.types import Block, LogicalDocument, PageRef


class EventKind(str, Enum):
    PAGE_CLASSIFIED = "page_classified"
    BLOCK_EMITTED = "block_emitted"
    BOUNDARY_PROPOSED = "boundary_proposed"
    DOCUMENT_OPENED = "document_opened"
    SECTION_OPENED = "section_opened"
    SECTION_CLOSED = "section_closed"
    DOCUMENT_CLOSED = "document_closed"
    PACKET_CLOSED = "packet_closed"


class Event(BaseModel):
    kind: EventKind
    page: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def page_classified(cls, ref: PageRef) -> Event:
        return cls(
            kind=EventKind.PAGE_CLASSIFIED,
            page=ref.index,
            payload=ref.model_dump(),
        )

    @classmethod
    def block_emitted(cls, block: Block) -> Event:
        return cls(
            kind=EventKind.BLOCK_EMITTED,
            page=block.page,
            payload=block.model_dump(),
        )

    @classmethod
    def boundary_proposed(
        cls, after_page: int, confidence: float, reason: str
    ) -> Event:
        return cls(
            kind=EventKind.BOUNDARY_PROPOSED,
            page=after_page,
            payload={"after_page": after_page, "confidence": confidence, "reason": reason},
        )

    @classmethod
    def document_opened(cls, doc_id: str, page: int) -> Event:
        return cls(
            kind=EventKind.DOCUMENT_OPENED,
            page=page,
            payload={"document_id": doc_id},
        )

    @classmethod
    def document_closed(cls, doc: LogicalDocument) -> Event:
        return cls(
            kind=EventKind.DOCUMENT_CLOSED,
            page=doc.page_end,
            payload={"document_id": doc.id, "title": doc.title, "pages": [doc.page_start, doc.page_end]},
        )

    @classmethod
    def packet_closed(cls, packet_id: str, document_count: int) -> Event:
        return cls(
            kind=EventKind.PACKET_CLOSED,
            payload={"packet_id": packet_id, "document_count": document_count},
        )
