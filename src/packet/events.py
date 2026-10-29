from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from packet.types import Block, LogicalDocument, PageRef

if TYPE_CHECKING:
    from packet.boundary import BoundarySignal
    from packet.lookahead import Decision


class EventKind(str, Enum):
    PAGE_CLASSIFIED = "page_classified"
    BLOCK_EMITTED = "block_emitted"
    BOUNDARY_PROPOSED = "boundary_proposed"
    BOUNDARY_DECIDED = "boundary_decided"
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
    def boundary_decided(
        cls,
        d: Decision,
        signal: BoundarySignal | None,
        page_num: int | None,
        page_total: int | None,
    ) -> Event:
        """The look-ahead verdict for page ``d.page`` (one per page, page 0 included)."""
        return cls(
            kind=EventKind.BOUNDARY_DECIDED,
            page=d.page,
            payload={
                "page": d.page,
                "split": d.split,
                "reason": d.reason,
                "confidence": d.confidence,
                "signal": (
                    None
                    if signal is None
                    else {"confidence": signal.confidence, "reason": signal.reason}
                ),
                "page_num": page_num,
                "page_total": page_total,
            },
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
