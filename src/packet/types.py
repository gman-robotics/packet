from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PageClass(str, Enum):
    TEXT = "text"
    SCAN = "scan"
    MIXED = "mixed"
    SUSPECT_ENCODING = "suspect_encoding"
    UNKNOWN = "unknown"


class SourceKind(str, Enum):
    NATIVE = "native"
    OCR = "ocr"
    VLM = "vlm"
    MERGED = "merged"


class BlockKind(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    SIGNATURE = "signature"
    STAMP = "stamp"
    HANDWRITING = "handwriting"
    CODE = "code"
    FORMULA = "formula"
    UNKNOWN = "unknown"


class BBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    unit: str = "pt"


class PageRef(BaseModel):
    packet_id: str
    index: int
    width: float | None = None
    height: float | None = None
    rotation: int = 0
    has_text_ops: bool | None = None
    image_coverage: float | None = None
    page_class: PageClass = PageClass.UNKNOWN
    class_confidence: float = 0.0


class Block(BaseModel):
    id: str
    page: int
    kind: BlockKind
    text: str = ""
    reading_order: int = 0
    bbox: BBox | None = None
    source: SourceKind = SourceKind.NATIVE
    confidence: float = 1.0
    heading_level: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    id: str
    heading: str | None = None
    level: int = 1
    page_start: int
    page_end: int
    blocks: list[Block] = Field(default_factory=list)
    children: list[Section] = Field(default_factory=list)


class LogicalDocument(BaseModel):
    id: str
    title: str | None = None
    page_start: int
    page_end: int
    sections: list[Section] = Field(default_factory=list)
    furniture: list[Block] = Field(default_factory=list)
    type_hint: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class PacketOrigin(BaseModel):
    path: str
    sha256: str | None = None
    media_type: str = "application/pdf"
    page_count: int = 0


class Packet(BaseModel):
    id: str
    origin: PacketOrigin
    documents: list[LogicalDocument] = Field(default_factory=list)
    unclassified_pages: list[int] = Field(default_factory=list)

    def document_count(self) -> int:
        return len(self.documents)
