from __future__ import annotations

import re
from typing import Protocol

from packet.types import Block, BlockKind, PageClass, PageRef, SourceKind


class Extractor(Protocol):
    def supports(self, page_class: PageClass) -> bool: ...
    def extract(self, ref: PageRef, text: str) -> list[Block]: ...


_NAMED_HEADING = re.compile(
    r"^(?:chapter|part|section|article|appendix|exhibit|schedule)\s+[\w.\-]+",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)+\s+\S")
_SIMPLE_NUMBERED = re.compile(r"^\d+\.\s+[A-Z].{0,80}$")
_ROMAN_HEADING = re.compile(r"^[IVXLCM]+\.\s+\S")
_ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9 ,.'-]{8,80}$")


class NativeTextExtractor:
    """Split a page text layer into coarse blocks. No OCR."""

    def supports(self, page_class: PageClass) -> bool:
        return page_class in {PageClass.TEXT, PageClass.MIXED, PageClass.UNKNOWN}

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        if not parts and raw.strip():
            parts = [line.strip() for line in raw.split("\n") if line.strip()]

        blocks: list[Block] = []
        for order, part in enumerate(parts):
            kind, level = _guess_kind(part, order == 0 and ref.index == 0)
            blocks.append(
                Block(
                    id=f"{ref.packet_id}-p{ref.index}-b{order}",
                    page=ref.index,
                    kind=kind,
                    text=part,
                    reading_order=order,
                    source=SourceKind.NATIVE,
                    confidence=0.8 if kind != BlockKind.UNKNOWN else 0.5,
                    heading_level=level,
                )
            )
        return blocks


def _is_heading(text: str) -> bool:
    return bool(
        _NAMED_HEADING.match(text)
        or _NUMBERED_HEADING.match(text)
        or _SIMPLE_NUMBERED.match(text)
        or _ROMAN_HEADING.match(text)
        or _ALL_CAPS.match(text)
    )


def _guess_kind(text: str, maybe_title: bool) -> tuple[BlockKind, int | None]:
    lines = text.splitlines()
    compact = text.strip()
    if maybe_title and len(compact) < 140 and "\n" not in compact:
        return BlockKind.TITLE, 0
    if len(lines) == 1 and _is_heading(compact):
        level = compact.count(".") + 1 if compact[:1].isdigit() else 1
        return BlockKind.HEADING, min(level, 6)
    if compact.startswith(("- ", "* ", "• ")) or re.match(r"^\d+[.)]\s", compact):
        return BlockKind.LIST_ITEM, None
    return BlockKind.PARAGRAPH, None


class Router:
    """Send a page to the first extractor that claims it."""

    def __init__(self, extractors: list[Extractor]):
        self.extractors = extractors

    def extract(self, ref: PageRef, text: str) -> list[Block]:
        for ext in self.extractors:
            if ext.supports(ref.page_class):
                return ext.extract(ref, text)
        return []
