from __future__ import annotations

import re
from dataclasses import dataclass, field

from packet.types import Block, BlockKind, PageRef

_PAGE_NUM = re.compile(
    r"\b(?:page\s*)?(\d+)\s*(?:of|/)\s*(\d+)\b",
    re.IGNORECASE,
)


@dataclass
class BoundarySignal:
    after_page: int
    confidence: float
    reason: str


@dataclass
class BoundaryDetector:
    """Generic, domain-free boundary hints."""

    lookback: int = 3
    _page_nums: list[tuple[int, int | None]] = field(default_factory=list)
    _headers: list[str] = field(default_factory=list)
    _sizes: list[tuple[float | None, float | None, int]] = field(default_factory=list)

    def observe(self, ref: PageRef, blocks: list[Block]) -> BoundarySignal | None:
        page_num = _extract_page_number(blocks)
        header = _fingerprint(blocks, BlockKind.HEADER) or _leading_fingerprint(blocks)
        self._page_nums.append((ref.index, page_num))
        self._headers.append(header)
        self._sizes.append((ref.width, ref.height, ref.rotation))

        if ref.index == 0:
            return None

        reasons: list[str] = []
        score = 0.0

        prev_num = self._page_nums[-2][1]
        if page_num == 1 and prev_num is not None and prev_num > 1:
            reasons.append("page_number_reset")
            score += 0.45

        prev_header = self._headers[-2]
        if header and prev_header and header != prev_header:
            reasons.append("header_changed")
            score += 0.25

        prev_size = self._sizes[-2]
        curr_size = self._sizes[-1]
        if _size_jump(prev_size, curr_size):
            reasons.append("media_box_changed")
            score += 0.2

        if any(b.kind == BlockKind.TITLE for b in blocks) and ref.index > 0:
            reasons.append("title_block")
            score += 0.25

        if score < 0.4:
            return None
        return BoundarySignal(
            after_page=ref.index - 1,
            confidence=min(score, 0.95),
            reason="+".join(reasons),
        )


def _extract_page_number(blocks: list[Block]) -> int | None:
    for block in blocks:
        is_furniture = block.kind in {BlockKind.PAGE_NUMBER, BlockKind.FOOTER, BlockKind.HEADER}
        if not is_furniture and (block.kind is not BlockKind.PARAGRAPH or len(block.text) > 40):
            continue
        match = _PAGE_NUM.search(block.text)
        if match:
            return int(match.group(1))
    for block in blocks[-2:]:
        match = _PAGE_NUM.search(block.text)
        if match:
            return int(match.group(1))
    return None


def _fingerprint(blocks: list[Block], kind: BlockKind) -> str:
    texts = [b.text.strip().lower() for b in blocks if b.kind == kind and b.text.strip()]
    return " | ".join(texts[:2])


def _leading_fingerprint(blocks: list[Block]) -> str:
    if not blocks:
        return ""
    first = blocks[0].text.strip().lower()
    return first[:80]


def _size_jump(
    prev: tuple[float | None, float | None, int],
    curr: tuple[float | None, float | None, int],
) -> bool:
    if None in prev[:2] or None in curr[:2]:
        return False
    if prev[2] != curr[2]:
        return True
    dw = abs((prev[0] or 0) - (curr[0] or 0))
    dh = abs((prev[1] or 0) - (curr[1] or 0))
    return dw > 20 or dh > 20
