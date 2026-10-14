from __future__ import annotations

import re
from dataclasses import dataclass, field

from packet.types import Block, BlockKind, PageRef

_PAGE_NUM = re.compile(
    r"\b(?:page\s*)?(\d+)\s*(?:of|/)\s*(\d+)\b",
    re.IGNORECASE,
)


_BARE_PN = re.compile(r"^\s*(?:page\s*)?(\d{1,4})\s*$", re.IGNORECASE)

# Signal weights. A signal at or above PROPOSE is proposed alone; two or more
# independent weak signals together are lifted to COMBINED, which clears the
# pipeline's eager floor (0.55) but stays below the immediate-commit level, so
# look-ahead page-number evidence can still veto them.
W_PAGE_RESET = 0.45
W_SEQUENCE_COMPLETED = 0.2
W_HEADER_CHANGED = 0.25
W_MEDIA_BOX = 0.2
W_TITLE_TOP = 0.45  # a title that opens the page (below any running header): proposes alone
W_TITLE = 0.25  # a title lower on the page
PROPOSE = 0.4
COMBINED = 0.6
# Without a page-number reset, no combination may reach the splitter's immediate
# commit level (0.75): look-ahead numbering must always get a chance to veto.
CAP_WITHOUT_RESET = 0.74


@dataclass
class BoundarySignal:
    after_page: int
    confidence: float
    reason: str


@dataclass
class BoundaryDetector:
    """Generic, domain-free boundary hints."""

    lookback: int = 3
    last_page_number: int | None = None
    last_page_total: int | None = None
    _page_nums: list[tuple[int, int | None]] = field(default_factory=list)
    _page_totals: list[int | None] = field(default_factory=list)
    _tagged_header: list[bool] = field(default_factory=list)
    _headers: list[str] = field(default_factory=list)
    _sizes: list[tuple[float | None, float | None, int]] = field(default_factory=list)

    def observe(self, ref: PageRef, blocks: list[Block]) -> BoundarySignal | None:
        page_num, page_total = _extract_page_number(blocks)
        self.last_page_number = page_num
        self.last_page_total = page_total
        self._page_totals.append(page_total)
        # header_changed only means something when the previous page carried a tagged
        # running HEADER: it fires when that header stops. Ordinary pages always have
        # different first lines, so comparing those would fire on every page pair.
        header = _leading_fingerprint(blocks)
        self._tagged_header.append(bool(_fingerprint(blocks, BlockKind.HEADER)))
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
            score += W_PAGE_RESET
            prev_total = self._page_totals[-2]
            if prev_total is not None and prev_num == prev_total:
                # previous page was "N of N": its sequence is complete
                reasons.append("sequence_completed")
                score += W_SEQUENCE_COMPLETED

        prev_header = self._headers[-2]
        if self._tagged_header[-2] and header and prev_header and header != prev_header:
            reasons.append("header_changed")
            score += W_HEADER_CHANGED

        prev_size = self._sizes[-2]
        curr_size = self._sizes[-1]
        if _size_jump(prev_size, curr_size):
            reasons.append("media_box_changed")
            score += W_MEDIA_BOX

        title = _title_position(blocks)
        if title == "top":
            reasons.append("title_top")
            score += W_TITLE_TOP
        elif title == "lower":
            reasons.append("title_block")
            score += W_TITLE

        # "sequence_completed" refines the reset, it is not independent evidence, and a
        # title lower on the page is too common inside documents (OCR labels table and
        # section captions as titles) to count toward a combination.
        independent = [r for r in reasons if r not in {"sequence_completed", "title_block"}]
        if len(independent) >= 2:
            score = max(score, COMBINED)
        if "page_number_reset" not in reasons:
            score = min(score, CAP_WITHOUT_RESET)
        if score < PROPOSE:
            return None
        return BoundarySignal(
            after_page=ref.index - 1,
            confidence=min(score, 0.95),
            reason="+".join(reasons),
        )


def _title_position(blocks: list[Block]) -> str | None:
    """``top`` when the first non-furniture block is a TITLE, ``lower`` for any other TITLE."""
    body = [b for b in blocks if b.kind not in _FURNITURE]
    if body and body[0].kind == BlockKind.TITLE:
        return "top"
    return "lower" if any(b.kind == BlockKind.TITLE for b in body) else None


def _extract_page_number(blocks: list[Block]) -> tuple[int | None, int | None]:
    # A tagged PAGE_NUMBER wins: header/footer text often holds "N of M" or "N/M"
    # (a volume label, a date) that must not shadow the printed page number.
    for block in blocks:
        if block.kind == BlockKind.PAGE_NUMBER:
            match = _PAGE_NUM.search(block.text)
            if match:
                return int(match.group(1)), int(match.group(2))
            bare = _BARE_PN.match(block.text)
            if bare:
                return int(bare.group(1)), None
    for block in blocks:
        is_furniture = block.kind in {BlockKind.PAGE_NUMBER, BlockKind.FOOTER, BlockKind.HEADER}
        if not is_furniture and (block.kind is not BlockKind.PARAGRAPH or len(block.text) > 40):
            continue
        match = _PAGE_NUM.search(block.text)
        if match:
            return int(match.group(1)), int(match.group(2))
    for block in blocks[-2:]:
        match = _PAGE_NUM.search(block.text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


_FURNITURE = {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER}


def _fingerprint(blocks: list[Block], kind: BlockKind) -> str:
    texts = [b.text.strip().lower() for b in blocks if b.kind == kind and b.text.strip()]
    return " | ".join(texts[:2])


def _leading_fingerprint(blocks: list[Block]) -> str:
    if not blocks:
        return ""
    first = blocks[0].text.strip().lower().split("\n", 1)[0]
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
