from __future__ import annotations

import re
from dataclasses import dataclass, field

from packet.topic import MIN_WORDS, content_words, detect_language, topic_similarity
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
# Weak content evidence (GMA-218): never proposes alone, but counts as an independent
# signal next to a title, header change or page-number reset.
W_TOPIC_SHIFT = 0.2  # near-zero vocabulary overlap with the previous two pages
W_LANGUAGE = 0.25  # the page's language differs from the previous page's
TOPIC_SHIFT_MAX = 0.01  # cosine of content words at or below this is a topic shift
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
    _header_styles: list[tuple | None] = field(default_factory=list)
    _sizes: list[tuple[float | None, float | None, int]] = field(default_factory=list)
    _words: list = field(default_factory=list)
    _langs: list[str | None] = field(default_factory=list)

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
        self._header_styles.append(_header_style(blocks))
        self._sizes.append((ref.width, ref.height, ref.rotation))
        words = content_words(blocks)
        self._words.append(words)
        self._langs.append(detect_language(" ".join(
            b.text for b in blocks if b.kind not in _FURNITURE and b.text)))

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
        prev_style, style = self._header_styles[-2], self._header_styles[-1]
        same_template = prev_style is not None and prev_style == style
        # A layout page with no running-head line at all after one that had it (a
        # chapter opener drops its head) is an absence, not a new template.
        # Only a small-type running head counts: a body-size repeated first line that
        # stops (an inserted exhibit) is still a header change.
        # Only a chapter-opener shape (title at the top, same page geometry) counts: any
        # other page that loses the head is still a header change.
        dropped = (prev_style is not None and prev_style[2] and style is None and _is_layout(blocks)
                   and _title_position(blocks) == "top"
                   and not _size_jump(self._sizes[-2], self._sizes[-1]))
        # With layout styles on both pages, a running head whose text changes but whose
        # template (font size, position) does not is a chapter head, not a new document.
        if (self._tagged_header[-2] and header and prev_header and header != prev_header
                and not same_template and not dropped):
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

        if self._topic_shift():
            reasons.append("topic_shift")
            score += W_TOPIC_SHIFT
        prev_lang, lang = self._langs[-2], self._langs[-1]
        if prev_lang and lang and prev_lang != lang:
            reasons.append("language_changed")
            score += W_LANGUAGE

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


    def _topic_shift(self) -> bool:
        cur, prev = self._words[-1], self._words[-2]
        if sum(cur.values()) < MIN_WORDS or sum(prev.values()) < MIN_WORDS:
            return False
        # Only prose compares: tables, figures and forms carry no recognisable language
        # and their vocabulary says little about the topic.
        if not (self._langs[-1] and self._langs[-2]):
            return False
        before = prev + (self._words[-3] if len(self._words) >= 3 else type(prev)())
        return topic_similarity(before, cur) <= TOPIC_SHIFT_MAX


def _head_candidate(b: Block) -> bool:
    """A small top-band line FurnitureTagger may tag as a HEADER once its repeat is seen."""
    x = b.extra or {}
    return bool(x.get("layout") and x.get("band") == "top" and x.get("small"))


def _is_layout(blocks: list[Block]) -> bool:
    return any(b.extra.get("layout") for b in blocks)


def _header_style(blocks: list[Block]) -> tuple | None:
    """(font size, rounded top) of the page's running-head line: a layout HEADER, or a
    small top-band line not tagged yet (its repeat may lie on a page not seen yet)."""
    for b in blocks:
        x = b.extra or {}
        if x.get("layout") and (b.kind == BlockKind.HEADER
                                or (x.get("band") == "top" and x.get("small"))):
            return (x.get("size"), round(float(x.get("top") or 0) / 4), bool(x.get("small")))
    return None


def _title_position(blocks: list[Block]) -> str | None:
    """``top`` when the first non-furniture block is a TITLE, ``lower`` for any other TITLE."""
    body = [b for b in blocks if b.kind not in _FURNITURE and not _head_candidate(b)]
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
