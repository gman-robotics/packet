"""Tag page furniture (HEADER / FOOTER / PAGE_NUMBER) on native-text pages.

For each all-native page the first line (top) and last line (bottom) are
candidates; when the bottom line is a page number, the line above it is the
footer candidate. A candidate line is split into its own block only when it is
tagged, so pages without furniture keep their blocks unchanged.

- PAGE_NUMBER: ``Page 3``, ``Page 3 of 9``, ``3 of 9``, ``3/9``, ``- 3 -`` or a bare
  number that is plausible as a page number (not a year, not beyond the page count).
- HEADER: the exact top line also appears as the top line of another page within
  ``lookback`` pages.
- FOOTER: the bottom candidate, digit-normalized, also appears on another page
  within ``lookback`` pages (footers often carry dates or counters).

Streaming: ``tag_page`` tags against pages already seen. ``finalize`` re-tags a
document's earlier pages (e.g. page 0) once a later repeat is known; that change
shows in ``document_closed`` / the final Packet, not in earlier ``block_emitted``
events. OCR/VLM blocks keep the engine's labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from packet.types import Block, BlockKind, PageRef, SourceKind

_PAGE_LABELLED = re.compile(
    r"^\s*(?:(?:page|p\.?)\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?|\d{1,4}\s*(?:of|/)\s*\d{1,4}|[-–—]\s*\d{1,4}\s*[-–—])\s*$",
    re.IGNORECASE,
)
_BARE = re.compile(r"^\s*(\d{1,4})\s*$")
_DIGITS = re.compile(r"\d+")
FURNITURE = {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER}
MAX_EDGE_LEN = 120


def _norm(text: str) -> str:
    return _DIGITS.sub("#", " ".join(text.lower().split()))


def bare_number(line: str) -> int | None:
    m = _BARE.match(line)
    return int(m.group(1)) if m else None


def _footer_key(line: str) -> str:
    """Footer match key: digit-normalized, but only for short lines with at least two
    words of letters (rules out list items, amounts, divider labels)."""
    if len(line) > 80 or len(re.findall(r"[A-Za-z]{2,}", line)) < 2:
        return ""
    return _norm(line)


@dataclass
class _Edges:
    top: str
    bottom: str
    labelled_pn: bool  # bottom is "Page 3", "3 of 9", "- 3 -", ...
    bare_n: int | None  # bottom is a bare number (needs corroboration)
    footer: str  # footer candidate (bottom, or the line above a page number)


@dataclass
class FurnitureTagger:
    lookback: int = 3
    _edges: dict[int, _Edges] = field(default_factory=dict)

    def tag_page(self, ref: PageRef, blocks: list[Block]) -> list[Block]:
        if not blocks or any(b.source != SourceKind.NATIVE for b in blocks):
            return blocks
        lines = [ln.strip() for ln in "\n".join(b.text for b in blocks).split("\n") if ln.strip()]
        if len(lines) < 2:  # one-line pages (dividers, blanks) carry no furniture
            return blocks
        labelled = bool(_PAGE_LABELLED.match(lines[-1]))
        bare = None if labelled else bare_number(lines[-1])
        numbered = labelled or bare is not None
        footer = (lines[-2] if len(lines) > 2 else "") if numbered else lines[-1]
        self._edges[ref.index] = _Edges(lines[0], lines[-1], labelled, bare, footer)
        return self._apply(ref.index, [b.model_copy(deep=True) for b in blocks])

    def finalize(self, blocks: list[Block]) -> list[Block]:
        """Re-tag each page of a closing document with everything learned so far."""
        out: list[Block] = []
        page_blocks: dict[int, list[Block]] = {}
        order: list[int] = []
        for b in blocks:
            if b.page not in page_blocks:
                page_blocks[b.page] = []
                order.append(b.page)
            page_blocks[b.page].append(b.model_copy(deep=True))
        for page in order:
            pb = page_blocks[page]
            if page in self._edges and all(b.source == SourceKind.NATIVE for b in pb):
                pb = self._apply(page, pb)
            out.extend(pb)
        return out

    # -- internals --
    def _repeats(self, page: int, key_of, key: str) -> bool:
        if not key or len(key) > MAX_EDGE_LEN:
            return False
        for other in range(page - self.lookback, page + self.lookback + 1):
            e = self._edges.get(other)
            if other != page and e is not None and key_of(e) == key:
                return True
        return False

    def _bare_pn_ok(self, page: int, e: _Edges) -> bool:
        """A bare number is a page number if it equals the page position, or a
        neighbouring page also ends in a bare number offset by the page distance."""
        if e.bare_n is None:
            return False
        if e.bare_n == page + 1:
            return True
        for other in range(page - self.lookback, page + self.lookback + 1):
            o = self._edges.get(other)
            if (other != page and o is not None and o.bare_n is not None
                    and o.bare_n - e.bare_n == other - page):
                return True
        return False

    def _apply(self, page: int, blocks: list[Block]) -> list[Block]:
        e = self._edges[page]
        if e.labelled_pn or self._bare_pn_ok(page, e):
            blocks = _tag_edge(blocks, "bottom", e.bottom, BlockKind.PAGE_NUMBER)
        if e.footer and self._repeats(page, lambda x: _footer_key(x.footer), _footer_key(e.footer)):
            blocks = _tag_edge(blocks, "bottom", e.footer, BlockKind.FOOTER)
        if e.top != e.footer and self._repeats(page, lambda x: x.top, e.top):
            blocks = _tag_edge(blocks, "top", e.top, BlockKind.HEADER)
        for order, b in enumerate(blocks):
            b.reading_order = order
        return blocks


def _tag_edge(blocks: list[Block], edge: str, line: str, kind: BlockKind) -> list[Block]:
    """Split ``line`` off the top (or bottom, skipping blocks already tagged) and tag it."""
    idxs = range(len(blocks)) if edge == "top" else range(len(blocks) - 1, -1, -1)
    for i in idxs:
        b = blocks[i]
        if b.kind in FURNITURE:
            if b.text == line:
                return blocks  # already tagged
            continue
        lines = b.text.split("\n")
        target = 0 if edge == "top" else len(lines) - 1
        if lines[target].strip() != line:
            return blocks
        rest = lines[1:] if edge == "top" else lines[:-1]
        if not "\n".join(rest).strip():  # whole block is the line
            b.kind, b.heading_level = kind, None
            return blocks
        piece = Block(id=f"{b.id}-{kind.value}", page=b.page, kind=kind, text=line,
                      source=b.source, confidence=b.confidence, extra={"edge": edge})
        b.text = "\n".join(rest).strip()
        return blocks[:i] + [piece, b] + blocks[i + 1:] if edge == "top" else (
            blocks[:i] + [b, piece] + blocks[i + 1:]
        )
    return blocks
