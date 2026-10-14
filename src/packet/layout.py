"""Layout signals for text-layer pages (GMA-217).

``read_lines`` turns a pdfium text page into positioned lines with font size and
weight. ``layout_blocks`` turns those lines into Blocks the same way OCR pages get
them: the largest text near the top is a TITLE, short bold or larger lines are
HEADINGs, a bare number in the top or bottom band is a PAGE_NUMBER, and a line in a
band set smaller than the body is a HEADER or FOOTER. Repetition across pages is
left to ``FurnitureTagger`` (band lines get their own block, so a repeated band
line is the page's first or last line). Every layout block carries its style in
``extra`` (``size``, ``bold``, ``band``, ``top``) so later stages can compare
templates, not just text.
"""

from __future__ import annotations

import ctypes
import re
from collections import Counter
from dataclasses import dataclass

import pypdfium2.raw as pdfium_c

from packet.types import BBox, Block, BlockKind, PageRef, SourceKind

BAND = 0.08  # top / bottom fraction of the page height that holds furniture
TITLE_RATIO = 1.3  # title font size vs body size
HEADING_RATIO = 1.12
TITLE_ZONE = 0.45  # a title starts in the top 45% of the page
_PAGE_LABEL = re.compile(
    r"^\s*(?:(?:page|p\.?)\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$|^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$",
    re.IGNORECASE,
)


@dataclass
class TextLine:
    text: str
    x0: float
    top: float  # distance from the page's top edge to the line's top
    x1: float
    bottom: float
    size: float
    bold: bool


def read_lines(textpage, page_height: float) -> list[TextLine]:
    """Group a pdfium text page's characters into lines (top-down coordinates)."""
    raw = textpage.raw
    n = pdfium_c.FPDFText_CountChars(raw)
    lines: list[TextLine] = []
    chars: list[str] = []
    box = [1e9, 1e9, -1e9, -1e9]  # x0, top, x1, bottom
    sizes: Counter = Counter()
    bolds = 0
    lft, rgt, btm, top = (ctypes.c_double() for _ in range(4))
    name = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()

    def flush() -> None:
        nonlocal chars, box, sizes, bolds
        text = "".join(chars).strip()
        if text and sizes:
            size = sizes.most_common(1)[0][0]
            lines.append(TextLine(text, box[0], box[1], box[2], box[3], size,
                                  bolds * 2 > sum(sizes.values())))
        chars, box, sizes, bolds = [], [1e9, 1e9, -1e9, -1e9], Counter(), 0

    last_mid = None
    for i in range(n):
        ch = chr(pdfium_c.FPDFText_GetUnicode(raw, i))
        if ch in "\r\n":
            flush()
            last_mid = None
            continue
        pdfium_c.FPDFText_GetCharBox(raw, i, lft, rgt, btm, top)
        size = round(pdfium_c.FPDFText_GetFontSize(raw, i), 1)
        t, b = page_height - top.value, page_height - btm.value
        mid = (t + b) / 2
        if ch.strip() and last_mid is not None and abs(mid - last_mid) > max(size, 1) * 0.6:
            flush()
        chars.append(ch)
        if not ch.strip():
            continue
        last_mid = mid
        box = [min(box[0], lft.value), min(box[1], t), max(box[2], rgt.value), max(box[3], b)]
        sizes[size] += 1
        pdfium_c.FPDFText_GetFontInfo(raw, i, name, len(name), flags)
        weight = pdfium_c.FPDFText_GetFontWeight(raw, i)
        if weight >= 600 or b"bold" in name.value.lower() or b"black" in name.value.lower():
            bolds += 1
    flush()
    lines.sort(key=lambda ln: (round(ln.top), ln.x0))
    return lines


def body_size(lines: list[TextLine]) -> float:
    """The font size that carries the most characters."""
    weight: Counter = Counter()
    for ln in lines:
        weight[ln.size] += len(ln.text)
    return weight.most_common(1)[0][0] if weight else 0.0


def layout_blocks(ref: PageRef, lines: list[TextLine]) -> list[Block]:
    height = ref.height or 792.0
    body = body_size(lines)
    in_top = [ln.bottom <= height * BAND for ln in lines]
    in_bottom = [ln.top >= height * (1 - BAND) for ln in lines]
    content = [ln for ln, t, b in zip(lines, in_top, in_bottom) if not (t or b)]
    biggest = max((ln.size for ln in content), default=0.0)
    title_size = biggest if body and biggest >= body * TITLE_RATIO else None

    groups: list[tuple[BlockKind, str | None, list[TextLine]]] = []
    title_done = False
    for ln, t, b in zip(lines, in_top, in_bottom):
        band = "top" if t else "bottom" if b else None
        if band:
            kind = _band_kind(ln, body)
            groups.append((kind, band, [ln]))
            continue
        if (title_size and not title_done and ln.size == title_size
                and ln.top <= height * TITLE_ZONE):
            if groups and groups[-1][0] == BlockKind.TITLE:
                groups[-1][2].append(ln)
            else:
                groups.append((BlockKind.TITLE, None, [ln]))
            continue
        if groups and groups[-1][0] == BlockKind.TITLE:
            title_done = True
        kind = _content_kind(ln, body)
        prev = groups[-1] if groups else None
        if (kind == BlockKind.PARAGRAPH and prev and prev[0] == BlockKind.PARAGRAPH
                and prev[1] is None and ln.top - prev[2][-1].bottom < max(ln.size, 1) * 1.2):
            prev[2].append(ln)
        else:
            groups.append((kind, None, [ln]))

    blocks: list[Block] = []
    for order, (kind, band, glines) in enumerate(groups):
        lead = glines[0]
        blocks.append(Block(
            id=f"{ref.packet_id}-p{ref.index}-b{order}",
            page=ref.index,
            kind=kind,
            text="\n".join(g.text for g in glines),
            reading_order=order,
            bbox=BBox(page=ref.index, x0=min(g.x0 for g in glines), y0=lead.top,
                      x1=max(g.x1 for g in glines), y1=glines[-1].bottom),
            source=SourceKind.NATIVE,
            confidence=0.8,
            heading_level=0 if kind == BlockKind.TITLE else 1 if kind == BlockKind.HEADING else None,
            extra={"size": lead.size, "bold": lead.bold, "band": band, "top": round(lead.top, 1),
                   "layout": True},
        ))
    return blocks


def _band_kind(ln: TextLine, body: float) -> BlockKind:
    if _PAGE_LABEL.match(ln.text):
        return BlockKind.PAGE_NUMBER
    if body and ln.size < body * 0.95 and len(ln.text) <= 120:
        return BlockKind.HEADER if ln.top < 400 else BlockKind.FOOTER
    return BlockKind.PARAGRAPH  # left to FurnitureTagger's repetition check


def _content_kind(ln: TextLine, body: float) -> BlockKind:
    short = len(ln.text) <= 80 and not ln.text.rstrip().endswith((".", ","))
    if short and body and (ln.size >= body * HEADING_RATIO or (ln.bold and ln.size >= body * 0.95)):
        return BlockKind.HEADING
    return BlockKind.PARAGRAPH
