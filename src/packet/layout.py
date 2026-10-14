"""Layout signals for text-layer pages (GMA-217).

``read_lines`` turns a pdfium text page into positioned lines with font size and
weight. ``layout_blocks`` turns those lines into Blocks the way OCR pages get them:
the largest text near the top is a TITLE, short bold or larger lines are HEADINGs,
and a labelled page number ("Page 3", "3 of 9", "- 3 -") in the top or bottom band
is a PAGE_NUMBER. Anything that needs evidence from other pages (bare page
numbers, running heads and feet) is left to ``FurnitureTagger``: each band line
gets its own block at the page's first or last position, and the tagger tags it
when its text or its template (font size and position) repeats on a nearby page.
Every layout block carries its style in ``extra`` (``size``, ``bold``, ``band``,
``top``, ``small``) so later stages can compare templates, not just text.
"""

from __future__ import annotations

import ctypes
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import pypdfium2.raw as pdfium_c

from packet.types import BBox, Block, BlockKind, PageRef, SourceKind

BAND = 0.08  # top / bottom fraction of the page height that holds furniture
TITLE_RATIO = 1.3  # title font size vs body size
HEADING_RATIO = 1.12
TITLE_ZONE = 0.45  # a title starts in the top 45% of the page
_PAGE_LABELLED = re.compile(
    r"^\s*(?:(?:page|p\.)\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?|\d{1,4}\s*(?:of|/)\s*\d{1,4}"
    r"|[-–—]\s*\d{1,4}\s*[-–—])\s*$",
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


def read_lines(textpage, box_top: float, box_left: float = 0.0) -> list[TextLine]:
    """Group a pdfium text page's characters into lines, in content-stream order, with
    coordinates measured from the crop box's top-left corner. Unrotated pages only."""
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
        if ch.isspace():
            ch = " " if ch not in "\r\n" else ch  # NBSP, tabs, thin spaces separate words
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue  # pdfium soft-hyphen markers and other control/format characters
        pdfium_c.FPDFText_GetCharBox(raw, i, lft, rgt, btm, top)
        size = round(pdfium_c.FPDFText_GetFontSize(raw, i), 1)
        t, b = box_top - top.value, box_top - btm.value
        mid = (t + b) / 2
        # Compare with the line's anchor (its first visible char), not the previous
        # char, so superscripts and footnote markers do not chain into a new line.
        if ch.strip() and last_mid is not None and abs(mid - last_mid) > max(size, 1) * 0.6:
            flush()
            last_mid = None
        chars.append(ch)
        if not ch.strip():
            continue
        if last_mid is None:
            last_mid = mid
        x0, x1 = lft.value - box_left, rgt.value - box_left
        box = [min(box[0], x0), min(box[1], t), max(box[2], x1), max(box[3], b)]
        sizes[size] += 1
        pdfium_c.FPDFText_GetFontInfo(raw, i, name, len(name), flags)
        weight = pdfium_c.FPDFText_GetFontWeight(raw, i)
        fname = name.value.lower()
        if weight >= 600 or any(k in fname for k in (b"bold", b"black", b"heavy", b"semibold", b"demi")):
            bolds += 1
    flush()
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

    # Content keeps stream order (columns stay intact); band lines go to the page's
    # first and last positions, where FurnitureTagger looks for furniture.
    tops = sorted((ln for ln, t in zip(lines, in_top) if t), key=lambda ln: ln.top)
    bottoms = sorted((ln for ln, b in zip(lines, in_bottom) if b), key=lambda ln: ln.top)
    ordered = [(ln, True, False) for ln in tops] + [(ln, False, False) for ln in content]
    ordered += [(ln, False, True) for ln in bottoms]

    groups: list[tuple[BlockKind, str | None, list[TextLine]]] = []
    title_done = False
    for ln, t, b in ordered:
        band = "top" if t else "bottom" if b else None
        if band:
            groups.append((_band_kind(ln), band, [ln]))
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
                   "small": bool(body and lead.size < body * 0.95), "layout": True},
        ))
    return blocks


def _band_kind(ln: TextLine) -> BlockKind:
    """Only a labelled page number ("Page 3", "3 of 9", "- 3 -") is decided per page.
    Bare numbers and running heads/feet are left as their own block for
    FurnitureTagger, which needs corroboration from neighbouring pages."""
    return BlockKind.PAGE_NUMBER if _PAGE_LABELLED.match(ln.text) else BlockKind.PARAGRAPH


def _content_kind(ln: TextLine, body: float) -> BlockKind:
    short = len(ln.text) <= 80 and not ln.text.rstrip().endswith((".", ","))
    if short and body and (ln.size >= body * HEADING_RATIO or (ln.bold and ln.size >= body * 0.95)):
        return BlockKind.HEADING
    return BlockKind.PARAGRAPH
