"""Pure mappers from OCR engine output to Packet ``Block``s (no model imports)."""

from __future__ import annotations

import re
from typing import Any

from packet.boundary import is_page_label, match_trailing_page_label
from packet.types import BBox, Block, PageRef, SourceKind
from packet.types import BlockKind as K

LIGHTON_LABELS: dict[str, K] = {
    "text": K.PARAGRAPH, "title": K.HEADING, "list": K.LIST_ITEM, "header": K.HEADER,
    "footer": K.FOOTER, "page_number": K.PAGE_NUMBER, "footnote": K.PARAGRAPH,
    "caption": K.CAPTION, "formula": K.FORMULA, "code": K.CODE, "table": K.TABLE,
    "image": K.FIGURE, "chart": K.FIGURE, "header_image": K.HEADER, "footer_image": K.FOOTER,
    "aside_text": K.PARAGRAPH,
}
PADDLE_LABELS: dict[str, K] = {
    "doc_title": K.TITLE, "paragraph_title": K.HEADING, "text": K.PARAGRAPH, "content": K.PARAGRAPH,
    "abstract": K.PARAGRAPH, "header": K.HEADER, "footer": K.FOOTER, "number": K.PAGE_NUMBER,
    "table": K.TABLE, "image": K.FIGURE, "chart": K.FIGURE, "figure_title": K.CAPTION,
    "chart_title": K.CAPTION, "table_title": K.CAPTION, "formula": K.FORMULA,
    "display_formula": K.FORMULA, "inline_formula": K.FORMULA, "seal": K.STAMP, "algorithm": K.CODE,
    "footnote": K.PARAGRAPH, "vision_footnote": K.PARAGRAPH, "aside_text": K.PARAGRAPH,
    "reference": K.PARAGRAPH, "reference_content": K.PARAGRAPH, "header_image": K.HEADER,
    "footer_image": K.FOOTER,
}
_TAG = re.compile(r"!\[([a-z_]+)(\+?)\]\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)[ \t]*\n?")
_HASHES = re.compile(r"^(#+)\s*")
_FOOTER_HINT = re.compile(
    r"\b(confidential|proprietary|draft|internal|private)\b",
    re.IGNORECASE,
)


def _page_size(ref: PageRef) -> tuple[float, float]:
    return float(ref.width or 612.0), float(ref.height or 792.0)


def _is_footer_furniture_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or is_page_label(stripped):
        return False
    if len(stripped) > 40:
        return False
    if _FOOTER_HINT.search(stripped):
        return True
    if stripped.isupper():
        return True
    return len(stripped) <= 16 and " " not in stripped


def _split_trailing_page_furniture(text: str) -> list[tuple[K, str]]:
    """Split block text into body, page labels, and short footer lines (reading order)."""
    lines = [ln for ln in (text or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []

    tail: list[tuple[K, str]] = []
    footers: list[str] = []
    idx = len(lines) - 1
    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        if is_page_label(stripped):
            tail = [(K.PAGE_NUMBER, stripped)] + [(K.FOOTER, f) for f in footers]
            lines = lines[:idx]
            break
        if _is_footer_furniture_line(stripped):
            footers.insert(0, stripped)
            idx -= 1
            continue
        break

    pieces: list[tuple[K, str]] = []
    body_acc: list[str] = []

    def flush_body() -> None:
        if body_acc:
            pieces.append((K.PARAGRAPH, "\n".join(body_acc)))
            body_acc.clear()

    for line in lines:
        if not line.strip():
            continue
        inline = match_trailing_page_label(line)
        if inline:
            body, label = inline
            if body.strip():
                body_acc.append(body.rstrip())
            flush_body()
            pieces.append((K.PAGE_NUMBER, label))
        elif is_page_label(line.strip()):
            flush_body()
            pieces.append((K.PAGE_NUMBER, line.strip()))
        else:
            body_acc.append(line.rstrip())
    flush_body()
    pieces.extend(tail)
    return pieces


def parse_lighton_grounding(raw: str, ref: PageRef) -> list[Block]:
    """Grounding output: ``![label](x0,y0,x1,y1)`` (0-1000 page coords) before each block.
    Falls back to markdown paragraphs when the model answered without tags."""
    raw = raw or ""
    wpt, hpt = _page_size(ref)
    tags = list(_TAG.finditer(raw))
    if not tags:
        return _markdown_blocks(raw, ref)
    out: list[Block] = []
    pre = raw[: tags[0].start()].strip()
    if pre:  # text the model emitted before its first tag
        out.extend(_markdown_blocks(pre, ref))
    base = len(out)
    for j, m in enumerate(tags):
        i = base + j
        end = tags[j + 1].start() if j + 1 < len(tags) else len(raw)
        text = raw[m.end():end].strip()
        label, cont = m.group(1), m.group(2) == "+"
        kind = LIGHTON_LABELS.get(label, K.UNKNOWN)
        level = None
        if label == "title":
            h = _HASHES.match(text)
            level = len(h.group(1)) if h else 1
            text = _HASHES.sub("", text, count=1)
            if level == 1:
                kind = K.TITLE
        x0, y0, x1, y1 = (min(1000, max(0, int(m.group(k)))) for k in range(3, 7))
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        extra: dict[str, Any] = {"label": label, "continuation": cont, "engine": "lighton"}
        if kind == K.TABLE or label == "chart":
            extra["html"] = text
        out.append(Block(
            id=f"{ref.packet_id}-p{ref.index}-o{i}", page=ref.index, kind=kind, text=text,
            reading_order=i, source=SourceKind.VLM, confidence=0.9, heading_level=level,
            bbox=BBox(page=ref.index, x0=x0 / 1000 * wpt, y0=y0 / 1000 * hpt,
                      x1=x1 / 1000 * wpt, y1=y1 / 1000 * hpt),
            extra=extra,
        ))
    return _split_inline_page_furniture(out)


def _split_inline_page_furniture(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if block.kind not in {K.PARAGRAPH, K.UNKNOWN, K.HEADING, K.TITLE}:
            out.append(block)
            continue
        pieces = _split_trailing_page_furniture(block.text)
        if len(pieces) == 1 and pieces[0][1].strip() == block.text.strip():
            out.append(block)
            continue
        if not pieces:
            out.append(block)
            continue
        for kind, txt in pieces:
            out.append(
                block.model_copy(
                    update={
                        "id": f"{block.id}-f{len(out)}",
                        "kind": kind,
                        "text": txt,
                        "reading_order": len(out),
                        "heading_level": None if kind != K.HEADING else block.heading_level,
                        "extra": dict(block.extra),
                    }
                )
            )
    for i, block in enumerate(out):
        if block.reading_order != i:
            out[i] = block.model_copy(update={"reading_order": i})
    return out


def _markdown_blocks(raw: str, ref: PageRef) -> list[Block]:
    out: list[Block] = []
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    for i, p in enumerate(parts):
        kind, level = K.PARAGRAPH, None
        h = _HASHES.match(p)
        if h:
            level = len(h.group(1))
            kind = K.TITLE if level == 1 else K.HEADING
            p = _HASHES.sub("", p, count=1)
        elif p.startswith(("<table", "|")):
            kind = K.TABLE
        elif re.match(r"^([-*•]|\d+[.)])\s", p):
            kind = K.LIST_ITEM
        out.append(Block(id=f"{ref.packet_id}-p{ref.index}-o{i}", page=ref.index, kind=kind, text=p,
                         reading_order=i, source=SourceKind.VLM, confidence=0.8,
                         heading_level=level, extra={"engine": "lighton"}))
    return _split_inline_page_furniture(out)


def parse_paddle_blocks(raw: list[dict[str, Any]], ref: PageRef, dpi: float) -> list[Block]:
    """PaddleOCR-VL ``parsing_res_list`` entries (label, pixel bbox, text) -> Blocks."""
    s = 72.0 / dpi
    out: list[Block] = []
    for i, b in enumerate(raw):
        label = b.get("label") or ""
        box = b.get("bbox")
        bbox = None
        if box:
            x0, y0, x1, y1 = box
            bbox = BBox(page=ref.index, x0=x0 * s, y0=y0 * s, x1=x1 * s, y1=y1 * s)
        out.append(Block(
            id=f"{ref.packet_id}-p{ref.index}-o{i}", page=ref.index,
            kind=PADDLE_LABELS.get(label, K.UNKNOWN), text=(b.get("text") or "").strip(),
            reading_order=i, source=SourceKind.VLM, confidence=0.85, bbox=bbox,
            extra={"label": label, "engine": "paddle"},
        ))
    return out
