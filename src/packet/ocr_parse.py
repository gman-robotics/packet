"""Pure mappers from OCR engine output to Packet ``Block``s (no model imports).

LightOn grounding labels (`header`, `footer`, `page_number`, …) are defined on the
`lightonai/LightOnOCR-3-1B` model card (Prompt Modes / grounding); output uses
``![label](x0,y0,x1,y1)`` placeholders in 0–1000 page coordinates.
"""

from __future__ import annotations

import re
from typing import Any

from packet.boundary import keyword_page_label_fraction, match_trailing_page_label
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


def _page_size(ref: PageRef) -> tuple[float, float]:
    return float(ref.width or 612.0), float(ref.height or 792.0)


def _lighton_block_kind(label: str, text: str, kind: K) -> tuple[K, str | None]:
    if label == "page_number":
        return K.PAGE_NUMBER, None
    if label in {"header", "footer"} and keyword_page_label_fraction(text) is not None:
        return K.PAGE_NUMBER, "whole-label"
    return kind, None


def _split_trailing_page_label(
    text: str,
    body_kind: K,
    heading_level: int | None,
) -> list[tuple[K, str, int | None]]:
    """At most one trailing page label per block (``![text]`` / untagged markdown only)."""
    stripped = (text or "").strip()
    if not stripped:
        return []

    inline = match_trailing_page_label(stripped)
    if inline:
        body, label = inline
        if body.strip():
            return [(body_kind, body.strip(), heading_level), (K.PAGE_NUMBER, label, None)]
        return [(K.PAGE_NUMBER, label, None)]

    lines = stripped.splitlines()
    if len(lines) >= 2:
        last = lines[-1].strip()
        if keyword_page_label_fraction(last) is not None:
            body = "\n".join(lines[:-1]).strip()
            if body:
                return [(body_kind, body, heading_level), (K.PAGE_NUMBER, last, None)]

    return [(body_kind, stripped, heading_level)]


def _may_split_trailing_label(block: Block) -> bool:
    if block.kind == K.PAGE_NUMBER:
        return False
    label = block.extra.get("label")
    if label is not None and label != "text":
        return False
    return block.kind in {K.PARAGRAPH, K.TITLE, K.HEADING, K.LIST_ITEM, K.UNKNOWN}


def _extra_with_split(extra: dict[str, Any], split: str) -> dict[str, Any]:
    out = dict(extra)
    out["split"] = split
    return out


def _retype_if_whole_page_label(block: Block) -> Block:
    if block.kind == K.PAGE_NUMBER:
        return block
    if keyword_page_label_fraction((block.text or "").strip()) is not None:
        return block.model_copy(
            update={
                "kind": K.PAGE_NUMBER,
                "heading_level": None,
                "extra": _extra_with_split(block.extra, "whole-label"),
            }
        )
    return block


def _expand_trailing_page_labels(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        block = _retype_if_whole_page_label(block)
        if not _may_split_trailing_label(block):
            out.append(block)
            continue
        pieces = _split_trailing_page_label(block.text, block.kind, block.heading_level)
        if not pieces:
            out.append(block)
            continue
        if len(pieces) == 1:
            kind, txt, level = pieces[0]
            if kind == block.kind and txt == block.text and level == block.heading_level:
                out.append(block)
            else:
                out.append(
                    block.model_copy(
                        update={
                            "id": f"{block.id}-t{len(out)}",
                            "kind": kind,
                            "text": txt,
                            "reading_order": len(out),
                            "heading_level": level if kind != K.PAGE_NUMBER else None,
                            "extra": dict(block.extra),
                        }
                    )
                )
            continue
        for kind, txt, level in pieces:
            extra = dict(block.extra)
            if kind == K.PAGE_NUMBER:
                extra["split"] = "trailing-label"
            out.append(
                block.model_copy(
                    update={
                        "id": f"{block.id}-t{len(out)}",
                        "kind": kind,
                        "text": txt,
                        "reading_order": len(out),
                        "heading_level": level if kind != K.PAGE_NUMBER else None,
                        "extra": extra,
                    }
                )
            )
    for i, block in enumerate(out):
        if block.reading_order != i:
            out[i] = block.model_copy(update={"reading_order": i})
    return out


def parse_lighton_grounding(raw: str, ref: PageRef) -> list[Block]:
    """Grounding output: ``![label](x0,y0,x1,y1)`` (0-1000 page coords) before each block.
    Falls back to markdown paragraphs when the model answered without tags."""
    raw = raw or ""
    wpt, hpt = _page_size(ref)
    tags = list(_TAG.finditer(raw))
    if not tags:
        return _expand_trailing_page_labels(_markdown_blocks(raw, ref, expand_trailing=False))
    out: list[Block] = []
    pre = raw[: tags[0].start()].strip()
    if pre:  # text the model emitted before its first tag
        out.extend(_markdown_blocks(pre, ref, expand_trailing=False))
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
        kind, split_key = _lighton_block_kind(label, text, kind)
        x0, y0, x1, y1 = (min(1000, max(0, int(m.group(k)))) for k in range(3, 7))
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        extra: dict[str, Any] = {"label": label, "continuation": cont, "engine": "lighton"}
        if split_key:
            extra = _extra_with_split(extra, split_key)
        if kind == K.TABLE or label == "chart":
            extra["html"] = text
        out.append(Block(
            id=f"{ref.packet_id}-p{ref.index}-o{i}", page=ref.index, kind=kind, text=text,
            reading_order=i, source=SourceKind.VLM, confidence=0.9, heading_level=level,
            bbox=BBox(page=ref.index, x0=x0 / 1000 * wpt, y0=y0 / 1000 * hpt,
                      x1=x1 / 1000 * wpt, y1=y1 / 1000 * hpt),
            extra=extra,
        ))
    return _expand_trailing_page_labels(out)


def _markdown_blocks(raw: str, ref: PageRef, *, expand_trailing: bool = True) -> list[Block]:
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
    if expand_trailing:
        return _expand_trailing_page_labels(out)
    return out


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
