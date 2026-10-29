"""Self-contained HTML review of a parse trace. No network and no external assets."""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import ValidationError

from packet.events import Event, EventKind
from packet.types import Block, BlockKind, LogicalDocument, Packet, Section

IMAGE_SKIP_NOTE = "Page images skipped: rendering failed."
_LONGEST_PX = 800
_FURNITURE = frozenset({"header", "footer", "page_number"})
_FURNITURE_KINDS = frozenset({BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER})
CHECK_SECTION_OUTSIDE = "section_outside_document"
CHECK_FURNITURE_IN_BODY = "furniture_in_body"
CHECK_TITLE_EQUALS_FURNITURE = "title_equals_furniture"

_CSS = """
body { margin: 0; padding: 1.5rem; font: 15px/1.4 system-ui, sans-serif;
  background: #f4f5f7; color: #1c1917; }
h1 { font-size: 1.4rem; margin: 0 0 0.5rem; }
h2 { font-size: 1.1rem; margin: 0 0 0.4rem; }
h3 { font-size: 1rem; margin: 0 0 0.3rem; }
p { margin: 0.25rem 0; }
.note { background: #fff7ed; border: 1px solid #fdba74; padding: 0.5rem 0.75rem; }
.strip { display: flex; flex-direction: column; gap: 1rem; }
.page, .step { background: #fff; border: 1px solid #d6d3d1; padding: 0.75rem; }
.frame { position: relative; width: min(100%, 800px); background: #fafaf9;
  border: 1px solid #e7e5e4; }
.frame.has-size img, .frame.has-size .box { position: absolute; }
.frame.has-size img { inset: 0; width: 100%; height: 100%; object-fit: fill; }
.frame img.flow { display: block; width: 100%; height: auto; }
.box { box-sizing: border-box; overflow: hidden; font-size: 10px; line-height: 1.15; }
.block { white-space: pre-wrap; }
.unboxed, .sections, .sec-blocks { margin: 0.25rem 0 0.25rem 1rem; padding: 0; }
.kind-title { background: #fde68a; }
.kind-heading { background: #fcd34d; }
.kind-paragraph { background: #e0f2fe; }
.kind-list_item { background: #dbeafe; }
.kind-table { background: #d1fae5; }
.kind-figure { background: #ede9fe; }
.kind-caption { background: #fae8ff; }
.kind-header { background: #fecaca; }
.kind-footer { background: #fed7aa; }
.kind-page_number { background: #fbcfe8; }
.kind-signature { background: #e5e7eb; }
.kind-stamp { background: #fee2e2; }
.kind-handwriting { background: #ffedd5; }
.kind-code { background: #ecfccb; }
.kind-formula { background: #cffafe; }
.kind-unknown { background: #f3f4f6; }
.furniture { outline: 2px dashed #9f1239; }
.mark-TP { color: #166534; font-weight: 700; }
.mark-FP { color: #9a3412; font-weight: 700; }
.mark-FN { color: #9f1239; font-weight: 700; }
.stepper { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 1.5rem; }
.decision { margin-top: 0.5rem; font-family: ui-monospace, monospace; font-size: 13px; }
.checks { margin-top: 1.5rem; }
.check-hit { background: #fff7ed; border: 1px solid #fdba74; padding: 0.35rem 0.6rem;
  margin: 0.25rem 0; }
"""


@dataclass(frozen=True)
class ReviewFinding:
    """One metric-blind review hit. Checks never change a split or a score."""

    check: str
    document_id: str
    detail: str


def review_checks(packet: Packet) -> list[ReviewFinding]:
    """Section range, furniture text in body blocks, and title equal to furniture.

    ``render_html`` shows this same list. Callers can use the return value without
    parsing the HTML.
    """
    findings: list[ReviewFinding] = []
    for doc in packet.documents:
        findings.extend(_section_range_findings(doc))
        furniture = _furniture_strings(doc)
        findings.extend(_furniture_in_body_findings(doc, furniture))
        findings.extend(_title_findings(doc, furniture))
    return findings


def _walk_sections(sections: list[Section]):
    for section in sections:
        yield section
        yield from _walk_sections(section.children)


def _norm_line(text: str) -> str:
    return " ".join(text.split())


def _section_range_findings(doc: LogicalDocument) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for section in _walk_sections(doc.sections):
        if section.page_start < doc.page_start or section.page_end > doc.page_end:
            findings.append(ReviewFinding(
                CHECK_SECTION_OUTSIDE,
                doc.id,
                f"section {section.id} pages {section.page_start}-{section.page_end} "
                f"outside document pages {doc.page_start}-{doc.page_end}",
            ))
    return findings


def _furniture_blocks(doc: LogicalDocument) -> list[Block]:
    blocks = [block for block in doc.furniture if block.kind in _FURNITURE_KINDS]
    for section in _walk_sections(doc.sections):
        blocks.extend(block for block in section.blocks if block.kind in _FURNITURE_KINDS)
    return blocks


def _furniture_strings(doc: LogicalDocument) -> set[str]:
    return {text for block in _furniture_blocks(doc) if (text := _norm_line(block.text))}


def _furniture_in_body_findings(doc: LogicalDocument, furniture: set[str]) -> list[ReviewFinding]:
    if not furniture:
        return []
    findings: list[ReviewFinding] = []
    for section in _walk_sections(doc.sections):
        for block in section.blocks:
            if block.kind in _FURNITURE_KINDS or block.kind == BlockKind.TITLE:
                continue
            text = _norm_line(block.text)
            if text and text in furniture:
                findings.append(ReviewFinding(
                    CHECK_FURNITURE_IN_BODY,
                    doc.id,
                    f"block {block.id} text equals furniture {text}",
                ))
                continue
            # A longer block can carry the furniture as one of its own lines.
            # Match whole lines only: a substring test would fire on a bare
            # page number inside any sentence.
            line = next(
                (norm for raw in block.text.splitlines() if (norm := _norm_line(raw)) in furniture),
                None,
            )
            if line is not None:
                findings.append(ReviewFinding(
                    CHECK_FURNITURE_IN_BODY,
                    doc.id,
                    f"block {block.id} has a line equal to furniture {line}",
                ))
    return findings


def _title_findings(doc: LogicalDocument, furniture: set[str]) -> list[ReviewFinding]:
    if not furniture:
        return []
    findings: list[ReviewFinding] = []
    title = _norm_line(doc.title or "")
    if title and title in furniture:
        findings.append(ReviewFinding(
            CHECK_TITLE_EQUALS_FURNITURE,
            doc.id,
            f"title equals furniture {title}",
        ))
    for section in _walk_sections(doc.sections):
        for block in section.blocks:
            if block.kind != BlockKind.TITLE:
                continue
            text = _norm_line(block.text)
            if text and text in furniture:
                findings.append(ReviewFinding(
                    CHECK_TITLE_EQUALS_FURNITURE,
                    doc.id,
                    f"title block {block.id} equals furniture {text}",
                ))
    return findings


def load_trace(path: Path) -> list[Event]:
    """Read a ``packet parse --trace`` JSONL file into events."""
    events: list[Event] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trace line {line_no}: {exc.msg}") from exc
        try:
            events.append(Event.model_validate(row))
        except ValidationError as exc:
            raise ValueError(f"trace line {line_no}: invalid event") from exc
    return events


def render_html(
    packet: Packet,
    events: list[Event],
    pdf: Path | None = None,
    gold: list[int] | None = None,
) -> str:
    """One HTML document: page strip, decisions, document stepper, optional gold marks."""
    images: dict[int, str] = {}
    note: str | None = None
    if pdf is not None:
        images, note = _page_pngs(Path(pdf))

    sizes, classes = _page_geometry(events)
    blocks = _by_page(events, EventKind.BLOCK_EMITTED, _block_page)
    proposed = _by_page(events, EventKind.BOUNDARY_PROPOSED, _proposal_page)
    decided = _by_page(events, EventKind.BOUNDARY_DECIDED, lambda event: event.page)
    predicted = {doc.page_start for doc in packet.documents}
    gold_set = None if gold is None else set(gold)
    page_count = _page_count(packet, events)

    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{_esc(packet.id)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{_esc(packet.id)}</h1>",
    ]
    if note:
        parts.append(f"<p class=\"note\">{_esc(note)}</p>")
    if gold_set is not None:
        pred = ", ".join(str(n) for n in sorted(predicted)) or "none"
        gold_txt = ", ".join(str(n) for n in sorted(gold_set)) or "none"
        parts.append(
            "<p class=\"gold-summary\">"
            f"predicted starts {pred}; gold starts {gold_txt}</p>"
        )
    parts.append(_checks_html(packet))
    parts.append("<div class=\"strip\">")
    for index in range(page_count):
        mark = _boundary_mark(index, predicted, gold_set)
        parts.append(_page_html(
            index,
            sizes.get(index),
            classes.get(index),
            blocks.get(index, []),
            proposed.get(index, []),
            decided.get(index, []),
            images.get(index),
            mark,
        ))
    if gold_set is not None:
        # A gold start past the last page has no page card; still mark it FN.
        for index in sorted(page for page in gold_set if page >= page_count):
            parts.append(
                f"<section class=\"page beyond\" data-page=\"{index}\" data-mark=\"FN\">"
                f"<h2>Page {index}</h2><p class=\"mark mark-FN\">FN</p>"
                "<p class=\"note\">gold start past the last page</p></section>"
            )
    parts.append("</div>")
    parts.append("<section class=\"stepper\">")
    parts.append("<h2>Documents</h2>")
    step = 0
    for event in events:
        if event.kind != EventKind.DOCUMENT_CLOSED:
            continue
        parts.append(_step_html(step, event, packet))
        step += 1
    parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _checks_html(packet: Packet) -> str:
    findings = review_checks(packet)
    parts = ["<section class=\"checks\">", "<h2>Review checks</h2>"]
    if not findings:
        parts.append("<p class=\"checks-clear\">no review findings</p>")
    else:
        parts.append("<ul class=\"checks-list\">")
        for finding in findings:
            parts.append(
                f"<li class=\"check-hit\" data-check=\"{_esc(finding.check)}\" "
                f"data-doc=\"{_esc(finding.document_id)}\">{_esc(finding.detail)}</li>"
            )
        parts.append("</ul>")
    parts.append("</section>")
    return "".join(parts)


def _page_pngs(pdf: Path) -> tuple[dict[int, str], str | None]:
    try:
        from packet.source import PdfPageSource

        images: dict[int, str] = {}
        with PdfPageSource(pdf) as src:
            for index in range(src.page_count):
                img = src.render(index, longest_px=_LONGEST_PX)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images[index] = base64.b64encode(buf.getvalue()).decode("ascii")
        return images, None
    except Exception:  # noqa: BLE001 - images are optional; any render failure falls back to text
        return {}, IMAGE_SKIP_NOTE


def _page_count(packet: Packet, events: list[Event]) -> int:
    count = packet.origin.page_count
    seen = [event.page for event in events if isinstance(event.page, int)]
    if seen:
        count = max(count, max(seen) + 1)
    return max(count, 0)


def _page_geometry(events: list[Event]) -> tuple[dict[int, tuple[float, float]], dict[int, str]]:
    sizes: dict[int, tuple[float, float]] = {}
    classes: dict[int, str] = {}
    for event in events:
        if event.kind != EventKind.PAGE_CLASSIFIED or not isinstance(event.page, int):
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        width, height = _float(payload.get("width")), _float(payload.get("height"))
        if width is not None and height is not None and width > 0 and height > 0:
            sizes[event.page] = (width, height)
        page_class = payload.get("page_class")
        if page_class is not None:
            classes[event.page] = _plain(page_class)
    return sizes, classes


def _by_page(events: list[Event], kind: EventKind, page_of) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for event in events:
        if event.kind != kind:
            continue
        page = page_of(event)
        if not isinstance(page, int):
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        grouped.setdefault(page, []).append(payload)
    return grouped


def _block_page(event: Event) -> int | None:
    payload = event.payload if isinstance(event.payload, dict) else {}
    page = payload.get("page", event.page)
    return page if isinstance(page, int) else None


def _proposal_page(event: Event) -> int | None:
    """A proposal is emitted while reading the page after ``after_page``."""
    payload = event.payload if isinstance(event.payload, dict) else {}
    after = payload.get("after_page", event.page)
    if isinstance(after, bool) or not isinstance(after, int):
        return event.page if isinstance(event.page, int) else None
    if "after_page" in payload:
        return after + 1
    return after


def _boundary_mark(page: int, predicted: set[int], gold: set[int] | None) -> str | None:
    if gold is None:
        return None
    is_pred = page in predicted
    is_gold = page in gold
    if is_pred and is_gold:
        return "TP"
    if is_pred:
        return "FP"
    if is_gold:
        return "FN"
    return None


def _page_html(
    index: int,
    size: tuple[float, float] | None,
    page_class: str | None,
    blocks: list[dict],
    proposed: list[dict],
    decided: list[dict],
    image: str | None,
    mark: str | None,
) -> str:
    attrs = f"class=\"page\" data-page=\"{index}\""
    if mark:
        attrs += f" data-mark=\"{mark}\""
    parts = [f"<section {attrs}>", f"<h2>Page {index}</h2>"]
    if mark:
        parts.append(f"<p class=\"mark mark-{mark}\">{mark}</p>")
    if page_class:
        parts.append(f"<p class=\"pageclass\">page class {_esc(page_class)}</p>")
    parts.append(_frame_html(size, blocks, image))
    parts.append("<div class=\"decision\">")
    parts.append(_proposed_html(proposed))
    parts.append(_decided_html(decided))
    parts.append("</div></section>")
    return "".join(parts)


def _frame_html(
    size: tuple[float, float] | None,
    blocks: list[dict],
    image: str | None,
) -> str:
    ordered = sorted(enumerate(blocks), key=lambda item: (_reading_order(item[1]), item[0]))
    boxes: list[str] = []
    unboxed: list[str] = []
    for _, payload in ordered:
        token = _token(payload.get("kind"))
        classes = f"block kind-{token}"
        if token in _FURNITURE:
            classes += " furniture"
        text = _esc(payload.get("text", ""))
        bbox = payload.get("bbox") if isinstance(payload.get("bbox"), dict) else None
        # y grows down from the top edge (packet.layout.read_lines).
        style = _box_style(bbox, size) if bbox and size else None
        if style:
            boxes.append(f"<div class=\"{classes} box\" style=\"{style}\">{text}</div>")
        else:
            unboxed.append(f"<li class=\"{classes}\">{text}</li>")

    frame_class = "frame has-size" if size else "frame"
    style_attr = ""
    if size:
        width, height = size
        style_attr = f" style=\"aspect-ratio:{width:.4f}/{height:.4f}\""
    parts = [f"<div class=\"{frame_class}\"{style_attr}>"]
    if image:
        img_class = "" if size else " class=\"flow\""
        parts.append(f"<img{img_class} alt=\"\" src=\"data:image/png;base64,{image}\">")
    parts.extend(boxes)
    parts.append("</div>")
    if unboxed:
        parts.append("<ul class=\"unboxed\">")
        parts.extend(unboxed)
        parts.append("</ul>")
    return "".join(parts)


def _box_style(bbox: dict, size: tuple[float, float]) -> str | None:
    width, height = size
    x0, y0, x1, y1 = (_float(bbox.get(key)) for key in ("x0", "y0", "x1", "y1"))
    if None in (x0, y0, x1, y1):
        return None
    assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    left = 100.0 * x0 / width
    top = 100.0 * y0 / height
    box_w = 100.0 * (x1 - x0) / width
    box_h = 100.0 * (y1 - y0) / height
    return f"left:{left:.2f}%;top:{top:.2f}%;width:{box_w:.2f}%;height:{box_h:.2f}%"


def _proposed_html(rows: list[dict]) -> str:
    if not rows:
        return "<p class=\"proposed\">boundary proposed none</p>"
    parts = []
    for row in rows:
        after = row.get("after_page")
        after_txt = _esc(after) if after is not None else "none"
        parts.append(
            "<p class=\"proposed\">boundary proposed "
            f"after page {after_txt} "
            f"confidence {_esc(_num(row.get('confidence')))} "
            f"reason {_esc(row.get('reason'))}</p>"
        )
    return "".join(parts)


def _decided_html(rows: list[dict]) -> str:
    if not rows:
        return "<p class=\"decided\">boundary decided none</p>"
    parts = []
    for row in rows:
        split = row.get("split")
        if split is True:
            split_txt = "true"
        elif split is False:
            split_txt = "false"
        else:
            split_txt = _esc(split)
        parts.append(
            "<p class=\"decided\">boundary decided "
            f"split {split_txt} "
            f"reason {_esc(row.get('reason'))} "
            f"confidence {_esc(_num(row.get('confidence')))} "
            f"page_num {_esc(_num(row.get('page_num')))} "
            f"page_total {_esc(_num(row.get('page_total')))}</p>"
        )
    return "".join(parts)


def _step_html(index: int, event: Event, packet: Packet) -> str:
    payload = event.payload if isinstance(event.payload, dict) else {}
    doc_id = _plain(payload.get("document_id"))
    doc = next((item for item in packet.documents if item.id == doc_id), None)
    title = payload.get("title")
    if not title and doc is not None:
        title = doc.title
    pages = payload.get("pages")
    if isinstance(pages, list) and len(pages) >= 2:
        page_label = f"pages {pages[0]}-{pages[1]}"
    elif doc is not None:
        page_label = f"pages {doc.page_start}-{doc.page_end}"
    else:
        page_label = "pages none"
    if doc is None:
        tree = "<p>no section tree</p>"
    elif doc.sections:
        tree = "<ul class=\"sections\">" + "".join(_section_html(s) for s in doc.sections) + "</ul>"
    else:
        tree = "<p>no sections</p>"
    return (
        f"<article class=\"step\" data-step=\"{index}\">"
        f"<h3>{_esc(doc_id)}</h3>"
        f"<p class=\"doctitle\">{_esc(title or '(untitled)')}</p>"
        f"<p class=\"pagerange\">{_esc(page_label)}</p>"
        f"{tree}"
        f"{_flags_html(doc)}"
        "</article>"
    )


def _section_html(section: Section) -> str:
    heading = section.heading or "(untitled section)"
    parts = [
        "<li class=\"section\">",
        (
            f"<p>{_esc(heading)} "
            f"<span class=\"pages\">{section.page_start}-{section.page_end}</span></p>"
        ),
    ]
    if section.blocks:
        parts.append("<ul class=\"sec-blocks\">")
        for block in section.blocks:
            token = _token(block.kind)
            furn = " furniture" if token in _FURNITURE else ""
            parts.append(
                f"<li class=\"block kind-{token}{furn}\">"
                f"<span class=\"kind\">{_esc(token)}</span> {_esc(block.text)}</li>"
            )
        parts.append("</ul>")
    if section.children:
        parts.append("<ul>")
        for child in section.children:
            parts.append(_section_html(child))
        parts.append("</ul>")
    parts.append("</li>")
    return "".join(parts)


def _flags_html(doc: LogicalDocument | None) -> str:
    if doc is None or "packet.flags" not in doc.extras:
        return "<p class=\"flags\">no flags</p>"
    flags = doc.extras.get("packet.flags")
    if not isinstance(flags, list) or not flags:
        return "<p class=\"flags\">no flags</p>"
    items = []
    for flag in flags:
        kind, page, ref, detail = _flag_fields(flag)
        items.append(
            "<li>"
            f"kind {_esc(_plain(kind))} "
            f"page {_esc(_plain(page))} "
            f"ref {_esc(_plain(ref))} "
            f"detail {_esc(_detail_text(detail))}"
            "</li>"
        )
    return "<ul class=\"flags\">" + "".join(items) + "</ul>"


def _flag_fields(flag: object) -> tuple[object, object, object, object]:
    if isinstance(flag, dict):
        return flag.get("kind"), flag.get("page"), flag.get("ref"), flag.get("detail")
    return (
        getattr(flag, "kind", ""),
        getattr(flag, "page", None),
        getattr(flag, "ref", None),
        getattr(flag, "detail", None),
    )


def _detail_text(detail: object) -> str:
    if hasattr(detail, "model_dump") and not isinstance(detail, dict):
        detail = detail.model_dump()
    if isinstance(detail, (dict, list)):
        return json.dumps(detail, default=str, ensure_ascii=True)
    if detail is None:
        return ""
    return str(detail)


def _reading_order(payload: dict) -> int:
    value = payload.get("reading_order", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _num(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return _plain(value)


def _plain(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return ""
    return str(value)


def _token(value: object) -> str:
    text = _plain(value).strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text)
    return cleaned.strip("-") or "unknown"


def _esc(value: object) -> str:
    return html.escape(_plain(value), quote=True)
