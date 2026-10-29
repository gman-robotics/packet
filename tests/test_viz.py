"""GMA-458: self-contained HTML visualizer for a parse trace."""

from __future__ import annotations

import re
from html import escape as html_escape

import pytest

from packet import cli
from packet.eval.fixtures import build_builtin
from packet.events import EventKind
from packet.pipeline import Pipeline
from packet.viz import IMAGE_SKIP_NOTE, load_trace, render_html


def _fixture(tmp_path, name):
    return next(fx for fx in build_builtin(tmp_path) if fx.name == name)


def _collect(tmp_path, name):
    fx = _fixture(tmp_path, name)
    events = []
    packet = Pipeline().parse(fx.pdf, on_event=events.append)
    return fx, packet, events


def _assert_offline(html: str) -> None:
    assert "http://" not in html
    assert "https://" not in html
    assert 'src="//' not in html
    assert "src='//" not in html


@pytest.mark.parametrize("name", ["multi-doc", "shared-header", "unnumbered-cover"])
def test_render_offline_stepper_decisions_and_furniture(tmp_path, name):
    _fx, packet, events = _collect(tmp_path, name)
    html = render_html(packet, events)
    _assert_offline(html)
    closed = [event for event in events if event.kind == EventKind.DOCUMENT_CLOSED]
    assert html.count("data-step=") == len(closed)
    assert closed
    for event in closed:
        assert event.payload["document_id"] in html
        pages = event.payload["pages"]
        assert f"pages {pages[0]}-{pages[1]}" in html
        title = event.payload.get("title")
        if isinstance(title, str) and title:
            assert title in html
    for event in events:
        if event.kind == EventKind.BOUNDARY_DECIDED:
            assert event.payload["reason"]
            assert event.payload["reason"] in html
    assert re.search(r'class="[^"]*\bfurniture\b', html)
    assert 'style="left:' in html
    assert "no flags" in html
    assert "<script" not in html.lower()


def test_gold_marks_tp_fp_fn(tmp_path):
    _fx, packet, events = _collect(tmp_path, "multi-doc")
    assert {doc.page_start for doc in packet.documents} == {0, 2}

    match = render_html(packet, events, gold=[0, 2])
    assert 'data-page="0" data-mark="TP"' in match
    assert 'data-page="2" data-mark="TP"' in match
    assert 'data-mark="FP"' not in match
    assert 'data-mark="FN"' not in match

    mismatch = render_html(packet, events, gold=[0, 1])
    assert 'data-page="0" data-mark="TP"' in mismatch
    assert 'data-page="1" data-mark="FN"' in mismatch
    assert 'data-page="2" data-mark="FP"' in mismatch
    assert 'data-page="3" data-mark=' not in mismatch
    _assert_offline(mismatch)


def test_escapes_script_text_and_lists_flags(tmp_path):
    _fx, packet, events = _collect(tmp_path, "multi-doc")
    packet.documents[0].extras["packet.flags"] = [
        {
            "kind": "weak_boundary",
            "page": 1,
            "ref": "blk",
            "detail": {"note": "<script>alert(1)</script>"},
        }
    ]
    for event in events:
        if event.kind == EventKind.BLOCK_EMITTED:
            event.payload = dict(event.payload)
            event.payload["text"] = "<script>alert(1)</script>"
            break
    html = render_html(packet, events)
    assert "<script>" not in html
    assert "</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "weak_boundary" in html
    assert "no flags" in html
    _assert_offline(html)


def test_pdf_inlines_png_or_skip_note(tmp_path):
    fx, packet, events = _collect(tmp_path, "multi-doc")
    html = render_html(packet, events, pdf=fx.pdf)
    has_img = "data:image/png;base64," in html
    has_note = IMAGE_SKIP_NOTE in html
    assert has_img != has_note
    _assert_offline(html)

    missing = render_html(packet, events, pdf=tmp_path / "missing.pdf")
    assert IMAGE_SKIP_NOTE in missing
    assert "data:image/png;base64," not in missing
    _assert_offline(missing)


def test_cli_viz_roundtrip(tmp_path):
    fx = _fixture(tmp_path, "multi-doc")
    tree = tmp_path / "packet.json"
    trace = tmp_path / "trace.jsonl"
    report = tmp_path / "report.html"
    assert cli.main(["parse", str(fx.pdf), "-o", str(tree), "--trace", str(trace)]) == 0
    assert cli.main([
        "viz", str(trace), "-o", str(report), "--tree", str(tree),
        "--pdf", str(fx.pdf), "--gold", "0,2",
    ]) == 0
    assert report.is_file()
    html = report.read_text(encoding="utf-8")
    _assert_offline(html)
    events = load_trace(trace)
    closed = [event for event in events if event.kind == EventKind.DOCUMENT_CLOSED]
    assert html.count("data-step=") == len(closed)
    assert ("data:image/png;base64," in html) != (IMAGE_SKIP_NOTE in html)
    assert 'data-mark="TP"' in html
    assert 'data-mark="FP"' not in html


def test_cli_viz_bad_input(tmp_path):
    missing = tmp_path / "missing.jsonl"
    tree = tmp_path / "packet.json"
    tree.write_text("{}", encoding="utf-8")
    assert cli.main(["viz", str(missing), "-o", str(tmp_path / "out.html"), "--tree", str(tree)]) == 2

    fx = _fixture(tmp_path, "shared-header")
    trace = tmp_path / "trace.jsonl"
    good_tree = tmp_path / "good.json"
    assert cli.main(["parse", str(fx.pdf), "-o", str(good_tree), "--trace", str(trace)]) == 0
    bad_tree = tmp_path / "bad.json"
    bad_tree.write_text("{", encoding="utf-8")
    assert cli.main(["viz", str(trace), "-o", str(tmp_path / "out.html"), "--tree", str(bad_tree)]) == 2
    assert cli.main([
        "viz", str(trace), "-o", str(tmp_path / "out.html"), "--tree", str(good_tree), "--gold", "a,2",
    ]) == 2
    assert cli.main([
        "viz", str(trace), "-o", str(tmp_path / "out.html"), "--tree", str(good_tree),
        "--pdf", str(tmp_path / "nope.pdf"),
    ]) == 2


def _page_chunks(html: str) -> dict[int, str]:
    chunks: dict[int, str] = {}
    for match in re.finditer(r'<section class="page[^"]*" data-page="(\d+)"(.*?)</section>', html, re.DOTALL):
        chunks[int(match.group(1))] = match.group(2)
    return chunks


def _all_sections(sections):
    """Own walk of the section tree, so the test does not trust viz's walker."""
    out = []
    for section in sections:
        out.append(section)
        out.extend(_all_sections(section.children))
    return out


def _step_chunks(html: str) -> list[str]:
    return re.findall(r'<article class="step" data-step="\d+">(.*?)</article>', html, re.DOTALL)


@pytest.mark.parametrize("name", ["multi-doc", "shared-header", "unnumbered-cover"])
def test_section_tree_rendered_per_document(tmp_path, name):
    _fx, packet, events = _collect(tmp_path, name)
    html = render_html(packet, events)
    steps = _step_chunks(html)
    closed = [event for event in events if event.kind == EventKind.DOCUMENT_CLOSED]
    assert len(steps) == len(closed)
    seen_section = False
    for event, chunk in zip(closed, steps):
        doc = next(d for d in packet.documents if d.id == event.payload["document_id"])
        sections = _all_sections(doc.sections)
        assert chunk.count('<li class="section">') == len(sections)
        for section in sections:
            seen_section = True
            if section.heading:
                assert html_escape(section.heading) in chunk
            assert f'<span class="pages">{section.page_start}-{section.page_end}</span>' in chunk
            for block in section.blocks:
                if block.text.strip():
                    assert html_escape(block.text) in chunk
    assert seen_section


@pytest.mark.parametrize("name", ["multi-doc", "shared-header", "unnumbered-cover"])
def test_decided_and_proposed_on_their_own_page(tmp_path, name):
    _fx, packet, events = _collect(tmp_path, name)
    html = render_html(packet, events)
    chunks = _page_chunks(html)
    decided = [e for e in events if e.kind == EventKind.BOUNDARY_DECIDED]
    assert decided
    for page, chunk in chunks.items():
        assert "boundary proposed" in chunk
        reasons = [e.payload["reason"] for e in decided if e.page == page]
        assert chunk.count("boundary decided") == max(len(reasons), 1)
        for reason in reasons:
            assert f"reason {html_escape(reason)} " in chunk
    for event in events:
        if event.kind != EventKind.BOUNDARY_PROPOSED:
            continue
        after = event.payload["after_page"]
        chunk = chunks[after + 1]
        assert f"boundary proposed after page {after} " in chunk
        assert f"reason {html_escape(str(event.payload.get('reason')))}</p>" in chunk


def test_escapes_hostile_attribute_values(tmp_path):
    _fx, packet, events = _collect(tmp_path, "multi-doc")
    hostile = '"><img src=x onerror=alert(1)>'
    changed = 0
    for event in events:
        if event.kind == EventKind.BLOCK_EMITTED and changed < 2:
            event.payload = dict(event.payload)
            event.payload["kind"] = hostile
            if changed == 1:
                event.payload["bbox"] = {"x0": "0;background:url(x)", "y0": 0, "x1": 1, "y1": 1}
            changed += 1
    assert changed == 2
    html = render_html(packet, events)
    assert "<img src=x" not in html
    assert "onerror=alert(1)>" not in html
    assert 'class="block kind-img-src-x-onerror-alert-1' in html
    assert "url(" not in html
    _assert_offline(html)


def test_gold_start_past_last_page_is_fn(tmp_path):
    _fx, packet, events = _collect(tmp_path, "multi-doc")
    page_count = packet.origin.page_count
    html = render_html(packet, events, gold=[0, 2, page_count])
    assert f'data-page="{page_count}" data-mark="FN"' in html
    assert f'data-page="{page_count + 1}" data-mark=' not in html
    assert 'data-page="0" data-mark="TP"' in html


def test_pdf_inlines_png_with_renderer(tmp_path, monkeypatch):
    from packet import source

    class FakeImage:
        def save(self, buf, format):
            assert format == "PNG"
            buf.write(b"\x89PNGfake")

    class FakeSource:
        page_count = 2

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def render(self, index, longest_px):
            sizes.append(longest_px)
            return FakeImage()

    sizes: list[int] = []
    monkeypatch.setattr(source, "PdfPageSource", FakeSource)
    fx, packet, events = _collect(tmp_path, "multi-doc")
    html = render_html(packet, events, pdf=fx.pdf)
    assert html.count('src="data:image/png;base64,iVBOR2Zha2U="') == 2
    assert sizes == [800, 800]
    assert IMAGE_SKIP_NOTE not in html
    _assert_offline(html)


def test_cli_viz_more_bad_input(tmp_path):
    fx = _fixture(tmp_path, "shared-header")
    trace = tmp_path / "trace.jsonl"
    tree = tmp_path / "good.json"
    out = tmp_path / "out.html"
    assert cli.main(["parse", str(fx.pdf), "-o", str(tree), "--trace", str(trace)]) == 0
    assert cli.main(["viz", str(trace), "-o", str(out), "--tree", str(tmp_path / "none.json")]) == 2
    assert cli.main(["viz", str(trace), "-o", str(out), "--tree", str(tree), "--gold", "0,-1"]) == 2
    bad_trace = tmp_path / "bad.jsonl"
    bad_trace.write_text(trace.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    assert cli.main(["viz", str(bad_trace), "-o", str(out), "--tree", str(tree)]) == 2
    assert not out.exists()
