"""GMA-458: self-contained HTML visualizer for a parse trace."""

from __future__ import annotations

import re

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
