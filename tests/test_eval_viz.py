"""T4: packet eval --viz-dir, metric-blind review checks, stripped-numbers."""
from __future__ import annotations

import pytest

from packet import cli
from packet.eval.fixtures import build_builtin, build_fail_modes
from packet.eval.run import run_named_fixtures
from packet.events import EventKind
from packet.pipeline import Pipeline, PipelineConfig
from packet.types import Block, BlockKind, LogicalDocument, Packet, PacketOrigin, Section
from packet.viz import (
    CHECK_FURNITURE_IN_BODY,
    CHECK_SECTION_OUTSIDE,
    CHECK_TITLE_EQUALS_FURNITURE,
    render_html,
    review_checks,
)


def _packet(docs: list[LogicalDocument]) -> Packet:
    pages = max((doc.page_end for doc in docs), default=0) + 1
    return Packet(id="p", origin=PacketOrigin(path="x.pdf", page_count=pages), documents=docs)


def _block(kind: BlockKind, text: str, block_id: str = "b") -> Block:
    return Block(id=block_id, page=0, kind=kind, text=text)


def _doc(
    sections: list[Section],
    furniture: list[Block] | None = None,
    title: str | None = "Report",
    page_start: int = 0,
    page_end: int = 1,
    doc_id: str = "d0",
) -> LogicalDocument:
    return LogicalDocument(
        id=doc_id,
        title=title,
        page_start=page_start,
        page_end=page_end,
        sections=sections,
        furniture=furniture or [],
    )


def _findings(packet: Packet, check: str) -> list:
    return [item for item in review_checks(packet) if item.check == check]


def _assert_offline(html: str) -> None:
    assert "http://" not in html
    assert "https://" not in html
    assert 'src="//' not in html
    assert "src='//" not in html


def test_section_outside_document_positive():
    doc = _doc([
        Section(id="parent", page_start=0, page_end=1, children=[
            Section(id="child", page_start=0, page_end=4),
        ]),
    ])
    hits = _findings(_packet([doc]), CHECK_SECTION_OUTSIDE)
    assert len(hits) == 1
    assert hits[0].document_id == "d0"
    assert "child" in hits[0].detail
    html = render_html(_packet([doc]), [])
    assert f'data-check="{CHECK_SECTION_OUTSIDE}"' in html
    _assert_offline(html)


def test_section_outside_document_negative():
    doc = _doc([
        Section(id="parent", page_start=0, page_end=1, children=[
            Section(id="child", page_start=0, page_end=1),
        ]),
    ])
    packet = _packet([doc])
    assert _findings(packet, CHECK_SECTION_OUTSIDE) == []
    assert f'data-check="{CHECK_SECTION_OUTSIDE}"' not in render_html(packet, [])


FURNITURE_TEXT = {
    BlockKind.HEADER: "Binder Header",
    BlockKind.FOOTER: "Confidential - Binder 7",
    BlockKind.PAGE_NUMBER: "Page 3 of 9",
}


def test_section_outside_document_starts_before_document():
    doc = _doc(
        [Section(id="early", page_start=1, page_end=3)],
        page_start=2,
        page_end=3,
    )
    hits = _findings(_packet([doc]), CHECK_SECTION_OUTSIDE)
    assert len(hits) == 1 and "early" in hits[0].detail


@pytest.mark.parametrize("kind", sorted(FURNITURE_TEXT, key=lambda k: k.value))
def test_furniture_in_body_each_kind(kind):
    text = FURNITURE_TEXT[kind]
    furniture = _block(kind, text, "f")
    leaked = _block(BlockKind.PARAGRAPH, text, "p")
    doc = _doc([Section(id="s", page_start=0, page_end=1, blocks=[leaked])], furniture=[furniture])
    hits = _findings(_packet([doc]), CHECK_FURNITURE_IN_BODY)
    assert len(hits) == 1 and hits[0].document_id == "d0"


@pytest.mark.parametrize("kind", sorted(FURNITURE_TEXT, key=lambda k: k.value))
def test_furniture_line_inside_longer_block(kind):
    text = FURNITURE_TEXT[kind]
    furniture = _block(kind, text, "f")
    leaked = _block(
        BlockKind.PARAGRAPH,
        f"The chapter body runs on here.\n  {text}  \nand continues after it.",
        "p",
    )
    doc = _doc([Section(id="s", page_start=0, page_end=1, blocks=[leaked])], furniture=[furniture])
    packet = _packet([doc])
    hits = _findings(packet, CHECK_FURNITURE_IN_BODY)
    assert len(hits) == 1
    assert hits[0].document_id == "d0"
    assert text in hits[0].detail
    assert f'data-check="{CHECK_FURNITURE_IN_BODY}"' in render_html(packet, [])


def test_furniture_text_inside_a_line_does_not_fire():
    number = _block(BlockKind.PAGE_NUMBER, "3", "n")
    body = _block(BlockKind.PARAGRAPH, "See table 3 on the next page.\nIt has 3 rows.", "p")
    doc = _doc([Section(id="s", page_start=0, page_end=1, blocks=[body])], furniture=[number])
    assert _findings(_packet([doc]), CHECK_FURNITURE_IN_BODY) == []


@pytest.mark.parametrize("kind", sorted(FURNITURE_TEXT, key=lambda k: k.value))
def test_title_equals_furniture_each_kind(kind):
    text = FURNITURE_TEXT[kind]
    doc = _doc(
        [Section(id="s", page_start=0, page_end=1, blocks=[
            _block(BlockKind.TITLE, text, "t"),
            _block(BlockKind.PARAGRAPH, "Chapter body stays in the document.", "p"),
        ])],
        furniture=[_block(kind, text, "f")],
        title=text,
    )
    hits = _findings(_packet([doc]), CHECK_TITLE_EQUALS_FURNITURE)
    assert len(hits) == 2
    assert {hit.document_id for hit in hits} == {"d0"}


def test_furniture_in_body_positive():
    header = _block(BlockKind.HEADER, "Binder Header", "h")
    leaked = _block(BlockKind.PARAGRAPH, "Binder Header", "p")
    doc = _doc([Section(id="s", page_start=0, page_end=1, blocks=[leaked])], furniture=[header])
    hits = _findings(_packet([doc]), CHECK_FURNITURE_IN_BODY)
    assert len(hits) == 1 and hits[0].document_id == "d0"
    html = render_html(_packet([doc]), [])
    assert f'data-check="{CHECK_FURNITURE_IN_BODY}"' in html
    assert "Binder Header" in html
    _assert_offline(html)


def test_furniture_in_body_negative():
    header = _block(BlockKind.HEADER, "Binder Header", "h")
    body = _block(BlockKind.PARAGRAPH, "Chapter body stays in the document.", "p")
    doc = _doc([Section(id="s", page_start=0, page_end=1, blocks=[body])], furniture=[header])
    packet = _packet([doc])
    assert _findings(packet, CHECK_FURNITURE_IN_BODY) == []
    assert f'data-check="{CHECK_FURNITURE_IN_BODY}"' not in render_html(packet, [])


def test_title_equals_furniture_positive():
    header = _block(BlockKind.HEADER, "Binder Header", "h")
    doc = _doc(
        [Section(id="s", page_start=0, page_end=1, blocks=[
            _block(BlockKind.PARAGRAPH, "Chapter body stays in the document.", "p"),
        ])],
        furniture=[header],
        title="Binder Header",
    )
    hits = _findings(_packet([doc]), CHECK_TITLE_EQUALS_FURNITURE)
    assert len(hits) == 1 and hits[0].document_id == "d0"
    html = render_html(_packet([doc]), [])
    assert f'data-check="{CHECK_TITLE_EQUALS_FURNITURE}"' in html
    _assert_offline(html)


def test_title_equals_furniture_negative():
    header = _block(BlockKind.HEADER, "Binder Header", "h")
    doc = _doc(
        [Section(id="s", page_start=0, page_end=1, blocks=[
            _block(BlockKind.TITLE, "Chapter A", "t"),
            _block(BlockKind.PARAGRAPH, "Chapter body stays in the document.", "p"),
        ])],
        furniture=[header],
        title="Chapter A",
    )
    packet = _packet([doc])
    assert _findings(packet, CHECK_TITLE_EQUALS_FURNITURE) == []
    assert f'data-check="{CHECK_TITLE_EQUALS_FURNITURE}"' not in render_html(packet, [])


def test_viz_dir_help_names_packet_eval_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["eval", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--viz-dir" in help_text
    assert ".packet-eval/" in help_text


def test_viz_dir_writes_one_html_per_fixture(tmp_path):
    cache = tmp_path / "cache"
    viz = tmp_path / "viz"
    assert cli.main(["eval", "--dir", str(cache), "--viz-dir", str(viz)]) == 0
    fixtures = build_builtin(cache) + build_fail_modes(cache)
    files = sorted(path.name for path in viz.glob("*.html"))
    assert files == sorted(f"{fx.name}.html" for fx in fixtures)
    for fx in fixtures:
        html = (viz / f"{fx.name}.html").read_text(encoding="utf-8")
        _assert_offline(html)
        events = []
        Pipeline().parse(fx.pdf, on_event=events.append)
        closed = [event for event in events if event.kind == EventKind.DOCUMENT_CLOSED]
        assert closed
        assert html.count("data-step=") == len(closed)
    multi = (viz / "multi-doc.html").read_text(encoding="utf-8")
    assert 'data-page="0" data-mark="TP"' in multi
    assert 'data-page="2" data-mark="TP"' in multi
    merged = (viz / "stripped-numbers.html").read_text(encoding="utf-8")
    assert 'data-page="2" data-mark="FN"' in merged


def test_viz_dir_does_not_change_eval_output(tmp_path, capsys):
    plain_dir = tmp_path / "plain"
    viz_dir = tmp_path / "with"
    report = tmp_path / "html"
    assert cli.main(["eval", "--dir", str(plain_dir)]) == 0
    plain = capsys.readouterr().out
    assert cli.main(["eval", "--dir", str(viz_dir), "--viz-dir", str(report)]) == 0
    assert capsys.readouterr().out == plain
    assert cli.main(["eval", "--json", "--dir", str(plain_dir)]) == 0
    plain_json = capsys.readouterr().out
    assert cli.main(["eval", "--json", "--dir", str(viz_dir), "--viz-dir", str(report)]) == 0
    assert capsys.readouterr().out == plain_json
    assert "stripped-numbers" not in plain_json


def test_stripped_numbers_false_merges_below_fallback(tmp_path):
    fx = next(item for item in build_fail_modes(tmp_path) if item.name == "stripped-numbers")
    assert fx.name not in {item.name for item in build_builtin(tmp_path)}
    assert len(fx.gold.documents) == 2
    gated = run_named_fixtures(tmp_path)
    assert [row.name for row in gated] == [
        "text-layer", "multi-doc", "shared-header", "unnumbered-cover", "scan",
    ]
    assert all(row.passed_gate() and row.boundary.f1 == 1.0 for row in gated)

    events = []
    packet = Pipeline().parse(fx.pdf, on_event=events.append)
    assert len(packet.documents) == 1
    second = fx.gold.documents[1].page_start
    decided = next(
        event for event in events
        if event.kind == EventKind.BOUNDARY_DECIDED and event.page == second
    )
    assert decided.payload["confidence"] < 0.55
    signal = decided.payload["signal"]
    assert signal["reason"] == "title_top"
    assert signal["confidence"] < 0.55
    assert "header_changed" not in signal["reason"]
    assert "media_box_changed" not in signal["reason"]
    assert "topic_shift" not in signal["reason"]
    assert "language_changed" not in signal["reason"]


class _SplitEverywhere:
    def probability(self, features):
        return 0.99


def test_fail_mode_viz_ignores_boundary_weights(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    assert cli.main(["eval", "--dir", str(tmp_path / "c1"), "--viz-dir", str(plain)]) == 0
    expected = (plain / "stripped-numbers.html").read_text(encoding="utf-8")
    assert 'data-page="2" data-mark="FN"' in expected

    weighted = Pipeline(config=PipelineConfig(scorer=_SplitEverywhere()))
    fx = next(item for item in build_fail_modes(tmp_path / "c0") if item.name == "stripped-numbers")
    assert len(weighted.parse(fx.pdf).documents) > 1  # this pipeline would hide the false merge

    monkeypatch.setattr(cli, "_pipeline_for_boundary_weights", lambda path: weighted)
    viz = tmp_path / "weighted"
    cli.main(["eval", "--dir", str(tmp_path / "c2"), "--viz-dir", str(viz),
              "--boundary-weights", "unused.json"])
    assert (viz / "stripped-numbers.html").read_text(encoding="utf-8") == expected
