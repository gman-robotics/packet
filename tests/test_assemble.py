from packet.assemble import Assembler
from packet.types import Block, BlockKind, SourceKind


def _b(i: int, kind: BlockKind, text: str, level: int | None = None, page: int = 0) -> Block:
    return Block(
        id=f"b{i}",
        page=page,
        kind=kind,
        text=text,
        reading_order=i,
        source=SourceKind.NATIVE,
        heading_level=level,
    )


def test_heading_nesting():
    blocks = [
        _b(0, BlockKind.TITLE, "Annual Report"),
        _b(1, BlockKind.PARAGRAPH, "Intro text"),
        _b(2, BlockKind.HEADING, "1. Overview", 1),
        _b(3, BlockKind.PARAGRAPH, "Overview body"),
        _b(4, BlockKind.HEADING, "1.1 Markets", 2),
        _b(5, BlockKind.PARAGRAPH, "Markets body"),
        _b(6, BlockKind.HEADING, "2. Outlook", 1),
        _b(7, BlockKind.PARAGRAPH, "Outlook body"),
    ]
    doc = Assembler().build_document("d0", blocks)
    assert doc.title == "Annual Report"
    assert len(doc.sections) == 1
    title_sec = doc.sections[0]
    assert title_sec.heading == "Annual Report"
    assert any(b.text == "Intro text" for b in title_sec.blocks)
    headings = [c.heading for c in title_sec.children]
    assert headings == ["1. Overview", "2. Outlook"]
    overview = title_sec.children[0]
    assert [c.heading for c in overview.children] == ["1.1 Markets"]
