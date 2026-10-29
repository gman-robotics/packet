from packet.assemble import Assembler, _nest, repair_ranks
from packet.types import PACKET_FLAGS_KEY, Block, BlockKind, SourceKind


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
    assert PACKET_FLAGS_KEY not in doc.extras


def test_repair_ranks_nests_decimal_under_prefix():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1, page=0),
        _b(1, BlockKind.PARAGRAPH, "intro", page=0),
        _b(2, BlockKind.HEADING, "1.1", 1, page=1),
        _b(3, BlockKind.PARAGRAPH, "scope", page=1),
        _b(4, BlockKind.HEADING, "2", 1, page=2),
    ]
    doc = Assembler().build_document("d0", blocks)
    assert [section.heading for section in doc.sections] == ["1", "2"]
    one, two = doc.sections
    assert one.level == 1 and two.level == 1 and two.page_end == 2
    assert [child.heading for child in one.children] == ["1.1"]
    child = one.children[0]
    assert child.level == 2
    assert [block.text for block in child.blocks] == ["1.1", "scope"]
    assert one.page_end == 1
    assert doc.extras[PACKET_FLAGS_KEY] == [{
        "kind": "heading_rank_repaired",
        "page": 1,
        "ref": "b2-sec",
        "detail": {"from_parent": None, "to_parent": "b0-sec", "numbering": "1.1"},
    }]


def test_repair_ranks_nests_three_decimal_levels():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1),
        _b(1, BlockKind.HEADING, "1.1", 1),
        _b(2, BlockKind.HEADING, "1.1.2", 1),
        _b(3, BlockKind.HEADING, "2", 1),
    ]
    doc = Assembler().build_document("d0", blocks)
    assert [section.heading for section in doc.sections] == ["1", "2"]
    mid = doc.sections[0].children[0]
    assert mid.heading == "1.1" and mid.level == 2
    assert [child.heading for child in mid.children] == ["1.1.2"]
    assert mid.children[0].level == 3
    numbering = [flag["detail"]["numbering"] for flag in doc.extras[PACKET_FLAGS_KEY]]
    assert numbering == ["1.1", "1.1.2"]


def test_repair_moves_decimal_off_a_wrong_parent():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1, page=0),
        _b(1, BlockKind.HEADING, "2", 1, page=1),
        _b(2, BlockKind.HEADING, "1.1", 2, page=4),
    ]
    doc = Assembler().build_document("d0", blocks)
    one, two = doc.sections
    assert [child.heading for child in one.children] == ["1.1"]
    assert two.children == []
    assert one.page_end == 4
    assert two.page_end == 1
    detail = doc.extras[PACKET_FLAGS_KEY][0]["detail"]
    assert detail == {"from_parent": "b1-sec", "to_parent": "b0-sec", "numbering": "1.1"}


def test_repair_inserts_reparented_section_in_document_order():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1),
        _b(1, BlockKind.HEADING, "1.1", 2),
        _b(2, BlockKind.HEADING, "1.2", 3),
        _b(3, BlockKind.HEADING, "1.3", 2),
    ]
    doc = Assembler().build_document("d0", blocks)
    one = doc.sections[0]
    assert [child.heading for child in one.children] == ["1.1", "1.2", "1.3"]
    assert [child.level for child in one.children] == [2, 2, 2]
    assert one.children[0].children == []
    assert doc.extras[PACKET_FLAGS_KEY] == [{
        "kind": "heading_rank_repaired",
        "page": 0,
        "ref": "b2-sec",
        "detail": {"from_parent": "b1-sec", "to_parent": "b0-sec", "numbering": "1.2"},
    }]


def test_repair_recomputes_page_end_for_old_and_new_parents():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1, page=0),
        _b(1, BlockKind.HEADING, "1.1", 2, page=1),
        _b(2, BlockKind.HEADING, "1.2", 3, page=2),
        _b(3, BlockKind.PARAGRAPH, "detail", page=3),
        _b(4, BlockKind.HEADING, "1.3", 2, page=4),
        _b(5, BlockKind.HEADING, "2", 1, page=5),
        _b(6, BlockKind.HEADING, "8", 1, page=6),
        _b(7, BlockKind.HEADING, "2.1", 2, page=7),
        _b(8, BlockKind.HEADING, "2.1.1", 3, page=9),
    ]
    doc = Assembler().build_document("d0", blocks)
    one, two, eight = doc.sections
    assert [child.heading for child in one.children] == ["1.1", "1.2", "1.3"]
    assert one.children[0].page_end == 1
    assert one.children[1].page_end == 3
    assert one.page_end == 4
    assert [child.heading for child in two.children] == ["2.1"]
    assert [child.heading for child in two.children[0].children] == ["2.1.1"]
    assert two.children[0].page_end == 7
    assert two.page_end == 9
    assert eight.children == []
    assert eight.page_end == 6
    numbering = [flag["detail"]["numbering"] for flag in doc.extras[PACKET_FLAGS_KEY]]
    assert numbering == ["1.2", "2.1"]


def test_unnumbered_section_is_not_reparented():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1),
        _b(1, BlockKind.HEADING, "Notes", 1),
        _b(2, BlockKind.HEADING, "1.1", 1),
    ]
    doc = Assembler().build_document("d0", blocks)
    assert [section.heading for section in doc.sections] == ["1", "Notes"]
    assert [child.heading for child in doc.sections[0].children] == ["1.1"]
    assert doc.extras[PACKET_FLAGS_KEY] == [{
        "kind": "heading_rank_repaired",
        "page": 0,
        "ref": "b2-sec",
        "detail": {"from_parent": None, "to_parent": "b0-sec", "numbering": "1.1"},
    }]


def test_unnumbered_tree_is_unchanged_and_omits_flags():
    blocks = [
        _b(0, BlockKind.TITLE, "Annual Report"),
        _b(1, BlockKind.PARAGRAPH, "Intro text"),
        _b(2, BlockKind.HEADING, "Overview", 1),
        _b(3, BlockKind.PARAGRAPH, "Overview body"),
        _b(4, BlockKind.HEADING, "Markets", 2),
    ]
    nested = _nest(blocks)
    expected = [section.model_dump() for section in nested]
    doc = Assembler().build_document("d0", blocks)
    assert PACKET_FLAGS_KEY not in doc.extras
    assert [section.model_dump() for section in doc.sections] == expected
    repaired, flags = repair_ranks(doc.sections)
    assert flags == []
    assert repaired is doc.sections
    assert [section.model_dump() for section in repaired] == expected


def test_article_iv_is_depth_one_and_not_a_decimal_prefix():
    blocks = [
        _b(0, BlockKind.HEADING, "Terms", 1),
        _b(1, BlockKind.HEADING, "Article IV Powers", 3),
        _b(2, BlockKind.HEADING, "ARTICLE 4 Duties", 4),
        _b(3, BlockKind.HEADING, "4.1 Detail", 1),
    ]
    doc = Assembler().build_document("d0", blocks)
    terms, detail = doc.sections
    assert terms.heading == "Terms"
    article = terms.children[0]
    assert article.heading == "Article IV Powers" and article.level == 1
    duties = article.children[0]
    assert duties.heading == "ARTICLE 4 Duties" and duties.level == 1
    assert detail.heading == "4.1 Detail" and detail.level == 2 and detail.children == []
    assert PACKET_FLAGS_KEY not in doc.extras


def test_year_and_bare_amount_are_not_numbering():
    blocks = [
        _b(0, BlockKind.HEADING, "Overview", 1),
        _b(1, BlockKind.HEADING, "2024 Budget", 2),
        _b(2, BlockKind.HEADING, "100", 2),
    ]
    doc = Assembler().build_document("d0", blocks)
    overview = doc.sections[0]
    assert [(child.heading, child.level) for child in overview.children] == [
        ("2024 Budget", 2),
        ("100", 2),
    ]
    assert PACKET_FLAGS_KEY not in doc.extras


def test_leading_zeros_do_not_match_a_different_component():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1),
        _b(1, BlockKind.HEADING, "1.01", 1),
        _b(2, BlockKind.HEADING, "1.1.1", 1),
    ]
    doc = Assembler().build_document("d0", blocks)
    one, other = doc.sections
    assert one.heading == "1"
    assert [child.heading for child in one.children] == ["1.01"]
    assert one.children[0].children == []
    assert other.heading == "1.1.1" and other.level == 3
    assert doc.extras[PACKET_FLAGS_KEY] == [{
        "kind": "heading_rank_repaired",
        "page": 0,
        "ref": "b1-sec",
        "detail": {"from_parent": None, "to_parent": "b0-sec", "numbering": "1.01"},
    }]


def test_first_component_over_99_is_not_an_outline():
    blocks = [
        _b(0, BlockKind.HEADING, "10", 1),
        _b(1, BlockKind.HEADING, "100.00", 1),
        _b(2, BlockKind.HEADING, "10.31 Meeting", 1),
    ]
    doc = Assembler().build_document("d0", blocks)
    ten, amount = doc.sections
    assert ten.heading == "10"
    assert [child.heading for child in ten.children] == ["10.31 Meeting"]
    assert ten.children[0].level == 2
    assert amount.heading == "100.00" and amount.level == 1 and amount.children == []
    assert doc.extras[PACKET_FLAGS_KEY] == [{
        "kind": "heading_rank_repaired",
        "page": 0,
        "ref": "b2-sec",
        "detail": {"from_parent": None, "to_parent": "b0-sec", "numbering": "10.31"},
    }]


def test_section_keyword_nests_as_decimal():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1, page=0),
        _b(1, BlockKind.HEADING, "Section 1.2", 1, page=3),
    ]
    doc = Assembler().build_document("d0", blocks)
    assert [section.heading for section in doc.sections] == ["1"]
    child = doc.sections[0].children[0]
    assert child.heading == "Section 1.2" and child.level == 2
    assert doc.sections[0].page_end == 3
    assert doc.extras[PACKET_FLAGS_KEY][0]["detail"]["numbering"] == "1.2"


def test_title_level_zero_is_not_rewritten():
    blocks = [
        _b(0, BlockKind.TITLE, "1. Charter"),
        _b(1, BlockKind.HEADING, "1.1 Scope", 1),
    ]
    doc = Assembler().build_document("d0", blocks)
    title = doc.sections[0]
    assert title.level == 0
    assert [child.heading for child in title.children] == ["1.1 Scope"]
    assert title.children[0].level == 2
    assert PACKET_FLAGS_KEY not in doc.extras


def test_repair_keeps_cover_when_later_move_recomputes_parent():
    blocks = [
        _b(0, BlockKind.HEADING, "1", 1, page=0),
        _b(1, BlockKind.HEADING, "9", 1, page=1),
        _b(2, BlockKind.HEADING, "1.1", 1, page=2),
        _b(3, BlockKind.HEADING, "9.1", 2, page=3),
        _b(4, BlockKind.HEADING, "1.1.1", 2, page=4),
        _b(5, BlockKind.HEADING, "1.1.1.1", 3, page=15),
    ]
    doc = Assembler().build_document("d0", blocks)
    one = doc.sections[0]
    assert [child.heading for child in one.children] == ["1.1"]
    assert one.children[0].children[0].children[0].heading == "1.1.1.1"
    assert one.page_end == 15
