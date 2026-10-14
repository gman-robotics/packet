"""GMA-55 slice 3: tag page furniture (HEADER / FOOTER / PAGE_NUMBER) on native text and
make the Assembler's furniture-strip path live."""
from __future__ import annotations

from pathlib import Path

from packet.eval.pdfutil import write_text_pdf
from packet.furniture import FurnitureTagger
from packet.pipeline import Pipeline
from packet.types import Block, PageClass, PageRef, SourceKind
from packet.types import BlockKind as K


def _ref(i):
    return PageRef(packet_id="p", index=i, page_class=PageClass.TEXT)


def _native(i, text):
    return [Block(id=f"p-p{i}-b0", page=i, kind=K.PARAGRAPH, text=text, source=SourceKind.NATIVE)]


def _all_blocks(doc):
    out = []

    def walk(sections):
        for s in sections:
            out.extend(s.blocks)
            walk(s.children)

    walk(doc.sections)
    return out


def test_page_number_line_is_split_and_tagged():
    blocks = FurnitureTagger().tag_page(_ref(0), _native(0, "Body text here.\nPage 1 of 3"))
    assert [(b.kind, b.text) for b in blocks] == [(K.PARAGRAPH, "Body text here."),
                                                  (K.PAGE_NUMBER, "Page 1 of 3")]


def test_bare_and_dashed_page_numbers():
    t = FurnitureTagger()
    assert t.tag_page(_ref(0), _native(0, "Body.\n- 4 -"))[-1].kind == K.PAGE_NUMBER
    assert t.tag_page(_ref(4), _native(4, "Body.\n5"))[-1].kind == K.PAGE_NUMBER


def test_repeated_top_line_becomes_header_including_first_page_on_finalize():
    t = FurnitureTagger()
    p0 = t.tag_page(_ref(0), _native(0, "Binder Header\nChapter A body.\nPage 1 of 2"))
    p1 = t.tag_page(_ref(1), _native(1, "Binder Header\nChapter A continued.\nPage 2 of 2"))
    assert p1[0].kind == K.HEADER and p1[0].text == "Binder Header"
    done = t.finalize(p0 + p1)
    assert [b.kind for b in done if b.text == "Binder Header"] == [K.HEADER, K.HEADER]


def test_repeated_bottom_line_with_changing_digits_becomes_footer():
    t = FurnitureTagger()
    a = t.tag_page(_ref(0), _native(0, "Body one.\nACME Corp confidential 2026-01"))
    b = t.tag_page(_ref(1), _native(1, "Body two.\nACME Corp confidential 2026-02"))
    assert t.finalize(a + b)[-1].kind == K.FOOTER
    assert [x.kind for x in t.finalize(a + b) if "confidential" in x.text] == [K.FOOTER, K.FOOTER]


def test_unique_first_line_is_not_furniture():
    t = FurnitureTagger()
    blocks = t.tag_page(_ref(0), _native(0, "ACME Annual Report\n1. Overview\nBody of page one."))
    blocks += t.tag_page(_ref(1), _native(1, "Section continues.\nMore body."))
    assert not [b for b in t.finalize(blocks) if b.kind in {K.HEADER, K.FOOTER}]


def test_body_text_is_preserved_and_ids_unique():
    t = FurnitureTagger()
    blocks = t.tag_page(_ref(0), _native(0, "Hdr\nLine a\nLine b\nPage 1 of 1"))
    assert [b.text for b in blocks] == ["Hdr\nLine a\nLine b", "Page 1 of 1"]
    assert len({b.id for b in blocks}) == len(blocks)


def test_ocr_blocks_are_left_alone():
    blk = Block(id="o", page=0, kind=K.HEADER, text="X\nPage 1 of 2", source=SourceKind.VLM)
    assert FurnitureTagger().tag_page(_ref(0), [blk]) == [blk]


def test_pipeline_strips_furniture_from_body(tmp_path: Path):
    pdf = write_text_pdf(tmp_path / "sh.pdf", [
        ["Binder Header", "Chapter A body.", "Page 1 of 2"],
        ["Binder Header", "Chapter A continued.", "Page 2 of 2"],
    ])
    doc = Pipeline().parse(pdf).documents[0]
    body = " ".join(b.text for b in _all_blocks(doc))
    assert "Binder Header" not in body and "Page 1 of 2" not in body
    assert "Chapter A body." in body
    kinds = sorted(b.kind.value for b in doc.furniture)
    assert kinds == ["header", "header", "page_number", "page_number"]


def test_builtin_eval_still_passes(tmp_path):
    from packet.eval.run import run_named_fixtures

    assert all(r.boundary.f1 == 1.0 and r.passed_gate() for r in run_named_fixtures(tmp_path))


def test_reset_after_completed_page_sequence_splits_even_with_shared_header():
    from packet.boundary import BoundaryDetector

    det = BoundaryDetector()
    t = FurnitureTagger()
    sig = None
    for i, txt in enumerate(["Binder Header\nA body.\nPage 1 of 2", "Binder Header\nA more.\nPage 2 of 2",
                             "Binder Header\nB body.\nPage 1 of 2"]):
        sig = det.observe(_ref(i), t.tag_page(_ref(i), _native(i, txt)))
    assert sig is not None and sig.confidence >= 0.55 and "sequence_completed" in sig.reason


# ---- round-1 review cases ----

def test_year_or_amount_on_last_line_is_not_a_page_number():
    t = FurnitureTagger()
    assert t.tag_page(_ref(0), _native(0, "Founded in\n2024"))[-1].kind != K.PAGE_NUMBER
    assert t.tag_page(_ref(1), _native(1, "Total due\n1500"))[-1].kind != K.PAGE_NUMBER


def test_titles_differing_by_number_are_not_headers():
    t = FurnitureTagger()
    a = t.tag_page(_ref(0), _native(0, "INVOICE 1001\nBody a.\nPage 1 of 1"))
    b = t.tag_page(_ref(1), _native(1, "INVOICE 1002\nBody b.\nPage 1 of 1"))
    assert not [x for x in t.finalize(a + b) if x.kind == K.HEADER]


def test_non_furniture_pages_are_not_fragmented():
    t = FurnitureTagger()
    a = t.tag_page(_ref(0), _native(0, "First line of a wrapped\nparagraph continues here."))
    assert [(b.id, b.text) for b in t.finalize(a)] == [("p-p0-b0", "First line of a wrapped\nparagraph continues here.")]


def test_footer_above_page_number_is_tagged():
    t = FurnitureTagger()
    a = t.tag_page(_ref(0), _native(0, "Body a.\nACME Confidential\nPage 1"))
    b = t.tag_page(_ref(1), _native(1, "Body b.\nACME Confidential\nPage 2"))
    kinds = [(x.kind, x.text) for x in t.finalize(a + b)]
    assert (K.FOOTER, "ACME Confidential") in kinds and kinds.count((K.FOOTER, "ACME Confidential")) == 2


def test_header_known_only_near_its_repeats():
    t = FurnitureTagger()
    blocks = []
    for i in range(12):
        top = "Chapter Intro" if i in (0, 1, 11) else f"Unique top {chr(65 + i)}"
        blocks += t.tag_page(_ref(i), _native(i, f"{top}\nBody {i}."))
    tagged = [b.page for b in t.finalize(blocks) if b.kind == K.HEADER]
    assert tagged == [0, 1]


def test_split_piece_kind_and_remainder_kind():
    blk = Block(id="x", page=0, kind=K.LIST_ITEM, text="- item one\n- item two\nPage 1 of 1",
                source=SourceKind.NATIVE)
    out = FurnitureTagger().tag_page(_ref(0), [blk])
    assert [b.kind for b in out] == [K.LIST_ITEM, K.PAGE_NUMBER]


# ---- round-2 review cases ----

def test_single_line_pages_get_no_header_or_footer():
    t = FurnitureTagger()
    a = t.tag_page(_ref(0), _native(0, "Exhibit 1"))
    b = t.tag_page(_ref(1), _native(1, "Exhibit 2"))
    assert all(x.kind == K.PARAGRAPH for x in t.finalize(a + b))


def test_repeated_list_item_with_number_is_not_footer():
    t = FurnitureTagger()
    a = t.tag_page(_ref(0), _native(0, "Body a.\n- item 3"))
    b = t.tag_page(_ref(1), _native(1, "Body b.\n- item 4"))
    assert not [x for x in t.finalize(a + b) if x.kind == K.FOOTER]


def test_bare_small_amount_is_not_a_page_number_without_corroboration():
    t = FurnitureTagger()
    assert t.tag_page(_ref(4), _native(4, "Quantity ordered\n45"))[-1].kind != K.PAGE_NUMBER


def test_bare_page_number_matching_position_or_neighbours():
    t = FurnitureTagger()
    assert t.tag_page(_ref(0), _native(0, "Body.\n1"))[-1].kind == K.PAGE_NUMBER
    t2 = FurnitureTagger()
    a = t2.tag_page(_ref(0), _native(0, "Body.\n7"))
    b = t2.tag_page(_ref(1), _native(1, "Body.\n8"))
    assert [x.kind for x in t2.finalize(a + b) if x.text in {"7", "8"}] == [K.PAGE_NUMBER] * 2


# ---- Tom 9:19 PM: header_changed must not fire on ordinary pages ----

def test_page_number_reset_alone_does_not_split_continuous_doc(tmp_path):
    pdf = write_text_pdf(tmp_path / "cont.pdf", [
        ["Quarterly report body starts here.", "Page 1 of 2"],
        ["More discussion of results follows.", "Page 2 of 5"],
        ["Appendix tables continue the report.", "Page 1 of 3"],
    ])
    assert Pipeline().parse(pdf).document_count() == 1


def test_header_changed_fires_when_a_running_header_stops():
    from packet.boundary import BoundaryDetector

    det, t = BoundaryDetector(), FurnitureTagger()
    pages = ["Binder A\nBody one.", "Binder A\nBody two.", "Binder A\nBody three.", "Other Doc\nFresh body."]
    sigs = [det.observe(_ref(i), t.tag_page(_ref(i), _native(i, p))) for i, p in enumerate(pages)]
    assert sigs[1] is None and sigs[2] is None
    reasons = [s.reason for s in sigs if s]
    assert not any("header_changed" in r for r in reasons[:-1])
