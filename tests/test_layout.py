"""GMA-217 (P2): layout signals on text-layer pages."""
from packet.eval.pdfutil import write_layout_pdf
from packet.extract import NativeTextExtractor
from packet.pipeline import Pipeline
from packet.source import PdfPageSource
from packet.types import BlockKind, PageClass, PageRef


def _body(n=6, start=140):
    return [(f"Body sentence number {i} keeps going for a while.", 11, start + 16 * i) for i in range(n)]


def _blocks(tmp_path, pages, i=0):
    pdf = write_layout_pdf(tmp_path / "x.pdf", pages)
    with PdfPageSource(pdf) as src:
        ref = list(src.pages())[i].model_copy(update={"page_class": PageClass.TEXT})
        return NativeTextExtractor().extract(ref, src.text_for(i), layout=lambda: src.lines_for(i))


def test_lines_carry_font_size_bold_and_position(tmp_path):
    pdf = write_layout_pdf(tmp_path / "x.pdf", [[("Big Title", 24, 80, True), ("body", 11, 140)]])
    with PdfPageSource(pdf) as src:
        lines = src.lines_for(0)
    assert [ln.text for ln in lines] == ["Big Title", "body"]
    assert lines[0].size == 24 and lines[0].bold and not lines[1].bold
    assert lines[0].top < lines[1].top  # top-down coordinates


def test_largest_font_near_top_is_title(tmp_path):
    bl = _blocks(tmp_path, [[("A Study of Things", 22, 90, True)] + _body()])
    assert bl[0].kind == BlockKind.TITLE and bl[0].text == "A Study of Things"
    assert all(b.kind != BlockKind.TITLE for b in bl[1:])


def test_body_sized_page_has_no_title(tmp_path):
    bl = _blocks(tmp_path, [_body()])
    assert all(b.kind != BlockKind.TITLE for b in bl)


def test_labelled_number_in_bottom_band_is_page_number(tmp_path):
    bl = _blocks(tmp_path, [_body() + [("Page 12", 10, 760)]])
    assert bl[-1].kind == BlockKind.PAGE_NUMBER


def test_bare_band_number_is_own_block_but_needs_corroboration(tmp_path):
    bl = _blocks(tmp_path, [_body() + [("2003", 10, 760)]])
    assert bl[-1].text == "2003" and bl[-1].kind != BlockKind.PAGE_NUMBER
    pages = [_body() + [(str(n), 10, 760)] for n in (12, 13, 14)]
    pkt = Pipeline().parse(write_layout_pdf(tmp_path / "n.pdf", pages))
    nums = [b.text for d in pkt.documents for b in d.furniture if b.kind == BlockKind.PAGE_NUMBER]
    assert nums == ["12", "13", "14"]


def test_small_band_line_is_header_only_when_its_template_repeats(tmp_path):
    one = Pipeline().parse(write_layout_pdf(tmp_path / "a.pdf", [[("DRAFT", 8, 20)] + _body(), _body()]))
    assert not [b for d in one.documents for b in d.furniture if b.kind == BlockKind.HEADER]
    pages = [[(f"Chapter {c} - Running Head", 8, 20)] + _body() for c in range(3)]
    many = Pipeline().parse(write_layout_pdf(tmp_path / "b.pdf", pages))
    assert len([b for d in many.documents for b in d.furniture if b.kind == BlockKind.HEADER]) == 3


def test_small_footnote_in_bottom_band_is_not_footer(tmp_path):
    pages = [_body() + [("1 See the appendix for details of the method used.", 8, 745)], _body()]
    pkt = Pipeline().parse(write_layout_pdf(tmp_path / "f.pdf", pages))
    assert not [b for d in pkt.documents for b in d.furniture if b.kind == BlockKind.FOOTER]


def test_rotated_and_offset_pages_place_lines_correctly(tmp_path):
    spec = [[("Big Title", 24, 80, True)] + _body()]
    plain = write_layout_pdf(tmp_path / "p.pdf", spec)
    off = write_layout_pdf(tmp_path / "o.pdf", spec, origin=(30, 40))
    with PdfPageSource(plain) as a, PdfPageSource(off) as b:
        want, got = a.lines_for(0)[0], b.lines_for(0)[0]
    assert abs(got.top - want.top) < 1 and abs(got.x0 - want.x0) < 1
    rot = write_layout_pdf(tmp_path / "r.pdf", spec, rotate=90)
    with PdfPageSource(rot) as src:
        ref = next(iter(src.pages())).model_copy(update={"page_class": PageClass.TEXT})
        bl = NativeTextExtractor().extract(ref, src.text_for(0), layout=lambda: src.lines_for(0))
    assert not any(b.kind in {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER} for b in bl)


def test_two_columns_keep_stream_order(tmp_path):
    left = [(f"Left column line {i} text", 10, 100 + 14 * i, False, 50) for i in range(5)]
    right = [(f"Right column line {i} text", 10, 100 + 14 * i, False, 320) for i in range(5)]
    bl = _blocks(tmp_path, [left + right])
    text = "\n".join(b.text for b in bl)
    assert text.index("Left column line 4") < text.index("Right column line 0")


def test_soft_hyphen_control_chars_are_dropped(tmp_path):
    pdf = write_layout_pdf(tmp_path / "x.pdf", [[("plain line", 11, 100)]])
    with PdfPageSource(pdf) as src:
        assert all(ch.isprintable() for ln in src.lines_for(0) for ch in ln.text)


def test_layout_is_forwarded_through_escalating_extractor():
    from packet.backends import EscalatingExtractor
    from packet.extract import call_extractor
    from packet.layout import TextLine

    seen = []

    class Spy(NativeTextExtractor):
        def extract(self, ref, text, layout=None):
            seen.append(layout() if callable(layout) else layout)
            return []

    ref = PageRef(packet_id="p", index=0, page_class=PageClass.TEXT, height=792)
    line = TextLine("x", 0, 10, 5, 20, 11, False)
    call_extractor(EscalatingExtractor(Spy()), ref, "x", layout=lambda: [line])
    assert seen == [[line]]


def test_layout_blocks_carry_style(tmp_path):
    bl = _blocks(tmp_path, [[("Journal of Things", 8, 20)] + _body()])
    assert bl[0].extra["size"] == 8 and bl[0].extra["band"] == "top"


def test_text_only_call_still_works():
    ref = PageRef(packet_id="p", index=0, page_class=PageClass.TEXT)
    assert NativeTextExtractor().extract(ref, "Hello\n\nWorld")


def test_repeated_body_size_top_line_is_tagged_header_by_pipeline(tmp_path):
    pages = [[("Acme Corp Quarterly", 11, 20)] + _body() for _ in range(3)]
    pkt = Pipeline().parse(write_layout_pdf(tmp_path / "r.pdf", pages))
    heads = [b for d in pkt.documents for b in d.furniture if b.kind == BlockKind.HEADER]
    assert len(heads) == 3


def _doc_pages(head, title, n, head_size=8):
    pages = [[(head, head_size, 20), (title, 22, 90, True)] + _body()]
    pages += [[(head, head_size, 20)] + _body(8, 60) for _ in range(n - 1)]
    return pages


def test_unnumbered_chapters_with_changing_running_head_stay_one_document(tmp_path):
    # GMA-216 known risk: no page numbers; each chapter changes the running header text
    # (same style) and opens with a title. One document.
    pages = []
    for c in range(3):
        pages += _doc_pages(f"Handbook - Chapter {c + 1}", f"Chapter {c + 1}: Topic", 3)
    pkt = Pipeline().parse(write_layout_pdf(tmp_path / "book.pdf", pages))
    assert len(pkt.documents) == 1


def test_unnumbered_documents_with_different_header_style_split(tmp_path):
    pages = _doc_pages("Acme Annual Report", "Annual Report 2003", 3, head_size=8)
    pages += _doc_pages("CONTOSO LEASE AGREEMENT", "Lease Agreement", 3, head_size=10)
    pkt = Pipeline().parse(write_layout_pdf(tmp_path / "two.pdf", pages))
    assert [d.page_start for d in pkt.documents] == [0, 3]


def test_running_head_dropped_on_chapter_opener_stays_one_document(tmp_path):
    pages = []
    for c in range(3):
        pages.append([(f"Chapter {c + 1}: Topic", 22, 90, True)] + _body())
        pages += [[(f"Handbook - Chapter {c + 1}", 8, 20)] + _body(8, 60) for _ in range(2)]
    pkt = Pipeline().parse(write_layout_pdf(tmp_path / "opener.pdf", pages))
    assert len(pkt.documents) == 1


def test_same_template_header_text_change_is_not_header_changed():
    from packet.boundary import BoundaryDetector
    from packet.types import Block, SourceKind

    def hb(i, text, size):
        return Block(id=f"{i}", page=i, kind=BlockKind.HEADER, text=text, source=SourceKind.NATIVE,
                     extra={"layout": True, "size": size, "top": 20.0})

    def body(i):
        return Block(id=f"{i}b", page=i, kind=BlockKind.TITLE, text="Title", source=SourceKind.NATIVE)

    d = BoundaryDetector()
    d.observe(PageRef(packet_id="p", index=0), [hb(0, "Part A", 8), body(0)])
    sig = d.observe(PageRef(packet_id="p", index=1), [hb(1, "Part B", 8), body(1)])
    assert sig is None or "header_changed" not in sig.reason
    d.observe(PageRef(packet_id="p", index=2), [hb(2, "Part B", 8), body(2)])
    sig = d.observe(PageRef(packet_id="p", index=3), [hb(3, "OTHER CO", 10), body(3)])
    assert sig is not None and "header_changed" in sig.reason


def test_unicode_spaces_still_separate_words(tmp_path):
    pdf = write_layout_pdf(tmp_path / "s.pdf", [[("Section\u00a04 applies", 11, 100)]])
    with PdfPageSource(pdf) as src:
        text = src.lines_for(0)[0].text
    assert "Section 4" in text
