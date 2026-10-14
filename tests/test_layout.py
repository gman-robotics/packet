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


def test_bare_number_in_bottom_band_is_page_number(tmp_path):
    bl = _blocks(tmp_path, [_body() + [("12", 10, 760)]])
    assert bl[-1].kind == BlockKind.PAGE_NUMBER and bl[-1].text == "12"


def test_small_font_line_in_top_band_is_header(tmp_path):
    bl = _blocks(tmp_path, [[("Journal of Things, Vol 3", 8, 20)] + _body()])
    assert bl[0].kind == BlockKind.HEADER


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
