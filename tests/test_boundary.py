import pytest

from packet.boundary import (
    BoundaryDetector,
    _extract_page_number,
    _page_fraction_whole_line,
    keyword_page_label_fraction,
)
from packet.types import Block, BlockKind, PageRef, SourceKind


def _ref(i: int, w: float = 612, h: float = 792) -> PageRef:
    return PageRef(packet_id="p", index=i, width=w, height=h)


def _block(page: int, text: str, kind: BlockKind = BlockKind.PARAGRAPH) -> Block:
    return Block(id=f"{page}-{text[:8]}", page=page, kind=kind, text=text, source=SourceKind.NATIVE)


def test_page_number_reset_proposes_boundary():
    det = BoundaryDetector()
    assert det.observe(_ref(0), [_block(0, "Page 11 of 12", BlockKind.FOOTER)]) is None
    signal = det.observe(_ref(1), [_block(1, "Page 1 of 4", BlockKind.FOOTER)])
    assert signal is not None
    assert signal.after_page == 0
    assert "page_number_reset" in signal.reason


def test_same_document_no_boundary():
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "Page 1 of 4", BlockKind.FOOTER)])
    signal = det.observe(_ref(1), [_block(1, "Page 2 of 4", BlockKind.FOOTER)])
    assert signal is None


def test_printed_page_number_wins_over_n_of_m_in_header():
    # "Volume 2 of 5" in a running header must not shadow the printed "Page 7 of 12".
    d = BoundaryDetector()
    d.observe(_ref(0), [
        _block(0, "Volume 2 of 5", BlockKind.HEADER),
        _block(0, "Body."),
        _block(0, "Page 7 of 12", BlockKind.PAGE_NUMBER),
    ])
    assert (d.last_page_number, d.last_page_total) == (7, 12)


def test_spectroscopic_fraction_in_body_is_not_page_number():
    blocks = [_block(0, "2S\n1/2 → n′")]
    assert _extract_page_number(blocks) == (None, None)


def test_whole_line_page_fraction_in_footer():
    blocks = [_block(0, "Page 1 of 2", BlockKind.FOOTER)]
    assert _extract_page_number(blocks) == (1, 2)


def test_whole_line_p_prefix_page_labels():
    assert _extract_page_number([_block(0, "p. 1 of 2", BlockKind.FOOTER)]) == (1, 2)
    assert _extract_page_number([_block(0, "p 1 of 2", BlockKind.PAGE_NUMBER)]) == (1, 2)
    assert _extract_page_number([_block(0, "P. 1/2", BlockKind.FOOTER)]) == (1, 2)
    assert _extract_page_number([_block(0, "- 1 -", BlockKind.PAGE_NUMBER)]) == (None, None)


_OCR_PREFIXED_PAGE_LABEL_CORPUS = [
    "Page 1 of 2",
    "page 1 of 2",
    "page1 of 2",
    "Page1 of 2",
    "PAGE 1/2",
    "p. 1 of 2",
    "p 1 of 2",
    "P. 1/2",
]


@pytest.mark.parametrize("line", _OCR_PREFIXED_PAGE_LABEL_CORPUS)
def test_ocr_keyword_matches_native_on_prefixed_whole_lines(line):
    """OCR gate must track native ``page`` / ``p.`` prefix arms (not a duplicate helper)."""
    native = _page_fraction_whole_line(line)
    assert native is not None
    assert keyword_page_label_fraction(line) == native


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1 of 2", None),
        ("1/2", None),
        ("Chapter 1 of 2", None),
        ("Total Page 3", None),
        ("pages 1 of 2", None),
        ("see pp. 1 of 2", None),
    ],
)
def test_keyword_page_label_rejects_non_gate_lines(text, expected):
    assert keyword_page_label_fraction(text) == expected


def test_ocr_gate_rejects_bare_native_whole_line_fraction():
    """Fails if OCR gate diverges from native matchers on bare ``N of M``."""
    assert _page_fraction_whole_line("1 of 2") == (1, 2)
    assert keyword_page_label_fraction("1 of 2") is None


def test_ocr_gate_tracks_native_prefix_arms_not_wider_native_optional():
    """Fails if native optional prefix widens (e.g. ``pg``) without OCR following."""
    assert keyword_page_label_fraction("pg 1 of 2") == _page_fraction_whole_line("pg 1 of 2")


def test_trailing_page1_label_on_invoice_line():
    from packet.boundary import match_trailing_page_label

    split = match_trailing_page_label("Amount due: 12.00 page1 of 2")
    assert split == ("Amount due: 12.00", "page1 of 2")
    assert keyword_page_label_fraction(split[1]) == (1, 2)
