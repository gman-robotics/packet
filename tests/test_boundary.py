from packet.boundary import BoundaryDetector
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
