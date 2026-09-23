from packet.extract import NativeTextExtractor
from packet.types import BlockKind, PageClass, PageRef


def test_native_splits_and_tags_headings():
    ref = PageRef(packet_id="p", index=1, page_class=PageClass.TEXT)
    text = "1. Overview\n\nThis is the body.\n\n- first item\n\n- second item"
    blocks = NativeTextExtractor().extract(ref, text)
    kinds = [b.kind for b in blocks]
    assert BlockKind.HEADING in kinds
    assert BlockKind.PARAGRAPH in kinds
    assert BlockKind.LIST_ITEM in kinds
