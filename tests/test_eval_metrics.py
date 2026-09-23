from packet.eval.metrics import boundary_score, document_boundaries
from packet.types import LogicalDocument, Packet, PacketOrigin


def _pkt(docs: list[tuple[int, int]]) -> Packet:
    return Packet(
        id="x",
        origin=PacketOrigin(path="x.pdf", page_count=docs[-1][1] + 1 if docs else 0),
        documents=[
            LogicalDocument(id=str(i), page_start=a, page_end=b) for i, (a, b) in enumerate(docs)
        ],
    )


def test_boundaries_from_page_end():
    assert document_boundaries(_pkt([(0, 2), (3, 5)]).documents) == [2]


def test_perfect_split():
    gold = _pkt([(0, 1), (2, 3)])
    pred = _pkt([(0, 1), (2, 3)])
    s = boundary_score(gold, pred)
    assert s.f1 == 1.0


def test_single_doc_both_empty_boundaries():
    gold = _pkt([(0, 4)])
    pred = _pkt([(0, 4)])
    s = boundary_score(gold, pred)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_false_split():
    gold = _pkt([(0, 3)])
    pred = _pkt([(0, 1), (2, 3)])
    s = boundary_score(gold, pred)
    assert s.precision == 0.0
    assert s.recall == 0.0
