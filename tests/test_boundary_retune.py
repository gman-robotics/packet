"""GMA-216: weak signals combine, a top-of-page title weighs more, bare page numbers reset."""
from packet.boundary import BoundaryDetector
from packet.lookahead import LookaheadSplitter, PageObs
from packet.types import Block, BlockKind, PageRef, SourceKind


def _ref(i: int, w: float = 612, h: float = 792) -> PageRef:
    return PageRef(packet_id="p", index=i, width=w, height=h)


def _b(page: int, text: str, kind: BlockKind = BlockKind.PARAGRAPH) -> Block:
    return Block(id=f"{page}-{text[:8]}", page=page, kind=kind, text=text, source=SourceKind.NATIVE)


def _commits(signal) -> bool:
    """End-of-file decision with the default pipeline thresholds and no numbering evidence."""
    s = LookaheadSplitter()
    s.push(PageObs(0))
    s.push(PageObs(1, signal))
    return any(d.split for d in s.flush())


def test_two_weak_signals_commit_a_split():
    det = BoundaryDetector()
    det.observe(_ref(0, 594, 774), [_b(0, "Running head A", BlockKind.HEADER), _b(0, "body")])
    sig = det.observe(_ref(1, 595, 842), [_b(1, "Other head", BlockKind.HEADER), _b(1, "body")])
    assert sig is not None and "header_changed" in sig.reason and "media_box_changed" in sig.reason
    assert _commits(sig)


def test_single_weak_signal_does_not_commit():
    det = BoundaryDetector()
    det.observe(_ref(0, 594, 774), [_b(0, "body one")])
    sig = det.observe(_ref(1, 595, 842), [_b(1, "body two")])
    assert sig is None or not _commits(sig)


def test_title_at_top_of_page_proposes_and_commits():
    det = BoundaryDetector()
    det.observe(_ref(0), [_b(0, "body")])
    sig = det.observe(_ref(1), [_b(1, "A New Paper Title", BlockKind.TITLE), _b(1, "Abstract text")])
    assert sig is not None and "title_top" in sig.reason
    assert _commits(sig)


def test_title_lower_on_page_stays_weak():
    det = BoundaryDetector()
    det.observe(_ref(0), [_b(0, "body")])
    sig = det.observe(_ref(1), [_b(1, "para"), _b(1, "x"), _b(1, "y"), _b(1, "Mid Title", BlockKind.TITLE)])
    assert sig is None or not _commits(sig)


def test_title_below_running_header_still_counts_as_top():
    det = BoundaryDetector()
    det.observe(_ref(0), [_b(0, "body")])
    sig = det.observe(_ref(1), [_b(1, "Journal X", BlockKind.HEADER), _b(1, "New Title", BlockKind.TITLE)])
    assert sig is not None and "title_top" in sig.reason


def test_bare_tagged_page_number_reset():
    det = BoundaryDetector()
    det.observe(_ref(0), [_b(0, "body"), _b(0, "9", BlockKind.PAGE_NUMBER)])
    assert det.last_page_number == 9
    sig = det.observe(_ref(1), [_b(1, "body"), _b(1, "1", BlockKind.PAGE_NUMBER)])
    assert sig is not None and "page_number_reset" in sig.reason


def _decide(obs):
    s = LookaheadSplitter()
    out = []
    for o in obs:
        out += s.push(o)
    out += s.flush()
    return {d.page: d for d in out}


def test_unnumbered_page_implied_as_page_one_splits():
    # doc A numbered 1..3, doc B's unnumbered first page, then "2", "3"
    obs = [PageObs(0, None, 1), PageObs(1, None, 2), PageObs(2, None, 3),
           PageObs(3, None, None), PageObs(4, None, 2), PageObs(5, None, 3)]
    d = _decide(obs)
    assert d[3].split and d[3].reason == "implied_page_one"
    assert not any(d[i].split for i in (1, 2, 4, 5))


def test_two_unnumbered_opening_pages_then_three_splits_once():
    obs = [PageObs(0, None, 11), PageObs(1, None, 12), PageObs(2, None, None),
           PageObs(3, None, None), PageObs(4, None, 3), PageObs(5, None, 4)]
    d = _decide(obs)
    assert [i for i in d if d[i].split] == [2]


def test_unnumbered_insert_inside_numbered_doc_does_not_split():
    obs = [PageObs(0, None, 4), PageObs(1, None, 5), PageObs(2, None, None),
           PageObs(3, None, 6), PageObs(4, None, 7)]
    assert not any(x.split for x in _decide(obs).values())


def test_top_line_bare_page_number_is_tagged():
    from packet.furniture import FurnitureTagger
    t = FurnitureTagger()
    tagged = [t.tag_page(_ref(i), [_b(i, f"{i + 2}\nbody text line\nmore body")]) for i in range(3)]
    kinds = [[b.kind for b in page] for page in tagged]
    # Streaming: page 0 has no earlier neighbour yet; finalize() re-tags it.
    assert all(BlockKind.PAGE_NUMBER in k for k in kinds[1:])
    final = t.finalize([b for page in tagged for b in page])
    assert [b.text for b in final if b.kind == BlockKind.PAGE_NUMBER] == ["2", "3", "4"]
