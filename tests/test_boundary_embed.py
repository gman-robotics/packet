"""GMA-443: optional embed_shift signal. Off by default; weak; capped; one content vote."""
from packet.boundary import CAP_WITHOUT_RESET, PROPOSE, W_EMBED_SHIFT, BoundaryDetector
from packet.pipeline import PipelineConfig
from packet.types import Block, BlockKind, PageRef, SourceKind


def _ref(i):
    return PageRef(packet_id="p", index=i, width=612, height=792)


def _p(page, text, kind=BlockKind.PARAGRAPH):
    return Block(id=f"{page}-{kind.value}", page=page, kind=kind, text=text, source=SourceKind.NATIVE)


VEC = {"a": [1.0, 0.0], "b": [0.0, 1.0]}


def _run(pages, embedder=None, **kw):
    d = BoundaryDetector(embedder=embedder, **kw); out = []
    for i, blocks in enumerate(pages):
        out.append(d.observe(_ref(i), blocks))
    return d, out


def _emb(ref, blocks):
    return VEC[blocks[0].extra["v"]]


def _pg(i, v, title=False):
    b = _p(i, "Quarterly report on regional sales figures and staffing levels") if not title else _p(i, "Annual Report", BlockKind.TITLE)
    b.extra["v"] = v
    return [b]


def test_default_is_off():
    assert PipelineConfig().embedder is None and BoundaryDetector().embedder is None
    _, sig = _run([_pg(0, "a"), _pg(1, "b")])
    assert sig[1] is None


def test_embed_shift_alone_never_proposes():
    assert W_EMBED_SHIFT < PROPOSE
    d, sig = _run([_pg(0, "a"), _pg(1, "a"), _pg(2, "b")], _emb)
    assert sig[2] is None and d.last_embed_cosine == 0.0


def test_embed_shift_with_title_combines_under_cap():
    _, sig = _run([_pg(0, "a"), _pg(1, "a"), _pg(2, "b", title=True)], _emb)
    assert sig[2] is not None and "embed_shift" in sig[2].reason
    assert sig[2].confidence <= CAP_WITHOUT_RESET


def test_similar_page_no_shift():
    d, sig = _run([_pg(0, "a"), _pg(1, "a"), _pg(2, "a")], _emb)
    assert d.last_embed_cosine == 1.0 and sig[2] is None


def test_missing_embedding_is_ignored():
    d, sig = _run([_pg(0, "a"), _pg(1, "a")], lambda r, b: None)
    assert d.last_embed_cosine is None and sig[1] is None
