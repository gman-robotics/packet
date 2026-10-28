"""GMA-448 Exp B2 harness: pure logic only (no PDFs, no models). The real run is bench/hunch_b2/b2.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "hunch_b2"))
import b2

from packet import boundary as B
from packet.boundary import BoundaryDetector
from packet.types import Block, BlockKind, PageRef, SourceKind


def _ref(i):
    return PageRef(packet_id="p", index=i, width=612, height=792)


def _blocks(i, text, kind=BlockKind.PARAGRAPH):
    return [Block(id=f"{i}-b", page=i, kind=kind, text=text, source=SourceKind.NATIVE)]


def _observe(backend, n=3, alone=False):
    d = BoundaryDetector(); out = []
    ctx = b2.heuristics_off() if alone else b2._null()
    with ctx:
        for i in range(n):
            backend.fixture = "fx"
            out.append(d.observe(_ref(i), _blocks(i, f"page {i} body text"), decision_backend=backend))
    return out


def test_oracle_student_never_reaches_commit_level():
    be = b2.StudentBackend("oracle", gold={"fx": {1, 2}})
    sigs = [s for s in _observe(be) if s is not None]
    assert sigs and all(s.confidence <= 0.74 for s in sigs)


def test_student_below_threshold_is_ignored():
    be = b2.StudentBackend("student", probs={("fx", 1): 0.2, ("fx", 2): 0.99}, threshold=0.9)
    sigs = _observe(be)
    assert sigs[1] is None and sigs[2] is not None and sigs[2].confidence == 0.74


def test_missing_page_prob_falls_back_to_heuristic_only():
    be = b2.StudentBackend("student", probs={}, threshold=0.5)
    assert all(s is None for s in _observe(be))


def test_heuristics_off_restores_weights():
    before = (B.W_TITLE_TOP, B.W_PAGE_RESET, B.COMBINED)
    with b2.heuristics_off():
        assert B.W_TITLE_TOP == 0.0 and B.COMBINED == 0.0
    assert (B.W_TITLE_TOP, B.W_PAGE_RESET, B.COMBINED) == before


def test_heuristics_off_silences_a_heuristic_signal_and_restores_it():
    def title_run():
        d = BoundaryDetector()
        d.observe(_ref(0), _blocks(0, "body text one"))
        return d.observe(_ref(1), _blocks(1, "Schedule K-1", BlockKind.TITLE) + _blocks(1, "body text two"))
    assert title_run() is not None
    with b2.heuristics_off():
        assert title_run() is None
    assert title_run() is not None


def test_page_recorder_sees_page_zero_and_never_embeds():
    rec = b2.PageRecorder(); rec.fixture = "fx"; d = BoundaryDetector(embedder=rec)
    for i in range(2):
        d.observe(_ref(i), _blocks(i, f"page {i}"))
    assert [p for _, p, _ in rec.seen] == [0, 1] and d.last_embed_cosine is None


def test_feats_use_page_index_not_row_position():
    import numpy as np
    E = np.eye(4); ix = {"a": 0, "b": 1, "c": 2}
    rows = [{"page": 0, "key": "a"}, {"page": 2, "key": "c"}]  # page 1 missing
    X = b2._feats(rows, E, ix)
    assert X[0][-1] == 1.0 and X[1][-1] == 0.0
    assert list(X[1][4:8]) == [0.0, 0.0, 1.0, 0.0]  # e_2 - e_1, with e_1 missing -> zero vector


def test_page_text_keeps_every_kind_and_truncates():
    blocks = _blocks(0, "Header", BlockKind.HEADER) + _blocks(0, "x" * 5000)
    t = b2.page_text(blocks)
    assert t.startswith("Header ") and len(t) == b2.MAX_CHARS


def test_default_pipeline_has_no_decision_backend():
    from packet.pipeline import PipelineConfig
    assert PipelineConfig().decision_backend is None
