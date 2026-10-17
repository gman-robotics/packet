"""T9 (GMA-257): JSON logistic boundary scorer wired under observe (opt-in)."""

from __future__ import annotations

import json

import pytest

from packet.boundary import CAP_WITHOUT_RESET, BoundaryDetector
from packet.boundary_model import (
    CAP_WITHOUT_PAGE_RESET,
    FEATURE_NAMES,
    PROPOSE_FLOOR,
    BoundaryWeightsError,
    LogisticBoundaryScorer,
    combined_confidence,
    export_weights_json,
    featurize_from_facts,
    load_logistic_scorer,
    predict_probability,
    train_logistic,
)
from packet.lookahead import LookaheadSplitter, PageObs
from packet.pipeline import Pipeline, PipelineConfig
from packet.types import Block, BlockKind, PageRef, SourceKind


def _ref(i: int) -> PageRef:
    return PageRef(packet_id="p", index=i, width=612, height=792)


def _block(page: int, text: str, kind: BlockKind = BlockKind.PARAGRAPH) -> Block:
    return Block(id=f"{page}-{text[:8]}", page=page, kind=kind, text=text, source=SourceKind.NATIVE)


class _ConstantScorer:
    def __init__(self, p: float) -> None:
        self.p = p

    def probability(self, features: list[float]) -> float:
        return self.p


def _obs(i, conf=None, num=None, total=None):
    from packet.boundary import BoundarySignal

    sig = BoundarySignal(after_page=i - 1, confidence=conf, reason="t") if conf else None
    return PageObs(index=i, signal=sig, page_num=num, page_total=total)


def _run_splitter(obs, **kw):
    sp = LookaheadSplitter(**kw)
    out = []
    for o in obs:
        out += sp.push(o)
    return out + sp.flush()


def _splits(decisions):
    return [d.page for d in decisions if d.split]


def test_model_probability_capped_at_074_without_page_number_reset():
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "Chapter one body text here.")])
    sig = det.observe(
        _ref(1),
        [_block(1, "Chapter two starts without a page label.")],
        scorer=_ConstantScorer(1.0),
    )
    assert sig is not None
    assert sig.confidence <= CAP_WITHOUT_RESET


def test_pipeline_default_scorer_none_matches_heuristic_only(tmp_path):
    from packet.eval.pdfutil import write_text_pdf

    pdf = write_text_pdf(
        tmp_path / "two.pdf",
        [
            ["Doc A", "Page 1 of 2"],
            ["Doc A cont", "Page 2 of 2"],
            ["Doc B", "Page 1 of 2"],
            ["Doc B cont", "Page 2 of 2"],
        ],
    )
    a = Pipeline().parse(pdf).document_count()
    b = Pipeline(config=PipelineConfig(scorer=None)).parse(pdf).document_count()
    assert a == b == 2


def test_exported_then_loaded_scorer_matches_trained_model(tmp_path):
    X = [
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.5, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    ]
    y = [0, 1, 0]
    w = train_logistic(X, y)
    path = tmp_path / "weights.json"
    export_weights_json(path, w, training_packets=["fixture-a"])
    loaded = load_logistic_scorer(path)
    for row in X:
        assert predict_probability(w, row) == pytest.approx(loaded.probability(row), abs=1e-9)


def test_cleared_dlw00_replay_on_t5_splitter_without_model_injection():
    """T5 merge path: the cleared silent gap splits once page 8 shows bare 3
    (``implied_page_one``), with no injected model score."""
    cfg = PipelineConfig()
    kw = {
        "window": cfg.lookahead_window,
        "commit": cfg.boundary_commit_confidence,
        "fallback": cfg.boundary_min_confidence,
    }
    obs = [_obs(0, num=1), _obs(1, num=2)]
    obs += [_obs(i) for i in range(2, 6)]
    obs += [_obs(6), _obs(7), _obs(8, num=3)]
    d = _run_splitter(obs, **kw)
    assert 6 in _splits(d)
    assert any(x.page == 6 and x.reason == "implied_page_one" for x in d)


def test_cleared_dlw00_injected_055_splits_via_implied_page_one():
    """On the long-gap dlw-00 shape, a 0.55 model signal on page 6 still splits
    when page 8 is bare 3 outside the continuation window (gap > window + 1)."""
    cfg = PipelineConfig()
    kw = {
        "window": cfg.lookahead_window,
        "commit": cfg.boundary_commit_confidence,
        "fallback": cfg.boundary_min_confidence,
    }
    obs = [_obs(0, num=1), _obs(1, num=2)]
    obs += [_obs(i) for i in range(2, 6)]
    obs += [_obs(6, 0.55), _obs(7), _obs(8, num=3)]
    d = _run_splitter(obs, **kw)
    assert 6 in _splits(d)
    assert any(x.page == 6 and x.reason == "implied_page_one" for x in d)


def test_dlw12_page25_splits_at_055_not_at_045_even_after_number_reset():
    """Replay shape around dlw-12's silent gold cut (page index 24). Fallback is 0.55;
    0.45 stays below fallback even when a later page carries a numbering reset."""
    cfg = PipelineConfig()
    kw = {
        "window": cfg.lookahead_window,
        "commit": cfg.boundary_commit_confidence,
        "fallback": cfg.boundary_min_confidence,
    }
    prefix = [_obs(i) for i in range(24)]
    at_cut = _obs(24, 0.55)
    tail = [_obs(25, num=1, total=3), _obs(26, num=2, total=3)]
    assert 24 in _splits(_run_splitter(prefix + [at_cut] + tail, **kw))

    at_cut_weak = _obs(24, 0.45)
    assert 24 not in _splits(_run_splitter(prefix + [at_cut_weak] + tail, **kw))


@pytest.mark.parametrize(
    "payload,match",
    [
        ("missing", "not found"),
        ("{", "malformed"),
        ('{"schema_version": 99}', "schema_version"),
        (
            json.dumps({"schema_version": 1, "feature_names": ["x"], "mean": [], "std": [], "weights": [], "bias": 0}),
            "feature_names",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "feature_names": FEATURE_NAMES,
                    "mean": [0.0],
                    "std": [1.0],
                    "weights": [0.0],
                    "bias": 0.0,
                }
            ),
            "length",
        ),
    ],
)
def test_loader_errors_are_clear(tmp_path, payload, match):
    path = tmp_path / "w.json"
    if payload == "missing":
        path = tmp_path / "nope.json"
    else:
        path.write_text(payload)
    with pytest.raises(BoundaryWeightsError, match=match):
        load_logistic_scorer(path)


def test_cli_missing_weights_prints_one_line_stderr(tmp_path, capsys):
    from packet import cli

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    rc = cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json"), "--boundary-weights", str(tmp_path / "nope.json")])
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert err.startswith("packet:")


def test_combined_confidence_matches_t8_bench():
    assert combined_confidence(None, 0.99) == pytest.approx(CAP_WITHOUT_PAGE_RESET)
    assert combined_confidence(0.9, 0.1) == pytest.approx(0.9)
    assert combined_confidence(None, PROPOSE_FLOOR - 0.01) is None


def test_featurize_from_facts_matches_page_pair(tmp_path):
    facts = [
        {
            "page_number": 1,
            "page_total": 2,
            "title": None,
            "text": "hello world",
            "words": {"hello": 1, "world": 1},
            "lang": "en",
            "size": (612.0, 792.0, 0),
            "_size_jump": lambda a, b: False,
        },
        {
            "page_number": 2,
            "page_total": 2,
            "title": "top",
            "text": "abstract section",
            "words": {"abstract": 1},
            "lang": "en",
            "size": (612.0, 792.0, 0),
            "_size_jump": lambda a, b: False,
        },
    ]
    vec = featurize_from_facts(facts, 1)
    det = BoundaryDetector()
    det.observe(_ref(0), [_block(0, "hello world")])
    sig = det.observe(_ref(1), [_block(1, "abstract section", BlockKind.TITLE)], scorer=LogisticBoundaryScorer(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * len(FEATURE_NAMES),
        std=[1.0] * len(FEATURE_NAMES),
        weights=[0.0] * len(FEATURE_NAMES),
        bias=0.0,
    ))
    assert sig is not None
    assert len(vec) == len(FEATURE_NAMES)
