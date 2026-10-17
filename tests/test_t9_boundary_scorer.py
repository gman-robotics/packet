"""T9 (GMA-257): JSON logistic boundary scorer wired under observe (opt-in)."""

from __future__ import annotations

import json
import os
import stat

import pytest

from packet.boundary import CAP_WITHOUT_RESET, BoundaryDetector, _extract_page_number
from packet.boundary_model import (
    CAP_WITHOUT_PAGE_RESET,
    FEATURE_NAMES,
    PROPOSE_FLOOR,
    BoundaryWeightsError,
    combined_confidence,
    export_weights_json,
    featurize_from_facts,
    load_logistic_scorer,
    model_raised_confidence,
    page_fact_row,
    predict_probability,
    train_logistic,
)
from packet.eval.pdfutil import write_text_pdf
from packet.lookahead import LookaheadSplitter, PageObs
from packet.pipeline import Pipeline, PipelineConfig
from packet.types import Block, BlockKind, PageRef, SourceKind

DLW12_SILENT_CUT_PAGE = 25  # T8 holdout page index (page_end 24, key '25')


def _minimal_weights_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "mean": [0.0] * len(FEATURE_NAMES),
        "std": [1.0] * len(FEATURE_NAMES),
        "weights": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
    }
    payload.update(overrides)
    return json.dumps(payload)


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


def _trainer_facts(pages: list[list[Block]]) -> list[dict]:
    facts = []
    for i, blocks in enumerate(pages):
        ref = _ref(i)
        pn, pt = _extract_page_number(blocks)
        facts.append(page_fact_row(ref, blocks, page_number=pn, page_total=pt))
    return facts


def _runtime_facts(pages: list[list[Block]]) -> list[dict]:
    det = BoundaryDetector()
    for i, blocks in enumerate(pages):
        det.observe(_ref(i), blocks)
    return det._page_facts


def _assert_feature_parity(pages: list[list[Block]]) -> None:
    trainer = _trainer_facts(pages)
    runtime = _runtime_facts(pages)
    assert len(trainer) == len(runtime)
    for i in range(len(trainer)):
        assert trainer[i] == runtime[i]
    for i in range(1, len(pages)):
        t_vec = featurize_from_facts(trainer, i)
        r_vec = featurize_from_facts(runtime, i)
        for j, name in enumerate(FEATURE_NAMES):
            assert t_vec[j] == r_vec[j], f"page {i} feature {name}: trainer {t_vec[j]} != runtime {r_vec[j]}"


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


def test_featurize_trainer_runtime_parity_on_builtin_fixture(tmp_path):
    pdf = write_text_pdf(
        tmp_path / "shared.pdf",
        [
            ["Binder Header", "Chapter A body.", "Page 1 of 2"],
            ["Binder Header", "Chapter A continued.", "Page 2 of 2"],
            ["Binder Header", "Chapter B body.", "Page 1 of 2"],
            ["Binder Header", "Chapter B continued.", "Page 2 of 2"],
        ],
    )
    from packet.source import PdfPageSource

    pages_blocks: list[list[Block]] = []
    with PdfPageSource(pdf) as src:
        for ref in src.pages():
            text = src.text_for(ref.index)
            pages_blocks.append([_block(ref.index, line) for line in text.splitlines() if line.strip()])
    _assert_feature_parity(pages_blocks)


def test_lang_change_parity_english_header_then_french_body():
    english_header = " ".join(
        ["The quarterly report summarizes revenue, expenses, and outlook for investors."] * 6
    )
    french_page = " ".join(
        [
            "le la les des et est une un du pour dans que qui sur pas par au avec sont ce",
            "cette ces leur leurs nous vous ils elles mais ou donc car ni lors très aussi",
        ]
        * 6
    )
    pages = [
        [
            _block(0, english_header, BlockKind.HEADER),
            _block(0, "Note marginale."),
        ],
        [_block(1, french_page)],
    ]
    _assert_feature_parity(pages)
    facts = _trainer_facts(pages)
    assert facts[0]["lang"] == "en"
    assert facts[1]["lang"] == "fr"
    vec = featurize_from_facts(facts, 1)
    assert vec[FEATURE_NAMES.index("lang_change")] == 1.0


def test_cleared_dlw00_replay_on_t5_splitter_without_model_injection():
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
    cfg = PipelineConfig()
    kw = {
        "window": cfg.lookahead_window,
        "commit": cfg.boundary_commit_confidence,
        "fallback": cfg.boundary_min_confidence,
    }
    prefix = [_obs(i) for i in range(DLW12_SILENT_CUT_PAGE)]
    at_cut = _obs(DLW12_SILENT_CUT_PAGE, 0.55)
    tail = [_obs(26, num=1, total=3), _obs(27, num=2, total=3)]
    assert DLW12_SILENT_CUT_PAGE in _splits(_run_splitter(prefix + [at_cut] + tail, **kw))

    at_cut_weak = _obs(DLW12_SILENT_CUT_PAGE, 0.45)
    assert DLW12_SILENT_CUT_PAGE not in _splits(_run_splitter(prefix + [at_cut_weak] + tail, **kw))


def test_model_reason_only_when_model_raised_confidence():
    assert not model_raised_confidence(0.9, 0.1)
    assert model_raised_confidence(None, 0.6)

    det = BoundaryDetector()
    det.observe(
        _ref(0),
        [
            _block(0, "Running head", BlockKind.HEADER),
            _block(0, "Page 11 of 12", BlockKind.FOOTER),
            _block(0, "Body " * 40),
        ],
    )
    sig = det.observe(
        _ref(1),
        [
            _block(1, "New running head", BlockKind.HEADER),
            _block(1, "Page 1 of 4", BlockKind.FOOTER),
            _block(1, "Different opening", BlockKind.TITLE),
            _block(1, "Body " * 40),
        ],
        scorer=_ConstantScorer(0.1),
    )
    assert sig is not None
    assert sig.confidence >= 0.9
    assert "+model" not in sig.reason


@pytest.mark.parametrize(
    "payload,match",
    [
        ("missing", "not found"),
        ("{", "malformed"),
        ("[]", "JSON object"),
        ("null", "JSON object"),
        ("1", "JSON object"),
        ('"x"', "JSON object"),
        ('{"schema_version": 99}', "schema_version"),
        (_minimal_weights_json(schema_version=True), "schema_version"),
        (_minimal_weights_json(std=[0.0] * len(FEATURE_NAMES)), "std entries must be"),
        (
            _minimal_weights_json(weights=[int("9" * 309)] + [0.0] * (len(FEATURE_NAMES) - 1)),
            "invalid weights",
        ),
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
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "feature_names": FEATURE_NAMES,
                    "mean": [float("nan")] * len(FEATURE_NAMES),
                    "std": [1.0] * len(FEATURE_NAMES),
                    "weights": [0.0] * len(FEATURE_NAMES),
                    "bias": 0.0,
                }
            ),
            "non-finite",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "feature_names": FEATURE_NAMES,
                    "mean": [0.0] * len(FEATURE_NAMES),
                    "std": [1.0] * len(FEATURE_NAMES),
                    "weights": "bad",
                    "bias": 0.0,
                }
            ),
            "weights",
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


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_loader_unreadable_file(tmp_path):
    path = tmp_path / "w.json"
    export_weights_json(
        path,
        train_logistic([[0.0] * len(FEATURE_NAMES)], [0]),
        training_packets=["t"],
    )
    path.chmod(0)
    try:
        with pytest.raises(BoundaryWeightsError, match="cannot read"):
            load_logistic_scorer(path)
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_cli_missing_weights_prints_one_line_stderr(tmp_path, capsys):
    from packet import cli

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    rc = cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json"), "--boundary-weights", str(tmp_path / "nope.json")])
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert err.startswith("packet:")


def test_cli_huge_integer_coefficient_exits_2(tmp_path, capsys):
    from packet import cli

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    weights = tmp_path / "w.json"
    weights.write_text(
        _minimal_weights_json(weights=[int("9" * 309)] + [0.0] * (len(FEATURE_NAMES) - 1))
    )
    rc = cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json"), "--boundary-weights", str(weights)])
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert err.startswith("packet:")


def test_cli_zero_std_exits_2(tmp_path, capsys):
    from packet import cli

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    weights = tmp_path / "w.json"
    weights.write_text(_minimal_weights_json(std=[0.0] * len(FEATURE_NAMES)))
    rc = cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json"), "--boundary-weights", str(weights)])
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert err.startswith("packet:")


def test_cli_json_array_weights_exits_2(tmp_path, capsys):
    from packet import cli

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    weights = tmp_path / "w.json"
    weights.write_text("[]")
    rc = cli.main(["parse", str(pdf), "-o", str(tmp_path / "out.json"), "--boundary-weights", str(weights)])
    err = capsys.readouterr().err.strip()
    assert rc == 2
    assert err.count("\n") == 0
    assert err.startswith("packet:")


def test_combined_confidence_matches_t8_bench():
    assert combined_confidence(None, 0.99) == pytest.approx(CAP_WITHOUT_PAGE_RESET)
    assert combined_confidence(0.9, 0.1) == pytest.approx(0.9)
    assert combined_confidence(None, PROPOSE_FLOOR - 0.01) is None
