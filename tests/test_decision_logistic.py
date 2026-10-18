"""DB-1 (GMA-260): DecisionBackend protocol and logistic backend."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from packet.boundary_model import featurize_from_facts, load_logistic_scorer
from packet.classify import HeuristicClassifier
from packet.decision import (
    DecisionBackendError,
    DecisionRequest,
    LogisticDecisionBackend,
    format_default_message,
    load_decision_config,
    load_logistic_decision_backend,
)
from packet.eval.run import run_fixture
from packet.furniture import FurnitureTagger
from packet.types import BBox, Block, BlockKind, PageClass, PageRef, SourceKind

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "decision"

DEFAULT_INPUT_PAGE1 = (
    "has_num=1.000000\n"
    "num_is_one=1.000000\n"
    "has_total=0.000000\n"
    "prev_has_num=0.000000\n"
    "arxiv_stamp=0.000000\n"
    "title_top=1.000000\n"
    "title_lower=0.000000\n"
    "log_words=0.693147\n"
    "prev_log_words=0.000000\n"
    "head_abstract=0.000000\n"
    "head_intro=0.000000\n"
    "head_affil=0.000000\n"
    "prev_refs=0.000000\n"
    "media_jump=0.000000\n"
    "lang_change=0.000000\n"
    "topic_sim=1.000000\n"
    "text_excerpt=alpha\n"
    "page_number=1\n"
    "page_total=\n"
    "title_pos=top\n"
    "rotation=0\n"
    "has_header=0\n"
    "has_footer=0\n"
    "has_page_number_tag=0\n"
    "lang=\n"
    "page_class=mixed\n"
    "class_confidence=0.450000\n"
    "source_kinds=native\n"
    "min_block_confidence=0.800000\n"
)


def _load_facts() -> list[dict]:
    return json.loads((FIXTURES / "logistic_facts.json").read_text())


def test_logistic_facts_fixture_shape():
    facts = _load_facts()
    assert len(facts) == 2
    assert facts[0] == {
        "page_number": None,
        "page_total": None,
        "title": None,
        "text": "",
        "words": {},
        "lang": None,
        "size": [612, 792, 0],
    }
    assert facts[1] == {
        "page_number": 1,
        "page_total": None,
        "title": "top",
        "text": "alpha",
        "words": {"alpha": 1},
        "lang": None,
        "size": [612, 792, 0],
    }


def test_logistic_probability_fixture():
    facts = _load_facts()
    scorer = load_logistic_scorer(FIXTURES / "logistic_weights.json")
    features = featurize_from_facts(facts, 1)
    p = scorer.probability(features)
    expected = (FIXTURES / "logistic_probability.txt").read_text()
    assert f"{p:.6f}\n" == expected


def test_logistic_decide_uses_sixteen_floats_only():
    facts = _load_facts()
    backend = load_logistic_decision_backend(FIXTURES / "logistic_weights.json")
    ref = PageRef(packet_id="p", index=1)
    block = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
        bbox=BBox(page=1, x0=0, y0=0, x1=10, y1=10),
    )

    class _SpyScorer:
        def probability(self, features: list[float]) -> float:
            assert len(features) == 16
            return 0.5

    backend.scorer = _SpyScorer()  # type: ignore[assignment]
    result = backend.decide(
        DecisionRequest(
            task="boundary",
            page_index=1,
            facts=facts,
            blocks=[block],
            ref=ref,
        )
    )
    assert result.probability == 0.5


def test_format_default_message_page1():
    facts = _load_facts()
    ref = PageRef(packet_id="p", index=1, page_class=PageClass.MIXED, class_confidence=0.45)
    block = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
    )
    tagged = FurnitureTagger().tag_page(ref, [block])
    features = featurize_from_facts(facts, 1)
    message = format_default_message(
        features=features,
        fact=facts[1],
        blocks=tagged,
        ref=ref,
    )
    assert message == DEFAULT_INPUT_PAGE1


def test_load_logistic_config():
    backend = load_decision_config(FIXTURES / "config_logistic.json")
    assert isinstance(backend, LogisticDecisionBackend)
    caps = json.loads((FIXTURES / "logistic_capabilities.json").read_text())
    assert backend.capabilities.structured_output == caps["structured_output"]
    assert backend.capabilities.logprobs == caps["logprobs"]
    assert backend.capabilities.images == caps["images"]


def test_unknown_backend_config_fails_at_load():
    with pytest.raises(DecisionBackendError, match="unknown decision backend"):
        load_decision_config(FIXTURES / "config_unknown_backend.json")


def test_run_fixture_still_uses_default_pipeline():
    assert "pipeline or Pipeline()" in inspect.getsource(run_fixture)


def test_heuristic_classifier_on_alpha_fixture_text():
    ref = PageRef(packet_id="p", index=1)
    out = HeuristicClassifier().classify(ref, "alpha")
    assert out.page_class == PageClass.MIXED
    assert out.class_confidence == 0.45


def test_format_rotation_empty_when_missing_or_invalid():
    from packet.decision import _format_rotation

    assert _format_rotation(None) == ""
    assert _format_rotation([]) == ""
    assert _format_rotation([612, 792]) == ""
    assert _format_rotation([612, 792, None]) == ""
    assert _format_rotation([612, 792, 45]) == ""
    assert _format_rotation([612, 792, 90.5]) == ""
    assert _format_rotation([612, 792, "90"]) == ""
    assert _format_rotation([612, 792, False]) == ""
    assert _format_rotation([612, 792, 0]) == "0"
    assert _format_rotation([612, 792, 90]) == "90"


def test_format_default_message_rotation_line_empty_when_size_short():
    facts = _load_facts()
    facts_short = dict(facts[1])
    facts_short["size"] = [612, 792]
    ref = PageRef(packet_id="p", index=1, page_class=PageClass.MIXED, class_confidence=0.45)
    block = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
    )
    features = featurize_from_facts(facts, 1)
    message = format_default_message(
        features=features,
        fact=facts_short,
        blocks=[block],
        ref=ref,
    )
    assert message == DEFAULT_INPUT_PAGE1.replace("rotation=0\n", "rotation=\n")
