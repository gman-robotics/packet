"""DB-2 (GMA-261): in-process Hugging Face transformers decision backend."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from packet.classify import HeuristicClassifier
from packet.decision import (
    DecisionBackendError,
    DecisionRequest,
    load_decision_config,
)
from packet.furniture import FurnitureTagger
from packet.types import BBox, Block, BlockKind, PageClass, PageRef, SourceKind

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "decision"
SRC = Path(__file__).resolve().parents[1] / "src"

DEFAULT_INPUT_PAGE1 = (
    "has_num=1.000000\nnum_is_one=1.000000\nhas_total=0.000000\nprev_has_num=0.000000\n"
    "arxiv_stamp=0.000000\ntitle_top=1.000000\ntitle_lower=0.000000\nlog_words=0.693147\n"
    "prev_log_words=0.000000\nhead_abstract=0.000000\nhead_intro=0.000000\n"
    "head_affil=0.000000\nprev_refs=0.000000\nmedia_jump=0.000000\nlang_change=0.000000\n"
    "topic_sim=1.000000\ntext_excerpt=alpha\npage_number=1\npage_total=\ntitle_pos=top\n"
    "rotation=0\nhas_header=0\nhas_footer=0\nhas_page_number_tag=0\nlang=\npage_class=mixed\n"
    "class_confidence=0.450000\nsource_kinds=native\nmin_block_confidence=0.800000\n"
)


class FakeRunner:
    """Stands in for the real model: no weights, no network."""

    def __init__(self, responses: list[str], model: str, revision: str, device: str):
        self.responses = list(responses)
        self.model = model
        self.revision = revision
        self.device = device
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _fake_factory(responses: list[str], made: list[FakeRunner]):
    def factory(model: str, revision: str, device: str) -> FakeRunner:
        runner = FakeRunner(responses, model, revision, device)
        made.append(runner)
        return runner

    return factory


def _facts() -> list[dict]:
    return json.loads((FIXTURES / "logistic_facts.json").read_text())


def _page1_request(page_index: int = 1) -> DecisionRequest:
    ref = PageRef(packet_id="p", index=1)
    ref = HeuristicClassifier().classify(ref, "alpha")
    assert ref.page_class == PageClass.MIXED
    block = Block(
        id="b0",
        page=1,
        kind=BlockKind.TITLE,
        text="alpha",
        source=SourceKind.NATIVE,
        confidence=0.8,
        bbox=BBox(page=1, x0=0, y0=0, x1=10, y1=10),
    )
    blocks = FurnitureTagger().tag_page(ref, [block])
    facts = _facts()
    while len(facts) <= page_index:
        facts.append(dict(facts[-1]))
    return DecisionRequest(
        task="boundary", page_index=page_index, facts=facts, blocks=blocks, ref=ref
    )


def _load_with_fake(responses: list[str], config: str = "config_transformers.json"):
    made: list[FakeRunner] = []
    backend = load_decision_config(FIXTURES / config)
    backend.runner_factory = _fake_factory(responses, made)
    return backend, made


def test_imports_leave_torch_and_transformers_unloaded():
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import packet, packet.decision, packet.decision_transformers\n"
        "from packet.decision import load_decision_config\n"
        f"load_decision_config(pathlib.Path({str(FIXTURES / 'config_transformers.json')!r}))\n"
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    env = dict(os.environ)
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_config_loads_with_cpu_default_and_no_socket_or_key(monkeypatch):
    def no_socket(*a, **k):
        raise AssertionError("socket opened")

    reads: list[str] = []
    real_getenv = os.getenv

    def spy_getenv(name, *a):
        reads.append(name)
        return real_getenv(name, *a)

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(os, "getenv", spy_getenv)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    monkeypatch.setenv("PACKET_DECISION_API_KEY", "sentinel-not-read")
    backend = load_decision_config(FIXTURES / "config_transformers.json")
    assert backend.model == "packet-ci-fake"
    assert backend.revision == "ci-revision"
    assert backend.device == "cpu"
    assert "PACKET_DECISION_API_KEY" not in reads
    caps = backend.capabilities
    assert (caps.structured_output, caps.logprobs, caps.images) == (False, False, False)


def test_device_fixture_and_cuda_string_stored_verbatim(tmp_path):
    backend = load_decision_config(FIXTURES / "config_transformers_device.json")
    assert backend.device == "cpu"
    cfg = tmp_path / "cuda.json"
    cfg.write_text(
        json.dumps(
            {"backend": "transformers", "model": "m", "revision": "r", "device": "cuda"}
        )
    )
    assert load_decision_config(cfg).device == "cuda"


def test_runner_source_has_no_cuda_auto_path_or_hardcoded_model():
    text = (SRC / "packet" / "decision_transformers.py").read_text()
    assert "is_available" not in text
    assert "SmolLM" not in text and "huggingface.co" not in text


@pytest.mark.parametrize(
    "cfg",
    [
        {"backend": "transformers", "revision": "r"},
        {"backend": "transformers", "model": "m"},
        {"backend": "transformers", "model": "", "revision": "r"},
        {"backend": "transformers", "model": "m", "revision": "r", "device": ""},
        {"backend": "transformers", "model": "m", "revision": "r", "input": "bogus"},
        {"backend": "transformers", "model": "m", "revision": "r", "device": 0},
        {"backend": "transformers", "model": "m", "revision": "r", "max_chars": True},
        {"backend": "transformers", "model": "m", "revision": "r", "max_chars": 0},
    ],
)
def test_invalid_transformers_config_rejected(tmp_path, cfg):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(DecisionBackendError):
        load_decision_config(p)


def test_secret_field_rejected_for_transformers(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps({"backend": "transformers", "model": "m", "revision": "r", "token": "x"})
    )
    with pytest.raises(DecisionBackendError):
        load_decision_config(p)


def test_fake_records_model_revision_device_and_plain_response(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    plain = (FIXTURES / "tf_response_plain.txt").read_text()
    expected = json.loads((FIXTURES / "tf_plain_expected.json").read_text())
    backend, made = _load_with_fake([plain])
    result = backend.decide(_page1_request())
    assert len(made) == 1
    assert (made[0].model, made[0].revision, made[0].device) == (
        "packet-ci-fake",
        "ci-revision",
        "cpu",
    )
    assert result.probability == pytest.approx(expected["probability"])
    assert result.scores is expected["scores"]
    assert sorted(backend.flags) == expected["flags"]
    assert backend.degraded_pages == expected["degraded_pages"]
    assert backend.strikes == []


def test_runner_constructed_once_across_calls():
    backend, made = _load_with_fake(["0.1", "0.9"])
    backend.decide(_page1_request())
    backend.decide(_page1_request())
    assert len(made) == 1


def test_request_text_is_default_input_string():
    backend, made = _load_with_fake(["0.5"])
    backend.decide(_page1_request())
    assert made[0].prompts[0] == DEFAULT_INPUT_PAGE1


def test_garbage_response_is_one_strike():
    garbage = (FIXTURES / "tf_response_garbage.txt").read_text()
    expected = json.loads((FIXTURES / "tf_garbage_expected.json").read_text())
    backend, _ = _load_with_fake([garbage])
    result = backend.decide(_page1_request())
    assert result.probability is expected["probability"]
    assert backend.strikes == expected["strikes"]
    assert backend.degraded_pages == expected["degraded_pages"]


@pytest.mark.parametrize("raw", ["1.7", "62", "nan", "inf", "0_5", "-0.1", "true", "", "{}",
                                 '{"probability": true}', '{"probability": 2}', "[0.5]"])
def test_bad_numbers_are_unparseable(raw):
    backend, _ = _load_with_fake([raw])
    assert backend.decide(_page1_request()).probability is None
    assert backend.strikes == [{"page": 1, "reason": "unparseable"}]


@pytest.mark.parametrize("raw", ["0.62", " 0.62\n", ".5", "1", "0"])
def test_plain_numbers_accepted(raw):
    backend, _ = _load_with_fake([raw])
    assert backend.decide(_page1_request()).probability == pytest.approx(float(raw))
    assert backend.strikes == []


def test_bad_label_keeps_valid_probability():
    backend, _ = _load_with_fake(['{"label":"maybe","probability":0.4,"scores":"x"}'])
    result = backend.decide(_page1_request())
    assert result.probability == pytest.approx(0.4)
    assert result.label is None and result.scores is None


def test_features_and_page_text_inputs(tmp_path):
    for mode, check in (
        ("features", lambda p: p.startswith("has_num=1.000000\n") and "text_excerpt" not in p),
        ("page_text", lambda p: p == "al"),
    ):
        cfg = tmp_path / f"{mode}.json"
        cfg.write_text(json.dumps({"backend": "transformers", "model": "m", "revision": "r",
                                   "input": mode, "max_chars": 2}))
        made: list[FakeRunner] = []
        backend = load_decision_config(cfg)
        backend.runner_factory = _fake_factory(["0.5"], made)
        backend.decide(_page1_request())
        assert check(made[0].prompts[0]), (mode, made[0].prompts[0])


def test_max_output_key_loads_and_passes_secret_filter(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"backend": "transformers", "model": "m", "revision": "r",
                               "max_output": 8}))
    assert load_decision_config(cfg).max_output == 8


def test_no_top_level_heavy_imports_and_trust_remote_code_off():
    import ast

    text = (SRC / "packet" / "decision_transformers.py").read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert "torch" not in names and "transformers" not in names
    assert text.count("trust_remote_code=False") == 2
    assert "device_map" not in text


def test_injected_factory_never_calls_real_loader(monkeypatch):
    import packet.decision_transformers as dt

    def boom(*a, **k):
        raise AssertionError("real loader called")

    monkeypatch.setattr(dt, "load_transformers_runner", boom)
    backend, _ = _load_with_fake(["0.5"])
    backend.decide(_page1_request())


def test_json_probability_is_accepted():
    backend, _ = _load_with_fake(['{"label":"boundary","probability":0.3}'])
    result = backend.decide(_page1_request())
    assert result.probability == pytest.approx(0.3)
    assert result.label == "boundary"


def test_third_strike_raises_without_body():
    garbage = (FIXTURES / "tf_response_garbage.txt").read_text()
    expected = json.loads((FIXTURES / "tf_third_strike_expected.json").read_text())
    backend, _ = _load_with_fake([garbage] * 3)
    backend.decide(_page1_request(1))
    backend.decide(_page1_request(2))
    assert backend.strikes == expected["strikes"]
    assert backend.degraded_pages == expected["degraded_pages"]
    with pytest.raises(DecisionBackendError) as err:
        backend.decide(_page1_request(expected["raises_on_page"]))
    msg = str(err.value)
    for needle in expected["error_contains"]:
        assert needle in msg
    assert garbage not in msg and "boundary" not in msg


def test_reset_clears_strikes():
    backend, _ = _load_with_fake(["x", "0.5"])
    backend.decide(_page1_request())
    backend.reset()
    assert backend.strikes == [] and backend.degraded_pages == []


def test_reset_parse_budget_clears_strikes_and_degraded_pages():
    """GMA-298: reset_parse_budget must delegate to reset(), not be a no-op."""
    garbage = (FIXTURES / "tf_response_garbage.txt").read_text()
    backend, _ = _load_with_fake([garbage] * 3 + ["0.5"])
    backend.decide(_page1_request(1))
    backend.decide(_page1_request(2))
    assert len(backend.strikes) == 2
    assert backend.degraded_pages
    backend.reset_parse_budget()
    assert backend.strikes == []
    assert backend.degraded_pages == []
    # The budget is fresh: one more garbage page is a first strike, not the third.
    backend.decide(_page1_request(1))
    assert backend.strikes == [{"page": 1, "reason": "unparseable"}]
    assert backend.decide(_page1_request(1)).probability == pytest.approx(0.5)


def test_src_packet_does_not_import_bench_run_tf():
    for path in (SRC / "packet").rglob("*.py"):
        text = path.read_text()
        assert "run_tf" not in text and "ocr_compare" not in text, path


@pytest.mark.needs_models
def test_real_small_model_on_cpu(tmp_path):
    """Runs a real small public causal LM. CI deselects needs_models."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = os.environ.get("PACKET_TF_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    revision = os.environ.get("PACKET_TF_TEST_REVISION", "12fd25f77366fa6b3b4b768ec3050bf629380bac")
    cfg = tmp_path / "real.json"
    cfg.write_text(json.dumps({"backend": "transformers", "model": model, "revision": revision}))
    backend = load_decision_config(cfg)
    assert backend.device == "cpu"
    t0 = time.perf_counter()
    result = backend.decide(_page1_request())
    elapsed = time.perf_counter() - t0
    assert backend.last_output is not None
    print(
        f"\nmodel={model} revision={revision} device={backend.device} "
        f"elapsed_s={elapsed:.2f} raw={backend.last_output!r} "
        f"probability={result.probability} strikes={backend.strikes}"
    )
    assert result.probability is None or 0.0 <= result.probability <= 1.0
