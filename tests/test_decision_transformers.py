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
    DecisionCapabilities,
    DecisionRequest,
    load_decision_config,
)
from packet.decision_transformers import (
    TransformersDecisionBackend,
    load_transformers_runner,
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
    assert backend.send_images is False
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
        {"backend": "transformers", "model": "m", "revision": "r", "send_images": 1},
        {"backend": "transformers", "model": "m", "revision": "r", "send_images": "true"},
        {"backend": "transformers", "model": "m", "revision": "r", "send_images": None},
        {"backend": "transformers", "model": "m", "revision": "r", "capabilities": []},
        {"backend": "transformers", "model": "m", "revision": "r", "capabilities": "x"},
        {"backend": "transformers", "model": "m", "revision": "r", "capabilities": None},
        {"backend": "transformers", "model": "m", "revision": "r", "capabilities": {"images": 1}},
        {
            "backend": "transformers",
            "model": "m",
            "revision": "r",
            "capabilities": {"images": "true"},
        },
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
    assert text.count("trust_remote_code=False") == 4
    assert "AutoModelForCausalLM" in text
    assert "AutoModelForImageTextToText" in text
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


# GMA-272: sentence-style replies.
SENTENCE_REJECTS = [
    pytest.param(text, id=f"{kind}-{i}")
    for kind, texts in json.loads(
        (FIXTURES / "sentence_reject_cases.json").read_text()
    ).items()
    for i, text in enumerate(texts)
]


def test_sentence_reply_is_probability_not_strike():
    sentence = (FIXTURES / "tf_response_sentence.txt").read_text()
    expected = json.loads((FIXTURES / "tf_sentence_expected.json").read_text())
    backend, _ = _load_with_fake([sentence])
    result = backend.decide(_page1_request())
    assert result.probability == pytest.approx(expected["probability"])
    assert result.scores is expected["scores"]
    assert sorted(backend.flags) == expected["flags"]
    assert backend.strikes == expected["strikes"]
    assert backend.degraded_pages == expected["degraded_pages"]


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("The probability is 0.62.", 0.62),
        ("P(boundary)=0.7", 0.7),
        (".5 chance", 0.5),
        ("The probability is 1.", 1.0),
        ('"The probability is 0.62."', 0.62),
    ],
)
def test_sentence_accepts(raw, want):
    from packet.decision_transformers import parse_output

    result = parse_output(raw)
    assert result is not None and result.probability == pytest.approx(want)
    assert result.scores is None and result.label is None


@pytest.mark.parametrize("raw", SENTENCE_REJECTS)
def test_sentence_rejects_are_one_unparseable_strike(raw):
    backend, _ = _load_with_fake([raw])
    result = backend.decide(_page1_request())
    assert result.probability is None
    assert backend.strikes == [{"page": 1, "reason": "unparseable"}]
    assert backend.degraded_pages == [1]


def test_sentence_third_strike_message_omits_body():
    raw = "The probability is 0.62 or 0.7."
    backend, _ = _load_with_fake([raw] * 3)
    backend.decide(_page1_request(1))
    backend.decide(_page1_request(2))
    with pytest.raises(DecisionBackendError) as err:
        backend.decide(_page1_request(3))
    msg = str(err.value)
    assert "unparseable" in msg and "3" in msg and raw not in msg


class _RecordingRunner:
    """Records each ``generate`` call as ``(prompt, image)``. Image is absent when omitted."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def generate(self, prompt: str, *args: object) -> str:
        image = args[0] if args else None
        self.calls.append((prompt, image))
        return "0.5"


def _factory(runner):
    def factory(model: str, revision: str, device: str):
        return runner

    return factory


def _image_backend(runner, *, send_images: bool, images: bool) -> TransformersDecisionBackend:
    return TransformersDecisionBackend(
        model="m",
        revision="r",
        send_images=send_images,
        capabilities=DecisionCapabilities(images=images),
        runner_factory=_factory(runner),
    )


def test_images_on_calls_thunk_once_and_passes_the_pil_image():
    pytest.importorskip("PIL")
    from PIL import Image

    image = Image.new("RGB", (2, 2), "white")
    calls: list[int] = []

    def thunk():
        calls.append(1)
        return image

    runner = _RecordingRunner()
    backend = _image_backend(runner, send_images=True, images=True)
    result = backend.decide(_page1_request(), render=thunk)
    assert result.probability == pytest.approx(0.5)
    assert calls == [1]
    assert len(runner.calls) == 1
    prompt, got = runner.calls[0]
    assert prompt == DEFAULT_INPUT_PAGE1
    assert got is image
    assert "no_images" not in backend.flags


@pytest.mark.parametrize(
    ("send_images", "images"),
    [(True, False), (False, True), (False, False)],
)
def test_either_image_flag_false_does_not_call_thunk(send_images, images):
    def thunk():
        raise AssertionError("thunk called")

    class PromptOnly:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "0.5"

    runner = PromptOnly()
    backend = _image_backend(runner, send_images=send_images, images=images)
    result = backend.decide(_page1_request(), render=thunk)
    assert result.probability == pytest.approx(0.5)
    assert runner.prompts == [DEFAULT_INPUT_PAGE1]
    if images:
        assert "no_images" not in backend.flags
    else:
        assert "no_images" in backend.flags


def test_images_on_without_render_raises():
    runner = _RecordingRunner()
    backend = _image_backend(runner, send_images=True, images=True)
    with pytest.raises(DecisionBackendError, match="render thunk required"):
        backend.decide(_page1_request(), render=None)
    assert runner.calls == []


def test_text_only_runner_factory_still_takes_only_prompt():
    class PromptOnly:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "0.25"

    runner = PromptOnly()
    backend = TransformersDecisionBackend(model="m", revision="r", runner_factory=_factory(runner))
    assert backend.decide(_page1_request()).probability == pytest.approx(0.25)
    assert runner.prompts == [DEFAULT_INPUT_PAGE1]
    assert backend.send_images is False
    assert "no_images" in backend.flags


def test_image_config_keys_parsed_and_unknown_keys_ignored(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps(
            {
                "backend": "transformers",
                "model": "m",
                "revision": "r",
                "send_images": True,
                "capabilities": {"images": True, "extra": 1},
                "unused": "ignored",
            }
        )
    )
    backend = load_decision_config(cfg)
    assert backend.send_images is True
    assert backend.capabilities.images is True
    assert backend.capabilities.structured_output is False
    assert backend.capabilities.logprobs is False
    assert "no_images" not in backend.flags


def test_empty_capabilities_object_leaves_images_off(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps(
            {
                "backend": "transformers",
                "model": "m",
                "revision": "r",
                "capabilities": {},
            }
        )
    )
    backend = load_decision_config(cfg)
    assert backend.send_images is False
    assert backend.capabilities.images is False
    assert "no_images" in backend.flags


def test_secret_field_still_rejected_with_image_keys(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps(
            {
                "backend": "transformers",
                "model": "m",
                "revision": "r",
                "send_images": True,
                "capabilities": {"images": True, "token": "x"},
            }
        )
    )
    with pytest.raises(DecisionBackendError):
        load_decision_config(cfg)


def test_building_images_backend_imports_neither_torch_nor_transformers(tmp_path):
    cfg = tmp_path / "images.json"
    cfg.write_text(
        json.dumps(
            {
                "backend": "transformers",
                "model": "m",
                "revision": "r",
                "send_images": True,
                "capabilities": {"images": True},
            }
        )
    )
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import packet, packet.decision, packet.decision_transformers\n"
        "from packet.decision import load_decision_config\n"
        f"backend = load_decision_config(pathlib.Path({str(cfg)!r}))\n"
        "assert backend.send_images is True and backend.capabilities.images is True\n"
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_missing_vision_class_is_decision_backend_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *args, **kwargs):
        fromlist = kwargs.get("fromlist", ())
        if len(args) >= 3 and args[2]:
            fromlist = args[2]
        if name == "transformers" and fromlist and "AutoModelForImageTextToText" in fromlist:
            raise ImportError("cannot import name 'AutoModelForImageTextToText'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(DecisionBackendError, match="AutoModelForImageTextToText"):
        load_transformers_runner("m", "r", "cpu", images=True)


def test_missing_pillow_is_decision_backend_error(monkeypatch):
    import builtins
    import sys
    import types

    real = builtins.__import__
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForImageTextToText = object
    fake_tf.AutoProcessor = object

    def fake(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        if name == "transformers":
            return fake_tf
        return real(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(DecisionBackendError, match="Pillow"):
        load_transformers_runner("m", "r", "cpu", images=True)


def test_missing_pil_image_is_decision_backend_error(monkeypatch):
    import sys
    import types

    loaded: list[str] = []

    class _Proc:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loaded.append("processor")
            return object()

    class _Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loaded.append("model")
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = _Proc
    fake_tf.AutoModelForImageTextToText = _Model
    fake_pil = types.ModuleType("PIL")
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    with pytest.raises(DecisionBackendError, match="Pillow"):
        load_transformers_runner("m", "rev-pil", "cpu", images=True)
    assert loaded == []


def test_missing_torch_is_decision_backend_error(monkeypatch):
    import builtins
    import sys
    import types

    real = builtins.__import__
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForImageTextToText = object
    fake_tf.AutoProcessor = object
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil.Image = fake_pil_image

    def fake(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(DecisionBackendError, match="transformers backend images require torch"):
        load_transformers_runner("m", "rev-torch", "cpu", images=True)


def test_decide_asks_loader_for_vision_runner_only_when_images_on(monkeypatch):
    import packet.decision_transformers as dt

    seen: list[bool] = []

    def spy(model, revision, device, *, max_output=16, images=False):
        seen.append(images)
        return _RecordingRunner()

    monkeypatch.setattr(dt, "load_transformers_runner", spy)
    plain = TransformersDecisionBackend(model="m", revision="r")
    plain.decide(_page1_request())
    vision = TransformersDecisionBackend(
        model="m",
        revision="r",
        send_images=True,
        capabilities=DecisionCapabilities(images=True),
    )
    vision.decide(_page1_request(), render=lambda: object())
    assert seen == [False, True]


@pytest.mark.parametrize(
    ("send_images", "images"),
    [(True, False), (False, True)],
)
def test_loader_images_false_for_each_one_flag_off(monkeypatch, send_images, images):
    import packet.decision_transformers as dt

    seen: list[tuple] = []

    def spy(model, revision, device, *, max_output=16, images=False):
        seen.append((model, revision, device, max_output, images))
        return _RecordingRunner()

    monkeypatch.setattr(dt, "load_transformers_runner", spy)
    backend = TransformersDecisionBackend(
        model="packet-ci-fake",
        revision="ci-revision",
        device="meta",
        max_output=7,
        send_images=send_images,
        capabilities=DecisionCapabilities(images=images),
    )

    def thunk():
        raise AssertionError("render called")

    result = backend.decide(_page1_request(), render=thunk)
    assert result.probability == pytest.approx(0.5)
    assert seen == [("packet-ci-fake", "ci-revision", "meta", 7, False)]


def test_decide_loads_runner_before_render(monkeypatch):
    import packet.decision_transformers as dt

    order: list[str] = []

    def spy(model, revision, device, *, max_output=16, images=False):
        order.append("load")
        assert images is True
        assert (model, revision, device, max_output) == ("m", "r", "cpu", 16)
        raise DecisionBackendError("missing AutoModelForImageTextToText")

    monkeypatch.setattr(dt, "load_transformers_runner", spy)
    backend = TransformersDecisionBackend(
        model="m",
        revision="r",
        send_images=True,
        capabilities=DecisionCapabilities(images=True),
    )

    def thunk():
        order.append("render")
        return object()

    with pytest.raises(DecisionBackendError, match="AutoModelForImageTextToText"):
        backend.decide(_page1_request(), render=thunk)
    assert order == ["load"]


def test_send_images_true_capabilities_omitted_stays_text_only(tmp_path, monkeypatch):
    import packet.decision_transformers as dt

    seen: list[bool] = []
    made: list[_RecordingRunner] = []

    def spy(model, revision, device, *, max_output=16, images=False):
        seen.append(images)
        runner = _RecordingRunner()
        made.append(runner)
        return runner

    monkeypatch.setattr(dt, "load_transformers_runner", spy)
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps(
            {
                "backend": "transformers",
                "model": "packet-ci-fake",
                "revision": "ci-revision",
                "send_images": True,
            }
        )
    )
    backend = load_decision_config(cfg)
    assert backend.send_images is True
    assert backend.capabilities.images is False
    assert "no_images" in backend.flags

    def thunk():
        raise AssertionError("render called")

    result = backend.decide(_page1_request(), render=thunk)
    assert result.probability == pytest.approx(0.5)
    assert seen == [False]
    assert made[0].calls == [(DEFAULT_INPUT_PAGE1, None)]


def test_pipeline_passes_lazy_render_only_when_both_image_flags_on(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    from packet.eval.pdfutil import write_text_pdf
    from packet.extract import NativeTextExtractor
    from packet.pipeline import Pipeline, PipelineConfig
    from packet.source import PdfPageSource

    image = Image.new("RGB", (4, 4), "red")
    rendered: list[int] = []

    def fake_render(self, index: int, longest_px: int = 1540):
        rendered.append(index)
        return image

    monkeypatch.setattr(PdfPageSource, "render", fake_render)
    pdf = write_text_pdf(tmp_path / "two.pdf", [["Page one text."], ["Page two text."]])

    def run(name: str, payload: dict) -> _RecordingRunner:
        cfg = tmp_path / name
        cfg.write_text(
            json.dumps({"backend": "transformers", "model": "m", "revision": "r", **payload})
        )
        runner = _RecordingRunner()
        backend = load_decision_config(cfg)
        backend.runner_factory = _factory(runner)
        pipe = Pipeline(
            config=PipelineConfig(decision_backend=backend, split_on_boundary=False),
            extractors=[NativeTextExtractor()],
        )
        list(pipe.stream(pdf))
        return runner

    rendered.clear()
    off = run("off.json", {})
    assert len(off.calls) == 1
    assert off.calls[0][1] is None
    assert rendered == []

    for name, payload in (
        ("send-only.json", {"send_images": True}),
        ("images-only.json", {"capabilities": {"images": True}}),
    ):
        rendered.clear()
        one = run(name, payload)
        assert len(one.calls) == 1
        assert one.calls[0][1] is None
        assert rendered == []

    rendered.clear()
    on = run(
        "on.json",
        {"send_images": True, "capabilities": {"images": True}},
    )
    assert len(on.calls) == 1
    assert on.calls[0][1] is image
    assert rendered == [1]


class _FakeIds:
    def __init__(self, ids):
        self.shape = (1, len(ids))


class _FakeInputs(dict):
    def __init__(self, ids):
        super().__init__(input_ids=_FakeIds(ids), image_sizes=[[8, 9]])
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakeTok:
    pad_token_id = None
    eos_token_id = 7

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return "|".join(str(t) for t in tokens)


class _FakeProcessor:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.tokenizer = _FakeTok()
        self.calls = []
        self.last_inputs = None

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(("chat", messages, kwargs))
        return "rendered-chat"

    def __call__(self, **kwargs):
        self.calls.append(("call", kwargs))
        self.last_inputs = _FakeInputs([1, 2])
        return self.last_inputs


class _FakeVisionModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 9, 4]]


class _FakeTorch:
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def inference_mode(self):
        return self._Ctx()


def _vision_runner(chat_template):
    from packet.decision_transformers import _HFVisionRunner

    runner = _HFVisionRunner.__new__(_HFVisionRunner)
    runner._torch = _FakeTorch()
    runner.processor = _FakeProcessor(chat_template)
    runner.model = _FakeVisionModel()
    runner.device = "cpu"
    runner.max_output = 5
    return runner


@pytest.mark.parametrize("chat_template", ["{{ tpl }}", None])
def test_vision_runner_generate_sends_one_image_and_decodes_new_tokens(chat_template):
    from packet.decision_transformers import INSTRUCTION

    image = object()
    runner = _vision_runner(chat_template)
    runner.device = "unit-device"
    assert runner.generate("facts", image) == "9|4"
    if chat_template:
        assert len(runner.processor.calls) == 2
        kind, messages, kwargs = runner.processor.calls[0]
        assert kind == "chat"
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert messages == [
            {"role": "system", "content": INSTRUCTION},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "facts"},
                ],
            },
        ]
        call_kind, call_kw = runner.processor.calls[1]
        assert call_kind == "call"
        assert call_kw["images"] is image
        assert call_kw == {"text": "rendered-chat", "images": image, "return_tensors": "pt"}
    else:
        plain = INSTRUCTION + "\n\n" + "facts"
        assert runner.processor.calls == [
            ("call", {"text": plain, "images": image, "return_tensors": "pt"}),
        ]
        assert runner.processor.calls[0][1]["images"] is image
    sent = runner.processor.last_inputs
    assert sent.moved_to == "unit-device"
    kw = runner.model.kwargs
    assert kw["max_new_tokens"] == 5
    assert kw["do_sample"] is False
    assert kw["pad_token_id"] == 7
    assert kw["input_ids"] is sent["input_ids"]
    assert kw["input_ids"].shape == (1, 2)
    assert kw["image_sizes"] is sent["image_sizes"]
    assert kw["image_sizes"] == [[8, 9]]
    assert all(value is not image for value in kw.values())


def test_vision_from_pretrained_passes_revision_and_disables_remote_code(monkeypatch):
    import sys
    import types

    from packet.decision_transformers import _HFVisionRunner

    calls: list[tuple] = []
    processor_sentinel = object()

    class _Loaded:
        def __init__(self) -> None:
            self.moved = None
            self.evaluated = False

        def to(self, device):
            self.moved = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class _Processor:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append(("processor", model, kwargs.get("revision"), kwargs["trust_remote_code"]))
            return processor_sentinel

    class _Vision:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append(("model", model, kwargs.get("revision"), kwargs["trust_remote_code"]))
            return _Loaded()

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = _Processor
    fake_tf.AutoModelForImageTextToText = _Vision
    fake_torch = types.ModuleType("torch")
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil.Image = fake_pil_image
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)

    runner = load_transformers_runner(
        "packet-ci-fake", "ci-revision", "cpu", max_output=6, images=True
    )
    assert isinstance(runner, _HFVisionRunner)
    assert calls == [
        ("processor", "packet-ci-fake", "ci-revision", False),
        ("model", "packet-ci-fake", "ci-revision", False),
    ]
    assert calls[0][3] is False
    assert calls[1][3] is False
    assert runner.processor is processor_sentinel
    assert runner.model.moved == "cpu"
    assert runner.model.evaluated is True
    assert runner.device == "cpu"
    assert runner.max_output == 6
    assert runner._torch is fake_torch
