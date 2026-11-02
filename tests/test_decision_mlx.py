"""GMA-462: MLX-VLM decision backend. Stubs only; no weights and no network."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from packet.boundary_model import page_fact_row
from packet.decision import (
    DecisionBackendError,
    DecisionBudgetExhausted,
    DecisionCapabilities,
    DecisionRequest,
    build_decision_backend,
    load_decision_config,
)
from packet.decision_mlx import (
    MlxVlmDecisionBackend,
    _MlxRunner,
    build_mlx_vlm_backend,
    load_mlx_runner,
)
from packet.types import Block, BlockKind, PageClass, PageRef

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parent


def _ref(index: int) -> PageRef:
    return PageRef(
        packet_id="p",
        index=index,
        width=612,
        height=792,
        rotation=0,
        page_class=PageClass.TEXT,
        class_confidence=0.7,
    )


def _blocks(index: int, text: str) -> list[Block]:
    return [
        Block(
            id=f"b{index}",
            page=index,
            kind=BlockKind.PARAGRAPH,
            text=text,
            confidence=0.9,
        )
    ]


def _request(page_index: int = 1) -> DecisionRequest:
    facts = []
    ref = _ref(page_index)
    blocks = _blocks(page_index, "alpha")
    for i in range(page_index + 2):
        facts.append(
            page_fact_row(_ref(i), _blocks(i, "alpha"), page_number=None, page_total=None)
        )
    return DecisionRequest("confirm_boundary", page_index, facts, blocks, ref)


class _Rec:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.images: list = []

    def generate(self, prompt: str, image=None) -> str:
        self.prompts.append(prompt)
        self.images.append(image)
        return self.responses.pop(0)


def _backend(responses: list[str], *, images: bool = False):
    made: list[_Rec] = []

    def factory(_model: str, _revision: str) -> _Rec:
        runner = _Rec(responses)
        made.append(runner)
        return runner

    backend = MlxVlmDecisionBackend(
        model="m",
        revision="r",
        send_images=images,
        capabilities=DecisionCapabilities(images=images),
        runner_factory=factory,
    )
    return backend, made


def test_mlx_config_requires_pin_and_strict_bools(tmp_path):
    good = {
        "backend": "mlx_vlm",
        "model": "repo",
        "revision": "abc",
        "device": "cuda",
        "send_images": True,
        "capabilities": {"images": True},
        "note": "ignored",
    }
    backend = build_mlx_vlm_backend(good)
    assert isinstance(backend, MlxVlmDecisionBackend)
    assert backend.model == "repo" and backend.revision == "abc"
    assert backend.send_images is True and backend.capabilities.images is True
    path = tmp_path / "mlx.json"
    path.write_text(json.dumps(good))
    loaded = load_decision_config(path)
    assert isinstance(loaded, MlxVlmDecisionBackend)
    assert loaded.revision == "abc"
    with pytest.raises(DecisionBackendError, match="revision"):
        build_mlx_vlm_backend({"backend": "mlx_vlm", "model": "m"})
    with pytest.raises(DecisionBackendError, match="send_images"):
        build_mlx_vlm_backend(
            {"backend": "mlx_vlm", "model": "m", "revision": "r", "send_images": "yes"}
        )
    with pytest.raises(DecisionBackendError, match="capabilities.images"):
        build_mlx_vlm_backend(
            {
                "backend": "mlx_vlm",
                "model": "m",
                "revision": "r",
                "capabilities": {"images": 1},
            }
        )
    with pytest.raises(DecisionBackendError, match="unknown input"):
        build_mlx_vlm_backend(
            {"backend": "mlx_vlm", "model": "m", "revision": "r", "input": "nope"}
        )
    with pytest.raises(DecisionBackendError, match="credential"):
        build_decision_backend({**good, "api_key": "sek"}, base_dir=tmp_path)


def test_importing_decision_mlx_leaves_mlx_unloaded(tmp_path):
    cfg = tmp_path / "mlx.json"
    cfg.write_text(json.dumps({"backend": "mlx_vlm", "model": "m", "revision": "r", "device": "cpu"}))
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import packet.decision_mlx\n"
        "from packet.decision import load_decision_config\n"
        "from pathlib import Path\n"
        f"backend = load_decision_config(Path({str(cfg)!r}))\n"
        "assert backend.backend_label == 'mlx_vlm'\n"
        "bad = [m for m in ('mlx', 'mlx_vlm') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_decision_mlx_source_has_no_top_level_mlx_import():
    text = (SRC / "packet" / "decision_mlx.py").read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert "mlx" not in names and "mlx_vlm" not in names
    assert "trust_remote_code=False" in text
    assert "local_files_only" not in text
    assert "huggingface.co" not in text


def test_construct_does_not_load_missing_stack_until_decide(monkeypatch):
    import packet.decision_mlx as dm

    calls: list[str] = []

    def boom(*_args, **_kwargs):
        calls.append("load")
        raise DecisionBackendError("mlx_vlm backend requires mlx-vlm")

    monkeypatch.setattr(dm, "load_mlx_runner", boom)
    backend = MlxVlmDecisionBackend(model="m", revision="r")
    assert calls == []
    with pytest.raises(DecisionBackendError, match="mlx-vlm"):
        backend.decide(_request())
    assert calls == ["load"]


def test_missing_render_raises_before_load(monkeypatch):
    import packet.decision_mlx as dm

    def boom(*_args, **_kwargs):
        raise AssertionError("loader called")

    monkeypatch.setattr(dm, "load_mlx_runner", boom)
    backend = MlxVlmDecisionBackend(
        model="m",
        revision="r",
        send_images=True,
        capabilities=DecisionCapabilities(images=True),
    )
    with pytest.raises(DecisionBackendError, match="render thunk required"):
        backend.decide(_request(), render=None)


def test_missing_mlx_and_mlx_vlm_and_pillow_messages(monkeypatch):
    import builtins

    import packet.decision_mlx as dm

    def missing_mlx(name, message):
        raise DecisionBackendError(message)

    monkeypatch.setattr(dm, "_import_named", missing_mlx)
    with pytest.raises(DecisionBackendError, match=r"requires mlx$"):
        dm._import_mlx_stack()

    monkeypatch.setattr(dm, "_import_named", lambda _name, _message: None)
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm"):
            raise ImportError("no mlx_vlm")
        if name == "huggingface_hub" or name.startswith("huggingface_hub"):
            raise ImportError("no hub")
        return real(name, *args, **kwargs)

    for mod in ("mlx_vlm", "mlx_vlm.prompt_utils", "huggingface_hub"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(DecisionBackendError, match="requires mlx-vlm"):
        dm._import_mlx_stack()

    monkeypatch.setattr(builtins, "__import__", real)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    def no_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pil")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    with pytest.raises(DecisionBackendError, match="Pillow"):
        dm._require_mlx_pillow()


def test_load_snapshots_then_loads_local_path(monkeypatch):
    import packet.decision_mlx as dm

    calls: list[tuple] = []

    def snap(model, revision=None, **kwargs):
        calls.append(("snap", model, revision, tuple(sorted(kwargs))))
        return "/cache/weights"

    def load(path, trust_remote_code=True, **kwargs):
        calls.append(("load", path, trust_remote_code, tuple(sorted(kwargs))))
        return object(), object()

    monkeypatch.setattr(dm, "_import_mlx_stack", lambda: (snap, load, lambda *_a, **_k: None, lambda *_a, **_k: None))
    monkeypatch.setattr(dm, "_require_mlx_pillow", lambda: None)
    runner = load_mlx_runner("repo", "rev")
    assert isinstance(runner, _MlxRunner)
    assert calls == [
        ("snap", "repo", "rev", ()),
        ("load", "/cache/weights", False, ()),
    ]


def test_load_failure_wraps_and_pillow_runs_before_snapshot(monkeypatch):
    import packet.decision_mlx as dm

    def snap(*_args, **_kwargs):
        raise AssertionError("snapshot called")

    monkeypatch.setattr(
        dm, "_import_mlx_stack", lambda: (snap, lambda *_a, **_k: None, None, None)
    )

    def no_pillow():
        raise DecisionBackendError("mlx_vlm backend requires Pillow")

    monkeypatch.setattr(dm, "_require_mlx_pillow", no_pillow)
    with pytest.raises(DecisionBackendError, match="Pillow"):
        load_mlx_runner("repo", "rev")

    def snap_fail(model, revision=None):
        raise RuntimeError(f"disk {model} {revision}")

    monkeypatch.setattr(dm, "_require_mlx_pillow", lambda: None)
    monkeypatch.setattr(
        dm, "_import_mlx_stack", lambda: (snap_fail, lambda *_a, **_k: None, None, None)
    )
    with pytest.raises(DecisionBackendError, match="failed to load 'repo'") as err:
        load_mlx_runner("repo", "rev")
    assert isinstance(err.value.__cause__, RuntimeError)


def test_runner_generate_kwargs_and_text_only():
    calls: list[tuple] = []

    def generate(_model, _processor, prompt, image=None, **kwargs):
        calls.append((prompt, image, kwargs))
        return SimpleNamespace(text="0.25")

    def apply(_processor, _config, messages, num_images=0):
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "facts"
        return f"formatted-{num_images}"

    model = SimpleNamespace(config=object())
    runner = _MlxRunner(model, object(), generate, apply)
    image = object()
    assert runner.generate("facts", image) == "0.25"
    assert calls[0][0] == "formatted-1"
    assert calls[0][1] == [image]
    assert calls[0][2] == {"max_tokens": 64, "temperature": 0.0, "verbose": False}
    assert runner.generate("facts", None) == "0.25"
    assert calls[1][0] == "formatted-0"
    assert calls[1][1] is None


def test_third_strike_is_budget_exhausted_and_does_not_append():
    backend, made = _backend(["nope", "nope", "nope"])
    backend.decide(_request(1))
    backend.decide(_request(2))
    assert len(backend.strikes) == 2
    assert backend.degraded_pages == [1, 2]
    with pytest.raises(DecisionBudgetExhausted, match=r"unparseable output on page 3 \(strike 3\)"):
        backend.decide(_request(3))
    assert len(backend.strikes) == 2
    assert backend.degraded_pages == [1, 2]
    assert len(made[0].prompts) == 3


def test_neighbor_heads_fold_newlines_and_cap_at_600():
    backend, made = _backend(["0.5"])
    facts = [
        page_fact_row(_ref(0), _blocks(0, "before line\r\nmore"), page_number=None, page_total=None),
        page_fact_row(_ref(1), _blocks(1, "middle"), page_number=None, page_total=None),
        page_fact_row(_ref(2), _blocks(2, "Z" * 700), page_number=None, page_total=None),
    ]
    request = DecisionRequest("confirm_boundary", 1, facts, _blocks(1, "middle"), _ref(1))
    backend.decide(request)
    prompt = made[0].prompts[0]
    assert "before_head=before line more\n" in prompt
    assert f"after_head_1={'Z' * 600}\n" in prompt
    assert "Z" * 601 not in prompt
    assert "after_head_2=" not in prompt


@pytest.mark.needs_mlx
def test_pinned_qwen_smoke():
    if os.environ.get("PACKET_MLX_CONFIRM") != "1":
        pytest.skip("set PACKET_MLX_CONFIRM=1 to load the pinned MLX model")
    image_mod = pytest.importorskip("PIL.Image")
    image = image_mod.new("RGB", (8, 8), "white")
    backend = MlxVlmDecisionBackend(
        model="mlx-community/Qwen3.5-4B-4bit",
        revision="0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
        send_images=True,
        capabilities=DecisionCapabilities(images=True),
    )
    result = backend.decide(_request(), render=lambda: image)
    assert result.probability is None or 0.0 <= result.probability <= 1.0
    assert backend.load_seconds is not None and backend.load_seconds >= 0.0


def test_neighbor_heads_labels_and_page_zero():
    import packet.decision_mlx as dm

    facts = [{"text": "zero"}, {"text": "one"}, {"text": "two"}, {"text": "three"}]
    heads = dm._neighbor_heads(facts, 0)
    assert "before_head" not in heads
    assert "after_head_1=one" in heads and "after_head_2=two" in heads
    heads = dm._neighbor_heads(facts, 1)
    assert "before_head=zero" in heads
    assert "after_head_1=two" in heads and "after_head_2=three" in heads


def test_send_images_without_capability_skips_render_and_load_time_is_small():
    backend, made = _backend(["0.7"], images=False)
    backend.send_images = True
    calls = []
    result = backend.decide(_request(), render=lambda: calls.append(1))
    assert result.probability == 0.7
    assert calls == []
    assert made[0].images == [None]
    assert 0.0 <= backend.load_seconds < 60.0


def test_positive_int_accepts_one():
    from packet.decision_config import config_positive_int

    assert config_positive_int({"k": 1}, "k", 5, "x") == 1
    with pytest.raises(DecisionBackendError):
        config_positive_int({"k": 0}, "k", 5, "x")


def test_logistic_config_rejects_blank_weights(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"backend": "logistic", "weights": "  "}))
    with pytest.raises(DecisionBackendError, match="weights"):
        load_decision_config(path)


def test_completion_row_keeps_a_prompt_length_row():
    import packet.decision_transformers as dt

    model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    assert dt._completion_row(model, [1, 2, 3], 3) == [1, 2, 3]
    assert dt._completion_row(model, [1, 2, 3, 4], 3) == [4]
