"""Opt-in Apple MLX vision decision backend (GMA-462).

Importing this module does not import ``mlx`` or ``mlx_vlm``. The first
``decide`` loads weights. Packet defaults stay off this path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from packet.decision import (
    DecisionBackendError,
    DecisionCapabilities,
    DecisionRequest,
    DecisionResult,
    message_for_request,
)
from packet.decision_config import (
    config_images,
    config_positive_int,
    config_send_images,
    config_str,
    require_pillow,
)
from packet.decision_transformers import (
    DEFAULT_MAX_CHARS,
    INPUT_MODES,
    INSTRUCTION,
    parse_or_strike,
)

_LABEL = "mlx_vlm backend"
_HEAD_CHARS = 600
_HEAD_WINDOW = 2
_MISSING_RENDER = "mlx_vlm backend: render thunk required when images are enabled"


class MlxRunner(Protocol):
    def generate(self, prompt: str, image: Any = None) -> str: ...


RunnerFactory = Callable[[str, str], MlxRunner]


class _MlxRunner:
    """One pinned MLX-VLM model. ``generate`` is ``mlx_vlm.generate``."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        generate: Callable[..., Any],
        apply_chat: Callable[..., Any],
    ):
        self.model = model
        self.processor = processor
        self._generate = generate
        self._apply = apply_chat

    def generate(self, prompt: str, image: Any = None) -> str:
        formatted = self._format(prompt, image)
        return self._complete(formatted, image)

    def _format(self, prompt: str, image: Any) -> Any:
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        num_images = 0 if image is None else 1
        try:
            return self._apply(
                self.processor, self.model.config, messages, num_images=num_images
            )
        except DecisionBackendError:
            raise
        except Exception as exc:
            raise DecisionBackendError("mlx_vlm backend: generate failed") from exc

    def _complete(self, formatted: Any, image: Any) -> str:
        try:
            result = self._generate(
                self.model,
                self.processor,
                formatted,
                image=None if image is None else [image],
                max_tokens=64,
                temperature=0.0,
                verbose=False,
            )
        except DecisionBackendError:
            raise
        except Exception as exc:
            raise DecisionBackendError("mlx_vlm backend: generate failed") from exc
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise DecisionBackendError("mlx_vlm backend: generate returned no text")
        return text


def _import_named(name: str, message: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise DecisionBackendError(message) from exc


def _import_mlx_stack() -> tuple[Any, Any, Any, Any]:
    _import_named("mlx", "mlx_vlm backend requires mlx")
    try:
        from huggingface_hub import snapshot_download
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise DecisionBackendError("mlx_vlm backend requires mlx-vlm") from exc
    return snapshot_download, load, generate, apply_chat_template




def _load_mlx_weights(snapshot_download: Any, load: Any, model: str, revision: str) -> tuple[Any, Any]:
    try:
        path = snapshot_download(model, revision=revision)
        return load(path, trust_remote_code=False)
    except DecisionBackendError:
        raise
    except Exception as exc:
        raise DecisionBackendError(f"mlx_vlm backend: failed to load {model!r}") from exc


def load_mlx_runner(model: str, revision: str) -> _MlxRunner:
    """The only function that imports mlx / mlx_vlm."""
    snapshot_download, load, generate, apply_chat = _import_mlx_stack()
    _require_mlx_pillow()
    loaded, processor = _load_mlx_weights(snapshot_download, load, model, revision)
    return _MlxRunner(loaded, processor, generate, apply_chat)


def _fold_head(text: str) -> str:
    folded = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return folded[:_HEAD_CHARS]


def _head_line(label: str, fact: dict[str, Any]) -> str:
    return f"{label}={_fold_head(str(fact.get('text') or ''))}"


def _neighbor_heads(facts: list[dict[str, Any]], page_index: int) -> str:
    lines: list[str] = []
    if page_index > 0:
        lines.append(_head_line("before_head", facts[page_index - 1]))
    last = min(len(facts) - 1, page_index + _HEAD_WINDOW)
    step = 1
    index = page_index + 1
    while index <= last:
        lines.append(_head_line(f"after_head_{step}", facts[index]))
        step += 1
        index += 1
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _append_heads(base: str, heads: str) -> str:
    if not heads:
        return base
    if base.endswith("\n"):
        return base + heads
    return base + "\n" + heads


@dataclass
class MlxVlmDecisionBackend:
    model: str
    revision: str
    input_mode: str = "default"
    max_chars: int = DEFAULT_MAX_CHARS
    capabilities: DecisionCapabilities = field(default_factory=DecisionCapabilities)
    send_images: bool = False
    degraded_pages: list[int] = field(default_factory=list)
    strikes: list[dict[str, Any]] = field(default_factory=list)
    runner_factory: RunnerFactory | None = None
    last_output: str | None = None
    last_request: DecisionRequest | None = None
    load_seconds: float | None = None
    _runner: MlxRunner | None = field(default=None, repr=False)

    @property
    def backend_label(self) -> str:
        return "mlx_vlm"

    def reset_parse_budget(self) -> None:
        self.strikes.clear()
        self.degraded_pages.clear()

    def _images_on(self) -> bool:
        return self.send_images and self.capabilities.images

    def _reject_missing_render(self, render: Callable[[], Any] | None) -> None:
        if self._images_on() and render is None:
            raise DecisionBackendError(_MISSING_RENDER)

    def _image_argument(self, render: Callable[[], Any] | None) -> Any:
        if not self._images_on():
            return None
        assert render is not None
        return render()

    def _build_runner(self) -> MlxRunner:
        if self.runner_factory is not None:
            return self.runner_factory(self.model, self.revision)
        return load_mlx_runner(self.model, self.revision)

    def _get_runner(self) -> MlxRunner:
        if self._runner is not None:
            return self._runner
        start = time.perf_counter()
        runner = self._build_runner()
        self._runner = runner
        self.load_seconds = time.perf_counter() - start
        return runner

    def build_message(self, request: DecisionRequest) -> str:
        base = message_for_request(
            request, input_mode=self.input_mode, max_chars=self.max_chars
        )
        return _append_heads(base, _neighbor_heads(request.facts, request.page_index))

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        self.last_request = request
        self._reject_missing_render(render)
        runner = self._get_runner()
        output = runner.generate(self.build_message(request), self._image_argument(render))
        self.last_output = output
        return parse_or_strike("mlx_vlm", output, request.page_index, self.strikes, self.degraded_pages)












def _input_mode(data: dict[str, Any]) -> str:
    mode = data.get("input", "default")
    if mode not in INPUT_MODES:
        raise DecisionBackendError(f"mlx_vlm backend: unknown input {mode!r}")
    return mode


def build_mlx_vlm_backend(data: dict[str, Any]) -> MlxVlmDecisionBackend:
    """Validate ``backend: mlx_vlm``. Imports nothing heavy."""
    return MlxVlmDecisionBackend(
        model=config_str(data, "model", _LABEL),
        revision=config_str(data, "revision", _LABEL),
        input_mode=_input_mode(data),
        max_chars=config_positive_int(data, "max_chars", DEFAULT_MAX_CHARS, _LABEL),
        capabilities=DecisionCapabilities(images=config_images(data, _LABEL)),
        send_images=config_send_images(data, _LABEL),
    )


def _require_mlx_pillow() -> None:
    require_pillow("mlx_vlm backend requires Pillow")
