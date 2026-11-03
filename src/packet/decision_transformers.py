"""DB-2 (GMA-261): in-process Hugging Face transformers decision backend.

This module imports only the standard library and ``packet``. ``torch`` and
``transformers`` are imported inside :func:`load_transformers_runner`, which runs
on the first :meth:`TransformersDecisionBackend.decide` call. Loading a config
builds the backend object only: no model download, no socket, no key read.

Device is ``cpu`` unless the config sets ``device``. There is no CUDA auto path.
Model id and revision always come from config. A page image is forwarded only
when ``send_images`` and ``capabilities.images`` are both true.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from packet.decision import (
    DecisionBackendError,
    DecisionCapabilities,
    DecisionRequest,
    DecisionResult,
    extract_sentence_probability,
    message_for_request,
    record_unparseable,
)
from packet.decision_config import (
    config_images,
    config_positive_int,
    config_send_images,
    config_str,
    require_pillow,
)

INPUT_MODES = ("default", "features", "page_text")
_LABEL = "transformers backend"
_PLAIN_NUMBER = re.compile(r"\d*\.?\d+")
_TEMPLATE_REQUIRED = "transformers backend images require a processor chat template"
_PROCESSOR_REJECTED = "transformers backend images: processor rejected the page"
DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_OUTPUT = 16
_MISSING_IMAGE = object()

INSTRUCTION = (
    "You score document boundaries. Given the page facts below, reply with only one "
    "number between 0 and 1: the probability that this page starts a new document."
)


class TransformersRunner(Protocol):
    def generate(self, prompt: str, image: Any = None) -> str: ...


RunnerFactory = Callable[[str, str, str], TransformersRunner]


class _HFRunner:
    """Greedy text generation with a Hugging Face causal LM (real weights)."""

    def __init__(self, model: str, revision: str, device: str, max_output: int):
        import torch  # lazy: the first import of the stack
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, trust_remote_code=False)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_output = max_output

    def generate(self, prompt: str) -> str:
        tok = self.tokenizer
        text = _text_prompt(tok, prompt)
        inputs = tok(text, return_tensors="pt").to(self.device)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output,
                do_sample=False,
                pad_token_id=pad_id,
            )
        return _decode_completion(self.model, tok, out, inputs["input_ids"].shape[1])


class _HFVisionRunner:
    """Greedy image-text generation with one image part (real weights)."""

    def __init__(self, model: str, revision: str, device: str, max_output: int):
        torch = _import_vision_torch()
        model_cls, processor_cls = _import_vision_classes()
        _require_pillow()
        self._torch = torch
        self.processor = processor_cls.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self.model = model_cls.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_output = max_output

    def generate(self, prompt: str, image: Any = None) -> str:
        proc = self.processor
        text = _vision_prompt(proc, prompt)
        inputs = _vision_inputs(proc, text, image, self.device)
        tok = getattr(proc, "tokenizer", proc)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output,
                do_sample=False,
                pad_token_id=pad_id,
            )
        return _decode_completion(self.model, tok, out, inputs["input_ids"].shape[1])


def _completion_row(model: Any, row: Any, prompt_len: int) -> Any:
    config = getattr(model, "config", None)
    if getattr(config, "is_encoder_decoder", False):
        return row
    if len(row) > prompt_len:
        return row[prompt_len:]
    return row


def _decode_completion(model: Any, tokenizer: Any, out: Any, prompt_len: int) -> str:
    return tokenizer.decode(_completion_row(model, out[0], prompt_len), skip_special_tokens=True)


def _text_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return INSTRUCTION + "\n\n" + prompt


def _vision_prompt(processor: Any, prompt: str) -> str:
    if not getattr(processor, "chat_template", None):
        raise DecisionBackendError(_TEMPLATE_REQUIRED)
    try:
        return processor.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (TypeError, ValueError) as exc:
        raise DecisionBackendError(_PROCESSOR_REJECTED) from exc


def _vision_inputs(processor: Any, text: str, image: Any, device: str) -> Any:
    try:
        inputs = processor(text=text, images=image, return_tensors="pt")
    except (TypeError, ValueError) as exc:
        raise DecisionBackendError(_PROCESSOR_REJECTED) from exc
    return inputs.to(device)


def _import_vision_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DecisionBackendError("transformers backend images require torch") from exc
    return torch


def _import_vision_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise DecisionBackendError(
            "transformers backend images require AutoModelForImageTextToText"
        ) from exc
    return AutoModelForImageTextToText, AutoProcessor




def load_transformers_runner(
    model: str,
    revision: str,
    device: str,
    *,
    max_output: int = DEFAULT_MAX_OUTPUT,
    images: bool = False,
) -> TransformersRunner:
    """The only function that imports torch/transformers."""
    if images:
        return _HFVisionRunner(model, revision, device, max_output)
    return _HFRunner(model, revision, device, max_output)


def _as_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    p = float(value)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        return None
    return p


def _valid_scores(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("label") not in ("boundary", "continue"):
            return None
        p = _as_probability(item.get("probability"))
        if p is None:
            return None
        out.append({"label": item["label"], "probability": p})
    return out


def _lenient(text: str) -> DecisionResult | None:
    """Sentence-style fallback (GMA-272); see ``extract_sentence_probability``."""
    p = extract_sentence_probability(text)
    if p is None:
        return None
    return DecisionResult(probability=p, scores=None, label=None)


def parse_output(text: str) -> DecisionResult | None:
    """Return a usable P(boundary), or None when the output is unparseable."""
    body = text.strip()
    if _PLAIN_NUMBER.fullmatch(body):
        p = _as_probability(float(body))
        if p is None:
            return None
        return DecisionResult(probability=p, scores=None, label=None)
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return _lenient(body)
    if isinstance(obj, str):
        return _lenient(obj)
    if not isinstance(obj, dict):
        return None
    p = _as_probability(obj.get("probability"))
    if p is None:
        return None
    label = obj.get("label") if obj.get("label") in ("boundary", "continue") else None
    scores = _valid_scores(obj.get("scores"))
    return DecisionResult(probability=p, scores=scores, label=label)


@dataclass
class TransformersDecisionBackend:
    model: str
    revision: str
    device: str = "cpu"
    input_mode: str = "default"
    max_chars: int = DEFAULT_MAX_CHARS
    max_output: int = DEFAULT_MAX_OUTPUT
    capabilities: DecisionCapabilities = field(default_factory=DecisionCapabilities)
    send_images: bool = False
    degraded_pages: list[int] = field(default_factory=list)
    strikes: list[dict[str, Any]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    runner_factory: RunnerFactory | None = None
    last_output: str | None = None
    _runner: TransformersRunner | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        caps = self.capabilities
        if not caps.structured_output:
            self.flags.add("no_structured_output")
        if not caps.logprobs:
            self.flags.add("no_logprobs")
        if not caps.images:
            self.flags.add("no_images")

    def reset(self) -> None:
        """Clear per-parse strikes and degraded page indexes (DB-4/DB-5)."""
        self.strikes.clear()
        self.degraded_pages.clear()

    @property
    def backend_label(self) -> str:
        return "transformers"

    def reset_parse_budget(self) -> None:
        self.reset()

    def _get_runner(self) -> TransformersRunner:
        if self._runner is None:
            if self.runner_factory is not None:
                self._runner = self.runner_factory(self.model, self.revision, self.device)
            else:
                kwargs: dict[str, Any] = {"max_output": self.max_output}
                if self._images_on():
                    kwargs["images"] = True  # text-only callers keep the old signature
                self._runner = load_transformers_runner(
                    self.model, self.revision, self.device, **kwargs
                )
        return self._runner

    def _images_on(self) -> bool:
        return self.send_images and self.capabilities.images

    def _reject_missing_render(self, render: Callable[[], Any] | None) -> None:
        if self._images_on() and render is None:
            raise DecisionBackendError(
                "transformers backend: render thunk required when images are enabled"
            )

    def _image_argument(self, render: Callable[[], Any] | None) -> Any:
        if not self._images_on():
            return _MISSING_IMAGE
        return render()

    def build_message(self, request: DecisionRequest) -> str:
        # The config's ``input`` selects the payload.
        return message_for_request(
            request, input_mode=self.input_mode, max_chars=self.max_chars
        )

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        self._reject_missing_render(render)
        runner = self._get_runner()
        image = self._image_argument(render)
        prompt = self.build_message(request)
        output = _run_generate(runner, prompt, image)
        self.last_output = output
        return parse_or_strike("transformers", output, request.page_index, self.strikes, self.degraded_pages)




def _run_generate(runner: TransformersRunner, prompt: str, image: Any) -> str:
    if image is _MISSING_IMAGE:
        return runner.generate(prompt)
    return runner.generate(prompt, image)










def build_transformers_backend(data: dict[str, Any]) -> TransformersDecisionBackend:
    """Validate a ``backend: transformers`` config. Imports nothing heavy."""
    model = config_str(data, "model", _LABEL)
    revision = config_str(data, "revision", _LABEL)
    device = "cpu"
    if "device" in data:
        device = config_str(data, "device", _LABEL)
    input_mode = data.get("input", "default")
    if input_mode not in INPUT_MODES:
        raise DecisionBackendError(f"transformers backend: unknown input {input_mode!r}")
    return TransformersDecisionBackend(
        model=model,
        revision=revision,
        device=device,
        input_mode=input_mode,
        max_chars=config_positive_int(data, "max_chars", DEFAULT_MAX_CHARS, _LABEL),
        max_output=config_positive_int(data, "max_output", DEFAULT_MAX_OUTPUT, _LABEL),
        capabilities=DecisionCapabilities(images=config_images(data, _LABEL)),
        send_images=config_send_images(data, _LABEL),
    )


def _require_pillow() -> None:
    require_pillow("transformers backend images require Pillow")


def parse_or_strike(
    kind: str, output: str, page_index: int, strikes: Any, degraded_pages: list[int]
) -> DecisionResult:
    """Parsed result, or one unparseable strike (third raises ``DecisionBudgetExhausted``)."""
    parsed = parse_output(output)
    if parsed is not None:
        return parsed
    return record_unparseable(kind, page_index, strikes, degraded_pages)
