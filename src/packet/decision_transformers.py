"""DB-2 (GMA-261): in-process Hugging Face transformers decision backend.

This module imports only the standard library and ``packet``. ``torch`` and
``transformers`` are imported inside :func:`load_transformers_runner`, which runs
on the first :meth:`TransformersDecisionBackend.decide` call. Loading a config
builds the backend object only: no model download, no socket, no key read.

Device is ``cpu`` unless the config sets ``device``. There is no CUDA auto path.
Model id and revision always come from config.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from packet.boundary_model import featurize_from_facts
from packet.decision import (
    DecisionBackendError,
    DecisionCapabilities,
    DecisionRequest,
    DecisionResult,
    format_default_message,
    format_features_message,
)

INPUT_MODES = ("default", "features", "page_text")
MAX_STRIKES = 3
_PLAIN_NUMBER = re.compile(r"\d*\.?\d+")
DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_OUTPUT = 16

INSTRUCTION = (
    "You score document boundaries. Given the page facts below, reply with only one "
    "number between 0 and 1: the probability that this page starts a new document."
)


class TransformersRunner(Protocol):
    def generate(self, prompt: str) -> str: ...


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
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(
                [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = INSTRUCTION + "\n\n" + prompt
        inputs = tok(text, return_tensors="pt").to(self.device)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output,
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return tok.decode(new_tokens, skip_special_tokens=True)


def load_transformers_runner(
    model: str, revision: str, device: str, *, max_output: int = DEFAULT_MAX_OUTPUT
) -> TransformersRunner:
    """The only function that imports torch/transformers."""
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
        return None
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
        """Extra to the protocol: the per-parse caller (DB-4) clears strikes here."""
        self.strikes.clear()
        self.degraded_pages.clear()

    def _get_runner(self) -> TransformersRunner:
        if self._runner is None:
            if self.runner_factory is not None:
                self._runner = self.runner_factory(self.model, self.revision, self.device)
            else:
                self._runner = load_transformers_runner(
                    self.model, self.revision, self.device, max_output=self.max_output
                )
        return self._runner

    def build_message(self, request: DecisionRequest) -> str:
        mode = self.input_mode  # the config's ``input`` selects the payload
        if mode == "page_text":
            fact = request.facts[request.page_index]
            return str(fact.get("text") or "")[: self.max_chars]
        features = featurize_from_facts(request.facts, request.page_index)
        if mode == "features":
            return format_features_message(features)
        return format_default_message(
            features=features,
            fact=request.facts[request.page_index],
            blocks=request.blocks,
            ref=request.ref,
        )

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        del render  # images capability is false for this backend
        prompt = self.build_message(request)
        output = self._get_runner().generate(prompt)
        self.last_output = output
        result = parse_output(output)
        if result is not None:
            return result
        if len(self.strikes) + 1 >= MAX_STRIKES:
            raise DecisionBackendError(
                f"transformers backend: unparseable output on page {request.page_index} "
                f"(strike {MAX_STRIKES})"
            )
        self.strikes.append({"page": request.page_index, "reason": "unparseable"})
        self.degraded_pages.append(request.page_index)
        return DecisionResult(probability=None, scores=None, label=None)


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DecisionBackendError(f"transformers backend requires non-empty {key!r}")
    return value


def _positive_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DecisionBackendError(f"transformers backend {key!r} must be a positive integer")
    return value


def build_transformers_backend(data: dict[str, Any]) -> TransformersDecisionBackend:
    """Validate a ``backend: transformers`` config. Imports nothing heavy."""
    model = _require_str(data, "model")
    revision = _require_str(data, "revision")
    device = "cpu"
    if "device" in data:
        device = _require_str(data, "device")
    input_mode = data.get("input", "default")
    if input_mode not in INPUT_MODES:
        raise DecisionBackendError(f"transformers backend: unknown input {input_mode!r}")
    return TransformersDecisionBackend(
        model=model,
        revision=revision,
        device=device,
        input_mode=input_mode,
        max_chars=_positive_int(data, "max_chars", DEFAULT_MAX_CHARS),
        max_output=_positive_int(data, "max_output", DEFAULT_MAX_OUTPUT),
    )
