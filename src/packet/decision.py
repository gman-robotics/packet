"""Decision-model backend protocol (DB-1+): shared formatter and logistic backend."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from packet.boundary_model import (
    FEATURE_NAMES,
    LogisticBoundaryScorer,
    featurize_from_facts,
)
from packet.types import Block, BlockKind, PageRef, SourceKind

_ALLOWED_SOURCE_KINDS = frozenset(
    {SourceKind.NATIVE, SourceKind.OCR, SourceKind.VLM, SourceKind.MERGED}
)
_VALID_ROTATIONS = frozenset({0, 90, 180, 270})
_FORBIDDEN_CONFIG_KEYS = re.compile(
    r"^(api[_-]?key|authorization|token)$", re.IGNORECASE
)
_FORBIDDEN_QUERY_KEYS = frozenset({"api_key", "token", "authorization"})


class DecisionBackendError(Exception):
    """Raised when decision config or runtime cannot proceed."""


@dataclass(frozen=True)
class DecisionCapabilities:
    structured_output: bool = False
    logprobs: bool = False
    images: bool = False


@dataclass(frozen=True)
class DecisionResult:
    probability: float | None
    scores: list[dict[str, Any]] | None = None
    label: str | None = None


@dataclass(frozen=True)
class DecisionRequest:
    task: str
    page_index: int
    facts: list[dict[str, Any]]
    blocks: list[Block]
    ref: PageRef
    input_mode: str = "default"


class DecisionBackend(Protocol):
    capabilities: DecisionCapabilities
    degraded_pages: list[int]

    @property
    def backend_label(self) -> str: ...

    def reset_parse_budget(self) -> None: ...

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult: ...


def decision_backend_passes_images(backend: DecisionBackend) -> bool:
    """True when ``decide`` should receive the page render thunk."""
    return bool(getattr(backend, "send_images", False)) and backend.capabilities.images


def resolve_decision_input_mode(backend_mode: str, request_mode: str) -> str:
    """Observe leaves ``default`` on the request; the backend config picks the wire mode."""
    if request_mode != "default":
        return request_mode
    return backend_mode


def _reject_secret_fields(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            if _FORBIDDEN_CONFIG_KEYS.search(key_str):
                raise DecisionBackendError(
                    f"decision config rejects credential field {key_str!r}"
                )
            _reject_secret_fields(value, f"{path}.{key_str}" if path else key_str)
    elif isinstance(obj, list):
        for item in obj:
            _reject_secret_fields(item, path)


def _reject_url_secrets(base_url: str) -> None:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(base_url)
    if parsed.username or parsed.password:
        raise DecisionBackendError("decision config rejects base_url userinfo")
    query = parse_qs(parsed.query, keep_blank_values=True)
    for name in query:
        if name.lower() in _FORBIDDEN_QUERY_KEYS:
            raise DecisionBackendError(f"decision config rejects query parameter {name!r}")


def _fold_text_excerpt(text: str, cap: int = 400) -> str:
    folded = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return folded[:cap]


def _format_page_int(value: int | None) -> str:
    if value is None or value < 1 or value > 9999:
        return ""
    return str(int(value))


def _format_rotation(size: tuple[Any, ...] | list[Any] | None) -> str:
    if not size or len(size) < 3:
        return ""
    rot = size[2]
    if rot is None or type(rot) is not int:
        return ""
    if rot in _VALID_ROTATIONS:
        return str(rot)
    return ""


def _furniture_flags(blocks: list[Block]) -> tuple[str, str, str]:
    has_header = any(b.kind == BlockKind.HEADER for b in blocks)
    has_footer = any(b.kind == BlockKind.FOOTER for b in blocks)
    has_page_number = any(b.kind == BlockKind.PAGE_NUMBER for b in blocks)
    return (
        "1" if has_header else "0",
        "1" if has_footer else "0",
        "1" if has_page_number else "0",
    )


def _source_kinds(blocks: list[Block]) -> str:
    kinds = sorted(
        {b.source.value for b in blocks if b.source in _ALLOWED_SOURCE_KINDS}
    )
    return ",".join(kinds)[:32]


def _min_block_confidence(blocks: list[Block]) -> str:
    if not blocks:
        return ""
    return f"{min(b.confidence for b in blocks):.6f}"


def _format_lang(lang: str | None) -> str:
    if not lang:
        return ""
    return str(lang)[:16]


def _format_title_pos(title: str | None) -> str:
    if title in ("top", "lower"):
        return title
    return ""


def format_features_message(features: list[float]) -> str:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {len(features)}")
    lines = [f"{name}={value:.6f}" for name, value in zip(FEATURE_NAMES, features)]
    return "\n".join(lines) + "\n"


def format_default_message(
    *,
    features: list[float],
    fact: dict[str, Any],
    blocks: list[Block],
    ref: PageRef,
) -> str:
    """Default decision payload: 16 feature lines plus additive page context."""
    has_header, has_footer, has_page_number = _furniture_flags(blocks)
    additive = [
        f"text_excerpt={_fold_text_excerpt(fact.get('text') or '')}",
        f"page_number={_format_page_int(fact.get('page_number'))}",
        f"page_total={_format_page_int(fact.get('page_total'))}",
        f"title_pos={_format_title_pos(fact.get('title'))}",
        f"rotation={_format_rotation(fact.get('size') or ())}",
        f"has_header={has_header}",
        f"has_footer={has_footer}",
        f"has_page_number_tag={has_page_number}",
        f"lang={_format_lang(fact.get('lang'))}",
        f"page_class={ref.page_class.value}",
        f"class_confidence={ref.class_confidence:.6f}",
        f"source_kinds={_source_kinds(blocks)}",
        f"min_block_confidence={_min_block_confidence(blocks)}",
    ]
    return format_features_message(features) + "\n".join(additive) + "\n"


@dataclass
class LogisticDecisionBackend:
    scorer: LogisticBoundaryScorer
    capabilities: DecisionCapabilities = field(default_factory=DecisionCapabilities)
    degraded_pages: list[int] = field(default_factory=list)

    @property
    def backend_label(self) -> str:
        return "logistic"

    def reset_parse_budget(self) -> None:
        """Logistic backend has no per-parse HTTP budget; no-op."""

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        del render  # logistic scores features only; no images
        features = featurize_from_facts(request.facts, request.page_index)
        probability = self.scorer.probability(features)
        return DecisionResult(probability=probability, scores=None, label=None)


def load_logistic_decision_backend(path: Path) -> LogisticDecisionBackend:
    from packet.boundary_model import load_logistic_decision_backend as _load

    return _load(path)


def load_decision_config(path: Path) -> DecisionBackend:
    if not path.exists():
        raise DecisionBackendError(f"decision config not found: {path}")
    try:
        raw = path.read_text()
    except OSError as exc:
        raise DecisionBackendError(f"cannot read decision config {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisionBackendError(f"malformed decision config JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DecisionBackendError(
            f"malformed decision config: expected JSON object, got {type(parsed).__name__}"
        )
    data: dict[str, Any] = parsed
    _reject_secret_fields(data)
    if isinstance(data.get("base_url"), str):
        _reject_url_secrets(data["base_url"])
    backend = data.get("backend")
    if backend == "logistic":
        weights = data.get("weights")
        if not isinstance(weights, str) or not weights.strip():
            raise DecisionBackendError("logistic backend requires weights path")
        return load_logistic_decision_backend(path.parent / weights)
    if backend == "http":
        from packet.decision_http import load_http_decision_backend

        return load_http_decision_backend(data)
    raise DecisionBackendError(f"unknown decision backend {backend!r}")
