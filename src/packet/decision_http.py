"""OpenAI-compatible HTTP decision backend (DB-3). Stdlib only."""

from __future__ import annotations

import base64
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from packet.boundary_model import featurize_from_facts
from packet.decision import (
    DecisionBackendError,
    DecisionCapabilities,
    DecisionRequest,
    DecisionResult,
    extract_sentence_probability,
    format_default_message,
    format_features_message,
    resolve_decision_input_mode,
)

_ENV_API_KEY = "PACKET_DECISION_API_KEY"
_RETRY_STATUSES = frozenset({429, 502, 503})
_BACKOFF_S = (0.5, 1.0)


class _HttpRetryExhausted(Exception):
    """All retryable attempts failed for one decide call."""


_RETRYABLE_ERRORS = (
    TimeoutError,
    urllib.error.URLError,
    OSError,
    http.client.IncompleteRead,
)

_BOUNDARY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "boundary_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "probability", "scores"],
            "properties": {
                "label": {"type": "string", "enum": ["boundary", "continue"]},
                "probability": {"type": "number"},
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "probability"],
                        "properties": {
                            "label": {"type": "string", "enum": ["boundary", "continue"]},
                            "probability": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/chat/completions"


def _safe_http_error(status: int | str, page_index: int) -> str:
    return f"decision http {status} page {page_index}"


def _check_api_key(auth: str) -> str | None:
    if auth == "none":
        return None
    if auth != "env":
        raise DecisionBackendError(f"unsupported auth mode {auth!r}")
    value = os.environ.get(_ENV_API_KEY)
    if value is None or value == "":
        raise DecisionBackendError(f"{_ENV_API_KEY} is unset or empty")
    return value


@dataclass
class HttpDecisionBackend:
    base_url: str
    model: str
    capabilities: DecisionCapabilities = field(default_factory=DecisionCapabilities)
    timeout_s: float = 30.0
    max_retries: int = 2
    max_calls: int = 0
    auth: str = "env"
    input_mode: str = "default"
    send_images: bool = False
    max_chars: int = 4000
    degraded_pages: list[int] = field(default_factory=list)
    no_structured_output: bool = field(init=False)
    no_logprobs: bool = field(init=False)
    no_images: bool = field(init=False)
    _strike_count: int = field(default=0, repr=False)
    _calls_made: int = field(default=0, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _urlopen: Callable[..., Any] = field(default=urllib.request.urlopen, repr=False)

    def __post_init__(self) -> None:
        self.no_structured_output = not self.capabilities.structured_output
        self.no_logprobs = not self.capabilities.logprobs
        self.no_images = not self.capabilities.images

    @property
    def backend_label(self) -> str:
        return "http"

    def reset_parse_budget(self) -> None:
        """Reset per-parse strike and call counters (DB-5 wiring)."""
        self.degraded_pages.clear()
        self._strike_count = 0
        self._calls_made = 0

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        if self._calls_made >= self.max_calls:
            raise DecisionBackendError(
                f"decision http max_calls {self.max_calls} exceeded at page {request.page_index}"
            )
        self._calls_made += 1

        api_key = _check_api_key(self.auth)
        body = self._build_request_body(request, render=render)
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        url = _chat_completions_url(self.base_url)
        try:
            status, raw_body = self._post_with_retries(url, headers, payload, request.page_index)
        except _HttpRetryExhausted:
            self._record_strike(request.page_index)
            return DecisionResult(probability=None, scores=None, label=None)

        if status != 200:
            raise DecisionBackendError(_safe_http_error(status, request.page_index))

        try:
            envelope = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._record_strike(request.page_index)
            return DecisionResult(probability=None, scores=None, label=None)

        content = self._try_extract_message_content(envelope)
        if content is None:
            self._record_strike(request.page_index)
            return DecisionResult(probability=None, scores=None, label=None)

        result = self._parse_message_content(content)
        if result.probability is None:
            self._record_strike(request.page_index)
            return result
        return result

    def _record_strike(self, page_index: int) -> None:
        self.degraded_pages.append(page_index)
        self._strike_count += 1
        if self._strike_count >= 3:
            raise DecisionBackendError(f"decision http unparseable page {page_index}")

    def _build_request_body(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None,
    ) -> dict[str, Any]:
        mode = resolve_decision_input_mode(self.input_mode, request.input_mode)
        fact = request.facts[request.page_index]
        features = featurize_from_facts(request.facts, request.page_index)

        if mode == "features":
            text = format_features_message(features)
        elif mode == "page_text":
            text = (fact.get("text") or "")[: self.max_chars]
        else:
            text = format_default_message(
                features=features,
                fact=fact,
                blocks=request.blocks,
                ref=request.ref,
            )

        use_images = self.send_images and self.capabilities.images
        if use_images:
            if render is None:
                raise DecisionBackendError("render thunk required when images are enabled")
            png_bytes = render()
            if not isinstance(png_bytes, (bytes, bytearray)):
                raise DecisionBackendError("render thunk must return PNG bytes")
            b64 = base64.standard_b64encode(png_bytes).decode("ascii")
            message_content: Any = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        else:
            message_content = text

        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": message_content}],
        }
        if self.capabilities.structured_output:
            body["response_format"] = _BOUNDARY_RESPONSE_FORMAT
        if self.capabilities.logprobs:
            body["logprobs"] = True
        return body

    def _post_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        page_index: int,
    ) -> tuple[int, bytes]:
        attempts = 0
        max_attempts = 1 + self.max_retries

        while attempts < max_attempts:
            attempts += 1
            try:
                status, body = self._single_post(url, headers, payload)
            except _RETRYABLE_ERRORS:
                if attempts >= max_attempts:
                    raise _HttpRetryExhausted()
                self._sleep(_BACKOFF_S[min(attempts - 1, len(_BACKOFF_S) - 1)])
                continue

            if status == 200:
                return status, body
            if status in _RETRY_STATUSES:
                if attempts >= max_attempts:
                    raise _HttpRetryExhausted()
                self._sleep(_BACKOFF_S[min(attempts - 1, len(_BACKOFF_S) - 1)])
                continue
            if 300 <= status < 400:
                raise DecisionBackendError(_safe_http_error(status, page_index))
            raise DecisionBackendError(_safe_http_error(status, page_index))

        raise _HttpRetryExhausted()

    def _single_post(
        self,
        url: str,
        headers: dict[str, str],
        payload: bytes,
    ) -> tuple[int, bytes]:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise DecisionBackendError(f"unsupported URL scheme {parsed.scheme!r}")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        def _open(request: urllib.request.Request, timeout: float) -> Any:
            if self._urlopen is urllib.request.urlopen:
                opener = urllib.request.build_opener(_NoRedirectHandler())
                return opener.open(request, timeout=timeout)
            return self._urlopen(request, timeout=timeout)

        try:
            with _open(req, self.timeout_s) as resp:
                status = resp.status
                body = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read()
            if status in _RETRY_STATUSES or 300 <= status < 400:
                return status, body
            if status == 200:
                return status, body
            return status, body
        return status, body

    @staticmethod
    def _try_extract_message_content(envelope: Any) -> str | None:
        if not isinstance(envelope, dict):
            return None
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, str):
            return None
        return content

    def _parse_message_content(self, content: str) -> DecisionResult:
        stripped = content.strip()
        if not stripped:
            return DecisionResult(probability=None, scores=None, label=None)

        as_plain = self._try_plain_probability(stripped)
        if as_plain is None:
            as_plain = self._try_sentence_probability(stripped)
        if as_plain is not None:
            if self.capabilities.structured_output:
                self.no_structured_output = True
            if self.capabilities.logprobs:
                self.no_logprobs = True
            return DecisionResult(probability=as_plain, scores=None, label=None)

        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return DecisionResult(probability=None, scores=None, label=None)

        if not isinstance(obj, dict):
            return DecisionResult(probability=None, scores=None, label=None)

        prob = obj.get("probability")
        label = obj.get("label")
        scores_raw = obj.get("scores")
        if not isinstance(prob, (int, float)):
            return DecisionResult(probability=None, scores=None, label=None)

        p = float(prob)
        if not math.isfinite(p):
            return DecisionResult(probability=None, scores=None, label=None)
        scores: list[dict[str, Any]] | None = None
        if isinstance(scores_raw, list):
            parsed_scores: list[dict[str, Any]] = []
            for item in scores_raw:
                if not isinstance(item, dict):
                    continue
                sl = item.get("label")
                sp = item.get("probability")
                if isinstance(sl, str) and isinstance(sp, (int, float)):
                    parsed_scores.append({"label": sl, "probability": float(sp)})
            scores = parsed_scores if parsed_scores else None

        if self.capabilities.structured_output and scores is None:
            self.no_structured_output = True

        label_str = label if isinstance(label, str) else None
        return DecisionResult(probability=p, scores=scores, label=label_str)

    @staticmethod
    def _try_sentence_probability(text: str) -> float | None:
        """GMA-272 lenient fallback; JSON that decodes to a non-string stays strict."""
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return extract_sentence_probability(text)
        if isinstance(obj, str):
            return extract_sentence_probability(obj)
        return None

    @staticmethod
    def _try_plain_probability(text: str) -> float | None:
        try:
            value = float(text)
        except ValueError:
            return None
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return None
        return value


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def load_http_decision_backend(data: dict[str, Any]) -> HttpDecisionBackend:
    base_url = data.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise DecisionBackendError("http backend requires base_url")
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise DecisionBackendError("http backend requires model")

    caps_raw = data.get("capabilities")
    if caps_raw is None:
        caps_raw = {}
    if not isinstance(caps_raw, dict):
        raise DecisionBackendError("capabilities must be an object")

    capabilities = DecisionCapabilities(
        structured_output=bool(caps_raw.get("structured_output")),
        logprobs=bool(caps_raw.get("logprobs")),
        images=bool(caps_raw.get("images")),
    )

    auth = data.get("auth", "env")
    if auth not in ("env", "none"):
        raise DecisionBackendError(f"unsupported auth {auth!r}")

    return HttpDecisionBackend(
        base_url=base_url.strip(),
        model=model.strip(),
        capabilities=capabilities,
        timeout_s=float(data.get("timeout_s", 30)),
        max_retries=int(data.get("max_retries", 2)),
        max_calls=int(data.get("max_calls", 0)),
        auth=str(auth),
        input_mode=str(data.get("input", "default")),
        send_images=bool(data.get("send_images", False)),
        max_chars=int(data.get("max_chars", 4000)),
    )
