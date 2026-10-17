"""T8/T9 (GMA-256, GMA-257): deterministic logistic boundary scorer and JSON weights."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from packet.boundary_layout import size_jump, title_position
from packet.topic import content_words, detect_language, same_language, topic_similarity
from packet.types import Block, PageRef

SCHEMA_VERSION = 1
# Match ``packet.boundary`` propose / cap (T8 combined_confidence parity).
PROPOSE_FLOOR = 0.4
CAP_WITHOUT_PAGE_RESET = 0.74

FEATURE_NAMES: list[str] = [
    "has_num",
    "num_is_one",
    "has_total",
    "prev_has_num",
    "arxiv_stamp",
    "title_top",
    "title_lower",
    "log_words",
    "prev_log_words",
    "head_abstract",
    "head_intro",
    "head_affil",
    "prev_refs",
    "media_jump",
    "lang_change",
    "topic_sim",
]

ARXIV = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
AFFIL = re.compile(
    r"universit|institut|department|laborator|@|college|school of",
    re.IGNORECASE,
)
REFS = re.compile(r"^\s*(references|bibliography)\b", re.IGNORECASE | re.MULTILINE)

Weights = tuple[list[float], list[float], list[float], float]  # mean, sd, coef, bias


class BoundaryWeightsError(Exception):
    """Raised when boundary weights JSON cannot be loaded or used."""


class BoundaryScorer(Protocol):
    def probability(self, features: list[float]) -> float: ...


def train_logistic(
    X: list[list[float]],
    y: list[int],
    l2: float = 1.0,
    epochs: int = 3000,
    lr: float = 0.2,
) -> Weights:
    """Deterministic batch gradient descent on standardized features, class-balanced
    (positives are rare), L2 on the coefficients."""
    n, k = len(X), len(X[0])
    mean = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in X) / n) or 1.0 for j in range(k)]
    Z = [[(r[j] - mean[j]) / sd[j] for j in range(k)] for r in X]
    pos = sum(y) or 1
    wpos, wneg = n / (2 * pos), n / (2 * max(n - pos, 1))
    coef, bias = [0.0] * k, 0.0
    for _ in range(epochs):
        g, gb = [l2 * c / n for c in coef], 0.0
        for z, t in zip(Z, y):
            p = _sigmoid(bias + sum(c * v for c, v in zip(coef, z)))
            e = (p - t) * (wpos if t else wneg) / n
            gb += e
            for j in range(k):
                g[j] += e * z[j]
        coef = [c - lr * gj for c, gj in zip(coef, g)]
        bias -= lr * gb
    return (mean, sd, coef, bias)


def predict_probability(w: Weights, x: list[float]) -> float:
    mean, sd, coef, bias = w
    return _sigmoid(
        bias + sum(c * (v - m) / s for c, v, m, s in zip(coef, x, mean, sd))
    )


def combined_confidence(heur: float | None, p: float) -> float | None:
    """Model score as a signal: capped at 0.74, ignored below the propose level, and
    never lowers the detector's own confidence."""
    m = min(p, CAP_WITHOUT_PAGE_RESET) if p >= PROPOSE_FLOOR else None
    vals = [v for v in (heur, m) if v is not None]
    return max(vals) if vals else None


def model_raised_confidence(heur: float | None, model_p: float) -> bool:
    """True when the model strictly increases the combined score before the observe cap."""
    lifted = combined_confidence(heur, model_p)
    return lifted is not None and (heur is None or lifted > heur)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


def page_fact_row(
    ref: PageRef,
    blocks: list[Block],
    *,
    page_number: int | None,
    page_total: int | None,
) -> dict[str, Any]:
    """One page row for training or serving (matches ``bench/t8_holdout.page_facts``)."""
    body = "\n".join(b.text for b in blocks if b.text)
    return {
        "page_number": page_number,
        "page_total": page_total,
        "title": title_position(blocks),
        "text": body,
        "words": content_words(blocks),
        "lang": detect_language(body),
        "size": (ref.width, ref.height, ref.rotation),
    }


def featurize_from_facts(facts: list[dict], i: int) -> list[float]:
    """16-feature vector for page index ``i`` (trainer and runtime share this)."""
    if i <= 0 or i >= len(facts):
        raise IndexError(f"page index {i} out of range for {len(facts)} fact rows")
    f, prev = facts[i], facts[i - 1]
    head = f["text"][:600].lower()
    nwords = sum(f["words"].values())
    pw = sum(prev["words"].values())
    return [
        float(f["page_number"] is not None),
        float(f["page_number"] == 1),
        float(f["page_total"] is not None),
        float(prev["page_number"] is not None),
        float(bool(ARXIV.search(f["text"]))),
        float(f["title"] == "top"),
        float(f["title"] == "lower"),
        math.log1p(nwords),
        math.log1p(pw),
        float("abstract" in head),
        float("introduction" in head),
        float(bool(AFFIL.search(head))),
        float(bool(REFS.search(prev["text"]))),
        float(bool(size_jump(prev["size"], f["size"]))),
        float(not same_language(prev["lang"], f["lang"])),
        topic_similarity(prev["words"], f["words"]) if nwords and pw else 1.0,
    ]


@dataclass
class LogisticBoundaryScorer:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float

    def probability(self, features: list[float]) -> float:
        if len(features) != len(self.weights):
            raise BoundaryWeightsError(
                f"expected {len(self.weights)} features, got {len(features)}"
            )
        w: Weights = (self.mean, self.std, self.weights, self.bias)
        return predict_probability(w, features)

    def as_weights(self) -> Weights:
        return (self.mean, self.std, self.weights, self.bias)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def export_weights_json(
    path: Path,
    w: Weights,
    *,
    training_packets: list[str],
    git_sha: str | None = None,
) -> None:
    mean, sd, coef, bias = w
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean,
        "std": sd,
        "weights": coef,
        "bias": bias,
        "training_packets": sorted(training_packets),
        "git_sha": git_sha if git_sha is not None else _git_sha(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _finite_coefficient(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoundaryWeightsError(f"malformed boundary weights: invalid {label}")
    try:
        out = float(value)
    except OverflowError as exc:
        raise BoundaryWeightsError(f"malformed boundary weights: invalid {label}") from exc
    if not math.isfinite(out):
        raise BoundaryWeightsError(f"malformed boundary weights: non-finite {label}")
    return out


def load_logistic_scorer(path: Path) -> LogisticBoundaryScorer:
    if not path.exists():
        raise BoundaryWeightsError(f"boundary weights not found: {path}")
    try:
        raw = path.read_text()
    except OSError as exc:
        raise BoundaryWeightsError(f"cannot read boundary weights {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BoundaryWeightsError(f"malformed boundary weights JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BoundaryWeightsError(
            f"malformed boundary weights: expected JSON object, got {type(parsed).__name__}"
        )
    data: dict[str, Any] = parsed
    schema = data.get("schema_version")
    if isinstance(schema, bool) or schema != SCHEMA_VERSION:
        raise BoundaryWeightsError(
            f"unsupported boundary weights schema_version {schema!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    names = data.get("feature_names")
    if names != FEATURE_NAMES:
        raise BoundaryWeightsError(
            f"boundary weights feature_names mismatch (expected {FEATURE_NAMES!r}, got {names!r})"
        )
    for key in ("mean", "std", "weights"):
        if key not in data or not isinstance(data[key], list):
            raise BoundaryWeightsError(f"malformed boundary weights: missing or invalid {key!r}")
        if len(data[key]) != len(FEATURE_NAMES):
            raise BoundaryWeightsError(
                f"malformed boundary weights: {key} length {len(data[key])} "
                f"!= {len(FEATURE_NAMES)} features"
            )
        for item in data[key]:
            _finite_coefficient(item, key)
    if "bias" not in data:
        raise BoundaryWeightsError("malformed boundary weights: missing or invalid bias")
    bias = _finite_coefficient(data["bias"], "bias")
    mean = [_finite_coefficient(x, "mean") for x in data["mean"]]
    std = [_finite_coefficient(x, "std") for x in data["std"]]
    if any(s <= 0 for s in std):
        raise BoundaryWeightsError("malformed boundary weights: std entries must be > 0")
    weights = [_finite_coefficient(x, "weights") for x in data["weights"]]
    return LogisticBoundaryScorer(
        feature_names=list(FEATURE_NAMES),
        mean=mean,
        std=std,
        weights=weights,
        bias=bias,
    )
