"""T8/T9 (GMA-256, GMA-257): deterministic logistic boundary scorer and JSON weights."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


@dataclass(frozen=True)
class _PageSlice:
    page_number: int | None
    page_total: int | None
    title: str | None
    body: str
    words: Any
    lang: str | None
    size: tuple[float | None, float | None, int]


def featurize_from_facts(facts: list[dict], i: int) -> list[float]:
    """Feature vector for page index ``i`` (bench / training parity)."""
    f, prev = facts[i], facts[i - 1] if i > 0 else None
    cur = _PageSlice(
        f["page_number"],
        f["page_total"],
        f["title"],
        f["text"],
        f["words"],
        f["lang"],
        f["size"],
    )
    prev_slice = None
    if prev is not None:
        prev_slice = _PageSlice(
            prev["page_number"],
            prev["page_total"],
            prev["title"],
            prev["text"],
            prev["words"],
            prev["lang"],
            prev["size"],
        )
    return _featurize_pair(cur, prev_slice)


def featurize_page_pair(
    ref: PageRef,
    blocks: list[Block],
    *,
    page_number: int | None,
    page_total: int | None,
    prev_page_number: int | None,
    prev_body: str,
    prev_words: Any,
    prev_lang: str | None,
    prev_size: tuple[float | None, float | None, int],
) -> list[float]:
    from packet.boundary import _title_position

    body = "\n".join(b.text for b in blocks if b.text)
    words = content_words(blocks)
    lang = detect_language(body)
    cur = _PageSlice(
        page_number,
        page_total,
        _title_position(blocks),
        body,
        words,
        lang,
        (ref.width, ref.height, ref.rotation),
    )
    prev_slice = _PageSlice(
        prev_page_number,
        None,
        None,
        prev_body,
        prev_words,
        prev_lang,
        prev_size,
    )
    return _featurize_pair(cur, prev_slice)


def _featurize_pair(cur: _PageSlice, prev: _PageSlice | None) -> list[float]:
    from packet.boundary import _size_jump

    head = cur.body[:600].lower()
    nwords = sum(cur.words.values())
    pw = sum(prev.words.values()) if prev else 0
    return [
        float(cur.page_number is not None),
        float(cur.page_number == 1),
        float(cur.page_total is not None),
        float(prev is not None and prev.page_number is not None),
        float(bool(ARXIV.search(cur.body))),
        float(cur.title == "top"),
        float(cur.title == "lower"),
        math.log1p(nwords),
        math.log1p(pw),
        float("abstract" in head),
        float("introduction" in head),
        float(bool(AFFIL.search(head))),
        float(bool(prev and REFS.search(prev.body))),
        float(bool(prev and _size_jump(prev.size, cur.size))),
        float(bool(prev and not same_language(prev.lang, cur.lang))),
        topic_similarity(prev.words, cur.words) if prev and nwords and pw else 1.0,
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


def load_logistic_scorer(path: Path) -> LogisticBoundaryScorer:
    if not path.exists():
        raise BoundaryWeightsError(f"boundary weights not found: {path}")
    try:
        raw = path.read_text()
    except OSError as exc:
        raise BoundaryWeightsError(f"cannot read boundary weights {path}: {exc}") from exc
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BoundaryWeightsError(f"malformed boundary weights JSON in {path}: {exc}") from exc
    if data.get("schema_version") != SCHEMA_VERSION:
        raise BoundaryWeightsError(
            f"unsupported boundary weights schema_version {data.get('schema_version')!r} "
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
    if "bias" not in data or not isinstance(data["bias"], (int, float)):
        raise BoundaryWeightsError("malformed boundary weights: missing or invalid bias")
    try:
        mean = [float(x) for x in data["mean"]]
        std = [float(x) for x in data["std"]]
        weights = [float(x) for x in data["weights"]]
        bias = float(data["bias"])
    except (TypeError, ValueError) as exc:
        raise BoundaryWeightsError("malformed boundary weights: non-numeric coefficients") from exc
    return LogisticBoundaryScorer(
        feature_names=list(FEATURE_NAMES),
        mean=mean,
        std=std,
        weights=weights,
        bias=bias,
    )
