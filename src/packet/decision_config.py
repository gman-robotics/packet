"""Shared config checks for local decision backends (transformers, mlx_vlm)."""

from __future__ import annotations

from typing import Any

from packet.decision import DecisionBackendError


def config_str(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DecisionBackendError(f"{label} requires non-empty {key!r}")
    return value


def config_bool(value: Any, key: str, label: str) -> bool:
    if not isinstance(value, bool):
        raise DecisionBackendError(f"{label} {key!r} must be a bool")
    return value


def config_images(data: dict[str, Any], label: str) -> bool:
    if "capabilities" not in data:
        return False
    caps = data["capabilities"]
    if not isinstance(caps, dict):
        raise DecisionBackendError(f"{label} capabilities must be an object")
    if "images" not in caps:
        return False
    return config_bool(caps["images"], "capabilities.images", label)


def config_send_images(data: dict[str, Any], label: str) -> bool:
    if "send_images" not in data:
        return False
    return config_bool(data["send_images"], "send_images", label)


def config_positive_int(data: dict[str, Any], key: str, default: int, label: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DecisionBackendError(f"{label} {key!r} must be a positive integer")
    return value


def require_pillow(message: str) -> None:
    try:
        __import__("PIL.Image")
    except ImportError as exc:
        raise DecisionBackendError(message) from exc
