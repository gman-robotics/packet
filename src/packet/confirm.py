"""Mid-band boundary confirmer (GMA-462). Off unless a caller passes one.

``observe_page`` stores post-classify facts. ``confirm`` asks the decision
backend only for an in-band page after page 0. Page 0 stays on the eager rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packet.boundary_model import page_fact_row
from packet.decision import (
    DecisionBackendError,
    DecisionBudgetExhausted,
    DecisionRequest,
    build_decision_backend,
    read_config_object,
)

_DEFAULT_LOW = 0.35
_DEFAULT_HIGH = 0.74


def _split_verdict(probability: float | None) -> bool | None:
    if probability is None:
        return None
    return probability >= 0.5


@dataclass
class BackendConfirmer:
    backend: Any
    low: float
    high: float
    task: str = "confirm_boundary"
    window: int = 2
    disabled: bool = False
    degraded_pages: list[int] = field(default_factory=list)
    _facts: list[dict[str, Any]] = field(default_factory=list)
    _stored: dict[int, tuple[Any, list[Any], Any]] = field(default_factory=dict)

    def observe_page(
        self,
        classified: Any,
        blocks: list[Any],
        render: Any,
        page_number: int | None,
        page_total: int | None,
    ) -> None:
        if classified.index == 0:
            self._reset_file()
        self._append_fact(classified, blocks, page_number, page_total)
        self._stored[classified.index] = (classified, blocks, render)
        self._drop_stale(classified.index)

    def confirm(self, before, page, after) -> bool | None:
        try:
            return self._verdict(page, after)
        finally:
            self.release(page.index)

    def release(self, index: int) -> None:
        self._stored.pop(index, None)

    def _reset_file(self) -> None:
        self.disabled = False
        self.degraded_pages.clear()
        self._facts.clear()
        self._stored.clear()
        self.backend.reset_parse_budget()

    def _append_fact(
        self,
        classified: Any,
        blocks: list[Any],
        page_number: int | None,
        page_total: int | None,
    ) -> None:
        if classified.index != len(self._facts):
            raise DecisionBackendError(
                f"confirm observer saw page {classified.index} after {len(self._facts)} rows"
            )
        self._facts.append(
            page_fact_row(
                classified, blocks, page_number=page_number, page_total=page_total
            )
        )

    def _drop_stale(self, current: int) -> None:
        cutoff = current - self.window - 1
        stale = [index for index in self._stored if index < cutoff]
        for index in stale:
            del self._stored[index]

    def _verdict(self, page, after) -> bool | None:
        if page.index == 0 or self.disabled or not self._in_band(page):
            return None
        if len(self._facts) <= page.index:
            return None
        return self._call(page, after)

    def _in_band(self, page) -> bool:
        signal = page.signal
        if signal is None:
            return False
        confidence = signal.confidence
        if confidence is None:
            return False
        return self.low <= confidence <= self.high

    def _fact_end(self, page, after) -> int:
        end = page.index + 1
        if after:
            end = max(end, after[-1].index + 1)
        return end

    def _stored_page(self, index: int) -> tuple[Any, list[Any], Any]:
        found = self._stored.get(index)
        if found is None:
            raise DecisionBackendError(f"confirm lost page {index} before decide")
        return found

    def _call(self, page, after) -> bool | None:
        classified, blocks, render = self._stored_page(page.index)
        request = DecisionRequest(
            task=self.task,
            page_index=page.index,
            facts=self._facts[0 : self._fact_end(page, after)],
            blocks=blocks,
            ref=classified,
            input_mode="default",
        )
        try:
            result = self.backend.decide(request, render=render)
        except DecisionBudgetExhausted:
            self._disable(page.index)
            return None
        return _split_verdict(result.probability)

    def _disable(self, index: int) -> None:
        self.degraded_pages.append(index)
        self.disabled = True


def _band_bound(data: dict[str, Any], key: str, default: float) -> float:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionBackendError(f"confirm config {key!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DecisionBackendError(f"confirm config {key!r} must be a finite number")
    return number


def _task_name(data: dict[str, Any]) -> str:
    task = data.get("task", "confirm_boundary")
    if not isinstance(task, str) or not task.strip():
        raise DecisionBackendError("confirm config task must be a non-empty string")
    return task


def _window_value(data: dict[str, Any]) -> int:
    if "window" not in data:
        return 2
    value = data["window"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecisionBackendError(
            "confirm config 'window' must be a non-negative integer"
        )
    return value


def _decision_object(data: dict[str, Any]) -> dict[str, Any]:
    decision = data.get("decision")
    if not isinstance(decision, dict):
        raise DecisionBackendError("confirm config decision must be an object")
    return decision


def confirmer_from_object(data: dict[str, Any], *, base_dir: Path) -> BackendConfirmer:
    low = _band_bound(data, "low", _DEFAULT_LOW)
    high = _band_bound(data, "high", _DEFAULT_HIGH)
    if low > high:
        raise DecisionBackendError("confirm config low must be <= high")
    backend = build_decision_backend(_decision_object(data), base_dir=base_dir)
    return BackendConfirmer(
        backend=backend,
        low=low,
        high=high,
        task=_task_name(data),
        window=_window_value(data),
    )


def load_confirm_config(path: Path) -> BackendConfirmer:
    return confirmer_from_object(
        read_config_object(path, "confirm config"), base_dir=path.parent
    )
