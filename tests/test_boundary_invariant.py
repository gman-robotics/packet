"""GMA-492: one boundary observation and one decision outcome per page, including page 0.

``BoundaryDetector.observe`` returns before the decision backend on page 0. A dump that
records only ``decide()`` calls must not treat later fact rows as if page 0 never existed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from packet.boundary import BoundaryDetector
from packet.decision import DecisionCapabilities, DecisionRequest, DecisionResult
from packet.eval.pdfutil import write_text_pdf
from packet.events import EventKind
from packet.pipeline import Pipeline, PipelineConfig

# First token on each page is PAGEk so fact row k can be told from its neighbors.
_PAGES: dict[str, list[list[str]]] = {
    "one-page": [
        ["PAGE0 only document body.", "Page 1 of 1"],
    ],
    "one-document": [
        ["PAGE0 alpha report opens.", "Page 1 of 3"],
        ["PAGE1 alpha report continues.", "Page 2 of 3"],
        ["PAGE2 alpha report closes.", "Page 3 of 3"],
    ],
    "unnumbered-cover": [
        ["PAGE0 fabrikam service agreement cover."],
        ["PAGE1 scope of services described.", "Page 2 of 3"],
        ["PAGE2 term of the agreement.", "Page 3 of 3"],
    ],
    "two-document": [
        ["PAGE0 northwind invoice amount.", "Page 1 of 2"],
        ["PAGE1 northwind invoice thanks.", "Page 2 of 2"],
        ["PAGE2 contoso lease premises.", "Page 1 of 2"],
        ["PAGE3 contoso lease signatures.", "Page 2 of 2"],
    ],
    "three-document": [
        ["PAGE0 invoice stands alone.", "Page 1 of 1"],
        ["PAGE1 lease body starts.", "Page 1 of 2"],
        ["PAGE2 lease body ends.", "Page 2 of 2"],
        ["PAGE3 cover letter opens.", "Page 1 of 1"],
    ],
}


@dataclass
class _RecordingBackend:
    """Decision backend that records ``decide`` requests and leaves the heuristic score."""

    calls: list[DecisionRequest] = field(default_factory=list)
    capabilities: DecisionCapabilities = field(default_factory=DecisionCapabilities)
    degraded_pages: list[int] = field(default_factory=list)

    @property
    def backend_label(self) -> str:
        return "recording"

    def reset_parse_budget(self) -> None:
        self.degraded_pages.clear()

    def decide(
        self,
        request: DecisionRequest,
        *,
        render: Callable[[], Any] | None = None,
    ) -> DecisionResult:
        del render
        self.calls.append(request)
        return DecisionResult(probability=None)


def _run(tmp_path, monkeypatch, name: str, pages: list[list[str]]):
    observed: list[int] = []
    real = BoundaryDetector.observe

    def spy(self, ref, blocks, *args, **kwargs):
        observed.append(ref.index)
        return real(self, ref, blocks, *args, **kwargs)

    monkeypatch.setattr(BoundaryDetector, "observe", spy)
    backend = _RecordingBackend()
    events: list = []
    packet = Pipeline(config=PipelineConfig(decision_backend=backend)).parse(
        write_text_pdf(tmp_path / f"{name}.pdf", pages),
        on_event=events.append,
    )
    return packet, events, observed, backend


@pytest.mark.parametrize(("name", "pages"), list(_PAGES.items()), ids=list(_PAGES))
def test_observe_called_once_per_page_in_order(tmp_path, monkeypatch, name, pages):
    _packet, _events, observed, _backend = _run(tmp_path, monkeypatch, name, pages)
    assert observed == list(range(len(pages)))


@pytest.mark.parametrize(("name", "pages"), list(_PAGES.items()), ids=list(_PAGES))
def test_decide_skips_page_zero_and_prefixes_its_facts(tmp_path, monkeypatch, name, pages):
    _packet, _events, _observed, backend = _run(tmp_path, monkeypatch, name, pages)
    n = len(pages)
    assert [request.page_index for request in backend.calls] == list(range(1, n))
    for request in backend.calls:
        assert len(request.facts) == request.page_index + 1
        for index in range(request.page_index + 1):
            assert f"PAGE{index}" in request.facts[index]["text"]


@pytest.mark.parametrize(("name", "pages"), list(_PAGES.items()), ids=list(_PAGES))
def test_each_page_has_one_contiguous_outcome(tmp_path, monkeypatch, name, pages):
    packet, events, _observed, _backend = _run(tmp_path, monkeypatch, name, pages)
    n = len(pages)
    assert packet.origin.page_count == n
    assert packet.documents
    assert packet.documents[0].page_start == 0

    covered: list[int] = []
    previous_end = -1
    for doc in packet.documents:
        assert doc.page_start == previous_end + 1
        assert doc.page_end >= doc.page_start
        covered.extend(range(doc.page_start, doc.page_end + 1))
        previous_end = doc.page_end
    assert covered == list(range(n))

    decided = [
        event.payload["page"]
        for event in events
        if event.kind == EventKind.BOUNDARY_DECIDED
    ]
    assert decided == list(range(n))
    starts = {doc.page_start for doc in packet.documents}
    splits = {
        event.payload["page"]
        for event in events
        if event.kind == EventKind.BOUNDARY_DECIDED and event.payload["split"]
    }
    assert starts == {0} | splits

    proposed = [
        event.payload["after_page"]
        for event in events
        if event.kind == EventKind.BOUNDARY_PROPOSED
    ]
    assert all(after_page >= 0 for after_page in proposed)
    assert len(proposed) == len(set(proposed))
