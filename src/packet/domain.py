from __future__ import annotations

from typing import Protocol

from packet.types import LogicalDocument


class DomainAdapter(Protocol):
    """Optional enrichment after a logical document is closed."""

    name: str

    def applies(self, doc: LogicalDocument) -> bool:
        ...

    def enrich(self, doc: LogicalDocument) -> LogicalDocument:
        ...


class NoOpDomain:
    name = "noop"

    def applies(self, doc: LogicalDocument) -> bool:
        return False

    def enrich(self, doc: LogicalDocument) -> LogicalDocument:
        return doc


def apply_domains(
    doc: LogicalDocument, adapters: list[DomainAdapter]
) -> LogicalDocument:
    for adapter in adapters:
        if adapter.applies(doc):
            doc = adapter.enrich(doc)
    return doc
