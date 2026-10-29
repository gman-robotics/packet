from __future__ import annotations

import copy
from typing import Protocol

from packet.types import PACKET_FLAGS_KEY, LogicalDocument


class DomainAdapter(Protocol):
    """Optional enrichment after a logical document is closed.

    Set ``type_hint`` and add ``extras``. Do not overwrite
    ``extras["packet.flags"]``. Core owns that key. ``apply_domains`` puts
    an existing value back and drops one an adapter inserts.
    """

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
    """Run adapters. Core owns ``packet.flags``; an adapter cannot change it.

    The value present before this call is put back afterward. A value an
    adapter inserts when the key was absent is removed. Other extras stay
    as the adapters left them.
    """
    owned = (
        copy.deepcopy(doc.extras[PACKET_FLAGS_KEY])
        if PACKET_FLAGS_KEY in doc.extras
        else None
    )
    # Hand adapters a copy so an in-place edit cannot change ``owned`` or the
    # list the caller still holds.
    if owned is not None:
        doc.extras[PACKET_FLAGS_KEY] = copy.deepcopy(owned)
    for adapter in adapters:
        if adapter.applies(doc):
            doc = adapter.enrich(doc)
    if owned is None:
        doc.extras.pop(PACKET_FLAGS_KEY, None)
    else:
        doc.extras[PACKET_FLAGS_KEY] = owned
    return doc
