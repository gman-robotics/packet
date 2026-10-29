"""Citation ids named from a document's ``extras`` (core, domain-free).

``citation_integrity`` is ``ok / total`` over the whole packet. The known block
ids are gathered from every document, and an empty ``citation_keys`` scans every
extras value. Repeats count. ``unresolved_citation_ids`` is the second element of
``citation_id_counts``.
"""

from __future__ import annotations

from packet.types import LogicalDocument, Section


def block_ids(doc: LogicalDocument) -> set[str]:
    """Block ids in this document's sections and furniture."""
    found: set[str] = set()
    for section in doc.sections:
        _collect_block_ids(section, found)
    found.update(block.id for block in doc.furniture)
    return found


def citation_id_counts(
    doc: LogicalDocument,
    known_ids: set[str],
    citation_keys: set[str] | None = None,
) -> tuple[int, list[str]]:
    """Return ``(total, unresolved)`` for citation ids in ``doc.extras``.

    ``total`` counts every cited id, including repeats. ``unresolved`` keeps
    repeats and lists only ids absent from ``known_ids``. ``citation_keys``
    limits which extras keys are citations; ``None`` or an empty set scans
    every value, matching ``citation_integrity``.
    """
    keys = citation_keys or set()
    total = 0
    unresolved: list[str] = []
    for extra_key, value in doc.extras.items():
        if keys and extra_key not in keys:
            continue
        cited = _as_citation_ids(value)
        if not cited:
            continue
        for cid in cited:
            total += 1
            if cid not in known_ids:
                unresolved.append(cid)
    return total, unresolved


def unresolved_citation_ids(
    doc: LogicalDocument,
    known_ids: set[str],
    citation_keys: set[str] | None = None,
) -> list[str]:
    """Citation ids in ``doc`` that are not in ``known_ids`` (repeats kept)."""
    return citation_id_counts(doc, known_ids, citation_keys)[1]


def _collect_block_ids(section: Section, ids: set[str]) -> None:
    ids.update(block.id for block in section.blocks)
    for child in section.children:
        _collect_block_ids(child, ids)


def _as_citation_ids(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []
