from __future__ import annotations

from dataclasses import dataclass

from packet.citations import block_ids, citation_id_counts
from packet.types import LogicalDocument, Packet, SourceKind


@dataclass
class BoundaryScore:
    precision: float
    recall: float
    f1: float
    gold: list[int]
    predicted: list[int]
    matched: int
    tolerance: int = 0


def document_boundaries(docs: list[LogicalDocument]) -> list[int]:
    """Page index after which a new LogicalDocument starts (EVAL.md §1)."""
    ordered = sorted(docs, key=lambda d: d.page_start)
    return [d.page_end for d in ordered[:-1]]


def _greedy_match(gold: list[int], pred: list[int], tolerance: int) -> int:
    used_pred: set[int] = set()
    matched = 0
    for g in gold:
        best_i = None
        best_d = None
        for i, p in enumerate(pred):
            if i in used_pred:
                continue
            d = abs(p - g)
            if d <= tolerance and (best_d is None or d < best_d):
                best_d = d
                best_i = i
        if best_i is not None:
            used_pred.add(best_i)
            matched += 1
    return matched


def micro_boundary_prf(results: list) -> tuple[float, float, float]:
    """Pool boundary matches across fixtures (matched/pred/gold counts)."""
    matched = sum(r.boundary.matched for r in results)
    pred = sum(len(r.boundary.predicted) for r in results)
    gold = sum(len(r.boundary.gold) for r in results)
    if pred + gold == 0:
        return 1.0, 1.0, 1.0
    p = matched / pred if pred else 0.0
    r = matched / gold if gold else 0.0
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def boundary_score(
    gold: Packet,
    predicted: Packet,
    *,
    tolerance: int = 0,
) -> BoundaryScore:
    g = document_boundaries(gold.documents)
    p = document_boundaries(predicted.documents)
    if not g and not p:
        return BoundaryScore(1.0, 1.0, 1.0, g, p, 0, tolerance)
    matched = _greedy_match(g, p, tolerance)
    precision = matched / len(p) if p else 0.0
    recall = matched / len(g) if g else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return BoundaryScore(precision, recall, f1, g, p, matched, tolerance)


def citation_integrity(packet: Packet, citation_keys: set[str] | None = None) -> float:
    ids: set[str] = set()
    for doc in packet.documents:
        ids.update(block_ids(doc))

    total = 0
    unresolved = 0
    for doc in packet.documents:
        doc_total, missing = citation_id_counts(doc, ids, citation_keys)
        total += doc_total
        unresolved += len(missing)
    if total == 0:
        return 1.0
    return (total - unresolved) / total


def false_ocr_hard_fail(packet: Packet, pages_with_text: set[int]) -> list[int]:
    """Pages that had a trustworthy text layer but emitted OCR/VLM blocks."""
    fails: list[int] = []
    for doc in packet.documents:
        for section in doc.sections:
            _scan_ocr(section, pages_with_text, fails)
        for block in doc.furniture:
            if block.page in pages_with_text and block.source in {SourceKind.OCR, SourceKind.VLM}:
                fails.append(block.page)
    return sorted(set(fails))


def _scan_ocr(section, pages_with_text: set[int], fails: list[int]) -> None:
    for block in section.blocks:
        if block.page in pages_with_text and block.source in {SourceKind.OCR, SourceKind.VLM}:
            fails.append(block.page)
    for child in section.children:
        _scan_ocr(child, pages_with_text, fails)
