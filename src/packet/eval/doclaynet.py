from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packet.eval.pdfutil import write_text_pdf
from packet.types import LogicalDocument, Packet, PacketOrigin

LABEL_MAP = {
    "title": "title",
    "section-header": "heading",
    "text": "paragraph",
    "list-item": "list_item",
    "table": "table",
    "picture": "figure",
    "caption": "caption",
    "formula": "formula",
    "page-header": "header",
    "page-footer": "footer",
    "footnote": "footer",
}


@dataclass
class DocRecord:
    doc_name: str
    split: str
    category: str
    page_paths: list[Path] = field(default_factory=list)
    page_nos: list[int] = field(default_factory=list)


@dataclass
class SynthSpec:
    name: str
    slice: str
    split: str
    seed: int
    documents: list[str]
    page_ranges: list[tuple[int, int]]


def load_catalog(catalog_path: Path) -> dict[str, list[DocRecord]]:
    """JSONL: split, doc_name, category, page_no, pdf_path."""
    by_split: dict[str, dict[str, DocRecord]] = defaultdict(dict)
    with catalog_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            split = row["split"]
            name = row["doc_name"]
            rec = by_split[split].get(name)
            if rec is None:
                rec = DocRecord(name, split, row.get("category", "unknown"))
                by_split[split][name] = rec
            rec.page_paths.append(Path(row["pdf_path"]))
            rec.page_nos.append(int(row["page_no"]))
    grouped: dict[str, list[DocRecord]] = {}
    for split, docs in by_split.items():
        records = []
        for rec in docs.values():
            order = sorted(zip(rec.page_nos, rec.page_paths))
            rec.page_nos = [n for n, _ in order]
            rec.page_paths = [p for _, p in order]
            records.append(rec)
        grouped[split] = records
    return grouped


def plan_packets(
    records: list[DocRecord],
    *,
    slice_name: str,
    split: str,
    seed: int = 20260922,
    packet_count: int = 8,
    docs_per_packet: int = 2,
    max_pages: int = 40,
) -> list[SynthSpec]:
    import random

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    specs: list[SynthSpec] = []
    i = 0
    n = 0
    while n < packet_count and i < len(pool):
        chosen: list[DocRecord] = []
        pages = 0
        while len(chosen) < docs_per_packet and i < len(pool):
            rec = pool[i]
            i += 1
            if pages + len(rec.page_paths) > max_pages and chosen:
                continue
            if slice_name == "mono-seq" and chosen and rec.category != chosen[0].category:
                continue
            if slice_name == "mono-hard" and chosen and rec.category != chosen[0].category:
                continue
            if slice_name.startswith("poly") and chosen and rec.category == chosen[0].category:
                continue
            chosen.append(rec)
            pages += len(rec.page_paths)
        if len(chosen) < 2:
            continue
        ranges = []
        cursor = 0
        names = []
        for rec in chosen:
            end = cursor + len(rec.page_paths) - 1
            ranges.append((cursor, end))
            names.append(rec.doc_name)
            cursor = end + 1
        specs.append(
            SynthSpec(
                name=f"dlnet-{slice_name}-{n:04d}",
                slice=slice_name,
                split=split,
                seed=seed,
                documents=names,
                page_ranges=ranges,
            )
        )
        n += 1
    return specs


def materialize_placeholder_packet(spec: SynthSpec, dest: Path) -> tuple[Path, Packet]:
    """Stand-in PDF when DocLayNet extras are not cached."""
    dest.mkdir(parents=True, exist_ok=True)
    pages: list[list[str]] = []
    gold_ranges: list[tuple[int, int]] = []
    cursor = 0
    for i, doc_name in enumerate(spec.documents):
        n = spec.page_ranges[i][1] - spec.page_ranges[i][0] + 1
        n = max(n, 1)
        start = cursor
        for p in range(1, n + 1):
            pages.append([doc_name[:80], f"synthetic page {p}", f"Page {p} of {n}"])
            cursor += 1
        gold_ranges.append((start, cursor - 1))
    pdf = write_text_pdf(dest / f"{spec.name}.pdf", pages)
    gold = Packet(
        id=spec.name,
        origin=PacketOrigin(path=str(pdf), page_count=len(pages)),
        documents=[
            LogicalDocument(id=f"{spec.name}-d{i}", title=name, page_start=a, page_end=b)
            for i, (name, (a, b)) in enumerate(zip(spec.documents, gold_ranges))
        ],
    )
    (dest / f"{spec.name}.gold.json").write_text(gold.model_dump_json(indent=2))
    (dest / f"{spec.name}.spec.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "slice": spec.slice,
                "split": spec.split,
                "seed": spec.seed,
                "documents": spec.documents,
                "page_ranges": spec.page_ranges,
                "note": "placeholder-pdf; replace with concatenated DocLayNet extras",
            },
            indent=2,
        )
    )
    return pdf, gold


def write_manifest(path: Path, specs: list[SynthSpec]) -> None:
    payload = [
        {
            "name": s.name,
            "slice": s.slice,
            "split": s.split,
            "seed": s.seed,
            "documents": s.documents,
            "page_ranges": s.page_ranges,
        }
        for s in specs
    ]
    path.write_text(json.dumps(payload, indent=2))
