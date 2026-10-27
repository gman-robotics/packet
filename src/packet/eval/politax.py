"""PoliTax-Split corpus (``packet-politax``).

Extend-AI/PoliTax-Split on Hugging Face: 30 whole PDF packets with CC0 sub-document
labels. PDFs and ``annotations.jsonl`` are pinned by HTTPS URL and sha256, downloaded
into a local cache at eval time, and never committed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from packet.eval.fixtures import NamedFixture
from packet.types import LogicalDocument, Packet, PacketOrigin

MANIFEST = Path(__file__).resolve().parents[3] / "evals" / "politax.json"
POLITAX_FIXTURE_CONDITION = "pub-tax-return"


class FetchError(RuntimeError):
    pass


class GoldAdapterError(ValueError):
    pass


@dataclass
class AnnotationsRef:
    url: str
    file: str
    sha256: str
    license: str


@dataclass
class PoliTaxDoc:
    subject: str
    file_name: str
    file: str
    url: str
    pages: int
    sha256: str
    license: str
    gold_cuts: int


@dataclass
class Manifest:
    name: str
    version: int
    license_note: str
    annotations: AnnotationsRef
    docs: list[PoliTaxDoc]


@dataclass
class SubDocument:
    start_page: int
    end_page: int
    identifier: str = ""
    type: str = ""


@dataclass
class FileAnnotation:
    file_name: str
    page_count: int
    sub_documents: list[SubDocument]


def load_manifest(path: Path | None = None) -> Manifest:
    raw = json.loads((path or MANIFEST).read_text())
    ann = raw["annotations"]
    docs = [
        PoliTaxDoc(
            d["subject"],
            d["file_name"],
            d["file"],
            d["url"],
            d["pages"],
            d["sha256"],
            d["license"],
            int(d.get("gold_cuts", 0)),
        )
        for d in raw["docs"]
    ]
    return Manifest(
        raw["name"],
        raw["version"],
        raw["license_note"],
        AnnotationsRef(ann["url"], ann["file"], ann["sha256"], ann["license"]),
        docs,
    )


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "packet-eval/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_verified(path: Path, data: bytes, expected: str, label: str) -> None:
    if _sha(data) != expected:
        raise FetchError(f"{label}: sha256 mismatch (re-pin the manifest)")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch(
    m: Manifest,
    cache: Path,
    opener: Callable[[str], bytes] = _download,
    pause: float = 0.5,
) -> Path:
    """Download PDFs and annotations into ``cache``; verify every pinned sha256."""
    cache.mkdir(parents=True, exist_ok=True)
    ann_path = cache / m.annotations.file
    if not (ann_path.exists() and _sha(ann_path.read_bytes()) == m.annotations.sha256):
        data = opener(m.annotations.url)
        _write_verified(ann_path, data, m.annotations.sha256, m.annotations.file)
    downloaded = 0
    for d in m.docs:
        path = cache / d.file
        if path.exists() and _sha(path.read_bytes()) == d.sha256:
            continue
        if downloaded and pause:
            time.sleep(pause)
        data = opener(d.url)
        downloaded += 1
        if _sha(data) != d.sha256:
            raise FetchError(f"{d.file_name}: sha256 mismatch for {d.url} (re-pin the manifest)")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return ann_path


def load_annotations(path: Path) -> dict[str, FileAnnotation]:
    out: dict[str, FileAnnotation] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        subs = [
            SubDocument(s["start_page"], s["end_page"], s.get("identifier", ""), s.get("type", ""))
            for s in raw["sub_documents"]
        ]
        out[raw["file_name"]] = FileAnnotation(raw["file_name"], raw["page_count"], subs)
    return out


def validate_sub_documents(subs: list[SubDocument], page_count: int) -> None:
    """Reject overlap, gap, or ranges that do not cover ``1..page_count`` (1-indexed)."""
    if not subs:
        raise GoldAdapterError("no sub_documents")
    ordered = sorted(subs, key=lambda s: s.start_page)
    if ordered[0].start_page != 1:
        raise GoldAdapterError(f"first sub-document starts at {ordered[0].start_page}, expected 1")
    if ordered[-1].end_page != page_count:
        raise GoldAdapterError(
            f"last sub-document ends at {ordered[-1].end_page}, expected {page_count}"
        )
    for s in ordered:
        if s.start_page > s.end_page:
            raise GoldAdapterError(f"inverted range {s.start_page}..{s.end_page}")
        if s.start_page < 1 or s.end_page > page_count:
            raise GoldAdapterError(f"range {s.start_page}..{s.end_page} outside 1..{page_count}")
    for prev, nxt in itertools.pairwise(ordered):
        if nxt.start_page <= prev.end_page:
            raise GoldAdapterError(
                f"overlap: {prev.start_page}..{prev.end_page} and {nxt.start_page}..{nxt.end_page}"
            )
        if nxt.start_page != prev.end_page + 1:
            raise GoldAdapterError(
                f"gap: {prev.end_page} then {nxt.start_page} (expected {prev.end_page + 1})"
            )


def logical_documents_from_gold(
    file_name: str, ann: FileAnnotation, *, fixture_id: str
) -> list[LogicalDocument]:
    validate_sub_documents(ann.sub_documents, ann.page_count)
    docs: list[LogicalDocument] = []
    for i, sub in enumerate(ann.sub_documents):
        docs.append(
            LogicalDocument(
                id=f"{fixture_id}-d{i}",
                title=sub.identifier or sub.type or f"doc-{i}",
                page_start=sub.start_page - 1,
                page_end=sub.end_page - 1,
            )
        )
    return docs


def _fixture_name(file_name: str) -> str:
    return Path(file_name).stem.replace(" ", "_")


def build_fixtures(m: Manifest, cache: Path, annotations: dict[str, FileAnnotation]) -> list[NamedFixture]:
    fixtures: list[NamedFixture] = []
    for d in m.docs:
        ann = annotations.get(d.file_name)
        if ann is None:
            raise GoldAdapterError(f"missing annotations for {d.file_name}")
        if ann.page_count != d.pages:
            raise GoldAdapterError(f"{d.file_name}: manifest pages {d.pages}, labels {ann.page_count}")
        pdf_path = cache / d.file
        name = _fixture_name(d.file_name)
        docs = logical_documents_from_gold(d.file_name, ann, fixture_id=name)
        gold = Packet(
            id=name,
            origin=PacketOrigin(path=str(pdf_path), page_count=d.pages),
            documents=docs,
        )
        fixtures.append(NamedFixture(name, pdf_path, gold, POLITAX_FIXTURE_CONDITION, notes=d.file_name))
    return fixtures


def corpus_gold_label_totals(m: Manifest) -> tuple[int, int]:
    """Return ``(gold_boundaries, sub_documents)`` summed across all manifest PDFs.

    Manifest ``gold_cuts`` counts sub-documents per whole PDF; internal boundaries are
    ``sub_documents - packet_count`` (one packet per PDF).
    """
    sub_documents = sum(d.gold_cuts for d in m.docs)
    boundaries = sub_documents - len(m.docs)
    return boundaries, sub_documents


def format_gold_label_totals(boundaries: int, sub_documents: int) -> str:
    return f"gold labels: {boundaries} boundaries, {sub_documents} sub-documents"


def subject_totals_from_manifest(m: Manifest, manifest_path: Path | None = None) -> dict[str, dict[str, int]]:
    path = manifest_path or MANIFEST
    raw_docs = {d["file_name"]: d for d in json.loads(path.read_text())["docs"]}
    totals: dict[str, dict[str, int]] = {}
    for d in m.docs:
        bucket = totals.setdefault(d.subject, {"pages": 0, "gold_cuts": 0, "pdfs": 0})
        bucket["pages"] += d.pages
        bucket["pdfs"] += 1
        bucket["gold_cuts"] += int(raw_docs[d.file_name].get("gold_cuts", 0))
    return totals


def text_layer_census(
    fixtures: Iterable[NamedFixture], *, subjects: dict[str, str], min_chars: int = 40
) -> list[dict]:
    from packet.eval.run import text_layer_pages

    rows: list[dict] = []
    for fx in fixtures:
        pages = fx.gold.origin.page_count
        text_pages = len(text_layer_pages(fx.pdf, min_chars=min_chars))
        file_name = fx.notes or fx.pdf.name
        rows.append(
            {
                "file_name": file_name,
                "subject": subjects.get(file_name, ""),
                "pages": pages,
                "text_layer_pages": text_pages,
            }
        )
    return rows


def format_census(rows: list[dict]) -> str:
    lines = ["text-layer census (native chars >= 40):"]
    total_pages = total_text = 0
    for row in rows:
        total_pages += row["pages"]
        total_text += row["text_layer_pages"]
        lines.append(
            f"  {row['file_name']}: {row['pages']} pages, {row['text_layer_pages']} text-layer"
        )
    lines.append(f"  total: {total_pages} pages, {total_text} text-layer")
    return "\n".join(lines)
