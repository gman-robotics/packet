"""Whole-document joined DocLayNet packets (``packet-dlnet-whole``).

DocLayNet pages are samples, so joining them never gives a real packet: page
numbers never restart. This set resolves DocLayNet document names to the whole
public PDF (pinned URL, page count and sha256 in ``evals/dlnet_whole.json``),
downloads them into a local cache at eval time, and joins whole documents into
packets with a seeded plan. Nothing third-party is committed.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from packet.eval.fixtures import NamedFixture
from packet.types import LogicalDocument, Packet, PacketOrigin

MANIFEST = Path(__file__).resolve().parents[3] / "evals" / "dlnet_whole.json"


class FetchError(RuntimeError):
    pass


@dataclass
class WholeDoc:
    name: str
    url: str
    file: str
    pages: int
    license: str
    sha256: str


@dataclass
class Manifest:
    name: str
    version: int
    seed: int
    docs_per_packet: tuple[int, int]
    license_note: str
    docs: list[WholeDoc]


def load_manifest(path: Path | None = None) -> Manifest:
    raw = json.loads((path or MANIFEST).read_text())
    docs = [WholeDoc(d["doclaynet_name"], d["url"], d["file"], d["pages"], d["license"], d["sha256"])
            for d in raw["docs"]]
    lo, hi = raw["docs_per_packet"]
    return Manifest(raw["name"], raw["version"], raw["seed"], (lo, hi), raw["license_note"], docs)


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "packet-eval/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch(m: Manifest, cache: Path, opener: Callable[[str], bytes] = _download) -> list[Path]:
    """Download missing docs into ``cache``; every file (cached or new) must match its sha256."""
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for d in m.docs:
        path = cache / d.file
        if not (path.exists() and _sha(path.read_bytes()) == d.sha256):
            data = opener(d.url)
            if _sha(data) != d.sha256:
                raise FetchError(f"{d.name}: sha256 mismatch for {d.url}")
            path.write_bytes(data)
        paths.append(path)
    return paths


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan(m: Manifest) -> list[list[int]]:
    """Seeded partition of all docs into packets of ``docs_per_packet`` docs (each doc once)."""
    rng = random.Random(m.seed)
    order = list(range(len(m.docs)))
    rng.shuffle(order)
    lo, hi = m.docs_per_packet
    packets: list[list[int]] = []
    while order:
        k = rng.randint(lo, hi)
        if len(order) - k < lo:
            k = len(order)  # never leave a packet with fewer than ``lo`` docs
        packets.append(order[:k])
        order = order[k:]
    return packets


def build_fixtures(m: Manifest, cache: Path, out: Path) -> list[NamedFixture]:
    """Join cached docs into packet PDFs under ``out`` with gold document ranges."""
    out.mkdir(parents=True, exist_ok=True)
    fixtures: list[NamedFixture] = []
    for n, idxs in enumerate(plan(m)):
        name = f"dlw-{n:02d}"
        pdf_path = out / f"{name}.pdf"
        joined = pdfium.PdfDocument.new()
        docs: list[LogicalDocument] = []
        start = 0
        for i in idxs:
            d = m.docs[i]
            src = pdfium.PdfDocument(str(cache / d.file))
            if len(src) != d.pages:
                raise FetchError(f"{d.name}: {len(src)} pages, manifest says {d.pages}")
            joined.import_pages(src)
            docs.append(LogicalDocument(id=f"{name}-d{len(docs)}", page_start=start,
                                        page_end=start + d.pages - 1, title=d.name))
            start += d.pages
        joined.save(str(pdf_path))
        gold = Packet(id=name, origin=PacketOrigin(path=str(pdf_path), page_count=start),
                      documents=docs)
        fixtures.append(NamedFixture(name, pdf_path, gold, "born-digital",
                                     "+".join(m.docs[i].name for i in idxs)))
    return fixtures
