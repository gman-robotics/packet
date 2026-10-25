"""GMA-274: opt-in PoliTax eval that reads cached GMA-275 OCR text (no new OCR).

``packet eval --set politax --ocr-cache`` routes scan pages to the cached LightOn
grounding text under ``<data-dir>/ocr/<pdf_sha256>/pNNNNN.json`` instead of an OCR
engine. Default eval never imports this module and never reads the cache.
A page with no valid record yields no blocks (same as OCR unavailable) and is
counted in ``missing``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from packet.backends import EscalatingExtractor
from packet.eval.politax_ocr import SCHEMA, record_path
from packet.extract import NativeTextExtractor
from packet.ocr_parse import parse_lighton_grounding
from packet.pipeline import Pipeline, PipelineConfig
from packet.types import Block, PageClass, PageRef

_OCR_CLASSES = {PageClass.SCAN, PageClass.MIXED, PageClass.SUSPECT_ENCODING}


@dataclass
class CacheStats:
    used: int = 0
    missing: int = 0
    truncated: int = 0
    engines: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {"pages_used": self.used, "pages_missing": self.missing,
                "pages_truncated": self.truncated,
                "engines": dict(sorted(self.engines.items()))}


class CachedOcrExtractor:
    """Reads one PDF's cached OCR records; never loads an engine or renders a page."""

    needs_image = False

    def __init__(self, data_dir: Path, pdf_sha256: str, stats: CacheStats):
        self.data_dir = Path(data_dir)
        self.pdf_sha256 = pdf_sha256
        self.stats = stats

    def supports(self, page_class: PageClass) -> bool:
        return page_class in _OCR_CLASSES

    def _record(self, index: int) -> dict | None:
        try:
            rec = json.loads(record_path(self.data_dir, self.pdf_sha256, index).read_text())
        except (OSError, ValueError):
            return None
        if not (isinstance(rec, dict) and rec.get("schema") == SCHEMA
                and rec.get("pdf_sha256") == self.pdf_sha256 and rec.get("page") == index
                and isinstance(rec.get("raw"), str)):
            return None
        return rec

    def extract(self, ref: PageRef, text: str, image=None) -> list[Block]:
        rec = self._record(ref.index)
        if rec is None:
            self.stats.missing += 1
            return []
        self.stats.used += 1
        self.stats.truncated += int(rec.get("truncated") is True)
        eng = rec.get("engine") or {}
        self.stats.engines[f"{eng.get('name')}:{eng.get('model')}@{eng.get('revision')}"] += 1
        return parse_lighton_grounding(rec["raw"], ref)


def ocr_cache_pipeline(data_dir: Path, pdf_sha256: str, stats: CacheStats,
                       config: PipelineConfig | None = None) -> Pipeline:
    ocr = CachedOcrExtractor(data_dir, pdf_sha256, stats)
    return Pipeline(extractors=[EscalatingExtractor(NativeTextExtractor(), ocr)], config=config)


def run_with_ocr_cache(m, fixtures, data_dir: Path, config: PipelineConfig | None = None):
    """One eval row per fixture, scan pages fed from the OCR cache. Returns (results, stats)."""
    from packet.eval.run import run_fixture

    sha_by_file = {d.file_name: d.sha256 for d in m.docs}
    stats = CacheStats()
    results = []
    for fx in fixtures:
        sha = sha_by_file[fx.notes or fx.pdf.name]
        results.append(run_fixture(fx, ocr_cache_pipeline(data_dir, sha, stats, config)))
    return results, stats
