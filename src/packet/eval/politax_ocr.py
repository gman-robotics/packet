"""Optional OCR text cache for PoliTax scan pages (GMA-275).

Outside the eval gate and outside CI. Pages whose stripped native text is under
``OCR_MIN_NATIVE_CHARS`` (the census threshold) get one JSON record each:

    <data-dir>/ocr/<pdf_sha256>/p00000.json   # 0-based page index, as PdfPageSource
    <data-dir>/ocr/_runs/<utc>-<pid>.json     # one log per invocation

Records hold the raw engine output only. This module never imports an OCR engine;
the recognizer arrives as a callable from ``packet politax-ocr``. Nothing else in
``packet`` imports this module (guarded by tests/test_politax_ocr_guard.py).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

OCR_MIN_NATIVE_CHARS = 40
SCHEMA = 1
OCR_DIR = "ocr"
RUNS_DIR = "_runs"

_replace = os.replace  # tests inject a failure between the temp write and the replace


@dataclass(frozen=True)
class EngineId:
    """Strict skip-rule identity: any field change makes cached pages stale."""

    name: str
    model: str
    revision: str
    mode: str
    max_tokens: int
    render_px: int


@dataclass(frozen=True)
class PagePlan:
    pdf_sha256: str
    file_name: str
    file: str
    index: int
    native_chars: int


@dataclass
class RunSummary:
    started: str
    ended: str = ""
    wall_s: float = 0.0
    gen_s: float = 0.0
    planned: int = 0
    ocr_done: int = 0
    redone: int = 0
    skipped_cached: int = 0
    truncated: int = 0
    interrupted: bool = False
    engine: dict = field(default_factory=dict)


Recognize = Callable[[PagePlan], tuple[str, int | None]]


def native_chars(text: str | None) -> int:
    """Same measure as ``text_layer_pages``: stripped native text length."""
    return len((text or "").strip())


def plan_pages(docs: Iterable, data_dir: Path, *, subject: str | None = None,
               file: str | None = None) -> tuple[list[PagePlan], int]:
    """Pages that need OCR, plus the count of pages skipped for native text.

    ``docs`` are manifest ``PoliTaxDoc`` rows; the key is the manifest sha256, which
    ``politax.fetch`` has verified for the file on disk."""
    from packet.source import PdfPageSource

    plans: list[PagePlan] = []
    native = 0
    for d in docs:
        if subject and d.subject != subject:
            continue
        if file and file not in (d.file, d.file_name):
            continue
        with PdfPageSource(Path(data_dir) / d.file) as src:
            for i in range(src.page_count):
                n = native_chars(src.text_for(i))
                if n >= OCR_MIN_NATIVE_CHARS:
                    native += 1
                    continue
                plans.append(PagePlan(d.sha256, d.file_name, d.file, i, n))
    return plans, native


def record_path(data_dir: Path, pdf_sha256: str, index: int) -> Path:
    return Path(data_dir) / OCR_DIR / pdf_sha256 / f"p{index:05d}.json"


def _state(plan: PagePlan, data_dir: Path, engine: EngineId) -> str:
    path = record_path(data_dir, plan.pdf_sha256, plan.index)
    if not path.exists():
        return "pending"
    try:
        rec = json.loads(path.read_text())
        ok = (rec["schema"] == SCHEMA and rec["pdf_sha256"] == plan.pdf_sha256
              and rec["page"] == plan.index and isinstance(rec["raw"], str))
    except (OSError, ValueError, KeyError, TypeError):
        return "corrupt"
    if not ok:
        return "corrupt"
    return "cached" if rec.get("engine") == asdict(engine) else "stale"


def classify(plans: Iterable[PagePlan], data_dir: Path, engine: EngineId) -> Counter:
    out: Counter = Counter({"cached": 0, "stale": 0, "corrupt": 0, "pending": 0})
    for p in plans:
        out[_state(p, data_dir, engine)] += 1
    return out


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def _clean_temp(data_dir: Path) -> None:
    root = Path(data_dir) / OCR_DIR
    if root.is_dir():
        for tmp in root.glob("*/.*.tmp"):
            tmp.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(plans: list[PagePlan], recognize: Recognize, data_dir: Path, engine: EngineId, *,
        limit: int | None = None, clock: Callable[[], float] = time.monotonic,
        versions: dict[str, str] | None = None) -> RunSummary:
    """OCR every pending, stale or corrupt page; skip current records. Single writer.

    ``limit`` caps ``recognize`` calls. A log is written even on Ctrl-C or an engine
    error; the page in flight is the most that is lost."""
    data_dir = Path(data_dir)
    _clean_temp(data_dir)
    summary = RunSummary(started=_now(), planned=len(plans), engine=asdict(engine))
    t_start = clock()
    try:
        for plan in plans:
            state = _state(plan, data_dir, engine)
            if state == "cached":
                summary.skipped_cached += 1
                continue
            if limit is not None and summary.ocr_done + summary.redone >= limit:
                break
            t0 = clock()
            raw, tokens = recognize(plan)
            secs = clock() - t0
            truncated = tokens is not None and tokens >= engine.max_tokens
            _atomic_write_json(record_path(data_dir, plan.pdf_sha256, plan.index), {
                "schema": SCHEMA,
                "pdf_sha256": plan.pdf_sha256,
                "file_name": plan.file_name,
                "page": plan.index,
                "native_chars": plan.native_chars,
                "engine": asdict(engine),
                "versions": dict(versions or {}),
                "raw": raw,
                "seconds": round(secs, 3),
                "generation_tokens": tokens,
                "truncated": truncated,
                "created": _now(),
            })
            summary.gen_s += secs
            summary.truncated += int(truncated)
            if state == "pending":
                summary.ocr_done += 1
            else:
                summary.redone += 1
    except BaseException:
        summary.interrupted = True
        raise
    finally:
        summary.wall_s = round(clock() - t_start, 3)
        summary.gen_s = round(summary.gen_s, 3)
        summary.ended = _now()
        log = data_dir / OCR_DIR / RUNS_DIR / f"{summary.started.replace(':', '')}-{os.getpid()}.json"
        if log.exists():
            log = log.with_name(f"{log.stem}-{time.monotonic_ns()}.json")
        _atomic_write_json(log, asdict(summary))
    return summary


def report(data_dir: Path, engine: EngineId) -> dict:
    """Read-only totals for the PR: records on disk plus runs that called the engine."""
    root = Path(data_dir) / OCR_DIR
    pages = truncated = current = 0
    pdfs = set()
    versions: set[str] = set()
    if root.is_dir():
        for path in root.glob("*/p*.json"):
            try:
                rec = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            pages += 1
            pdfs.add(rec.get("pdf_sha256"))
            truncated += int(bool(rec.get("truncated")))
            current += int(rec.get("engine") == asdict(engine))
            v = rec.get("versions", {}).get("mlx_vlm")
            if v:
                versions.add(v)
    runs = wall = gen = 0.0
    if (root / RUNS_DIR).is_dir():
        for path in (root / RUNS_DIR).glob("*.json"):
            try:
                log = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if log.get("ocr_done", 0) + log.get("redone", 0) == 0:
                continue
            runs += 1
            wall += log.get("wall_s", 0.0)
            gen += log.get("gen_s", 0.0)
    return {
        "pages_cached": pages,
        "pages_current_engine": current,
        "pdfs": len(pdfs),
        "truncated": truncated,
        "engine": asdict(engine),
        "mlx_vlm": sorted(versions),
        "runs": int(runs),
        "wall_s_total": round(wall, 3),
        "gen_s_total": round(gen, 3),
    }


def format_report(rep: dict) -> str:
    e = rep["engine"]
    return (
        f"pages_cached {rep['pages_cached']} (current engine {rep['pages_current_engine']})  "
        f"pdfs {rep['pdfs']}  truncated {rep['truncated']}\n"
        f"engine {e['name']}  {e['model']}@{e['revision']}  {e['mode']}  "
        f"max_tokens {e['max_tokens']}  render {e['render_px']}\n"
        f"mlx_vlm {','.join(rep['mlx_vlm']) or '-'}  runs {rep['runs']}  "
        f"wall_s_total {rep['wall_s_total']}  gen_s_total {rep['gen_s_total']}"
    )
