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
    engine_calls: int = 0
    ocr_done: int = 0
    redone: int = 0
    skipped_cached: int = 0
    truncated: int = 0
    interrupted: bool = False
    engine: dict = field(default_factory=dict)
    setup_s: float = 0.0
    render_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    write_s: float = 0.0
    stops: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PageResult:
    raw: str
    tokens: int | None
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    stop: str = "eos"
    render_s: float = 0.0


Recognize = Callable[[PagePlan], PageResult]


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
        versions: dict[str, str] | None = None, setup_s: float = 0.0) -> RunSummary:
    """OCR every pending, stale or corrupt page; skip current records. Single writer.

    ``limit`` caps ``recognize`` calls. A log is written even on Ctrl-C or an engine
    error; the page in flight is the most that is lost."""
    data_dir = Path(data_dir)
    _clean_temp(data_dir)
    summary = RunSummary(started=_now(), planned=len(plans), engine=asdict(engine),
                         setup_s=setup_s)
    t_start = clock()
    try:
        for plan in plans:
            state = _state(plan, data_dir, engine)
            if state == "cached":
                summary.skipped_cached += 1
                continue
            if limit is not None and summary.engine_calls >= limit:
                break
            t0 = clock()
            summary.engine_calls += 1
            result = recognize(plan)
            secs = clock() - t0
            truncated = result.tokens is not None and result.tokens >= engine.max_tokens
            t_write = time.perf_counter()
            _atomic_write_json(record_path(data_dir, plan.pdf_sha256, plan.index), {
                "schema": SCHEMA,
                "pdf_sha256": plan.pdf_sha256,
                "file_name": plan.file_name,
                "page": plan.index,
                "native_chars": plan.native_chars,
                "engine": asdict(engine),
                "versions": dict(versions or {}),
                "raw": result.raw,
                "seconds": round(secs, 3),
                "generation_tokens": result.tokens,
                "truncated": truncated,
                "created": _now(),
                "render_s": round(result.render_s, 3),
                "prefill_s": round(result.prefill_s, 3),
                "decode_s": round(result.decode_s, 3),
                "prompt_tokens": result.prompt_tokens,
                "stop": result.stop,
            })
            summary.write_s += time.perf_counter() - t_write
            summary.gen_s += secs
            summary.render_s += result.render_s
            summary.prefill_s += result.prefill_s
            summary.decode_s += result.decode_s
            summary.stops[result.stop] = summary.stops.get(result.stop, 0) + 1
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
        summary.setup_s = round(summary.setup_s, 3)
        summary.render_s = round(summary.render_s, 3)
        summary.prefill_s = round(summary.prefill_s, 3)
        summary.decode_s = round(summary.decode_s, 3)
        summary.write_s = round(summary.write_s, 3)
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
    stops: Counter = Counter()
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
            stop = rec.get("stop")
            if stop:
                stops[stop] += 1
    runs = wall = gen = 0.0
    setup = render = prefill = decode = write = 0.0
    if (root / RUNS_DIR).is_dir():
        for path in (root / RUNS_DIR).glob("*.json"):
            try:
                log = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if log.get("engine_calls", log.get("ocr_done", 0) + log.get("redone", 0)) <= 0:
                continue
            runs += 1
            wall += log.get("wall_s", 0.0)
            gen += log.get("gen_s", 0.0)
            setup += log.get("setup_s", 0.0)
            render += log.get("render_s", 0.0)
            prefill += log.get("prefill_s", 0.0)
            decode += log.get("decode_s", 0.0)
            write += log.get("write_s", 0.0)
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
        "setup_s": round(setup, 3),
        "render_s": round(render, 3),
        "prefill_s": round(prefill, 3),
        "decode_s": round(decode, 3),
        "write_s": round(write, 3),
        "stops": {"eos": int(stops.get("eos", 0)), "max_tokens": int(stops.get("max_tokens", 0)),
                  **{k: int(v) for k, v in stops.items() if k not in {"eos", "max_tokens"}}},
    }


def format_report(rep: dict) -> str:
    e = rep["engine"]
    stops = rep.get("stops") or {}
    ordered = ["eos", "max_tokens", *[k for k in stops if k not in {"eos", "max_tokens"}]]
    stop_s = " ".join(f"{k}={stops.get(k, 0)}" for k in ordered)
    return (
        f"pages_cached {rep['pages_cached']} (current engine {rep['pages_current_engine']})  "
        f"pdfs {rep['pdfs']}  truncated {rep['truncated']}\n"
        f"engine {e['name']}  {e['model']}@{e['revision']}  {e['mode']}  "
        f"max_tokens {e['max_tokens']}  render {e['render_px']}\n"
        f"mlx_vlm {','.join(rep['mlx_vlm']) or '-'}  runs {rep['runs']}  "
        f"wall_s_total {rep['wall_s_total']}  gen_s_total {rep['gen_s_total']}\n"
        f"setup_s {rep.get('setup_s', 0)}  render_s {rep.get('render_s', 0)}  "
        f"prefill_s {rep.get('prefill_s', 0)}  decode_s {rep.get('decode_s', 0)}  "
        f"write_s {rep.get('write_s', 0)}\n"
        f"stops {stop_s}"
    )
