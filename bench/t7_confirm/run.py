"""T7 mid-band confirm bench.

Importing this module keeps mlx out of module import. The model loads on the first
``decide`` of a real run. ``--write-config`` only writes JSON.

See ``bench/t7_confirm/README.md`` for the bands, the trump train files, and
the holdout rule. This script does not switch models and does not run holdout
after a train pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

BANDS = ((0.35, 0.74), (0.45, 0.74), (0.50, 0.65))
MODEL = "mlx-community/Qwen3.5-4B-4bit"
REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
CONFIRMED = {"confirmed_split", "confirmed_continue"}
HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"


class _TimedBackend:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.latencies: list[float] = []

    def decide(self, request, *, render=None):
        start = time.perf_counter()
        try:
            return self.backend.decide(request, render=render)
        finally:
            self.latencies.append(time.perf_counter() - start)

    def reset_parse_budget(self) -> None:
        self.backend.reset_parse_budget()

    def __getattr__(self, name):
        return getattr(self.backend, name)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench/t7_confirm/run.py")
    parser.add_argument("--set", choices=["builtin", "dlnet-whole", "politax", "all"], default="builtin")
    parser.add_argument("--subset", choices=["train", "holdout"], default=None)
    parser.add_argument("--low", type=float, default=None)
    parser.add_argument("--high", type=float, default=None)
    parser.add_argument("--write-config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _band_error(args) -> str | None:
    if (args.low is None) != (args.high is None):
        return "pass both --low and --high, or neither"
    if args.low is not None and args.low > args.high:
        return "low must be <= high"
    return None


def _set_error(args) -> str | None:
    if args.set == "politax" and args.subset not in {"train", "holdout"}:
        return "--set politax needs --subset train or --subset holdout"
    if args.set != "politax" and args.subset is not None:
        return "--subset is only for --set politax"
    return None


def _selected_band(args) -> tuple[float, float]:
    if args.low is None:
        return BANDS[0]
    return (args.low, args.high)


def _run_bands(args) -> list[tuple[float, float]]:
    if args.low is None:
        return list(BANDS)
    return [(args.low, args.high)]


def _confirm_document(low: float, high: float) -> dict:
    return {
        "low": low,
        "high": high,
        "task": "confirm_boundary",
        "decision": {
            "backend": "mlx_vlm",
            "model": MODEL,
            "revision": REVISION,
            "send_images": True,
            "capabilities": {"images": True},
        },
    }


def _write_config(args) -> int:
    low, high = _selected_band(args)
    path = args.write_config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_confirm_document(low, high), indent=2) + "\n")
    print(f"wrote {path}")
    return 0


def _false_splits(rows) -> int:
    return sum(len(row.boundary.predicted) - row.boundary.matched for row in rows)


def _macro_f1(rows) -> float:
    if not rows:
        return 0.0
    return sum(row.boundary.f1 for row in rows) / len(rows)


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. ``pct`` is 0-100."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    index = min(len(ordered), max(rank, 1)) - 1
    return ordered[index]


def _peak_rss_bytes() -> int:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _new_timed():
    from packet.decision import DecisionCapabilities
    from packet.decision_mlx import MlxVlmDecisionBackend

    backend = MlxVlmDecisionBackend(
        model=MODEL,
        revision=REVISION,
        send_images=True,
        capabilities=DecisionCapabilities(images=True),
    )
    return _TimedBackend(backend)


def _confirmer(timed, low: float, high: float):
    from packet.confirm import BackendConfirmer

    timed.latencies.clear()
    return BackendConfirmer(
        backend=timed, low=low, high=high, task="confirm_boundary", window=2
    )


def _score_one(fx, pipe):
    from packet.eval.metrics import boundary_score, false_ocr_hard_fail
    from packet.eval.run import EvalResult, text_layer_pages
    from packet.events import EventKind

    events = []
    predicted = pipe.parse(fx.pdf, on_event=events.append)
    score = boundary_score(fx.gold, predicted, tolerance=0)
    fails = false_ocr_hard_fail(predicted, text_layer_pages(fx.pdf))
    confirmed = 0
    for event in events:
        if event.kind != EventKind.BOUNDARY_DECIDED:
            continue
        if event.payload.get("reason") in CONFIRMED:
            confirmed += 1
    result = EvalResult(
        name=fx.name,
        condition=fx.condition,
        pages=fx.gold.origin.page_count,
        gold_docs=fx.gold.document_count(),
        pred_docs=predicted.document_count(),
        boundary=score,
        false_ocr_pages=fails,
        notes=fx.notes,
    )
    return result, confirmed


def _score_many(fixtures, pipe, confirmer):
    rows = []
    confirmed = 0
    degraded = []
    for fx in fixtures:
        result, pages = _score_one(fx, pipe)
        rows.append(result)
        confirmed += pages
        if confirmer is not None:
            degraded.append({"file": result.name, "pages": list(confirmer.degraded_pages)})
    return rows, confirmed, degraded


def _builtin_fixtures(args):
    from packet.eval.fixtures import build_builtin

    return build_builtin(args.dir)


def _dlnet_fixtures(args):
    from packet.eval import wholedoc
    from packet.eval.run import DEFAULT_WHOLEDOC_CACHE

    data = args.data_dir or DEFAULT_WHOLEDOC_CACHE
    manifest = wholedoc.load_manifest()
    wholedoc.fetch(manifest, cache=data)
    out = (args.dir or Path(".packet-eval")) / "dlnet-whole"
    return wholedoc.build_fixtures(manifest, data, out)


def _politax_bundle(args):
    from packet.eval.politax_split import partition_docs
    from packet.eval.run import DEFAULT_POLITAX_CACHE, prepare_politax

    data = args.data_dir or DEFAULT_POLITAX_CACHE
    manifest, fixtures = prepare_politax(data)
    train, holdout = partition_docs(manifest)
    chosen = train if args.subset == "train" else holdout
    names = {doc.file_name for doc in chosen}
    picked = [fx for fx in fixtures if (fx.notes or fx.pdf.name) in names]
    return manifest, picked, data


def _plain_rows(fixtures):
    from packet.pipeline import Pipeline

    rows, _confirmed, _degraded = _score_many(fixtures, Pipeline(), None)
    return rows


def _politax_rows(manifest, fixtures, data, config):
    from packet.eval.politax_ocr_eval import CacheStats, ocr_cache_pipeline

    sha = {doc.file_name: doc.sha256 for doc in manifest.docs}
    stats = CacheStats()
    confirmer = None if config is None else config.confirmer
    rows = []
    confirmed = 0
    degraded = []
    for fx in fixtures:
        pipe = ocr_cache_pipeline(data, sha[fx.notes or fx.pdf.name], stats, config)
        result, pages = _score_one(fx, pipe)
        rows.append(result)
        confirmed += pages
        if confirmer is not None:
            degraded.append({"file": result.name, "pages": list(confirmer.degraded_pages)})
    return rows, confirmed, degraded, stats


def _changes(base, got) -> int:
    by_name = {row.name: row for row in base}
    changed = 0
    for row in got:
        if by_name[row.name].boundary.predicted != row.boundary.predicted:
            changed += 1
    return changed


def _fixture_rows(base, got) -> list[dict]:
    by_name = {row.name: row for row in base}
    out = []
    for row in got:
        other = by_name[row.name]
        out.append({
            "name": row.name,
            "notes": row.notes,
            "f1": row.boundary.f1,
            "false_splits": len(row.boundary.predicted) - row.boundary.matched,
            "predicted": list(row.boundary.predicted),
            "baseline_predicted": list(other.boundary.predicted),
            "false_ocr_pages": list(row.false_ocr_pages),
        })
    return out


def _payload(args, set_name, low, high, timed, got, base, confirmed, degraded, stats, wall) -> dict:
    got_false = _false_splits(got)
    base_false = _false_splits(base)
    body = {
        "model": MODEL,
        "revision": REVISION,
        "set": set_name,
        "subset": args.subset,
        "low": low,
        "high": high,
        "load_seconds": timed.backend.load_seconds,
        "wall_seconds": wall,
        "peak_rss_bytes": _peak_rss_bytes(),
        "pages_confirmed": confirmed,
        "decision_changes": _changes(base, got),
        "macro_f1": _macro_f1(got),
        "false_splits": got_false,
        "baseline_macro_f1": _macro_f1(base),
        "baseline_false_splits": base_false,
        "eligible": got_false <= base_false,
        "latency_p50": _percentile(timed.latencies, 50),
        "latency_p95": _percentile(timed.latencies, 95),
        "degraded_pages": degraded,
        "fixtures": _fixture_rows(base, got),
    }
    if stats is not None:
        body["ocr_cache"] = stats.as_dict()
    return body


def _evaluate(args, set_name, confirmer):
    from packet.pipeline import Pipeline, PipelineConfig

    if set_name == "politax":
        manifest, fixtures, data = _politax_bundle(args)
        base, _c, _d, _stats = _politax_rows(manifest, fixtures, data, None)
        start = time.perf_counter()
        got, confirmed, degraded, stats = _politax_rows(
            manifest, fixtures, data, PipelineConfig(confirmer=confirmer)
        )
        return got, base, confirmed, degraded, stats, time.perf_counter() - start
    fixtures = _builtin_fixtures(args) if set_name == "builtin" else _dlnet_fixtures(args)
    base = _plain_rows(fixtures)
    pipe = Pipeline(config=PipelineConfig(confirmer=confirmer))
    start = time.perf_counter()
    got, confirmed, degraded = _score_many(fixtures, pipe, confirmer)
    return got, base, confirmed, degraded, None, time.perf_counter() - start


def _result_path(args, set_name: str, low: float, high: float) -> Path:
    root = args.out or RESULT_DIR
    subset = args.subset or "na"
    return root / f"{set_name}-{subset}-{low:.2f}-{high:.2f}.json"


def _emit(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if "best_eligible" in payload:
        best = payload["best_eligible"]
        picked = "none" if best is None else f"[{best['low']}, {best['high']}]"
        print(f"summary {payload['set']} {payload['subset']} best {picked} -> {path}")
        return
    print(
        f"{payload['set']} {payload['subset']} "
        f"[{payload['low']}, {payload['high']}] "
        f"macro {payload['macro_f1']:.4f} false_splits {payload['false_splits']} "
        f"eligible {payload['eligible']} -> {path}"
    )


def _best_eligible(rows: list[dict]) -> dict | None:
    best = None
    for row in rows:
        if not row["eligible"]:
            continue
        if best is None or row["macro_f1"] > best["macro_f1"]:
            best = {"low": row["low"], "high": row["high"], "macro_f1": row["macro_f1"]}
    return best


def _emit_summary(args, set_name: str, rows: list[dict]) -> None:
    root = args.out or RESULT_DIR
    subset = args.subset or "na"
    path = root / f"summary-{set_name}-{subset}.json"
    compact = [
        {
            "low": row["low"],
            "high": row["high"],
            "macro_f1": row["macro_f1"],
            "false_splits": row["false_splits"],
            "baseline_false_splits": row["baseline_false_splits"],
            "eligible": row["eligible"],
        }
        for row in rows
    ]
    body = {
        "set": set_name,
        "subset": args.subset,
        "bands": compact,
        "best_eligible": _best_eligible(rows),
        "holdout": "not run by this script",
    }
    _emit(path, body)


def _sets(args) -> list[str]:
    if args.set == "all":
        return ["builtin", "dlnet-whole"]
    return [args.set]


def _run(args) -> int:
    timed = _new_timed()
    for set_name in _sets(args):
        summaries = []
        for low, high in _run_bands(args):
            confirmer = _confirmer(timed, low, high)
            got, base, confirmed, degraded, stats, wall = _evaluate(args, set_name, confirmer)
            payload = _payload(
                args, set_name, low, high, timed, got, base, confirmed, degraded, stats, wall
            )
            path = _result_path(args, set_name, low, high)
            _emit(path, payload)
            summaries.append(payload)
        if len(summaries) > 1:
            _emit_summary(args, set_name, summaries)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    band_error = _band_error(args)
    if band_error:
        print(band_error, file=sys.stderr)
        return 2
    if args.write_config is not None:
        return _write_config(args)
    set_error = _set_error(args)
    if set_error:
        print(set_error, file=sys.stderr)
        return 2
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
