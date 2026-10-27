from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from packet.backends import OcrConfigError, default_extractors
from packet.pipeline import Pipeline, PipelineConfig

_DISTRIBUTION = "packet-dl"


def _unit_interval(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not (0.0 <= x <= 1.0):  # rejects nan, inf, negatives, > 1
        raise argparse.ArgumentTypeError(f"must be within [0, 1]: {value!r}")
    return x


def _ocr_config_stderr_line(exc: OcrConfigError) -> str:
    msg = str(exc)
    if exc.pip_extra == "ocr-mlx":
        return (
            f"packet: {msg}; install with pip install '{_DISTRIBUTION}[ocr-mlx]' "
            "on Apple silicon"
        )
    if exc.pip_extra == "ocr-paddle":
        return f"packet: {msg}; install with pip install '{_DISTRIBUTION}[ocr-paddle]'"
    return f"packet: {msg}"


def _fail_ocr_config(exc: OcrConfigError) -> None:
    print(_ocr_config_stderr_line(exc), file=sys.stderr)
    raise SystemExit(2) from exc


def _extractors_or_exit(force_inspect: bool = False):
    try:
        return default_extractors(force_inspect=force_inspect)
    except OcrConfigError as exc:
        _fail_ocr_config(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packet", description="Document language understanding")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="classify pages and print events")
    inspect.add_argument("path")
    inspect.add_argument("--force-inspect", action="store_true", help="OCR every page, even text pages")

    parse = sub.add_parser("parse", help="write a PacketTree JSON file")
    parse.add_argument("path", nargs="+",
                       help="one or more PDFs; several files share one loaded OCR model")
    parse.add_argument("-o", "--output", default="-", help="output file for a single PDF")
    parse.add_argument("--out-dir", default=None,
                       help="write <name>.json per PDF here (required for several PDFs)")
    parse.add_argument("--timings", default=None,
                       help="append one JSON line per PDF: seconds, pages, OCR page seconds")
    parse.add_argument("--no-split", action="store_true", help="force a single logical document")
    parse.add_argument("--force-inspect", action="store_true", help="OCR every page, even text pages")
    parse.add_argument(
        "--boundary-weights",
        default=None,
        help="JSON logistic boundary weights (optional; load errors exit 2)",
    )
    parse.add_argument(
        "--decision-config",
        default=None,
        help="JSON decision backend config for parse (optional; load errors exit 2)",
    )

    ev = sub.add_parser("eval", help="run built-in packet fixtures")
    ev.add_argument("--json", action="store_true")
    ev.add_argument("--dir", default=None, help="cache dir for generated PDFs")
    ev.add_argument(
        "--set",
        default="builtin",
        choices=["builtin", "dlnet-whole", "politax"],
        help="builtin fixtures (default), packet-dlnet-whole, or packet-politax (PoliTax-Split)",
    )
    ev.add_argument(
        "--data-dir",
        default=None,
        help="download cache for --set dlnet-whole (default ~/.cache/packet/dlnet-whole) "
        "or politax (default ~/.cache/packet/politax)",
    )
    ev.add_argument("--min-f1", type=_unit_interval, default=None,
                    help="per-fixture boundary F1 floor in [0,1] (tolerance 0); default 1.0, "
                         "which suits the deterministic built-in fixtures; lower it for "
                         "real-model or DocLayNet runs")
    ev.add_argument("--min-macro-f1", type=_unit_interval, default=None,
                    help="macro-average boundary F1 floor in [0,1]; default = the set floor "
                         "(politax --ocr-cache 0.41 on holdout; builtin = --min-f1)")
    ev.add_argument(
        "--boundary-weights",
        default=None,
        help="JSON logistic boundary weights for eval (optional; load errors exit 2)",
    )
    ev.add_argument(
        "--ocr-cache",
        action="store_true",
        help="--set politax only (GMA-274, opt-in): feed scan pages from the cached "
             "politax-ocr text under <data-dir>/ocr; no OCR runs; off by default; "
             "gates holdout macro F1 >= 0.41 (GMA-274 Q7)",
    )

    pt = sub.add_parser(
        "politax-train-weights",
        help="train schema-1 logistic weights on PoliTax trump PDFs only",
    )
    pt.add_argument(
        "--data-dir",
        default=None,
        help="PoliTax download cache (default ~/.cache/packet/politax)",
    )
    pt.add_argument(
        "-o",
        "--output",
        required=True,
        help="write exported JSON weights (not committed by default)",
    )

    po = sub.add_parser(
        "politax-ocr",
        help="optional OCR text cache for PoliTax scan pages (LightOn MLX; outside eval/CI)",
    )
    po.add_argument("--data-dir", default=None,
                    help="PoliTax download cache (default ~/.cache/packet/politax)")
    po.add_argument("--subject", default=None, help="only this manifest subject")
    po.add_argument("--file", default=None, help="only this PDF (manifest file or file_name)")
    po.add_argument("--limit-pages", type=int, default=None,
                    help="stop after this many engine calls (cached pages do not count)")
    po.add_argument("--max-tokens", type=int, default=4096)
    po.add_argument("--render-px", type=int, default=1540)
    po.add_argument(
        "--loop-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop a page when its tail compresses under the zlib loop rule "
             "(default: on; --no-loop-stop runs to eos or max tokens)",
    )
    mode = po.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="fetch PDFs, print selected/cached/pending pages; no model load")
    mode.add_argument("--report", action="store_true",
                      help="read-only totals for the PR (pages, engine, wall time)")
    ps = sub.add_parser(
        "politax-ship-check",
        help="run PoliTax default-on ship gate (GMA-291); does not change defaults",
    )
    ps.add_argument(
        "--weights",
        default=None,
        help="existing JSON weights; if omitted, trains trump weights to --train-output",
    )
    ps.add_argument(
        "--train-output",
        default=None,
        help="path to write freshly trained weights when --weights is omitted",
    )
    ps.add_argument("--data-dir", default=None, help="PoliTax cache (default ~/.cache/packet/politax)")
    ps.add_argument(
        "--dlnet-cache",
        default=None,
        help="dlnet-whole cache for item 4 (default ~/.cache/packet/dlnet-whole)",
    )
    ps.add_argument("--json", action="store_true")

    plan = sub.add_parser("eval-plan", help="plan DocLayNet synthetic packets from a catalog")
    plan.add_argument("--catalog", required=True)
    plan.add_argument("--slice", default="poly-seq")
    plan.add_argument("--split", default="val")
    plan.add_argument("--out", default="evals/generated")
    plan.add_argument("--count", type=int, default=8)

    args = parser.parse_args(argv)

    if args.cmd == "eval":
        return _cmd_eval(args)
    if args.cmd == "politax-train-weights":
        return _cmd_politax_train_weights(args)
    if args.cmd == "politax-ocr":
        return _cmd_politax_ocr(args)
    if args.cmd == "politax-ship-check":
        return _cmd_politax_ship_check(args)
    if args.cmd == "eval-plan":
        return _cmd_eval_plan(args)

    if args.cmd == "parse":
        return _cmd_parse(args)

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2
    pipe = Pipeline(extractors=_extractors_or_exit(force_inspect=args.force_inspect))
    try:
        for event in pipe.stream(path):
            print(f"{event.kind.value:20} page={event.page} {json.dumps(event.payload)[:180]}")
    except OcrConfigError as exc:
        _fail_ocr_config(exc)
    return 0


def _parse_pipeline_config(args) -> PipelineConfig | None:
    from packet.boundary_model import BoundaryWeightsError, load_logistic_scorer
    from packet.decision import DecisionBackendError, load_decision_config
    from packet.pipeline import PipelineConfig

    if args.boundary_weights and args.decision_config:
        print(
            "packet: --decision-config cannot be used with --boundary-weights",
            file=sys.stderr,
        )
        return None
    config = PipelineConfig()
    if args.boundary_weights:
        try:
            config.scorer = load_logistic_scorer(Path(args.boundary_weights))
        except BoundaryWeightsError as exc:
            print(f"packet: {exc}", file=sys.stderr)
            return None
    if args.decision_config:
        try:
            config.decision_backend = load_decision_config(Path(args.decision_config))
        except (DecisionBackendError, BoundaryWeightsError) as exc:
            print(f"packet: {exc}", file=sys.stderr)
            return None
    return config


def _timing_decision_fields(config: PipelineConfig) -> dict:
    backend = config.decision_backend
    if backend is None:
        return {}
    return {
        "decision_backend": backend.backend_label,
        "decision_degraded_pages": list(backend.degraded_pages),
    }


def _cmd_parse(args) -> int:
    paths = [Path(p) for p in args.path]
    if len(paths) > 1 and not args.out_dir:
        print("packet parse: several PDFs need --out-dir", file=sys.stderr)
        return 2
    if len(paths) > 1 and args.output != "-":
        print("packet parse: -o takes one PDF; use --out-dir for several", file=sys.stderr)
        return 2
    stems = [p.stem for p in paths]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        print(f"packet parse: several PDFs have the same name ({', '.join(dupes)}); "
              "their outputs would overwrite each other", file=sys.stderr)
        return 2
    missing = [p for p in paths if not p.exists()]
    if len(paths) == 1 and missing:
        print(f"not found: {missing[0]}", file=sys.stderr)
        return 2

    # One pipeline and one set of extractors for the whole run; OCR engines also sit in
    # the process-level cache, so the model loads once however many files there are.
    extractors = _extractors_or_exit(force_inspect=args.force_inspect)
    from packet.decision import DecisionBackendError

    config = _parse_pipeline_config(args)
    if config is None:
        return 2
    pipe = Pipeline(extractors=extractors, config=config)
    if args.no_split:
        pipe.config.split_on_boundary = False
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in paths:
        backend = config.decision_backend
        if backend is not None:
            backend.reset_parse_budget()
        for ext in extractors:
            if hasattr(ext, "ocr_timings"):
                ext.ocr_timings.clear()
        start = time.perf_counter()
        row: dict = {"file": path.name}
        try:
            if not path.exists():
                raise FileNotFoundError(f"not found: {path}")
            packet = pipe.parse(path)
            text = json.dumps(packet.model_dump(), indent=2)
            if out_dir:
                out = out_dir / f"{path.stem}.json"
            elif args.output != "-":
                out = Path(args.output)
            else:
                out = None
            if out is None:
                print(text)
            else:
                out.write_text(text)
                print(f"wrote {out} ({packet.document_count()} document(s), "
                      f"{packet.origin.page_count} pages)")
            row.update(ok=True, pages=packet.origin.page_count, documents=packet.document_count())
            row.update(_timing_decision_fields(config))
        except OcrConfigError as exc:
            _fail_ocr_config(exc)
        except DecisionBackendError as exc:
            print(f"packet: {exc}", file=sys.stderr)
            row.update(ok=False, error=str(exc))
            _write_timing(args.timings, row, start, extractors)
            return 2
        except Exception as exc:  # one bad file must not stop a batch
            failed += 1
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            _write_timing(args.timings, row, start, extractors)
            if len(paths) == 1:
                raise
            print(f"packet: {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        _write_timing(args.timings, row, start, extractors)
    return 1 if failed else 0


def _write_timing(path: str | None, row: dict, start: float, extractors) -> None:
    if not path:
        return
    row["seconds"] = round(time.perf_counter() - start, 3)
    row["ocr_page_seconds"] = [round(t["seconds"], 3) for ext in extractors
                               for t in getattr(ext, "ocr_timings", [])]
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _pipeline_for_boundary_weights(path: str | None) -> Pipeline | None:
    if not path:
        return Pipeline()
    from packet.boundary_model import BoundaryWeightsError, load_logistic_scorer

    try:
        scorer = load_logistic_scorer(Path(path))
    except BoundaryWeightsError as exc:
        print(f"packet: {exc}", file=sys.stderr)
        return None
    return Pipeline(config=PipelineConfig(scorer=scorer))


def _cmd_politax_ship_check(args) -> int:
    from packet.eval.politax_ship import evaluate_politax_ship, train_and_evaluate_ship
    from packet.eval.run import DEFAULT_POLITAX_CACHE, DEFAULT_WHOLEDOC_CACHE

    politax = Path(args.data_dir) if args.data_dir else DEFAULT_POLITAX_CACHE
    dlnet = Path(args.dlnet_cache) if args.dlnet_cache else DEFAULT_WHOLEDOC_CACHE
    if args.weights:
        report = evaluate_politax_ship(Path(args.weights), politax, dlnet)
    else:
        out = Path(args.train_output or "/tmp/packet-politax-ship-weights.json")
        report = train_and_evaluate_ship(politax, out, dlnet)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"ship={'pass' if report.ship else 'FAIL'} outcome ({report.outcome})")
        for item in report.items:
            mark = "PASS" if item.passed else "FAIL"
            print(f"  {item.number}. [{mark}] {item.title}: {item.detail}")
        print(f"max holdout model p={report.max_holdout_model_p:.4f}")
    return 0


def _cmd_politax_train_weights(args) -> int:
    from packet.eval.politax_train import train_trump_logistic
    from packet.eval.run import DEFAULT_POLITAX_CACHE

    data = Path(args.data_dir) if args.data_dir else DEFAULT_POLITAX_CACHE
    out = Path(args.output)
    _, rows = train_trump_logistic(data, out)
    print(
        f"wrote {out} ({len(rows.training_packets)} trump PDFs, "
        f"{len(rows.X)} training pages, {sum(rows.y)} gold cuts)"
    )
    return 0


def _cmd_politax_ocr(args) -> int:
    import importlib.metadata
    import importlib.util

    from packet.backends import LightOnExtractor, LoopRule
    from packet.eval import politax
    from packet.eval import politax_ocr as po
    from packet.eval.run import DEFAULT_POLITAX_CACHE

    data = (Path(args.data_dir) if args.data_dir else DEFAULT_POLITAX_CACHE).expanduser()
    repo = Path(__file__).resolve().parents[2]
    if data.resolve().is_relative_to(repo):
        print(f"packet: --data-dir {data} is inside the repo checkout; use a path outside git",
              file=sys.stderr)
        return 2
    if args.limit_pages is not None and args.limit_pages < 0:
        print("packet: --limit-pages must be >= 0", file=sys.stderr)
        return 2
    engine = po.EngineId("lighton-mlx", LightOnExtractor.model, LightOnExtractor.revision,
                         LightOnExtractor.mode, args.max_tokens, args.render_px)
    if args.report:
        rep = po.report(data, engine)
        print(json.dumps(rep, indent=2))
        print(po.format_report(rep))
        return 0
    m = politax.load_manifest()
    t_setup = time.perf_counter()
    politax.fetch(m, cache=data)
    plans, native = po.plan_pages(m.docs, data, subject=args.subject, file=args.file)
    states = po.classify(plans, data, engine)
    setup_s = time.perf_counter() - t_setup
    print(f"selected {len(plans)} pages (<{po.OCR_MIN_NATIVE_CHARS} native chars), "
          f"native-text skipped {native}; cached {states['cached']} stale {states['stale']} "
          f"corrupt {states['corrupt']} pending {states['pending']}")
    if args.dry_run:
        return 0
    if importlib.util.find_spec("mlx_vlm") is None:
        _fail_ocr_config(OcrConfigError("politax-ocr needs mlx-vlm", pip_extra="ocr-mlx"))
    from packet.source import PdfPageSource

    ext = LightOnExtractor(max_tokens=args.max_tokens)
    sources: dict[str, PdfPageSource] = {}
    loop_rule = LoopRule() if args.loop_stop else None

    def recognize(plan):
        t0 = time.perf_counter()
        src = sources.get(plan.file)
        if src is None:
            for old in sources.values():
                old.close()
            sources.clear()
            src = sources[plan.file] = PdfPageSource(data / plan.file)
        img = src.render(plan.index, longest_px=engine.render_px)
        render_s = time.perf_counter() - t0
        rec = ext.recognize(img, loop_rule=loop_rule)
        return po.PageResult(raw=rec.raw, tokens=rec.tokens, prompt_tokens=rec.prompt_tokens,
                             prefill_s=rec.prefill_s, decode_s=rec.decode_s, stop=rec.stop,
                             render_s=render_s)

    versions = {"packet": _version_or_dash(importlib.metadata, _DISTRIBUTION),
                "mlx_vlm": _version_or_dash(importlib.metadata, "mlx-vlm")}
    try:
        s = po.run(plans, recognize, data, engine, limit=args.limit_pages, versions=versions,
                   setup_s=setup_s)
    except Exception as exc:  # noqa: BLE001 - any engine failure exits 1; log is kept
        print(f"packet: politax-ocr engine error: {exc}", file=sys.stderr)
        return 1
    finally:
        for src in sources.values():
            src.close()
    print(f"ocr_done {s.ocr_done} redone {s.redone} skipped_cached {s.skipped_cached} "
          f"truncated {s.truncated} wall_s {s.wall_s} gen_s {s.gen_s}")
    return 0


def _version_or_dash(md, dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "-"


def _cmd_eval(args) -> int:
    from packet.eval import run as ev

    root = Path(args.dir) if args.dir else None
    floors = (ev.OCR_CACHE_FLOORS if args.ocr_cache else ev.SET_FLOORS).get(
        args.set, ev.SET_FLOORS[args.set])
    min_f1 = floors[0] if args.min_f1 is None else args.min_f1
    min_macro = floors[1] if args.min_macro_f1 is None else args.min_macro_f1
    politax_census = None
    politax_gold_totals = None
    politax_holdout_baseline = None
    politax_holdout_scorer = None
    ocr_stats = None
    macro_results = None
    if args.ocr_cache and args.set != "politax":
        print("packet: --ocr-cache needs --set politax", file=sys.stderr)
        return 2
    if args.ocr_cache and args.boundary_weights:
        print("packet: --ocr-cache cannot be used with --boundary-weights", file=sys.stderr)
        return 2
    pipe = _pipeline_for_boundary_weights(args.boundary_weights)
    if pipe is None:
        return 2
    if args.set == "dlnet-whole":
        data = Path(args.data_dir) if args.data_dir else ev.DEFAULT_WHOLEDOC_CACHE
        results = ev.run_dlnet_whole(data, root, pipeline=pipe)
    elif args.set == "politax":
        from packet.eval import politax
        from packet.eval.politax_split import compute_holdout_baseline
        from packet.eval.politax_train import holdout_scorecard

        data = Path(args.data_dir) if args.data_dir else ev.DEFAULT_POLITAX_CACHE
        m, fixtures = ev.prepare_politax(data)
        subjects = {d.file_name: d.subject for d in m.docs}
        politax_gold_totals = politax.corpus_gold_label_totals(m)
        politax_census = politax.text_layer_census(fixtures, subjects=subjects)
        if args.ocr_cache:
            from packet.eval.politax_ocr_eval import run_with_ocr_cache
            from packet.eval.politax_split import holdout_results

            results_heur, ocr_stats = run_with_ocr_cache(m, fixtures, data)
            macro_results = holdout_results(m, fixtures, results_heur)
        else:
            heur_pipe = Pipeline() if args.boundary_weights else pipe
            results_heur = [ev.run_fixture(fx, heur_pipe) for fx in fixtures]
        politax_holdout_baseline = compute_holdout_baseline(m, fixtures, results_heur)
        if args.boundary_weights:
            results = [ev.run_fixture(fx, pipe) for fx in fixtures]
            politax_holdout_scorer = holdout_scorecard(m, fixtures, results)
        else:
            results = results_heur
    else:
        results = ev.run_named_fixtures(root, pipeline=pipe)
    gate = ev.evaluate_gate(results, min_f1=min_f1, min_macro_f1=min_macro,
                            macro_results=macro_results)
    if args.json:
        payload = {
            "set": args.set,
            "fixtures": ev.results_as_dict(results, min_f1),
            "gate": {
                "passed": gate.passed,
                "failures": gate.failures,
                "macro_f1": gate.macro_f1,
                "min_f1": gate.min_f1,
                "min_macro_f1": gate.min_macro_f1,
                "macro_scope": "holdout" if macro_results is not None else "all",
            },
        }
        if args.set == "politax":
            from packet.eval.metrics import micro_boundary_prf

            mp, mr, mf = micro_boundary_prf(results)
            payload["micro_boundary_p"] = mp
            payload["micro_boundary_r"] = mr
            payload["micro_boundary_f1"] = mf
            if politax_gold_totals is not None:
                boundaries, sub_documents = politax_gold_totals
                payload["gold_label_totals"] = {
                    "boundaries": boundaries,
                    "sub_documents": sub_documents,
                }
            if politax_census is not None:
                payload["text_layer_census"] = politax_census
            if ocr_stats is not None:
                payload["ocr_cache"] = ocr_stats.as_dict()
            if politax_holdout_baseline is not None:
                from packet.eval.politax_split import holdout_baseline_as_dict

                payload["holdout_heuristic"] = holdout_baseline_as_dict(politax_holdout_baseline)
            if politax_holdout_scorer is not None:
                payload["holdout_scorer"] = {
                    "macro_f1": politax_holdout_scorer.macro_f1,
                    "micro_boundary_p": politax_holdout_scorer.micro_p,
                    "micro_boundary_r": politax_holdout_scorer.micro_r,
                    "micro_boundary_f1": politax_holdout_scorer.micro_f1,
                    "per_packet_f1": politax_holdout_scorer.per_packet_f1,
                }
        print(json.dumps(payload, indent=2))
        return 0 if gate.passed else 1
    print(f"{'fixture':16} {'cond':14} pages gold pred   P     R    F1  ocr  gate")
    for r in results:
        mark = "PASS" if r.passed_gate(min_f1) else "FAIL"
        print(
            f"{r.name:16} {r.condition:14} {r.pages:5} {r.gold_docs:4} {r.pred_docs:4} "
            f"{r.boundary.precision:5.2f} {r.boundary.recall:5.2f} {r.boundary.f1:5.2f} "
            f"{len(r.false_ocr_pages):4}  {mark}"
        )
    if politax_census is not None:
        from packet.eval import politax
        from packet.eval.metrics import micro_boundary_prf

        print(politax.format_census(politax_census))
        if politax_gold_totals is not None:
            print(politax.format_gold_label_totals(*politax_gold_totals))
        mp, mr, mf = micro_boundary_prf(results)
        print(f"micro boundary P {mp:.2f} R {mr:.2f} F1 {mf:.2f} (all {len(results)} packets)")
        if ocr_stats is not None:
            st = ocr_stats.as_dict()
            print(f"ocr cache (GMA-274 opt-in): pages used {st['pages_used']} "
                  f"missing {st['pages_missing']} truncated {st['pages_truncated']}")
        if politax_holdout_baseline is not None:
            from packet.eval.politax_split import format_holdout_baseline

            print(format_holdout_baseline(politax_holdout_baseline))
        if politax_holdout_scorer is not None:
            print("holdout with boundary weights:")
            print(
                f"  macro F1 {politax_holdout_scorer.macro_f1:.4f}  "
                f"micro P {politax_holdout_scorer.micro_p:.4f} "
                f"R {politax_holdout_scorer.micro_r:.4f} "
                f"F1 {politax_holdout_scorer.micro_f1:.4f}"
            )
            for name, f1 in politax_holdout_scorer.per_packet_f1.items():
                print(f"    {name}: F1 {f1:.4f}")
    scope = " holdout" if macro_results is not None else ""
    print(f"macro{scope} F1 {gate.macro_f1:.2f} (min {gate.min_macro_f1:.2f}; "
          f"per-fixture min {min_f1:.2f})")
    if gate.passed:
        print("GATE PASS")
        return 0
    print("GATE FAIL")
    for f in gate.failures:
        print(f"  - {f}")
    return 1


def _cmd_eval_plan(args) -> int:
    from packet.eval.doclaynet import (
        load_catalog,
        materialize_placeholder_packet,
        plan_packets,
        write_manifest,
    )

    catalog = Path(args.catalog)
    if not catalog.exists():
        print(f"not found: {catalog}", file=sys.stderr)
        return 2
    grouped = load_catalog(catalog)
    records = grouped.get(args.split, [])
    if not records:
        print(f"no records for split={args.split}", file=sys.stderr)
        return 2
    specs = plan_packets(
        records, slice_name=args.slice, split=args.split, packet_count=args.count
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.json", specs)
    for spec in specs:
        materialize_placeholder_packet(spec, out)
    print(f"planned {len(specs)} packets under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
