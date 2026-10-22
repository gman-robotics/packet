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
                    help="macro-average boundary F1 floor in [0,1]; default = --min-f1")

    plan = sub.add_parser("eval-plan", help="plan DocLayNet synthetic packets from a catalog")
    plan.add_argument("--catalog", required=True)
    plan.add_argument("--slice", default="poly-seq")
    plan.add_argument("--split", default="val")
    plan.add_argument("--out", default="evals/generated")
    plan.add_argument("--count", type=int, default=8)

    args = parser.parse_args(argv)

    if args.cmd == "eval":
        return _cmd_eval(args)
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


def _cmd_eval(args) -> int:
    from packet.eval import run as ev

    root = Path(args.dir) if args.dir else None
    floors = ev.SET_FLOORS[args.set]
    min_f1 = floors[0] if args.min_f1 is None else args.min_f1
    min_macro = floors[1] if args.min_macro_f1 is None else args.min_macro_f1
    politax_census = None
    politax_holdout_baseline = None
    if args.set == "dlnet-whole":
        data = Path(args.data_dir) if args.data_dir else ev.DEFAULT_WHOLEDOC_CACHE
        results = ev.run_dlnet_whole(data, root)
    elif args.set == "politax":
        from packet.eval import politax
        from packet.eval.politax_split import compute_holdout_baseline

        data = Path(args.data_dir) if args.data_dir else ev.DEFAULT_POLITAX_CACHE
        m, fixtures = ev.prepare_politax(data)
        subjects = {d.file_name: d.subject for d in m.docs}
        politax_census = politax.text_layer_census(fixtures, subjects=subjects)
        results = [ev.run_fixture(fx) for fx in fixtures]
        politax_holdout_baseline = compute_holdout_baseline(m, fixtures, results)
    else:
        results = ev.run_named_fixtures(root)
    gate = ev.evaluate_gate(results, min_f1=min_f1, min_macro_f1=min_macro)
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
            },
        }
        if args.set == "politax":
            from packet.eval.metrics import micro_boundary_prf

            mp, mr, mf = micro_boundary_prf(results)
            payload["micro_boundary_p"] = mp
            payload["micro_boundary_r"] = mr
            payload["micro_boundary_f1"] = mf
            if politax_census is not None:
                payload["text_layer_census"] = politax_census
            if politax_holdout_baseline is not None:
                from packet.eval.politax_split import holdout_baseline_as_dict

                payload["holdout_baseline"] = holdout_baseline_as_dict(politax_holdout_baseline)
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
        mp, mr, mf = micro_boundary_prf(results)
        print(f"micro boundary P {mp:.2f} R {mr:.2f} F1 {mf:.2f} (all {len(results)} packets)")
        if politax_holdout_baseline is not None:
            from packet.eval.politax_split import format_holdout_baseline

            print(format_holdout_baseline(politax_holdout_baseline))
    print(f"macro F1 {gate.macro_f1:.2f} (min {gate.min_macro_f1:.2f}; per-fixture min {min_f1:.2f})")
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
