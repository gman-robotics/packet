from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from packet.backends import default_extractors
from packet.pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packet", description="Document language understanding")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="classify pages and print events")
    inspect.add_argument("path")
    inspect.add_argument("--force-inspect", action="store_true", help="OCR every page, even text pages")

    parse = sub.add_parser("parse", help="write a PacketTree JSON file")
    parse.add_argument("path")
    parse.add_argument("-o", "--output", default="-")
    parse.add_argument("--no-split", action="store_true", help="force a single logical document")
    parse.add_argument("--force-inspect", action="store_true", help="OCR every page, even text pages")

    ev = sub.add_parser("eval", help="run built-in packet fixtures")
    ev.add_argument("--json", action="store_true")
    ev.add_argument("--dir", default=None, help="cache dir for generated PDFs")
    ev.add_argument("--min-f1", type=float, default=None,
                    help="per-fixture boundary F1 floor (tolerance 0); default 1.0")
    ev.add_argument("--min-macro-f1", type=float, default=None,
                    help="macro-average boundary F1 floor; default = --min-f1")

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

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    pipe = Pipeline(extractors=default_extractors(force_inspect=args.force_inspect))
    if args.cmd == "parse" and args.no_split:
        pipe.config.split_on_boundary = False

    if args.cmd == "inspect":
        for event in pipe.stream(path):
            print(f"{event.kind.value:20} page={event.page} {json.dumps(event.payload)[:180]}")
        return 0

    packet = pipe.parse(path)
    payload = packet.model_dump()
    text = json.dumps(payload, indent=2)
    if args.output == "-":
        print(text)
    else:
        out = Path(args.output)
        out.write_text(text)
        print(f"wrote {out} ({packet.document_count()} document(s), {packet.origin.page_count} pages)")
    return 0


def _cmd_eval(args) -> int:
    from packet.eval import run as ev

    root = Path(args.dir) if args.dir else None
    min_f1 = ev.DEFAULT_MIN_F1 if args.min_f1 is None else args.min_f1
    results = ev.run_named_fixtures(root)
    gate = ev.evaluate_gate(results, min_f1=min_f1, min_macro_f1=args.min_macro_f1)
    if args.json:
        print(json.dumps({
            "fixtures": ev.results_as_dict(results, min_f1),
            "gate": {"passed": gate.passed, "failures": gate.failures, "macro_f1": gate.macro_f1,
                     "min_f1": gate.min_f1, "min_macro_f1": gate.min_macro_f1},
        }, indent=2))
        return 0 if gate.passed else 1
    print(f"{'fixture':16} {'cond':14} pages gold pred   P     R    F1  ocr  gate")
    for r in results:
        mark = "PASS" if r.passed_gate(min_f1) else "FAIL"
        print(
            f"{r.name:16} {r.condition:14} {r.pages:5} {r.gold_docs:4} {r.pred_docs:4} "
            f"{r.boundary.precision:5.2f} {r.boundary.recall:5.2f} {r.boundary.f1:5.2f} "
            f"{len(r.false_ocr_pages):4}  {mark}"
        )
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
