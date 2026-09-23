from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from packet.pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packet", description="Document language understanding")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="classify pages and print events")
    inspect.add_argument("path")

    parse = sub.add_parser("parse", help="write a PacketTree JSON file")
    parse.add_argument("path")
    parse.add_argument("-o", "--output", default="-")
    parse.add_argument("--no-split", action="store_true", help="force a single logical document")

    args = parser.parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    pipe = Pipeline()
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


if __name__ == "__main__":
    raise SystemExit(main())
