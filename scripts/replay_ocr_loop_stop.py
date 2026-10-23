"""Read-only replay of cached politax-ocr pages against the zlib loop rule.

Walks page JSON under the OCR cache and checks that non-truncated ``raw`` text
does not trip ``LoopRule`` (last 1,500 characters, zlib level 9, ratio under
0.02). Truncated pages that would trip are counted and do not fail the run.
This script does not write the cache and does not load MLX.

Usage:
    python scripts/replay_ocr_loop_stop.py
    python scripts/replay_ocr_loop_stop.py --data-dir ~/.cache/packet/politax/ocr

The default directory is the politax-ocr cache root
(``~/.cache/packet/politax/ocr``), one level under the politax data dir. Exit
status is 0 when no non-truncated page fires, and 1 when one or more do. A
missing cache directory prints a note and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from packet.backends import LoopRule, _loop_fired


def _page_files(root: Path):
    for path in sorted(root.rglob("p*.json")):
        if "_runs" in path.parts:
            continue
        stem = path.stem
        if len(stem) > 1 and stem[0] == "p" and stem[1:].isdigit():
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert cached politax-ocr pages do not trip the zlib loop rule.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "packet" / "politax" / "ocr",
        help="OCR cache root (default: ~/.cache/packet/politax/ocr)",
    )
    args = parser.parse_args(argv)
    root = args.data_dir.expanduser()
    if not root.is_dir():
        print(f"cache dir not found: {root} (nothing to check)")
        return 0
    rule = LoopRule()
    pages = checked = fired = truncated_pages = truncated_would = 0
    failures: list[Path] = []
    for path in _page_files(root):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"skip unreadable {path}", file=sys.stderr)
            continue
        raw = rec.get("raw") if isinstance(rec, dict) else None
        if not isinstance(raw, str):
            continue
        pages += 1
        hit = _loop_fired(raw, rule)
        if rec.get("truncated") is True:
            truncated_pages += 1
            truncated_would += int(hit)
            continue
        checked += 1
        if hit:
            fired += 1
            failures.append(path)
    print(
        f"pages {pages} checked {checked} fired {fired} "
        f"truncated {truncated_pages} truncated_would_fire {truncated_would}"
    )
    for path in failures:
        print(f"fired {path}", file=sys.stderr)
    return 1 if fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
