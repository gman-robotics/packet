# Parse to JSON

Parse runs the full pipeline on a PDF and emits a PacketTree as JSON, either to stdout or to a file via `-o`.

## Sub-features

- `parse-stdout` prints indented JSON for a single-document fixture.
- `parse-file` writes JSON to a path and prints a summary line with document and page counts.
- `parse-no-split` forces one logical document when boundary splitting would otherwise apply.

## How to get to it (user POV)

- Run `packet parse path/to/file.pdf` to print JSON.
- Run `packet parse path/to/file.pdf -o packet.json` to write a file.
- Run `packet parse path/to/file.pdf -o packet.json --no-split` to disable boundary splitting.

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- Fixture `multi-doc` materialized (expects **two** logical documents by default).

- **Write JSON file.** Set `OUT="$PACKET_VERIFY_ROOT/scratch/multi-doc.json"` and `mkdir -p "$(dirname "$OUT")"`. Run `PACKET_VERIFY_CMD_LABEL=parse-multi-doc .cursor/skills/verify-packet/control-packet cli -- parse "$(.cursor/skills/verify-packet/control-packet fixture multi-doc)" -o "$OUT"`. Exit code `0`. The summary line is on **stdout only** (captured in `parse-multi-doc.stdout`) and mentions `2 document(s)` and `4 pages`. File `$OUT` contains a top-level `"documents"` array with length 2.
- **No-split.** Run the same with `--no-split` and assert the summary reports `1 document(s)` when boundary splitting is disabled.
- **Proof.** Copy or reference `$OUT` under `$PACKET_VERIFY_ARTIFACTS/parse-multi-doc.json`. Append `feature=parse fixture=multi-doc run_id=$PACKET_VERIFY_RUN_ID` to `proof.txt`.

## Gotchas

- Default pipeline splits on boundary signals; use `multi-doc` when asserting multiple documents, `text-layer` for a single document.
- `-o -` means stdout; file side effects are then the captured stdout artifact only.
