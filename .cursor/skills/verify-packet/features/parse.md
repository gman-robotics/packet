# Parse to JSON

Parse runs the full pipeline on a PDF and emits a PacketTree as JSON, either to stdout or to a file via `-o`.

## Sub-features

- `parse-stdout` prints indented JSON for a single-document fixture.
- `parse-file` writes JSON to a path and prints a summary line with document and page counts.
- `parse-no-split` forces one logical document when boundary splitting would otherwise apply.
- `parse-batch` parses several PDFs in one process with `--out-dir` (one shared pipeline and OCR model load when OCR runs).
- `parse-timings` appends one JSON line per input file via `--timings` (wall seconds, page count, per-page OCR seconds when applicable).

## How to get to it (user POV)

- Run `packet parse path/to/file.pdf` to print JSON.
- Run `packet parse path/to/file.pdf -o packet.json` to write a file.
- Run `packet parse path/to/file.pdf -o packet.json --no-split` to disable boundary splitting.
- Run `packet parse a.pdf b.pdf --out-dir out/` to write `out/a.json` and `out/b.json` in one run.
- Run `packet parse a.pdf b.pdf --out-dir out/ --timings run.jsonl` to record per-file timing rows.

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- Fixture `multi-doc` materialized (expects **two** logical documents by default).

- **Write JSON file.** Set `OUT="$PACKET_VERIFY_ROOT/scratch/multi-doc.json"` and `mkdir -p "$(dirname "$OUT")"`. Run `PACKET_VERIFY_CMD_LABEL=parse-multi-doc .cursor/skills/verify-packet/control-packet cli -- parse "$(.cursor/skills/verify-packet/control-packet fixture multi-doc)" -o "$OUT"`. Exit code `0`. The summary line is on **stdout only** (captured in `parse-multi-doc.stdout`) and mentions `2 document(s)` and `4 pages`. File `$OUT` contains a top-level `"documents"` array with length 2.
- **No-split.** Run the same with `--no-split` and assert the summary reports `1 document(s)` when boundary splitting is disabled.
- **Batch.** Set `BATCH="$PACKET_VERIFY_ROOT/scratch/batch-out"` and `mkdir -p "$BATCH"`. Run `PACKET_VERIFY_CMD_LABEL=parse-batch .cursor/skills/verify-packet/control-packet cli -- parse "$(.cursor/skills/verify-packet/control-packet fixture text-layer)" "$(.cursor/skills/verify-packet/control-packet fixture multi-doc)" --out-dir "$BATCH" --timings "$PACKET_VERIFY_ARTIFACTS/parse-batch.jsonl"`. Exit code `0`. Stdout has two `wrote …` lines (one document / three pages, then two documents / four pages). `$BATCH/text-layer.json` and `$BATCH/multi-doc.json` exist. Each successful `--timings` line is one JSON object with `file`, `ok`, `pages`, `documents`, `seconds`, and `ocr_page_seconds`. Mixed pages still go through OCR (`EscalatingExtractor`); with default env the legacy `FallbackExtractor` tries Paddle only when LightOn cannot run or **raises** (an empty LightOn result is final). `ocr_page_seconds` stays `[]` when no OCR timing row was recorded for that file—common when `doctor` reports `ocr_lighton: missing-mlx` and `ocr_paddle: missing`, or when `PACKET_OCR=off`. When at least one engine succeeds, mixed pages (for example `text-layer` page 2, `multi-doc` page 3) get per-page seconds in `ocr_page_seconds` (check `ocr_lighton` and `ocr_paddle` separately in doctor). `PACKET_OCR=auto|lighton|paddle` selects a single engine or auto-routing per [ocr.md](./ocr.md); a bad factory config aborts the whole run with exit `2`.
- **Batch errors.** `packet parse a.pdf b.pdf` without `--out-dir` exits `2` with `several PDFs need --out-dir` on stderr. Several inputs that share the same basename exit `2` before parsing starts.
- **Proof.** Copy `$OUT` to `$PACKET_VERIFY_ARTIFACTS/parse-multi-doc.json`. Append `feature=parse fixture=multi-doc run_id=$PACKET_VERIFY_RUN_ID` to `proof.txt`. For batch, copy `$BATCH/*.json` into `$PACKET_VERIFY_ARTIFACTS` (same as the single-file JSON), and keep `parse-batch.jsonl` and `parse-batch.stdout`.

## Gotchas

- Default pipeline splits on boundary signals; use `multi-doc` when asserting multiple documents, `text-layer` for a single document.
- `-o -` means stdout; file side effects are then the captured stdout artifact only.
- With several PDFs, `-o` is only valid for a single input; use `--out-dir` instead.
- In a multi-file run, one bad file is logged on stderr and skipped (exit `1` if any failed); a corrupt or unreadable file writes a `--timings` row with `ok: false` when `--timings` is set. A **missing** path for a **single** input exits `2` with `not found: …` on stderr and writes **no** timings row (the CLI returns before the parse loop).
