# Run builtin eval

Eval regenerates synthetic builtin PDFs (if needed), runs the pipeline with gold logical-document boundaries, and prints boundary precision/recall/F1 and false-OCR page counts with a per-fixture PASS/FAIL mark, then a macro-F1 line and an overall `GATE PASS` / `GATE FAIL` (with reasons).

## Sub-features

- `eval-table` human-readable table with `ocr` and `gate` columns, a `macro F1` line, and `GATE PASS`/`GATE FAIL`.
- `eval-json` machine-readable results via `--json`: an object `{"fixtures": [per-fixture rows with gate_pass], "gate": {"passed", "failures", "macro_f1", "min_f1", "min_macro_f1"}}` (not a bare list).
- `eval-cache` uses `--dir` to isolate generated PDFs per verification run.

## How to get to it (user POV)

- Run `packet eval` from the repository (uses default cache `.packet-eval` when `--dir` omitted).
- Run `packet eval --dir /tmp/my-cache` for a disposable cache location.
- Run `packet eval --json` for JSON output.
- Run `packet eval --min-f1 0.9 --min-macro-f1 0.95` to set the F1 floors (defaults 1.0; macro defaults to `--min-f1`).

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- `$PACKET_VERIFY_CACHE` is empty or owned by this `PACKET_VERIFY_RUN_ID`.

- **Builtin gates.** Run `PACKET_VERIFY_CMD_LABEL=eval-builtin .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE"`. Exit code `0`. Stdout lists fixtures `text-layer`, `multi-doc`, `shared-header`, `scan` each with gate `PASS`, and the last line is `GATE PASS`.
- **JSON.** Run `PACKET_VERIFY_CMD_LABEL=eval-json .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE" --json`. Exit code `0`. Stdout is a JSON object: `.gate.passed` is `true`, `.gate.failures` is empty, and `.fixtures` has all four builtin fixtures, each with `gate_pass: true`.
- **Proof.** Append `feature=eval run_id=$PACKET_VERIFY_RUN_ID cache=$PACKET_VERIFY_CACHE` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`. Retain `eval-builtin.stdout` and `eval-builtin.exitcode`.

## Gotchas

- Without `--dir`, eval writes under `.packet-eval` in the cwd; prefer `$PACKET_VERIFY_CACHE` for agent runs.
- A single FAIL row yields exit code `1` even if other fixtures pass. A row fails on any false-OCR page or a boundary F1 below `--min-f1`; the run also fails if macro F1 is below `--min-macro-f1`.
