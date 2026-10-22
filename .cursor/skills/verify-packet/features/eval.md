# Run builtin eval

Eval runs the native-text pipeline (no OCR) over a fixture set: the synthetic builtins (default) or `dlnet-whole`, which joins real public PDFs into packets. For each set it regenerates the PDFs if needed, runs the pipeline with gold logical-document boundaries, and prints boundary precision/recall/F1 and false-OCR page counts with a per-fixture PASS/FAIL mark, then a macro-F1 line and an overall `GATE PASS` / `GATE FAIL` (with reasons).

## Sub-features

- `eval-table` human-readable table with `ocr` and `gate` columns, a `macro F1` line, and `GATE PASS`/`GATE FAIL`.
- `eval-dlnet` `--set dlnet-whole`: 13 packets (`dlw-00` to `dlw-12`) built from whole public PDFs that are downloaded to `--data-dir` (default `~/.cache/packet/dlnet-whole`) and checked against pinned sha256 values.
- `eval-politax` `--set politax`: 30 PoliTax-Split PDFs (one fixture each), report-only floors 0.0, macro + micro boundary metrics, text-layer census. Downloads to `--data-dir` (default `~/.cache/packet/politax`).
- `eval-json` machine-readable results via `--json`: an object `{"set": "builtin"|"dlnet-whole", "fixtures": [per-fixture rows with gate_pass], "gate": {"passed", "failures", "macro_f1", "min_f1", "min_macro_f1"}}` (not a bare list).
- `eval-cache` uses `--dir` to isolate generated PDFs per verification run.

## How to get to it (user POV)

- Run `packet eval` from the repository (uses default cache `.packet-eval` when `--dir` omitted).
- Run `packet eval --dir /tmp/my-cache` for a disposable cache location.
- Run `packet eval --json` for JSON output.
- Run `packet eval --min-f1 0.9 --min-macro-f1 0.95` to set the F1 floors (builtin defaults: 1.0 and 1.0; dlnet-whole defaults: per-fixture 0.0, macro 0.6).
- Run `packet eval --set dlnet-whole --data-dir ~/.cache/packet/dlnet-whole` for the real-PDF set.

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- `$PACKET_VERIFY_CACHE` is empty or owned by this `PACKET_VERIFY_RUN_ID`.

- **Builtin gates.** Run `PACKET_VERIFY_CMD_LABEL=eval-builtin .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE"`. Exit code `0`. Stdout lists fixtures `text-layer`, `multi-doc`, `shared-header`, `scan` each with gate `PASS`, and the last line is `GATE PASS`.
- **JSON.** Run `PACKET_VERIFY_CMD_LABEL=eval-json .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE" --json`. Exit code `0`. Stdout is a JSON object: `.gate.passed` is `true`, `.gate.failures` is empty, and `.fixtures` has all four builtin fixtures, each with `gate_pass: true`.
- **dlnet-whole.** Run `PACKET_VERIFY_CMD_LABEL=eval-dlnet .cursor/skills/verify-packet/control-packet cli -- eval --set dlnet-whole --dir "$PACKET_VERIFY_CACHE/dlnet" --json`. Exit `0`. `.set` is `"dlnet-whole"`, `.gate.passed` is `true`, `.gate.min_macro_f1` is `0.6`, `.gate.macro_f1` is at least `0.6`, and `.fixtures` has 13 rows.
- **Proof.** Append `feature=eval run_id=$PACKET_VERIFY_RUN_ID cache=$PACKET_VERIFY_CACHE` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`. Retain `eval-builtin.stdout` and `eval-builtin.exitcode`.

## Gotchas

- The first `--set dlnet-whole` run downloads the PDFs (a pause between each to be polite to arXiv), so it needs network and takes minutes; later runs reuse `--data-dir` and take about 5 s. A sha256 mismatch stops the run. Without network and an empty cache, report `eval-dlnet` as blocked.
- `dlw-00` scores boundary F1 `1.0` on the current heuristic head (saturated with `dlw-01`); the residual macro drag is mainly `dlw-12`. Per-fixture floor for this set remains `0.0`, so a single low-F1 packet does not fail the gate unless macro F1 drops below `0.6`.

- Without `--dir`, eval writes under `.packet-eval` in the cwd; prefer `$PACKET_VERIFY_CACHE` for agent runs.
- A single FAIL row yields exit code `1` even if other fixtures pass. A row fails on any false-OCR page or a boundary F1 below `--min-f1`; the run also fails if macro F1 is below `--min-macro-f1`.
