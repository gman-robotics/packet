# Run builtin eval

Eval runs the native-text pipeline (no OCR) over a fixture set: the synthetic builtins (default) or `dlnet-whole`, which joins real public PDFs into packets. For each set it regenerates the PDFs if needed, runs the pipeline with gold logical-document boundaries, and prints boundary precision/recall/F1 and false-OCR page counts with a per-fixture PASS/FAIL mark, then a macro-F1 line and an overall `GATE PASS` / `GATE FAIL` (with reasons).

## Sub-features

- `eval-table` human-readable table with `ocr` and `gate` columns, a `macro F1` line, and `GATE PASS`/`GATE FAIL`.
- `eval-dlnet` `--set dlnet-whole`: 13 packets (`dlw-00` to `dlw-12`) built from whole public PDFs that are downloaded to `--data-dir` (default `~/.cache/packet/dlnet-whole`) and checked against pinned sha256 values.
- `eval-politax` `--set politax`: 30 PoliTax-Split PDFs (one fixture each), per-fixture floors 0.0, macro + micro boundary metrics, text-layer census, and holdout heuristic baseline on stdout. Downloads to `--data-dir` (default `~/.cache/packet/politax`).
- `eval-politax-ocr-cache` `--set politax --ocr-cache`: scan pages read from `<data-dir>/ocr` (populated by `packet politax-ocr`; no OCR in eval). Macro gate uses **holdout** F1 with floor `0.41` (GMA-274 Q7); stdout includes `ocr cache (GMA-274 opt-in): pages used …` and `macro holdout F1 …`.
- `eval-json` machine-readable results via `--json`: an object `{"set": "builtin"|"dlnet-whole"|"politax", "fixtures": [per-fixture rows with gate_pass], "gate": {"passed", "failures", "macro_f1", "min_f1", "min_macro_f1", "macro_scope": "all"|"holdout"}}` (not a bare list). For `politax`, also includes `micro_boundary_p|r|f1`, `text_layer_census`, optional `holdout_heuristic`, optional `holdout_scorer` (with `--boundary-weights`), and optional `ocr_cache` stats when `--ocr-cache` is set.
- `eval-cache` uses `--dir` to isolate generated PDFs per verification run.

## How to get to it (user POV)

- Run `packet eval` from the repository (uses default cache `.packet-eval` when `--dir` omitted).
- Run `packet eval --dir /tmp/my-cache` for a disposable cache location.
- Run `packet eval --json` for JSON output.
- Run `packet eval --min-f1 0.9 --min-macro-f1 0.95` to set the F1 floors (builtin defaults: 1.0 and 1.0; dlnet-whole defaults: per-fixture 0.0, macro 0.6).
- Run `packet eval --set dlnet-whole --data-dir ~/.cache/packet/dlnet-whole` for the real-PDF set.
- Run `packet eval --set politax --data-dir ~/.cache/packet/politax` for PoliTax-Split (network on first download).
- Run `packet eval --set politax --ocr-cache --json` after `politax-ocr` has cached scan text under `<data-dir>/ocr`.

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- `$PACKET_VERIFY_CACHE` is empty or owned by this `PACKET_VERIFY_RUN_ID`.

- **Builtin gates.** Run `PACKET_VERIFY_CMD_LABEL=eval-builtin .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE"`. Exit code `0`. Stdout lists fixtures `text-layer`, `multi-doc`, `shared-header`, `scan` each with gate `PASS`, and the last line is `GATE PASS`.
- **JSON.** Run `PACKET_VERIFY_CMD_LABEL=eval-json .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE" --json`. Exit code `0`. Stdout is a JSON object: `.gate.passed` is `true`, `.gate.failures` is empty, and `.fixtures` has all four builtin fixtures, each with `gate_pass: true`.
- **dlnet-whole.** Run `PACKET_VERIFY_CMD_LABEL=eval-dlnet .cursor/skills/verify-packet/control-packet cli -- eval --set dlnet-whole --dir "$PACKET_VERIFY_CACHE/dlnet" --json`. Exit `0`. `.set` is `"dlnet-whole"`, `.gate.passed` is `true`, `.gate.min_macro_f1` is `0.6`, `.gate.macro_f1` is at least `0.6`, and `.fixtures` has 13 rows.
- **politax.** Run `PACKET_VERIFY_CMD_LABEL=eval-politax .cursor/skills/verify-packet/control-packet cli -- eval --set politax --data-dir "$HOME/.cache/packet/politax" --json`. Needs network on an empty cache. Exit `0`. `.set` is `"politax"`, `.gate.min_f1` is `0.0`, `.fixtures` has 30 rows, and `text_layer_census` is present. Without network and cache, report `eval-politax` as blocked (same as `eval-dlnet`).
- **politax ocr-cache.** After `politax-ocr` has populated `<data-dir>/ocr`, run `PACKET_VERIFY_CMD_LABEL=eval-politax-ocr-cache .cursor/skills/verify-packet/control-packet cli -- eval --set politax --ocr-cache --data-dir "$HOME/.cache/packet/politax" --json`. Exit `0`. `.gate.macro_scope` is `"holdout"`, `.gate.min_macro_f1` is `0.41`, and `.gate.passed` is `true`. Without cached OCR text, expect missing-page failures or a gate FAIL — not a headless CI step.
- **Proof.** Append `feature=eval run_id=$PACKET_VERIFY_RUN_ID cache=$PACKET_VERIFY_CACHE` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`. Retain `eval-builtin.stdout` and `eval-builtin.exitcode`.

## Gotchas

- The first `--set dlnet-whole` run downloads the PDFs (a pause between each to be polite to arXiv), so it needs network and takes minutes; later runs reuse `--data-dir` and take about 5 s. A sha256 mismatch stops the run. Without network and an empty cache, report `eval-dlnet` as blocked.
- `dlw-00` scores boundary F1 `1.0` on the current heuristic head (saturated with `dlw-01`); the residual macro drag is mainly `dlw-12`. Per-fixture floor for this set remains `0.0`, so a single low-F1 packet does not fail the gate unless macro F1 drops below `0.6`.

- Without `--dir`, eval writes under `.packet-eval` in the cwd; prefer `$PACKET_VERIFY_CACHE` for agent runs.
- A single FAIL row yields exit code `1` even if other fixtures pass. A row fails on any false-OCR page or a boundary F1 below `--min-f1`; the run also fails if macro F1 is below `--min-macro-f1`.
- `--ocr-cache` requires `--set politax` and cannot combine with `--boundary-weights` (exit `2`). Default politax eval does not read the OCR cache; ship gating for default-on weights uses `packet politax-ship-check` (see [politax-tools.md](./politax-tools.md)).
