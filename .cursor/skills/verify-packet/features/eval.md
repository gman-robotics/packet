# Run builtin eval

Eval runs the native-text pipeline (no OCR) over a fixture set: the synthetic builtins (default) or `dlnet-whole`, which joins real public PDFs into packets. For each set it regenerates the PDFs if needed, runs the pipeline with gold logical-document boundaries, and prints boundary precision/recall/F1 and false-OCR page counts with a per-fixture PASS/FAIL mark, then a macro-F1 line and an overall `GATE PASS` / `GATE FAIL` (with reasons).

## Sub-features

- `eval-table` human-readable table with `ocr` and `gate` columns, a `macro F1` line, and `GATE PASS`/`GATE FAIL`.
- `eval-dlnet` `--set dlnet-whole`: 13 packets (`dlw-00` to `dlw-12`) built from whole public PDFs that are downloaded to `--data-dir` (default `~/.cache/packet/dlnet-whole`) and checked against pinned sha256 values.
- `eval-politax` `--set politax`: 30 PoliTax-Split PDFs (one fixture each), report-only floors 0.0, macro + micro boundary metrics, text-layer census. Downloads to `--data-dir` (default `~/.cache/packet/politax`).
- `eval-politax-ocr-cache` `--set politax --ocr-cache`: scan pages read from `<data-dir>/ocr` (populated by `packet politax-ocr`; eval does not run OCR). Macro gate uses **holdout** F1 with floor `0.41` (GMA-274 Q7); stdout includes `ocr cache (GMA-274 opt-in): pages used …` and `macro holdout F1 …`.
- `eval-viz-dir` `--viz-dir DIR`: after eval finishes, writes one offline HTML file per fixture under `DIR` (builtins plus fail-mode fixtures such as `stripped-numbers.html`). Metrics and gate stdout are unchanged; viz always uses the default heuristic pipeline (ignores `--boundary-weights` for HTML).
- `eval-json` machine-readable results via `--json`: an object with `"set"` (`"builtin"`, `"dlnet-whole"`, or `"politax"`), `"fixtures"` (per-fixture rows with `gate_pass`), and `"gate"` (`passed`, `failures`, `macro_f1`, `min_f1`, `min_macro_f1`, `macro_scope` always `all` or `holdout`). Not a bare list. For `"politax"`, also expect `micro_boundary_p` / `micro_boundary_r` / `micro_boundary_f1`, `gold_label_totals` (`boundaries`, `sub_documents`), `text_layer_census`, and `holdout_heuristic` plus the C2 alias `holdout_baseline` (same object, native-text heuristic) on every run; `holdout_scorer` with `--boundary-weights` and `ocr_cache` with `--ocr-cache` only.
- `eval-cache` uses `--dir` to isolate generated PDFs per verification run.

## How to get to it (user POV)

- Run `packet eval` from the repository (uses default cache `.packet-eval` when `--dir` omitted).
- Run `packet eval --dir /tmp/my-cache` for a disposable cache location.
- Run `packet eval --json` for JSON output.
- Run `packet eval --min-f1 0.9 --min-macro-f1 0.95` to set the F1 floors (builtin defaults: 1.0 and 1.0; dlnet-whole defaults: per-fixture 0.0, macro 0.6).
- Run `packet eval --set dlnet-whole --data-dir ~/.cache/packet/dlnet-whole` for the real-PDF set.
- Run `packet eval --dir /tmp/cache --viz-dir /tmp/eval-viz` to emit HTML reviews alongside the usual table.
- Run `packet eval --set politax --ocr-cache --json` after `politax-ocr` has cached scan text under `<data-dir>/ocr`.

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- `$PACKET_VERIFY_CACHE` is empty or owned by this `PACKET_VERIFY_RUN_ID`.

- **Builtin gates.** Run `PACKET_VERIFY_CMD_LABEL=eval-builtin .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE"`. Exit code `0`. Stdout lists fixtures `text-layer`, `multi-doc`, `shared-header`, `scan` each with gate `PASS`, and the last line is `GATE PASS`.
- **JSON.** Run `PACKET_VERIFY_CMD_LABEL=eval-json .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE" --json`. Exit code `0`. Stdout is a JSON object: `.gate.passed` is `true`, `.gate.failures` is empty, and `.fixtures` has all four builtin fixtures, each with `gate_pass: true`.
- **dlnet-whole.** Run `PACKET_VERIFY_CMD_LABEL=eval-dlnet .cursor/skills/verify-packet/control-packet cli -- eval --set dlnet-whole --dir "$PACKET_VERIFY_CACHE/dlnet" --json`. Exit `0`. `.set` is `"dlnet-whole"`, `.gate.passed` is `true`, `.gate.min_macro_f1` is `0.6`, `.gate.macro_f1` is at least `0.6`, and `.fixtures` has 13 rows.
- **Viz dir.** Set `VIZ="$PACKET_VERIFY_ARTIFACTS/eval-viz"`. Run `PACKET_VERIFY_CMD_LABEL=eval-viz-dir .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE" --viz-dir "$VIZ"`. Exit `0` with the same stdout gate as **Builtin gates**. Expect HTML files including `multi-doc.html` and `stripped-numbers.html`; open one and confirm offline HTML (no external URLs, no `<script`).
- **PoliTax ocr-cache (optional).** After `politax-ocr` populated `<data-dir>/ocr`, run `PACKET_VERIFY_CMD_LABEL=eval-politax-ocr-cache .cursor/skills/verify-packet/control-packet cli -- eval --set politax --ocr-cache --data-dir "$HOME/.cache/packet/politax" --json`. Exit `0`. `.gate.macro_scope` is `"holdout"`, `.gate.min_macro_f1` is `0.41`, `.gate.passed` is `true`, and `.ocr_cache.pages_missing` is `0`. With real PoliTax PDFs on disk but **no** OCR cache files, `--ocr-cache` still exits `0`: missing pages increment `.ocr_cache.pages_missing`, extraction falls back to native text, and holdout heuristic macro can clear `0.41` (typical full-corpus holdout macro ≈ `0.4159`). Exit `1` with `.gate.passed` false when holdout macro is below `0.41` — for example OCR cached on training PDFs only while holdout stays on native text (`tests/test_politax_ocr_eval.py`), or the blank-PDF unit corpus where holdout macro is `0.0`. Failures use the usual holdout line in `.gate.failures` (`macro (holdout) boundary F1 … < 0.41`), not a separate OCR error string. Not a headless CI step.
- **Proof.** Append `feature=eval run_id=$PACKET_VERIFY_RUN_ID cache=$PACKET_VERIFY_CACHE` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`. Retain `eval-builtin.stdout` and `eval-builtin.exitcode`.

## Gotchas

- The first `--set dlnet-whole` run downloads the PDFs (a pause between each to be polite to arXiv), so it needs network and takes minutes; later runs reuse `--data-dir` and take about 5 s. A sha256 mismatch stops the run. Without network and an empty cache, report `eval-dlnet` as blocked.
- `dlw-00` scores boundary F1 `1.0` on the current heuristic head (saturated with `dlw-01`); the residual macro drag is mainly `dlw-12`. Per-fixture floor for this set remains `0.0`, so a single low-F1 packet does not fail the gate unless macro F1 drops below `0.6`.

- Without `--dir`, eval writes under `.packet-eval` in the cwd; prefer `$PACKET_VERIFY_CACHE` for agent runs.
- A single FAIL row yields exit code `1` even if other fixtures pass. A row fails on any false-OCR page or a boundary F1 below `--min-f1`; the run also fails if macro F1 is below `--min-macro-f1`.
- `--ocr-cache` requires `--set politax` and cannot combine with `--boundary-weights` (exit `2`). Default politax eval does not read the OCR cache; ship gating for default-on weights uses `packet politax-ship-check` (see [politax-tools.md](./politax-tools.md)).
