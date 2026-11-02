# PoliTax tooling (train, OCR cache, ship gate)

Three CLI entry points support the PoliTax-Split eval set outside the builtin `packet eval` headless path: training exported logistic weights, optionally caching LightOn OCR text for scan pages, and running the default-on ship gate (GMA-291).

## Sub-features

- `politax-train-weights` fits schema-1 logistic boundary weights on **trump** training PDFs only and writes JSON via `-o`.
- `politax-ocr` (optional; needs `mlx-vlm` plus LightOn **model weights** on disk for a full run) renders PoliTax scan pages and writes cached text under `<data-dir>/ocr` for later `packet eval --set politax --ocr-cache`. `--dry-run` and `--report` return before `mlx_vlm` is imported (dry-run still fetches PoliTax PDFs and lists pending pages).
- `politax-ship-check` evaluates the ship gate against existing weights or freshly trains trump weights when `--weights` is omitted; needs PoliTax and dlnet-whole caches.

## How to get to it (user POV)

- Run `packet politax-train-weights -o /tmp/weights.json` after PoliTax PDFs are in the cache.
- Run `packet politax-ocr --dry-run` to see pending scan pages, then run without `--dry-run` to fill `<data-dir>/ocr`.
- Run `packet politax-ship-check --weights path/to/weights.json`, or omit `--weights` to train into `--train-output` (default `/tmp/packet-politax-ship-weights.json`).

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- `DATA="${DATA:-$HOME/.cache/packet/politax}"` and `DLNET="${DLNET:-$HOME/.cache/packet/dlnet-whole}"`.
- PoliTax PDFs under `$DATA` and dlnet-whole PDFs under `$DLNET` (first fetch for each needs network). If either cache is missing or empty, report the whole feature as **blocked** — do not treat partial runs as proof.

- **Train weights.** Set `OUT="$PACKET_VERIFY_ROOT/scratch/politax-weights.json"` and `mkdir -p "$(dirname "$OUT")"`. Run `PACKET_VERIFY_CMD_LABEL=politax-train-weights .cursor/skills/verify-packet/control-packet cli -- politax-train-weights --data-dir "$DATA" -o "$OUT"`. Exit `0`. Stdout matches `wrote … trump PDFs`.
- **OCR dry-run.** Run `PACKET_VERIFY_CMD_LABEL=politax-ocr-dry .cursor/skills/verify-packet/control-packet cli -- politax-ocr --data-dir "$DATA" --dry-run`. Exit `0` without `ocr-mlx` when only listing pages. Full OCR run needs `doctor` `ocr_lighton: ready` (not headless CI).
- **Ship check.** With both caches populated, run `politax-ship-check` with trained weights from **Train weights** above — not `tests/fixtures/decision/logistic_weights.json` (all-zero weights give p=0.5 everywhere, so holdout scorer-on cannot beat scorer-off). Example: `PACKET_VERIFY_CMD_LABEL=politax-ship-check .cursor/skills/verify-packet/control-packet cli -- politax-ship-check --weights "$OUT" --data-dir "$DATA" --dlnet-cache "$DLNET"`. Or omit `--weights` to train trump weights to `--train-output` (default `/tmp/packet-politax-ship-weights.json`) inside the ship command. Exit `0` when the gate passes and `1` when it fails; a missing or invalid `--weights` file exits `2` with a single `packet:` line on stderr before the checklist runs. **Text stdout:** `ship=pass outcome (a)` or `ship=FAIL outcome (b)`, numbered lines `N. [PASS] …` / `N. [FAIL] …`, then `max holdout model p=…`. **JSON stdout (`--json`):** object with `"ship"`, `"outcome"`, and `"items"` (each item has `number`, `title`, `passed`, `detail`); use `.ship === true` and every item `.passed === true` for proof — do not assume exit `0` without reading the report when weights load failed (exit `2`).
- **Proof.** When preconditions are met, ship exits `0`, and stdout (or `--json`) shows pass, append `feature=politax-tools data_dir=$DATA run_id=$PACKET_VERIFY_RUN_ID` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`.

## Gotchas

- `politax-ocr` and full `politax-ship-check` are intentionally outside CI; builtin `ci-headless` stays on synthetic fixtures only.
- Ship check item 4 calls `run_dlnet_whole`; populate `$DLNET` first (e.g. `packet eval --set dlnet-whole --data-dir "$DLNET"`). An empty cache often aborts with an uncaught network/traceback from `wholedoc.fetch`, not a numbered `[FAIL]` line.
- Without PoliTax data on disk, report each recipe as blocked rather than skipping verification silently.
