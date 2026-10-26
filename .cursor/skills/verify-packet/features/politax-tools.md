# PoliTax tooling (train, OCR cache, ship gate)

Three CLI entry points support the PoliTax-Split eval set outside the builtin `packet eval` headless path: training exported logistic weights, optionally caching LightOn OCR text for scan pages, and running the default-on ship gate (GMA-291).

## Sub-features

- `politax-train-weights` fits schema-1 logistic boundary weights on **trump** training PDFs only and writes JSON via `-o`.
- `politax-ocr` (optional, MLX + weights) renders PoliTax scan pages and writes cached text under `<data-dir>/ocr` for later `packet eval --set politax --ocr-cache`. Supports `--dry-run` and `--report` without loading the model.
- `politax-ship-check` evaluates the ship gate against existing weights or freshly trains trump weights to `--train-output`; needs PoliTax and dlnet-whole caches.

## How to get to it (user POV)

- Run `packet politax-train-weights -o /tmp/weights.json` after PoliTax PDFs are in the cache.
- Run `packet politax-ocr --dry-run` to see pending scan pages, then run without `--dry-run` to fill `<data-dir>/ocr`.
- Run `packet politax-ship-check --weights path/to/weights.json` (or omit `--weights` to train into `/tmp/packet-politax-ship-weights.json`).

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- PoliTax manifest PDFs fetched under `$DATA` (default `~/.cache/packet/politax`); first fetch needs network.

- **Train weights.** Set `OUT="$PACKET_VERIFY_ROOT/scratch/politax-weights.json"` and `mkdir -p "$(dirname "$OUT")"`. Run `PACKET_VERIFY_CMD_LABEL=politax-train-weights .cursor/skills/verify-packet/control-packet cli -- politax-train-weights --data-dir "$DATA" -o "$OUT"`. Exit `0`. Stdout matches `wrote … trump PDFs`.
- **OCR dry-run.** Run `PACKET_VERIFY_CMD_LABEL=politax-ocr-dry .cursor/skills/verify-packet/control-packet cli -- politax-ocr --data-dir "$DATA" --dry-run`. Exit `0` without `ocr-mlx` when only listing pages. Full OCR run needs `doctor` `ocr_lighton: ready` (not headless CI).
- **Ship check.** With committed fixture `tests/fixtures/decision/logistic_weights.json`, run `PACKET_VERIFY_CMD_LABEL=politax-ship-check .cursor/skills/verify-packet/control-packet cli -- politax-ship-check --weights tests/fixtures/decision/logistic_weights.json --data-dir "$DATA" --dlnet-cache "$HOME/.cache/packet/dlnet-whole"`. Exit `0`. Stdout contains `ship=pass` and numbered `PASS` lines (needs both caches populated). Use `--json` for `report.as_dict()`.
- **Proof.** When preconditions are met, append `feature=politax-tools data_dir=$DATA run_id=$PACKET_VERIFY_RUN_ID` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`.

## Gotchas

- `politax-ocr` and full `politax-ship-check` are intentionally outside CI; builtin `ci-headless` stays on synthetic fixtures only.
- Ship check item 4 touches dlnet-whole; an empty dlnet cache fails or blocks until `packet eval --set dlnet-whole` has run once.
- Without PoliTax data on disk, report each recipe as blocked rather than skipping verification silently.
