# Plan DocLayNet eval packets

Eval-plan reads a DocLayNet-style catalog on disk and writes a manifest plus placeholder packet PDFs for offline eval layout work.

## Sub-features

- `eval-plan-manifest` writes `manifest.json` under an output directory.
- `eval-plan-placeholders` materializes one PDF per planned packet.

## How to get to it (user POV)

- Run `packet eval-plan --catalog path/to/catalog.json --out evals/generated` with optional `--slice`, `--split`, and `--count`.

## Driving it with control-packet

Preconditions:

- `control-packet doctor` reports `ok: true`.
- A DocLayNet catalog JSON file is available on disk (not vendored in this repository).

- **Plan when catalog exists.** Run `PACKET_VERIFY_CMD_LABEL=eval-plan .cursor/skills/verify-packet/control-packet cli -- eval-plan --catalog "$CATALOG" --out "$PACKET_VERIFY_ROOT/scratch/generated" --count 2`. Exit code `0`. Stdout contains `planned`. Output directory contains `manifest.json` and packet subfolders.
- **Proof.** When preconditions are met, append `feature=eval-plan catalog=$CATALOG` to `proof.txt` and keep stdout capture.

## Gotchas

- There is no committed catalog in-repo; verification of this feature is **blocked** until the operator supplies `CATALOG`. Report skip rather than fabricating catalog data.
- Default output path `evals/generated` is gitignored policy; use `$PACKET_VERIFY_ROOT/scratch` for agent runs.
