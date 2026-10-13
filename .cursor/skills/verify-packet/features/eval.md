# Run builtin eval

Eval regenerates synthetic builtin PDFs (if needed), runs the pipeline with gold logical-document boundaries, and prints boundary precision/recall with a per-fixture PASS/FAIL gate.

## Sub-features

- `eval-table` human-readable table with gate column.
- `eval-json` machine-readable results via `--json`.
- `eval-cache` uses `--dir` to isolate generated PDFs per verification run.

## How to get to it (user POV)

- Run `packet eval` from the repository (uses default cache `.packet-eval` when `--dir` omitted).
- Run `packet eval --dir /tmp/my-cache` for a disposable cache location.
- Run `packet eval --json` for JSON output.

## Driving it with control-packet

Preconditions:

- `.cursor/skills/verify-packet/control-packet doctor` reports `ok: true`.
- `$PACKET_VERIFY_CACHE` is empty or owned by this `PACKET_VERIFY_RUN_ID`.

- **Builtin gates.** Run `PACKET_VERIFY_CMD_LABEL=eval-builtin .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE"`. Exit code `0`. Stdout lists fixtures `text-layer`, `multi-doc`, `shared-header`, `scan` each with gate `PASS`.
- **JSON.** Run `PACKET_VERIFY_CMD_LABEL=eval-json .cursor/skills/verify-packet/control-packet cli -- eval --dir "$PACKET_VERIFY_CACHE" --json`. Exit code `0`. Stdout JSON includes all fixtures with passing gates.
- **Proof.** Append `feature=eval run_id=$PACKET_VERIFY_RUN_ID cache=$PACKET_VERIFY_CACHE` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`. Retain `eval-builtin.stdout` and `eval-builtin.exitcode`.

## Gotchas

- Without `--dir`, eval writes under `.packet-eval` in the cwd; prefer `$PACKET_VERIFY_CACHE` for agent runs.
- A single FAIL row yields exit code `1` even if other fixtures pass.
