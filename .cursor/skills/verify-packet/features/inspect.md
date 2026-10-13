# Inspect a packet

Inspect streams the pipeline event log for each page of a PDF: classification, blocks, and document lifecycle events, printed as human-readable lines on stdout.

## Sub-features

- `inspect-events` prints `page_classified`, `block_emitted`, and boundary-related events for a born-digital PDF.
- `inspect-missing` exits non-zero when the path does not exist.

## How to get to it (user POV)

- Run `packet inspect path/to/file.pdf` in a terminal.

## Driving it with control-packet

Preconditions:

- `control-packet doctor` reports `ok: true`.
- Builtin fixture `text-layer` exists (`control-packet materialize-fixtures`).

- **Stream events.** Run `PACKET_VERIFY_CMD_LABEL=inspect-text-layer .cursor/skills/verify-packet/control-packet cli -- inspect "$(.cursor/skills/verify-packet/control-packet fixture text-layer)"`. Exit code `0`. Stdout contains at least one line starting with `page_classified` and one with `block_emitted`.
- **Missing file.** Run `PACKET_VERIFY_CMD_LABEL=inspect-missing .cursor/skills/verify-packet/control-packet cli -- inspect /tmp/packet-verify-no-such-file.pdf`. Exit code `2`. Stderr contains `not found`.
- **Proof.** Append `feature=inspect fixture=text-layer run_id=$PACKET_VERIFY_RUN_ID` to `$PACKET_VERIFY_ARTIFACTS/proof.txt`. Keep `inspect-text-layer.stdout` and `inspect-text-layer.exitcode`.

## Gotchas

- Large PDFs produce long stdout; use `head` for spot checks but capture full stdout for proof when feasible.
- Scanned fixtures classify as `scan`; native text extraction may emit few blocks until OCR backends are wired (see README status).
