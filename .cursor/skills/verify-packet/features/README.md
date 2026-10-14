# Packet verification map

This directory is the maintained source for verifying user-facing **Packet CLI** behavior. Read the index before driving commands, then open the matching feature file for the recipe.

## Baseline preconditions

- Repository checkout with `pyproject.toml` and `src/packet/`.
- `export PACKET_VERIFY_RUN_ID` to a unique value per verification run.
- `export PATH="${HOME}/.local/bin:${PATH}"`.
- Run `.cursor/skills/verify-packet/control-packet launch` from the repo root.
- Run `.cursor/skills/verify-packet/control-packet doctor` and require `ok: true`.
- Materialize builtin PDFs: `.cursor/skills/verify-packet/control-packet materialize-fixtures`.
- Never assert against PDFs in the git tree; builtins are generated under `$PACKET_VERIFY_CACHE/builtin/`.

## Driving conventions

- Start every recipe from the baseline unless its feature file says otherwise.
- Pass PDF paths as positional arguments exactly as the CLI expects.
- Use `.cursor/skills/verify-packet/control-packet cli --` for commands whose output is proof; set `PACKET_VERIFY_CMD_LABEL` to a stable basename.
- Use `.cursor/skills/verify-packet/control-packet fixture <name>` to resolve builtin PDF paths (`text-layer`, `multi-doc`, `shared-header`, `scan`).
- Run `.cursor/skills/verify-packet/control-packet cleanup` when finished; do not delete files under `$PACKET_VERIFY_ARTIFACTS`.

## Proof and skip reporting

- CLI proof includes stdout, stderr, exit code, and any `-o` output file named in the recipe.
- Include `feature_id` and input path in `$PACKET_VERIFY_ARTIFACTS/proof.txt`.
- If a feature needs data that is not in-repo (DocLayNet catalog for `eval-plan`), report the missing precondition in Gotchas — do not mark the feature verified.

## Feature entry contract

Each feature file has an H1 title, one introductory paragraph, then exactly four H2 sections in order: `Sub-features`, `How to get to it (user POV)`, `Driving it with control-packet`, `Gotchas`.

## Features

- [Inspect a packet](./inspect.md) — stream page classification and pipeline events for a PDF.
- [Parse to JSON](./parse.md) — build a PacketTree and write JSON (stdout or `-o`).
- [Run builtin eval](./eval.md) — boundary metrics on synthetic fixtures with pass/fail gates.
- [Plan DocLayNet eval packets](./eval-plan.md) — generate a manifest and placeholder PDFs from an external catalog.
