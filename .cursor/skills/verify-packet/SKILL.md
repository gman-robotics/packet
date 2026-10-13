---
name: verify-packet
description: "Drive the Packet document-understanding CLI (inspect, parse, eval) from a clean install, capture terminal evidence, and tear down scratch caches. Use when proving Packet pipeline or CLI behavior after a change."
---

# Verify Packet

Packet is a **CLI** (`packet`), not a long-running server. Verification means an editable install, disposable PDF fixtures under a per-run cache directory, driving subcommands the way a user would, and keeping proof transcripts outside cleanup paths.

## Launch

From the repository root:

```bash
export PACKET_VERIFY_RUN_ID="${PACKET_VERIFY_RUN_ID:-$(date +%s)-$$}"
export PATH="${HOME}/.local/bin:${PATH}"
.cursor/skills/verify-packet/control-packet launch
eval "$(.cursor/skills/verify-packet/control-packet env)"
```

`launch` runs `pip install -e ".[dev]"` when `packet` is missing, creates `$PACKET_VERIFY_ROOT` (default `/tmp/packet-verify-$PACKET_VERIFY_RUN_ID`), and prints doctor fields.

**Ready** when `control-packet doctor` prints `ok: true` with a resolved `packet` path and `package_version` matching `pyproject.toml`.

**Teardown:** run `control-packet cleanup` after each verification session (see Cleanup). There is no daemon to keep alive between drives.

## Doctor

Read-only health check before driving:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
.cursor/skills/verify-packet/control-packet doctor
```

Require `ok: true`, a `packet` binary on PATH (or `~/.local/bin/packet` after install), and `repo_root` pointing at this checkout. If `package_version` is `unknown`, reinstall with `pip install -e ".[dev]"`.

## Drive

Harness: **`control-packet`** in this directory. It wraps the real `packet` entry point (`packet.cli:main`).

Conventions:

- Set `PACKET_VERIFY_RUN_ID` so concurrent agents do not share cache dirs.
- Materialize built-in eval PDFs once per run: `control-packet materialize-fixtures` (writes under `$PACKET_VERIFY_CACHE`, default `$PACKET_VERIFY_ROOT/cache`).
- Resolve a fixture path: `control-packet fixture text-layer` (also `multi-doc`, `shared-header`, `scan`).
- Run CLI commands with captured evidence:

```bash
PACKET_VERIFY_CMD_LABEL=inspect-text-layer \
  .cursor/skills/verify-packet/control-packet cli -- inspect "$(control-packet fixture text-layer)" | head -20
```

Each `cli --` invocation writes `$ARTIFACTS_DIR/<label>.{stdout,stderr,exitcode}`.

Feature recipes live under `features/`. Start from `features/README.md`, drive **every entry point listed** for the feature under test (not only the happiest path).

## Evidence

Default artifacts directory: `$PACKET_VERIFY_ROOT/artifacts` (override with `PACKET_VERIFY_ARTIFACTS`).

Proof standards:

- Exercise the **public CLI** (`packet inspect|parse|eval|eval-plan`), not internal Python imports, unless the feature map explicitly allows a library-only path.
- Capture **command, stdout, stderr, and exit code** for CLI proof (`control-packet cli --` files).
- For `parse`, also keep the **output JSON file** the user asked for (`-o`); assert document count and page count in the file match CLI stderr summary when written to a path.
- For `eval`, require exit code `0` and a `PASS` gate for every builtin fixture row (or JSON `passed` equivalents with `--json`).
- Record `feature_id` and the exact fixture or input path in a one-line `proof.txt` beside the captures:

```bash
echo "feature=eval run_id=$PACKET_VERIFY_RUN_ID" >> "$PACKET_VERIFY_ARTIFACTS/proof.txt"
```

Mocks: built-in eval PDFs are **synthetic fixtures** generated locally; that is the intended boundary for builtin gates. DocLayNet `eval-plan` needs an external catalog file (not shipped in-repo).

## Cleanup

```bash
.cursor/skills/verify-packet/control-packet cleanup
```

Removes `$PACKET_VERIFY_CACHE` and `$PACKET_VERIFY_ROOT/scratch` only. **Does not** delete `$PACKET_VERIFY_ARTIFACTS`. Never `pkill packet` — there is no persistent process.

Optional: `PACKET_VERIFY_DELETE_ROOT=1` removes everything under `$PACKET_VERIFY_ROOT` except the `artifacts/` subtree.

After cleanup, confirm proof files still exist:

```bash
test -f "$PACKET_VERIFY_ARTIFACTS/proof.txt" && ls -la "$PACKET_VERIFY_ARTIFACTS"
```

## Helpers

| Command | Purpose |
|--------|---------|
| `.cursor/skills/verify-packet/control-packet launch` | Install if needed, create dirs, run doctor |
| `.cursor/skills/verify-packet/control-packet env` | Print `export` lines for verify paths (eval in shell) |
| `.cursor/skills/verify-packet/control-packet doctor` | Read-only readiness YAML |
| `.cursor/skills/verify-packet/control-packet materialize-fixtures` | Run `packet eval --dir $CACHE` to build PDFs |
| `.cursor/skills/verify-packet/control-packet fixture <name>` | Print path to a builtin PDF |
| `.cursor/skills/verify-packet/control-packet cli -- <args>` | Run `packet` with captured stdout/stderr/exit |
| `.cursor/skills/verify-packet/control-packet cleanup` | Drop cache/scratch; keep artifacts |

Maintenance: run `/maintain-verification-skill` when CLI flags or fixtures change.
