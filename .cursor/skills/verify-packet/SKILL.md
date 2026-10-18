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

`launch` runs `pip install -e ".[dev]"` when `packet` is missing, creates `$PACKET_VERIFY_ROOT` (default `/tmp/packet-verify-$PACKET_VERIFY_RUN_ID`), writes an ownership marker under that root, and prints doctor fields. Always `eval "$(.cursor/skills/verify-packet/control-packet env)"` in the same shell so later invocations reuse the persisted `PACKET_VERIFY_RUN_ID`.

**Ready** when `.cursor/skills/verify-packet/control-packet doctor` prints `ok: true` with a resolved `packet` path and `package_version` matching `pyproject.toml`. Doctor also prints `ocr_lighton` (`ready`, `missing-mlx` or `missing-weights`), `ocr_paddle` and `machine`; the OCR recipes need `ocr_lighton: ready`.

Doctor also checks that `import packet` resolves to this checkout. After editable-install fixes (Refs GMA-219), `pip install -e ".[dev]"` should load `src/packet` directly. If an older environment still shadows the checkout with a `site-packages/packet` copy, doctor requires every `.py` file there to match `src/packet` byte-for-byte; any divergence fails doctor — reinstall with `pip install -e ".[dev]"` from the repo root.

**Teardown:** run `.cursor/skills/verify-packet/control-packet cleanup` after each verification session (see Cleanup). There is no daemon to keep alive between drives.

## Doctor

Health check before driving (does not create cache directories; run `launch` first):

```bash
export PATH="${HOME}/.local/bin:${PATH}"
.cursor/skills/verify-packet/control-packet doctor
```

Require `ok: true`, a `packet` binary on PATH (or `~/.local/bin/packet` after install), and `repo_root` pointing at this checkout. If `package_version` is `unknown`, reinstall with `pip install -e ".[dev]"`.

## Drive

Harness: **`control-packet`** in this directory. It wraps the real `packet` entry point (`packet.cli:main`).

Conventions:

- Set `PACKET_VERIFY_RUN_ID` so concurrent agents do not share cache dirs.
- Materialize built-in eval PDFs once per run: `.cursor/skills/verify-packet/control-packet materialize-fixtures` (writes under `$PACKET_VERIFY_CACHE`, default `$PACKET_VERIFY_ROOT/cache`).
- Build an image-only copy of a fixture for OCR checks: `.cursor/skills/verify-packet/control-packet make-scan` (default `multi-doc`; prints the path under `$PACKET_VERIFY_CACHE`).
- Resolve a fixture path: `.cursor/skills/verify-packet/control-packet fixture text-layer` (also `multi-doc`, `shared-header`, `scan`).
- Run CLI commands with captured evidence:

```bash
PACKET_VERIFY_CMD_LABEL=inspect-text-layer \
  .cursor/skills/verify-packet/control-packet cli -- inspect \
  "$(.cursor/skills/verify-packet/control-packet fixture text-layer)"
head -20 "$PACKET_VERIFY_ARTIFACTS/inspect-text-layer.stdout"
```

Each `cli --` invocation writes `$ARTIFACTS_DIR/<label>.{stdout,stderr,exitcode}` and exits with the same code as `packet` (paths are printed on stdout).

Feature recipes live under `features/`. Start from `features/README.md`, drive **every entry point listed** for the feature under test (not only the happiest path).

## Evidence

Default artifacts directory: `$PACKET_VERIFY_ROOT/artifacts` (override with `PACKET_VERIFY_ARTIFACTS`).

Proof standards:

- Exercise the **public CLI** (`packet inspect|parse|eval|eval-plan`), not internal Python imports, unless the feature map explicitly allows a library-only path.
- Capture **command, stdout, stderr, and exit code** for CLI proof (`control-packet cli --` files).
- For `parse`, also keep the **output JSON file** the user asked for (`-o` or `--out-dir/<stem>.json` in batch mode); assert document count and page count in the file match each CLI **stdout** summary line (`wrote … (N document(s), M pages)`). With `--timings`, keep the JSONL file: successful rows include `file`, `ok`, `pages`, `documents`, `seconds`, and `ocr_page_seconds`; failed rows include only `file`, `ok`, `error`, `seconds`, and `ocr_page_seconds` (no `pages` or `documents`).
- For `eval`, require exit code `0`, a `PASS` mark on every builtin fixture row, and a final `GATE PASS` line. With `--json`, stdout is an object `{"fixtures": [...], "gate": {...}}`: require `.gate.passed == true` and `.fixtures[].gate_pass == true` for every fixture.
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
| `.cursor/skills/verify-packet/control-packet make-scan [name]` | Write an image-only copy of a builtin, so only OCR can read it |
| `.cursor/skills/verify-packet/control-packet cli -- <args>` | Run `packet` with captured stdout/stderr/exit; exit code matches `packet` |
| `.cursor/skills/verify-packet/control-packet cleanup` | Drop cache/scratch; keep artifacts |
| `.cursor/skills/verify-packet/control-packet ci-headless` | CI smoke: doctor, fixtures, inspect, parse (incl. `--boundary-weights` fixture), builtin eval `GATE PASS` (no OCR / eval-plan) |

Maintenance: when CLI flags or fixtures change, update this skill and the `features/` recipes to match (optional: pstack `/create-verification-skill` flow if you use it elsewhere).
