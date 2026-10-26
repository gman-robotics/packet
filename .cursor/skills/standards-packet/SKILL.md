---
name: standards-packet
description: "gman-robotics/packet (packet-dl): PR/issue shape, planning, implementation order, and CI-aligned verification. Load on every agent session before opening or reviewing PRs."
---

# standards-packet

Repository **Packet** (PyPI-style name `packet-dl`, CLI `packet`). Skill id is `standards-packet` (not hub default `standards-packet-dl` from `pyproject.toml` `[project].name`). Harness skills live under `.cursor/skills/`; verification harness is `verify-packet` at `.cursor/skills/verify-packet/SKILL.md`.

## PR format

Every pull request description uses this **header order** (values on the same line after the label):

1. `Tier:` — T0–T3 (table below)
2. `Models:` — implementer and reviewer model ids used for the change
3. `Linear:` — full `https://linear.app/...` issue URL, or `N/A`
4. `Summary:` — what changed and why (1–3 sentences)
5. `Gates:` — advisory only (see Verification); report local/CI command outcomes and optional unclebob tool results

| Tier | Meaning |
| --- | --- |
| **T0** | Small change, docs-only, or check-only (hub Markdown, typos, config tweak) |
| **T1** | Medium feature or fix in an existing module |
| **T2** | Larger feature, new module, or cross-cutting behavior change |
| **T3** | Infra, migration, or new service / major architectural move |

After the five headers, add a **Verification** subsection listing commands you ran and their exit codes, aligned with the CI job `verify` in `.github/workflows/ci.yml`. Use checklist items only when they map to real checks.

Example header block (illustrative values):

```
Tier: T1
Models: composer-2.5 implementer, grok-4.7 review
Linear: N/A
Summary: Fix boundary scorer tie-break when page numbers repeat across documents.
Gates: advisory — ruff 0, pytest 0, packet eval GATE PASS, verify-packet ci-headless ok; dryer/crapper skipped (not on PATH)
```

## Issue format

GitHub issue forms under `.github/ISSUE_TEMPLATE/` expect:

- **Problem** — observable failure or missing capability
- **Acceptance criteria** — testable bullets (name pytest files or `packet eval` expectations when relevant)
- **Non-goals** — what this issue explicitly will not do
- **Verification plan** — commands to run before close (default: CI-equivalent subset from Verification below)

Use only labels that exist on the repo (`gh label list`). Current labels include `bug`, `enhancement`, `documentation`, `question`, `help wanted`, `good first issue`, `GMA-217`, `GMA-218`.

## Planning

- **T0–T1** — inline plan in the PR `Summary` is enough; name files and risks.
- **T2–T3** — use hub `architect` for design and tradeoffs; use hub `plan-code-review-workflow` when the change spans multiple modules, eval gates, or CLI contracts. Plans must list files, rollback, and named checks per **sequence-verifiable-units**.
- Every plan names verification commands (ruff, filtered pytest, `packet eval`, `verify-packet`) before implementation starts.

## Implementation

- Prefer **write-tests** (hub skill) order: failing test or eval expectation, then minimal fix, then refactor.
- Python package layout: editable install `pip install -e ".[dev]"` from repo root; import `packet` from `src/packet` (see GMA-219 — do not shadow checkout with stale `site-packages/packet`).
- **ub-mode / unclebob:** `dryer`, `crapper`, and `mutator` are **not** CI blockers here. When installed locally, run `dryer` / `crapper` **advisory** only and summarize on the `Gates:` line; there is no mutator gate. Before advisory `dryer`, `rm -f .metrics/dry.edn` when that path exists (GMA-411).
- Commit messages: imperative subject, reference Linear issue id in body when `Linear:` is not `N/A`.

## Verification

**CI pass bar** (job `verify`, Ubuntu, Python 3.12.15) — every step exits 0:

1. **Install package** — `python -m pip install --upgrade pip` then `python -m pip install -e ".[dev]"`.
2. **Editable install imports the checkout (GMA-219)** — from `$RUNNER_TEMP`:  
   `python -c "import packet, pathlib, os; p=pathlib.Path(packet.__file__).resolve(); r=(pathlib.Path(os.environ['GITHUB_WORKSPACE'])/'src'/'packet').resolve(); assert p.is_relative_to(r), p; print(p)"`
3. **Ruff** — `python -m ruff check src tests`
4. **Pytest (unit)** — `python -m pytest -q -m "not needs_network and not needs_gpu and not needs_models and not needs_ocr_extras" --tb=short`
5. **Builtin eval GATE** (`PYTHONPATH: src`) —  
   `set -o pipefail`  
   `export PATH="${HOME}/.local/bin:${PATH}"`  
   `python -c "import packet, pathlib; p=pathlib.Path(packet.__file__).resolve(); r=(pathlib.Path('src')/'packet').resolve(); assert p.is_relative_to(r), p"`  
   `CACHE="${RUNNER_TEMP}/packet-eval"`  
   `mkdir -p "$CACHE"`  
   `packet eval --dir "$CACHE" | tee "${RUNNER_TEMP}/eval-builtin.txt"`  
   `grep -q 'GATE PASS' "${RUNNER_TEMP}/eval-builtin.txt"`
6. **verify-packet headless smoke** (`PACKET_VERIFY_RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}`, `PYTHONPATH: src`) —  
   `set -o pipefail`  
   `export PATH="${HOME}/.local/bin:${PATH}"`  
   `python -c "import packet, pathlib; p=pathlib.Path(packet.__file__).resolve(); r=(pathlib.Path('src')/'packet').resolve(); assert p.is_relative_to(r), p"`  
   `.cursor/skills/verify-packet/control-packet launch`  
   `eval "$(.cursor/skills/verify-packet/control-packet env)"`  
   `.cursor/skills/verify-packet/control-packet ci-headless`

**Local pre-push** (from repo root; mirror CI order; Python 3.11+ per `pyproject.toml`):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
REPO_ROOT="$(pwd)"
cd "${TMPDIR:-/tmp}"
GITHUB_WORKSPACE="$REPO_ROOT" python -c "import packet, pathlib, os; p=pathlib.Path(packet.__file__).resolve(); r=(pathlib.Path(os.environ['GITHUB_WORKSPACE'])/'src'/'packet').resolve(); assert p.is_relative_to(r), p; print(p)"
cd "$REPO_ROOT"
python -m ruff check src tests
python -m pytest -q -m "not needs_network and not needs_gpu and not needs_models and not needs_ocr_extras" --tb=short
set -o pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH=src
python -c "import packet, pathlib; p=pathlib.Path(packet.__file__).resolve(); r=(pathlib.Path('src')/'packet').resolve(); assert p.is_relative_to(r), p"
CACHE="${TMPDIR:-/tmp}/packet-eval-local"
mkdir -p "$CACHE"
packet eval --dir "$CACHE" | tee "${TMPDIR:-/tmp}/packet-eval-local.txt"
grep -q 'GATE PASS' "${TMPDIR:-/tmp}/packet-eval-local.txt"
export PACKET_VERIFY_RUN_ID="local-$(date +%s)-$$"
.cursor/skills/verify-packet/control-packet launch
eval "$(.cursor/skills/verify-packet/control-packet env)"
.cursor/skills/verify-packet/control-packet ci-headless
```

Deep CLI proof after behavioral changes: follow `.cursor/skills/verify-packet/SKILL.md` and `features/` recipes (OCR paths need optional extras and are outside default CI).

**Gates line (advisory):** Summarize ruff/pytest/eval/ci-headless exit codes. Optional: if `dryer` or `crapper` is on PATH, run and note pass/fail without blocking merge. Do not require mutator. PR description **format** lint (header order) is advisory — no workflow enforces it in this repo.

**Drift:** When CI or verify-packet changes, run hub `maintain-repo-standards` (trigger only; no scheduled routine).

Principles (names only): **encode-lessons-in-structure**, **prove-it-works**, **sequence-verifiable-units**.
