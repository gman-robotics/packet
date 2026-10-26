<!--
Tier scale: T0 docs/check-only; T1 medium fix in existing module; T2 larger/cross-cutting; T3 infra/migration/major architecture.
Linear: use a full https://linear.app/... issue URL or N/A only.
Gates: advisory — report ruff, pytest, packet eval GATE PASS, verify-packet ci-headless; optional dryer/crapper if on PATH; no mutator; header format is not CI-blocking.
-->

## Tier

<!-- T0 | T1 | T2 | T3 — see HTML comment at top of this file -->

Tier:

## Models

<!-- implementer model, reviewer model (e.g. composer-2.5 implementer, grok-4.7 review) -->

Models:

## Linear

<!-- Example: https://linear.app/gman-personal-projects/issue/GMA-412 — or N/A -->

Linear:

## Summary

<!-- What changed and why; link DESIGN.md or eval docs when behavior shifts -->

Summary:

## Gates

<!-- Advisory only: exit codes for ruff, filtered pytest, packet eval (GATE PASS), verify-packet ci-headless; note dryer/crapper if run -->

Gates:

## Verification

- [ ] `python -m pip install --upgrade pip` and `python -m pip install -e ".[dev]"` (exit 0)
- [ ] `python -m ruff check src tests` (exit 0)
- [ ] `python -m pytest -q -m "not needs_network and not needs_gpu and not needs_models and not needs_ocr_extras" --tb=short` (exit 0)
- [ ] `PYTHONPATH=src packet eval --dir ${TMPDIR:-/tmp}/packet-eval-local` prints `GATE PASS` with `set -o pipefail` and tee (exit 0)
- [ ] `.cursor/skills/verify-packet/control-packet ci-headless` after `launch` + `env` (exit 0)
