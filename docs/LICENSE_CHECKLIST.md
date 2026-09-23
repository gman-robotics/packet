# License checklist

Status: **checklist only — not all rows below are cleared yet.** Packet's
own code is MIT (`LICENSE`), but Packet *names* several third-party
backends (in `DESIGN.md`, `README.md`, and as optional dependencies in
`pyproject.toml`) whose licenses do not automatically become MIT just
because Packet's glue code is. This file exists so "we mentioned the
license in DESIGN.md" is never treated as equivalent to actually clearing
each backend before it ships. A soft mention in prose is not a substitute
for this file.

Legend: ☐ not checked · ☑ checked/cleared · N/A not applicable at this
package's current integration state.

## How to use this checklist

For each backend below, before it is (a) added as a non-optional
dependency, or (b) bundled/vendored (model weights or code shipped inside
a Packet artifact rather than pip-installed by the user at their own
discretion), confirm every row and record the date, the license version
checked, and who checked it. Do not check a row from memory — link the
actual license file or page you read.

## 1. Packet itself (MIT spine)

| Check | Status | Notes |
|---|---|---|
| `LICENSE` file present at repo root and matches `pyproject.toml`'s `license = { text = "MIT" }` | ☑ | Present; both say MIT. |
| Every source file's copyright/attribution is consistent with the MIT grant (no per-file conflicting headers) | ☑ | No per-file license headers exist in `src/packet/*`; nothing to conflict. |
| Third-party code copy-pasted (not imported) into `src/` is inventoried separately from this table | ☑ | None today — `src/packet` has no vendored third-party source. |

## 2. Docling (bundled models)

Referenced in `DESIGN.md` (Backend adapters) and as the `docling` optional
extra in `pyproject.toml`. `DoclingExtractor` in `src/packet/backends.py`
is currently a stub (`NotImplementedError`) — **not wired in**, so this row
set is a pre-flight check to clear before `DoclingExtractor` ships, not a
retroactive audit.

| Check | Status | Notes |
|---|---|---|
| Docling's own code license identified (project license, e.g. MIT/Apache-2.0) and recorded here with version | ☐ | Not yet checked against a pinned Docling version. |
| Docling's **bundled model weights** license(s) identified separately from the code license — Docling ships layout/table/OCR models that may carry their own (sometimes more restrictive, e.g. research-only) terms | ☐ | Model weights can have different terms than the wrapping library; must be checked per model Docling pulls in, not just the `docling` package. |
| Confirmed whether any bundled model is non-commercial / research-only, which would block Packet shipping it as a default | ☐ | |
| Attribution requirements (NOTICE file, model cards) captured for redistribution if Packet ever bundles/vendors instead of pip-installs | ☐ | |

## 3. Marker (model weights)

Referenced in `DESIGN.md` (Why this exists, Backend adapters) as a
comparable/alternative converter and named as a future scan-path option.
No Marker integration exists in code yet.

| Check | Status | Notes |
|---|---|---|
| Marker's code license identified and recorded with version | ☐ | |
| Marker's model weights license identified separately (Marker's layout/OCR/table weights have historically carried non-commercial terms in some releases — must be re-verified against whatever version is actually integrated, not assumed from an older release) | ☐ | This is the specific risk this checklist exists to catch: "we mentioned Marker in DESIGN.md" is not a clearance. |
| Confirmed license terms as of the exact version pinned in `pyproject.toml` before adding a `marker` extra | ☐ | No `marker` extra exists yet; add this check before one does. |

## 4. PaddleOCR-VL

Referenced in `DESIGN.md` (Escalation policy, Backend adapters) as the
scan/OCR backend. `PaddleExtractor` in `src/packet/backends.py` is
currently a stub (`NotImplementedError`) — not wired in.

| Check | Status | Notes |
|---|---|---|
| PaddleOCR / PaddleOCR-VL code license identified and recorded with version | ☐ | |
| Model weight license(s) for the specific detection/recognition/VL models Packet would pull identified separately from the code license | ☐ | PaddleOCR historically distributes pretrained models under terms that need per-model verification (some Apache-2.0, some under PaddlePaddle-specific terms) — do not assume one blanket license covers every model file. |
| Confirmed no GPL/AGPL-licensed component would be pulled in transitively via the `paddlepaddle` runtime dependency chain | ☐ | |
| Attribution/NOTICE requirements captured | ☐ | |

## 5. pdf-inspector (Firecrawl)

Referenced in `DESIGN.md` (Escalation policy) and as the `inspector`
optional extra in `pyproject.toml`. `InspectorClassifier` in
`src/packet/classify.py` currently only checks that `pdf_inspector` is
importable and then delegates to `HeuristicClassifier` — it does not call
into pdf-inspector's actual functionality yet, so this is also a
pre-flight check.

| Check | Status | Notes |
|---|---|---|
| pdf-inspector's license identified and recorded with version | ☐ | |
| Confirmed pdf-inspector has no bundled model weights with separate terms (if it wraps a model, treat it like Docling/Marker/Paddle above; if it is pure heuristics/code, note that explicitly here) | ☐ | |
| Confirmed license compatibility with MIT distribution of Packet (i.e., pdf-inspector's terms do not force a stricter license onto Packet's own MIT code when used only as an optional runtime dependency via the `inspector` extra) | ☐ | |

## Sign-off

This checklist must have every non-N/A row in sections 2-5 at ☑ (with the
version and date recorded) before the corresponding `Extractor` /
`Classifier` stub is wired into the default `Router`/`Pipeline` — i.e.
before `NotImplementedError` is removed from `DoclingExtractor` or
`PaddleExtractor`, or before `InspectorClassifier` calls real
pdf-inspector functionality. Do not clear a row based on a general
familiarity with a project's license; link the specific license file or
page read for the specific pinned version.
