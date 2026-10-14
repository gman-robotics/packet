# License checklist

Status: **sections 1, 4 and 6 cleared 2026-10-13; 2, 3 and 5 still open.** Packet's
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

Referenced in `DESIGN.md` (Escalation policy, Backend adapters) and wired
as the **fallback** OCR engine (`PaddleExtractor`, PaddleOCR-VL-1.6 page
pipeline) behind LightOnOCR-3 (section 6). Packet never bundles or vendors
the Paddle code or weights: the user installs `paddleocr`/`paddlepaddle`
themselves and the weights are downloaded by paddleocr/paddlex at first
use into the user's cache.

Checked 2026-10-13 by Bo (bench eng, for Thomas Ginter).

| Check | Status | Notes |
|---|---|---|
| PaddleOCR / PaddleOCR-VL code license identified and recorded with version | ☑ | `paddleocr==3.7.0`, `paddlex==3.7.2`, `paddlepaddle==3.2.1` (versions installed and benchmarked in `bench/ocr_compare`). All Apache-2.0: read <https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE> (Apache License 2.0, repo `main` @ `dab3fe3`) and the installed wheels' metadata (`License: Apache License 2.0` / `Apache-2.0` / `Apache Software License`). |
| Model weight license(s) for the specific detection/recognition/VL models Packet would pull identified separately from the code license | ☑ | **PaddleOCR-VL-1.6** (`PaddlePaddle/PaddleOCR-VL-1.6` @ `c5630abae1d940eafe0697512a0325494b02ab42`): repo ships a `LICENSE` file, Apache License 2.0, "Copyright (c) 2025 PaddlePaddle Authors" — read <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/LICENSE>; card metadata `license: apache-2.0`. **PP-DocLayoutV3** (layout stage, fetched by paddlex; `PaddlePaddle/PP-DocLayoutV3` @ `241f8bdfc77a7c7bee915a5057aaee58c235a8d3`): card metadata `license: apache-2.0` (<https://huggingface.co/PaddlePaddle/PP-DocLayoutV3>); **no LICENSE file in that repo** — same caveat as LightOn below. paddlex fetches it unpinned; pin before relying on it. No other Paddle models are pulled by the PaddleOCR-VL pipeline as configured. |
| Confirmed no GPL/AGPL-licensed component would be pulled in transitively via the `paddlepaddle` runtime dependency chain | ☑ | Scanned every distribution in the benchmark venv (`paddleocr 3.7.0` + deps) on 2026-10-13: **no GPL or AGPL**. LGPL-only transitive deps: `cssutils 2.15.0` (LGPL-3.0+), `encutils 1.0.0` (LGPL-3.0+), `python-bidi 0.6.11` (LGPL), `crc32c 2.9.post0` (LGPL-2.1+). These are dynamically imported by the user's own install, not shipped by Packet, so they do not affect Packet's MIT license. Re-scan if Packet ever ships a lockfile or bundle. |
| Attribution/NOTICE requirements captured | ☑ | Apache-2.0 §4 applies only on redistribution. Packet does not redistribute Paddle code or weights, so no NOTICE is required today. If that ever changes, ship the Apache-2.0 text plus the PaddlePaddle copyright line above. |

## 6. LightOnOCR-3 (default OCR engine)

Wired as the **default** OCR engine: `LightOnOCR-3-1B` in grounding mode,
run with MLX (`mlx-vlm`) on Apple silicon, per
`bench/ocr_compare/REPORT.md`. Packet never bundles or vendors the
weights; they are downloaded by the user (or lazily from Hugging Face at
first use) into the user's HF cache. `mlx-vlm` is an optional extra the
user installs.

Checked 2026-10-13 by Bo (bench eng, for Thomas Ginter).

| Check | Status | Notes |
|---|---|---|
| Model weight license identified and recorded with version | ☑ | `lightonai/LightOnOCR-3-1B` @ `23d544085c886bbdace647bcfab91203299417e2`: Apache-2.0 — read the model card <https://huggingface.co/lightonai/LightOnOCR-3-1B> ("All models are released under the Apache 2.0 license for research and commercial use"; "License: Apache License 2.0"; metadata `license: apache-2.0`). Same terms on the 0.8B (`36d8636…`) and 4B (`b71010f…`) cards. |
| License text present as a file alongside the weights | ⚠ | **The LightOnOCR-3 license appears only on the model card. None of the three Hugging Face model repos contains a LICENSE file** (checked the repo file lists via the HF API on 2026-10-13). Note: LightOn's separate *code* repo <https://github.com/lightonai/LightOnOCR> (client/CLI/benchmarks, `main` @ `36755d4`) does carry an Apache-2.0 `LICENSE`, but that covers the code, not the weights. Accepted for a user-downloaded optional model; revisit before any bundling. |
| Confirmed the weights are not non-commercial / research-only | ☑ | Card states "research and commercial use". |
| Runtime code license (`mlx-vlm`, `mlx`) | ☑ | `mlx-vlm 0.7.6` (MIT) and `mlx 0.32.3` (MIT), as installed in the benchmark venv; optional, user-installed. |
| Attribution/NOTICE requirements captured | ☑ | No redistribution, so nothing to ship. If bundled later: Apache-2.0 text plus the LightOn citation from the card, and first ask LightOn to add a LICENSE file to the weights repos. |

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

Packet's own code stays MIT. Model weights are always downloaded by the
user (or lazily at runtime), never bundled or vendored into a Packet
artifact.

This checklist must have every non-N/A row in sections 2-6 at ☑ (with the
version and date recorded) before the corresponding `Extractor` /
`Classifier` stub is wired into the default `Router`/`Pipeline` — i.e.
before `NotImplementedError` is removed from `DoclingExtractor` or
`PaddleExtractor`, or before `InspectorClassifier` calls real
pdf-inspector functionality. Do not clear a row based on a general
familiarity with a project's license; link the specific license file or
page read for the specific pinned version.
