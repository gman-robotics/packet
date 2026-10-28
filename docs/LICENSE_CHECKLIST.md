# License checklist

Status: **section 1 cleared; sections 4 and 6 cleared with ⚠ caveats (no LICENSE file beside some weights) on 2026-10-13; 2, 3 and 5 still open.** Packet's
own code is MIT (`LICENSE`), but Packet *names* several third-party
backends (in `DESIGN.md`, `README.md`, and as optional dependencies in
`pyproject.toml`) whose licenses do not automatically become MIT just
because Packet's glue code is. This file exists so "we mentioned the
license in DESIGN.md" is never treated as equivalent to actually clearing
each backend before it ships. A soft mention in prose is not a substitute
for this file.

Legend: ☐ not checked · ☑ checked/cleared · ⚠ checked, accepted with a
recorded caveat (counts as cleared for the sign-off gate, but must be
re-checked before bundling) · N/A not applicable at this package's current
integration state.

## How to use this checklist

For each backend below, before it is (a) added as a non-optional
dependency, or (b) bundled/vendored (model weights or code shipped inside
a Packet artifact rather than pip-installed by the user at their own
discretion, or downloaded lazily at runtime into the user's own cache),
confirm every row and record the date, the license version
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

Referenced in `DESIGN.md` (Escalation policy, Backend adapters) and
selected as the **fallback** OCR engine (`PaddleExtractor`, PaddleOCR-VL-1.6
page pipeline) behind LightOnOCR-3 (section 6), per
`bench/ocr_compare/REPORT.md`. `PaddleExtractor` is implemented in
`src/packet/backends.py` (PaddleOCR-VL-1.6 page pipeline). Packet never bundles or vendors
the Paddle code or weights: the user installs `paddleocr`/`paddlepaddle`
themselves and the weights are downloaded by paddleocr/paddlex at first
use into the user's cache.

Checked 2026-10-13 by Bo (bench eng, for Thomas Ginter).

| Check | Status | Notes |
|---|---|---|
| PaddleOCR / PaddleOCR-VL code license identified and recorded with version | ☑ | `paddleocr==3.7.0`, `paddlex==3.7.2`, `paddlepaddle==3.2.1` (versions installed and benchmarked in `bench/ocr_compare`). All Apache-2.0: read <https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE> (Apache License 2.0, repo `main` @ `dab3fe3`) and the installed wheels' metadata (`License: Apache License 2.0` / `Apache-2.0` / `Apache Software License`). |
| Model weight license(s) for the specific detection/recognition/VL models Packet would pull identified separately from the code license | ⚠ | **PaddleOCR-VL-1.6** (`PaddlePaddle/PaddleOCR-VL-1.6` @ `c5630abae1d940eafe0697512a0325494b02ab42`): repo ships a `LICENSE` file, Apache License 2.0, "Copyright (c) 2025 PaddlePaddle Authors" — read <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/LICENSE>; card metadata `license: apache-2.0`. **PP-DocLayoutV3** (layout stage, fetched by paddlex; `PaddlePaddle/PP-DocLayoutV3` @ `241f8bdfc77a7c7bee915a5057aaee58c235a8d3`): card metadata `license: apache-2.0` (<https://huggingface.co/PaddlePaddle/PP-DocLayoutV3>); **no LICENSE file in that repo** — same caveat as LightOn below. paddlex fetches it unpinned; pin before relying on it. Verified by listing the paddlex model cache after the full benchmark run (`paddlex-cache/official_models/` contained only `PP-DocLayoutV3` and `PaddleOCR-VL-1.6`); orientation/unwarping models were not fetched with the pipeline defaults used. Re-check if pipeline options change. Base-model ancestry (ERNIE-4.5-0.3B LM) is covered by the Apache-2.0 grant on the released weights. |
| Confirmed no GPL/AGPL-licensed component would be pulled in transitively via the `paddlepaddle` runtime dependency chain | ☑ | Scanned every distribution in the benchmark venv (`paddleocr 3.7.0` + deps) on 2026-10-13: **no GPL or AGPL**. LGPL-only transitive deps: `cssutils 2.15.0` (LGPL-3.0+), `encutils 1.0.0` (LGPL-3.0+), `python-bidi 0.6.11` (LGPL), `crc32c 2.9.post0` (LGPL-2.1+). These are dynamically imported by the user's own install, not shipped by Packet, so they do not affect Packet's MIT license. Re-scan if Packet ever ships a lockfile or bundle. |
| Attribution/NOTICE requirements captured | ☑ | Apache-2.0 §4 applies only on redistribution. Packet does not redistribute Paddle code or weights, so no NOTICE is required today. If that ever changes, ship the Apache-2.0 text plus the PaddlePaddle copyright line above. |

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

## 6. LightOnOCR-3 (default OCR engine)

Selected as the **default** OCR engine: `LightOnOCR-3-1B` in grounding
mode, run with MLX (`mlx-vlm`) on Apple silicon, per
`bench/ocr_compare/REPORT.md`. `LightOnExtractor` is implemented in
`src/packet/backends.py` (LightOnOCR-3-1B grounding on MLX).
Only the 1B is used; the 0.8B/4B cards were read for comparison only. Packet never bundles or vendors the
weights; they are downloaded by the user (or lazily from Hugging Face at
first use) into the user's HF cache. `mlx-vlm` is not a Packet dependency or extra; the user installs it.

Checked 2026-10-13 by Bo (bench eng, for Thomas Ginter).

| Check | Status | Notes |
|---|---|---|
| Model weight license identified and recorded with version | ☑ | `lightonai/LightOnOCR-3-1B` @ `23d544085c886bbdace647bcfab91203299417e2`: Apache-2.0 — read the model card <https://huggingface.co/lightonai/LightOnOCR-3-1B> ("All models are released under the Apache 2.0 license for research and commercial use"; "License: Apache License 2.0"; metadata `license: apache-2.0`). Same terms on the 0.8B (`36d8636…`) and 4B (`b71010f…`) cards. |
| License text present as a file alongside the weights | ⚠ | **The LightOnOCR-3 license appears only on the model card. None of the three Hugging Face model repos contains a LICENSE file** (checked the repo file lists via the HF API on 2026-10-13). Note: LightOn's separate *code* repo <https://github.com/lightonai/LightOnOCR> (client/CLI/benchmarks, `main` @ `36755d4`) does carry an Apache-2.0 `LICENSE`, but that covers the code, not the weights. Accepted for a user-downloaded optional model; revisit before any bundling. |
| Confirmed the weights are not non-commercial / research-only | ☑ | Card states "research and commercial use". |
| Runtime code license (`mlx-vlm`, `mlx`) | ☑ | `mlx-vlm 0.7.6` (MIT) and `mlx 0.32.3` (MIT); optional, user-installed. Scanned every distribution in a venv with `packet` + `mlx-vlm 0.7.6` (incl. `transformers 5.19.0`) on 2026-10-13: no GPL/AGPL/LGPL. Base-model ancestry (Pixtral vision + Qwen3 LM, per the card: LightOnOCR-2 architecture) is covered by the Apache-2.0 grant on the released weights. |
| Attribution/NOTICE requirements captured | ☑ | No redistribution, so nothing to ship. If bundled later: Apache-2.0 text plus the LightOn citation from the card, and first ask LightOn to add a LICENSE file to the weights repos. |

## 7. Experimental model candidates (PROVISIONAL)

These are bench candidates from GMA-441, GMA-445 and GMA-443 only. **None of them is wired into Packet, and none is a default.** Every
row is **PROVISIONAL**: it records what the Hugging Face card, the API metadata and the repo file list said on 2026-10-28
for the pinned revision. Before any integration it needs a full section-style check (runtime code licenses, NOTICE,
ancestry). Weights stay user-downloaded into the user's own HF cache and are never bundled.

| Model @ revision | License (as read) | LICENSE file in weights repo | Status | Notes |
|---|---|---|---|---|
| `lightonai/LightOnOCR-3-1B` @ `23d544085c886bbdace647bcfab91203299417e2` | Apache-2.0 (card + metadata `license: apache-2.0`) | No | PROVISIONAL (already ☑/⚠ as the default in section 6) | Same revision as section 6; re-listed here so the candidate set is complete. |
| `LiquidAI/LFM3-VL-2B` @ `f83d2b0a2296aaa277f394990a9b8b949ddf9691` | **LFM Open License v1.0** (metadata `license: other`, `license_name: lfm1.0`) — <https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/LICENSE> | Yes | PROVISIONAL | **Not Apache.** §5 Commercial Use Limitation: commercial use is licensed only for entities under the $10M annual-revenue "Threshold". Needs owner sign-off before any non-research use. Runs only in the sandbox venv (`trust_remote_code`). |
| `LiquidAI/LFM2.5-VL-1.6B-Extract` @ `21073aa65b326c125ceb07d80fa08900bc2eb0d9` | **LFM Open License v1.0** (metadata `license: other`, `license_name: lfm1.0`) — <https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract/blob/main/LICENSE> | Yes | PROVISIONAL | Same revenue-threshold caveat. Its LICENSE text has a different md5 from the LFM3 file; the two were not diffed. Read this file on its own before relying on it. |
| `openbmb/MiniCPM-OCR-1.0` @ `6796197bb3a8e224b2f64ae462bfbb3f8a15ac4b` | Apache-2.0 (metadata `license: apache-2.0`) — <https://huggingface.co/openbmb/MiniCPM-OCR-1.0> | No | PROVISIONAL | License appears only on the card, the same caveat as LightOn. Base-model ancestry has not been checked. |
| `openbmb/MiniCPM-OCR-Jev` @ `90bc4c4768c082f769f5352479fa0df84da42c1a` | Apache-2.0 (metadata `license: apache-2.0`) — <https://huggingface.co/openbmb/MiniCPM-OCR-Jev> | No | PROVISIONAL | Region checker (GMA-445). Card-only license. Upstream scripts from `OpenBMB/MiniCPM-OCR` @ `458718f8` were read for parity only and not vendored. |
| `google/embeddinggemma-2` @ `26711154b641697c443124e316f3d21587f9ff5d` | Apache-2.0 (metadata `license: apache-2.0`; the card links <https://ai.google.dev/gemma/docs/gemma_4_license> labelled "Apache 2.0") — <https://huggingface.co/google/embeddinggemma-2> | No | PROVISIONAL | Not gated (API `gated: false`), checked 2026-10-28. Card-only license. Before use, read the linked Gemma page to confirm it is plain Apache-2.0 with no added use policy. GMA-443 candidate. |

## Sign-off

Packet's own code stays MIT. Model weights are always downloaded by the
user (or lazily at runtime), never bundled or vendored into a Packet
artifact.

This checklist must have every non-N/A row in sections 2-6 at ☑ or ⚠ (with the
version and date recorded) before the corresponding `Extractor` /
`Classifier` stub is wired into the default `Router`/`Pipeline` — i.e.
before `NotImplementedError` is removed from `DoclingExtractor`,
`PaddleExtractor` or `LightOnExtractor`, or before `InspectorClassifier` calls real
pdf-inspector functionality. Do not clear a row based on a general
familiarity with a project's license; link the specific license file or
page read for the specific pinned version.
