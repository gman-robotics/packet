# GMA-207: OCR model comparison for Packet (PaddleOCR-VL-1.6 vs LightOnOCR-3)

Parent: Packet / GMA-55. This compares models only; nothing in Packet's `src/` was changed.
Mac: M5 Pro, 64 GB. Started 20:18 MDT, report written about 21:25 MDT (Mac clock), so about 1 h 10 m of the 3 h limit was used.
Everything lives in an ocr-compare work dir outside the repo (`$OCR_COMPARE_ROOT`): `venv-tf/`, `venv-mlx/`, `venv-paddle/`, `hf/` (weights), `paddlex-cache/`, `runs/` (raw outputs), `logs/`, the scripts in `bench/ocr_compare/`, `testset/`.

## Models and pins (all Apache-2.0 per the HF card metadata)
| Model | HF repo @ revision | Arch | License evidence |
|---|---|---|---|
| PaddleOCR-VL-1.6 (0.9B VLM) | PaddlePaddle/PaddleOCR-VL-1.6 @ c5630abae1d940eafe0697512a0325494b02ab42 | ERNIE-4.5-0.3B + NaViT encoder, custom_code | LICENSE file (Apache 2.0) in repo + card `license: apache-2.0` |
| PP-DocLayoutV3 (layout stage, fetched by paddleocr) | PaddlePaddle/PP-DocLayoutV3 @ 241f8bdfc77a7c7bee915a5057aaee58c235a8d3 (HEAD on 2026-10-12; paddlex fetches unpinned) | Paddle inference model | card `license: apache-2.0`, Apache badge |
| LightOnOCR-3-0.8B | lightonai/LightOnOCR-3-0.8B @ 36d8636925327c671bd8e479c517a004f7566bd9 | Qwen3.5 VL | card metadata + text "Apache 2.0 … research and commercial use"; **no LICENSE file in repo** |
| LightOnOCR-3-1B | lightonai/LightOnOCR-3-1B @ 23d544085c886bbdace647bcfab91203299417e2 | Mistral3/Pixtral vision + Qwen3 LM (LightOnOCR-2 arch) | same; **no LICENSE file** |
| LightOnOCR-3-4B | lightonai/LightOnOCR-3-4B @ b71010f095dc3735de7ee6b9969bff8f8f50b39a | Qwen3.5 VL | same; **no LICENSE file** |

The paddleocr pipeline downloaded its own copy of the VL weights. Its sha256 (`85a479d5…71db`) is identical to the pinned HF blob.
Model cards are saved in `cards/`.

## Test set (fixed, documented; `bench/ocr_compare/make_testset.py`, seed 20261012, 150 dpi)
- **Packet fixtures** (`packet.eval.fixtures.build_builtin`, rendered by `bench/ocr_compare/packet_fixtures.py`): 13 pages (text-layer ×3, multi-doc ×4, shared-header ×4, scan ×2). Gold text comes from the PDF text layer. The `scan` fixture is blank, so it serves as a hallucination test where the model should output nothing.
- **Synthetic multi-doc packet** `testset/synth/multidoc.pdf`, md-00…05: an invoice (2 pp), a lease (3 pp) and a bank statement (1 pp). Each page has a header, footer, page number (in both "Page x of y" and bare forms), body text, and 3 tables. 4 of the 6 pages are scan-degraded (skew, blur, noise). Gold covers text, tables, furniture and the doc boundaries (new docs start at pages 2 and 5).
- **Hard scans** hs-00/03/05: the same layouts, downsampled to 2/3 size, 3° skew, heavy noise, JPEG q30.
- **Public samples** without shipped ground truth, scored with the manual rubric in `bench/ocr_compare/public_rubric.py`:
  - SROIE receipt (hf-internal-testing/fixtures_ocr, a ScanSnap scan): 9 hand-transcribed key fields.
  - PaddleOCR demo page (a 4-column Chinese newspaper with a photo and caption): 7 anchors in true reading order, scored for hits and pairwise order.

Metrics (`bench/ocr_compare/score.py`):
- CER/WER (jiwer), order-sensitive, with markup stripped.
- Line recall.
- Table cell recall.
- Furniture recall: header, footer and page-number text is present.
- Furniture labeled: the model's own label is correct (synth and hard-scan pages only).
- Characters emitted on the blank pages.

Boundaries: `bench/ocr_compare/to_blocks.py` maps each output to `packet.types.Block` and runs **Packet's own `BoundaryDetector`** over md-00…05.

## Results (24 pages; accuracy over the 22 pages with gold)
| Model / mode / backend | CER | Line recall | Table cells | Furniture present / labeled | Boundaries (Packet detector) | Receipt fields | News anchors / order | Blank-page chars | Pages/min | Peak mem |
|---|---|---|---|---|---|---|---|---|---|---|
| LightOn 0.8B plain, MLX bf16 | 0.0002 | 0.991 | 1.00 | 0.98 / n/a | ✅ [2,5] | 0.67 | 0.86 / 1.0 | 0 | **33.2** | 3.1 GB MLX |
| LightOn 0.8B grounding, MLX | 0.0005 | 0.981 | 1.00 | 0.98 / 0.89 | ✅ | 0.67 | 0.86 / 1.0 | 0 | 24.0 | 3.1 GB |
| LightOn 1B plain, MLX (template fixed) | 0.0001 | 0.996 | 1.00 | 0.98 / n/a | ✅ | 0.89 | 0.71 / 1.0 | 0 | 21.0 | 10.2 GB |
| LightOn 1B grounding, MLX | 0.0004 | 0.986 | 1.00 | 0.98 / 0.89 | ✅ | 0.89 | 0.86 / 1.0 | 0 | 16.0 | 10.2 GB |
| LightOn 1B plain, transformers MPS fp32 | 0.0001 | 0.996 | 1.00 | 0.98 / n/a | ✅ | 0.89 | 0.57* / 1.0 | 0 | 7.8 | 5.0 GB RSS |
| LightOn 1B grounding, transformers MPS fp32 | 0.0004 | 0.986 | 1.00 | 0.98 / 0.89 | ✅ | 0.89 | 0.57* / 1.0 | 0 | 5.9 | 5.0 GB RSS |
| LightOn 4B plain, MLX | **0.0000** | **1.000** | 1.00 | **1.00** / n/a | ✅ | 0.89 | 0.86 / 1.0 | 0 | 7.1 | 10.7 GB |
| LightOn 4B grounding, MLX | 0.0002 | 0.990 | 1.00 | 1.00 / **0.93** | ✅ | 0.89 | 0.86 / 1.0 | 0 | 5.2 | 10.7 GB |
| PaddleOCR-VL-1.6 page pipeline (layout+VL), paddle CPU | 0.0279 | 0.953 | 1.00 | 0.97 / 0.81 | ❌ as mapped (✅ with "first heading → title" rule) | 0.78 | 0.86 / 0.8 | 0 | 4.4 | 10.2 GB RSS |
| PaddleOCR-VL-1.6 VLM only ("OCR:"), MLX, no layout | 0.0113 | 0.955 | **0.00** (plain text, no table) | 0.97 / none | ❌ | 0.78 | 0.86 / 1.0 | 0 | 50.8 | 2.9 GB |

\* The transformers runs had max_new_tokens leaked at 1500, which truncated the newspaper page. Re-running the newspaper on MLX with a 6000-token cap gave 0.86 / 1.0 for all three LightOn sizes (`runs/news6k-*`). Every model misses the same anchor, so that anchor may be my transcription error.
Memory: "MLX" is `mx.get_peak_memory()`. "RSS" is the process peak from psutil plus `/usr/bin/time -l`. MPS driver allocation at the end of the fp32 1B run was 7.1 GB.

**Caveats.** The synthetic and Packet pages are clean typeset English, so most models are near the ceiling (CER < 0.1%) on them, and they mostly separate furniture labelling and pipeline failures rather than raw OCR. The real discriminators are the hard scans, the receipt and the newspaper. With 24 pages, these numbers support a direction, not a benchmark.

Specific failures seen:
- Paddle pipeline: on pk-shared-header-1 the layout stage missed a whole text line ("Chapter A continued."). Content outside detected regions is lost.
- Paddle pipeline: on hs-05 it misread "Page 1 of 1" as `P10201501` and dropped a paragraph.
- Paddle pipeline labels doc titles `paragraph_title` rather than `doc_title`.
- LightOn 0.8B dropped the receipt's document number and item line.
- All models missed or mangled the receipt barcode number (9556939040118).

## Speed / runtime support (MPS, MLX, CPU)
- **LightOnOCR-3 0.8B / 4B (Qwen3.5) on transformers + MPS: unusable.** The first page took 325 s, because the gated-delta and causal-conv1d layers fall back to reference PyTorch (the fla and causal_conv1d kernels are CUDA-only). I aborted that run (`logs/lighton-0.8b-plain-bf16.ABORTED.log`). This is the same failure mode as d3-flash.
- **LightOnOCR-3 on MLX (mlx-vlm 0.7.6, mlx 0.32.3): works for all three sizes**, loading the pinned official snapshots directly (bf16, no third-party conversions).
- **LightOnOCR-3-1B on transformers 5.17 + MPS:** works only with a workaround. The card's `from_pretrained` loads **zero LM/vision weights**: the config says `model_type: mistral3` and the checkpoint keys are `language_model.model.*`. I built `LightOnOcrConfig` from the same dict and remapped the keys (see `bench/ocr_compare/run_tf.py`). fp32 per the card.
- **mlx-vlm gotcha (1B):** mlx-vlm's chat template adds a stray `\n` after the image token on an empty prompt, which switches the 1B into grounding output. Patched in `bench/ocr_compare/run_mlx.py`. The first run is kept as `runs/mlx-lighton-1b-plain-BADTEMPLATE`.
- **PaddleOCR-VL-1.6:**
  - paddlepaddle 3.2.1 on macOS arm64 is **CPU only** (no Metal), giving 4.4 pages/min for the full page pipeline.
  - The VLM alone runs on MLX at 50 pages/min, but without the layout stage you lose tables, bboxes and labels.
  - paddleocr has an `--vl_rec_backend mlx-vlm-server` option (layout on CPU, VLM on MLX) that would likely close most of the gap. **Not tested** in this run.
  - On transformers the card only supports element-level use, and not on MPS (its example uses cuda/cpu).

## Mapping to Packet Blocks (`bench/ocr_compare/to_blocks.py`, outputs in `runs/*/…packet_blocks.json`)
- **LightOn grounding:**
  - Labels map 1:1: text→PARAGRAPH, title→TITLE (H1) / HEADING (H2+, level from `#`), list→LIST_ITEM, header/footer/page_number→HEADER/FOOTER/PAGE_NUMBER, caption, formula, code, table, image/chart→FIGURE, header_image/footer_image→HEADER/FOOTER.
  - Bboxes are normalized 0–1000 and convert to pt via page size. Reading order is emission order.
  - Lost or approximated:
    - footnote and aside_text have no BlockKind (mapped to PARAGRAPH, label kept in `extra`).
    - The `+` continuation marker has no field (`extra.continuation`).
    - Tables and charts are HTML strings (`extra.html`). There is no cell structure in Block.
    - No per-block confidence (set to 1.0).
- **LightOn plain:** markdown only. No bboxes and no furniture labels, so headers, footers and page numbers become PARAGRAPH. Packet's boundary detector still worked here, through the header fingerprint and title.
- **Paddle pipeline:**
  - Richest label set: doc_title, paragraph_title, header, footer, number, seal→STAMP, algorithm→CODE, figure/table/chart titles→CAPTION, display/inline formula, and others. Pixel bboxes. Explicit reading order from the layout model.
  - Lost or approximated: abstract, reference, footnote and aside have no BlockKind. Table HTML has no Block cell structure.
  - Titles often come out as `paragraph_title`, so with a straight mapping Packet's detector found **no** boundaries. Adding a "first heading on a page → TITLE" rule gave exact [2,5].
- **Packet-side gap (independent of model):** `boundary._PAGE_NUM` needs "x of y". The bare page numbers on the lease ("1", "2", "3"), although correctly labeled `page_number` by LightOn grounding and Paddle, give no page-number-reset signal. Boundaries were found only via header change + title.

## Extractor integration effort
| | Deps | Install pain on the Mac | Extractor shape |
|---|---|---|---|
| LightOnOCR-3 (MLX) | mlx, mlx-vlm (pulls transformers 5.19) | Low: one pip install, ~2 min. Must patch the 1B empty-prompt template; weights 1.6/3.4/8.5 GB | One call per page image → parse `![label](bbox)` → Blocks (~150 lines, done in to_blocks.py) |
| LightOnOCR-3 (transformers) | torch 2.14, transformers ≥5.17 | 1B needs the key-remap hack; 0.8B/4B impractically slow on MPS (fine on CUDA) | same |
| PaddleOCR-VL pipeline | paddlepaddle 3.2.1 + paddleocr 3.7/paddlex 3.7.2 (~100 deps: opencv-contrib, modelscope, aistudio-sdk, openai, …) | Medium: big dependency tree, CPU only, 7–47 s cold load, downloads layout weights at runtime (set `PADDLE_PDX_MODEL_SOURCE=huggingface`, `PADDLE_PDX_CACHE_HOME`) | `predict()` → `parsing_res_list` with label, bbox, content → Blocks. Fills the existing `PaddleExtractor` stub naturally |

Transitive licences: venv-paddle has LGPL packages (cssutils, encutils via premailer; python-bidi; crc32c). No GPL or AGPL found by package metadata. venv-tf and venv-mlx had no GPL, LGPL or AGPL hits.

## LICENSE_CHECKLIST (packet `docs/LICENSE_CHECKLIST.md`)
§4 PaddleOCR-VL:
- Code (paddleocr) and weights (VL-1.6, PP-DocLayoutV3) are Apache-2.0 at the pins above.
- No GPL/AGPL found. LGPL deps were noted.
- NOTICE: Apache LICENSE ships in the weights repo.
- I'm proposing these as **evidence only. I did not tick the boxes**, because the checklist wants an owner sign-off with links.

LightOnOCR-3 has **no row in the checklist**, so a new section is needed. Its repos have no LICENSE file; the Apache-2.0 grant is only in card metadata and prose. That is weaker evidence and worth confirming with LightOn before shipping weights by default.

## Recommendation
- **Default: LightOnOCR-3-1B, grounding mode, via MLX.**
  - Best measured balance here: 16 pages/min, 0.89 furniture labeling, correct boundaries, 0.89 receipt fields, typed blocks with bboxes.
  - Light install.
  - It also keeps LightOnOCR-2 compatibility.
  - Use **0.8B** where throughput matters: 24–33 pages/min at 3 GB, but weaker on the noisy receipt.
  - Use **4B** for a quality pass: best on every synthetic metric, 5–7 pages/min.
- **Fallback: the PaddleOCR-VL-1.6 page pipeline.**
  - Richest layout taxonomy and explicit layout reading order, and it is what the `PaddleExtractor` stub expects.
  - But it is CPU only on macOS, slower (4.4 pages/min), and has a heavy dependency tree.
  - It showed layout-miss content loss in this set, and needs a title-mapping rule for Packet boundaries.
  - Worth testing its `mlx-vlm-server` backend before treating it as a Mac default.
- **Packet follow-ups (not done, out of scope):**
  - Accept bare page numbers when a block is already typed PAGE_NUMBER.
  - Decide BlockKinds or `extra` conventions for footnote, aside and continuation.
  - Add a LightOnOCR row to LICENSE_CHECKLIST.

## Commands run (abridged; full scripts in `bench/ocr_compare/`; run from the work dir, or set `OCR_COMPARE_ROOT` for the `.sh` drivers)
```
python3.13 -m venv venv-tf && venv-tf/bin/pip install transformers==5.17.0 torch torchvision pillow pypdfium2 accelerate jiwer psutil huggingface_hub numpy pydantic
python3.13 -m venv venv-paddle && venv-paddle/bin/pip install paddlepaddle==3.2.1 "paddleocr[doc-parser]>=3.6.0" psutil pypdfium2
python3.13 -m venv venv-mlx && venv-mlx/bin/pip install mlx-vlm psutil          # mlx-vlm 0.7.6, mlx 0.32.3
HF_HOME=$PWD/hf HF_HUB_DISABLE_XET=1 hf download <repo> --revision <sha>        # xet transfers stalled; plain HTTP worked
venv-tf/bin/python bench/ocr_compare/make_testset.py testset/synth; venv-tf/bin/python bench/ocr_compare/packet_fixtures.py
bench/ocr_compare/run_all.sh lighton-1b:plain:fp32 lighton-1b:grounding:fp32               # transformers/MPS
bench/ocr_compare/batch_mlx.sh lighton-0.8b:plain lighton-0.8b:grounding lighton-1b:plain lighton-1b:grounding lighton-4b:plain lighton-4b:grounding paddle-vl-1.6:paddle
bench/ocr_compare/batch3.sh                                                                  # 1B plain template fix + newspaper at 6000 tokens
venv-paddle/bin/python bench/ocr_compare/run_paddle.py runs/paddle-vl-1.6-pipeline-cpu <imgs>   # PADDLE_PDX_MODEL_SOURCE=huggingface
venv-tf/bin/python bench/ocr_compare/score.py runs/*; bench/ocr_compare/to_blocks.py runs/*; bench/ocr_compare/public_rubric.py runs/*
```
Raw outputs are in `runs/<run>/<page>.txt`, with `_summary.json` (speed and memory), `_scores.json`, `_blocks.json`, `_public_rubric.json` and `*.packet_blocks.json` alongside. Logs are in `logs/`.

Harness copy: `packet` clone at `ocr-compare/packet`, branch `bo/ocr-compare`, `bench/ocr_compare/` only. Committed locally, **not pushed, no PR**.
