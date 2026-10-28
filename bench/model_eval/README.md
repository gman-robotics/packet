# Packet model evaluation harness (GMA-441, Exp A)

Harness only. Nothing here changes Packet defaults or `src/`. Outputs go outside git (`~/.cache/packet/model-eval/p1/`).
Plan: Linear GMA-439 / GMA-441, "Packet Model Evaluation Plan" (Tom, 2026-10-27).

| Script | What it does |
|---|---|
| `run_lfm_extract.py` | LiquidAI/LFM2.5-VL-1.6B-Extract @21073aa6 on mlx-vlm. Card prompt (system: "Extract the following from the image" + YAML fields + "Respond with only a JSON object"; user turn: image only), greedy. JSON fields render to markdown. |
| `run_lfm3_mps.py` | LiquidAI/LFM3-VL-2B @f83d2b0a on transformers + torch MPS, `trust_remote_code`, no-think, greedy. **Run only from the sandbox venv** `/Users/Shared/gman-sandbox/lfm3-mps/venv` with `HF_HUB_OFFLINE=1`. |
| `run_minicpm_tf.py` | openbmb/MiniCPM-OCR-1.0 @6796197b on transformers (native `minicpm_ocr`, no remote code) + MPS, page prompt `Text Recognition:`. Sandbox venv. |
| `run_minicpm_llamacpp.py` | Same model as GGUF on llama.cpp `llama-server`. Kept for when a llama.cpp release loads projector type `minicpm_ocr_1_0` (Homebrew 0.7.0 build 11669 cannot; `--HEAD` fails to build). |
| `exp_a.py` | Feeds saved page text through `packet.ocr_parse.parse_lighton_grounding`, `BoundaryDetector` and the eval gate: builtin fixtures (false-OCR gate on), synth md-00..05 boundary F1 at tolerance 0, PoliTax holdout `--ocr-cache`-style eval on a candidate data dir (PDF symlinks; the LightOn cache is never written). |
| `summarize.py` | Per-slice table (born-digital, blank, synth clean/degraded, hard-scan) from `bench/ocr_compare/score.py` output. |
| `cold_start.sh` | Cold load without `sudo purge`: `ditto --nocache` copy of each snapshot, then a fresh process (cold) and a second fresh process (warm). |

LightOn baseline runs use `bench/ocr_compare/run_mlx.py` (plain-mode template fix included). Testset: `bench/ocr_compare/make_testset.py` + `packet_fixtures.py` (seed 20261012) under `testset/` (not committed).

```bash
bench/ocr_compare/run_mlx.py lighton-1b plain|grounding OUT IMGS...
bench/model_eval/run_lfm_extract.py x OUT IMGS...
/Users/Shared/gman-sandbox/lfm3-mps/venv/bin/python bench/model_eval/run_lfm3_mps.py x OUT IMGS...
bench/ocr_compare/score.py OUT; bench/ocr_compare/to_blocks.py OUT; bench/ocr_compare/public_rubric.py OUT
bench/model_eval/exp_a.py md-f1 OUT; bench/model_eval/exp_a.py builtin OUT
bench/model_eval/exp_a.py politax-link ~/.cache/packet/politax DST && bench/model_eval/exp_a.py politax-ocr-lfm DST
bench/model_eval/exp_a.py politax-eval DST
```

## Exp A2 (GMA-445): MiniCPM-OCR-Jev region checker

`openbmb/MiniCPM-OCR-Jev` @90bc4c47 (Apache-2.0) is a 2-class verifier: it sees the original crop, the candidate OCR text,
and a render of that text, and returns prob_yes ("consistent"). It runs on transformers + torch MPS in the sandbox venv
(native `qwen3_5`, no remote code, `HF_HUB_OFFLINE=1`). Packet defaults are not changed.

| File | What it does |
|---|---|
| `jev_score.py` | Model wrapper and scorer. The prompt bytes are checked against the upstream md5. The `score.weight` head runs on the last token. A PIL renderer stands in for upstream Chromium/KaTeX. `parity` scores the upstream examples; `testset` writes `RUN/_jev.json`; `politax` writes resumable jsonl for the holdout. |
| `jev_a2.py` | Analysis only (repo venv). `dev` computes labels, AUROC and the tuned thresholds on the testset (dev split). `drop-testset` / `drop-politax` remove blocks below the threshold into new run or candidate dirs. `holdout` reports flag fractions and a spot-check. The holdout is never used to tune. |

```
PY=/Users/Shared/gman-sandbox/lfm3-mps/venv/bin/python; export HF_HUB_OFFLINE=1
$PY bench/model_eval/jev_score.py parity <upstream>/examples
$PY bench/model_eval/jev_score.py testset ~/.cache/packet/model-eval/p1/runs/{lighton-1b-grounding,lfm25x,lfm3-mps,minicpm-mps}
.venv/bin/python bench/model_eval/jev_a2.py dev                    # thresholds from dev only
STRIDE=2 PAGE_LEVEL=0 $PY bench/model_eval/jev_score.py politax ~/.cache/packet/politax P1/a2/politax-lighton.jsonl   # read-only source
.venv/bin/python bench/model_eval/jev_a2.py drop-testset T; .venv/bin/python bench/model_eval/jev_a2.py drop-politax T
.venv/bin/python bench/model_eval/exp_a.py politax-eval P1/politax-lighton-jevdrop
```
Regions: LightOn grounding output is scored per `![label](bbox)` block, cropped from the page image. Untagged engines
(LFM2.5-Extract, LFM3, MiniCPM-OCR) have no bboxes, so each page is scored as one region. That input is outside
Jev's training distribution of block crops.
