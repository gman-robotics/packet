# Packet model evaluation harness (GMA-441, Exp A)

Harness only. Nothing here changes Packet defaults or `src/`. Outputs go outside git (`~/.cache/packet/model-eval/p1/`).
Plan: Linear GMA-439 / GMA-441, "Packet Model Evaluation Plan" (Tom, 2026-10-27).

| Script | What it does |
|---|---|
| `run_lfm_extract.py` | LiquidAI/LFM2.5-VL-1.6B-Extract @21073aa6 on mlx-vlm. Card prompt (system: "Extract the following from the image" + YAML fields + "Respond with only a JSON object"; user turn: image only), greedy. JSON fields render to markdown. |
| `run_lfm3_mps.py` | LiquidAI/LFM3-VL-2B @f83d2b0a on transformers + torch MPS, `trust_remote_code`, no-think, greedy. **Run only from the sandbox venv** `/Users/Shared/gman-sandbox/lfm3-mps/venv` with `HF_HUB_OFFLINE=1`. |
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
