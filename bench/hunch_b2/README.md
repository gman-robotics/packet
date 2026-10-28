# Exp B2 (GMA-448): Hunch-pattern teacher-student boundary signal

**Teacher:** PoliTax gold boundary labels. **Student:** the google/embeddinggemma-2 @26711154 page-head embedding (CPU fp32, 256 tokens, "Document" prompt) + logistic regression on `[e_i, e_i − e_{i−1}, cos(e_i, e_{i−1}), cos(e_i, mean(e_{i−1}, e_{i−2})), first-page]`.

The student enters Packet through the existing `decision_backend` hook, using a bench-only adapter (`StudentBackend`), so **no `src/` change is needed**. Its probability passes through `combined_confidence` (floor 0.40, cap 0.74), then the 0.74 no-reset cap, then the real `LookaheadSplitter` (commit 0.75, fallback 0.55).

The student emits 0.74 when p ≥ t and 0 otherwise. The student-alone arm zeroes every `W_*` weight and `COMBINED` in a context manager. The look-ahead page-number veto and implied-page-one still run.

## Splits and rules
- **Train:** politax-train, 23 docs, 2,882 pages, 920 boundaries. 5 folds, grouped by fixture.
- **Threshold rule, declared before any holdout run:** the highest train macro F1 with false splits ≤ the heuristic's train count (41).
- **Holdout:** politax-holdout, 7 docs, 429 pages, 202 boundaries. Frozen, and never used for fitting or choosing.
- **Regression sets:** dlnet-whole and builtin.

## Run
```
.venv/bin/python bench/hunch_b2/b2.py labels|ceiling|dump
HF_HUB_OFFLINE=1 <sandbox>/bin/python bench/hunch_b2/embed_cpu.py
.venv/bin/python bench/hunch_b2/b2.py train
WITH_MINICPM=1 ONLY_FILES=<3 pdfs> .venv/bin/python bench/hunch_b2/b2.py ablate
HF_HUB_OFFLINE=1 FILES=<pdfs> <sandbox>/bin/python bench/hunch_b2/ocr_minicpm.py ~/.cache/packet/model-eval/b2/politax-minicpm1
```
Artifacts are in `~/.cache/packet/model-eval/b2/`.

## Results (2026-10-28)
- **Ceiling (step 3)**, oracle student at 0.74 on every gold start, LightOn tagged. The look-ahead veto, not the cap, sets the ceiling.

  | split | heuristic | oracle alone | oracle + heuristic |
  |---|---|---|---|
  | holdout macro F1 | 0.4159 (19 false splits) | 0.8618 (2 false splits) | 0.8371 (13 false splits) |
  | train macro F1 | 0.2343 (41 false splits) | 0.9646 (8 false splits) | 0.9282 (33 false splits) |

  With untagged OCR the oracle reaches 0.9865, because no page numbers parse and so nothing vetoes.
- **Training:** out-of-fold AUROC 0.897. The rule picks t = 0.90 for student-alone (train F1 0.295 vs 0.234, 25 false splits) and t = 0.97 for student+heuristic (0.238, 41 false splits).
- **Holdout under the rule:** the highest holdout probability is 0.898, so at t = 0.90 the student never fires on any backend. There is **no holdout lift under the declared rule**. The threshold does not carry over from the Trump business returns to individual returns.
- **Diagnostic only:** t = 0.6 is the best training F1 without the false-split rule. It was never selected, but it shows the signal is there. On holdout it scores:
  - LightOn: 0.640 (9 false splits)
  - untagged LightOn: 0.745 (8 false splits)
  - MiniCPM 3-doc subset: 0.761
  - LFM2.5: 0.125

  The false-split rule counts raw splits on 2,882 pages, which is too strict.
- **Latency:** CPU fp32, 8 threads, batch 1, measured while MiniCPM ran on the GPU: embed + classify p50 184.5 ms and p95 201.5 ms per page. Bulk throughput is 88 ms/page. That isn't Hunch's 22 ms (eg2cpu was not built here), but it is about 2% of a MiniCPM page.
- **MiniCPM-OCR 1.0 holdout:** 5.38 pages/min wall, under the 6 pages/min bar, so I cut it to a stratified subset of 3 whole docs, one per subject: harris_2023, obama_2013 and biden_2018 (157 pages planned, 154 OCR pages scored). The output is in a new cache dir.

Full tables are in the PR body.
