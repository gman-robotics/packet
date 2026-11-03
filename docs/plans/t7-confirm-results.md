# T7 mid-band VLM confirmer: results (GMA-462)

Date: 2026-11-02. Host: Tom's Mac (M5 Pro). Code: `db56609` (holdout run).
Model: `mlx-community/Qwen3.5-4B-4bit`, revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, through mlx-vlm.
The confirmer is opt-in. The default stays off. A default change needs Tom's OK.

## Result

The confirmer meets the done bar, but it does not help. On the PoliTax holdout it
changed 0 decisions. Holdout macro F1 and false splits are equal to the baseline.

| Set | Band | Macro F1 (on) | Macro F1 (off) | False splits (on / off) | Pages confirmed | Decision changes |
|---|---|---|---|---|---|---|
| builtin | all 3 bands | 1.0000 | 1.0000 | 0 / 0 | 0 | 0 |
| dlnet-whole | all 3 bands | 0.9846 | 0.9846 | 0 / 0 | 0 | 0 |
| PoliTax train | [0.35, 0.74] | 0.1131 | 0.2343 | 23 / 41 | 587 | 21 |
| PoliTax train | [0.45, 0.74] | 0.1131 | 0.2343 | 23 / 41 | 587 | 21 |
| PoliTax train | [0.50, 0.65] | 0.2315 | 0.2343 | 41 / 41 | 90 | 1 |
| PoliTax holdout (single pass) | [0.50, 0.65] | 0.4159 | 0.4159 | 19 / 19 | 21 | 0 |

## Band choice

Rule (plan, step 2): best train macro F1 with false splits not more than the train baseline.
All 3 bands pass the false-split limit. [0.50, 0.65] has the best train macro F1 (0.2315), so it is the band.
Its train macro F1 is 0.0028 lower than the baseline. The holdout pass ran once with this band. There was no re-tuning after it.

The wide bands cut train false splits from 41 to 23, but macro F1 drops by half (0.2343 to 0.1131).
The model removes true boundaries as well as false ones. That trade is not acceptable.

## Cost

- Per-call latency: p50 1.43 s, p95 1.49 s (holdout); p50 about 1.6 s, p95 1.7 to 1.9 s (train).
- Peak RSS: 3.98 GB (holdout), up to 4.23 GB (train).
- Model load: 1.6 to 1.75 s (warm cache).
- Wall time: holdout 33 s; train 148 s ([0.50, 0.65]) and about 16 min (wide bands).
- Degraded pages: none on the holdout.

## Recommendation

Keep the seam and keep the confirmer off. With this model and these bands it adds about 1.5 s per
mid-band page and does not change holdout results. Next options, if Tom wants more work here:
a stronger VLM, a prompt that gives more page context, or the work on the boundary scorer itself.

Raw results: `bench/t7_confirm/results/`.
