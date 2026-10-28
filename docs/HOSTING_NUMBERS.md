# Local hosting numbers: OCR engines and checker (GMA-442, provisional)

These numbers are re-reported from existing Exp A ([GMA-441](https://linear.app/gman-robotics/issue/GMA-441), PR #61) and
Exp A2 ([GMA-445](https://linear.app/gman-robotics/issue/GMA-445), PR #62) artifacts. Nothing was re-run for this doc
except the thermal check. Every number names its source file under `~/.cache/packet/model-eval/p1/` (written `P1/`).
This is a docs record only. No Packet defaults change.

**Host:** Mac17,9, Apple M5 Pro, 64 GB unified memory, macOS 27.0.1. Single process, one engine at a time, bf16 checkpoint weights.

## Runtime and version

| Engine | HF repo @ revision | Runtime | Venv |
|---|---|---|---|
| LightOnOCR-3-1B (default) | `lightonai/LightOnOCR-3-1B` @ `23d544085c886bbdace647bcfab91203299417e2` | mlx-vlm 0.7.10 / mlx 0.32.4 | repo `.venv` |
| LFM2.5-VL-1.6B-Extract | `LiquidAI/LFM2.5-VL-1.6B-Extract` @ `21073aa65b326c125ceb07d80fa08900bc2eb0d9` | mlx-vlm 0.7.10 / mlx 0.32.4 | repo `.venv` |
| LFM3-VL-2B | `LiquidAI/LFM3-VL-2B` @ `f83d2b0a2296aaa277f394990a9b8b949ddf9691` | transformers 5.21.0 / torch 2.14.1 MPS, `trust_remote_code` | sandbox `/Users/Shared/gman-sandbox/lfm3-mps/venv`, `HF_HUB_OFFLINE=1` |
| MiniCPM-OCR 1.0 | `openbmb/MiniCPM-OCR-1.0` @ `6796197bb3a8e224b2f64ae462bfbb3f8a15ac4b` | transformers 5.21.0 / torch 2.14.1 MPS, native | sandbox venv |
| MiniCPM-OCR-Jev (checker) | `openbmb/MiniCPM-OCR-Jev` @ `90bc4c4768c082f769f5352479fa0df84da42c1a` | transformers 5.21.0 / torch 2.14.1 MPS, native `qwen3_5` + manual `score.weight` head | sandbox venv |

Sources: `P1/runs/*/_summary.json` (`backend`, `revision`); `P1/runs/*/_jev.json` (`rev`).
MiniCPM-OCR on Homebrew llama.cpp 0.7.0 failed (`unknown projector type minicpm_ocr_1_0`), so it is not a hosting option today.

## Cold and warm start

Method (`bench/model_eval/cold_start.sh`, PR #61): no `sudo purge`. Each pinned snapshot is copied with `ditto --nocache`
so its pages are not resident. A fresh process loads it (cold), then a second fresh process loads it again (warm). One page (`md-00`) each.

| Engine | Cold: imports + load = process→loaded (s) | Warm: process→loaded (s) | Cold / warm wall incl. 1 page (s) | 1st page (s) |
|---|---|---|---|---|
| LightOn-1B | 1.65 + 0.41 = 2.06 | 1.96 | 6.22 / 6.05 | 3.73 (cold), 3.69 (warm) |
| LFM2.5-Extract | 2.63 + 0.39 = 3.02 | 1.93 | 5.57 / 4.49 | 2.1 |
| LFM3-VL-2B | 1.39 + 1.56 = 2.95 | not in artifact | 6.55 / 5.98 | 3.2 |
| MiniCPM-OCR 1.0 | 1.41 + 1.44 = 2.85 | 2.21 | 7.66 / 7.08 | 4.4 |
| MiniCPM-OCR-Jev | not measured | load 4.96 (snapshot in page cache) | not measured | n/a |

Sources: `P1/coldstart/{engine}-{cold,warm}.log` (`SUMMARY` line and `/usr/bin/time -l` "real"); `P1/coldstart/*/_summary.json`.
Jev load comes from the Exp A2 parity run log (`load_s`).

## Throughput and memory

| Engine | Testset pages/min (24 pp) | PoliTax holdout pages/min | Peak memory, 1-page cold run (`/usr/bin/time -l` footprint) | Accelerator peak, full testset run |
|---|---|---|---|---|
| LightOn-1B plain | 24.29 | — | 10.29 GB | mlx 10.21 GB |
| LightOn-1B grounding | 20.2 | 3.87 | 10.29 GB | mlx 10.21 GB |
| LFM2.5-Extract | 21.83 | 26.1 | 5.63 GB | mlx 4.1 GB |
| LFM3-VL-2B | 27.63 | 4.88 (49 pp) | 5.48 GB | MPS driver 4.78 GB (RSS 2.12 GB) |
| MiniCPM-OCR 1.0 | 16.81 | not run | 6.31 GB | MPS driver 7.65 GB (RSS 1.1 GB) |
| Jev checker, per region (LightOn blocks) | 15.5 | 6.89 | not measured | not measured |
| Jev checker, whole page | 22.9–25.1 | 12.22 (LFM3), 26.72 (LFM2.5x) | not measured | not measured |

Sources: testset numbers from `P1/runs/*/_summary.json` (`pages_per_min`, `mlx_peak_gb`, `mps_driver_peak_gb`, `peak_rss_gb`).
PoliTax OCR numbers from the Exp A run summaries (`pages_per_min_wall`) as reported in PR #61. Jev numbers from
`P1/a2/dev.json` and `P1/a2/holdout.json` (`jev_pages_per_min`). PoliTax pages are dense tax forms, so throughput is much lower there than on the testset.

## Thermals

`pmset -g therm` (no sudo) was sampled 12 times at 5 s intervals during a 59 s Jev whole-page run on MPS
(24 pages, 24.4 pages/min). Every sample reported "No thermal warning level has been recorded", "No performance warning
level has been recorded" and "No CPU power status has been recorded". This shows that no OS thermal or performance
warning was raised. It is not a temperature reading. Die temperature, fan speed and sustained-run throttling are
**not measured** (they need `powermetrics`, which needs sudo). The OCR engines were not thermally checked: **not measured**.
Sources: `P1/a2/therm_before.txt`, `P1/a2/therm_during.txt`, `P1/a2/therm_run.log`.
