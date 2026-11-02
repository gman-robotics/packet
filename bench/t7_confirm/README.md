# T7 mid-band VLM confirm

Opt-in only. `PipelineConfig.confirmer` stays `None` unless this harness, or
`--confirm-config`, passes a `BackendConfirmer`. The 0.74 cap is untouched.
This directory does not download a model and does not contain results.
`docs/plans/t7-confirm-results.md` is written after a local run, for Morgan or
Piper to copy into tmg-wiki. That wiki checkout is not on this Mac.

## Model

Primary, already cached under `~/.cache/huggingface/hub`:

- `mlx-community/Qwen3.5-4B-4bit`
- revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`
- `model_type` `qwen3_5`

Manual fallback only, if that load or its answers fail. Do not auto-switch.
No revision is pinned here; do not invent one:

- `mlx-community/gemma-4-e2b-it-4bit`

Runtime, with the worktree on `PYTHONPATH` (this tree is not installed into
that venv; do not pip install):

```
PY=/Users/tginter/dev/gman-robotics/packet-gma389/.venv/bin/python
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
```

`HF_HUB_OFFLINE=1` keeps `snapshot_download` on the cache. The venv already
has Python 3.13, mlx 0.32.4, mlx_vlm 0.7.10, and transformers 5.21.0.

Wrap a real run with `/usr/bin/time -l` and keep that log next to the JSON.
The JSON `peak_rss_bytes` is `ru_maxrss` (bytes on macOS). Stop at 6 hours
from the first model load.

## Bands

Train picks one band. The 0.74 cap stays where it is.

- [0.35, 0.74]
- [0.45, 0.74]
- [0.50, 0.65]

`--write-config PATH` writes one confirm JSON for `--low` / `--high`, or for
[0.35, 0.74] when those flags are omitted. It does not load the model.

## Train files

PoliTax has no dev split. Train is `subject=trump` from `evals/politax.json`
(`POLITAX_TRAIN_SUBJECT` / `partition_docs`). Holdout is obama, harris, and
biden, whole files. The bench filters with `partition_docs`; this list is the
lock:

- trump_business_2015_DJT Holdings 2015.pdf
- trump_business_2015_DJT Holdings Managing Member LLC 2015.pdf
- trump_business_2015_LFB Acquisition LLC 2015.pdf
- trump_business_2016_DJT Holdings 2016.pdf
- trump_business_2016_DJT Holdings Managing Member LLC 2016.pdf
- trump_business_2016_LFB Acquisition LLC 2016.pdf
- trump_business_2017_DJT Holdings 2017.pdf
- trump_business_2017_LFB Acquisition LLC 2017.pdf
- trump_business_2018_DTTM Operations 2018.pdf
- trump_business_2018_LFB Acquisition LLC 2018.pdf
- trump_business_2019_DJT Holdings Managing Member LLC 2019 - 2.pdf
- trump_business_2019_DTTM Operations 2019.pdf
- trump_business_2020_DJT Holdings 2020.pdf
- trump_business_2020_DJT Holdings Managing Member LLC 2020 - 1.pdf
- trump_business_2020_DTTM Operations 2020.pdf
- trump_business_2020_LFB Acquisition LLC 2020.pdf
- trump_individual_2015_Form 1040 2015-1.pdf
- trump_individual_2015_amt_Form 1040 2015-3.pdf
- trump_individual_2016_Form 1040 2016-2.pdf
- trump_individual_2017_Form 1040 2017-1.pdf
- trump_individual_2018_Form 1040 2018-2.pdf
- trump_individual_2019_Form 1040 2019-3.pdf
- trump_individual_2020_Form 1040 2020-2.pdf

Do not tune with `packet eval --set politax --ocr-cache`. That parses every
fixture and gates on holdout. PoliTax rows here go through `ocr_cache_pipeline`,
the same cache feed as `run_with_ocr_cache`, and only the selected subset.

## Run

```
HF_HUB_OFFLINE=1 PYTHONPATH=src \
  /usr/bin/time -l $PY bench/t7_confirm/run.py --set politax --subset train
HF_HUB_OFFLINE=1 PYTHONPATH=src \
  $PY bench/t7_confirm/run.py --set builtin
HF_HUB_OFFLINE=1 PYTHONPATH=src \
  $PY bench/t7_confirm/run.py --set dlnet-whole
HF_HUB_OFFLINE=1 PYTHONPATH=src \
  $PY bench/t7_confirm/run.py --set all
```

`--set all` is builtin plus dlnet-whole. PoliTax is never inside `all`.

One train pass over the three bands. Pick the eligible band with the best
train macro F1. Eligible means false splits (`len(predicted) - matched`,
summed) are less than or equal to the confirmer-off baseline on that same
subset. Then one holdout pass, and only that pass:

```
HF_HUB_OFFLINE=1 PYTHONPATH=src \
  /usr/bin/time -l $PY bench/t7_confirm/run.py \
  --set politax --subset holdout --low LOW --high HIGH
```

The script does not launch holdout by itself. A summary JSON records
`best_eligible` and `"holdout": "not run by this script"`.

Done when holdout macro F1 is at least 0.41 and holdout false splits are less
than or equal to the holdout baseline. If that misses, the seam still stays
off by default.

## Result JSON

Written under `bench/t7_confirm/results/` only when the script runs:

- `model`, `revision`, `set`, `subset`, `low`, `high`
- `load_seconds`, `wall_seconds`, `peak_rss_bytes`
- `pages_confirmed` (`confirmed_split` + `confirmed_continue`)
- `decision_changes` (fixtures whose predicted boundaries differ from confirmer-off)
- `macro_f1`, `false_splits`, `baseline_macro_f1`, `baseline_false_splits`, `eligible`
- `latency_p50`, `latency_p95` (nearest-rank, seconds per `decide`)
- `degraded_pages` copied after each file, before the next file's index-0 reset
- `fixtures` and, for PoliTax, `ocr_cache`

Confirmer-off on PoliTax uses the same OCR cache and no confirmer.
