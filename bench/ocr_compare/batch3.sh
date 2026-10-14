#!/bin/zsh
BENCH=${0:A:h}  # this script's directory (bench/ocr_compare)
cd "${OCR_COMPARE_ROOT:?set OCR_COMPARE_ROOT to the ocr-compare work dir: venvs, hf/, testset/, runs/, logs/}"
export HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
while pgrep -f batch_mlx.sh >/dev/null; do sleep 5; done
IMGS=(testset/synth/*.png testset/packet/pk-*.png testset/public/*)
mv runs/mlx-lighton-1b-plain runs/mlx-lighton-1b-plain-BADTEMPLATE
echo "=== $(date +%T) 1b plain fixed"; MAXTOK=2048 /usr/bin/time -l venv-mlx/bin/python $BENCH/run_mlx.py lighton-1b plain runs/mlx-lighton-1b-plain $IMGS > logs/mlx-lighton-1b-plain.log 2>&1; grep pages_per logs/mlx-lighton-1b-plain.log
for k in lighton-0.8b lighton-1b lighton-4b; do echo "=== $(date +%T) news $k"; MAXTOK=6000 venv-mlx/bin/python $BENCH/run_mlx.py $k grounding runs/news6k-$k testset/public/paddle-demo.png > logs/news6k-$k.log 2>&1; grep pages_per logs/news6k-$k.log; done
echo B3_DONE
