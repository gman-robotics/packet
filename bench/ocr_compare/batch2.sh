#!/bin/zsh
cd /Users/Shared/gman-sandbox/ocr-compare; export HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
for m in plain grounding; do mkdir -p runs/tmp-$m; DT=fp32 venv-tf/bin/python bench/run_tf.py lighton-1b $m runs/tmp-$m testset/synth/hs-*.png > logs/lighton-1b-$m-hs.log 2>&1; cp runs/tmp-$m/hs-*.txt runs/lighton-1b-$m-fp32/; mv runs/tmp-$m/_summary.json runs/lighton-1b-$m-fp32/_summary_hs.json; echo "1b $m hs done $(date +%T)"; done
bench/run_all.sh lighton-0.8b:plain:bf16 lighton-0.8b:grounding:bf16 lighton-4b:plain:bf16 lighton-4b:grounding:bf16
unset HF_HUB_OFFLINE; export PADDLE_PDX_CACHE_HOME=$PWD/paddlex-cache PADDLE_PDX_MODEL_SOURCE=huggingface PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
echo "=== $(date +%T) START paddle"; /usr/bin/time -l venv-paddle/bin/python bench/run_paddle.py runs/paddle-vl-1.6-pipeline-cpu testset/synth/*.png testset/packet/pk-*.png testset/public/* > logs/paddle-pipeline.log 2>&1; echo "=== $(date +%T) END paddle rc=$?"; grep -E '"pages_per_min"' logs/paddle-pipeline.log; grep "maximum resident" logs/paddle-pipeline.log
echo BATCH2_DONE
