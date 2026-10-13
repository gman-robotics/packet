#!/bin/zsh
# GMA-207 driver: sequential runs, one model at a time (memory isolation).
cd /Users/Shared/gman-sandbox/ocr-compare
export HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 MAXTOK=${MAXTOK:-2048}
IMGS=(testset/synth/*.png testset/packet/pk-*.png testset/public/*)
for spec in "$@"; do
  key=${spec%%:*}; rest=${spec#*:}; mode=${rest%%:*}; dt=${rest#*:}
  out=runs/$key-$mode-$dt
  echo "=== $(date +%T) START $out"
  DT=$dt /usr/bin/time -l venv-tf/bin/python bench/run_tf.py $key $mode $out $IMGS > logs/$key-$mode-$dt.log 2>&1
  echo "=== $(date +%T) END $out rc=$?"; grep -E '"pages_per_min"' logs/$key-$mode-$dt.log | tail -1
  grep "maximum resident" logs/$key-$mode-$dt.log
done
echo ALL_DONE
