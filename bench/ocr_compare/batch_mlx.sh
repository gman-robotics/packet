#!/bin/zsh
cd /Users/Shared/gman-sandbox/ocr-compare; export HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 MAXTOK=2048
while pgrep -f run_paddle.py >/dev/null; do sleep 5; done
IMGS=(testset/synth/*.png testset/packet/pk-*.png testset/public/*)
for spec in "$@"; do key=${spec%%:*}; mode=${spec#*:}; out=runs/mlx-$key-$mode
  echo "=== $(date +%T) START $out"; /usr/bin/time -l venv-mlx/bin/python bench/run_mlx.py $key $mode $out $IMGS > logs/mlx-$key-$mode.log 2>&1
  echo "=== $(date +%T) END $out rc=$?"; grep '"pages_per_min"' logs/mlx-$key-$mode.log; grep "maximum resident" logs/mlx-$key-$mode.log; done
echo MLX_DONE
