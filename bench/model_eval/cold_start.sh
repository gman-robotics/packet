#!/bin/zsh
# GMA-441 cold start without sudo purge: copy each pinned snapshot with `ditto --nocache` (bypasses the unified
# buffer cache, so the new files' pages are not resident), then load from the copy in a fresh process (cold),
# then load again from the same copy in a second fresh process (warm). One page each (md-00). Run from the worktree.
# usage: cold_start.sh OUTROOT
set -e; OUT=$1; mkdir -p $OUT; export HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface
HUB=~/.cache/huggingface/hub; IMG=testset/synth/md-00.png
copy() { local src=$HUB/models--${1//\//--}/snapshots/$2 dst=$OUT/weights/$3; rm -rf $dst; mkdir -p $dst
  for f in $src/*(.,@); do ditto --nocache "${f:A}" "$dst/${f:t}"; done; }
copy lightonai/LightOnOCR-3-1B 23d544085c886bbdace647bcfab91203299417e2 lighton
copy LiquidAI/LFM2.5-VL-1.6B-Extract 21073aa65b326c125ceb07d80fa08900bc2eb0d9 lfm25x
copy LiquidAI/LFM3-VL-2B f83d2b0a2296aaa277f394990a9b8b949ddf9691 lfm3
for t in cold warm; do
  /usr/bin/time -l .venv/bin/python bench/model_eval/run_lfm_extract.py $OUT/weights/lfm25x $OUT/lfm25x-$t $IMG > $OUT/lfm25x-$t.log 2>&1
  /usr/bin/time -l /Users/Shared/gman-sandbox/lfm3-mps/venv/bin/python bench/model_eval/run_lfm3_mps.py $OUT/weights/lfm3 $OUT/lfm3-$t $IMG > $OUT/lfm3-$t.log 2>&1
  /usr/bin/time -l .venv/bin/python -c "
import time; T0=time.time()
import json, mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
T1=time.time(); m,p=load('$OUT/weights/lighton'); mx.eval(m.parameters()); T2=time.time()
fp=apply_chat_template(p,m.config,'',num_images=1).replace('<|image_pad|>\n<|im_end|>','<|image_pad|><|im_end|>')
r=generate(m,p,fp,image=['$IMG'],max_tokens=2048,temperature=0.0,verbose=False); T3=time.time()
print('SUMMARY',json.dumps({'imports_s':round(T1-T0,2),'load_s':round(T2-T1,2),'process_to_loaded_s':round(T2-T0,2),'first_page_s':round(T3-T2,2)}))
" > $OUT/lighton-$t.log 2>&1
done
echo COLD_DONE
