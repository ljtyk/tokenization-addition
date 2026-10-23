#!/usr/bin/env bash
# Matched-compute extension for k=3 (chat 2026-10-18): at 16k steps the k3 cells have used only
# ~48% of the k1 cells' training FLOPs (2.09e15 vs 4.34e15; same 4.1M problems, but ~2.1x fewer
# tokens). Warm-start all 24 k3 cells from tokstudy/out/k3_16k to 33k total steps (~1.31e11 FLOPs
# per k3 step), which matches k1's FLOPs at 16k. Same recipe as run_stage1_gpt2_31k.sh: fresh
# 500-step warmup + cosine over the continuation, new optimizer, new data seeds.
# Usage (from the repo root): bash tokstudy/run_k3_33k.sh
set -u
PY=./.venv/Scripts/python.exe
SRC=tokstudy/out/k3_16k
OUT=tokstudy/out/k3_33k
mkdir -p $OUT tokstudy/logs/k3_33k
COMMON="--rope_pos scale --max_iters 33000 --warmup 500 --eval_interval 1000 --log_interval 1000"

for seed in 0 1 2; do
  for scheme in k3-right-rev k3-left-fwd k3-right-fwd k3-left-rev; do
    for pos in irope rope; do
      name=${scheme}_${pos}_s${seed}
      if [ -f $OUT/$name/eval_results.json ]; then echo "=== skip $name (done)"; continue; fi
      if [ ! -f $SRC/$name/ckpt.pt ]; then echo "=== missing $SRC/$name, skipped"; continue; fi
      echo "=== start $name $(date +%H:%M:%S)"
      $PY -u tokstudy/train.py --scheme $scheme --pos $pos --seed $seed $COMMON \
          --init_from $SRC/$name --out_dir $OUT/$name > tokstudy/logs/k3_33k/$name.log 2>&1
      echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/k3_33k/$name.log)"
    done
  done
done
echo "=== all done $(date +%H:%M:%S)"
