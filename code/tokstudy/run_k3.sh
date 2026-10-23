#!/usr/bin/env bash
# k=3 chunking (chat 2026-10-18): k3 x {right, left} x {rev, fwd} x {place-value (irope), standard (rope)}
# x 3 seeds = 24 runs, 16k steps from scratch (cosine over 16k, EMA 0.999). Seeds are the outer loop
# and the place-aligned (right-rev, scheme 1) and GPT-style (left-fwd, scheme 3) cells come first
# within each seed. k3 sequences are ~2.4x shorter than k1's, so 16k steps is ~40% of k1's
# 16k-step FLOPs; a matched-compute warm-start extension may follow. Finished cells are skipped.
# Usage (from the repo root): bash tokstudy/run_k3.sh
set -u
PY=./.venv/Scripts/python.exe
OUT=tokstudy/out/k3_16k
mkdir -p $OUT tokstudy/logs/k3_16k
COMMON="--rope_pos scale --max_iters 16000 --eval_interval 500 --log_interval 1000"

for seed in 0 1 2; do
  for scheme in k3-right-rev k3-left-fwd k3-right-fwd k3-left-rev; do
    for pos in irope rope; do
      name=${scheme}_${pos}_s${seed}
      if [ -f $OUT/$name/eval_results.json ]; then echo "=== skip $name (done)"; continue; fi
      echo "=== start $name $(date +%H:%M:%S)"
      $PY -u tokstudy/train.py --scheme $scheme --pos $pos --seed $seed $COMMON \
          --out_dir $OUT/$name > tokstudy/logs/k3_16k/$name.log 2>&1
      echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/k3_16k/$name.log)"
    done
  done
done
echo "=== all done $(date +%H:%M:%S)"
