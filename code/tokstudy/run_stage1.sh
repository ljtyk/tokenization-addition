#!/usr/bin/env bash
# Stage 1 (design doc section 8): {k1, gpt2} x {fwd, rev} x {standard, place-value} x 3 seeds = 24 runs.
#   standard    = scaled RoPE on all dims, no place-value index    (--pos rope)
#   place-value = half the RoPE pairs by scaled sequence position, half by the place-value
#                 index, index stretch s~U[1,4], no index table     (--pos irope)
# All: EMA 0.999, 4k steps, periodic checks every 250 steps. Seeds are the OUTER loop, so a
# partial run already covers every cell with seed 0. Cells whose eval_results.json exists are
# skipped, so the script can be re-run after an interruption. Runs one at a time (~7 min each).
# Usage (from the repo root): bash tokstudy/run_stage1.sh
set -u
PY=./.venv/Scripts/python.exe
OUT=tokstudy/out/stage1
mkdir -p $OUT tokstudy/logs/stage1
COMMON="--rope_pos scale --max_iters 4000 --eval_interval 250 --log_interval 500"

for seed in 0 1 2; do
  for scheme in k1-rev k1-fwd gpt2-rev gpt2-fwd; do
    for pos in irope rope; do
      name=${scheme}_${pos}_s${seed}
      if [ -f $OUT/$name/eval_results.json ]; then echo "=== skip $name (done)"; continue; fi
      echo "=== start $name $(date +%H:%M:%S)"
      $PY -u tokstudy/train.py --scheme $scheme --pos $pos --seed $seed $COMMON \
          --out_dir $OUT/$name > tokstudy/logs/stage1/$name.log 2>&1
      echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/stage1/$name.log)"
    done
  done
done
echo "=== all done $(date +%H:%M:%S)"
