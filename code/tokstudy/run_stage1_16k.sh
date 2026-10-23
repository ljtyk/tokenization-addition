#!/usr/bin/env bash
# Stage 1 continued to 16k steps (chat 2026-10-17): every 4k-step stage 1 cell (tokstudy/out/stage1,
# kept as the learning-speed snapshot) is warm-started and trained to 16k total steps. The
# continuation gets a fresh 500-step warmup + cosine over steps 4k-16k, a re-initialized optimizer
# and new data seeds (see train.py --init_from). Seeds are the outer loop; finished cells are
# skipped, so the script can be re-run after an interruption.
# Usage (from the repo root): bash tokstudy/run_stage1_16k.sh
set -u
PY=./.venv/Scripts/python.exe
SRC=tokstudy/out/stage1
OUT=tokstudy/out/stage1_16k
mkdir -p $OUT tokstudy/logs/stage1_16k
COMMON="--rope_pos scale --max_iters 16000 --warmup 500 --eval_interval 500 --log_interval 1000"

for seed in 0 1 2; do
  for scheme in k1-rev k1-fwd gpt2-rev gpt2-fwd; do
    for pos in irope rope; do
      name=${scheme}_${pos}_s${seed}
      if [ -f $OUT/$name/eval_results.json ]; then echo "=== skip $name (done)"; continue; fi
      echo "=== start $name $(date +%H:%M:%S)"
      $PY -u tokstudy/train.py --scheme $scheme --pos $pos --seed $seed $COMMON \
          --init_from $SRC/$name --out_dir $OUT/$name > tokstudy/logs/stage1_16k/$name.log 2>&1
      echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/stage1_16k/$name.log)"
    done
  done
done
echo "=== all done $(date +%H:%M:%S)"
