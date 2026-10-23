#!/usr/bin/env bash
# Matched-compute extension (chat 2026-10-18): at 16k steps the GPT-2 BPE cells have used only ~52%
# of the k1 cells' training FLOPs (shorter sequences: 2.26e15 vs 4.34e15) and were still improving.
# Warm-start the 12 GPT-2 cells from tokstudy/out/stage1_16k to 31k total steps, which matches the
# k1 cells' FLOPs at 16k (~1.41e11 FLOPs per GPT-2 step). Same continuation recipe as the 16k
# stage: fresh 500-step warmup + cosine over the continuation, new optimizer, new data seeds.
# Usage (from the repo root): bash tokstudy/run_stage1_gpt2_31k.sh
set -u
PY=./.venv/Scripts/python.exe
SRC=tokstudy/out/stage1_16k
OUT=tokstudy/out/stage1_31k
mkdir -p $OUT tokstudy/logs/stage1_31k
COMMON="--rope_pos scale --max_iters 31000 --warmup 500 --eval_interval 1000 --log_interval 1000"

for seed in 0 1 2; do
  for scheme in gpt2-rev gpt2-fwd; do
    for pos in irope rope; do
      name=${scheme}_${pos}_s${seed}
      if [ -f $OUT/$name/eval_results.json ]; then echo "=== skip $name (done)"; continue; fi
      echo "=== start $name $(date +%H:%M:%S)"
      $PY -u tokstudy/train.py --scheme $scheme --pos $pos --seed $seed $COMMON \
          --init_from $SRC/$name --out_dir $OUT/$name > tokstudy/logs/stage1_31k/$name.log 2>&1
      echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/stage1_31k/$name.log)"
    done
  done
done
echo "=== all done $(date +%H:%M:%S)"
