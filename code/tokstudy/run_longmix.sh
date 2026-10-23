#!/usr/bin/env bash
# Does the model learn a robust "nothing here" on its own if long problems make it necessary? (chat 2026-10-19)
# k1-rev, index-RoPE + stretch (the main study's place-value encoding, no attention changes), warm start
# from stage1_16k at 16k -> 24k steps (fresh warmup + cosine, see train.py --init_from), 3 seeds.
#   mix:     10% of training problems have a 20-40 digit long operand (--long_frac 0.1)
#   control: the same continuation on the normal 1-10 digit data (matched steps)
# Reach is then probed to 400 digits: a coverage dial predicts ~4x the longest training length (~160);
# a learned mechanism would go well beyond.
# Usage (from the repo root): bash tokstudy/run_longmix.sh mix|control
set -u
PY=./.venv/Scripts/python.exe
SRC=tokstudy/out/stage1_16k
OUT=tokstudy/out/longmix
mkdir -p $OUT tokstudy/logs/longmix
arm=$1
EXTRA=""
[ "$arm" = mix ] && EXTRA="--long_frac 0.1 --long_min 20 --long_max 40"
for s in 0 1 2; do
  name=k1-rev_irope_${arm}_s$s
  if [ -f $OUT/$name/eval_results.json ]; then echo "=== skip $name (done)"; continue; fi
  echo "=== start $name $(date +%H:%M:%S)"
  $PY -u tokstudy/train.py --scheme k1-rev --pos irope --rope_pos scale --seed $s --max_iters 24000 --warmup 500 \
      --eval_interval 1000 --log_interval 1000 $EXTRA --init_from $SRC/k1-rev_irope_s$s --out_dir $OUT/$name \
      > tokstudy/logs/longmix/$name.log 2>&1
  echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/longmix/$name.log)"
done
echo "=== all done $(date +%H:%M:%S)"
