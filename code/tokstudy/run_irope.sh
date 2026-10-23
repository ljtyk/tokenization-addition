#!/usr/bin/env bash
# Index-RoPE test (chat 2026-10-17): k1-rev, half the RoPE pairs by scaled sequence position,
# half by place-value index (no index table), EMA, 4k steps. With index stretch s~U[1,4]
# (is4) and without (is1), 3 seeds each, run one at a time.
# Usage (from the repo root): bash tokstudy/run_irope.sh
set -u
PY=./.venv/Scripts/python.exe
COMMON="--scheme k1-rev --pos irope --rope_pos scale --max_iters 4000 --eval_interval 250 --log_interval 500"
mkdir -p tokstudy/logs
run() {  # name, extra args...
  local name=$1; shift
  echo "=== start $name $(date +%H:%M:%S)"
  $PY -u tokstudy/train.py $COMMON --out_dir tokstudy/out/$name "$@" > tokstudy/logs/$name.log 2>&1
  echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/$name.log)"
}
for s in 0 1 2; do run ema4k_k1-rev_irope-is4_s$s --seed $s --index_rope_scale_max 4; done
for s in 0 1 2; do run ema4k_k1-rev_irope-is1_s$s --seed $s --index_rope_scale_max 1; done
echo "=== all done $(date +%H:%M:%S)"
