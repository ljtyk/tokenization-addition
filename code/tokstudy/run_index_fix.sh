#!/usr/bin/env bash
# Index-fix comparison (see design doc / chat 2026-10-17): k1-rev, scaled RoPE + place-value,
# EMA, 4k steps. Baseline seed 2 (seeds 0-1 already exist as ema4k_k1-rev_pv-scale_s{0,1}),
# separator end-index x3 seeds, stretched index (stride 1-3) x3 seeds. Runs one at a time:
# two concurrent runs were slower in total than sequential ones on the 4080 Laptop.
# Usage (from the repo root): bash tokstudy/run_index_fix.sh
set -u
PY=./.venv/Scripts/python.exe
COMMON="--scheme k1-rev --pos pv --rope_pos scale --max_iters 4000 --eval_interval 250 --log_interval 500"
mkdir -p tokstudy/logs

run() {  # name, extra args...
  local name=$1; shift
  echo "=== start $name $(date +%H:%M:%S)"
  $PY -u tokstudy/train.py $COMMON --out_dir tokstudy/out/$name "$@" > tokstudy/logs/$name.log 2>&1
  echo "=== end $name $(date +%H:%M:%S) exit $? | $(grep -h 'late_avg' tokstudy/logs/$name.log)"
}

run ema4k_k1-rev_pv-scale_s2 --seed 2
for s in 0 1 2; do run ema4k_k1-rev_pv-scale-sep_s$s --seed $s --sep_index; done
for s in 0 1 2; do run ema4k_k1-rev_pv-scale-st3_s$s --seed $s --index_stride_max 3; done
echo "=== all done $(date +%H:%M:%S)"
