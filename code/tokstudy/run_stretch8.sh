#!/usr/bin/env bash
# Coverage-dial test (chat 2026-10-17): is index-RoPE's ~40-50 digit limit set by the stretch
# factor? k1-rev irope, EMA, 4k steps. Both stretches (sequence RoPE and index) raised 4 -> 8
# on 2 seeds, plus the index stretch alone raised to 8 (sequence stays 4) on 1 seed, to see
# which stretch binds. Compare with ema4k_k1-rev_irope-is4_s* via probe_length.py.
# Usage (from the repo root): bash tokstudy/run_stretch8.sh
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
for s in 0 1; do run ema4k_k1-rev_irope-is8-rs8_s$s --seed $s --index_rope_scale_max 8 --rope_scale_max 8; done
run ema4k_k1-rev_irope-is8-rs4_s0 --seed 0 --index_rope_scale_max 8 --rope_scale_max 4
echo "=== all done $(date +%H:%M:%S)"
