#!/usr/bin/env bash
# Task 076 M1: run one measurement point of train_m1.py with VRAM polling.
# usage: m1_run_point.sh <tag> <train_m1.py cli args...>
set -u
cd "$(dirname "$0")"
PY="$HOME/task074_frameworks/venv_pgx/bin/python"
tag="$1"; shift
mkdir -p m1_logs
log="m1_logs/${tag}.log"
vram="m1_logs/${tag}.vram"
( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done > "$vram" 2>/dev/null ) &
poller=$!
WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 5400 "$PY" train_m1.py "$@" > "$log" 2>&1
rc=$?
kill "$poller" 2>/dev/null
wait "$poller" 2>/dev/null
max_vram=$(sort -n "$vram" 2>/dev/null | tail -1)
echo "=== POINT_DONE tag=$tag rc=$rc max_vram_mib=${max_vram:-NA} ==="
tail -4 "$log"
