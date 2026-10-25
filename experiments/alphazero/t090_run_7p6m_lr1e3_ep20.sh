#!/usr/bin/env bash
# Task 090 Phase 1: 7.6M (d320/L6/H8) @ lr 1e-3, epochs 20 = epochs-only extension of 089's
# 7.6M (which was STABLE + undertrained at 8ep, gap~0, still rising). Tests whether more
# training alone lets 7.6M reach/beat 3.6M (T1: val 0.936/0.472). One-variable vs 089 = epochs.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m4_distill_train.py \
  --arch transformer --d-model 320 --num-layers 6 --n-heads 8 --ff-mult 4 \
  --lr 1e-3 --batch 192 --epochs 20 --seed 0 --warmup 300 --out t090_7p6m_lr1e3ep20
