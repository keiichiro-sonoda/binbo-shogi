#!/usr/bin/env bash
# Task 090 Phase 2 (branch, only if epochs-only@1e-3 plateaus below 3.6M): 7.6M @ lr 3e-4,
# epochs 20. Tests whether a lower (stable) lr finds a better optimum than 1e-3's plateau.
# One-variable vs Phase 1 = lr (1e-3 -> 3e-4). Lower lr converges slower -> watch for plateau.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m4_distill_train.py \
  --arch transformer --d-model 320 --num-layers 6 --n-heads 8 --ff-mult 4 \
  --lr 3e-4 --batch 192 --epochs 20 --seed 0 --warmup 300 --out t090_7p6m_lr3e4ep20
