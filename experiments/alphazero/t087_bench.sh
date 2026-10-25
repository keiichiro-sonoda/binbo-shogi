#!/usr/bin/env bash
# Task 087 T1 step 3: play-time per-move latency, Transformer (WS-T) vs CNN (WS2-128),
# identical sims sweep & position count. JIT-warm call is discarded inside bench().
# states/s ~= sims / (ms/move); relative FLOP-efficiency = CNN_ms / T_ms at equal sims.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="$HOME/task074_frameworks/venv_pgx/bin/python"
SWEEP="4,8,16,32,64,128"
N=30
echo "===== BENCH transformer (WS-T = t087_wst/ws2.ckpt, 3.585M) ====="
"$PY" m2_usi.py --ckpt t087_wst/ws2.ckpt --bench "$N" --bench-sims "$SWEEP"
echo "===== BENCH cnn (WS2-128 = m6_ws2/ws2.ckpt, 3.462M) ====="
"$PY" m2_usi.py --ckpt m6_ws2/ws2.ckpt --bench "$N" --bench-sims "$SWEEP"
echo "BENCH_ALL_DONE"
