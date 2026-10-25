#!/usr/bin/env bash
# Task 087 T1 step 2 gate: WS-T (transformer, best net) vs WS2-128 (CNN, established
# best warm-start). Both AZ USI engines, sims128, byoyomi 100ms. score>0.5 = WS-T wins.
# Economical gate: if WS-T is competitive here, proceed to vs-champion; if crushed, stop.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m2_match.py \
  --az-ckpt t087_wst/ws2.ckpt --az-sims 128 \
  --opponent az2 --opp-ckpt m6_ws2/ws2.ckpt --opp-sims 128 \
  --games 40 --byoyomi 100 --tsv t087_wst_vs_ws2128.tsv
