#!/usr/bin/env bash
# Task 087 T1 step 2 MAIN JUDGE: WS-T (transformer) vs champion (task064 32M NNUE,
# YaneuraOu). sims128 / byoyomi 100ms. Direct comparison to WS2-128's 2-3-95 (elo -576).
# GAMES via $1 (default 40 probe; expand to 100-200 if competitive).
set -euo pipefail
GAMES="${1:-40}"
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m2_match.py \
  --az-ckpt t087_wst/ws2.ckpt --az-sims 128 \
  --opponent yane --games "$GAMES" --byoyomi 100 --tsv t087_wst_vs_champion.tsv
