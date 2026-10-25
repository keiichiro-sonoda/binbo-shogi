#!/usr/bin/env bash
# Task 087 T1 strict arch-only match: WS-T (transformer, b256) vs CNN-b256 (same batch,
# epochs, lr, seed -> ONLY arch differs). Removes the batch/step confound from the
# headline 34-6-0 (which used CNN-b1024). sims128 / byoyomi 100ms. score>0.5 = WS-T.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m2_match.py \
  --az-ckpt t087_wst/ws2.ckpt --az-sims 128 \
  --opponent az2 --opp-ckpt t087_cnn_b256/ws2.ckpt --opp-sims 128 \
  --games 40 --byoyomi 100 --tsv t087_wst_vs_cnnb256.tsv
