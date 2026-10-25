#!/usr/bin/env bash
# Task 087 T1 step 2 anchor: WS-T (transformer) vs gikou@6 (DepthLimit 6). sims128 /
# byoyomi 100ms, 100 games. Compare to WS2-128 = 0-6-94.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m2_match.py \
  --az-ckpt t087_wst/ws2.ckpt --az-sims 128 \
  --opponent gikou --games 100 --byoyomi 100 --tsv t087_wst_vs_gikou.tsv
