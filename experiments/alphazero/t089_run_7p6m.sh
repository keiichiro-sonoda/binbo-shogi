#!/usr/bin/env bash
# Task 089 T2-scale point: 7.6M Transformer (d320/L6/H8/ff4) warm-start distill on 2M.
# One-variable vs T1's 3.6M (d240/L5): only params differ; data/epochs/lr/seed/warmup match.
# batch 192 (256 OOMs on 16GB: XLA arena can't place the large contiguous MHA intermediate).
# Operating point from an OOM probe; batch spread (3.6M@256 / 7.6M@192 / 14.4M@128) is a minor
# documented confound (train-val gap diagnostic + match noise dominate over it).
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m4_distill_train.py \
  --arch transformer --d-model 320 --num-layers 6 --n-heads 8 --ff-mult 4 \
  --lr 1e-3 --batch 192 --epochs 8 --seed 0 --warmup 300 --out t089_7p6m
