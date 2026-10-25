#!/usr/bin/env bash
# Task 089 T2-scale point: 14.4M Transformer (d384/L8/H8/ff4) warm-start distill on 2M.
# batch 128 (256/192 OOM on 16GB). Operating point from OOM probe (documented confound).
# On 2M this is ~7:1 params:examples = the overfit-boundary diagnostic point (watch gap_*).
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m4_distill_train.py \
  --arch transformer --d-model 384 --num-layers 8 --n-heads 8 --ff-mult 4 \
  --lr 1e-3 --batch 128 --epochs 8 --seed 0 --warmup 300 --out t089_14p4m
