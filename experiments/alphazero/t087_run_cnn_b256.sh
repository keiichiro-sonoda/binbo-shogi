#!/usr/bin/env bash
# Task 087 T1 fairness control: CNN (WS2-128 arch, 128ch/10L) at batch 256 — matches
# the Transformer run EXACTLY except arch (batch 256, 8 epochs, lr 1e-3, seed 0). This
# isolates architecture from the batch/step-count confound (transformer was forced to
# 256 by MHA memory; WS2-128 was trained at 1024, so equal-epoch != equal-steps).
# Compare WS-T (transformer, b256) vs this CNN-b256 = strict arch-only on val corr/top1.
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m4_distill_train.py \
  --arch cnn --num-channels 128 --num-layers 10 \
  --lr 1e-3 --batch 256 --epochs 8 --seed 0 --out t087_cnn_b256
