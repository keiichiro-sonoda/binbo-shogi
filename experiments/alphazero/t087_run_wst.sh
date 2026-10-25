#!/usr/bin/env bash
# Task 087 T1: WS-T = Transformer warm-start distill on m4_data 2M (value+policy).
# One-variable vs WS2-128 (CNN 128ch/10L, m6_ws2): arch differs; lr/seed match.
# batch 256 (NOT 1024): MHA's O(batch*seq^2*d) intermediate OOMs — the fatal train_step
# tensor was 9.9GiB@1024, 4.95GiB@512, both crossing XLA's ~12.9GB per-proc cap on the
# 16GB 4060 Ti. 256 -> ~2.5GiB tensor, ~6.5GB working set = safe. Documented operating
# point + a real finding: the transformer needs ~4x smaller batch than the CNN (1024).
# 8 epochs (best-ckpt selected) -> epoch-4 row = ~equal-budget compare, best-of-8 = convergence.
# d=240/L=5/H=8/ff=4 = 3.585M params (WS2-128 = 3.462M).
set -euo pipefail
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/task074_frameworks/venv_pgx/bin/python" m4_distill_train.py \
  --arch transformer --d-model 240 --num-layers 5 --n-heads 8 --ff-mult 4 \
  --lr 1e-3 --batch 256 --epochs 8 --seed 0 --out t087_wst
