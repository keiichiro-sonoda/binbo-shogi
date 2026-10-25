#!/usr/bin/env bash
# Task 089 T2-scale evaluation for one scale point (same物差し as T1): internal vs the
# 3.6M anchor (t087_wst), then external vs champion + gikou@6, then throughput bench.
# Usage: t089_eval.sh <ckpt-rel-path> <tag>   e.g. t089_eval.sh t089_7p6m/ws2.ckpt 7p6m
set -euo pipefail
CKPT="$1"; TAG="$2"
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="$HOME/task074_frameworks/venv_pgx/bin/python"
echo "===== EVAL $TAG ($CKPT) ====="
echo "----- internal: $TAG vs WS-T-3.6M (t087_wst), sims128 -----"
"$PY" m2_match.py --az-ckpt "$CKPT" --az-sims 128 --opponent az2 \
  --opp-ckpt t087_wst/ws2.ckpt --opp-sims 128 --games 40 --byoyomi 100 --tsv "t089_${TAG}_vs_3p6m.tsv"
echo "----- external: $TAG vs champion (task064 NNUE), sims128/0.1s -----"
"$PY" m2_match.py --az-ckpt "$CKPT" --az-sims 128 --opponent yane \
  --games 40 --byoyomi 100 --tsv "t089_${TAG}_vs_champion.tsv"
echo "----- external: $TAG vs gikou@6, sims128/0.1s -----"
"$PY" m2_match.py --az-ckpt "$CKPT" --az-sims 128 --opponent gikou \
  --games 100 --byoyomi 100 --tsv "t089_${TAG}_vs_gikou.tsv"
echo "----- throughput bench -----"
"$PY" m2_usi.py --ckpt "$CKPT" --bench 30 --bench-sims 4,8,16,32,64,128
echo "EVAL_DONE_$TAG"
