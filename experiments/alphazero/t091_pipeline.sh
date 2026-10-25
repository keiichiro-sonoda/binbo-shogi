#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: t091_pipeline.sh <sims:32|48> <selfplay_batch:64|128|256> <iter_seconds_estimate> [base_iters=75] [states_per_iter=65536] [training_batch=128]" >&2
  exit 2
fi

REPO="$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel)"
source "$REPO/scripts/env.sh"
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DATA_ROOT LOGS_DIR MODELS_DIR TEACHER_DIR VENV_PY YANE_EVAL BINBO

exec nice -n 10 "$HOME/task074_frameworks/venv_pgx/bin/python" t091_pipeline.py \
  --sims "$1" --selfplay-batch "$2" --iter-seconds-estimate "$3" \
  --base-iters "${4:-75}" --states-per-iter "${5:-65536}" \
  --training-batch "${6:-128}" --hard-cap 150 --budget-hours 24
