#!/usr/bin/env bash
set -euo pipefail

REPO="$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel)"
source "$REPO/scripts/env.sh"
cd "$HOME/task074_frameworks/pgx/examples/alphazero"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DATA_ROOT LOGS_DIR MODELS_DIR TEACHER_DIR VENV_PY YANE_EVAL BINBO

exec nice -n 10 "$HOME/task074_frameworks/venv_pgx/bin/python" t092_bridge_calibration.py
