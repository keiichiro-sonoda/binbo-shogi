#!/usr/bin/env bash
# Task 149: explicit engine selection, template-free wide/default 0cp gate.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$VENV_PY" -m trainer.nnue_gate "$@"
