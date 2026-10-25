#!/usr/bin/env bash
# Task 058 QAT residual driver: for a QATSharedNNUE model dir, export its BNNQ, then
# report the quantization residual on tools/parity_positions.sfen:
#   - continuous float eval (what the trainer optimized, fake_quant OFF)   [cp]
#   - deployed int eval, two independent implementations that must agree:
#       * Python int-faithful (trainer/qat_eval.py)        -> reference
#       * binbo Rust Mode::Quant nnue-eval on the BNNQ     -> deployment
#   residual = int - float (the "536cp -> quant-error" headline; small = loads undistorted).
#
#   scripts/qat_residual.sh <model_dir> [weights.pt] [positions.sfen]
#
# Requires env.sh (BINBO, VENV_PY). Writes the BNNQ next to the .pt (<weights>.bnnq).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
PY="${VENV_PY:-python3}"
MODEL="${1:?usage: qat_residual.sh <model_dir> [weights.pt] [positions.sfen]}"
WEIGHTS="${2:-nnue_best.pt}"
POS="${3:-tools/parity_positions.sfen}"
BNNQ="$MODEL/${WEIGHTS%.pt}.bnnq"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# 1) export BNNQ from the trained QAT checkpoint.
"$PY" -m trainer.export_nnue "$MODEL" --weights "$WEIGHTS" --out "$(basename "$BNNQ")" >/dev/null

# 2) Python reference (continuous float + deployed int) + 3) binbo Rust deployment.
"$PY" -m trainer.qat_eval "$MODEL" "$POS" --weights "$WEIGHTS" --out-int "$WORK/pyint.txt"
"${BINBO}" nnue-eval "$BNNQ" "$POS" "$WORK/rustint.txt" 2>/dev/null

"$PY" - "$WORK/rustint.txt" "$WORK/pyint.txt" <<'PY'
import sys, numpy as np
ri = np.array([int(float(x)) for x in open(sys.argv[1])])
pi = np.array([int(x) for x in open(sys.argv[2])])
ok = bool((ri == pi).all())
print(f"  binbo Rust BNNQ deploy vs Python int : exact={int((ri==pi).sum())}/{len(ri)} "
      f"max|diff|={int(np.abs(ri-pi).max())} -> {'GREEN(deploy==reference)' if ok else 'RED'}")
PY
