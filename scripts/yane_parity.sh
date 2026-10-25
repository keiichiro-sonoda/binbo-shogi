#!/usr/bin/env bash
# Task 057 forward-parity driver: compare binbo's view of a YaneuraOu nn.bin against the
# real engine. Establishes the reusable procedure (it works the same for the 256 本命 eval).
#
#   scripts/yane_parity.sh [positions.sfen]
#
# Produces, for each position:
#   - YaneuraOu `e` (raw static NNUE eval)  = ground truth
#   - Python int-faithful forward (remap + exact int math)  -> must equal `e` exactly
#   - binbo Rust nnue-eval on the converted BNN3            -> correlation only (plain ReLU)
#
# Requires env.sh (YANE, YANE_EVAL, BINBO, VENV_PY). Reads the eval from $YANE_EVAL/nn.bin.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
PY="${VENV_PY:-python3}"
POS="${1:-tools/parity_positions.sfen}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
FIFO="$WORK/par.fifo"; mkfifo "$FIFO"

# 1) YaneuraOu `e` for each position (read-write FIFO keeps stdin open past each eval).
{ printf 'usi\nsetoption name EvalDir value %s\nsetoption name USI_OwnBook value false\nisready\n' "$YANE_EVAL"
  while IFS= read -r s; do [ -n "$s" ] && printf 'position sfen %s\ne\n' "$s"; done < "$POS"
  printf 'quit\n'; } > "$WORK/cmds.txt"
timeout 60 "$YANE" < "$FIFO" > "$WORK/par.txt" 2>&1 &
exec 3<>"$FIFO"; cat "$WORK/cmds.txt" >&3; wait; exec 3>&-
grep -E 'eval = ' "$WORK/par.txt" | sed 's/eval = //' > "$WORK/yane.txt"

# 2) convert tanuki -> binbo BNN3 (float, plain ReLU) and BNNQ (int, quant-aware;
#    Task 058), run both binbo forwards.
"$PY" -m trainer.yane_to_binbo "$YANE_EVAL/nn.bin" --convert "$WORK/binbo.bin" >/dev/null
"${BINBO}" nnue-eval "$WORK/binbo.bin" "$POS" "$WORK/binbo.txt" 2>/dev/null
"$PY" -m trainer.yane_to_binbo "$YANE_EVAL/nn.bin" --convert-quant "$WORK/binbo.bnnq" >/dev/null
"${BINBO}" nnue-eval "$WORK/binbo.bnnq" "$POS" "$WORK/binbo_quant.txt" 2>/dev/null

# 3) int-faithful forward + compare (float = correlation only; quant = bit-exact).
"$PY" - "$POS" "$WORK/yane.txt" "$WORK/binbo.txt" "$YANE_EVAL/nn.bin" "$WORK/binbo_quant.txt" <<'PY'
import sys, numpy as np, cshogi
from trainer import yane_nnue as yn, yane_to_binbo as yb
pos, yane_f, binbo_f, nn, binbo_q_f = sys.argv[1:6]
sfens=[l.strip() for l in open(pos) if l.strip()]
yane=np.array([int(x) for x in open(yane_f)],dtype=float)
binbo=np.array([float(x) for x in open(binbo_f)],dtype=float)
binbo_q=np.array([int(x) for x in open(binbo_q_f)],dtype=float)
parsed=yb.yn.parse(nn); yb._attach_int_views(parsed)
pyint=np.array([yb.yane_forward_int(parsed,cshogi.Board(s)) for s in sfens],dtype=float)
ex=int((pyint==yane).sum())
print(f"N={len(sfens)}")
print(f"  int-faithful forward vs YaneuraOu e : exact={ex}/{len(sfens)}  max|diff|={int(np.abs(pyint-yane).max())}  -> {'GREEN(exact)' if ex==len(sfens) else 'RED'}")
r=np.corrcoef(binbo,yane)[0,1]
A=np.vstack([binbo,np.ones_like(binbo)]).T; sl,ic=np.linalg.lstsq(A,yane,rcond=None)[0]
resid=yane-(sl*binbo+ic)
print(f"  binbo float forward (BNN3) vs yane  : Pearson r={r:.3f}  resid std={resid.std():.0f}cp (loose: plain ReLU != clipped int) [BEFORE]")
exq=int((binbo_q==yane).sum())
print(f"  binbo quant forward (BNNQ) vs yane  : exact={exq}/{len(sfens)}  max|diff|={int(np.abs(binbo_q-yane).max())}  resid std={(binbo_q-yane).std():.2f}cp  -> {'GREEN(bit-exact, 536->0)' if exq==len(sfens) else 'RED'} [AFTER]")
PY