#!/usr/bin/env bash
# Task 059 writer driver: serialize a binbo BNNQ to a YaneuraOu nn.bin and prove the
# self-made eval loads bit-exact into the real engine.
#
#   scripts/binbo_to_yane.sh <bnnq> [template_nn] [positions.sfen]
#
# Pipeline: BNNQ --(trainer/binbo_to_yane)--> nn.bin --> load in YaneuraOu, then on
# tools/parity_positions.sfen compare:
#   - A  : engine `e` (written) == int-faithful forward (yane_forward_int, written)   [writer valid]
#   - fid: binbo `nnue-eval` Mode::Quant (source BNNQ) == engine `e` (written)         [round-trip 0cp]
# For a tanuki-derived BNNQ the binbo forward equals the ORIGINAL tanuki `e` (Task 058),
# so `fid` GREEN also means engine(written) == original tanuki (engine-grounded bit-exact).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
PY="${VENV_PY:-python3}"
BNNQ="${1:?usage: binbo_to_yane.sh <bnnq> [template_nn] [positions.sfen]}"
TEMPLATE="${2:-$YANE_EVAL/nn.bin}"
POS="${3:-tools/parity_positions.sfen}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/eval"

# 1) write nn.bin into an EvalDir.
"$PY" -m trainer.binbo_to_yane --from-bnnq "$BNNQ" --template "$TEMPLATE" --out "$WORK/eval/nn.bin" >/dev/null

# 2) engine `e` on the written eval (read-write FIFO keeps stdin open past each eval).
FIFO="$WORK/par.fifo"; mkfifo "$FIFO"
{ printf 'usi\nsetoption name Threads value 1\nsetoption name EvalDir value %s\nsetoption name USI_OwnBook value false\nisready\n' "$WORK/eval"
  while IFS= read -r s; do [ -n "$s" ] && printf 'position sfen %s\ne\n' "$s"; done < "$POS"
  printf 'quit\n'; } > "$WORK/cmds.txt"
timeout 60 "$YANE" < "$FIFO" > "$WORK/par.txt" 2>&1 &
exec 3<>"$FIFO"; cat "$WORK/cmds.txt" >&3; wait; exec 3>&-
grep -E 'eval = ' "$WORK/par.txt" | sed 's/eval = //' > "$WORK/e_written.txt"

# 3) binbo Mode::Quant forward on the SOURCE BNNQ (the eval being deployed).
"${BINBO}" nnue-eval "$BNNQ" "$POS" "$WORK/binbo_src.txt" 2>/dev/null

# 4) int-faithful forward on the WRITTEN nn.bin + compare.
"$PY" - "$POS" "$WORK/e_written.txt" "$WORK/binbo_src.txt" "$WORK/eval/nn.bin" <<'PY'
import sys, numpy as np, cshogi
from trainer import yane_nnue as yn, yane_to_binbo as yb
pos, e_written_f, binbo_src_f, nn_written = sys.argv[1:5]
sfens=[l.strip() for l in open(pos) if l.strip()]
ew=np.array([int(x) for x in open(e_written_f)], dtype=np.int64)
bs=np.array([int(float(x)) for x in open(binbo_src_f)], dtype=np.int64)
p=yn.parse(nn_written); yb._attach_int_views(p)
pi=np.array([yb.yane_forward_int(p, cshogi.Board(s)) for s in sfens], dtype=np.int64)
n=len(sfens)
if n == 0 or len(ew) != n or len(bs) != n:  # Task 149: fail closed on missing rows.
    raise SystemExit(f"RED count mismatch: positions={n} engine={len(ew)} binbo={len(bs)}")
exA=int((ew==pi).sum()); exF=int((ew==bs).sum())
print(f"N={n}")
print(f"  A  engine e(written) vs int-faithful(written) : exact={exA}/{n} max|d|={int(np.abs(ew-pi).max())} -> {'GREEN(0cp)' if exA==n else 'RED'}")
print(f"  fid binbo Mode::Quant(src) vs engine e(written): exact={exF}/{n} max|d|={int(np.abs(ew-bs).max())} -> {'GREEN(0cp, bit-exact in engine)' if exF==n else 'RED'}")
sys.exit(0 if exA == n and exF == n else 1)  # Task 149
PY
