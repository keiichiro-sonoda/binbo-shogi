#!/bin/bash
# Task 177: WS-T2 on all 32.4M positions with the fast trainer (t174). First a short fp32-vs-bf16 equivalence check
# (2,000 steps each from ws2, same seed; eval on the 1,184 probe roots + 50k held-out positions). If bf16's probe top-1 and
# held-out top-1 are within 0.5 pt of fp32 and its held-out Brier within 0.002, the full run uses bf16, otherwise fp32.
# Full run: 1 epoch from the original WS-T (ws2), batch 256, lr 1e-5 (t173 defaults), then the full eval.
set -eu
cd $HOME/GitHub/binbo-shogi; REPO=$PWD; source scripts/env.sh
W=$DATA_ROOT/work/task177_wst2_full; PGX=$HOME/task074_frameworks/venv_pgx/bin/python; T=experiments/alphazero/t174_fast_wst.py
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $W
for c in fp32 bf16; do
  [ -f $W/check_$c/latest.json ] || $PGX $T --device gpu train --out $W/check_$c --compute $c --batch 256 --steps 2000
  L=$($PGX -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' $W/check_$c/latest.json)
  [ -f $W/check_eval_$c/eval.json ] || $PGX $T --device gpu eval --checkpoint $W/check_$c/$L --out $W/check_eval_$c --val-limit 50000 --allow-subset
done
C=$($PGX - $W <<'PY'
import json,sys
w=sys.argv[1]; r={}
for c in ("fp32","bf16"):
    e=json.load(open(f"{w}/check_eval_{c}/eval.json"))["results"]["wst2"]
    r[c]=(e["probe"]["top1"], e["heldout"]["top1"], e["heldout"]["value_brier"])
ok=abs(r["fp32"][0]-r["bf16"][0])<=0.005 and abs(r["fp32"][1]-r["bf16"][1])<=0.005 and abs(r["fp32"][2]-r["bf16"][2])<=0.002
print("bf16" if ok else "fp32"); print(json.dumps(r), file=sys.stderr)
PY
)
echo "T177_COMPUTE $C"
$PGX $T --device gpu train --out $W/full --compute $C --batch 256 --epochs 1
echo T177_TRAIN_DONE
L=$($PGX -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' $W/full/latest.json)
$PGX $T --device gpu eval --checkpoint $W/full/$L --out $W/eval_full
echo T177_EVAL_DONE
