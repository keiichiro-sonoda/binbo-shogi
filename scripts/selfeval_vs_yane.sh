#!/usr/bin/env bash
# Task 060 search-engine comparison driver: take a trained (tied QAT) model dir and run the
# SAME self-made eval on both ① binbo's internal Rust search and ② YaneuraOu's search,
# one variable only (the search engine; the eval is bit-exact identical on both sides).
#
#   scripts/selfeval_vs_yane.sh <model_dir> [weights=nnue_best.pt] [movetime_ms=100] \
#                               [max_games=200] [seed] [opening_plies] [eval_dir]
#
# Pipeline:
#   1) export  : model_dir/<weights>  --(trainer.export_nnue, auto tie)--> <weights>.bnnq
#   2) write   : BNNQ  --(trainer.binbo_to_yane)--> EvalDir/nn.bin  (YaneuraOu eval)
#   3) GATE    : scripts/binbo_to_yane.sh  =>  engine `e` == binbo Mode::Quant == int-faithful = 0cp.
#                The match runs ONLY if this is green: otherwise an eval mismatch would
#                contaminate the search-engine comparison (can't attribute the result).
#   4) match   : binbo `nnue-vs-yane` (movetime, trinomial SPRT early-stop) on the SAME EvalDir.
#
# The EvalDir + BNNQ + per-game TSV are persistent (under $DATA_ROOT), git-untracked, GPL eval
# kept out of the repo. Records the nn.bin sha256 so the report pins the exact eval played.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
PY="${VENV_PY:-python3}"

MODEL="${1:?usage: selfeval_vs_yane.sh <model_dir> [weights] [movetime_ms] [max_games] [seed] [opening_plies] [eval_dir]}"
# Accept either a full model dir or a bare tag resolved under $MODELS_DIR.
if [ ! -d "$MODEL" ] && [ -d "$MODELS_DIR/$MODEL" ]; then
  MODEL="$MODELS_DIR/$MODEL"
fi
WEIGHTS="${2:-nnue_best.pt}"
MOVETIME_MS="${3:-100}"
MAX_GAMES="${4:-200}"
SEED="${5:-20260622}"
OPENING_PLIES="${6:-6}"
MODEL_TAG="$(basename "$MODEL")"
EVAL_DIR="${7:-$DATA_ROOT/eval_dirs/$MODEL_TAG}"
BNNQ="$MODEL/${WEIGHTS%.pt}.bnnq"
TSV="$LOGS_DIR/task060_${MODEL_TAG}_mt${MOVETIME_MS}.tsv"

[ -x "$BINBO" ] || { echo "missing binbo binary: $BINBO (cargo build --release)"; exit 1; }
[ -x "$YANE" ]  || { echo "missing YaneuraOu binary: $YANE"; exit 1; }
[ -f "$MODEL/$WEIGHTS" ] || { echo "missing weights: $MODEL/$WEIGHTS"; exit 1; }
mkdir -p "$EVAL_DIR" "$LOGS_DIR"

echo "== 1) export BNNQ (auto tie via model_config.json) =="
"$PY" -m trainer.export_nnue "$MODEL" --weights "$WEIGHTS" --out "$(basename "$BNNQ")"

echo "== 2) write YaneuraOu nn.bin into EvalDir =="
"$PY" -m trainer.binbo_to_yane --from-bnnq "$BNNQ" --template "$YANE_EVAL/nn.bin" --out "$EVAL_DIR/nn.bin"
EVAL_SHA="$(sha256sum "$EVAL_DIR/nn.bin" | cut -d' ' -f1)"
echo "eval_dir=$EVAL_DIR"
echo "nn.bin sha256=$EVAL_SHA"

echo "== 3) PRE-MATCH GATE: eval identity (engine e == binbo Mode::Quant == int-faithful = 0cp) =="
# binbo_to_yane.sh (059, untouched) prints GREEN/RED but exits 0 either way, so enforce the
# verdict HERE: the match must NOT run on a mismatched eval (it would confound the search-engine
# comparison). Require both checks GREEN and zero RED.
GATE_OUT="$(scripts/binbo_to_yane.sh "$BNNQ")"
echo "$GATE_OUT"
GREENS="$(printf '%s\n' "$GATE_OUT" | grep -c 'GREEN' || true)"
REDS="$(printf '%s\n' "$GATE_OUT" | grep -c 'RED' || true)"
if [ "$REDS" -ne 0 ] || [ "$GREENS" -lt 2 ]; then
  echo "GATE FAILED (greens=$GREENS reds=$REDS): eval not bit-exact on both sides — aborting before match." >&2
  exit 1
fi
echo "GATE PASSED: eval is bit-exact identical on binbo and YaneuraOu (0cp)."

echo "== 4) search-engine match: SAME eval, binbo search vs YaneuraOu search =="
echo "(movetime_ms=$MOVETIME_MS max_games=$MAX_GAMES seed=$SEED opening_plies=$OPENING_PLIES)"
echo "tsv=$TSV"
"$BINBO" nnue-vs-yane "$BNNQ" "$YANE" "$EVAL_DIR" "$MOVETIME_MS" "$SEED" "$MAX_GAMES" "$OPENING_PLIES" | tee "$TSV"

echo "== done. eval sha256=$EVAL_SHA tsv=$TSV =="
