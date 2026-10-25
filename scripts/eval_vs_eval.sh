#!/usr/bin/env bash
# Task 061 generation-comparison driver: confirm that a newly trained candidate eval is
# stronger than the current baseline by playing them against each other under a FIXED search
# (YaneuraOu both sides, different EvalDir). One variable: the eval.
#
#   scripts/eval_vs_eval.sh <cand_model_dir> <baseline_eval_dir> \
#                           [weights=nnue_best.pt] [movetime_ms=100] [max_games=400] [seed] [opening_plies]
#
# Pipeline (candidate side only — the baseline EvalDir is assumed already gated when adopted):
#   1) export : cand_model_dir/<weights>  --(trainer.export_nnue, auto tie)--> <weights>.bnnq
#   2) write  : BNNQ  --(trainer.binbo_to_yane)--> eval_dirs/<cand_tag>/nn.bin
#   3) GATE   : scripts/binbo_to_yane.sh  =>  engine `e` == binbo Mode::Quant == int-faithful = 0cp.
#               (RED => abort: a mismatched eval would confound the comparison.)
#   4) match  : binbo `yane-vs-yane` (trinomial SPRT, H1 = candidate stronger) cand vs baseline.
#
# Eval dirs + BNNQ + per-game TSV are persistent (under $DATA_ROOT), git-untracked, GPL eval
# kept out of the repo. Records both nn.bin sha256s so the report pins the exact evals played.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
PY="${VENV_PY:-python3}"

CAND="${1:?usage: eval_vs_eval.sh <cand_model_dir> <baseline_eval_dir> [weights] [movetime_ms] [max_games] [seed] [opening_plies]}"
BASE_EVAL_DIR="${2:?missing <baseline_eval_dir>}"
# Accept either a full model dir or a bare tag resolved under $MODELS_DIR.
if [ ! -d "$CAND" ] && [ -d "$MODELS_DIR/$CAND" ]; then
  CAND="$MODELS_DIR/$CAND"
fi
WEIGHTS="${3:-nnue_best.pt}"
MOVETIME_MS="${4:-100}"
MAX_GAMES="${5:-400}"
SEED="${6:-20260622}"
OPENING_PLIES="${7:-6}"
CAND_TAG="$(basename "$CAND")"
CAND_EVAL_DIR="$DATA_ROOT/eval_dirs/$CAND_TAG"
BNNQ="$CAND/${WEIGHTS%.pt}.bnnq"
TSV="$LOGS_DIR/task061_${CAND_TAG}_vs_$(basename "$BASE_EVAL_DIR")_mt${MOVETIME_MS}.tsv"

[ -x "$BINBO" ] || { echo "missing binbo binary: $BINBO (cargo build --release)"; exit 1; }
[ -x "$YANE" ]  || { echo "missing YaneuraOu binary: $YANE"; exit 1; }
[ -f "$CAND/$WEIGHTS" ] || { echo "missing weights: $CAND/$WEIGHTS"; exit 1; }
[ -f "$BASE_EVAL_DIR/nn.bin" ] || { echo "missing baseline eval: $BASE_EVAL_DIR/nn.bin"; exit 1; }
mkdir -p "$CAND_EVAL_DIR" "$LOGS_DIR"

echo "== 1) export candidate BNNQ (auto tie via model_config.json) =="
"$PY" -m trainer.export_nnue "$CAND" --weights "$WEIGHTS" --out "$(basename "$BNNQ")"

echo "== 2) write candidate YaneuraOu nn.bin into EvalDir =="
"$PY" -m trainer.binbo_to_yane --from-bnnq "$BNNQ" --template "$YANE_EVAL/nn.bin" --out "$CAND_EVAL_DIR/nn.bin"
CAND_SHA="$(sha256sum "$CAND_EVAL_DIR/nn.bin" | cut -d' ' -f1)"
BASE_SHA="$(sha256sum "$BASE_EVAL_DIR/nn.bin" | cut -d' ' -f1)"
echo "cand_eval_dir=$CAND_EVAL_DIR  sha256=$CAND_SHA"
echo "base_eval_dir=$BASE_EVAL_DIR  sha256=$BASE_SHA"
if [ "$CAND_SHA" = "$BASE_SHA" ]; then
  echo "candidate and baseline nn.bin are identical (sha match) — nothing to compare." >&2
  exit 1
fi

echo "== 3) PRE-MATCH GATE: candidate eval identity (engine e == binbo Mode::Quant == int-faithful = 0cp) =="
GATE_OUT="$(scripts/binbo_to_yane.sh "$BNNQ")"
echo "$GATE_OUT"
GREENS="$(printf '%s\n' "$GATE_OUT" | grep -c 'GREEN' || true)"
REDS="$(printf '%s\n' "$GATE_OUT" | grep -c 'RED' || true)"
if [ "$REDS" -ne 0 ] || [ "$GREENS" -lt 2 ]; then
  echo "GATE FAILED (greens=$GREENS reds=$REDS): candidate eval not bit-exact — aborting before match." >&2
  exit 1
fi
echo "GATE PASSED: candidate eval is bit-exact (binbo == YaneuraOu, 0cp)."

echo "== 4) generation match: candidate vs baseline, FIXED YaneuraOu search =="
echo "(movetime_ms=$MOVETIME_MS max_games=$MAX_GAMES seed=$SEED opening_plies=$OPENING_PLIES)"
echo "tsv=$TSV"
"$BINBO" yane-vs-yane "$YANE" "$CAND_EVAL_DIR" "$BASE_EVAL_DIR" "$MOVETIME_MS" "$SEED" "$MAX_GAMES" "$OPENING_PLIES" | tee "$TSV"

echo "== done. cand sha256=$CAND_SHA base sha256=$BASE_SHA tsv=$TSV =="
