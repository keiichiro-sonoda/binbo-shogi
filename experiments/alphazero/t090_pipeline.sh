#!/usr/bin/env bash
# Task 090 Phase 1 pipeline (sequential, unattended — 089 lesson): distill 7.6M@1e-3@20ep
# then eval. Killing the distill on plateau/diverge advances to eval (|| + [-f ckpt] guard).
# Usage: t090_pipeline.sh <run-script> <out-dir> <tag>
set -uo pipefail
RUN="${1:-t090_run_7p6m_lr1e3_ep20.sh}"; OUT="${2:-t090_7p6m_lr1e3ep20}"; TAG="${3:-7p6m_lr1e3ep20}"
cd "$HOME/task074_frameworks/pgx/examples/alphazero"

echo "########## STAGE 1/2: distill ($RUN -> $OUT) ##########"
bash "$RUN" || echo "STAGE1_ENDED rc=$?"

echo "########## STAGE 2/2: eval ($TAG) ##########"
if [ -f "$OUT/ws2.ckpt" ]; then
  bash t089_eval.sh "$OUT/ws2.ckpt" "$TAG" || echo "STAGE2_FAILED rc=$?"
else
  echo "STAGE2_SKIPPED: $OUT/ws2.ckpt missing"
fi
echo "PIPELINE_DONE"
