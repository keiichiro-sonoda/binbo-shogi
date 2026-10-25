#!/usr/bin/env bash
# Task 089: run the remaining scale-study stages SEQUENTIALLY in one detached job so it
# completes unattended (no per-stage re-trigger needed — the fix for the missed-completion
# stall). Order: 7.6M eval first (its ckpt exists -> external result soon), then 14.4M
# distill, then 14.4M eval. Each sub-script prints its own DONE marker.
set -uo pipefail   # NOT -e: one failing stage must not abort the rest (each is independent-ish)
cd "$HOME/task074_frameworks/pgx/examples/alphazero"

echo "########## STAGE 1/3: 7.6M eval ##########"
bash t089_eval.sh t089_7p6m/ws2.ckpt 7p6m || echo "STAGE1_FAILED rc=$?"

echo "########## STAGE 2/3: 14.4M distill ##########"
bash t089_run_14p4m.sh || echo "STAGE2_FAILED rc=$?"

echo "########## STAGE 3/3: 14.4M eval ##########"
if [ -f t089_14p4m/ws2.ckpt ]; then
  bash t089_eval.sh t089_14p4m/ws2.ckpt 14p4m || echo "STAGE3_FAILED rc=$?"
else
  echo "STAGE3_SKIPPED: t089_14p4m/ws2.ckpt missing (distill failed)"
fi
echo "PIPELINE_DONE"
