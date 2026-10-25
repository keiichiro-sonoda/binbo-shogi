#!/usr/bin/env bash
# Task 102 driver (2026-09-12, pre-registered): one detached job for the night.
#   102b  scale control  : t102_scale_control.py --stage train (GPU, ~12 min) then
#                          --stage match in the background (CPU, ~35 min), after which
#                          the bounded +32M teacher generation (Task 101 stage 1) resumes
#                          on 12 workers until $T102_GEN_STOP (before 102a's students).
#   102a  scale curve 4M : waits for the detached az_rescore of shared_8m (raw labels,
#                          log = $T102_RAW_LOG), then t100_search_student.py on the first
#                          4,000,000 rows of shared_8m (run2m_c parts reused from 100b).
# Launch: T102_RAW_LOG=<log> scripts/detached.sh task102 bash experiments/alphazero/t102_pipeline.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
source scripts/env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
PGX="$HOME/task074_frameworks/venv_pgx/bin/python"
RAW="$HOME/binbo-shogi-data/work/task102a/rescore_raw"
RAWLOG="${T102_RAW_LOG:?set T102_RAW_LOG to the detached az_rescore log}"
GEN_STOP="${T102_GEN_STOP:-04:30}"
DLOG="$LOGS_DIR/task102b_driver.log"

echo "=== T102 START $(date) gen_stop=$GEN_STOP raw_log=$RAWLOG ==="
"$PGX" experiments/alphazero/t102_scale_control.py --stage train || { echo "=== T102B_TRAIN_FAILED ==="; exit 1; }

(
  "$PGX" experiments/alphazero/t102_scale_control.py --stage match
  echo "=== T102B_MATCH_DONE rc=$? $(date) ==="
  secs=$(( $(date -d "$GEN_STOP" +%s) - $(date +%s) ))
  if [ "$secs" -gt 600 ]; then
    MAX_SECONDS="$secs" WORKERS=12 scripts/detached.sh gen64m_b bash "$HOME/binbo-shogi-data/teacher/run32m_g.gen.sh"
  else
    echo "=== GEN SKIPPED (only ${secs}s left before $GEN_STOP) ==="
  fi
) > "$DLOG" 2>&1 &
BPID=$!

until grep -q 'az_rescore=ok' "$RAWLOG"; do
  if grep -q -E 'DETACHED_DONE rc=[1-9]|Traceback' "$RAWLOG"; then echo "=== RAW RESCORE FAILED ==="; exit 1; fi
  sleep 30
done
echo "=== T102A RAW READY $(date) ==="
"$PGX" experiments/alphazero/t100_search_student.py --run-tag task102a \
  --cache "$HOME/binbo-shogi-data/caches/shared_8m" --raw-dir "$RAW" \
  --limit-train 4000000 --no-clock-rule
rc=$?
wait "$BPID"
echo "=== T102_PIPELINE_DONE rc_a=$rc $(date) ==="
exit "$rc"
