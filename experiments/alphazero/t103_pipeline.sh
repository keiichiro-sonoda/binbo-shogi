#!/usr/bin/env bash
# Task 103 driver (2026-09-12 daytime, pre-registered): one detached job.
#   103c  absolute frames  : t103_absolute.py in the background (CPU only) from the start:
#                            both 4M students (102a) vs champion and vs gikou@6, 4 x 800 games.
#   103a  scale curve 8M   : t100_search_student.py on ALL 7,950,000 train rows of shared_8m
#                            (405 shard parts reused from 102a; ~3.95M new rows), raw student
#                            trained AFTER the rescore (no GPU contention), then 800 games.
#   103b  sims ladder S=8  : t100_search_student.py --sims 8 on the Task 100 corpus (1M),
#                            raw student = task100_student_raw (reused), 800 games.
# Launch: scripts/detached.sh task103 bash experiments/alphazero/t103_pipeline.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
source scripts/env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
PGX="$HOME/task074_frameworks/venv_pgx/bin/python"
D="$HOME/binbo-shogi-data"

echo "=== T103 START $(date) ==="
( "$PGX" experiments/alphazero/t103_absolute.py; echo "=== T103C_DONE rc=$? $(date) ===" ) \
  > "$LOGS_DIR/task103c_driver.log" 2>&1 &
CPID=$!

"$PGX" experiments/alphazero/t100_search_student.py --run-tag task103a \
  --cache "$D/caches/shared_8m" --raw-dir "$D/work/task102a/rescore_raw" \
  --limit-train 7950000 --no-clock-rule --raw-student-after
rc_a=$?
echo "=== T103A_DONE rc=$rc_a $(date) ==="

"$PGX" experiments/alphazero/t100_search_student.py --run-tag task103b \
  --sims 8 --limit-train 1000000 --raw-student "$D/models/task100_student_raw" \
  --label-moved-min-cp 30 --no-clock-rule --skip-cargo
rc_b=$?
echo "=== T103B_DONE rc=$rc_b $(date) ==="

wait "$CPID"
echo "=== T103_PIPELINE_DONE rc_a=$rc_a rc_b=$rc_b $(date) ==="
[ "$rc_a" -eq 0 ] && [ "$rc_b" -eq 0 ]
