#!/usr/bin/env bash
# Task 078 M3: overnight loop wrapper — external watchdog (hang detection) + auto-resume.
# Launch via: scripts/detached.sh m3_loop bash ~/task074_frameworks/pgx/examples/alphazero/m3_run.sh [max_iters]
#
# Anomaly detection layer 2 (layer 1 = WARN_SLOW inside m3_loop.py):
#   - progress file m3_progress.log gets one flushed line per iteration (~290s expected)
#   - no write for 30 min -> WARN_HANG marker; 45 min -> safe kill (ckpt-resumable)
#   - m3_loop.py resume=auto restarts from the latest m3_ckpt/*.ckpt (max 5 attempts)
set -u
cd "$(dirname "$0")"
PY="$HOME/task074_frameworks/venv_pgx/bin/python"
MAX_ITERS="${1:-150}"
# lr=1e-4 chosen by probe ladder (2026-07-02): fresh-adam lr=1e-3 wipes the distilled
# value in ONE iteration (val_corr_m2 0.90->0.0075); 1e-4 retains 0.57/0.50 after
# iters 1/2 while policy_loss still moves. Documented in reports/task078_m3_loop.md.
LR="${2:-1e-4}"
CORR_INT="${3:-1}"  # Task 079: per-iter val_corr_m2 logging by default
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")        # Task 080: extra m3_loop.py key=value args (init_ckpt=, ckpt_dir=, m2_val_dir=, ...)
PROGRESS="${M3_PROGRESS:-m3_progress.log}"
HANG_WARN_S=1800
HANG_KILL_S=2700

run_once() {
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" m3_loop.py \
    max_num_iters="$MAX_ITERS" learning_rate="$LR" corr_interval="$CORR_INT" resume=auto \
    ${EXTRA[@]+"${EXTRA[@]}"} >> "$PROGRESS" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    local age=$(( $(date +%s) - $(stat -c %Y "$PROGRESS") ))
    if [ "$age" -ge "$HANG_KILL_S" ]; then
      echo "WARN_HANG_KILL age=${age}s $(date)" >> "$PROGRESS"
      kill "$pid" 2>/dev/null; sleep 15; kill -9 "$pid" 2>/dev/null
      return 1
    elif [ "$age" -ge "$HANG_WARN_S" ]; then
      echo "WARN_HANG age=${age}s $(date)" >> "$PROGRESS"
    fi
  done
  wait "$pid"
}

touch "$PROGRESS"
for attempt in 1 2 3 4 5; do
  echo "=== M3_RUN attempt=$attempt max_iters=$MAX_ITERS $(date) ===" | tee -a "$PROGRESS"
  run_once
  if grep -q M3_LOOP_DONE "$PROGRESS"; then
    echo "=== M3_RUN_COMPLETE $(date) ===" | tee -a "$PROGRESS"
    exit 0
  fi
  echo "=== M3_RESTART after attempt=$attempt $(date) ===" | tee -a "$PROGRESS"
done
echo "=== M3_RUN_GAVE_UP $(date) ===" | tee -a "$PROGRESS"
exit 1
