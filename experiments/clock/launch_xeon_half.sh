#!/usr/bin/env bash
# Start the Xeon half of Task 180 once Task 181's Xeon judging is done (see reports/task180_sente_clock.md, amendment 2).
set -uo pipefail
LOG=$HOME/binbo-shogi-data/logs/task181_20261024-020818.log
R=${BINBO_WORKER_ROOT:?set BINBO_WORKER_ROOT to the home directory on the worker}/binbo-work/data
WORKER=${BINBO_WORKER_HOST:-xeon-worker}   # ssh の接続先の別名 (~/.ssh/config で定義する)
until grep -q "T181_DELTA\|DETACHED_DONE" "$LOG"; do sleep 30; done
lo=$(grep -m1 T181_DELTA "$LOG" | sed -E 's/.*ci=\[([-0-9.]+),.*/\1/')
echo "T181 ci_low=${lo:-none}"
if [ -n "$lo" ] && python3 -c "import sys; sys.exit(0 if float('$lo') > 0 else 1)"; then
  # the runner would play the confirmation's candidate side here, where the clock half leaves no memory for it
  pkill -f t181_mixfinish.py && echo "KILLED_T181_RUNNER: confirm M later on the Xeon"
fi
until ssh "$WORKER" "mkdir $R/jobs/.busy" 2>/dev/null; do sleep 60; done   # the frame lock: no judging during this run
echo "XEON_LOCK_TAKEN $(date +%T)"
ssh "$WORKER" "cd $R/work/task180_clock && setsid -f bash -c 'export DATA_ROOT=$R PYTHONPATH=$BINBO_WORKER_ROOT/binbo-work/pylib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; python3 -B run_experiment.py --config $R/work/task180_clock/xeon_scaled_v1.json --out $R/work/task180_clock/xeon_prereg_v3 --games 2667 --parallel 48 --cpu-budget 48 --threads 1 --hash-mb 256 --black-base-ms 18000 --white-base-ms 60000 --black-inc-ms 200 --white-inc-ms 200 --network-margin-ms 0 --charge-quantum-ms 0 --opening-plies 4 --max-plies 512 --cap-ms 3000 --k 17 --seed 2026181333 > xeon_prereg_v3.log 2>&1; echo \$? > xeon_prereg_v3.rc; rmdir $R/jobs/.busy' > /dev/null 2>&1 < /dev/null"
echo "XEON_HALF_STARTED $(date +%T)"
