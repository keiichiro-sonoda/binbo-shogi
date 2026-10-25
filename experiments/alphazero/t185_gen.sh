#!/usr/bin/env bash
# Task 185: round-1 self-play of the student lineage (task166_S_R2 vs itself, depth 9, to mate), generated and
# quiescence-replaced on the Xeon, then brought home, assembled and compacted here (same recipe as Task 154/159).
set -euo pipefail
R=${BINBO_WORKER_ROOT:?set BINBO_WORKER_ROOT to the home directory on the worker}/binbo-work
WORKER=${BINBO_WORKER_HOST:-xeon-worker}   # ssh の接続先の別名 (~/.ssh/config で定義する)
W=$R/data/work/task185
EVAL=$R/data/eval_dirs/task166_S_R2
ENV="export DATA_ROOT=$R/data PYTHONPATH=$R/repo:$R/pylib YANE=$R/data/engines/yaneuraou/source/YaneuraOu-by-gcc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"
GEN="python3 -B $R/repo/experiments/alphazero/t153_selfplay_gen.py"
ssh "$WORKER" "mkdir -p $W && cd $R/repo && setsid -f bash -c '$ENV; $GEN gen --games 150000 --depth 9 --procs 48 --out $W --eval-limit 32000 --max-ply 320 --nodes-cap 10000000 --eval-dir $EVAL --seed 20271850 > $W/gen.log 2>&1 && $GEN qs --work $W --procs 48 --eval-dir $EVAL > $W/qs.log 2>&1; echo \$? > $W/DONE' > /dev/null 2>&1 < /dev/null"
echo "T185_STARTED $(date +%T)"
until ssh "$WORKER" "test -f $W/DONE" 2>/dev/null; do sleep 120; done
rc=$(ssh "$WORKER" "cat $W/DONE"); echo "T185_XEON_DONE rc=$rc $(date +%T)"; [ "$rc" = 0 ] || exit 1
L=$HOME/binbo-shogi-data/work/task185
mkdir -p "$L" && rsync -a "$WORKER:$W/" "$L/"
cd $HOME/GitHub/binbo-shogi && source scripts/env.sh
PYTHONPATH=$PWD "$VENV_PY" -B experiments/alphazero/t153_selfplay_gen.py assemble --work "$L" --out "$L/cache"
PYTHONPATH=$PWD "$VENV_PY" -B -m trainer.compact_cache "$L/cache" "$L/cache_compact"
echo "T185_DATA_READY $(python3 -c "import json;m=json.load(open('$L/cache/meta.json'));print(m['n_train'], m.get('counts'))")"
