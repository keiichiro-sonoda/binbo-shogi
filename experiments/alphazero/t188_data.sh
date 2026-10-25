#!/usr/bin/env bash
# Task 188 data pipeline, in the order the disk allows (free ~45GB; each 64M cache ~30GB, the .001 volume 24GB).
set -euo pipefail
cd "$HOME/GitHub/binbo-shogi" && source scripts/env.sh
L=$HOME/binbo-shogi-data/logs
PY="env PYTHONPATH=$PWD $VENV_PY -B experiments/alphazero/t188_student2x_data.py"
until grep -q DETACHED_DONE $L/archive_selfplay_20261024-233338.log; do sleep 60; done
grep -q "^DONE" $L/archive_selfplay_20261024-233338.log || { echo "selfplay archive failed"; exit 1; }
bash $HOME/binbo-shogi-data/work/xeon/archive_wst2.sh
echo "T188_STAGE cache-a $(date +%T)"; $PY cache-a
until grep -q "DL_OK\|DETACHED_DONE" $L/dl_suisho5_001_20261024-233309.log; do sleep 60; done
grep -q DL_OK $L/dl_suisho5_001_20261024-233309.log || { echo "download failed"; exit 1; }
echo "T188_STAGE extract $(date +%T)"; $PY extract
rm -f $HOME/binbo-shogi-data/public/nodchip_suisho5_depth9/shuffled.7z.001   # re-downloadable; sha256 file kept
echo "T188_STAGE cache-b $(date +%T)"; $PY cache-b
$PY manifests
df -h $HOME | tail -1
echo "T188_DATA_READY $(date +%T)"
