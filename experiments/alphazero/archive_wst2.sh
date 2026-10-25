#!/usr/bin/env bash
# Move the gikou2-lineage training material (not used by the student lineage, decision 2026-10-24) to the Xeon HDD.
# Copy -> sha256 on both sides -> delete local only on a match -> leave an ARCHIVED.txt note in each source dir.
set -uo pipefail
D=$HOME/binbo-shogi-data
DEST=${BINBO_ARCHIVE_ROOT:?set BINBO_ARCHIVE_ROOT to the archive directory on the worker}/binbo-shogi-data
WORKER=${BINBO_WORKER_HOST:-xeon-worker}   # ssh の接続先の別名 (~/.ssh/config で定義する)
FILES=()
DIRS=(work/task173_wst2 work/task177_wst2_full)
cd "$D"
list=$(mktemp)
for f in "${FILES[@]}"; do echo "$f"; done > "$list"
for d in "${DIRS[@]}"; do find "$d" -type f; done >> "$list"
echo "FILES $(wc -l < "$list") BYTES $(xargs -d '\n' du -cb < "$list" | tail -1 | cut -f1)"
rsync -a --files-from="$list" "$D/" "$WORKER:$DEST/" || { echo "RSYNC_FAILED"; exit 1; }
echo "COPIED $(date +%T)"
xargs -d '\n' sha256sum < "$list" | sort -k2 > "$list.local"
ssh "$WORKER" "cd $DEST && xargs -d '\n' sha256sum" < "$list" | sort -k2 > "$list.remote"
if ! diff -q "$list.local" "$list.remote" > /dev/null; then echo "SHA_MISMATCH"; diff "$list.local" "$list.remote" | head; exit 1; fi
echo "SHA_OK $(wc -l < "$list.local") files"
stamp=$(date +%FT%T)
for d in work/task173_wst2 work/task177_wst2_full; do
  { echo "Archived $stamp to Xeon (ssh $WORKER) $DEST/$d"
    echo "Reason: WS-T2 (Task 173/177) training data and checkpoints; not used by the student lineage before the event."
    echo "Restore: rsync -a $WORKER:$DEST/$d/ $D/$d/   (sha256 below)"
    grep " $d/" "$list.local"; } > "$D/$d/ARCHIVED.txt.new"
done
xargs -d '\n' rm -f < "$list"
for d in "${DIRS[@]}"; do find "$D/$d" -type d -empty -delete 2>/dev/null; done
for d in work/task173_wst2 work/task177_wst2_full; do mkdir -p "$D/$d"; mv "$D/$d/ARCHIVED.txt.new" "$D/$d/ARCHIVED.txt" 2>/dev/null || true; done
cp "$list.local" "$D/work/xeon/archive_wst2_sha256.txt"
echo "DONE $(date +%T)"; df -h "$D" | tail -1
