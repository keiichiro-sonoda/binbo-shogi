#!/usr/bin/env bash
# SSOT wiring (Task 088): this repo dir (experiments/alphazero/) holds the REAL source.
# The runtime lives in a pgx clone; this script replaces each same-named file there with
# a symlink back to the real file in this repo so there is one source of truth (no drift).
#
# Idempotent — rerun after adding a new .py/.sh here to (re)link it into the clone.
# pgx-original files (only train.py) are NOT in this repo, so they are never touched.
# Data/ckpt/log dirs stay in the clone (scripts resolve them via abspath(__file__)).
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"                 # .../binbo-shogi/experiments/alphazero
CLONE="${1:-$HOME/task074_frameworks/pgx/examples/alphazero}"
[ -d "$CLONE" ] || { echo "clone dir not found: $CLONE" >&2; exit 1; }

n=0
for f in "$REPO"/*.py "$REPO"/*.sh; do
  b="$(basename "$f")"
  [ "$b" = "link_into_clone.sh" ] && continue        # don't link the linker itself
  rm -f "$CLONE/$b"                                   # drop clone copy (or stale symlink)
  ln -s "$f" "$CLONE/$b"                              # point clone -> repo file
  n=$((n + 1))
done
echo "linked $n files: $CLONE/<name> -> $REPO/<name>"
