#!/bin/bash
# Task 158: binbo (YaneuraOu + adopted eval task154_R_result30) vs full-strength Gikou2 v2.0.2 at equal time.
# Design (fixed before the run): byoyomi 1000 ms per move for both, Threads 1 each, no ponder, no book, Gikou2
# DepthLimit 100 (= no limit), 6 random opening plies, colours swapped on the same opening, max 320 plies.
# 8 workers x 50 games = 400 games; each worker (both engines) is pinned to one logical CPU of a different P core
# (0,2,...,14) so the two engines share one core and never think at the same time. Hard stop at 23:45 regardless of
# results (games finished by then count). SPRT is disabled (alpha = beta = 1e-100).
# Gikou2 is built from source (v2.0.2 + docs commit ca25edf) with one local fix: softmax() on an empty valarray
# (no legal moves) is UB and segfaults with libstdc++ 13; it now returns the empty array (no effect on play).
set -u
D=$HOME/binbo-shogi-data; OUT=$D/work/task158; mkdir -p "$OUT"
deadline=$(date -d "today 23:45" +%s); now=$(date +%s); secs=$((deadline - now))
for w in 0 1 2 3 4 5 6 7; do
  cpu=$((2 * w)); seed=$((20291000 + 100 * w))
  timeout "$secs" taskset -c "$cpu" "$(dirname "$0")/../../target/release/binbo-shogi" yane-vs-gikou \
    "$D/engines/yaneuraou/source/YaneuraOu-by-gcc" "$D/eval_dirs/task154_R_result30" "$D/engines/gikou/gikou" \
    1000 100 "$seed" 50 6 320 1000 0 10 1e-100 1e-100 > "$OUT/main_w$w.tsv" 2> "$OUT/main_w$w.err" &
done
wait
echo "T158_DONE $(date)"
