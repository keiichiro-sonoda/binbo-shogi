#!/bin/bash
# Task 169: gate 0 of the GPU idea ("ask the GPU policy whether extra search is worth it"), fixed before running.
# Question: does last night's post-hoc gain reproduce on new positions? Rule (same as the post-hoc analysis on Task 167):
# if the old WS-T policy's top-1 equals the N-node search move, stop at N; otherwise give a fraction of the disagreeing
# roots 4N and the rest 1.6N so that the mean budget is 1.6N (uniform baseline 1.6N). Primary = the implementable variant
# ('ll': N then +3N keeping the hash table). PASS if its reduction vs uniform has a 95% bootstrap CI lower bound > 0.
# Also reported: the cold-4N variant, and both with the 2 largest contributors removed (robustness).
# New roots: 1,536 from ROUND 1 self-play games (work/task154/games, games of task142; last night used round 2 games),
# seed 20271690; same eval task154_R_result30, same budgets (N=100k; 1.6N, 4N, 20N, 40N), same tool t167_probe_rescue.py.
set -eu
source scripts/env.sh
export PYTHONDONTWRITEBYTECODE=1
P=experiments/alphazero/t167_probe_rescue.py; WK=$DATA_ROOT/work/strategy/gate0
"$VENV_PY" $P sample --work $WK --games-dir $DATA_ROOT/work/task154/games --count 1536 --seed 20271690
"$HOME/task074_frameworks/venv_pgx/bin/python" $P policy --work $WK --device auto 2>&1 | grep -v -i "cuda\|plugin\|warning" | tail -2
"$VENV_PY" $P search --work $WK --procs 26
"$VENV_PY" $P report --work $WK --bootstrap 2000
"$VENV_PY" experiments/alphazero/t167_posthoc_agree_alloc.py $WK | tee $WK/gate0_rule.txt
echo T169_DONE
