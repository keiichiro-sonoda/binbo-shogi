#!/usr/bin/env bash
# Task 076 M1: remaining measurement grid (center point already measured).
# One axis varied at a time from center = sims32 / ch64 L5 / batch256 / steps256.
# Riskiest points (steps512, net256x20) last so earlier data survives an OOM/crash.
set -u
cd "$(dirname "$0")"
COMMON="env_id=shogi training_batch_size=1024 eval_interval=100 max_num_iters=4 seed=0"

./m1_run_point.sh sims16    $COMMON num_simulations=16 selfplay_batch_size=256 max_num_steps=256 num_channels=64  num_layers=5
./m1_run_point.sh sims64    $COMMON num_simulations=64 selfplay_batch_size=256 max_num_steps=256 num_channels=64  num_layers=5
./m1_run_point.sh net128x10 $COMMON num_simulations=32 selfplay_batch_size=256 max_num_steps=256 num_channels=128 num_layers=10
./m1_run_point.sh batch128  $COMMON num_simulations=32 selfplay_batch_size=128 max_num_steps=256 num_channels=64  num_layers=5
./m1_run_point.sh batch512  $COMMON num_simulations=32 selfplay_batch_size=512 max_num_steps=256 num_channels=64  num_layers=5
./m1_run_point.sh steps128  $COMMON num_simulations=32 selfplay_batch_size=256 max_num_steps=128 num_channels=64  num_layers=5
# instrumented animal_shogi @ M0 config for the like-for-like dobutsu comparison
./m1_run_point.sh animal    env_id=animal_shogi training_batch_size=1024 eval_interval=100 max_num_iters=4 seed=0 num_simulations=32 selfplay_batch_size=512 max_num_steps=64 num_channels=64 num_layers=5
# risky points last
./m1_run_point.sh steps512  $COMMON num_simulations=32 selfplay_batch_size=256 max_num_steps=512 num_channels=64  num_layers=5
./m1_run_point.sh net256x20 $COMMON num_simulations=32 selfplay_batch_size=256 max_num_steps=256 num_channels=256 num_layers=20
echo "=== GRID_ALL_DONE ==="
