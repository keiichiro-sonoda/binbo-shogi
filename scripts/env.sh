#!/usr/bin/env bash
# Canonical paths for binbo-shogi generation/training (Task 044).
# Source this in any generation/training script instead of hardcoding /tmp:
#     source "$(git rev-parse --show-toplevel)/scripts/env.sh"
# Rationale + persistent/temp classification: docs/data_storage.md.
# /tmp is wiped on WSL reboot/sleep, so anything costly to reacquire lives under
# ~/binbo-shogi-data (persistent, outside any git repo). KP caches may stay in
# /tmp (regenerable in minutes via trainer/prepare_data).

DATA_ROOT="${DATA_ROOT:-$HOME/binbo-shogi-data}"

# Teacher engine (Gikou v2.0.2) — binary + params live together (engine cwd = dir).
export GIKOU="$DATA_ROOT/engines/gikou/gikou"
# Python env: torch 2.12.1+cu130 (CUDA) + numpy + cshogi. Works for CPU and GPU.
export VENV_PY="$DATA_ROOT/venv/bin/python"
# Data / models / logs (persistent, git-untracked).
export TEACHER_DIR="$DATA_ROOT/teacher"
export MODELS_DIR="$DATA_ROOT/models"
export LOGS_DIR="$DATA_ROOT/logs"
# Persistent cargo cache so offline/locked builds survive a /tmp wipe.
export CARGO_HOME="$DATA_ROOT/cargo"
# binbo engine binary (in the repo tree, which is itself persistent).
if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
  export BINBO="$(git rev-parse --show-toplevel)/target/release/binbo-shogi"
fi

# やねうら王 (Task 056 方向転換の土台, GPL-3.0). Pristine upstream clone + g++ ビルド済み。
# EvalDir に YANE_EVAL を渡す(または eval/nn.bin を cwd に置く)。eval/本体は非コミット・永続。
export YANE_DIR="$DATA_ROOT/engines/yaneuraou"
export YANE="$YANE_DIR/source/YaneuraOu-by-gcc"   # YaneuraOu NNUE 9.41git 64AVX2 (halfkp_256x2-32-32)
export YANE_EVAL="$YANE_DIR/eval"                  # smoke用 nn.bin(tanuki- Háo, GPL-3.0)。大会用は封印。

# KP feature cache: regenerable, may stay in /tmp (NOT persistent).
export CACHE_DIR="${CACHE_DIR:-/tmp/binbo-cache}"
