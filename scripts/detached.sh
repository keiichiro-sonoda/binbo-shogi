#!/usr/bin/env bash
# scripts/detached.sh — 任意の長時間ジョブを完全 detach 起動する汎用 launcher (Task 063).
#
# なぜ必要か(再診断済みの真因, docs/long_running_jobs.md 参照):
#   harness の background ジョブは Claude プロセスの「子」なので、Claude/シェルが終わると
#   道連れで kill される。さらに WSL2 では distro に残プロセスが無くなると VM ごと自動停止
#   する(/tmp も wipe)。ホストスリープは無関係(AC standby=0=既に無効、確認済み)。
#   → setsid で新セッション/PID1 へ再親付けすれば、(1) セッション終了に耐え、
#     (2) 実行中はプロセスが残るので WSL VM も保持される。これ単体で事故は防げる。
#
# 使い方:
#   scripts/detached.sh <tag> <cmd> [args...]
#     <tag>  ログ名の stem。log = $LOGS_DIR/<tag>_<YYYYmmdd-HHMMSS>.log(永続・git 非追跡)
# 例:
#   scripts/detached.sh gen8m   bash teacher/run8m_e.gen.sh
#   scripts/detached.sh sprt_x  "$BINBO" yane-vs-yane "$YANE" eval_a eval_b 100
#
# 完了判定: ログ末尾に必ず `=== DETACHED_DONE rc=<N> <date> ===` を追記する(失敗時も rc 付き)。
#   PID は setsid の再 fork 次第で transient を掴むことがあるため、待機はこの marker を正とする:
#     while pgrep -af '<distinctive>' >/dev/null && ! grep -q DETACHED_DONE "$log"; do sleep 20; done
#
# 注: detached.sh は detach+log の汎用層のみ。プロセス死亡時の自動再開(resume ループ)は
#     ジョブ固有の関心事なので、渡すコマンド側に持たせる(例: teacher/*.overnight.sh の流儀)。
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: scripts/detached.sh <tag> <cmd> [args...]" >&2
  exit 2
fi

cd "$(git rev-parse --show-toplevel)"
# 永続パス($LOGS_DIR, $BINBO, $GIKOU ...)を読み込む。/tmp をハードコードしない(Task 044)。
source scripts/env.sh

tag="$1"; shift
mkdir -p "$LOGS_DIR"
log="$LOGS_DIR/${tag}_$(date +%Y%m%d-%H%M%S).log"

# bash -c '"$@"; ...' でコマンドを実行し、終了後に完了 sentinel を同じログへ追記する。
# set -e はこの子シェルに伝播しないので、コマンドが失敗しても marker は必ず付く(rc を記録)。
# 先頭に RUN_META(git HEAD + dirty 数)を刻む = 後から「どの版で回したか / 未コミット版か」を
# 必ず特定可能に(Task 088)。CWD=repo root を継承ゆえ git は効く。git 失敗は 2>/dev/null で無害化。
setsid nohup bash -c 'echo "=== RUN_META $(date "+%F %T") HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo NA) dirty=$(git status --porcelain 2>/dev/null | wc -l) exp_dirty=$(git status --porcelain -- experiments/alphazero 2>/dev/null | wc -l) ==="; "$@"; rc=$?; echo "=== DETACHED_DONE rc=$rc $(date) ==="' _ "$@" \
  </dev/null >"$log" 2>&1 &
pid=$!

echo "detached: tag=$tag pid=$pid log=$log"
echo "  follow: tail -f $log"
echo "  done? : grep -q DETACHED_DONE $log   # 完了は marker で判定(PID には頼らない)"
