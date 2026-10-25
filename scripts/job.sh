#!/usr/bin/env bash
# scripts/job.sh — 長時間ジョブを「途中で止めて、後で続きから再開」するための操作(Task 106)。
#
#   scripts/job.sh start  <tag> <cmd> [args...]   起動(detached.sh + 熱 guard)。コマンドを記録する
#   scripts/job.sh pause  <tag> [--now]           止める。既定 = 区切りの良いところで保存して止まる(待つ)
#                                                 --now = すぐ止める(最後のチェックポイント以降は失う)
#   scripts/job.sh resume <tag>                   記録したコマンドで再開(完了済みの段は飛ばされる)
#   scripts/job.sh status [<tag>]                 実行中か・最後のログ行
#
# 仕組み:
#   - 停止要求 = ファイル $DATA_ROOT/run/STOP_<tag>。ジョブには環境変数 BINBO_STOP_FILE で渡す。
#     対応しているドライバ(t106 以降)は、学習なら数分ごとの区切りで状態を保存して rc=75 で終わり、
#     rescore はシャード(1 万局面、約 10 秒)ごと、対局は 100 局のかたまりごとに止まる。
#   - 再開は同じコマンドを流し直すだけ。各段は「出来上がった成果物があれば飛ばす」作りなので、
#     学習は train_state.pt から、rescore は保存済み part から、対局は終わったかたまりから続く。
#   - WSL の自動更新などで強制終了した場合も resume で同じように続く(失うのは最後の保存以降だけ)。
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
RUN_DIR="${DATA_ROOT:-$HOME/binbo-shogi-data}/run"
mkdir -p "$RUN_DIR"

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ "$#" -ge 1 ] || usage
sub="$1"; shift

latest_log() { ls -t "$LOGS_DIR/$1"_[0-9]*.log 2>/dev/null | head -n1 || true; }

# そのログを開いているプロセス(= ジョブ本体)の PID
holders() {
  local log; log="$(readlink -f "$1")"
  for fd in /proc/[0-9]*/fd/*; do
    [ "$(readlink "$fd" 2>/dev/null)" = "$log" ] && echo "${fd#/proc/}" | cut -d/ -f1
  done | sort -u
}

running() { local log; log="$(latest_log "$1")"; [ -n "$log" ] && ! grep -q DETACHED_DONE "$log" && [ -n "$(holders "$log")" ]; }

case "$sub" in
  start|resume)
    tag="${1:?tag}"; shift || true
    if running "$tag"; then echo "$tag: 実行中です(先に pause するか、終わるのを待ってください)"; exit 1; fi
    if [ "$sub" = start ]; then
      [ "$#" -ge 1 ] || usage
      printf '%q ' "$@" > "$RUN_DIR/$tag.cmd"
    fi
    [ -f "$RUN_DIR/$tag.cmd" ] || { echo "$tag: 記録されたコマンドがありません(start で起動したジョブだけ再開できます)"; exit 1; }
    rm -f "$RUN_DIR/STOP_$tag"
    # shellcheck disable=SC2046
    eval "BINBO_STOP_FILE=$RUN_DIR/STOP_$tag scripts/detached.sh $tag $(cat "$RUN_DIR/$tag.cmd")" >/dev/null
    sleep 2
    log="$(latest_log "$tag")"
    # The thermal guard is optional (its monitor is not part of the public tree).
    if [ -f scripts/monitor/hwwatch.py ]; then
      scripts/detached.sh "guard_$tag" python3 scripts/monitor/hwwatch.py --guard-log "$log" --no-log >/dev/null
    fi
    echo "$tag: ${sub} しました。log=$log"
    ;;
  pause)
    tag="${1:?tag}"; now="${2:-}"
    log="$(latest_log "$tag")"
    if ! running "$tag"; then echo "$tag: 実行中ではありません"; exit 0; fi
    touch "$RUN_DIR/STOP_$tag"
    if [ "$now" = "--now" ]; then
      for pid in $(holders "$log"); do
        pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
        [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null || true
      done
      echo "$tag: すぐに止めました(再開: scripts/job.sh resume $tag)"
    else
      echo "$tag: 停止を要求しました。区切りで保存して止まるのを待ちます(学習なら数分、対局なら最長 20 分ほど)…"
      while running "$tag"; do sleep 10; done
      echo "$tag: 止まりました(再開: scripts/job.sh resume $tag)"
    fi
    tail -2 "$log"
    ;;
  status)
    for f in "$RUN_DIR"/*.cmd; do
      [ -e "$f" ] || continue
      tag="$(basename "$f" .cmd)"
      [ -n "${1:-}" ] && [ "$1" != "$tag" ] && continue
      log="$(latest_log "$tag")"
      if running "$tag"; then st="実行中"; elif [ -n "$log" ] && grep -q "DETACHED_DONE rc=0" "$log"; then st="完了";
      elif [ -n "$log" ] && grep -q "DETACHED_DONE rc=75" "$log"; then st="一時停止中";
      else st="停止(resume で再開可)"; fi
      echo "$tag: $st  log=${log:-なし}"
      [ -n "$log" ] && tail -1 "$log" | cut -c1-160 | sed 's/^/    /'
    done
    ;;
  *) usage ;;
esac
