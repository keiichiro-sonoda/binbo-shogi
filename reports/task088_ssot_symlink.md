# Task 088 — コード管理の一本化: symlink SSOT + 実行版トレーサビリティ

**目的**: 087 でソースを追跡下へ移したが二重コピー(repo `experiments/alphazero/` = 正史 / pgx クローン内 = 実行場所)が残り、「手で同期」= 記憶依存でいつか drift → レポート再現性が静かに壊れる。本タスクは①クローンを repo 実体への **symlink** に置換して SSOT 化(同期という故障モードを消す)、②起動時に **git HEAD + dirty** をログへ刻む(どの版で回したか特定可能)。**実験・学習なし・コード管理のみ**。実測 2026-07-12。Rust・champion・(c) データ/ckpt 不触。

## やったこと
- **linker `experiments/alphazero/link_into_clone.sh`(新規・コミット・冪等)**: repo 内の各 `*.py`/`*.sh` について、クローンの同名を削除→ repo 実体へ `ln -s`。**26 本を symlink 化**(将来ファイル追加時は再実行 1 発)。pgx 原本 `train.py` は repo に無いゆえ非対象=実体のまま。
- **置換前 diff = 0**(クローン=repo 26 本 byte 一致)→ 乖離 reconcile 不要のクリーン cutover。**tar 退避**(scratch `alphazero_code_backup_pre088.tgz` 36K、repo に committed 版があるので保険)。
- **`scripts/detached.sh`**: wrapper 先頭に `RUN_META $(date) HEAD=$(git rev-parse --short HEAD) dirty=<repo全体> exp_dirty=<experiments/alphazero>` を追記。CWD=binbo root を継承ゆえ git 有効、`2>/dev/null` で非 repo でも無害。**087 の全実験ジョブがこれ経由**=1 箇所で全実行に版情報が乗る(各 t0xx ランチャは無改変)。

## 検証(学習なし最小、全 PASS)
| 項目 | 結果 |
|---|---|
| 全 26 symlink → repo 実体 | `readlink -f` で repo パス、逸脱 0 / `train.py` は実体のまま ✓ |
| `abspath(__file__)` 解決先 | クローン(`…/m4_data`)= **データ/出力は repo に漏れない** ✓ |
| 相互 import via symlink | `import m2_moves, m4_distill_train, m2_usi; from network import TransformerNet, AZNet` OK(`__file__`=クローン, `realpath`=repo)✓ |
| no-train smoke | `m2_usi.py --ckpt t087_wst/ws2.ckpt --bench 3 --bench-sims 8` → `sims=8: 8 ms/move`、ckpt をクローン側で解決 ✓ |
| detached RUN_META(dirty) | ダミージョブ = `RUN_META dirty=2 exp_dirty=1`、job body + `DETACHED_DONE rc=0` ✓ |
| detached RUN_META(clean) | 本コミット後にダミー再走で `dirty=0`(下記)✓ |
| binbo 本体ジョブ非破壊 | env.sh 読込・DETACHED_DONE 付与は不変(ダミーで確認)、header 1 行が増えるのみ ✓ |

## 注意点(記録)
- **`network.py` だけ pgx-tracked**(他 25 本は untracked)。symlink 置換は pgx git 上 typechange(現状すでに M)。クローンで `git checkout network.py`/`git pull` すると symlink が上流版に潰れる。**pgx は v2.6.0 pin・pull しない**運用ゆえ実害なし。README にも明記。
- 名前空間: 自作は `m1-m4_*`/`az_*`/`t087_*`/`network`/`train_m*` の接頭辞で pgx 原本(`train.py`)と分離、衝突なし。

## 成果物 / 規律
- 新規: `experiments/alphazero/link_into_clone.sh`、`reports/task088_ssot_symlink.md`。更新: `scripts/detached.sh`、`experiments/alphazero/README.md`、`STATUS.md`。作者のメモに**昇格ルール**(使い捨て探索コードが次タスクでも使う土台に育った時点で git 追跡へ=074→087 見逃しの再発防止)。
- クローン内 symlink 群・tar 退避は非 repo。`git status` 差分 = 上記 5 ファイルのみ。**Rust・champion・(c) データ/ckpt 不触。push は人間承認後。**
