# Task 153: binbo 自身の自己対局から勝敗つき教師を作る

2026-10-21、Codex。**自己対局 → 現 eval の QS leaf → 勝敗つき学習キャッシュを実装した。全量生成・学習・棋力判定はしていない。** [教師の方針](../docs/teacher_direction_2026-10-20.md)と [Task 150 調査の次の実験 3](task150_why_codex.md)に対応する。

追加ファイルは [生成器](../experiments/alphazero/t153_selfplay_gen.py)、[実エンジンの軽量試験](../experiments/alphazero/t153_selfplay_gen_test.py)、本報告の 3 点。既存ファイル・STATUS は変更せず、commit / push なし。GPU 不使用。永続データへの書き込みは `$DATA_ROOT/work/task153_dryrun/` 内だけ。Python は `scripts/env.sh` のものを使い、BLAS / OpenMP は各 1 スレッド、探索・QS の実行は最大 8 プロセス。

## 生成条件と勝敗

`gen --games N --depth D --procs P --out WORK` は **binbo の採用 eval 同士**をやねうら王で対局させる。Threads 1、Hash 16 MiB、MultiPV 1、FV_SCALE 16、book / ponder なし。やねうら王と重みのパスは `DATA_ROOT` / `YANE` を参照し、`--engine` / `--eval-dir` でも指定できる。暗黙の設定上書きを避けるため `eval_options.txt` のある eval は拒否する。

- 1 worker に USI エンジン 1 個を保持し、**各局の先頭でだけ `usinewgame` + `isready`**。初回もこの 1 回だけ。Task 146 追記の、局面ごとの約 133 ms の初期化を避ける。局内の TT / 履歴は引き継ぐ。
- 毎手 `position startpos moves ...` で、ランダム序盤を含む全手順を渡す。SFEN だけの入力で反復履歴を失うことはない。
- 既定の最初の 6〜16 ply を合法手から一様ランダムに選ぶ。`--opening-min` / `--opening-max`、`--seed` で変更可能。追加の `--random-move-prob`（既定 0）と `--random-until-ply`（既定 32）もある。追加乱数手では探索するが、その行は記録しない。MultiPV-2 は実装していない。
- `--write-minply` は既定 16。ply は **すでに指された手数、0 始まり**。ランダム部分と、この閾値未満の局面は記録しない。閾値前に終局する空の part も正しい完成局として扱う。
- 評価値は Task 146 と同じ **やねうら王内部 Value・その局面の手番視点**。表示 cp から `sign(cp)*ceil(abs(cp)*90/100)` で戻す。mate は ±32000（`-0` も負）。通常値の範囲外は失敗し、切り詰めない。
- `--nodes-cap K` を深さ制限と併用できる。上限到達時は最後の bound のない PV の値を採る。実到達深さ・その PV 出力の nodes・元の score 表示も保存する。nodes は探索全体の厳密な消費数ではなく、上限も厳密停止ではない。上限はレシピの一部で、再開時に変更できない。

記録する各行には SFEN、評価値、選択した合法手、ply、game ID、手番、到達深さ、nodes、mate 区分、表示値、最終勝敗がある。通常の指し手は全件 cshogi の合法手と照合する。勝敗は手番側の **勝ち +1 / 引き分け 0 / 負け −1**。全行が既知の対局結果を持ち、0 は「結果欠測」ではない。

終局方針は `manifest.json` と cache の `meta.json` にも保存する。

| 条件 | 結果 |
|---|---|
| 合法手なし | 相手の勝ち |
| 同一の盤面・持駒・手番が 4 回出現 | 引き分け。連続王手も、この生成器の裁定として引き分け |
| 27 点宣言が可能 | 宣言側の勝ち（cshogi LAW_27 / USI CSARule27） |
| `--max-ply`（既定 256）に到達 | 引き分け |
| 探索値の絶対値 ≥ `--eval-limit`（既定 3000 内部値） | 探索値が正なら手番側、負なら相手の勝ち |
| USI `resign` | 相手の勝ち。既定 ResignValue=99999 なので通常の評価値投了は抑止 |

評価値裁定は選択手を指す前。閾値に達した探索行も記録条件を満たせば保存するので、最終行の選択手だけは棋譜へ未適用のことがある。宣言と resign の終端応答には通常の選択手がないため、その問い合わせ自体は学習行にしない。評価値裁定は gensfen と同様の近似的な勝敗であり、すべてを実際の詰みまで指した結果ではない。

**反復は cshogi `is_draw()` に直接任せない。** 実機の king shuffle では 4 ply、最初の再出現で同 API が反応した。生成器は SFEN の盤面・持駒・手番を別に数え、4 回目で止める。連続王手の反則負けを区別しない点は上表の明示的な方針である。

## 決定性と再開

game seed は `SHA256("t153:基底seed:game_id")` の先頭 64 bit。各局は独立した乱数列・初期化境界を持つため、worker の割り当てやプロセス数で変わらない。**同一のエンジン・重み・コード・依存バージョン・設定**が再現性の条件。

`games/game_000000000000.json` のように **1 局 = 1 atomic part**。内容の SHA256 を持ち、書き込みは一時ファイル → flush / fsync → rename → 親 directory fsync。PID、経過時間、出力先を part に入れない。完成局を再利用し、欠損・破損局だけ再生成する。`--start-game G --games N` で範囲を追加し、manifest は依頼済み範囲の和集合を保持する。QS / assemble はその全範囲を要求する。

EOF / timeout / 非合法手等はエンジンを終了し、新しいエンジンで **その局全体を再実行**（既定 2 回再試行）。途中の局から継続しない。同じ WORK の操作は flock で排他し、Linux parent-death signal で親 SIGKILL 後の worker / engine の残存計算を防ぐ。深さ・seed・重み・スクリプト等が変わった再開は拒否する。`--procs` / `--timeout` / `--retries` は変更可能。

## QS と assemble

`qs --work WORK [--procs P]` の既定 eval は **task142_finetune**。`qs_replace.py` と同じ `qsearch_psv` 拡張・score=+1 の marker を使う。1 プロセスにつき 1 engine / Threads 1、局 part ごとに初期化し、QS 内の順番も固定する。並列数による葉の違いを避けるため、1 engine の QS worker は常に 1。

葉が元局面と変わった行は既定で捨てる。残る行のラベルは元局面と一致したまま。王手中の葉も捨てる。`--changed inherit` は変わった葉も残す実験用指定で、**元の探索ラベルを継承するだけで再ラベルではない**。PV が奇数手なら評価値と勝敗の両方を反転する。packed SFEN の手番・復元した葉の手番でも parity を照合する。全 kept 行で 77 列の特徴から SFEN への往復を確認する。元の ply / move は source row を介して追跡できるが、葉での合法手として扱ってはいけない。

`assemble --work WORK --out CACHE` は SQLite の **77 列 int32 の全バイト一致**で集合内の重複を除く。ハッシュ衝突による誤除去はない。既定は shared_64m の val および左右反転局面も除外し、`val_*` は同キャッシュへの symlink、validation 用別ラベルは Task 142 `labels_public` への symlink にする。元の 64M train との重複は比較しない。

`--holdout-fraction .1 --split-seed S` なら game ID の hash で局全体を train / val に分ける。この場合の `val_*` は新自己対局の holdout 配列で、shared val を置き換える。val を先に処理し、同一局面と val の左右反転を train に入れない。各 split に少なくとも 1 行必要。重複ラベルを平均せず、val 優先・同 split 内では小さい game ID / source row の値を残す。

出力は次の形式。labels を CACHE 内に置くことで、キャッシュとラベルを 1 回の directory rename で同時公開する。失敗・中断した staging は完成物として認識しない。assemble 自体は途中再開せず、新しい staging でやり直す。

```text
CACHE/
  train_indices.npy       int32 (N,77)
  train_evals.npy         int16 (N,)   内部 Value、葉の手番側
  train_wdl.npy           int8  (N,)   +1/0/-1、葉の手番側
  train_game_ids.npy      int64 (N,)
  train_plies.npy         int32 (N,)   根局面の ply
  val_*.npy                          既定は共有 val への symlink
  labels/
    train_dlshogi_evals.npy           ../train_evals.npy への symlink
    val_dlshogi_evals.npy             共有ラベルまたは ../val_evals.npy
    meta.json
  meta.json                          エンジン/重み/コード SHA256、深さ、seed、
                                     範囲、勝敗方針、QS、件数、part SHA256 等
```

trainer の `--eval-npy CACHE/labels` で既存形式の別ラベルとして読める。train は cache 自体の値と同じ。共有 val を Task 142 のラベルで評価するにはこの指定を使う（共有 cache の `val_evals.npy` は元のラベル）。**trainer の `--wdl-weight` の既定は 0** なので、勝敗を損失へ混ぜる比率は次の学習実験で別途指定する。このタスクでは学習条件を変更していない。

## 実行方法

リポジトリ直下から実行する。以下は小規模 pilot。出力先は新しいディレクトリを指定する。

```bash
source scripts/env.sh
export DATA_ROOT
S=experiments/alphazero/t153_selfplay_gen.py
W="$DATA_ROOT/work/task153_dryrun/my_pilot"

"$VENV_PY" -B "$S" gen --games 256 --depth 9 --procs 8 --out "$W"
# 同じコマンドで再開。範囲を追加する場合（それ以外の生成条件は同一）:
"$VENV_PY" -B "$S" gen --start-game 256 --games 256 --depth 9 --procs 8 --out "$W"
"$VENV_PY" -B "$S" qs --work "$W" --procs 8
"$VENV_PY" -B "$S" assemble --work "$W" --out "$W/cache"
# assemble が表示するコマンド:
"$VENV_PY" -B -m trainer.compact_cache "$W/cache" "$W/cache_compact"

# 同じ局面集合から、局単位 holdout を持つ別キャッシュ:
"$VENV_PY" -B "$S" assemble --work "$W" --out "$W/cache_holdout" \
  --holdout-fraction .1 --split-seed 20271531

# depth 7 / 8 / 9、同じ game seeds、各 256 局の計測:
"$VENV_PY" -B "$S" bench --games 256 --procs 8 \
  --out "$DATA_ROOT/work/task153_dryrun/bench_new"
# 試験（新規の出力先）:
"$VENV_PY" -B experiments/alphazero/t153_selfplay_gen_test.py \
  --out "$DATA_ROOT/work/task153_dryrun/tests_new"
```

## 実測と費用

以下は最終版の `bench_final/bench.json` に基づく。CPU は i7-14700KF、論理 28、本測定は **P=8**、Threads=1、affinity 指定なし。各深さ 256 局、seed 20271530、ランダム序盤 6〜16 ply、追加乱数手なし、write-minply 16、eval-limit 3000、max-ply 256、node cap なし。worker 起動・局の初期化・part 保存を含む。QS も P=8 で、生成終了後に別工程として実施。kept は QS の既定 changed-drop 後・集合全体の dedup 前。

| 深さ | 生成 秒 | QS 秒 | 記録行 | QS kept 行 | kept 率 |
|---|---:|---:|---:|---:|---:|
| 7 | 12.16 | 6.03 | 26,783 | 16,619 | 62.1% |
| 8 | 17.68 | 5.92 | 26,863 | 16,693 | 62.1% |
| 9 | 30.16 | 6.14 | 28,510 | 17,541 | 61.5% |

| 深さ | 局/時（生成） | 記録 万行/時（生成） | QS kept 換算 万行/時（生成） | 生成＋QS 万 kept行/時 |
|---|---:|---:|---:|---:|
| 7 | 75,784 | 792.9 | 492.0 | 329.0 |
| 8 | 52,115 | 546.9 | 339.8 | 254.7 |
| 9 | 30,554 | 340.3 | 209.4 | 173.9 |

10M / 32M / 64M **QS kept 行**に必要な時間（時間単位）:

| 条件 | 10M | 32M | 64M |
|---|---:|---:|---:|
| d7、P=8 実測速度 | 3.04 | 9.73 | 19.45 |
| d7、28 CPU 外挿・効率 60〜80% | 1.09〜1.45 | 3.47〜4.63 | 6.95〜9.26 |
| d8、P=8 実測速度 | 3.93 | 12.57 | 25.13 |
| d8、28 CPU 外挿・効率 60〜80% | 1.40〜1.87 | 4.49〜5.98 | 8.98〜11.97 |
| d9、P=8 実測速度 | 5.75 | 18.40 | 36.80 |
| d9、28 CPU 外挿・効率 60〜80% | 2.05〜2.74 | 6.57〜8.76 | 13.14〜17.52 |

終局理由は d7 が評価値裁定 252 / 最大手数 4、d8 が 252 / 2 / 反復 2、d9 が 244 / 8 / 反復 4。宣言・詰みまで指した局はこの pilot にはない。Task 146 の約 1,800 局面/秒・26 本という再ラベル実績とは、自己対局の TT 引継ぎ・序盤・結果確定・QS 棄却率が異なる。

再現に使った SHA256:

```text
engine 0831b1c1925c4e7577fdd3cecc2ac3234f724686b44c4d495a678c969fd186b7
eval   651ff9275a2fa416f195697270a1191967349d1601950f991b9bf522b9c8cece
tool   384df4763275119fdb4c756ab8331226994f3d61b93f32fc5a60c0447e990789
```

Python 3.12.3 / numpy 1.26.4 / cshogi 1.0.9。生の JSON / NPZ、ログと数値は `$DATA_ROOT/work/task153_dryrun/bench_final/`（起動ログは `bench_final.log`）。`bench_v1` / `bench_v2` は初期の反復検出条件の参考値で、最終表には用いていない。

28 CPU は **実測していない**。28/8 の線形倍率に対して効率 60〜80% を仮定する。P/E コア・HT・帯域・熱・長時間の裾で変わるため、この幅は保証ではない。計算式は `必要行数 / (8本実測の局面毎時 × 28/8 × 効率)`。本表は生成＋QS を含み、dedup / assemble / compact は含まない。最終的に dedup で r 割を捨てるなら、生成部分はさらに概ね `1/(1-r)` 倍必要。

**推奨は depth 9 / Threads 1 / Hash 16 / pilot P=8。** 深さ 7 は安いが、Task 146 / 150 と同じ深さの本線を保つ。教師品質や棋力の優位を速度だけから主張しない。本番の P=20 / 26 は別の短い較正で決める。今回は実行していない。

## 検証と未解決

最終版の `tests_fourfold/results.json` は全項目合格。

- depth 3、36 局、最大 160 ply、eval-limit 2000、追加乱数手 10%。1,846 記録行を棋譜の先頭から独立に再生し、局面・全選択手・実際の指し手の合法性、random / minply の除外、最終局面を照合。勝者の手番 939 行のうち 662 行（70.5%）が正の値。すべての行の結果を game winner と照合した。
- 同じ seed の **P=1 / 4 と ID 範囲分割**、親 SIGKILL → 再開、engine SIGKILL → 局単位 retry、内容を壊した JSON → 修復、のすべてで完成 game part が **バイト一致**。再実行は 36/36 局を再利用。設定変更は拒否。
- QS は 1,846 → 1,229 行、changed 617 行。`inherit` の奇数手 leaf **403 行**を含め、評価値と結果の両方の符号を元行と照合。QS の P=1 / 4 でも NPZ がバイト一致。再実行で完成 QS を再利用。
- default shared val の cache と、holdout 25% の cache（train 932 / val 290、重複等 7 行除去）を作成。game ID の非交差、train / val の同一局面の非交差、集合内重複除去、配列の dtype / shape と別ラベルの整列を確認。
- 両 cache を **`python -m trainer.compact_cache` が受理**し、保存後に復号した全 77 列が全行一致。欠損 QS part を含む assemble は拒否して完成ディレクトリを残さない。
- depth 9 + nodes cap 1000 の実対局も再生。両色の既知の 1 手詰め、最大手数 draw、27 点宣言の勝者、king shuffle の **2 / 3 回目は継続・4 回目だけ draw** の境界 fixture も合格。

さらに深さ 7 / 8 / 9 各 256 局、**合計 82,156 記録行**の全選択手を再生して合法性を照合した。反復 draw も実際に 4 回出現したことを確認。d9 の勝者手番は 13,224 行中 10,025 行（75.8%）が正。これらの符号分布は配線の健全性の参考で、探索値や裁定結果を外部正解として採点したものではない。

d9 の assemble は 17,541 行 → **17,503 train 行**、38 行除去。共有 val 50,000 行への mirror 除外用索引作成を含め **17.23 秒**（別途、軽量試験と並行して計測）、compact 本体は **0.2 秒**、train / val とも mismatch 0。この固定費を含む小集合の時間を 64M へ単純比例しない。SQLite が大きくなった場合の I/O 時間は未測定。

数値・ログは `$DATA_ROOT/work/task153_dryrun/tests_fourfold/`、速度測定後の再生・assemble の数値は `bench_final/validation.json`、compact の全行照合は `bench_final/compact.log`。初期版の試験ディレクトリは履歴として残し、最終結果と混ぜない。

本体の Rust、合法手生成、局面更新の実装には触れていないので Cargo の回帰試験は対象外。正しさの照合には既存 USI 実装と独立した cshogi の合法手・SFEN 往復・既知の詰み / 反復局面を使った。

残る論点:

1. **教師品質と棋力は未確認。** 小集合の局単位 holdout、同一の新局面で d9 / d12、採用重みからの小規模学習の順で調べる。Task 146 の旧分布への再ラベルとは条件が違うので、改善も失敗も先取りしない。
2. eval-limit 裁定・max-ply・ランダム序盤の強さと長さ・WDL 混合比。特に評価値裁定の勝敗が多く、独立した真の勝敗を追加したのと同義ではない。連続王手を反則負けにする変更は別レシピとして扱う。
3. 28 論理 CPU の実速、熱、長い探索の timeout、node cap を入れた際の到達深さの裾。全量を起動する前に短い較正が必要。
4. 保存容量とファイル数。1 局 1 JSON / QS NPZ は監査・局単位再開に向くが、大規模では inode と容量の費用がある。d9 pilot の raw game は約 9.23 MB、QS part は約 1.30 MB。これらと 323 bytes/row の dense cache（ID / ply 含む）・156 bytes/row の compact を保持すると、pilot の比率では **64M で約 64 GiB**、さらに filesystem と assemble 中の SQLite / 一時 raw / npy が要る。現在の空き約 52 GiB に収まるとは見込まない。全量前に容量を確保し、保持方針を決める。生成正本は再生成に時間がかかるため `/tmp` へ移さない。
