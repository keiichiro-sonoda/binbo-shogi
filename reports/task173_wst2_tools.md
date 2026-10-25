# Task 173: WS-T2 の作成・学習・評価ツール

2026-10-23。納品物はこの作業領域内の `experiments/alphazero/t173_wst2.py` と
`tests/test_t173_wst2.py`。既存リポジトリのファイル、STATUS、元データ、元チェックポイントは変更せず、
コミットもしていない。ソースの共通処理は `--repo` から読み込む。

**CPU の小規模試験まで完了。GPU 本学習・全自己対局データでの改善判定は未実施。**
ここにある4ステップの重みは動作確認用であり、WS-T2 の採用候補ではない。

## 実行環境と準備

リポジトリ直下で実行する。パスの共通設定は既存の `scripts/env.sh` を利用し、
Python だけ JAX/pgx 環境を明示する。出力先はすべて永続領域。

```bash
REPO="$PWD"
source "$REPO/scripts/env.sh"
TASK173_WORK="$DATA_ROOT/work/task173_wst2"
PGX_PY="${PGX_PY:-$HOME/task074_frameworks/venv_pgx/bin/python}"
TOOLS="$TASK173_WORK/experiments/alphazero/t173_wst2.py"
export PYTHONDONTWRITEBYTECODE=1
```

検証した環境: Python 3.12 / JAX・jaxlib 0.10.2 / pgx 2.6.0 / dm-haiku 0.0.16 /
optax 0.2.8 / cshogi 1.0.2。CPU の affinity は既定4論理CPU。
GPU は `--device gpu` で必須指定でき、未検出なら失敗する。
`--device auto` は利用可能ならGPU、なければCPUを選ぶ。
観測・合法手生成はベクトル化した pgx をCPUで動かし、バッチ単位でネットのデバイスへ渡す。

## build: 元ゲームの除外と分割

```bash
"$PGX_PY" "$TOOLS" --repo "$REPO" --device cpu build \
  --games "$DATA_ROOT/work/task154/games" "$DATA_ROOT/work/task159/games" \
  --out "$TASK173_WORK/dataset"
```

既定で `strategy/gate0/roots.json` と `strategy/probe13/main/roots.json` の
**全1,792ゲーム**を、安定性や詰みの有無にかかわらず除外する。参照元のファイルSHA、
game_id、元ゲームの内容チェックサムを照合する。
元ファイルの絶対パスに加えて `(recipe_id, game_id)` でも照合するため、別パスへコピーした同じゲームも除外する。
各ラウンドで game_id が0から再開するため、番号だけの照合では両ラウンドを区別できない。
同じキーの入力重複は一度だけ使い、内容が異なればエラー。

除外後のキーを `SHA256([20271730, key])` 順に並べ、先頭 `floor(ゲーム数*0.02)` を検証用にする。
小規模試験でも検証用は最低1ゲーム、学習用は最低1ゲーム必要。
検証ゲームは学習に一切出さず、入力ディレクトリを指定する順序によって分割は変わらない。
`games.jsonl` に元ファイル、キー、内容SHA、分割、採用局面数を記録する。
`meta.json` は全データを書き終えてから作る完了印。
既存出力先への build は拒否するので、失敗したディレクトリは別名に退避するか新しい出力先を選ぶ。

局面は元の探索行をそのまま使い、QSへの置換は行わない。
ply 16未満、opening_plies未満、random_pliesに含まれる行を除く。
非詰みは深さ9以上を要求し、指し手が元局面のcshogi合法手に含まれること、手番・勝敗・手数の対応を確認する。
詰みを見つけて深さ9より前に完了した行も残す。
policy は `m2_moves.usi_to_action` の pgx action ID。

value は元行の `score`（**手番側の internal Value**）を使う。`raw` のUSI cpには換算しない。

```text
p = sigmoid(clip(score, -3000, 3000) / 600)
mate=True または abs(score)=32000 の場合: p=1（正）または0（負）
勝ち: target = 2*(0.7*p + 0.3*1)-1
負け: target = 2*(0.7*p + 0.3*0)-1
引分: target = 2*p-1
```

`train.bin` / `val.bin` は208 bytes/局面の構造化レコード。
SFEN（最大192 bytes、超過は切り捨てずエラー）、action uint16、value float32、勝敗 int8、
ゲーム索引 uint32、ply uint16、元score int16、mate uint8を格納する。
2,000万局面ならデータ本体は約4.16 GB（約3.87 GiB）で、観測 `(9,9,119)` は保存しない。
学習・評価開始時にデータのSHA、ゲームの除外、各レコードの分割先を再検査する。

## train: WS-Tから継続、約3 GPU時間を起動時に較正

```bash
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu train \
  --data "$TASK173_WORK/dataset" --out "$TASK173_WORK/train"
```

初期重みは `PGX_HOME/pgx/examples/alphazero/t087_wst/ws2.ckpt`
（PGX_HOMEの既定は `~/task074_frameworks`）。
SHA256 `20bf5e826e307cf191078f1cb86bf2c6f7bf79ef6e89bb2d40829844950fe2b3` を必須照合する。
形は元チェックポイントと同じ、Transformer、81 tokens、d_model=240、5層、8 heads、ff_mult=4。
大きい形の新設は今回のツールには含めていない。
保存したチェックポイントは既存 `m2_usi.load_model` で読めることも確認した。

既定値:

|項目|既定|
|---|---|
|optimizer|Adam、lr=1e-5、通常のb1/b2/eps|
|loss|合法手だけのpolicy CE + value MSE（係数各1）|
|batch|GPU: 32, 64, 128, 256, 512, 1024を順に試し、成功した最大値。CPU: 8|
|計測|各候補を2回ウォームアップ後、5ステップ。`--bench-steps` で変更可能|
|学習量|`--hours 3`。較正時間を引いた残り時間の90%を、実測秒/stepで割って総stepを固定|
|lr schedule|最初のmin(100, 総stepの10%)で線形warmup、その後cosineで初期lrの0.1倍まで|
|保存|1,000stepごとと実行終端。`--checkpoint-every` で変更可能|
|乱数|20271730。ゲーム分割とは別の、学習局面のepoch順列に使用|

計測にはメモリマップのバッチ抽出、SFENパース、pgxの合法手・観測生成、転送、
forward/backward、Adam更新、重み・optimizerまでの同期を含め、JITコンパイルを除く。
ベンチマーク中の更新は捨て、元WS-Tと新規Adam状態から本学習を開始する。
`throughput.json` に全候補の速度、選択したbatch、JIT時間、推定時間を残す。
GPUメモリ不足は次の候補を止め、それまで成功したbatchを使う。1024超は明示的な `--batch` で指定できる。
この環境でGPUは試せていないため、GPUでの実際のbatch・速度・最終step数を事前の測定値としては記載できない。
3時間は固定step数を決めるための見積もりで、実行中の負荷変動や保存時間まで保証するタイマーではない。

総更新数は `--steps 30000`、または `--epochs 2` でも指定可能（両方同時は不可）。
各epochはseed付き全局面順列で、端数は次epochへつないで固定batchを作るため、捨てる行はない。
検証・プローブを使ったチェックポイント選択はこの学習器では行わない。

```bash
# latest.jsonの相対ファイル名を得る（train完了後、または中断後）
LATEST=$("$PGX_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
  "$TASK173_WORK/train/latest.json")

"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu train \
  --data "$TASK173_WORK/dataset" --out "$TASK173_WORK/train" \
  --resume "$TASK173_WORK/train/$LATEST"
```

再開は重み、Adamのmoment/count、固定schedule、seed、batch、絶対stepから復元する。
`--steps` / `--epochs` の再指定は拒否し、その他の学習設定も保存値を使う。
データID、学習スクリプト、依存ソース、ライブラリの版が変わった再開は拒否する。
`--stop-after 100` はscheduleを変更せず100更新で保存して戻るので、段階的な試走にも使える。
チェックポイントの本体、`latest.json` は一時ファイルから置換し、flush/fsyncする。

## eval: 旧版も再推論して比較

```bash
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu eval \
  --data "$TASK173_WORK/dataset" \
  --checkpoint "$TASK173_WORK/train/$LATEST" \
  --out "$TASK173_WORK/eval"
```

出力は `eval.json`、`ws2_policy.json`、`wst2_policy.json`。
`reference_best_move` を参照正解とし、stable=true / mate=false の **gate0 1,022 + probe13 162 = 1,184局面**
で旧WS-TとWS-T2のtop-1 / top-2を測る。想定件数と異なる場合は失敗する。
両方のモデルで全プローブのpgx/cshogi合法手集合の完全一致を確認し、合法手の中で順位・確率を計算する。
SFENの二桁持駒は一桁ずつ解釈せず、整数としてパースしてpgxの `_from_board` へバッチで渡す。
観測は既存t167と同じ `_observe(..., flip=True)`。

検証ゲームは全局面についてdepth-9指し手へのtop-1 / top-2、value Brier、学習targetのMSEを測定する。
Brierは `p=(value+1)/2`、実結果は負け0・勝ち1・引分0.5として `(p-z)^2` の平均。
勝敗が決まった行と引分の内訳も出す。長いゲームほど寄与が大きい局面平均。
検証側の全合法手集合照合を追加する場合は `--check-val-legal`。

関門0は新しいpolicy top-1と既存N探索の手の一致を取り直し、既存の探索損失を使う。
`t167_posthoc_agree_alloc.py` の `rule` と同じ計算を関数化した
（元スクリプトはimport時にI/Oを実行するため、直接importはしない）。
一致ならNで止め、不一致には1.6Nと4Nを混ぜ、平均1.6Nにそろえる。
独立4N (`l4`) とN+3N継続 (`ll`) の両方、確率0.3条件、寄与の大きい2局面を除く条件、
seed固定2,000回のbootstrapを、全体と各プローブ群に出す。
不一致0件のとき0を返す元スクリプトの規約も保持し、その場合をJSONで明示する。
探索手や参照値は変えないため、新policyの手を改めて探索する必要はない。
これは既存集合上の配分規則の再計算であり、実際の時間管理での棋力判定ではない。

部分データや `--probe-limit` / `--val-limit` を用いる評価は、明示的な `--allow-subset` が必要。
出力にも `subset=true` を残す。候補の学習データIDと、現在の全プローブゲームの除外を再検査する。

## CPU検証結果

```bash
"$PGX_PY" "$TASK173_WORK/tests/test_t173_wst2.py"
```

毎回 `tests_output/cpu_*` に独立したデータとチェックポイントを作る。
両ラウンドの実ゲーム24件×12局面=288局面（学習276、検証12）で、次の6試験を実施した。

1. buildのバイト単位の再現性、入力順序を反転した分割一致、同じ番号の異ラウンド識別、
   プローブ元を別名コピーした2件の除外、除外監査を改変したデータの拒否。
2. 値の符号・clamp・勝敗混合・引分・詰みの数式と、実行から保存したaction/value/plyの照合。
3. 288局面と二桁持駒を持つ先後2局面の全合法手集合照合。
   t167方式の観測との一致。巨大な非合法logitを加えた推論と学習でのマスク、非合法logitへの勾配ゼロ。
4. epoch末尾をまたぐサンプラーの再現性と全局面の出現数、lr scheduleの端点。
5. 既存スクリプトからASTで取り出した `rule` と、新関数の全1,184局面・4条件での一致（絶対誤差1e-12以内）。
6. 元WS-Tを4step学習し、2step+保存+再開との重み・Adam全leafの**ビット一致**。
   小規模eval、部分評価の誤表示防止、既存USI用ローダーでの読込み。

6試験すべて成功。ログは `tests/cpu_final.log`。
ビット一致の保証は検証したCPU・同一バージョン内の結果で、異なるデバイス間のビット一致は主張しない。
CPU単独の計測は `smoke/train_cpu_final/throughput.json` と `smoke/train_cpu_final.log`。
batch=8、4 CPU threads、計測20stepで **86.96局面/秒、0.0920秒/step**。
ウォームアップ/JITは4.22秒。GPUへの速度換算はしていない。

追加で `smoke/dataset`（24ゲーム、288局面）から学習した4step版について、
**全1,184プローブ**と検証12局面、bootstrap 2,000回のevalを実行した。
結果は `smoke/eval_final/eval.json`（最終ソースと計測用の4stepチェックポイントで再確認）。
小規模データのため `subset=true`。

|指標|旧WS-T|4step動作確認版|
|---|---:|---:|
|全1,184プローブ reference top-1|45.9459%|45.9459%|
|全1,184プローブ reference top-2|61.2331%|61.3176%|
|検証12局面 depth-9 top-1|33.3333%|33.3333%|
|検証12局面 Brier|0.141185|0.141290|
|gate0群 関門0・継続4N 改善cp|+4.6238 [+0.6746, +9.8410]|同じ|
|gate0群 上位寄与2局面除外・継続4N 改善cp|+3.5160 [+1.1166, +6.1656]|同じ|
|両群合計 関門0・継続4N 改善cp|+4.9737 [+1.5537, +9.5054]|同じ|

旧WS-Tの再推論top-1は既存GPU保存結果と1,184/1,184一致し、Task 169の関門0 +4.6cp も再現した。
top-2は1,183/1,184で一致。残るgate0 r0383では既存GPU結果の2位と3位の確率差が約1e-5で、
CPU再計算時に入れ替わった。最大top-1確率差は約0.0015。
本比較は両モデルを同じCPU・同じバッチ条件で再計算している。
4stepの差は改善の証拠ではなく、学習・評価の経路が動くことの確認である。

## 前提と残る実行

元ゲームは既存Task 153形式で、同じrecipe/idは同じゲームとみなす。プローブのsourceファイルとSHAが読めない場合は、
除外を推測して進めずエラーにする。全ゲーム同士の同一局面の重複除去は行わず、分割は依頼どおりゲーム単位。
SFENから復元できない履歴は既存t167と同じ扱いで、現在局面の観測を作る。
元ゲームの勝敗には元の終局・打ち切り規約が反映され、新しい勝敗の付け直しはしない。

残るのは全量build、GPU起動時のbatch/速度較正、約3時間の本学習、その候補の全検証ゲーム・全プローブ評価。
CPU試験ではGPUのメモリ不足処理やGPU実測速度は確認できていない。
出力容量はデータ本体に加えて保存回数分のAdam付きチェックポイントが必要で、長時間実行前に永続領域の空き容量を確認する。
既存の運用規約に従い、無人実行前には作者側でマシンのスリープ無効化を確認する。
