# Task 167: Probe 1 / 3 の実行ツール

`experiments/alphazero/t167_probe_rescue.py` を追加。既存ファイル・対局・モデルは変更せず、コミットしない。本番256局面の計測は未実施。

## 本番の実行

リポジトリのルートで実行する。先にマシンのスリープ無効化を確認する。生成物は永続領域に置く。

```bash
source scripts/env.sh
export PYTHONDONTWRITEBYTECODE=1
PROBE=experiments/alphazero/t167_probe_rescue.py
WORK="$DATA_ROOT/work/strategy/probe13"
PGX_PY="$HOME/task074_frameworks/venv_pgx/bin/python"

"$VENV_PY" "$PROBE" sample --work "$WORK"
"$PGX_PY" "$PROBE" policy --work "$WORK" --device auto
"$VENV_PY" "$PROBE" search --work "$WORK" --procs 20
"$VENV_PY" "$PROBE" report --work "$WORK" --bootstrap 2000
```

出力は `roots.json`、`policy.json`、`search_manifest.json`、`search/rXXXX.json`、`report.json`、短い日本語の `summary.md`。policy を先に走らせれば、search は上位2手の子局面も一度で評価できる。sample → search → policy の順でもよいが、最後に search をもう一度実行して子局面を補充する。report は欠測を黙って落とさずエラーにする。

search は同じコマンドで再開する。各根の各探索結果が揃うたびに、一時ファイルへの書き込み・fsync・rename でJSONを更新する。pool の各プロセスが専用エンジンを持ち、探索エラー時はエンジンを作り直して1回再試行する。途中の根も完了済みの探索は再利用する。ソース、根、エンジン、eval、予算、設定のハッシュが異なれば再利用を拒否する。コード変更後は別の `--work` を使う。sample は既存 `roots.json` を上書きしない。policy と report は自分の生成物を再生成できる。同じworkへの同時書き込みは避ける。

既定Nは100,000、予算は100,000 / 160,000 / 400,000 / 2,000,000 / 4,000,000。`--nodes` でNを変更すると比率を維持する。Threads=1、Hash=64MB、定跡なし、FV_SCALE=16などは既存Task 146の固定設定に準拠。エンジンとevalは指定された採用版を使い、実際のSHAをmanifestに保存する。`--engine`、`--eval-dir` で明示変更も可能。`eval_options.txt` による上書きは拒否する。

各予算・MultiPV2・各子局面は `usinewgame` + `isready` で独立に探索する。後述の「N＋追加3N」だけは、同じ根でNからTTを持ち越す。SFENから開始するため、元の対局の千日手履歴は復元しない。

## 標本と参照値

- 実際のTask 159の生対局は `~/binbo-shogi-data/work/task159/games/game_*.json`。指定文中の `.npz` は存在しない。Task 153の `task153-selfplay-v1` を読み、保存済みチェックサムも検証する。派生QSのnpzは使わない。明示的なSFEN/score/ply/game_id列を持つraw npzにも対応する。
- seed=20271670。ファイル順を固定してシャッフルし、条件を満たす最初の256局から、それぞれ適格行を一様に1行選ぶ。plyは指し終えた手数の20〜150（両端含む）、保存された**内部Value**の絶対値1500未満、非王手、合法手あり、入玉宣言可能局面を除外。ゲームIDは重複させず、元ファイルのパス・SHA・plyも保存する。
- 勝敗や今回の探索結果で標本を選ばない。今回の判断用に局単位で分離する意味でのholdoutであり、過去の全学習集合との非重複を新たに証明する処理ではない。
- 集計のcpは**USI表示cp**。Task 153の内部Valueと混同しない。`ref(m) = -score_cp(child(m), 20N)`、`ref(best)` は根20Nの最善手を同じ子探索で評価した値。N / 1.6N / 4N / 20N / 40N の最善手とpolicy上位2手の和集合を子探索20N/40Nで評価する。
- 主集計は根20N/40Nの最善手が一致し、根の値の変化が100cp以下、かつ詰み関連でない根。最善手不一致 **または** 値差100cp超を不安定とする。子探索の値差100cp以下も要求する感度分析を別に保存する。参照最善手の子より他候補が高くなることがあるため、損失を0に丸めず符号つきで残し、その件数も記録する。
- ノード制限では未完了の反復の `lowerbound` を持つ手が `bestmove` になることがある。値・depth・PVには最後の確定反復を使い、終了時info（bound込み）、実際のbestmove、両者の不一致フラグを保存する。boundsを確定値として読まない。nodesは受信infoの最大値を保存し、スコアを得た時点のnodesも別に残す。
- 詰みは符号を保って±3000cp（`mate -0` も負）。根予算・候補の子・MultiPVに詰み値があれば詰み群に分ける。合法手なしの子は手番側−3000、入玉宣言可能な子は＋3000として明示的に記録する。不安定・詰み・全件の統計もJSONに保存する。

## 2つの測定

Probe 1は、主集計のうちNの参照損失100cp以上が分母。その中で、policy上位2手の良い方がNの手から50cp以上回復する割合を計算する。対象0なら率はnullで、0%とは報告しない。

Probe 3は一様1.6Nの平均参照損失から、配分後の平均参照損失を引く。正が改善。

| JSONキー | 重点20%の選び方・探索 |
|---|---|
| `oracle_loss` | Nの参照損失が大きい順（依頼のoracle）。重点根は冷たい4N |
| `oracle_gain` | N→4Nの実現改善量順。この二択配分に対する本当のhindsight上限 |
| `cheap` | 小さいMultiPV2のNのgap、Nの反復間の最善手変更回数が多い、Nのdepthが浅い、の3つの経験midrankの和。重点根は冷たい4N |
| `cheap_free` | gapを除き、元の単一PVのNだけの2特徴で選ぶ。重点根は冷たい4N |
| `cheap_live` | 同じ2特徴で選び、Nの後にTTを保持して追加3N。重点根の選別費用込みで4N |

安いルールの重みは固定、参照値で調整しない。PV変更はPV先頭手の変更。gapは同じ完了depthの第1・第2PVの差、欠測時は0相当（件数を報告）。tieはroot_id順で決める。

冷たい4Nを使う数値は事後的な配分の比較であり、`cheap` のMultiPV2追加Nは診断費用として別勘定。Nで全根を選別した後に冷たい4Nを読み直す実装は平均**1.8N**になる。この混同を避けるため、追加MultiPVも再探索のNの二重払いも不要な **`cheap_live` を実行費用込みの同予算値**として併記する。continuationは最初のNを再現して一致を確認し、そのTTのまま `go nodes 3N` を実行する。元の冷たい4Nと同じ手になるとは仮定しない。この追加候補も参照子探索で採点する。

256や、安定性で絞った後の件数は5で割り切れない。20%枠の端数1局を確率的に重点化した**期待損失**として計算する。各再標本でも重点枠を再計算するので、要求ノードの期待総量は常に一様1.6Nと一致する。整数枠に丸めた結果を同予算とは呼ばない。実際のエンジンのnode overshootはログに残すが補正しない。policy・参照・子探索などの測定費用は配分予算に含まない。

CIはseed=20271673、既定2000回の局単位の対応つきpercentile bootstrap（95%）。各根1局なので行の再標本化が局の再標本化に一致する。各反復で選別順位・重点枠も再計算する。Probe 1の分母が0になる再標本は率を未定義とし、有効反復数をJSONに記載する。

## 旧WS-Tと推論環境

チェックポイント:

```text
~/task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt
sha256 20bf5e826e307cf191078f1cb86bf2c6f7bf79ef6e89bb2d40829844950fe2b3
```

Task 087のTransformerによるWS2。Task 091以降の自己対局更新版ではない。`m4_distill_train.py` の保存形式、`m2_usi.py` のarch dispatch / SFEN観測を確認し、`m2_distill_train.make_forward` → `network.TransformerNet` を直接使う。MCTSを挟まず、合法手内softmaxで正規化した確率の上位k（既定8）を保存する。チェックポイントのSHAはコードでも固定して誤用を拒否する。

環境は `~/task074_frameworks/venv_pgx/bin/python`。pgx実装は同venvの `lib/python3.12/site-packages/pgx/`、実験クローンの場所は `~/task074_frameworks/pgx/examples/alphazero/`。`PGX_HOME` または `--checkpoint` で場所を変更できるが、SHAは同じものを要求する。JAX・pgx・Haiku・cshogiのバージョン、実装ファイルのSHA、checkpoint config/epoch、デバイスを `policy.json` に保存する。

pgxの `from_sfen` はこの版で持駒の二桁個数を一桁ずつ上書きするため、そのケースだけ、確認済みの同じ `_from_board` APIで正しい持駒配列を渡す。全根でSFEN先頭3フィールドの往復一致と、**pgx/cshogiの全合法手集合の完全一致**を要求する。上位手だけの確認ではない。

`--device auto` はJAXの自動バックエンド選択でGPUを優先し、CUDAが使えなければCPUへ落ちる。GPUを見せる実行では `JAX_PLATFORMS=cpu` / `CUDA_VISIBLE_DEVICES=''` を残さない。CPU指定は `--device cpu`。CPU affinityは既定4コア以内（`--cpu-threads`）、BLASは1スレッドに制限する。searchの20プロセスとは別の設定である。

latencyは単局面・batch 8を既定100反復、各shapeのJITをwarmupした後、**SFEN parse・合法手mask/観測符号化・stack・GPU転送・forward・logitsのCPUへの同期転送まで**を時間に含む。top-k整列は除外。p50/p99はbatch全体のmsであり、batch 8を8で割った値ではない。8局面未満のsmokeでは入力を巡回して8件にする。

## スモーク検証

次の手順で4局面、N=2000、予算2000 / 3200 / 8000 / 40000 / 80000を実行。searchは2プロセス、同時に走らせるpolicyはCPU affinity 2、BLAS各1スレッドで、計4スレッド以内に制限した。

```bash
source scripts/env.sh
export PYTHONDONTWRITEBYTECODE=1
PROBE=experiments/alphazero/t167_probe_rescue.py
SMOKE="$DATA_ROOT/work/strategy/probe13/smoke_final"
"$VENV_PY" "$PROBE" sample --work "$SMOKE" --count 4
"$VENV_PY" "$PROBE" search --work "$SMOKE" --nodes 2000 --procs 2
"$HOME/task074_frameworks/venv_pgx/bin/python" "$PROBE" policy \
  --work "$SMOKE" --device cpu --cpu-threads 2 --latency-repeats 30
"$VENV_PY" "$PROBE" search --work "$SMOKE" --nodes 2000 --procs 2
"$VENV_PY" "$PROBE" report --work "$SMOKE" --bootstrap 1000
```

探索→policy→探索の順で、既存結果の再利用とpolicy候補の追加評価を検証する。さらにsearch再実行でJSONのSHA・更新時刻が変わらないことを確認する。小標本の数値は実行経路の確認用で、方向Aの採否には使わない。

最終版の結果は `~/binbo-shogi-data/work/strategy/probe13/smoke_final/{report.json,summary.md}`。

| 確認項目 | 結果 |
|---|---|
| 標本 | 異なる4局、適格局面を得るまで5ファイルを走査 |
| 根の安定性 | 安定3、不安定1、詰み関連0、gap欠測0 |
| Probe 1 | 主集計の損失100cp以上は0件、救済率はnull（算出不能） |
| Probe 3、一様の平均参照損失 | 53.33cp |
| 参照損失順 / 実現改善順oracleの削減 | ともに53.33cp、95% CI [0.00, 128.00] |
| cheap / cheap_free / cheap_live の削減 | ともに21.33cp、95% CI [0.00, 128.00] |
| 単局面CPU推論、30反復 | p50 9.334ms / p99 10.951ms |
| batch 8 CPU推論、30反復 | p50 63.408ms / p99 70.084ms（batch全体） |

4根の子候補数は2 / 3 / 2 / 3。各候補の40,000・80,000ノード評価、MultiPV2、N＋追加3N、1000回bootstrapまで完走。再実行で4つの探索JSONのSHA・mtimeが全て不変。`stat_checks.json` / `parser_checks.json` / `resume_checks.json` に追加確認の記録を置いた。

閾値100/50cpちょうどの包含、50cpに1cp足りない非救済例、損失削減の符号、期待ノード総量（1/4/5/6/256根）、同順位のmidrank、空の群、bootstrapを人工例でも照合した。別途、先後それぞれ持駒18歩のSFENをpolicyに通し、修正した二桁持駒の往復・合法手集合一致を確認した。GPU非公開のauto実行はCUDAプラグインの診断をstderrに出した後CPUにフォールバックして完走した。GPUの実推論はこのsandboxでは未確認。

USI parserは再現可能な応答列でも `mate +` / `mate -0` / `mate -` の符号、負cp、下界だけで最善手が変わる終了、実測nodes、同じdepthのMultiPV差を照合した。cshogiで合法手0と確認した子局面の−3000cp処理も通した。Python構文確認と `git diff --check` は成功。Rustのルール実装や既存コードには触れていない。
