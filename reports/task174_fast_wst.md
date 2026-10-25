# Task 174: WS-T2 の学習を測ってから速くする

2026-10-23。出力はすべて `~/binbo-shogi-data/work/task174_fast_wst/` 配下。
既存リポジトリ、Task 173 のデータ・重み・STATUS は変更せず、コミットしていない。
納品コードは [t174_fast_wst.py](../experiments/alphazero/t174_fast_wst.py)。

## 結論

**入力生成は約22倍に短縮できた。学習全体の10倍は達成・実測していない。**
batch 256 の SFEN→観測・合法手マスク生成は旧版 **96.55 ms**、新しい圧縮入力生成・CPUデバイス転送は
**4.43 ms**。新しい入力は約57,819局面/秒。観測、合法手マスク、圧縮復元、教師値は
学習データから無作為に選んだ **4,096局面**でビット一致し、二桁持駒・先後・王手の追加3局面も一致した。

GPU はこの環境に存在しない。旧GPUログと今回のCPU分解測定を組み合わせると、同じ FP32 ネット計算を維持した
新経路は条件付きで **約2,826〜2,972局面/秒**と見積もる。旧較正値807局面/秒の約3.5〜3.7倍、
旧実学習の定常値1,400局面/秒の約2.0〜2.1倍。**GPUで測った新速度ではない。**
GPUで次に測るべき候補として、任意の bf16 計算と勾配蓄積を実装した。

さらに、旧 `throughput.json` の807局面/秒には初回の32.4M行の乱数順列生成が混入していた。
これを「GPUの定常性能」と解釈して長時間学習の量を決めるのも無駄の原因だった。

## 測定の手順と環境

最初に `AGENTS.md`、`STATUS.md`、`DESIGN.md`、`t173_wst2.py`、`network.py`、
`m2_distill_train.py`、Task 173 レポートを読み、venv内の pgx の `_from_board`、`_observe`、
`_legal_action_mask` と cshogi の Cython/Python インターフェースを確認した。
最適化前の独立プロファイルは [profile_first.py](../measurements/profile_first.py)、
[profile_first.json](../measurements/profile_first.json)。batch 8/32/64/128/256 を順に測定した。

Python 3.12、JAX/jaxlib 0.10.2、pgx 2.6.0、dm-haiku 0.0.16、optax 0.2.8、
cshogi 1.0.2、NumPy 2.5.0、jmp 0.0.4。CPU affinity は4論理CPU、
OMP/OpenBLAS は1スレッドに制限。入力生成の先読みはワーカー1本で、同じ affinity を継承する。
背景では別の作業の時間つき対局が動いている。負荷・ページキャッシュを排他的には制御していない。

主測定は [bench256_final/bench.json](../measurements/bench256_final/bench.json)、
ログは [bench256_final.log](../measurements/bench256_final.log)。
各stageは初回をウォームアップとして除外し、その後5回の中央値を使用。
各stage末尾でデバイスを同期する。パイプラインだけは5stepを連続投入し最後に同期する。
測定時のソースは [measured_trainer.py](../measurements/bench256_final/measured_trainer.py) に固定保存し、
`bench.json` の `run_meta.script_sha256` と一致する。納品版の `bench` も実行版を自動保存する。
benchは6.87 GB全体のSHA読出しをstage計測に混ぜず、metaとファイルサイズで読み込むことをJSONに明記する。
train / encode / evalは元データのSHA・ゲーム分割検査を省略しない。

## CPU 分解測定

以下はすべて batch 256。転送の数値は **host→CPUデバイス**であり、H2Dの実測ではない。
`--device gpu bench` では同じ測定点で実際のH2Dを測る。

|stage|ms / batch|局面/秒|
|---|---:|---:|
|同じ行の再読み出し|0.0041|62.4M|
|異なる無作為行の読み出し、後述の温まった状態|0.403|634,925|
|SFENのASCIIデコード|0.111|2.30M|
|旧Python SFENパースとstack|4.155|61,609|
|先後の正規化|0.084|3.07M|
|pgx合法手マスク|68.493|3,738|
|pgx観測|23.812|10,751|
|旧パース＋合法手＋観測の融合経路|96.554|2,651|
|新C++観測＋合法手|3.823|66,968|
|新C++生成＋714-byte圧縮|4.122|62,109|
|圧縮したhostバッファの準備|0.011|22.8M|
|圧縮入力のhost→CPU転送|0.148|1.73M|
|旧bool入力のhost→CPU転送|0.101|2.52M|
|圧縮入力のデバイス上の復元|1.293|197,925|
|新圧縮入力の生成＋host→CPU転送|4.428|57,819|
|小規模memmapキャッシュの読み出し＋転送|0.184|1.39M|
|forward/backward、圧縮復元を含む|2,641.045|96.93|
|Adam、勾配は既にデバイスにある|3.936|65,043|
|融合した学習step、入力はデバイスにある|2,657.191|96.34|
|旧入力＋学習、同期あり|2,731.681|93.72|
|新入力＋学習、同期あり|2,641.684|96.91|
|新先読み＋donation、異なる行の連続5step|2,665.587|96.04|

独立stageの合計は融合したstepと同じとは限らない。CPU学習はforward/backwardが大半を占め、
入力の約92 msを削っても全体は約3%の短縮にとどまる。CPUのforward/backwardをGPU時間として換算していない。

読み出しには大きな温度差があった。先行測定の `bench256/bench.json` では同じ無作為行の変更パターンが
**89.70 ms/batch**、上表の再測定では **0.403 ms**。
OSキャッシュを落とす操作はしていない。「読込0.004 ms」や小さな圧縮キャッシュの数字を
32M行の未読ランダムアクセスに外挿しない。90 msの読込待ちが毎batch残れば、先読みしても
CPU入力側だけで約2,700局面/秒が上限になる。キャッシュを増やすことが必ず速くなるとは限らない。

## 何を変えたか

### 1. 学習stepからpgxのルール再計算を外した

pgx `_observe` は駒のone-hotだけでなく、各駒の利き、利きの枚数、持駒、王手を作る。
合法手生成も81升×方向に対する大量の配列演算を使う。毎batchこれをCPUで再計算する必要はなかった。

venvに既にある `cshogi.dlshogi.make_input_features(board, float32[62,9,9], float32[57,9,9])`
が同じ特徴をC++で作れる。空バッファへ毎回ゼロ初期化し、62空間面＋57定数面を
pgxと同じ `rot90(k=3)` に並べる。合法手は cshogi の `legal_moves` と `make_move_label` で作る。
pgx のaction IDとの一致は、既存 `m2_moves.usi_to_action` とも照合する。
駒・利き・持駒の意味や、SFENに履歴が無いときの扱いを変えていない。

### 2. 714 bytes/局面を転送し、jit内で観測を復元する

|保存フィールド|bytes|内容|
|---|---:|---|
|pieces|81|空=0、自駒1〜14、相手駒15〜28。28個のone-hot面を代替|
|effects|345|利き・利き枚数の34面×81bitをpackbits|
|scalars|8|持駒・王手の57bit|
|legal|274|2,187 actionsのbitmask|
|action / value|2 / 4|元データのuint16 / float32をそのまま保存|
|合計|**714**|bit順は上位bitから|

actionはデバイスに渡すときint32にするので転送量は実際には716 bytes/局面。
旧入力はbool観測9,639＋boolマスク2,187＋int32 action 4＋float32 value 4 = **11,834 bytes/局面**。
転送量を約16.5分の1にする。観測への復元は整数・bool演算のみで、浮動小数点の近似はない。
これは利きをGPUで再計算する案より、GPUでの81×81の一時配列を減らせる。

オンライン生成を既定にした。約4 ms/batchなら、FP32の推定GPU計算と十分重なる。
事前キャッシュが必要なら `encode` を使える。
全train＋val 33,046,893局面なら **23,595,481,602 bytes = 23.60 GB / 21.98 GiB**。
元SFENデータ6.87 GBに追加で必要になる。単純な119面＋合法手のpackbits約49 GBを避けた。
`encode` は元データを完全検査し、ファイルSHAとデータIDを保存、最後のmeta.jsonを完了印とする。
中途半端なキャッシュや `--limit` 付き測定キャッシュは学習で拒否する。

全量データのSHA・ゲーム分割を検査したうえで、train/valの先頭各32,768局面を実際に変換して保存した。
[cache_prefix32768/meta.json](../measurements/cache_prefix32768/meta.json) に記録がある。
読込・C++符号化・圧縮・書込・fsyncの時間はtrain **0.588秒（55,754局面/秒）**、
val **0.617秒（53,092局面/秒）**。起動時の全データ検査、サンプル照合、出力の最終SHA計算はこの区間に含まない。
この速度なら全量の変換区間は約10分だが、23.6 GBを持続して書く場合の速度は未測定。
全量変換は行っておらず、既定のオンライン経路ならこの前処理時間と追加容量を使わず開始できる。

### 3. 先読み、同期の削減、donation、再コンパイルの削減

先読みは1ワーカー、既定深さ2。t173の同じseed・epoch順列・各batch内ソート・末尾折返しを使い、
取り出す行と順序は変えない。GPUに次の圧縮入力を渡している間に現在のstepを実行する。
lossの有限性はデバイス上で蓄積し、ログ・保存時に確認する。毎stepの `np.asarray(loss)` による同期を外した。
modelとAdamバッファをdonateし、較正の更新には基準重みのコピーを使う。

Adamのcountだけが初期状態ではuncommitted、初回更新後はcommittedとなり、
それだけで2回目のjitコンパイルが起きていた。初期Adam状態全体をデバイスに明示配置した。
調査では batch 8 の初回1.50秒、2回目1.29秒、3回目0.088秒だったものが、
明示配置後は初回1.42秒、以後0.087秒、コンパイル済みsignatureは1個になった。
新規学習では較正と本学習で同じjit関数も再利用する。

自動batch選択は既知の32/64/128/256のうち**実測の局面/秒が最大**のものを選ぶ。
旧版の「入ったうちで最大batch」から変えた。同じ更新列を比較したい場合は `--batch 256` を指定する。
512以上は、既知のFP32 OOMを自動で繰り返さず、明示的な `--batch` で試す。

## GPU速度の推定と10倍に届かない理由

元の `train/throughput.json` は32/64/128/256でそれぞれ150.63/165.71/220.40/317.38 ms/step。
`benchmark()` はタイマー開始後に最初の `Sampler.indices(0)` を呼ぶので、
全32.4M行の順列生成が5stepに一度だけ入る。今回この初期化は **0.682秒**、5分割で **136.48 ms/step**。
317.38−136.48=180.90 msは、実学習ログとよく整合する。
このCPU値が過去のGPU実行時にも同じだったと断定はしない。

実学習 `train/history.jsonl` の隣接ログ間隔をstep差で割った中央値は **182.815 ms/step = 1,400.32局面/秒**。
最後の30,203stepは5,568.61秒、7,731,968局面（0.239 epoch）。
約3時間という設定から較正で固定されたstep数であり、ログ上の学習ループ実時間は約1.55時間だった。
較正が遅く見えたため、時間枠を十分使う前に予定stepに到達している。

温まった読込を仮定すると、今回の旧CPU入力96.67 msを引いた残りは **86.15 ms/step**。
この残りはGPUのforward/backward/Adamに加え、H2D・同期やCPU/GPUの実行条件差も含む。

|同じFP32ネット時間を保つ条件付き推定|局面/秒|
|---|---:|
|新入力4.43 msを直列に足す|2,826|
|新入力がGPUと完全に重なる|2,972|
|全32.4M trainを1周する時間|約3.0〜3.2時間|

**FP32の残り約86 msが保たれる限り、入力をゼロコストにしても旧定常速度の約2.12倍、
旧較正値の約3.68倍まで。** 807局面/秒の10倍にはstep全体を **31.74 ms以下**、
真の定常速度の10倍なら **18.28 ms以下**にする必要がある。
入力を速くするだけでは届かず、ネット計算自体をそれぞれ約2.7倍／4.7倍以上短縮する必要がある。
これは別実行間の引き算に基づく条件付きの限界であり、GPUハードウェア全体の絶対限界ではない。
新 `bench` は元の較正値、実学習中央値、初期順列、読込条件を分けてJSONに出す。

## 数値が変わる部分

- **既定 FP32、accumulation=1**: 観測・マスク・教師値、lossの式、Adam、lr schedule、
  同じbatch指定時の抽出列は同じ。CPUの同じ8局面・同じ重みから1更新したモデル全leaf、Adam全leaf、lossは
  旧版・新版・donation版の間でビット一致。別デバイス・別XLA最適化間の重みのビット一致までは保証しない。
- **`--compute bf16`**: パラメータとAdamはFP32、Transformerの線形層・attention・残差などをbf16にする。
  LayerNorm計算とlossはFP32、LayerNorm出力などはbf16へ丸める。マスク・教師値は変更しない。
  attention softmaxを含む丸めが変わり、同じ学習結果ではない。CPUの8局面の初回lossは
  FP32 **3.35017943**、bf16 **3.34061456**。有限性と保存パラメータFP32は確認した。GPUの速度・収束・棋力は未確認。
- **`--accumulation K`**: `--batch`は実効batch。その中をK個の同サイズmicrobatchに分け、
  勾配を平均してAdamを1回更新する。statelessなTransformerだけを許す。
  同じ実効batchなら目的関数は同じだが加算順序が変わる。
  batch 8を2分割した初回更新の最大パラメータ差は **9.8348e-6**。
  Adamは微小勾配に敏感なので、更新までビット一致とは言わない。
  実効batch自体を256から512に増やせば、Adam更新回数・scheduleあたりの局面数も変わる。
- **eval**: bf16学習した重みも含め、既定で既存ネットのFP32推論を行う。
  元t173の評価・除外監査・関門0集計を再利用する。

## 確認したこと

[test_t174_fast_wst.py](../tests/test_t174_fast_wst.py) の3統合試験は成功。
最終ログは [cpu_release.log](../tests/cpu_release.log)。

1. 旧版／新入力／donationの更新全leaf一致。Adamを明示配置すると3更新でjit signatureが1個。
2. microbatch蓄積と直接更新のloss一致許容差、実更新の差分計測。bf16の有限lossとFP32パラメータ。
3. 実ゲーム由来の既存288局面fixtureでキャッシュ生成、全ファイルSHA検査、
   キャッシュの破損・部分データの拒否、オンライン4更新とキャッシュ2更新＋再開2更新のビット一致。
   t173と同じeval引数で16プローブ＋検証12局面も実行し、subset表示を確認。

主benchの4,096局面は全32.4M trainから無作為抽出した。
別に [bench_accumulation/bench.json](../measurements/bench_accumulation/bench.json) で
batch 8 / accumulation 2 の分解測定とパイプラインも完走。
大きな実効batchのbenchでは、分解するforward/backwardもmicrobatchサイズで測り、
全batchを単独F/Bに渡してメモリ節約を打ち消さないようにした。
本学習、GPUのOOM境界、全量キャッシュ、新しい重みの棋力判定は実行していない。

## 実行方法

リポジトリ直下から実行。出力先は毎回新しいディレクトリを指定する。

```bash
REPO="$PWD"
source "$REPO/scripts/env.sh"
export DATA_ROOT PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
TASK174_WORK="$DATA_ROOT/work/task174_fast_wst"
PGX_PY="${PGX_PY:-$HOME/task074_frameworks/venv_pgx/bin/python}"
TOOLS="$TASK174_WORK/experiments/alphazero/t174_fast_wst.py"
DATA="$DATA_ROOT/work/task173_wst2/dataset"

# CPUでstage分解とビット一致を再現する
"$PGX_PY" "$TOOLS" --repo "$REPO" --device cpu bench \
  --data "$DATA" --batch 256 --bench-steps 5 --check-sample 4096 \
  --out "$TASK174_WORK/bench_cpu_new"

# GPU実測。まずFP32を基準にし、その後bf16との差を測る
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu bench \
  --data "$DATA" --batch 256 --bench-steps 30 \
  --out "$TASK174_WORK/bench_gpu_fp32_new"
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu bench \
  --data "$DATA" --batch 256 --compute bf16 --bench-steps 30 \
  --out "$TASK174_WORK/bench_gpu_bf16_new"

# 任意: batch 512をmicrobatch 256×2で測る。実効batchを増やす実験
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu bench \
  --data "$DATA" --batch 512 --accumulation 2 --bench-steps 30 \
  --out "$TASK174_WORK/bench_gpu_acc2_new"

# 通常はオフライン変換なしで学習できる。既存t173と同じ初期WS-TをSHAで照合
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu train \
  --data "$DATA" --batch 256 --hours 3 --bench-steps 30 \
  --out "$TASK174_WORK/train_fp32_new"

# キャッシュを使う場合のみ。まず短い実測。--limit版は学習には使えない
"$PGX_PY" "$TOOLS" --repo "$REPO" --device cpu encode \
  --data "$DATA" --limit 32768 --out "$TASK174_WORK/cache_trial_new"
# 全量は --limit を省略し、新しい --out を指定。容量と実測所要時間を先に確認する
# train に --cache "$TASK174_WORK/cache_full_new" を追加すれば読み込める

# 再開はTask 174のチェックポイント専用。保存済みschedule/batch/seed/精度を復元
LATEST=$("$PGX_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
  "$TASK174_WORK/train_fp32_new/latest.json")
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu train \
  --data "$DATA" --out "$TASK174_WORK/train_fp32_new" \
  --resume "$TASK174_WORK/train_fp32_new/$LATEST"

# evalは旧Task 173の候補ckptも読める。既定で全プローブ・全検証ゲームを評価
"$PGX_PY" "$TOOLS" --repo "$REPO" --device gpu eval \
  --data "$DATA" --checkpoint "$TASK174_WORK/train_fp32_new/$LATEST" \
  --out "$TASK174_WORK/eval_new"

BINBO_REPO="$REPO" "$PGX_PY" "$TASK174_WORK/tests/test_t174_fast_wst.py"
```

`train` の `--steps` / `--epochs`、`--warmup`、`--seed`、`--lr`、`--stop-after`、
ログ・保存間隔、および `eval` のprobes・bootstrap・subset関連引数はt173と同じ。
再開時はデータID、トレーナー／依存ソースのSHA、ライブラリ版、backendを検査する。
既存のTask 173 optimizer checkpointからの直接resumeは追加していない。
元のcheckpointを書き換えることなく評価でき、Task 174の新規学習は識別済みWS-Tから開始する。

GPUの30step benchで入力・GPU計算・転送の時間を確認してから長時間学習の精度とbatchを選ぶ。
bf16の実験では、同じデータ量でFP32と検証損失・既存の棋力判定を比較してから採用する。
無人の長時間実行の前には、従来の運用規約どおり作者側でスリープ無効化を確認する。

## GPU での実測 (Claude, 10/23 夜, 判定の対局と同時に動かした状態)

| 経路 | batch | 局面/秒 (学習全体, 毎回違う行を読む) |
|---|---|---|
| 旧 t173 (実測) | 256 | 1,377 |
| 新 t174, FP32 | 256 | 2,862 |
| 新 t174, FP32 | 512 | 2,748 |
| **新 t174, bf16** | **256** | **4,763** |
| 新 t174, bf16 | 512 | 4,412 |

- 入力はもう律速ではなく, GPU の順伝播・逆伝播が律速 (FP32 で 2,880 局面/秒, bf16 で 4,775 局面/秒)。
- bf16 なら旧版の約 3.5 倍。3,238 万局面の 1 周が約 1.9 時間。bf16 は計算の数値が変わるので, 使うなら FP32 の版と同じ評価 (1,184 局面・検証用の対局) で差がないかを先に確かめる。
- 記録 `~/binbo-shogi-data/work/task174_fast_wst/gpu_bench/`。
