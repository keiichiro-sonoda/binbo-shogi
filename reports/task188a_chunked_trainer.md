# Task 188a: resident チャンクを切り替える学習器 (2026-10-24)

`trainer/train_nnue.py --chunk-manifests A.json,B.json,...` を追加した。
毎 epoch 指定順に学習し、各チャンク内は既存 manifest の shuffle と keep mask を使う。
モデル、Adam の moments/step、shuffle/mirror/dropout の乱数、epoch、scheduler は 1 run で継続する。
共通 val レコードは完全一致を要求し、未学習時と epoch 末に各 1 回評価する。
使い方は `trainer/README.md`。コミットは作成していない。

## 保存・履歴

- `--manifest` / `--compact-cache` と排他。既存 manifest と同様、label swap と limit-train は不可。
  複数チャンクでは外付け row-mask / row-weight は不可。mask は各 manifest に持たせる。
- CUDA の前チャンクの codes/kings/target/WDL を解放し、`empty_cache()` 後に次をロードする。
  val・モデル・optimizer は継続。チャンクごとに peak を reset し、allocated/reserved bytes を
  `CUDA_CHUNK` と `history[].chunks` に記録する (共通常駐分・バッチの作業領域も含む)。
- checkpoint 有効時は全チャンク境界で保存。時間間隔によるバッチ途中保存も対応する。
  chunk index、行 cursor、損失、既済みチャンク、Adam、scheduler、各 RNG を保存し、
  shuffle は保存した Generator 状態から再生成する。`--stop-file` は複数チャンクでは毎バッチ確認。
- `--max-steps` は全チャンク・epoch を通算。上限到達時の部分 epoch も 1 回検証して終了し、
  検証前の中断から再開しても余分な更新をしない。
- 複数チャンクの `history.json` に順序付き paths + stamps (size/mtime_ns/SHA256)、
  各 epoch 各 chunk の訪問行数・平均損失を保存する。全体損失は訪問行数による加重平均。
  再開時は引数・順序・manifest stamps の変更を拒否する。
- **1 チャンクは既存 `--manifest X.json` に正規化する**。要求された履歴の同値性と既存形式を守るため、
  この場合は追加の `chunks` キーを作らない。単一パス/内容 stamp は既存の
  `run_identity.manifest` / `manifest_sha256`、チャンク損失は `history[].train_loss` に対応する。

## 同値性・回帰の証拠

PyTorch 2.12.1+cu130、CPU 2 threads 以下。次の **23 tests が成功**した。

```bash
source scripts/env.sh
"$VENV_PY" -B -m unittest trainer.test_chunked_cache -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$VENV_PY" -B experiments/alphazero/t172_test.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  "$VENV_PY" -B experiments/alphazero/t135_fast_equiv.py --cpu-checks
```

- 新規 9 件。変更前 HEAD の plain / compact / manifest と現行を直接比較し、
  `nnue_final.pt` / `nnue_best.pt` / model_config / history がバイト一致。
  1 チャンク指定も manifest とバイト一致 (all-true / masked、fast 有無、mirror、dropout、cosine、更新上限)。
  history 比較時は出力先を同一にし、実時刻を固定した。
- 2 チャンク・2 epoch、mask の端数バッチを含む fixture で、独立に作った randperm/mirror RNG の列と一致。
  val は初回 + epoch 末だけ。Adam の各 step は通算 10。target 変換 7 レシピも既存経路と完全一致。
- チャンク境界、次 epoch 境界、バッチ途中、上限到達後の検証直前で強制中断して再開。
  dropout + mirror + fused Adam + cosine を含め、final/best のバイト列、Adam 状態、Generator、
  数値履歴が連続実行と一致 (履歴の実測 train_secs は除外)。1 チャンクの dropout 再開も一致。
- 既存 Task 172 の 9 件、Task 135 の CPU 5 件も成功。Rust のルール実装は変更していない。

## メモリと限界

CPU fixture で実 tensor payload を計測: resident train (kings/codes/half) は
**A 1,708 bytes → B 1,396 bytes**、各 epoch で同じ。
weakref で前の cache 本体・codes・kings・target が次のロード前に消滅することを確認した。
GPU branch の empty/reset/load/peak/release の呼出順も CPU 上の mock で検証した。

実機 GPU は `nvidia-smi` がドライバと通信できず、`torch.cuda.is_available()` も False。
**GPU smoke・GPU peak 実測・128M 本番学習は未実施**。空きを待つ処理はしていない。
容量の算術目安は、escape のない codes+kings が 156 bytes/物理行なので、
128M 一括は 19.968 GB、64M を 2 チャンクならその部分は同時最大 9.984 GB。
これに共通 val、target、モデル、Adam、勾配、バッチの作業領域が加わる。実測 VRAM 値とは区別する。

各チャンク + 共通 val + 学習状態は単独で GPU に収まる必要がある。
入替ごとにディスクから読み、GPU へ転送するため、その時間は増える。
チャンク間をまたぐ全体 shuffle や端数バッチの連結は行わない。
重複除去・train/val 漏洩監査は manifest 作成側で行う。
再開一致を実証したのは CPU fixture。同じ CUDA 実行の数値決定性や GPU 上の再開一致は未検証。

## Claude の追加確認 (10/24 23:55, 14700KF の GPU)
- CPU の試験 (新規 9 件, Task 172 の 9 件, Task 135 の 5 件) をこちらでも実行して全部通った。
- GPU で 2 チャンク (Task 181 の M.json 1,707 万行 → Task 182 の MR.json 2,735 万行) をまたいで 19,000 歩: 載せ替えが動き, 割り当て 3.72GB → 5.36GB (1 行 約 160 バイト)。
  6,395 万行のチャンクは約 11GB の見込みで 16GB に収まる。
