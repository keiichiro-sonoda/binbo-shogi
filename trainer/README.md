# Trainer Data Pipeline

## 複数の resident チャンク (Task 188a)

GPU に全量が載らないときは、各々が単独で載る既存の multi-cache manifest を指定する。
位置引数は全 manifest 共通の val cache。manifest の `val` レコードはパスを解決した後で完全一致が必要。

```bash
source scripts/env.sh
"$VENV_PY" -m trainer.train_nnue "$DATA_ROOT/path/to/shared-val-cache" \
  --chunk-manifests "$DATA_ROOT/path/to/A.json,$DATA_ROOT/path/to/B.json" \
  --qat --qat-tie --factorize --fast --feature-dim 256 --hidden-dim 32 \
  --epochs 12 --batch-size 2048 --lr 6e-5 --seed 20260621 \
  --checkpoint-minutes 10 --resume --out "$MODELS_DIR/chunked_student"
```

- 毎 epoch **CLI の指定順 A → B → …**。各チャンク内は既存 `--manifest` と同じ RNG と keep mask の
  shuffle (mask がある場合、従来どおり全行 randperm も消費)。端数バッチはチャンクをまたがない。
- モデル、Adam の moments/step、scheduler、shuffle/mirror/dropout の RNG を継続する。
  val は最初の未学習時と各 epoch 末に各 1 回、scheduler も epoch 末に 1 回。
- 切替時に前の codes/kings/target/WDL を解放して `empty_cache()`。
  共通 val・モデル・Adam は常駐。`CUDA_CHUNK` と各 chunk の履歴に allocated/reserved のピーク bytes を記録
  (共通の常駐分やバッチの作業領域を含む、チャンク読込直前からの区間ピーク)。
- `--checkpoint-minutes` はバッチ途中でも保存する。これか `--resume` / `--stop-file` が有効なら
  全チャンク境界でも保存。中断した chunk の順序は RNG から復元し、既済みの損失は保存値を使う。
  `--stop-file` は複数チャンクでは毎バッチ確認。再開時はファイルを取り除く。
- `--max-steps` は全チャンク・全 epoch の optimizer 更新の通算。到達時はその部分 epoch を検証して終了。
  チャンク境界で中断して検証前に再開しても余分な更新を行わない。
- 複数チャンクの `history.json` は `chunks` に manifest の絶対パス・size・mtime_ns・SHA256、
  `history[].chunks` に 0 始まり chunk 番号・実際に訪問した rows・train_loss・GPU peak を記録。
  全体 train_loss は訪問行数による加重平均。resume 時は順序・引数・stamp の変化を拒否する。
- **1 チャンク指定は `--manifest X.json` の完全な別名**。従来の履歴との同値性を保つため、この場合だけ
  `chunks` キーを追加せず、リスト長 1 のパス/内容 stamp は既存の `run_identity.manifest` /
  `manifest_sha256`、チャンク損失は既存の `history[].train_loss` に対応する。
  実時刻の `train_secs` と出力先パスは実行ごとに変わる。
- `--manifest` / `--compact-cache` と排他。`--eval-npy` / `--limit-train` も既存 manifest と同じく不可。
  複数チャンクでは `--row-mask` / `--row-weight` は非対応。各 manifest の mask を使う。
  cross-chunk 重複除去や train/val 漏洩監査は入力 manifest を作る側の責任。

CPU 回帰: `"$VENV_PY" -B -m unittest trainer.test_chunked_cache -v` (2 threads)。

## KP 特徴の付番

**正本は [`../docs/kp_spec.md`](../docs/kp_spec.md)** にあります。付番式・次元の内訳・座標系・
駒種ID(金の位置が盤と持ち駒で違う罠を含む)・「77特徴」の由来・検算例・正準 HalfKP との
差異・検証経緯は、すべてそちらに記載しています。重複(=ドリフト)を避けるため、ここでは
要点とこのパイプライン固有の事項のみを記します。

要点:

- 入力は手番側視点に正規化した 2視点 HalfKP 系(view 0=手番側玉基準、view 1=相手玉基準)。
- `TOTAL_FEATURES = 353486`、1局面で立つ特徴は常に 77。
- 実装: 正解基準は `kp.py`(Python)、推論側 `src/eval/kp.rs`(Rust)が完全一致(Task 018)。

このパイプライン固有の事項:

- このディレクトリは Python/PyTorch 側の学習パイプライン用。**保存正本は `hcpe`** であり、
  KP インデックス列は読み込み時に作る派生物として扱う(`DESIGN.md`)。
- 評価値 `eval` は hcpe の値をそのまま手番側視点として扱う。

## 最小 NNUE スモーク

Task 014 の目的は、本格学習ではなく「KP 入力から勝率ラベルへの学習ループが回る」ことの確認です。

実行環境は Python + `cshogi` + `numpy` + CPU版 `torch` です。GPU はまだ使いません。

### 評価値から勝率への変換

hcpe の `eval` は centipawn、手番側視点として扱います。初期の変換は次のシグモイドに固定します。

```text
winrate = sigmoid(eval_cp / 600)
```

`600 cp` は暫定値です。`eval=0` が `0.5`、`+600` が約 `0.731`、`-600` が約 `0.269` になり、スモーク用の正負ラベルが飽和しすぎないため採用します。最適化は後段で行います。

### モデル構造

最小モデルは `trainer/nnue.py` の `MinimalNNUE` です。

```text
KP active indices
  -> EmbeddingBag(TOTAL_FEATURES, feature_dim=32, sum)
  -> Linear(32, 16)
  -> ReLU
  -> Linear(16, 16)
  -> ReLU
  -> Linear(16, 1)
  -> sigmoid(logit)
```

`EmbeddingBag` は、立っている KP 特徴の行だけを足し合わせます。巨大な one-hot ベクトルは作らず、NNUE の特徴変換層を小さく実装した形です。

### 損失と optimizer

- 損失: `BCEWithLogitsLoss`
  - 勝率の soft label を logit に直接当てる標準的で数値安定な損失として採用。
- optimizer: `Adam`
  - 最小リハーサルで学習配管を安定して通すための標準選択。
- ゴール: loss が有限、逆伝播で重みが変わる、数ステップ NaN/Inf なく回ること。

### 再現手順

少量の多様局面を既存 Rust コマンドで作り、Python で eval だけを既知のばらつき値に差し替えます。

```bash
cargo run --release --locked -- hcpe-selfplay /tmp/binbo-task014/base.hcpe 4 64 20260621 1
python -m trainer.make_smoke_hcpe /tmp/binbo-task014/base.hcpe /tmp/binbo-task014/smoke_eval.hcpe --count 96
python -m trainer.train_smoke /tmp/binbo-task014/smoke_eval.hcpe --steps 20 --batch-size 16
```

この手順は、評価値が本物かどうかではなく、`hcpe -> KP -> NNUE -> loss -> backward -> optimizer.step` の配管を確認するためのものです。
