# Task 017: 100万局面で NNUE 本格学習 → 検証ロスで汎化を確認

## 結論

- **検証ロスは下がった = 技巧2の評価を未知対局で汎化できた**。標準設定(wd=0)で 0.6980(未学習)/
  0.6931(定数予測器)→ **0.6330(epoch1)**、予測勝率と目標勝率の相関 val_corr ≈ 0.61。
- 標準設定は **epoch1 以降すぐ過学習**(val 上昇・train 下降、乖離 +0.08→+0.16)。観測した過学習に対し
  weight decay を 1 点だけ probe した結果、**wd=1e-3 で過学習が大きく緩和し best val=0.6182(epoch9)、
  乖離 +0.08、val_corr 0.64** まで改善。wd を強めるほど一貫して改善(下表)。
- **1 エポック = 約 24 秒(CPU)**。15 エポックで約 6 分。**GPU は現規模では不要**。
- 次段(Rust 推論で対局)に渡す重みは **wd=1e-3 の best checkpoint(epoch9, val 0.6182)**。

## データと分割

- 入力: Task 016 の100万局面(技巧2 depth-8 `score cp`)、`/tmp/binbo-task016/run1m/`(100 shard)。
- **分割は shard 単位**(`trainer/prepare_data.py`): 末尾 5 shard(index 95–99 = 5万局面)を検証用に
  held-out、残り 95 shard(95万局面)を訓練。**同一対局が train/val に跨らない**ので、検証ロスは
  「未知の対局」での汎化を測る(局面単位分割だと同一対局の隣接局面が train/val に分かれて相関リーク
  するため避けた)。再現可能(shard index で決定的)。
- KP 変換は既存 `trainer/kp.py`(検証済み)。全局面が**ちょうど 77 特徴**(駒数保存の不変量)であることを
  prepare_data が assert。indices を `(N, 77)` int32 の密行列でキャッシュ(.npy)。
  - KP 変換実測: 100万局面を **39.3 秒(25,471 局面/秒)**で変換し .npy にキャッシュ(以後の学習は再利用)。
- ラベル: `winrate = sigmoid(eval_cp / 600)`(Task 014 の規約)。手番側視点で統一(既存パイプライン準拠)。

## モデルと学習設定

- モデル: Task 014 の `MinimalNNUE`(構造不変):
  `EmbeddingBag(TOTAL_FEATURES=353486, dim=32, sum) -> Linear(32,16) -> ReLU -> Linear(16,16) -> ReLU -> Linear(16,1)`。
- 損失: `BCEWithLogitsLoss`(勝率 soft label を logit に当てる)。optimizer: `Adam(lr=1e-3)`。
- バッチ: 1024。エポック: 15。seed 固定。CPU(torch 2.12.1+cpu, 14 threads)。
- 実装: `trainer/train_nnue.py`。エポックごとに train_loss / val_loss、val 勝率 MAE、
  予測勝率と目標勝率の Pearson 相関を記録。**定数予測器(val 目標平均)の val ロスをベースライン**として併記。

## 結果(標準設定: lr=1e-3, batch=1024, weight_decay=0, 15 epochs)

- ベースライン(定数予測器=val 目標平均 0.5036): val_loss = **0.6931**。未学習モデル: val_loss = 0.6980。

| epoch | train_loss | val_loss | gap(val−train) | val_mae | val_corr |
|---|---|---|---|---|---|
| 1 | 0.5533 | **0.6330** ← 最良 | +0.080 | 0.1634 | 0.612 |
| 2 | 0.5134 | 0.6350 | +0.122 | 0.1616 | 0.611 |
| 3 | 0.5054 | 0.6348 | +0.129 | 0.1615 | 0.617 |
| 5 | 0.4995 | 0.6391 | +0.140 | 0.1629 | 0.607 |
| 10 | 0.4945 | 0.6432 | +0.149 | 0.1640 | 0.601 |
| 15 | 0.4923 | 0.6511 | +0.159 | 0.1663 | 0.590 |

- **検証ロスは下がった**: 0.6980(未学習)/0.6931(定数)→ **0.6330(epoch1)**。下げ幅 0.065。
  val_corr ≈ 0.61(予測勝率と技巧2由来の目標勝率の Pearson 相関)→ **未知の対局でも技巧2の評価を
  有意に真似られている**(汎化している)。
- **ただし epoch1 以降は過学習**: train_loss は単調減少(0.553→0.492)だが val_loss は epoch1 を底に
  上昇(0.633→0.651)。train/val の乖離が +0.08 → +0.16 に拡大。**最良 val は epoch1**。
  → この設定では **early stopping(epoch1 相当)が最良汎化**で、それ以上回すと汎化が悪化する。

## 実時間と GPU 要否

- **1 エポック = 約 24.1 秒(CPU, torch 2.12.1+cpu, 14 threads)**。15 エポックで約 6 分。
- 内訳: train 95万局面の 1 パス + val 5万局面の評価。EmbeddingBag(密勾配)+ Adam でも CPU で軽い。
- KP 変換は事前 1 回 39.3 秒(.npy キャッシュ、学習側は再利用)。
- **GPU は現規模では不要**。1 エポック 24 秒なので、エポック数を増やしても・軽い HP 探索をしても
  CPU で現実的(例: 100 エポックでも約 40 分)。モデル(11.3M params, 主に embedding)もデータ
  (KP indices 密行列 ~310MB)も小さい。将来モデル/データを大幅に拡大したら再評価する。

## 所見

- **判定**: 検証ロスは下がった(DoD 達成)。生徒 NNUE は技巧2 depth-8 の評価を未知局面で
  有意に汎化(val_corr 0.61, val_mae 0.166 = 勝率誤差約 16.6%)。一方で**速やかに過学習**する。
- **過学習の原因(推定)**: モデル容量(embedding 353,486×32 ≈ 11.3M params)が信号に対して大きく、
  95万例でも 1 パスで train を memorize しがち。最小ネットワーク部(32→16→16→1)は小さいので、
  過学習は主に巨大な特徴埋め込みに由来すると見られる。
- **対策 probe(weight decay)**: 過学習を観測したので Adam の L2 weight decay を 2 点だけ試した
  (15 epochs each, 他は標準のまま)。**weight decay を強めるほど一貫して best val が下がり、過学習が
  後退し、相関が上がった**:

  | 設定 | best val | @epoch | final val | final val_corr | 最終乖離(val−train) |
  |---|---|---|---|---|---|
  | wd=0(標準) | 0.6330 | 1 | 0.6511 | 0.590 | +0.159 |
  | wd=1e-4 | 0.6279 | 6 | 0.6340 | 0.628 | +0.118 |
  | **wd=1e-3** | **0.6182** | 9 | 0.6255 | 0.637 | +0.079 |

  → 過学習は「巨大な特徴埋め込みの容量過多」が主因という推定と整合(弱い正則化で大きく改善)。
  best val は wd=1e-3 が最良。さらに強い wd や容量削減・データ増で改善余地はあるが、
  HP 探索はスコープ外なのでここで止める(次段で強さ測定と合わせて詰める)。
- **次段への申し送り**:
  - 推論に載せる重みは **wd=1e-3 の最良 val checkpoint(`model_wd1e-3/nnue_best.pt`, epoch9, val 0.6182,
    val_corr 0.637)**を使う(標準 wd=0 の epoch1 より汎化が良い)。`model_config.json` に層次元・scale を保存済み。
  - さらなる改善候補: より強い weight decay / dropout / 容量削減(feature_dim 縮小)/ データ増
    (Task 016 基盤で 500万〜へ拡大可)。今回はスコープ外(構造探索・大規模 HP 探索はしない)。次段で
    強さ測定と合わせて検討。
  - 視点は既存パイプライン準拠(手番側、eval をそのまま sigmoid(eval/600))。新規の視点処理なし。

## 成果物の場所

- 学習スクリプト: `trainer/prepare_data.py`, `trainer/train_nnue.py`。
- キャッシュ(非コミット): `/tmp/binbo-task017/cache/`(train/val の .npy、約 310MB)。
- 学習済み重み(非コミット、各 ~45MB):
  - 標準: `/tmp/binbo-task017/model/{nnue_best.pt,nnue_final.pt,model_config.json,history.json}`
  - weight decay: `/tmp/binbo-task017/model_wd1e-4/`, `/tmp/binbo-task017/model_wd1e-3/`
  - **次段推奨: `/tmp/binbo-task017/model_wd1e-3/nnue_best.pt`(epoch9, val 0.6182)**。
- 再現コマンド:
  - `python -m trainer.prepare_data /tmp/binbo-task016/run1m /tmp/binbo-task017/cache --val-shards 5`
  - `python -m trainer.train_nnue /tmp/binbo-task017/cache --epochs 15 --batch-size 1024 --lr 1e-3 --weight-decay 1e-3 --out /tmp/binbo-task017/model_wd1e-3`
