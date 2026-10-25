# Task 020: 学習済み NNUE を Rust で順伝播し、Python の logit と数値一致

## 結論

- 学習済み重み(`nnue_best.pt`)を Rust が読める自己記述バイナリにエクスポートし、Rust で NNUE
  順伝播(EmbeddingBag sum 相当 + Linear/ReLU 3層)を実装した。
- 同一局面で **Python の logit と Rust の logit が誤差内で一致**:
  **20,000 局面で最大絶対誤差 1.145e-06、平均 1.29e-07、1e-4 超過ゼロ**(閾値 1e-3 に対し3桁の余裕)。
- 「Python で学んだ評価が、Rust の推論で正しく再現できる」ことを担保した。次段(αβ探索 +
  `ThinkingEngine` 接続)で評価器として使える。

## 1. 重みのエクスポート(`trainer/export_nnue.py`)

- `.pt`(state_dict)→ 自己記述バイナリ `nnue_f32.bin`。Rust 側は JSON 依存なしで読める。
- レイアウト(リトルエンディアン):
  - magic `BNN1`(4B) / `total_features` u32 / `feature_dim` u32 / `hidden_dim` u32 / `scale_cp` f32
  - 続けて 7 テンソルを **f32・C 順(行優先)** で固定順に連結:
    1. `feature_transform.weight` (353486×32)
    2. `network.0.weight` (16×32) / 3. `network.0.bias` (16)
    4. `network.2.weight` (16×16) / 5. `network.2.bias` (16)
    6. `network.4.weight` (1×16) / 7. `network.4.bias` (1)
- **転置なし**: PyTorch Linear の weight は `(out_features, in_features)` で `y = W x + b`。Rust も同じ
  `(out, in)` 行優先で読み、同じ式で計算するので転置不要(転置ズレは典型バグ源 → 設計で回避)。
- 構造は実チェックポイントから再確認(推測しない): state_dict キー・形状・dtype(全 f32)を
  `export_nnue.py` が assert。出力 45,249,496 B = 20(ヘッダ)+ 11,312,369×4。
- 再利用可能(学習し直すたびに `export_nnue.py <model_dir>` で再エクスポート)。

## 2. Rust 順伝播(`src/eval/nnue.rs`)

- `Nnue::load(path)`: magic/次元を読み、7 テンソルを f32 ベクタに展開。`total_features` が
  `kp::TOTAL_FEATURES` と一致するか検証。末尾の余りバイトも検査。
- `forward(&Position) -> f32`:
  1. `kp::kp_indices`(Task 018/019 で Python 一致済み)で立っている 77 インデックスを得る。
  2. **EmbeddingBag(sum) 相当**: 該当する埋め込み行(各32次元)を合計 → 32次元 accumulator
     (`docs/kp_spec.md §7` の通り、2視点を分けず全77を1本に合算 = Python の `mode=sum` と同じ)。
  3. `Linear(32→16) → ReLU → Linear(16→16) → ReLU → Linear(16→1)`、各 `y=Wx+b`、ReLU=`max(0,x)`。
  4. 出力は logit(シグモイド前)。
- 全計算 f32(Python モデルの dtype に合わせる)。差分更新・SIMD・量子化はしない(スコープ外)。
- 単体テスト: 小さな手組み重み(feature_dim=2, hidden=2)で accumulate と network_forward を
  手計算値(29.0 等)に照合 → 転置/ReLU/バイアスの向きを単独で検証。

## 3. 評価値の扱い

- 比較は **logit(シグモイド前)** で行う。学習は `BCEWithLogitsLoss`(logit を勝率に当てる)で、
  シグモイドは単調なので logit が一致すれば勝率も一致する。
- αβ探索で使う評価値(logit / 勝率 / cp 相当)の最終決定は次段の論点。本タスクは
  「Python と同じ logit を出せる」確認に留める。

## 4. 検証(`trainer/verify_nnue_rust.py`)— Python↔Rust クロスチェック

- 手法: Python が hcpe を読み、`.pt` をロードした MinimalNNUE で logit を計算 + SFEN を出力。
  Rust `nnue-eval` が同じ SFEN 群でエクスポート重みから logit を計算。局面ごとに差を測る。
- 局面源: Task 016 の100万局面(`run1m` 100 shard を stride サンプルで序盤〜終盤網羅)。
- 判定基準(浮動小数点なので完全一致でなく誤差内):
  - 閾値 **1e-3 abs**。根拠: f32 の相対 ε ≈ 1.19e-7、logit の大きさは最大 ~5.8。順伝播は
    77項の埋め込み和 + 3回の行列積(32/16/16項)で、丸め誤差の累積は概ね 1e-6 オーダー。
    閾値 1e-3 はその約1000倍の安全側(logit 差 1e-3 は勝率差 ~2.5e-4 で評価・探索に無影響)。

### 結果

| サンプル | 局面数 | max_abs_err | mean_abs_err | 1e-4 超過 | 判定 |
|---|---|---|---|---|---|
| run1m stride | 5,000 | 1.145e-06 | 1.28e-07 | 0 | ok |
| run1m stride | 20,000 | 1.145e-06 | 1.29e-07 | 0 | ok |

- logit 範囲 [-5.54, 5.77]、両手番を含む(黒10,245/白9,755 @20k)。
- 最大誤差の局面でも py=2.9788 / rust=2.9788(差 1.1e-6)。差は f32 の加算順序・丸めのみで、
  これは f32 単精度の分解能そのもの(値 ~3 で f32 ε ≈ 3.6e-7、累積で ~1e-6)。期待通り。
- max_rel_err はゼロ近傍 logit で見かけ上大きく出る(abs が正しい指標)。abs では全局面 1e-6 未満。

## 5. 一致しない場合の切り分け(今回は不要だった)

- 設計段階で層ごと検証できるようにした: `accumulate`(EmbeddingBag 相当)と `network_forward`
  (Linear/ReLU 連鎖)を分離し、単体テストで各々を手計算値に照合。実データでも一発で誤差内一致
  したため、層ごとの乖離追跡は発生しなかった。

## 回帰

- `cargo test --release --locked` 13/13(nnue の単体テスト2件を含む)。`cargo run -- verify`
  perft 1〜5 完全一致。ビルド警告なし。

## 成果物の場所

- スクリプト/実装(コミット): `trainer/export_nnue.py`, `src/eval/nnue.rs`,
  `trainer/verify_nnue_rust.py`, CLI `binbo-shogi nnue-eval`。
- 重み(非コミット、再生成可): `.pt` = `/tmp/binbo-task017/model_wd1e-3/nnue_best.pt`、
  エクスポート = `/tmp/binbo-task017/model_wd1e-3/nnue_f32.bin`(45MB)。
- 再現: `python -m trainer.export_nnue /tmp/binbo-task017/model_wd1e-3` →
  `python -m trainer.verify_nnue_rust /tmp/binbo-task016/run1m/shard_*.hcpe --limit 20000`。

## スコープ外(次段)

- αβ探索の実装・`ThinkingEngine` への評価器接続・実対局。
- 評価値の意味づけ(logit/勝率/cp)の最終決定。差分更新・SIMD・量子化。
