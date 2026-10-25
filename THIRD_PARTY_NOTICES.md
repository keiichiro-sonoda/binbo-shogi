# 第三者のソフトウェアとデータ

binbo-shogi の作者が書いた部分は GPL-3.0-or-later (`LICENSE`) で配布する。
次のものは他者の著作物で, それぞれ元のライセンスに従う。来歴と使い方の詳細は `docs/USED_COMPONENTS.md`。

## このリポジトリに含まれるもの

| もの | 場所 | ライセンス | 備考 |
|---|---|---|---|
| Pgx の AlphaZero の例 (Copyright 2023 The Pgx Authors) | `experiments/alphazero/network.py` の `AZNet` / `BlockV1` / `BlockV2`, `train_m0.py`, `train_m1.py` | Apache-2.0 (`LICENSES/Apache-2.0.txt`) | 将棋・どうぶつ将棋向けに改変した。各ファイルの先頭に帰属と改変を記載 |

## 依存するが同梱しないもの

| もの | 取得元 | ライセンス | 使い方 |
|---|---|---|---|
| やねうら王 (YaneuraOu) | https://github.com/yaneurao/YaneuraOu | GPL-3.0 | 対局時の探索部 (改変なし)。評価関数の形式, 静止局面への置き換え |
| yasai | https://github.com/sugyan/yasai | GPL-3.0 | 自前の Rust 探索の合法手生成 (`Cargo.toml` で rev を固定) |
| shogi_core / shogi_usi_parser | crates.io | MIT | 同上 |
| Pgx / mctx / JAX / dm-haiku / optax | PyPI | Apache-2.0 | 自作の Transformer の学習と探索ラベルの生成 |
| cshogi | https://github.com/TadaoYamaoka/cshogi | GPL-3.0 | 局面の検証・対局の審判 |
| PyTorch / NumPy | PyPI | BSD 系 | 評価関数の学習器 |

## 学習に使った他者のデータ (同梱しない)

| もの | ライセンス | 使い方 |
|---|---|---|
| 公開教師データ nodchip/shogi_suisho5_depth9 (Hugging Face) | MIT | 2026-10-24 からの評価関数の土台と仕上げ |
| 技巧2 (Gikou v2.0.2) | GPL-3.0 | それ以前の系統で, 教師にする局面の生成と蒸留元 |

## 強さの測定にだけ使ったもの (同梱しない, 学習・教師には使っていない)

tanuki- (GPL-3.0), AobaNNUE (GPL-3.0), つようら王11micro (MIT)。

学習済みの重み・教師データ・外部エンジンの実体はこのリポジトリに含まない。
