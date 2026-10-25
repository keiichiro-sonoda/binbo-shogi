# Task 018: Rust KP 変換を実装し、Python 付番と完全一致を検証

## 結論

- yasai 局面 → KP インデックス集合を計算する Rust 実装(`src/eval/kp.rs`)を追加した。付番は
  Python の正解基準(`trainer/kp.py`)に厳密一致させた(推測せず移植)。
- **30万局面・2分布(技巧2自己対局 20万 + RandomEngine 10万)で Rust と Python の KP が完全一致
  (不一致ゼロ)**。持ち駒・成駒・端の玉・両手番・玉位置81/81マスを含む。
- 学習(Python KP)と推論(Rust KP)で特徴の意味が一致することを担保した。次段(重み読み込み +
  NNUE 推論)の土台が整った。

## Python → Rust 付番の対応(移植のキモ)

| 項目 | Python(cshogi 基準) | Rust(shogi_core/yasai) |
|---|---|---|
| square 番号 | `(file-1)*9 + (rank-1)`、0..80 | `Square::index() - 1`(= 同式。hcpe 玉エンコードで既検証) |
| 視点正規化 | 黒=そのまま / 白=`80 - sq`(180度回転) | 同左 |
| 盤上駒ID(0..12) | `BOARD_PIECE_TO_ID`: P,L,N,S,B,R,**G=6**,+P..+R=7..12 | `board_piece_id()` で同順に明示 match |
| 手番側相対所有者 | `0=手番側 / 1=相手` | 同左(`color == turn`) |
| 2視点 | view0=手番側玉基準, view1=相手玉基準 | 同左 |
| 持ち駒 | 順 P,L,N,S,G,B,R / max [18,4,4,4,4,2,2] / offset [0,18,22,26,30,34,36] / slot 0始まり | 同左 |
| side feature | `353484 + turn`(黒0/白1) | 同左 |

- 次元: `FEATURES_PER_KING=2182`, `VIEW_DIM=176742`, `TOTAL_FEATURES=353486`, `SIDE_FEATURE_BASE=353484`
  (Python と一致、単体テストで定数も照合)。
- 駒種番号(Gold が飛の後 = id 6、成駒 7..12)は Task 015 で打つ手の駒種ズレがあった箇所と同種の罠。
  ここでは明示 match で Python と1対1対応させた。

## 検証方法(独立実装のクロスチェック)

SFEN を橋渡しにする(Rust に hcp デコーダは無い=スコープ外):

1. Python(`trainer/verify_kp_rust.py`)が hcpe を読み、各局面で cshogi board から
   **KP_python**(`trainer/kp.py`、正解基準)を計算し、その局面の **SFEN** を出力。
2. Rust の `kp-from-sfen` が同じ SFEN 群を yasai 局面に復元し **KP_rust** を計算してファイル出力。
3. Python が局面ごとに `KP_python == KP_rust`(集合の完全一致)を照合。

- 局面源は Task 016 の100万局面(`run1m` 100 shard、各 shard を stride サンプルで序盤〜終盤を網羅)と、
  Task 013 手法の RandomEngine 自己対局(`hcpe-selfplay`、別分布)。
- 多様性は実測で確認(下記)。単純局面だけに偏らないよう Task 013 の教訓を踏襲。

## 結果

- 技巧2 run1m 20万局面: **mismatches=0 / 200,000(100.0000%)**。
  手番 黒100,364/白99,636、持ち駒あり 174,388、成駒あり 112,058、端の玉 118,877、玉位置 **81/81**。
- RandomEngine 10万局面: **mismatches=0 / 100,000(100.0000%)**。
  手番 黒49,871/白50,129、持ち駒あり 79,614、成駒あり 86,865、端の玉 65,171、玉位置 **81/81**。
- 不一致は最終的にゼロ(途中の不一致なし。初期実装で startpos 単体テストを Python 値に照合して
  付番を固めてから大量検証した)。

## 補助検証 / 回帰

- 単体テスト(`src/eval/kp.rs`): 初期局面の KP が Python 値と完全一致(77特徴)、次元定数一致、
  全 index が範囲内 + side feature が丁度1つ。
- `cargo test --release --locked`: 11/11 通過。`cargo run --release -- verify`: perft 1〜5 完全一致。

## やらないこと(スコープ外、次段)

- 学習済み重みの読み込み・NNUE 順伝播・αβ探索・ThinkingEngine 接続。
- Rust 単独の逆変換(KP→局面)は任意=今回未実装(Python 側で Task 013 に可逆性確認済み、
  最優先の「Python との一致」を達成したため)。
- 差分更新(高速化)。まず正しく一致が先。
