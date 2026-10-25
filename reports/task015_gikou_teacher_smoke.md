# Task 015: 技巧2結合スモーク

## 結論

- Rust から技巧2を USI 子プロセスとして起動し、`score cp` と `bestmove` を取得して hcpe に書けることを確認した。
- 生成した 500 局面の本物評価値つき hcpe は、cshogi 照合、KP 可逆性、学習スモークを通過した。
- 500 局面実測では 4.070 ms/局面。単純外挿で 100万局面は約 1.13 時間、500万局面は約 5.65 時間。
- 生成データは `/tmp/binbo-task015/gikou_real_500.hcpe`。技巧2本体と生成データはコミットしない。

## 実行条件

- 技巧2: `/tmp/binbo-task007/gikou-run/release`
- 設定: `OwnBook=false`, `Threads=1`, `ResignScore=-32600`
- 生成コマンド:
  - `CARGO_HOME=/tmp/binbo-task011-cargo cargo run --release --locked -- hcpe-gikou /tmp/binbo-task007/gikou-run/release /tmp/binbo-task015/gikou_real_500.hcpe 500 8 6 256 20260621 false`
- 生成条件:
  - depth: 8
  - ランダム出だし: 6 plies
  - 最大手数: 256 plies
  - seed: 20260621

## 生成結果

- positions: 500
- games: 5
- elapsed: 2034.822 ms
- avg: 4.070 ms/position
- Gikou analysis avg: 3.836 ms/position
- eval range: -8932 .. +7175
- mate cutoffs: 4
- reported depth: 8 .. 8
- hcpe size: 19,000 bytes (= 500 * 38)

## 検証結果

- cshogi照合:
  - コマンド: `/tmp/binbo-task009/venv/bin/python tools/verify_hcpe.py /tmp/binbo-task015/gikou_real_500.hcpe /tmp/binbo-task015/gikou_real_500.hcpe.expected.tsv`
  - 結果: `entries=500`, `cshogi_dtype_itemsize=38`, `cshogi_readback=ok`
  - 技巧2の `score cp` は hcpe の `eval i16` として符号・スケール変換なしで一致。
- KP可逆性:
  - コマンド: `/tmp/binbo-task009/venv/bin/python -m trainer.verify_kp_bulk /tmp/binbo-task015/gikou_real_500.hcpe`
  - 結果: `reversible_match_rate=500/500`, `reversibility_mismatches=0`, `feature_count_mismatches=0`
- 学習スモーク:
  - コマンド: `/tmp/binbo-task009/venv/bin/python -m trainer.train_smoke /tmp/binbo-task015/gikou_real_500.hcpe --steps 20 --batch-size 16`
  - 結果: `nan_check=ok`, `train_smoke=ok`, `weight_update_max_abs_delta=0.0164308827`
- Rust回帰:
  - `CARGO_HOME=/tmp/binbo-task011-cargo cargo test --release --locked` → 5 tests passed
  - `CARGO_HOME=/tmp/binbo-task011-cargo cargo run --release --locked -- verify` → perft 1〜5 完全一致

## 発見と修正

- 初期局面を `OwnBook=false` で技巧2に投げると `score mate -2` / `nodes 0` になるため、ランダム出だし後の局面から収集する。これは設計上の「出だし数手をランダムで散らす」とも整合する。
- 終盤で `score mate` が出た局面は cp 教師値に混ぜず、その対局を打ち切る。今回の 500 局面では 4 回。
- cshogi照合で `bestMove16` の打つ手の駒種番号ズレが露出した。cshogi互換の drop 番号は `P,L,N,S,B,R,G = 1..7`。`shogi_core::PieceKind` の列挙順とは一致しないため、HCPE writer に明示マッピングと全drop種の回帰テストを追加した。

## 大規模化見積もり

- 時間:
  - 100万局面: 約 1.13 時間
  - 500万局面: 約 5.65 時間
  - 1000万局面: 約 11.3 時間
- ディスク:
  - hcpe は 38 bytes/局面。
  - 100万局面: 約 38 MB
  - 500万局面: 約 190 MB
  - 1000万局面: 約 380 MB
- 並列化余地:
  - 技巧2プロセスを複数並列に立てれば短縮余地がある。
  - ただし今回の実装は単一プロセス同期実行。並列化は大量生成タスクで別途実装する。

## 大量生成前の対処

- レジューム: 途中停止から追記再開できる manifest と shard 分割を入れる。
- タイムアウト: 技巧2のハングを検出し、プロセス再起動できるようにする。
- 定期検証: shard ごとに cshogi 照合、KP 可逆性、件数とファイルサイズ検査を回す。
- terminal処理: `score mate`、`bestmove resign`、最大手数、千日手候補を明示的に記録する。
- 並列化: 14700KF / Xeon 用に複数技巧2プロセスで shard を独立生成する。
- 再現性: Gikou path、depth、random plies、seed、git commit、生成日時を manifest に保存する。
