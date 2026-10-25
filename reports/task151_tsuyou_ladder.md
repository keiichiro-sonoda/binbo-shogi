# Task 151 — つようら王11micro を 3 本目の物差しにできるか (較正)

日付: 2026-10-21。コード: `experiments/alphazero/t151_tsuyou_ladder.py`。結果: `~/binbo-shogi-data/logs/task151_results.json`。
出どころ: やねうら王の FANBOX の公開記事 (2026-10-14, 全体公開)。`nn.bin`・`progress.bin` は MIT (「二次配布、商用利用などご自由に」)。
arch: micro `SFNN_ka2_2048_8_64_progress8` / mini `SFNN_ka2_4096_8_64_progress8` (入力は実質駒の位置だけ, 進行度で 8 通りに切り替え), FV_SCALE 24。
アーカイブ sha256: micro `d1326e9a…`, mini `bd5cd719…` (`public/tsuyouraou11/sha256.txt`)。

## エンジン
手元のやねうら王 (9.41) は進行度に未対応。GitHub の最新 (32165a8, 9.80git) を `engines/yaneuraou980/` に取り,
`make normal PYTHON=python3 TARGET_CPU=AVX2 COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_ka2_2048_8_64_progress8` でビルド
(sha256 `680dc8d4…`, mini 用 `6966ca2b…`)。今の実行ファイルは変えていない。1 スレッドで micro 112 万 nps, mini 75 万 nps (binbo は 約 180 万)。

## はしご (採用 eval task142_finetune @10 万ノード vs micro, 各段 104 局, seed 20286000 + 100i で序盤が重ならない)
| micro のノード数 | 1k | 2k | 4k | 8k | **11k** | 12k | 13k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| binbo の勝率 | 1.000 | 0.990 | 0.971 | 0.827 | **0.673** | 0.678 | 0.606 | 0.510 | 0.303 |

倍々の段 (1k〜32k) が 8k と 16k の間に落ちたので 11k / 12k / 13k を足した。**0.65 に最も近い段 = micro @11,000 ノード** (誤差 0 局)。
micro は 16,000 ノードで binbo の 10 万ノードとほぼ互角 = 1 ノードあたりの読みの質が約 6 倍。

## 次
標準の判定に加えるかは作者が決める。加えるなら, 比べる評価関数 (採用 eval, task129 など) を micro @11k で新しい 3 組 (s280/s281/s282) すべて指し直す。

## Task 152 (2026-10-21): Task 142 の改善は micro に対しても出るか
task142_finetune − task129, micro @11,000 ノードに対して, 新しい 3 組 (各 2,400 局): **+24.7 [+4.0, +45.4]** (s280 +24.7 / s281 +26.4 / s282 +23.2)。
普段の物差し (tanuki- + AobaNNUE) では +32.5 [+18.7, +46.4]。**系統の違う物差しでも改善は出る** = 「tanuki- に似ただけ」の疑いはほぼ晴れた。
