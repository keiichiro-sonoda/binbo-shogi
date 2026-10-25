# experiments/alphazero — (c) AlphaZero アークの実験コード(Tasks 073–087)

binbo-shogi の「技巧2 を超える」研究で試した **単機 AlphaZero(Pgx/JAX)サブアーク**の手書きソース。
これまで `~/task074_frameworks/pgx/examples/alphazero/`(第三者 pgx クローン内)に隔離・非コミット
だったものを、履歴・再現性・バックアップのためにバージョン管理下へ移した(Task 087 で決定)。

**大局・各タスクの結論は `docs/PROJECT_MAP_057-085.md` と `reports/task0NN_*.md` が正本。** ここはその
数字を生んだコード。

## SSOT: このディレクトリが唯一の実体(Task 088)
実行は pgx クローン `~/task074_frameworks/pgx/examples/alphazero/` で行うが、そこの `*.py`/`*.sh` は
**すべてこのディレクトリへの symlink**(`link_into_clone.sh` が張る)。→ **実体は repo だけ、手動同期は無い**
(編集はここ or symlink 経由のどちらでも同じ実体に効く)。データ/ckpt/log はクローン側に残る
(スクリプトは `abspath(__file__)` で解決するので symlink でもクローンを指す=repo に GB が漏れない)。

```
# 初回 / ファイルを増やした後: クローンへ symlink を張り直す(冪等)
bash experiments/alphazero/link_into_clone.sh
# 実行
cd ~/task074_frameworks/pgx/examples/alphazero && source ~/task074_frameworks/venv_pgx/bin/activate
```
- 起動は repo の `scripts/detached.sh <tag> <cmd>` 経由 → ログ先頭に `RUN_META HEAD=<sha> dirty=<n>` が
  刻まれ「どの版で回したか / 未コミット版か」が後から分かる(Task 088)。
- **注意**: `network.py` だけ pgx-tracked(他は untracked)。クローンで `git checkout network.py` すると
  symlink が上流版に潰れる。pgx は v2.6.0 pin・pull しない運用ゆえ実害なし。

## 依存
- **pgx**(`sotetsuk/pgx` Apache-2.0、site-packages 導入。`pgx._src.games.shogi._observe` 等に依存)
- **venv_pgx**: `jax`(CUDA)/ `dm-haiku` 0.0.16 / `mctx`(Gumbel MuZero)/ `optax` / `numpy` / `cshogi`(審判・hcpe)
- 教師データ・ckpt・cache は非コミット(`~/task074_frameworks/…` と `~/binbo-shogi-data/…`、`.gitignore` 参照)

## 帰属(attribution)
- `network.py`: `AZNet`/`BlockV1/V2` は **pgx の AlphaZero 例由来(Apache-2.0)**。`TransformerNet`/
  `EncoderLayer`/`_PGX_ORDER` は我々の追加(Task 087, ChessFormer arXiv 2605.19091 を将棋へ輸入)。
- `train.py`(pgx 原本)は**含めない**——上流 pgx にある。`train_m0.py`/`train_m1.py` はそれを将棋/
  どうぶつ将棋へ改変した我々版(M0/M1)。
- それ以外(`m1_*`/`m2_*`/`m3_*`/`m4_*`/`az_rescore.py`/`t087_*`)はすべて我々の手書き。

## ファイル地図(タスク別)
| ファイル | 役割 | 主タスク |
|---|---|---|
| `m1_parse.py` `m1_sfen_bridge.py` `m2_moves.py` `m2_verify_moves.py` | SFEN↔Pgx 橋・USI手↔action id(`dir*81+to`)・合法性検証 | 075–077 |
| `m2_distill_data.py` `m2_distill_train.py` | 価値のみ蒸留(warm-start WS1)+ `make_forward`(**arch 分岐点**) | 077 |
| `m2_usi.py` | USI エンジン(AZNet/TransformerNet + mctx Gumbel MCTS)。`--bench` = per-move latency。`load_model` に arch dispatch | 077, 087 |
| `m2_match.py` | usi-vs-usi 対局ドライバ(cshogi 審判、yane/gikou/az2、elo+CI) | 077+ |
| `m3_loop.py` `m3_sprt.py` `m3_run.sh` | MCTS 自己対局ループ + trinomial SPRT | 078–081 |
| `m4_distill_data.py` `m4_distill_train.py` | 価値+方策 蒸留(WS2)。`--arch transformer` 等 | 080, 087 |
| `az_rescore.py` | 技巧2@d8 探索cp を AZ value で上書き(nodchip 式 NNUE 教師) | 084, 085 |
| `network.py` : `TransformerNet` | 等パラメータ Transformer トランク(81マス=トークン) | **087** |
| `t087_*.sh` | 087 の蒸留/対局/bench ランチャ | 087 |

## 一変数の作法(重要)
各実験は「一変数」で比較する規約。例 087 = **アーキのみ**差(入力 obs・データ 2M・蒸留設定・
パラメータ数~・評価条件 sims128 を CNN と揃え、batch 差は CNN-b256 対照で分離)。詳細は各 report。
