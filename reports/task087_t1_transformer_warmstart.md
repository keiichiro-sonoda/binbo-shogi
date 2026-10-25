# Task 087 — T1: Transformer warm-start 検証(第2章の入り口、ループなし)

**目的**: 086 地図 §5「唯一開いた道=第2章(Transformer+規模で真に強い AZ)」の最初の一歩。Transformer ネットを Pgx/JAX(dm-haiku)で定義し、**warm-start 蒸留のみ**を CNN(WS2-128)と**同一の物差し**で比較 + 推論スループット実測で登山コストを外挿。ループなし。判定=warm-start の質だけが外部を動かす(082)ゆえ **vs champion が本判定**、WS2-128 の 2-3-95(アーク初勝利)が比較基準。実測 2026-07-12、隔離環境(~/task074_frameworks、非コミット)、**リポジトリは文書のみ・Rust/champion 無改変**。

## 設計(ChessFormer [arXiv 2605.19091, ICLR2026] を将棋へ輸入)
- **トークン化**: 既存 `(9,9,119)` obs をそのまま **81 マス×119 特徴**に reshape=**純粋なトランク差し替え**(入力は CNN と同一=一変数=アーキのみ)。各マストークン=その升の駒/利き/持駒(broadcast)/王手特徴。
- **policy head**: per-token(=行き先升 `to`)→ 27 方向 logits →(81,27)→ transpose →(27,81)→ 2187。Pgx action id = `direction*81 + to`(m2_moves 確認)に幾何整合。★obs 空間配置は Pgx 升 index(x*9+y)を `rot90(k=3)` した並び(shogi.py:813)ゆえ、reshape 後に 81 トークンを Pgx 升順へ再整列(`argsort(rot90(arange(81).reshape(9,9),3).flatten())`=構成で正・round-trip 検証済)→「行き先升トークンがその升への手を予測」の事前分布が幾何的に正確。
- **value head**: mean-pool(順序不変)→ Linear→gelu→Linear(1)→tanh。**位置エンコ**: 学習 pos-emb(81×D)(GAB は T1.5 温存)。
- **encoder**: pre-LN(`LN→MHA→残差, LN→FFN(4×,gelu)→残差`)。`hk.MultiHeadAttention`(haiku 0.0.16 内蔵=**新フレームワーク不要**、Flax 未導入)。LayerNorm=状態レス→`transform_with_state` の state 空→apply 署名が AZNet と同一→蒸留ループ/search/bench 無改変(`make_forward` に arch 分岐 1 箇所)。
- **等パラメータ**: 採用 **d_model 240 / L5 / H8 / ff4 = 3.585M**(WS2-128 CNN = 3.462M、+3.6%=わずかに大で「Transformer が小さいから」の言い訳を封じる)。

## 蒸留(m4_data 2M, 価値 MSE + 方策 CE、v=tanh(cp/1200)、lr 1e-3/seed 0 は CNN と一致)
### メモリ(実測の第一発見)
- MHA の O(batch·seq²·d) 中間が 16GB 4060Ti の XLA ~12.9GB/proc 上限を超過。**致命 alloc = batch1024:9.9GiB / 512:4.95GiB(線形)→ ともに OOM。batch 256 で working set ~6.5GB=安全**。=**Transformer は CNN の ~1/4 の batch しか載らない**(attention の学習時メモリ)。加えて小 batch=4× 多い CPU unpackbits 反復 → epoch 726s。transformer 用に adam+linear warmup 300 step(唯一のアーキ固有配慮、peak lr は CNN と同一)。

### val 学習曲線(val_corr / top1)
| epoch | Transformer(b256) | CNN WS2-128(b1024, 既存) | CNN-b256(対照) |
|---|---|---|---|
| 1 | 0.884 / 0.388 | 0.868 / 0.341 | 0.883 / 0.356 |
| 4 | 0.924 / 0.450 | **0.905 / 0.400(best)** | 0.914 / 0.410 |
| best | **0.936 / 0.472(ep8)** | 0.905 / 0.400 | 0.916 / 0.416(ep7) |

- **交絡の処理**: batch 256 の Transformer は同 epoch で CNN-b1024 の 4× の勾配更新=「equal epochs ≠ equal steps」。→ **CNN-b256 対照**(batch/step も揃え arch のみ差)を実施。小 batch は CNN にも効く(b1024 0.905/0.400 → b256 0.916/0.416)=生 val gap の約半分は batch 効果。
- **厳密 arch-only(両者 b256, 収束 best)**: Transformer 0.936/0.472 vs CNN-b256 0.916/0.416 = **corr +0.020 / top1 +0.056**。→ **交絡除去後も Transformer が両指標で明確勝ち、特に方策 top1(+0.056)**。epoch1 から一貫優位・サンプル効率も高い。

## 対局(同一物差し sims128 / byoyomi100ms / cshogi 審判、非合法手 全対局 0 件)
| 対戦 | W-D-L | score | elo(CI95) | 基準(CNN) |
|---|---|---|---|---|
| ① WS-T vs WS2-128(CNN-b1024) | 34-6-0 / 40 | 0.925 | **+436**(+330..+679) | — |
| ④ WS-T vs CNN-b256(**厳密 arch-only**) | 34-4-2 / 40 | 0.900 | **+382**(+265..+668) | — |
| ② WS-T vs **champion**(task064 NNUE) | 3-0-37 / 40 | 0.075 | **−436**(−2400..−293) | WS2-128 = 2-3-95, −576 |
| ③ WS-T vs gikou@6 | 11-8-81 / 100 | 0.150 | **−301**(−411..−226) | WS2-128 = 0-6-94, ~−604 |

- **内部棋力=等パラメータ・アーキ由来の圧勝**: ①34-6-0 / ④34-4-2 とも ~90%。**batch 一致(④)でも 34-4-2** ゆえ 34-6-0 は交絡でなくアーキ由来。val 優位が実棋力に完全転写。
- **外部転写も CNN 超え(アーク最良)**: champion **+140 elo**(勝率 7.5% vs 2%=3.75×、ただし依然 37/40 敗=不倒)、gikou@6 **+300 elo・11 勝 vs 0 勝**。=**net 質改善は外部に正転写、相手の探索が深いほど圧縮**(champion 深 αβ<gikou 浅 αβ)。

## スループット(m2_usi --bench, batch1・JIT-warm 除外, median ms/move)
| sims | Transformer | CNN | T/CNN |
|---|---|---|---|
| 16 | 29 | 22 | 1.32 |
| 32 | 29 | 50 | 0.58 |
| 64 | 59 | 94 | 0.63 |
| **128(対局条件)** | **110** | **145** | **0.76** |

- **対局域(sims64/128)で Transformer が ~1.3-1.6× 速い**(batch1 推論で CNN の 128ch×10 層 conv+BN=深く逐次より、Transformer の 5 層・81 トークン attention が安い)。低 sims は固定 dispatch overhead(~29ms plateau)。
- **★コストの非対称性(核心)**: メモリ大食いは **学習の backward(活性保持)のみ**。**推論/自己対局生成(forward)は逆に速い**。AZ ループの wall-clock 支配項=**自己対局生成(forward, batched)**ゆえ **Transformer でループは遅くならない・むしろ速い**。学習更新は小 batch(勾配蓄積で緩和可)。
- **登山コスト外挿**: 第2章を年単位化する要因は**規模(大 net×多い自己対局)=任意 AZ 共通**であって Transformer 固有の重さではない(推論は CNN 以下)。=**アーキ選択は第2章を重くしない**(ChessFormer 8-30× 容量効率の方向と整合)。

## 判定 = **GO(第2章を Transformer で続行)**
1. **等パラメータで Transformer が warm-start の質・内部棋力ともに明確勝ち**(arch-only: val corr+0.020/top1+0.056、対局 34-4-2 ~90%)=第2章の土台として CNN より良いことを実証。
2. **外部転写も CNN 超え(アーク最良)**: champion +140 / gikou +300。net 質改善は外部に正転写。
3. **コストは登山を重くしない**: 推論は CNN より速く(sims128=110 vs 145ms)、メモリ制約は学習 batch のみ(緩和可)。Transformer 固有の「年単位化」要因なし。
- **留保(過大評価しない)**: champion は sims128・warm-start のみでは依然不倒(37/40 敗)。**床を破るのは loop+規模(T2+)**=本番の高コスト・不確実部分(内部→外部転写は正だが深 αβ で強く圧縮=083/085 の壁は健在)。T1 は「土台が良い+アーキはコスト増でない」を確定しただけ。

## 申し送り(T2 以降)
- **T2 = ループ設計**を Transformer(d240/L5)で。self-play 生成は CNN 以下の推論コスト、学習 batch は 256(勾配蓄積で実効拡大可)。gen100 の value 圧縮(085)を監視する retention/教師品質モニタを設計に組込む。
- **規模軸(第2章の本命)**: 容量効率(ChessFormer)を活かすなら Transformer を大型化(d/L↑)して真に強い net → その value を NNUE 教師(085 の反転は「新情報なし」が原因、真に強い独立 net なら nodchip 水匠に相当=新情報)。
- **GAB(動的位置エンコ)= T1.5** の伸び代(learned pos-emb からの改善余地)。
- **資産**: WS-T ckpt=`t087_wst/ws2.ckpt`(ep8, 3.585M, corr0.936/top10.472)、CNN-b256 対照=`t087_cnn_b256/ws2.ckpt`、`network.py:TransformerNet`+`make_forward(arch=)`+`m2_usi` arch dispatch(CNN 後方互換)。全 TSV/log は ~/binbo-shogi-data。

## 規律・メタ
一変数=アーキのみ(入力 obs/データ 2M/蒸留設定/パラメータ数~等/評価条件 sims128 を CNN と揃え、batch は CNN-b256 対照で分離)。非合法手 0 継続。隔離・非コミット(network.py/m4_distill_train/m2_usi 拡張・ckpt・TSV)、既存 m4_data/m2_match/m3_sprt 無変更流用。**champion・Rust 無改変。push は人間承認後。**
