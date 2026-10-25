# Task 092-0 — NNUE 橋の較正

**目的**: Task 091 で使った「方策ラベルを training-only 補助 head へ入れ、head を捨てた NNUE evaluator の対局で教師品質を測る」橋が、そもそも方策品質差を検出できるか較正する。自己対局ループは回さず、固定2M局面・固定 WS-T value・同一 NNUE 条件で補助 head の容量と陽性対照を調べた。

## 実態と実験条件

- 現行 NNUE は **value-only evaluator** であり、対局時の方策 head は持たない。学習時だけ QAT NNUE の最終 hidden 32次元から `2187` 手クラスの補助 headを分岐し、`cp_MSE + 0.1 * 600^2 * policy_CE` で共学習する。export 時は補助 head を捨てる。
- したがってこれは Bonanza 型の探索一致学習でも、対局時に方策を使う構成でもない。方策ラベルが value trunk への正則化として残す差だけを、最終 evaluator の棋力で読む間接的な橋である。
- 固定データは Task 091 と同じ canonical-shared HalfKP 2M cache。value ラベルは全系統で WS-T の同一配列、12 epoch、同じ optimizer/seed。変数は補助 head と方策ラベルだけ。
- `random` は各局面の合法手から固定 seed で一様に1手を選んだ **教師ラベル**。対局・探索の指し手を確率化したものではない。
- 実装は `trainer/train_nnue_policy_aux.py`、`experiments/alphazero/t091_rescore.py`、`experiments/alphazero/t092_bridge_calibration.py`。長時間実走は clean HEAD から `scripts/detached.sh` で開始した。

## Step 1 — 補助 head の寄与と容量

比較した構成:

1. `none`: value-only。
2. `linear`: 現行 `Linear(32→2187)`。
3. `mlp`: `Linear(32→512)→ReLU→Linear(512→2187)`。
4. `ft-mlp`: MLP32 が事前ゲート(top-1 4%以上)に届かなかったため追加。2視点の feature-transformer 出力を連結した512次元から `512→512→2187`。

| head | value MSE | MAE(cp) | corr | pred/target std | policy CE | top-1 | `0.1×600²×CE` |
|---|---:|---:|---:|---:|---:|---:|---:|
| value-only | 603,674 | 497.64 | 0.87891 | 0.6795 | — | — | 0 |
| linear32 | 598,609 | **487.54** | 0.88195 | 0.6754 | 6.2776 | 2.15% | 225,992 |
| MLP32 | **593,724** | 488.33 | **0.88269** | 0.6783 | 5.9829 | 3.80% | 215,385 |
| FT-MLP512 | 595,772 | 489.96 | 0.88073 | **0.6822** | **5.8717** | **5.26%** | 211,381 |

### 判定

- **A: 補助正則化の value 寄与は部分的**。現行 linear は value-only より MSE、MAE(−10.10cp)、corr(+0.00304)を改善した一方、既に圧縮側の std 比は 0.6795→0.6754 とわずかに悪化した。全 value 指標改善を要求した機械ゲートは `false` だが、「全く効かない」ではなく **小さな混合効果**である。
- **B: 32次元 hidden が主ボトルネック**。MLP32 は top-1 2.15→3.80%まで改善したが事前4%ゲート未達。FT-MLP は **5.26%**へ上がったため、feature-transformer 幹には32次元 value hidden より多くの指し手情報がある。ただし絶対 top-1 は低い。
- policy 項の重み付き寄与は value MSE の約35〜38%で、ログ上ゼロでも無視可能でもない。補助 head 自体は有限 loss で学習している。

## Step 2 — 陽性対照

Step 1 最良の FT-MLP を固定し、方策ラベルだけを **WS-T合法argmax** と **合法一様random** に差し替えた。両モデルの WS-T value ラベル、局面、構造、学習条件は同一。0cp export gate 通過後、捨てた補助 headを含まない evaluator 同士を YaneuraOu 固定探索、100ms、opening 6手、seed `20260717`、逐次400局で比較した。

| 方策ラベル | value MAE | corr | std比 | policy CE | top-1 |
|---|---:|---:|---:|---:|---:|
| WS-T argmax | 489.96 | 0.88073 | 0.6822 | **5.8717** | **5.26%** |
| 合法 random | 490.22 | 0.88021 | 0.6827 | 6.2476 | 1.65% |

補助 head の検証指標では WS-T ラベルが random を明確に上回る。しかし export 後 evaluator の対局結果は次の通りだった。

| 候補 | W-L-D | score | elo(95% CI) | SPRT |
|---|---:|---:|---:|---:|
| WS-T補助教師 vs random補助教師 | **193-204-3** | **48.6%** | **−9.6 [−43.7, +24.4]** | continue、LLR −0.486 |

エラー0。候補/基準の平均 depth は 14.99/15.03、nodes は 142,052/142,443、思考時間はともに94.39msで、探索条件の偏りはない。事前 PASS 条件 `accept_h1 または 95% CI下限 > 0` を満たさない。

**判定: STEP2_FAIL_BRIDGE_NULL。既知の最大級の方策品質差さえ、現行の training-only 補助 head → value evaluator → 対局という橋では検出できなかった。**

## ゲートに従った見送り

- Step 3 の hard/soft × WS-T/iter25 は実施しない。WS-T argmax と合法 random の差が0 eloなら、それより小さい WS-T/iter25 や hard/soft の差をこの装置で判定する根拠がない。
- Task 092-B の共有幹+value正則化ループも実施しない。候補モデルを作る前に、その改善を測る教師品質 endpoint を作り直す必要がある。
- soft label は「指し手を非決定的にする」意味ではなく、学習ラベルを1手から上位手分布へ増やす意味である。本タスクでは条件ゲート不通過のため実装していない。

## 解釈

1. **WS-T 方策と random 方策が同品質という結果ではない**。補助 head の CE/top-1 は明確に分離した。分離しなかったのは、head を捨てた後の value evaluator 棋力である。
2. **Task 091 の教師0 eloは装置の null と区別できない**。091 で iter25 教師が WS-T 教師と互角だった事実は残るが、「ループ方策の教師価値が同等」とは確定できない。現行 bridge が方策差に鈍感だった可能性が大きい。
3. **器の容量だけでは解決しない**。FT-MLPで policy top-1を5.26%まで上げても evaluator差は0。headの表現力を上げて方策を読み出せることと、その情報が value evaluatorへ有用に残ることは別である。
4. 次にループ候補を測るなら、方策ネット自身を固定外部アンカーと直接対局させる、方策 head を推論時も保持する、または evaluator の手順位へ直接結びつく損失を設計するなど、**方策品質を捨てずに読む endpoint** が必要。追加実装・実走は別タスクで判断する。

## 再現ログと成果物

- detached master: `~/binbo-shogi-data/logs/task0920_bridge_20260717-195607.log` (`DETACHED_DONE rc=0`)
- 機械可読結果: `~/binbo-shogi-data/logs/task0920_results.json`
- 400局: `~/binbo-shogi-data/logs/task061_task0920_ft_mlp_wst_vs_task0920_ft_mlp_random_mt100.tsv`
- 学習ログ: `~/binbo-shogi-data/logs/task0920_{none,linear,mlp,ft_mlp}_wst.log`、`task0920_ft_mlp_random.log`
- checkpoint/eval/log は規律どおり `~/binbo-shogi-data` または再生成可能な `/tmp` に置き、コミットしない。

## 回帰確認

- Python: 変更4ファイルの `py_compile` 成功。各headの小規模forward/backward/export smokeは本走前に成功。
- Rust: `cargo test --release --locked --offline` **24/24 green**。
- perft: `cargo run --release --locked --offline -- verify` で depth 1〜5 = `30 / 900 / 25,470 / 719,731 / 19,861,490`、全完全一致。
