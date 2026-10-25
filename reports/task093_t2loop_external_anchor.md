# Task 093 — T2-loop checkpoint の外部アンカー測定

**目的**: Task 091 の WS-T / iter25 / iter50 / iter75 を、NNUE補助方策head橋を介さず技巧2 DepthLimit 6へ直接当てる。Transformer warm-startが Task 087 で score 0.15 の測定可能域へ入ったため、value固定・policy更新だけのT2-loopが外部棋力を動かしたかを初めて直接測る。

## 条件と再現性

- 対局器は既存 `m2_match.py` / `m2_usi.py` を無変更で使用。cshogi審判、Gumbel MCTS `sims=128`、`byoyomi=100ms`、技巧2 `OwnBook=false / Threads=1 / DepthLimit=6 / MinThinkingTime=10`。
- openingは合法一様乱数6手、seed `20260702`。100 openingを各2局、先後反転して1系列200局。4系列で同じ100 openingを共有した。最大320手、逐次実行、並列なし。
- WS-T=`t087_wst/ws2.ckpt`、T2-loop=`t091_s32_b128_ckpt/{000025,000050,000075}.ckpt`。4 checkpointのSHA-256とロードsmokeを実走前に確認した。T2 3世代のfrozen value digestは同一。
- Task 087旧100局とfresh WS-Tの先頭100局はopening列が完全一致した。一方、結果列は旧 `11-8-81` に対しfresh先頭100局 `10-10-80` で完全一致しなかった。時間制御対局には小さな実行揺らぎがあるため、全比較は今回同一実走のfresh 200局を基準とする。
- `scripts/detached.sh`、clean `dirty=0 exp_dirty=0`。実走は16:50〜18:35の約1時間45分。学習、checkpoint、Rust、既存実験ソースは変更していない。

## 絶対結果 — vs 技巧2@6

W-D-LはAZ側視点。score=`(W+0.5D)/200`、Eloは技巧2@6相対。

| 世代 | W-D-L | score | Elo | 95% CI | WS-T CIとの関係 |
|---:|---:|---:|---:|---:|---|
| WS-T / iter0 | **19-16-165** | **13.50%** | **−322.7** | **[−398.0, −265.6]** | 基準 |
| iter25 | 15-7-178 | 9.25% | −396.7 | [−495.7, −329.5] | overlap |
| iter50 | 12-7-181 | 7.75% | −430.3 | [−539.9, −359.3] | overlap |
| iter75 | 7-11-182 | 6.25% | **−470.4** | **[−585.9, −398.03]** | **negative / 非重複** |

主判定は事前固定どおり絶対Eloの95% CI非重複。iter75の上端 `−398.034` とWS-Tの下端 `−397.957` は約0.08 Eloだけ離れ、**形式的には非重複だが極めて境界的**である。単独の絶対CIだけなら強く言い過ぎない。

全800局でengine failure・非合法手0。終了理由はmateがWS-T/25/50/75で165/177/179/182、resignが19/15/12/7、千日手が16/7/7/11。iter25には連続王手千日手による `rep_lose` 1、iter50には `rep_win` 2があり、審判が規定どおり処理した。最大手数打切りは0。

## 同一opening paired解析

絶対CIは系列を独立扱いするため、同じ100 openingペアを共有した情報を補助解析に使った。openingペアを再標本化単位とし、seed `20260718`、200,000 bootstrapで `ΔElo vs WS-T` を求めた。

| 世代 | ΔElo vs WS-T | paired 95% CI |
|---:|---:|---:|
| iter25 | −74.0 | [−176.7, +23.1] |
| iter50 | −107.6 | [−225.7, **0.0**] |
| iter75 | **−147.8** | **[−266.8, −43.1]** |

iteration `0/25/50/75` への線形当てはめは **−47.7 Elo / 25 iter**、paired bootstrap 95% CI **[−85.5, −14.1]**。iter50の上端はちょうど0で境界だが、iter75と全4点の傾きは0を含まない。主判定のiter75非重複が僅差である一方、同一openingを使った補助解析も同方向を示した。

## 内部・旧教師・外部の3カーブ

外部列は今回のpaired `ΔElo vs WS-T`。旧教師列はTask 091の80局ゲートだが、Task 092-0でtraining-only補助head橋が陽性対照にも反応しないと判明したため、参考値へ格下げする。

| iter | 内部Elo vs WS-T | 旧NNUE教師Elo | 外部ΔElo vs WS-T | 技巧2手top-1 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0.4788 |
| 25 | −52 | +70 | −74 | 0.4570 |
| 50 | **+61** | 0 | **−108** | 0.4536 |
| 75 | −26 | 0 | **−148** | 0.4513 |

### 整合と乖離

1. **外部は単調に悪化**: score `13.50→9.25→7.75→6.25%`、相対Elo `0→−74→−108→−148`。Task 091のループは外部を「動かさなかった」のではなく、今回の条件では **悪い方向へ動かした**。
2. **内部と外部は非転写どころか逆向き**: iter50は内部 +61 Eloで3世代中最高だが、外部はWS-T比 −108 Elo。内部対局だけで世代選抜すると外部悪化を選ぶ具体例になった。
3. **静的top-1と外部は同方向**: 技巧2手一致率は世代とともに低下し、外部scoreも低下した。ただしtop-1は因果オラクルではなく、直接対局が主証拠である。
4. **value固定下のpolicy効果**: T091のvalue towerは全世代で同一digest・同一出力。MCTSのvalue情報を変えずpolicy towerだけ更新して外部が悪化したため、少なくともこの完全二塔ループでは **方策更新が外部悪化を生んだ範囲**まで切り分けられる。
5. **旧教師曲線は判定不能**: `+70/0/0` が外部と合わないのは、Task 092-0で確認したbridge nullと整合する。旧NNUE補助head教師ゲートを再利用しない。

## 判定とTask 092-Bへの申し送り

**判定: NEGATIVE_EXTERNAL_MOVEMENT。T2-loop完全二塔のpolicy更新は、WS-Tから技巧2@6への外部棋力を単調に悪化させた。** 絶対CIの主判定はiter75で境界的に非重複、paired傾きは明確に負で、両者が同方向を支持する。

Task 092-Bへ進む場合の前提:

- 完全二塔のpolicy-only更新は対照として外部負傾斜。共有幹+WS-T value蒸留正則化は、まず25 iterでこの悪化を防げるかを比較する。
- 補助方策head橋は判定から引退。候補自身のvalueで2M rescoreするTask 084方式の相対NNUE判定と、今回確立したgikou@6直接対局を使う。
- 2M PASSは32M本走の資格に留める。Task 085の2M→32M反転があるため、2M結果を最終教師改善とは呼ばない。
- 本タスクではλ、停止条件、共有幹実装を決めず、Task 092-Bの実装・実走にも着手しない。

## 成果物

- master log: `~/binbo-shogi-data/logs/task093_t2_external_20260718-165002.log` (`DETACHED_DONE rc=0`)
- machine-readable: `~/binbo-shogi-data/logs/task093_results.json`
- 各系列: `~/binbo-shogi-data/logs/task093_{wst,iter25,iter50,iter75}_vs_gikou6_s128_b100_seed20260702_n200.{tsv,console.log}`
- checkpoint/TSV/log/一回限りの解析コードは非コミット。数値と判断だけを本レポートへ蒸留した。
