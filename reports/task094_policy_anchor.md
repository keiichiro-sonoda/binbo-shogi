# Task 094 — 方策アンカー付き frozen-value T2-loop(093 の外部悪化を止められるか)

**目的**: Task 093 で「value 固定・policy だけ更新する完全二塔 T2-loop」は技巧2@6 の外部棋力を **単調悪化**させた(iter25/50/75 の paired ΔElo vs WS-T = −74/−108/−148、傾き −47.7/25iter)。悪化の機序候補は明快で、ループ方策が **WS-T(技巧2蒸留)の指し手を忘れる**(技巧2手 top-1 が 0.4788→0.4513、argmax 約36%変更)。本タスクは Task 091 のループに **一変数だけ**足す: policy 最適化 update の**固定割合を WS-T supervised best-move replay に置換**し(残りは従来どおり MCTS 訪問分布)、`r=0.25 / 0.50` の2armで **①外部悪化を止められるか ②止めた上で WS-T を超える正の転写が出るか** を測る。

## 条件と一変数
- **不変(091 と同一)**: 完全二塔・value 塔 byte-for-byte 凍結・`sims=32`・`selfplay_batch=128`・`states_per_iter=65,536`・`training_batch=128`・**512 updates/iter**・self-play 生成・optimizer・総 update 数。
- **一変数=アンカー率**: 512 updates のうち固定割合が WS-T 教師 best-move ラベル(`m4_data` の 2M、技巧2蒸留)、残りが MCTS 訪問分布。`r=0.25`→128 update、`r=0.50`→256 update が anchor。`schedule=batch_interleave`、`anchor_seed=20260714`。実測 `anchor_fraction` は両arm 厳密に 0.2500 / 0.5000。
- **value 凍結 = GREEN**: 全 iter で `value_model_digest == reference_value_digest = 2316885110a0…`、`value_max_abs_cp=0`、`reference_value_param_digest_exact=True`。091 と同じ完全保護を維持=**外部変化は policy 更新だけに帰属できる**。
- **判定器(093 と厳密同一物差し)**: iter25 checkpoint を `m2_match.py` 無変更で技巧2@6(`OwnBook=false/Threads=1/DepthLimit=6/MinThinkingTime=10`)へ直接、Gumbel MCTS `sims=128`・`byoyomi=100ms`・合法一様乱数6手 opening・`seed=20260702`・**200局**。paired ΔElo vs WS-T は 093 の WS-T 200局(同 100 opening ペア、score 0.135)を基準に 200,000 bootstrap。
- **事前ゲート**: 両arm を iter25 まで走らせ 200局判定し、**正arm(paired CI 下端>0)のみ 50iter へ延長**。
- `scripts/detached.sh`、clean `dirty=0 exp_dirty=0`。実走 2026-07-24 22:15〜07-25 07:44(約9.5h)。median iter ≈ 582s(r025)/578.5s(r050)、projected loop 8.06h。学習コード・champion・Rust・既存 ckpt は不触。

## 絶対結果 — vs 技巧2@6(iter25 / 200局、AZ視点)

| arm | W-D-L | score | 絶対Elo | 95% CI |
|---|---:|---:|---:|---:|
| **WS-T / iter0**(093 基準) | 19-16-165 | **13.50%** | −322.7 | [−398.0, −265.6] |
| **r=0.25** | 24-12-164 | **15.00%** | −301.3 | [−375.0, −244.7] |
| **r=0.50** | 31-17-152 | **19.75%** | −243.6 | [−306.6, −191.9] |

両arm 200局とも非合法手・engine failure 0。終了理由(r050)= mate 152 / resign 31 / sennichite 16 / max_plies 1。

## 同一opening paired 解析(vs WS-T)

| arm | paired ΔElo vs WS-T | paired 95% CI | 分類 |
|---|---:|---:|---|
| r=0.25 | **+21.3** | [−72.8, +118.6] | FLAT |
| r=0.50 | **+79.1** | [−11.1, +174.2] | FLAT |

- **両arm とも点推定は WS-T の上**(093 の anchor 無しは iter25 で **−74**)。r=0.50 は +79.1 で下端 −11.1 と 0 にごく僅かに届かない=**あと一歩で有意**だが FLAT。
- **正arm(CI 下端>0)は無し** → 事前ゲートどおり **50iter 延長は実施せず**。

## 機構 — アンカーは何を止め、何を足さなかったか

093/091 のループが起こした「WS-T の指し手忘却」が、アンカーで抑制された:

| 指標(iter25 近傍) | 091/093 ループ | 094 r=0.25 | 094 r=0.50 |
|---|---:|---:|---:|
| 技巧2手 top-1(val) | 0.4788→**0.4513** | **0.4781** | **0.4796** |
| argmax_change vs WS-T | ~0.36 | **0.217** | **0.209** |
| visit_argmax_agree(iter25) | — | 0.802 | 0.764 |
| value digest | 固定 | 固定(一致) | 固定(一致) |

1. **忘却の停止**: アンカー無し(093)は top-1 を 0.4788→0.4513 へ落としたが、094 は両arm とも **~0.478-0.480 に保持**、WS-T からの argmax 変更も 36%→~21% に縮小。=**supervised replay が方策を WS-T 近傍へ繋留し、093 の単調悪化を反転させた**(paired ΔElo −74 → +21/+79)。これは本タスクの明確な陽性所見。
2. **正の転写は無し**: 繋留の効果は「WS-T へ戻す」ところで頭打ちで、**WS-T を有意に超えない**(両arm CI が 0 を含む)。r=0.50 の方が anchor 比率が高く外部も良い(+79>+21)=**外部スコアは MCTS 由来の新規方策成分でなく、WS-T 成分の多さで説明**される向き。
3. **value 不変下の因果**: value 出力は全 iter で完全同一。∴ この差は純粋に policy 更新の帰属で、「MCTS 訪問分布への更新は外部を悪化(093)/せいぜい WS-T へ戻すだけ(094)」まで切り分けられた。

## 判定 = **FORGETTING_STOPPED_NO_POSITIVE_TRANSFER**

- **アンカーは 093 の外部悪化を止めた**(paired ΔElo −74 → +21/+79)=完全二塔ループの単調劣化は supervised replay で防げる、という機構的陽性。
- **だが WS-T を超える正の転写は出ない**(両arm FLAT、CI が 0 を含む)。効果は「WS-T へ引き戻す」上限で飽和し、MCTS 訪問分布側の新規情報は外部強度に寄与しなかった。
- **∴ このループ(sims32・value 凍結・訪問分布 policy 教師)は、アンカーを足しても WS-T 教師を改善しない**。091(教師 0 elo)・093(外部負傾斜)に続き、**完全二塔 policy-only ループは「外部を伸ばすレバー」でない**ことが3タスク連続で確定。正arm不在ゆえ 50iter・32M 還流・champion 再挑戦へは進まない。

## 申し送り(T2-loop / 第2章)
- **確定した壁**: value 凍結・訪問分布教師の完全二塔ループは、単調悪化(093)を防いでも(094)WS-T を超えない。**方策側の MCTS 訪問分布は sims32 では WS-T に無い外部有効情報を生んでいない**。次に動かすべき一変数は **policy 教師信号の質**(sims↑=48/64 で訪問分布を鋭くする、soft 訪問分布化)か、**value 側に新情報を入れる経路**(凍結を解いて WS-T value 蒸留正則化付きで幹を更新=BACKLOG の「共有幹+WS-T value 正則化」案)。
- **アンカー率は続けるなら>0.5 は無意味**: r=0.50 で既に「WS-T へ戻す」飽和。純増を狙うなら anchor でなく訪問分布側の情報量を上げる方向。
- **judge は健在**: gikou@6 直接対局(093 で確立)は −74→+79 の 150 Elo 幅を 200局で分離できた=方策変化の外部感度は十分。092-0 の bridge null とは別物として、外部 endpoint はこの直接対局で確定運用する。
- **2M PASS は 32M 確認資格まで**(085 の 2M→32M 反転ゆえ、2M 結果を最終教師改善と呼ばない)。共有幹案・λ・停止条件は別 /plan・人間判断。

## 成果物
- master log: `~/binbo-shogi-data/logs/task094_policy_anchor_20260724-221535.log`(`T094_PIPELINE_DONE verdict=FORGETTING_STOPPED_NO_POSITIVE_TRANSFER` / `DETACHED_DONE rc=0`)
- machine-readable: `~/binbo-shogi-data/logs/task094_results.json`
- 各arm: `~/binbo-shogi-data/logs/task094_r0{25,50}_iter025_vs_gikou6_s128_b100_seed20260702_n200.{tsv,console.log}`
- 実験コード `experiments/alphazero/t094_loop_anchored.py`(SSOT、 で追跡)。ckpt/TSV/log/loop 生成物はクローン側=非コミット。数値と判断のみ本レポートへ蒸留。

## 規律・メタ
一変数=アンカー率(net/data/sims/batch/value 凍結/総 update 数はすべて 091 固定)。detached.sh(RUN_META)・逐次パイプライン+一発完了 waiter で放置完走。value digest 全 iter 一致で凍結を機械照合、非合法手 0。判定器・opening・seed は 093 と厳密同一。**Rust・champion 不触。push は人間承認後。**
