# Task 089 — T2-scale: Transformer 規模スケーリング実測(warm-start は params でどう伸びるか)

**目的**: T1(087)で「等パラメータで Transformer が CNN 全軸勝ち・推論速い」と確定。本タスクは**規模(params)を一変数**に振り、warm-start 品質(corr/top1)・外部アンカー(champion/gikou)が params でどうスケールするかを**実測カーブ**で描く。データは 2M(=T1・一変数純度)、各点で **train-val gap** を測り「データ枯渇 vs 容量天井 vs 予算」を切り分け。ループなし。実測 2026-07-12〜13、隔離(experiments/alphazero SSOT・088)、全ジョブ RUN_META で版記録、Rust/champion 不触。

## 格子(一変数=params、他は T1 と同一: 2M・8ep・lr1e-3・warmup300・seed0)
d240/L5=3.58M(=T1 `t087_wst` 再利用)/ d320/L6=7.57M / d384/L8=14.43M、H8 固定。
**batch は OOM で per-point 動作点**: 3.6M@256 / 7.6M@192 / 14.4M@128(MHA の O(seq²·d) 中間が 16GB の XLA ~12.9GB 上限を超過。documented 交絡=gap 診断と対局ノイズが支配)。

## カーブ(全点 2M・8ep、sims128/mt0.1s、非合法手 全0)
| params | val corr/top1 | **gap** corr/top1 | 内部 vs 3.6M (W-D-L/40) | champion elo (W-D-L/40) | gikou elo (W-D-L/100) | bench sims128 |
|---|---|---|---|---|---|---|
| **3.6M** | **0.936 / 0.472** | −0.003 / +0.051 | (基準) | **−436** (3-0-37) | **−301** (11-8-81) | 110ms |
| **7.6M** | 0.918 / 0.442 | −0.007 / +0.012 | **−158** (5-13-22) | −511 (1-2-37) | −449 (5-4-91) | 126ms |
| **14.4M** | 0.829 / 0.307 * | −0.009 / +0.007 | **−407** (2-3-35) | **−2400** (0-0-40) | −727 (0-3-97) | 156ms |

(3.6M 外部・bench は 087 実測、gap は本タスクで補完。* **14.4M は不安定で ep1 が best**、詳細下。)

## 判定 = **規模は「同一 recipe/予算」では只のレバーでない(全軸で単調に悪化)**
- **params↑で warm-start 品質も棋力も単調に低下**: val 0.936→0.918→0.829、内部で大 net ほど 3.6M に大敗(−158→−407)、champion −436→−511→−2400(0勝)、gikou −301→−449→−727。**大きいほど弱い**。
- **機構は params で 2 つの別失敗**:
  - **7.6M = undertrained**: 全 epoch で 3.6M を下回り単調、**gap ~0(過学習でない)**、ep8 でも僅かに上昇中=8 epoch では収束しきらない(予算不足)。**2M はまだ枯渇させていない**(枯渇なら gap が開く)。
  - **14.4M = 不安定(lr 過大)**: lr 1e-3(3.6M 用)で **train loss が ep2 から上昇・発散**(v_mse 0.131→0.131→0.145、val 0.829→0.773→0.778)。best=ep1 のみ。ep3 で発散確定ゆえ**経済判断で ep3 打ち切り(kill, STAGE2_FAILED rc=143)**、best ep1 ckpt で評価。大 transformer は lr を下げる必要(既知: lr∝1/model size)。
  - **統一像**: **3.6M で調整した recipe(lr/epoch)が大 net に転写しない**。スケールは plug-and-play でない。
- **∴登山コストの現実**: 効かせるには **lr 再調整(↓)+ epoch↑ + data↑ + VRAM↑** が要る=単機 16GB/2M/8ep には無い資源・手間。第2章の「Transformer 大型化で強く」路線は、**アーキ優位(T1)とは別に、スケール自体が単機では素直に効かない**。しかも学習コストは params で超線形(大 net×小 batch×要多 epoch)、推論も 110→126→156ms と漸増。

## ★重要な留保(過小評価しない)
これは **equal-budget(lr1e-3・8ep 固定=一変数)** の結論=「**同じ recipe では** scale が効かない」。各 net を**適正 lr で収束まで**訓練すれば大 net が 3.6M を超える可能性は否定していない(7.6M は gap~0 で ep8 上昇中=伸びしろあり)。本タスクが確定したのは「T1 の recipe/budget では scale が逆効果・コスト増」。**clean な scale 研究 = follow-up**(下記)。

## 標準ネット提案
**単機 2M/8ep 域では 3.6M(d240/L5)が sweet spot**(本研究で全軸最良)。以降の (c) 標準ネット。大型化は「収束予算 + lr 調整 + 大データ」を伴わない限り非推奨。

## 申し送り(follow-up / T2-loop)
- **scale を clean に測る follow-up(安い順)**: ①**lr を model size で下げて** 7.6M/14.4M を再訓練(まず lr 3e-4 級)、②各 net を**収束まで**(epoch を budget 軸に)、③大データ(32M、disk 716G 空き)で枯渇を外す。gap がまだ小さいので枯渇対策(③)より lr/epoch(①②)が先。
- **T2-loop**: 標準 3.6M(d240/L5)で自己対局ループ設計へ(gen100 の value 圧縮監視付)。scale は loop で内部改善を得てから、適正 recipe で再検討。
- **警句**: 「大きいほど強い」は単機・固定 recipe では**成り立たない**。scale を第2章の主レバーにするなら、まず recipe(lr/epoch)を scale 対応させる小実験(①)が必須。

## 規律・メタ
一変数=params(data 2M/蒸留設定/評価条件 T1 同一、batch のみ OOM 動作点で明記、head_size 30/40/48 は規模の増やし方)。`m4_distill_train` に train-subset eval 追加(gap 診断、後方互換)。detached+RUN_META(全ジョブ HEAD 記録)。非合法手 全0。t089_* ckpt/log はクローン非コミット、コードは experiments/alphazero SSOT。**Rust・champion 不触。push は人間承認後。**
