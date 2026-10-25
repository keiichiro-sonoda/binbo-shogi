# Task 064 — 技巧2蒸留 教師データ増量(16M→32M): データレバーを絞り切る

**ゴール**: Task 062 で 8M→16M が +125 elo・頭打ちなしだった。lr 底打ち・ネット 256x2-32-32 固定・教師質(技巧2 depth8)据え置きで、
**残る一変数=データ量**を 16M→32M に倍増し、val/棋力がまだ productive か技巧2の天井(val_corr ~0.90)で頭打ちするかを確定する。
**一変数=データ量**。裏目的=「蒸留はここまで」の判断=自己強化ループ着手の最終合図。長時間・無人放置運用(Task 063 `detached.sh`)。

## 何をしたか(既存パイプライン流用, 新規コードなし)
- **追加 +16M 生成**(`gen-teacher`, 技巧2 depth8): 新 seed **20290629**(既存と非衝突=新規局面)、random_plies 6 / max_plies 256 /
  shard 10000 / **16 workers** / timeout 30s。resumable。`scripts/detached.sh gen32m` で無人起動、**1600/1600 shard を ~2h49m**(01:47→04:36)で生成、hcpe 3.2GB。
- **32M cache**(`prepare_data --canonical-shared`): primary=`run1m`(val=末尾5 shard)+ extra-runs=`run1m_b run2m_c run4m_d run8m_e run16m_f`。
  → **n_train 31,950,000 / n_val 50,000**。**val は shared_8m/16m と完全同一**(同 run1m・同 shard 95-99・同 50k=世代間で公平)。cache 9.7GB(train_indices 9.84GB)、convert 889.9s。
- **学習**(062 champion 設定固定, データのみ変数): `train_nnue --qat --qat-tie --feature-dim 256 --hidden-dim 32 --lr 1e-5 --epochs 12 --batch-size 1024 --scale-cp 600 --augment-mirror`
  (mirror perm=feature 空間=8M のを再利用)。`scripts/detached.sh train32m` で無人起動、**12 epoch 完走**(rc=0)。RSS ~11GB/22GB=OOM 余地十分(WSL RAM を 15→22GB に事前引上げ済)。train ~893s/epoch(mirror gather が CPU 律速)。

## 結果①: val(データはまだ productive・ただし逓減)
| 世代 | データ | best_ep | best_val | val_corr | val_mae | train@best | gap@best |
|---|---|---|---|---|---|---|---|
| 061 champion | 8M | 8 | 114530 | 0.8577 | 362.9 | 65050 | 49480 |
| 062 champion | 16M | 5 | 103318 | 0.8751 | 336.4 | 76434 | 26884 |
| **本タスク** | **32M** | **6** | **95358** | **0.8862** | **317.9** | 69703 | **25655** |

→ **データ倍増で val なお改善**: val_corr **0.8751→0.8862(+0.0111)**、val_loss **−7.7%**、val_mae 336.4→317.9。
**ただし逓減**: 8→16M は val_corr +0.0174 / val_loss −10%、16→32M は +0.0111 / −7.7%。
**過学習 gap は plateau**: 49480→26884→**25655**(8→16 で半減したが 16→32 はほぼ不変)。
**署名の変化**: 8→16M は train 上昇(65k→76k)で val 低下=「丸暗記しきれない」data律速。16→32M は **train も低下(76434→69703)しつつ val も低下**し、gap ~26k で動かない
=2× データで曲線全体が ~7k 下方シフトするが **gap 底(~26k)は縮まない**=この gap 底は **技巧2 depth8 ラベルの不可避ノイズ(教師の天井)**。val_corr 0.8862 は天井 ~0.90 に接近=蒸留で詰められる絶対量は縮小中。

## 結果②: 確証(eval-vs-eval SPRT, 探索器=YaneuraOu 固定, movetime 100ms)
両側 0cp ゲート通過(32M eval: engine `e`==binbo Mode::Quant==int-faithful=**32/32**, sha256 `3c25685e…`)。W/D/L は 32M 視点、相手=champion(`task062_16m_lr1e-5`, sha256 `b0aa6173…`)。

| 候補 | 相手 | 戦績(W-L-D) | score(CI95) | elo(CI95) | draw率 | SPRT |
|---|---|---|---|---|---|---|
| **32M**(lr1e-5) | champion(16M, `task062_16m_lr1e-5`) | 138-61-1 / 200 | **69.2%**(62.9–75.6) | **+141.0**(91.5–196.7) | 0.5% | continue(LLR2.152) |

→ **データ倍増は同一探索で +141 elo(score CI 下限 62.9%=決定的)**。SPRT は elo1=+10 閾値を 200局で形式未到達(LLR2.152)だが、これは
061/062 と同じ挙動(trinomial+小閾値は大差でも形式到達が遅い)で、**実質証拠は score CI**(69.2%, 下限 62.9%)=改善は明白。
速度は両側同一探索ゆえ拮抗(cand depth 15.54 ≈ base 15.20、~130k node 同等)=差は **eval のみ**。match ~37分(200局)。

## 採用 / 頭打ち判定 — **逓減は val だけ、棋力はまだ伸びる**
- **採用**: **`task064_32m_lr1e-5` を新 baseline 世代に昇格**。val 改善(corr 0.8751→0.8862)+ 対局 **+141 elo**(決定的)= pragmatic 基準を十分満たす。
  eval: `models/task064_32m_lr1e-5/nnue_best.{pt,bnnq}`、`eval_dirs/task064_32m_lr1e-5/nn.bin`(sha256 `3c25685e…`)。非コミット・永続。
- **頭打ち判定 = NO(まだ頭打ちでない)**。だが **val と棋力で逓減カーブが食い違う**のが本タスクの核心:
  - **val は逓減**(+0.0174→+0.0111、gap 底 ~26k で停止、天井 0.90 接近)=**蒸留の情報量は枯れつつある**。
  - **棋力は逓減せず**: 8→16M **+125**、16→32M **+141**(点推定はむしろ上、CI 重複=同程度)。**盤上ではデータがまだ強く効く**(val の小改善が桁違いの elo に化ける 061 以来の現象)。
- **逓減カーブ(64M 判断材料)**: val 基準なら 64M は天井 0.90 で頭打ち濃厚=旨味小。棋力基準なら 64M も +100 前後の見込みは残る。
  **ただし gap 底(教師ノイズ)が動かない**以上、蒸留だけの伸びには上限がある。**戦略判断: 蒸留の情報限界が見えた今、自己強化ループ(教師自体を上げる)の価値が最も高い**(BACKLOG 最上位)。64M は逓減覚悟の任意 follow-up。

## 回帰
- **Rust 無改変**(gen-teacher/prepare_data/train_nnue/`export_nnue`/`binbo_to_yane`/`eval_vs_eval.sh`/`yane-vs-yane` は全て既存流用)→ `cargo test` 24 green は自明(確認済)。perft 不変。
- 一変数=データ量(lr/net/teacher 質/057-059 bridge/Rust 探索 不触)。tied QAT 維持=engine 0cp(ゲートで実証)。
- **運用メモ**: 学習を `detached.sh` 経由で起動する際、`--qat-tie` の collapse-map(`trainer.binbo_to_yane` 内の `from trainer import kp`)解決のため `env PYTHONPATH=<repo>` を前置(または `-m trainer.train_nnue`)。コード変更なし=起動時の path 設定のみ。

## 再現 / 固定パラメータ
- 生成: `gen-teacher $GIKOU run16m_f 16000000 --depth 8 --random-plies 6 --max-plies 256 --shard-size 10000 --workers 16 --seed 20290629 --timeout-ms 30000`(resumable, `teacher/run16m_f.gen.sh` + `scripts/detached.sh gen32m`)。
- cache: `prepare_data run1m caches/shared_32m --canonical-shared --val-shards 5 --extra-runs run1m_b run2m_c run4m_d run8m_e run16m_f`(n_train 31.95M, val=同一50k)。
- 学習: `train_nnue caches/shared_32m --qat --qat-tie --feature-dim 256 --hidden-dim 32 --lr 1e-5 --epochs 12 --batch-size 1024 --scale-cp 600 --augment-mirror --mirror-cache <8m perm>`(best=epoch6, `scripts/detached.sh train32m env PYTHONPATH=<repo>`)。
- 確証: `scripts/eval_vs_eval.sh task064_32m_lr1e-5 eval_dirs/task062_16m_lr1e-5 nnue_best.pt 100 200 20260629 6`(movetime 100ms, SPRT elo0=0/elo1=+10/α=β=0.05, max_games 200)。
- ハード: Intel Core **i7-14700KF**(gen 16 workers, **WSL RAM 22GB**=.wslconfig で 15→22 引上げ)、RTX 4060 Ti(学習)、YaneuraOu 9.41git 64AVX2。disk: ext4 vhdx(/dev/sdd 760G avail / C: 336G free)=非制約。
- eval: `models/task064_32m_lr1e-5/nnue_best.{pt,bnnq}`、`eval_dirs/task064_32m_lr1e-5/nn.bin`(sha256 `3c25685e…`)。非コミット・永続。

## やらないこと / 次タスク(BACKLOG)
- **64M+ へのさらなる増量**(逓減覚悟の任意 follow-up=val は天井、棋力は残るが gap 底は不動)。**教師質の変更**(depth↑/別エンジン)。
- **自己強化ループ**(踏み台完成後の本命=本タスクで蒸留の情報限界が可視化): 強化蒸留 eval を初期化点に、ラベル元を自前 eval+深い探索へ→世代ごと SPRT 採用。
- Rust 探索の差分更新化+SIMD(060: 探索差の主因, 第二路線)。
