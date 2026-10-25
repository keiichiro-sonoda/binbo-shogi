# Task 062 — 技巧2蒸留 教師データ増量(8M→16M): データ量を一変数に

**ゴール**: Task 061 で lr を引き切った後の残るレバー=**教師データ量**。fd256 は 8M でまだ過学習(train ≪ val=データ律速)。
教師質(技巧2 depth8)・lr・ネット 256x2-32-32 を据え置き、**データ量だけ**を 8M→16M に倍増し、過学習が埋まって val/棋力が伸びるか測る。
**一変数=データ量**。裏目的=「蒸留の頭打ち確認 → 自己強化ループ着手の判断」。

## 何をしたか(既存パイプライン流用, 新規コードなし)
- **追加 8M 生成**(`gen-teacher`, 技巧2 depth8): 新 seed **20270627**(既存 8M の 20260621 と別=新規の異なる局面)、random_plies 6 / max_plies 256 /
  shard 10000 / **24 workers** / timeout 30s。resumable。実測 **depth8 ≈ 1.38ms/pos**、**+8M を ~70分**(09:39→10:49)で生成、hcpe 1.6GB。
- **16M cache**(`prepare_data --canonical-shared`): primary=`run1m`(val=末尾5 shard)+ extra-runs=`run1m_b run2m_c run4m_d run8m_e`。
  → **n_train 15,950,000 / n_val 50,000**。**val は shared_8m と完全同一**(同 run1m・同 shard 95-99・同 50k=世代間で公平)。cache 4.7GB。
- **学習**(061 champion 設定固定, データのみ変数): `train_nnue --qat --qat-tie --feature-dim 256 --hidden-dim 32 --lr 1e-5 --augment-mirror`
  (mirror perm=feature 空間=8M のを再利用)。disk/RAM 非制約。
  - **注**: 学習は session teardown で epoch 6 終了時に中断。だが **best-val(epoch 5)checkpoint は保存済**、かつ val 曲線は epoch 6 で既に反転
    (上昇)=epoch 5 が最適。`model_config.json` は arch 由来のみ(lr/epoch/データ量は不載)で champion と**バイト一致**=再構築して export(fudge でない)。

## 結果①: val(データで過学習が埋まった=データ律速の確認)
| 世代 | データ | best_ep | best_val | val_corr | val_mae | train@best | gap@best |
|---|---|---|---|---|---|---|---|
| champion(061) | 8M | 8 | 114530 | 0.8577 | 362.9 | 65050 | 49480 |
| **本タスク** | **16M** | 5 | **103318** | **0.8751** | **336.4** | 76434 | **26884** |

→ **データ倍増で val 明確改善**: val_corr **0.8577→0.8751(+0.0174)**、val_loss **−10%**、val_mae 362.9→336.4。
**train-val gap が 49480→26884 に半減**=過学習がデータで埋まった=**「fd256 は 8M では data律速」の仮説を実証**。
特徴的なのは **train_loss がむしろ上昇(65050→76434)しつつ val が低下**したこと=2× のデータは丸暗記しきれない→**容量が generalization に回った**(data律速の教科書的シグネチャ)。
(過去のデータ頭打ちは小器 fd8 の容量飽和。大器 fd256 では容量に余裕があり、データが効く、という見立てどおり。)
16M の val 曲線: e1 120244 → e3 107515 → **e5 103318(最小)** → e6 104027(反転)。mirror=on でもなお data で伸びた=対称水増しとは別の新情報。

## 結果②: 確証(eval-vs-eval SPRT, 探索器=YaneuraOu 固定, movetime 100ms)
両側 0cp ゲート通過(16M eval: engine `e`==binbo Mode::Quant==int-faithful=**32/32**, sha256 `b0aa6173…`)。W/D/L は 16M 視点、相手=champion(`task061_lr1e-5_e18`, sha256 `559cebd1…`)。

| 候補 | 相手 | 戦績(W-L-D) | score(CI95) | elo(CI95) | draw率 | SPRT |
|---|---|---|---|---|---|---|
| **16M**(lr1e-5) | champion(8M, `task061_lr1e-5_e18`) | 134-65-1 / 200 | **67.2%**(60.8–73.7) | **+125.0**(76–179) | 0.5% | continue(LLR1.919) |

→ **データ倍増は同一探索で +125 elo(score CI 下限 +76=決定的)**。SPRT は elo1=+10 閾値を 200局で形式到達せず(LLR1.919)だが、これは
061 の大勝と同じ挙動(trinomial+小閾値は大差でも形式到達が遅い)で、**実質証拠は score CI**(67.2%, CI 下限 60.8%)=改善は明白。
速度は両側同一探索ゆえ拮抗(cand depth 15.41 ≈ base 15.09)=差は **eval のみ**。

## 採用 / 頭打ち判定
- **採用**: **16M(`task062_16m_lr1e-5`)を新 baseline 世代に昇格**。val 明確改善(corr 0.8577→0.8751)+ 対局 **+125 elo**(決定的)= pragmatic 基準を十分満たす。eval: `models/task062_16m_lr1e-5/nnue_best.{pt,bnnq}`、`eval_dirs/task062_16m_lr1e-5/nn.bin`(sha256 `b0aa6173…`)。非コミット・永続。
- **頭打ち判定 = NO(まだ頭打ちでない)**: 8M→16M で **+125 elo / val_corr +0.017** = データはまだ productive。**32M も見込みあり**(逓減は想定)。
  ただし val_corr 0.8751 は**技巧2教師の天井(~0.90)に接近中**=蒸留だけで伸ばせる余地は縮小しつつある。
- **次の判断**: (a) 32M へさらに増量(逓減だが正の見込み)/ (b) **自己強化ループ着手**(踏み台が十分高くなった=本命)。蒸留の天井が近いことを踏まえると、(b) の戦略的価値が高い(BACKLOG 最上位)。

## 回帰
- **Rust 無改変**(gen-teacher/prepare_data/train_nnue/`eval_vs_eval.sh`/`yane-vs-yane` は全て既存流用)→ `cargo test` 24 green は自明(確認済)。perft 不変。
- 一変数=データ量(lr/net/teacher 質/057-059 bridge/Rust 探索 不触)。tied QAT 維持=engine 0cp(ゲートで実証)。

## 再現 / 固定パラメータ
- 生成: `gen-teacher $GIKOU run8m_e 8000000 --depth 8 --random-plies 6 --max-plies 256 --shard-size 10000 --workers 24 --seed 20270627 --timeout-ms 30000`(resumable)。
- cache: `prepare_data run1m caches/shared_16m --canonical-shared --val-shards 5 --extra-runs run1m_b run2m_c run4m_d run8m_e`。
- 学習: `train_nnue caches/shared_16m --qat --qat-tie --feature-dim 256 --hidden-dim 32 --lr 1e-5 --epochs 12 --batch-size 1024 --scale-cp 600 --augment-mirror --mirror-cache <8m perm>`。
- 確証: `scripts/eval_vs_eval.sh task062_16m_lr1e-5 <champion eval_dir>`(movetime 100ms, SPRT elo0=0/elo1=+10/α=β=0.05, max_games 200, seed 20260622)。
- ハード: Intel Core **i7-14700KF**(gen 24 workers)、RTX 4060 Ti(学習)、YaneuraOu 9.41git 64AVX2。disk `/mnt/c` 321G free=非制約。
- eval: `models/task062_16m_lr1e-5/nnue_best.{pt,bnnq}`、`eval_dirs/task062_16m_lr1e-5/nn.bin`(sha256 `b0aa6173…`)。非コミット・永続。

## やらないこと / 次タスク(BACKLOG)
- **32M+ へのさらなる増量**(本結果次第=逓減なら打ち止め)。**教師質の変更**(depth↑/別エンジン)。
- **自己強化ループ**(踏み台完成後の本命): 強化蒸留 eval を初期化点に、ラベル元を自前 eval+深い探索へ→世代ごと SPRT 採用。
- Rust 探索の差分更新化+SIMD(060: 探索差の主因, 第二路線)。
