# Task 061 — 技巧2蒸留 eval 強化(踏み台を高くする)+ 世代比較の検証ループ

**ゴール**: 自己強化ループ前の**踏み台**として、教師(技巧2蒸留8M)据え置きで**学習側だけ**を改善し、本命 256 self-eval を
できるだけ強くする。**主眼=「何が効いたか」が一変数ずつ分かる検証ループ**: 一次指標 val、確証は **eval-vs-eval SPRT**
(探索器=YaneuraOu を固定、eval だけ差し替え= Task 060 の逆軸)。ネットは 256x2-32-32 固定(writer/engine 確証の制約)、
教師・データ量・057-059 bridge・Rust 探索は不触。

## 検証ループ(新規ハーネス)
- **`yane-vs-yane`**(`src/main.rs`, 060 infra 再利用): YaneuraOu を 2 起動(別 EvalDir, 同一探索)→ **eval だけが変数**。色交替・
  ペアopening・千日手・trinomial SPRT(`src/sprt.rs`, H1=候補が強い)。回帰: `cargo test` 24 green・perft 1-5 不変・既存 `nnue-vs-yane`/`nnue-vs-gikou` 不変。
- **driver `scripts/eval_vs_eval.sh`**: 候補 model dir → export(BNNQ)→ writer(nn.bin)→ **0cp ゲート**(engine `e`==binbo Mode::Quant==int-faithful, RED で abort)→ `yane-vs-yane` 対局。
- **trainer**: 既存 `train_nnue.py --qat --qat-tie`(`--weight-decay`/`--dropout`/`--lr`/`--epochs`)+ 最小追加 `--lr-schedule {none,cosine,step}`(default none=従来不変)。

## 学習側チューニング(一変数, 一次=val)— baseline = Task 060 本命 `model_qat_tie_fd256_8m`(lr1e-3, 8ep)
| 軸 | 変更 | best_ep | best_val | val_corr | val_mae | 判定(val) |
|---|---|---|---|---|---|---|
| **baseline** | lr1e-3 e8 | 1 | 124494 | 0.8334 | 377.3 | 基準 |
| 正則化 | weight_decay 1e-4 | 1 | 126326 | 0.8312 | 381.6 | 悪化 |
| 正則化 | weight_decay 1e-3 | 1 | 126837 | 0.8301 | 382.2 | 悪化 |
| 正則化 | dropout 0.1 | 1 | 128203 | 0.8272 | 384.9 | 悪化(late overfit のみ緩和) |
| lr schedule | cosine lr1e-3 e12 | 1 | 124494 | 0.8334 | 377.3 | **変化なし** |
| **lr level** | lr 3e-4 e10 | 1 | 122234 | 0.8389 | 373.7 | 改善 |
| **lr level** | lr 2e-4 e12 | 1 | 120389 | 0.8429 | 372.1 | 改善 |
| **lr level** | lr 1e-4 e15 | 1 | 118256 | 0.8489 | 368.0 | 改善 |
| **lr level** | lr 5e-5 e12 | 3 | 117822 | 0.8493 | 367.5 | 改善 |
| **lr level** | lr 3e-5 e12 | 3 | 115224 | 0.8541 | 363.6 | 改善 |
| **lr level** | lr 1e-5 e18 | 8 | **114530** | **0.8577** | 362.9 | **最良(floor 付近)** |

**診断**:
- **正則化(weight_decay / dropout)は効かない**: best_val はむしろ悪化、best_epoch は依然 1。dropout は late overfit(final_val)だけ緩和。
- **cosine schedule(lr1e-3 起点)も効かない**: epoch 1 で既に val 最適に達し、その後 val は単調増加。decay は overfit 状態を凍結するだけ
  (epoch 1 の高 lr の overshoot が原因で、後段 decay では回復しない)。
- **唯一の lever = lr の水準**: lr を下げると val 単調改善(124494→114530, corr 0.8334→0.8577)。lr が高いほど epoch 1 で val 最適を
  overshoot していた=**Task 060 本命は lr が高すぎて過学習**だった。lr を下げると best_epoch が後ろへ(1→8, 収束が緩やか)、改善は
  lr 1e-5 付近で頭打ち(floor)。

## 確証(eval-vs-eval SPRT, 探索器=YaneuraOu 固定, movetime 100ms)
両側 eval は **0cp ゲート通過**(engine `e`==binbo Mode::Quant==int-faithful=32/32)。W/D/L は候補視点。

| 対局 | 候補 | 相手 | 戦績(W-L-D) | score(CI95) | elo(CI95) | draw率 | SPRT |
|---|---|---|---|---|---|---|---|
| 確証① | lr1e-4 | baseline(lr1e-3) | 148-44-2 / 194 | 76.8%(70.9–82.7) | **+208**(155–272) | 1.0% | accept_h1 |
| 確証② | lr3e-5 | baseline(lr1e-3) | 140-35-0 / 175 | 80.0%(74.1–85.9) | **+240.8**(182–314) | 0.0% | accept_h1 |
| 確証③(採用) | **lr1e-5** | baseline(lr1e-3) | 149-50-1 / 200 | 74.8%(68.7–80.8) | **+188.5**(137–249) | 0.5% | continue(LLR2.794≈上限) |
| 決勝(head-to-head) | lr1e-5 | lr3e-5 | 111-89-0 / 200 | 55.5%(48.6–62.4) | +38.4(−9.6–87.9) | 0.0% | continue(未決) |

→ **lr を下げた eval は 3 候補とも同一探索で baseline に圧勝**(+188〜+241 elo, draw ほぼ 0)。3 候補の vs-baseline elo は **CI が大きく重複(~+137〜+314)
=互いに統計的区別不可**(head-to-head lr1e-5 vs lr3e-5 も +38 で有意差なし)。確証③の SPRT は elo1=+10 の上限直前(LLR2.794)で 200 局打ち止めだが、
score 74.8%(CI 下限 68.7%)・elo 下限 +137 で**改善は決定的**(SPRT の形式的到達は標本揺らぎ、実質証拠は CI)。
**val の小改善(corr 0.833→0.858)が桁違いの elo 差**になるのは、baseline が過学習で**重要局面の評価を memorize 由来で誤り**、その失点が
aggregate な val_corr に表れにくいため(corr は易しい局面が支配的)。速度は両側同一探索ゆえ拮抗(cand depth ≈ base depth, ~同 node)=差は **eval のみ**に帰属。

## 採用
- **champion = `task061_lr1e-5_e18`**(lr 1e-5, best_epoch 8 / 18ep, val_corr **0.8577** / best_val 114530)。一次=最良 val、確証=baseline に
  **+188.5 elo(74.8%, CI +137–249)**、head-to-head で lr3e-5 に非劣化(55.5%)。pragmatic 基準(val 明確改善 + SPRT 非劣化)を満たす → **採用**。
- **新 baseline 世代に昇格**: 以後の世代比較・自己強化ループの初期化点は本 eval。eval: `models/task061_lr1e-5_e18/nnue_best.{pt,bnnq}`、
  nn.bin `eval_dirs/task061_lr1e-5_e18/nn.bin`(sha256 `559cebd1…`、BNNQ `dff0190d…`)。非コミット・永続。
- **踏み台は約 +190〜240 elo 上がった**(Task 060 本命比, 同一探索・同一教師・同一 arch=**学習 lr の修正だけ**で)。次の lever は**データ量増量**(BACKLOG)。

## 回帰(1変数の担保)
- Rust: `cargo build/test --release` = **24 green**(`yane-vs-yane` 追加, `sprt`/`usi_engine` 再利用・不変)、perft 1-5 不変、
  既存 `nnue-vs-yane`/`nnue-vs-gikou`・float modes 不変。**探索・057-059 bridge 不触**(追加ハーネスは探索に非干渉)。
- trainer: `--lr-schedule none`(default)で従来と数値同一。tied QAT 維持=候補は全て **engine 0cp 書出可**(ゲートで実証)。

## 再現 / 固定パラメータ
- 学習: `train_nnue.py <cache/shared_8m> --qat --qat-tie --feature-dim 256 --hidden-dim 32 --batch-size 1024 --scale-cp 600 --augment-mirror --lr <LR> --epochs <E>`。
- 確証一発: `scripts/eval_vs_eval.sh <候補 model dir> <baseline eval_dir> [weights] [movetime_ms] [max_games]`(export→writer→0cp ゲート→`yane-vs-yane`)。
- time control: movetime **100ms/手**、`go btime 0 wtime 0 byoyomi 100`、SPRT elo0=0/**elo1=+10**/α=β=0.05、opening_plies=6、max_plies=320、max_games=200、seed=20260622。
- 教師: `caches/shared_8m`(技巧2 depth8 cp, 7.95M train/50k val)。ハード: Intel Core **i7-14700KF**、YaneuraOu 9.41git 64AVX2(halfkp_256x2-32-32)、GPU 学習。
- per-game TSV: `~/binbo-shogi-data/logs/task061_*.tsv`(非コミット)。候補重み/BNNQ/nn.bin: `~/binbo-shogi-data/models|eval_dirs/`(非コミット・永続)。

## やらないこと / 次タスク(BACKLOG)
- **データ量増量**(本タスク非対象): 新規 gen-teacher 走 → 16M+。学習側(lr)が floor に達した今、**次の最有力 lever**。
- **自己強化ループ**(本命): 強化した蒸留 eval を初期化点に、ラベル元を自前eval+深い探索へ→世代ごと SPRT 採用。踏み台が高くなった今、着手可。
- Rust 探索の差分更新化+SIMD(060: 探索差の主因)。完全自己完結教師 / 提出フェーズの大会ルール原文再確認。
