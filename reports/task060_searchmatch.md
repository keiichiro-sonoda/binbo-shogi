# Task 060 — 本命 256 self-eval 本走 + 探索器比較対局(同一 eval × 自前 Rust 探索 vs YaneuraOu 探索)

**ゴール**: 057(読込)/058(量子化 forward)/059(書出)で閉じた「自前 eval を実エンジンに 0cp で載せる」三辺を**初めて本走で使う**。
技巧2蒸留の自前教師(大会合法ルート, GPL・事前生成)で **本命 256x2-32-32 eval** を tied QAT 学習し、**同一 eval** を
① 自前 Rust 探索 と ② YaneuraOu 探索 の両方に **bit-exact 同一**で載せて対戦させ、**洗練された探索の力**を測る。
**比較の変数は探索器のみ**(eval は両側で固定)。

## 何をしたか(1変数。eval は両側同一・057-059 bridge 不触)
- **Part 1 — binbo 探索が BNNQ(Mode::Quant)eval を正しいスケールで使う**(必須): `Mode::Quant` の `forward()` は **cp を直接返す**
  (float BNN1〜6 は logit×600)。`Nnue::is_quant()`(`src/eval/nnue.rs`)+ `Searcher::leaf_eval`(`src/search.rs`)を mode-aware に追加し、
  3 leaf 呼び出しを置換。**float modes は数値不変**(`eval_to_int` 温存)。→ binbo leaf cp == YaneuraOu leaf cp(058 bit-exact)。
- **Part 2 — 探索器比較ハーネス `nnue-vs-yane`**(`run_nnue_vs_gikou` を雛形): 内部 binbo 探索 vs 外部 YaneuraOu(汎用 USI driver
  `src/usi_engine.rs`)。色交替・ペア opening・千日手(key≥4)・W/D/L・per-side depth/nodes/time は流用。**SPRT 早期停止**は新規
  `src/sprt.rs`(trinomial/BayesElo: drawelo を観測から推定し W/D/L を確率モデル化、`LLR=Σ log P_H1/P_H0`、Wald 境界で打ち止め)。
  YaneuraOu options: **EvalDir / USI_OwnBook=false / Threads=1 / MinimumThinkingTime=0 / NetworkDelay=0 / NetworkDelay2=0**(0.1s 尊重)。
- **Part 3 — 本命 256 学習 + 書出 + 対局前ゲート**: `train_nnue.py --qat --qat-tie`(fd256/h32, **full 8M**, mirror, cp 目標, scale_cp 600,
  best-val 保存)→ `export_nnue`(auto tie)→ BNNQ → `binbo_to_yane`(059 writer)→ nn.bin。**対局前ゲート**(`scripts/binbo_to_yane.sh`)で
  engine `e` == binbo Mode::Quant == int-faithful = **0cp** を確認してから対局(ズレたら棋力差の切り分け不能)。driver `scripts/selfeval_vs_yane.sh`
  が export→writer→**ゲート(RED なら abort)**→match を一発。

## 本命 256 eval(full 8M, tied QAT, cp 目標)
| | best_epoch | val_loss | val_corr | val_mae | decisive_sign_agree |
|---|---|---|---|---|---|
| **本命 fd256 8M tied**(本タスク) | **1** | 124494 | **0.833** | 377cp | 0.889 |
| 参考: fd256 2M tied(059) | 1 | 160055 | 0.716 | — | 0.822 |

→ full 8M で corr 0.716→**0.833** と明確に向上。**epoch 1 以降は過学習**(train_loss 97k→23k と下がり続ける一方 val_loss 124k→157k と単調増加、
gap 27k→135k)。best-val 保存が epoch 1 を採用(早期停止が効くべきだった=次回 epochs を絞る論拠)。重みは epoch 1 checkpoint。

## 結果

### 対局前ゲート(両側 eval bit-exact 同一, 必須)
`tools/parity_positions.sfen`(32局面)、本命 BNNQ → writer → nn.bin → 実エンジン:

| 判定 | 内容 | 結果 |
|---|---|---|
| A | engine `e`(written) == int-faithful(`yane_forward_int`, written) | **32/32 exact, 0cp** |
| fid | binbo `nnue-eval`(Mode::Quant, src) == engine `e`(written) | **32/32 exact, 0cp** |

→ **両側の eval は完全に同一**。これを通してから対局=結果の差は**探索器のみ**に帰属する。

### 探索器比較対局(SPRT, movetime 100ms)
SAME eval、binbo 内部探索 vs YaneuraOu 探索。SPRT(H0: elo差0 / **H1: binbo ≥50 elo 弱い** / α=β=0.05):

| 指標 | 値 |
|---|---|
| **SPRT 判定** | **accept_h1**(19局で到達, LLR=3.027 ≥ 上限 2.944) |
| binbo 戦績(W-L-D) | **0-19-0**(全局 binbo が詰まされ負け) |
| 結論 | **YaneuraOu の探索は binbo の探索より ≥50 elo 強い(p<0.05)**。実際は 19/19 で桁違い。 |

**注**: 勝率 0/19 のため elo 点推定は clamp 飽和(−2400, CI [0,0])=厳密値でなく下界。**頑健な結論は SPRT 判定**(≥50 elo 弱い)+ 戦績 0/19。

### 速度差(同一 eval・同一 movetime での「総合探索力」)
| | avg_depth | max_depth | avg_nodes | max_nodes | avg_ms |
|---|---|---|---|---|---|
| **binbo**(自前 Rust 探索) | 3.81 | 7 | 34,558 | 62,527 | 78.9 |
| **YaneuraOu**(SIMD+差分更新) | **15.94** | **39** | **120,632** | 164,569 | 89.6 |

→ 同一 movetime・同一 eval で YaneuraOu は **約 4.2× の深さ / 約 3.5× の node**。これが 0-19-0 の正体=
**実装速度込みの探索器総合力**の差。純アルゴリズム分離は fixed-nodes が筋(任意 follow-up)。

## 回帰(1変数の担保)
- `cargo build --release` clean、`cargo test --release` = **24 green**(新規 `sprt::tests` 2 本含む)。
- **perft 1-5 不変**(30/900/25470/719731/19861490)。**float modes 数値不変**(`eval_to_int` 温存=既存 `nnue-vs-gikou`/`nnue-*match` 不変)。
- ゲート 32/32 0cp(両側 eval 同一)。**057-059 bridge 不触**。

## 再現
- 一発: `bash scripts/selfeval_vs_yane.sh model_qat_tie_fd256_8m nnue_best.pt 100 60`(export→writer→**0cp ゲート**→SPRT 対局)。
- 学習: `train_nnue.py caches/shared_8m --qat --qat-tie --feature-dim 256 --hidden-dim 32 --epochs 8 --batch-size 1024 --scale-cp 600 --augment-mirror`。
- 対局単体: `binbo-shogi nnue-vs-yane <bnnq> <yane_bin> <eval_dir> [movetime_ms] [seed] [max_games] [opening_plies] [max_plies] [depth] [elo0] [elo1] [alpha] [beta]`。

### 固定パラメータ(再計算用)
- eval: `model_qat_tie_fd256_8m/nnue_best.pt` → BNNQ sha256 `7e107cb58028febd5b86f0503c759a1f1f1ab1ee4208d329b67d7048c483444d`、
  nn.bin sha256 `3ce3eac6aabe7546fc8983dca27803145df142da00c5c038ceb7ae33d80ec058`。
- time control: movetime **100ms/手**(binbo は budget の 500ms margin を足し戻し)、YaneuraOu `go btime 0 wtime 0 byoyomi 100`。
- SPRT: elo0=0 / **elo1=−50** / α=β=0.05、opening_plies=6、max_plies=320、depth_cap=24、seed=20260622、max_games=60(19 で打ち止め)。
- per-game TSV: `~/binbo-shogi-data/logs/task060_model_qat_tie_fd256_8m_mt100.tsv`(elo/CI/SPRT 再計算用、非コミット)。
- ハード: Intel Core **i7-14700KF**(28 threads, 1 thread/engine)、YaneuraOu NNUE 9.41git 64AVX2(halfkp_256x2-32-32)、binbo `target/release`(avx2/bmi2)。

## やらないこと / 次タスク(BACKLOG)
- **fixed-nodes 比較**(純アルゴリズム分離。実装速度を外して探索アルゴリズムだけを測る)= 本結果が「速度込み総合力」なので筋の follow-up。
- **完全自己完結教師**(外部エンジン非依存=自前 eval self-play ラベル)。256 本命の棋力チューニング本格化(LR/epoch/正則化; 過学習対策)。
- 探索アルゴリズムの差し替え(binbo 側を強化)。提出フェーズで大会ルール原文の再確認。
