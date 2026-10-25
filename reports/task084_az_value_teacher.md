# Task 084 — AZ value 教師上書きの検証(野田式): AZ value は技巧2@d8 探索cp より良い NNUE 教師か

**目的**: 「技巧2 を超える」を binbo 本体(NNUE)に戻し、**教師の cp の出所だけ**を替える。技巧2@d8 探索cp を (c) の AZ ネット(gen100-128 / WS2-128)の静的 value で上書きし、同一 cache・同一設定で NNUE を学習、3系統を eval-vs-eval SPRT で相対比較。先行= nodchip 氏(NNUE 教師の探索cp を DL value で上書き→+R100)。**判定は (i) vs (ii)/(iii) の相対**(2M ≪ champion 32M ゆえ champion 超えは非対象)。実測 2026-07-10、隔離環境(az_rescore.py 非コミット)、**リポジトリは文書のみ・Rust 無改変**(yane-vs-yane は既存バイナリ)。**但し書き(055)**: 我々の AZ value は技巧2@d8 蒸留物=情報量は技巧2 を超えない。本質は「技巧2@d8 の noisy cp を AZ で denoise すると生徒 NNUE に効くか」。055 で dlshogi 静的 value は技巧2 探索cp より悪い教師だった前例あり=(i) 有利の事前確率。

## パイプライン(既存ツール流用 + az_rescore.py 1本のみ新規)
- **cache**: `prepare_data run2m_c --canonical-shared --val-shards 4` = train **1.96M** / val 40k、77 features(champion=tied QAT と同レイアウト)、58s。
- **AZ 上書き** `az_rescore.py`(dlshogi_rescore の cache-aligned 版、AZ 推論に差替): hcp→sfen→`_observe(flip=True)`→AZ forward value→**cp=1200·atanh(v) clip±5000**。**gikou-eval==cache の alignment assert 全 split 通過**(055 の順序事故を排除)。throughput ~11k/s(2M ~173s)。出力=`{train,val}_dlshogi_evals.npy`(train_nnue --eval-npy が読む名)。
- **学習**: 既存 `train_nnue --qat --qat-tie --feature-dim 256 --hidden-dim 32 --lr 1e-5 --epochs 12 --batch 1024 --scale-cp 600`(objective=cp)。3系統で同一、**差分は --eval-npy のみ**(= 教師 cp の出所)。mirror は 3系統共通の定数ゆえ省略(相対比較に無影響、と明記)。GPU 57s/epoch。
- **SPRT**: `eval_vs_eval.sh`(export→0cp ゲート→yane-vs-yane trinomial, movetime 0.1s)。

## ★上書きラベルの素性(教師としての一次指標)
| 教師 | cp std | 技巧2 との corr | |Δcp| median |
|---|---|---|---|
| (i) 技巧2@d8 探索cp(clip) | **1778** | — | — |
| (iii) WS2-128 value | 1471 | **0.942** | 248 |
| (ii) gen100 value | **698** | 0.896 | 365 |
- **WS2-128 は技巧2 に忠実**(std 1471≈1778、corr 0.942)=技巧2 の denoise 再構成。**gen100 は強く圧縮**(std 698、corr 0.896)=self-play ループで value が tanh 飽和側へドリフト(loop 中 corr_m2 0.905→0.83 と整合)。→ **(c) ループは gen100 を「強いプレイヤー」にしたが value は「平坦で decisive さの薄い」教師に変えた**。内部棋力↑ ≠ 教師品質、の予兆。

## 3系統の学習(同一 cache/設定, GPU 57s/epoch。val は各系統の自ラベル比=系統間非比較)
| 系統 | 教師 cp | best epoch | val_corr(自ラベル比) | val_mae |
|---|---|---|---|---|
| (i) | 技巧2@d8 探索cp | 7 | 0.752 | 471 |
| (ii) | gen100 value | 9 | 0.869 | 220 |
| (iii) | WS2-128 value | 12 | 0.858 | 327 |
- val_corr は**上書き先の自ラベルに対する値=系統間で比較不可**(gen100 の 0.869 は圧縮ラベルが fit 容易なだけ=強さでない)。棋力判定は下の SPRT。

## 相対 SPRT(eval のみ変数・探索固定・両側 0cp ゲート・movetime 0.1s、cap 300, 全 GATE PASSED)
| 対戦(cand vs base) | W-L-D | score | elo(CI95) | verdict |
|---|---|---|---|---|
| **(ii) gen100 vs (i) 技巧2** | 169-129-2 | 56.7% | **+46.6**(+7.5..+86.9) | continue@300(CI が 0 を除外=有意勝ち) |
| **(iii) WS2-128 vs (i) 技巧2** | 141-36-0 | 79.7% | **+237.2**(+179..+310) | **accept_h1@177(決定的勝ち)** |
| (ii) gen100 vs (iii) WS2-128 | 76-175-0 | 30.3% | **−144.9**(−195..−100) | accept_h0(gen100 が有意劣) |
- 推移律整合: **WS2-128(+237)> gen100(+47)> 技巧2(0)**。

## 判定 = **AZ value は技巧2@d8 探索cp より良い NNUE 教師(決定的)— ただし「良い教師」は WS2-128、ループは教師品質を悪化させた**
- **主判定 PASS(055 の事前予測に反する)**: (ii)(iii) とも (i) に有意勝ち。**WS2-128 value 教師は技巧2@d8 探索cp より +237 elo 良い**。nodchip の「DL value で探索cp を denoise すると NNUE が強くなる」が我々でも成立。
- **なぜ 055 と逆か**: 055 の dlshogi value は**別エンジンの判断**(生徒の技巧2ラベル局面と分布不一致)。我々の AZ value は**同じ技巧2@d8 信号の neural 平滑化**(分布一致・純 denoise)=noisy な探索cp(局面ごと ±jitter・重い裾)を well-behaved な回帰ターゲットへ整形。→ (i) は noisy ラベルを fit しきれず(val_corr 0.752)汎化悪、(iii) は平滑ラベルを良く fit(0.858)し eval も良。
- **★ループの逆説(内部棋力↑ ≠ 教師品質↑)**: **WS2-128(ループ前)が最良教師、gen100(ループ後)は −145 elo 劣**。(c) ループは gen100 を**強いプレイヤー**にしたが value を **std 698 の圧縮=decisive さの薄い平坦な教師**に変えた。棋力と教師品質は別軸。

## 申し送り
- **★champion 超えの道が再開**: WS2-128 value 教師が 技巧2@d8 に +237(2M 相対)=**全量 32M を WS2-128 value で上書き→champion 構成で本走→32M 技巧2 champion(task064)超えを狙う**。rescore は ~11k/s(32M ~48分)。066-072 の「技巧2@d8 天井」は**教師を替えれば動く**と実証された。
- **教師は WS2-128 を使う**(gen100 でない=ループが教師品質を落とした)。留保: WS2-128 value は技巧2@d8 蒸留物ゆえ**新情報でなく denoise**=「denoise で +237」であり「より賢い教師」ではない。32M denoise が 32M raw-技巧2 champion を超えるかは本走で確認要(2M 相対の +237 は強い示唆)。
- **可能性(投機)**: denoise 教師で新 champion → その champion で AZ warm-start やり直し → … の相互ブートストラップ。まず 32M 本走の green を見てから。

## 規律・メタ
一変数=教師 cp の出所のみ(局面/指し手/量/学習設定/net は 3系統同一)。(c) 資産は推論読取のみ、champion 不触。az_rescore.py は隔離 dir・非コミット(m4_distill_data/m2_usi/pgx 依存)。cache/eval/models/eval_dirs/TSV は `~/binbo-shogi-data` 永続・非コミット。既存 prepare_data/train_nnue/eval_vs_eval/dlshogi_rescore は無変更流用。**Rust 無改変**。**push は人間承認後。**
