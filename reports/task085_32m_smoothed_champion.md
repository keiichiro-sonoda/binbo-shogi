# Task 085 — 32M 全量を WS2-128 value で上書き → champion 超え挑戦(野田式スケール本走)

**目的**: 084 で「WS2-128 value は技巧2@d8 探索cp より良い NNUE 教師(2M で +237 elo)」が確定。32M 全量にスケールし champion 構成で学習、現 champion(task064 = 技巧2@d8 生ラベル 32M)に SPRT で挑む。**一変数=教師 train ラベルの出所のみ**(局面/量/net/学習設定は task064 と完全同一、mirror 込)。green なら 066-072 の「技巧2@d8 教師の天井」を教師の平滑化で初突破。実測 2026-07-10〜11、隔離環境(az_rescore.py 非コミット)、**リポジトリは文書のみ・Rust 無改変**。

**留保(本走で測る核心)**: 32M では生ラベルでも量による自己平均化が効く→平滑化の利得は 2M の +237 より逓減しうる。2M→32M カーブを実測。

## パイプライン(084 の配管を全て流用、既存 trainer 無変更)
- **cache**: `prepare_data run1m caches/shared_32m --canonical-shared --val-shards 5 --extra-runs run1m_b run2m_c run4m_d run8m_e run16m_f` = **n_train 31,950,000 / val 50,000 / 77 features**(task064 と同一 32M・同一 val 50k を再現)。
- **rescore**: `az_rescore --ckpt m6_ws2/ws2.ckpt`(cp=1200·atanh(v) clip±5000)、**alignment assert 全 split 通過**。**val は生技巧2 に passthrough**(`cp val_evals.npy → val_dlshogi_evals.npy`)=val_corr が champion 0.886 と直接比較可・early-stop 基準一致。
- **学習**: `train_nnue --qat --qat-tie --feature-dim 256 --hidden-dim 32 --lr 1e-5 --epochs 12 --batch 1024 --scale-cp 600 --augment-mirror --mirror-cache <8m perm> --eval-npy az_ws2_32m`(champion 構成、差分=train 教師のみ)。
- **判定**: `eval_vs_eval`(0cp ゲート→yane-vs-yane trinomial, mt0.1s, cap400)vs champion task064。

## rescore ラベル素性(2M→32M)
| | 技巧2 std | WS2-128 az std | corr | |Δcp| median |
|---|---|---|---|---|
| 2M(084) | 1778 | 1471 | 0.942 | 248 |
| **32M** | 1769 | **1462** | **0.941** | 246 |

→ 32M でも 2M と**ほぼ同一の平滑化特性**(std ~1462・corr 0.941)= WS2-128 は全量でも技巧2 に忠実な denoise ラベルを生成。alignment assert 全 split 通過。

## 学習(val_corr は生技巧2 val 比=champion 0.886 と直接比較)
- eval_npy 適用確認: train 31,950,000 swapped / val 50,000(生技巧2 passthrough)。12 epoch 完走(~900s/epoch)。
- **val_corr(生技巧2 val)= 0.873(best@ep12, mae 354)vs champion 0.886(best@ep6)**。平滑教師モデルは**生技巧2 val 予測でやや低い**(平滑ラベルで学習ゆえ当然)。ただし val_corr は棋力を決めない(084: 平滑教師は val 曖昧でも SPRT 勝ち)。判定は下の SPRT。
- decisive_sign 0.908 / |pred|_dec 1.06e3 = 決着局面の符号・振幅は健全。

## 主判定 SPRT — t085(WS2-128 平滑化 32M)vs champion(task064 生技巧2 32M)
- **0cp ゲート PASSED**(bit-exact)。**176k node 級の同一探索固定**、movetime 0.1s、cap 400。
- **結果: 178-219-3 / 400, score 44.9%, elo −35.7(CI −70.3..−1.9), verdict=continue(LLR −1.357)**。
- **CI 上端 −1.9 < 0 = t085 は champion より有意に弱い**(僅差だが 95% で負け)。**技巧2@d8 champion の天井は破れず**。

## ★2M→32M の利得反転(本タスクの核心的発見)
| データ量 | 平滑化(WS2-128)vs 生技巧2 |
|---|---|
| 2M(084) | **+237 elo**(accept_h1, 79.7%) |
| **32M(085)** | **−36 elo**(44.9%, 有意に負け) |
- **平滑化の利得は「小データの正則化」であって「教師改善」ではない**と確定。
  - **2M**: 生技巧2 ラベルの探索ノイズ(局面ごと ±jitter)を、生徒は 2M では平均化しきれない → AZ の denoise が大勝(+237)。
  - **32M**: 量による**生徒の自己平均化**でノイズが消える一方、AZ 平滑ラベルは**不可逆な圧縮損失**(tanh 飽和・有限 net 容量で実信号を捨てる)を抱える → 大データでは denoise 利得 < 圧縮損失 で**反転**(−36)。
- =066-072 の「技巧2@d8 教師天井」は、**同一情報の平滑化では(スケールでは)超えられない**。084 の +237 は天井突破でなく小データ正則化効果だった。

## 副次 — gikou@6 アンカー
- **t085 vs gikou@6 = 81-19-0, 81.0%(CI 73.3-88.7, elo +252)**。champion は task064 時点で 83% ゆえ**ほぼ同水準**(t085 が champion −36 と整合)。外部アンカーでも天井超えの兆候なし。

## 判定 = **RED(champion 超えならず)。ただし決定的に情報的な負け=「平滑化は小データ正則化、スケールでは天井を超えない」を確定**
- 主判定: t085(平滑化32M)vs champion = **−36 elo(有意に負け)**。副次 gikou 81%≈champion。**066-072 の技巧2@d8 教師天井は堅持**。
- **2M +237 → 32M −36 の反転**が本質: 平滑化(denoise)は生技巧2 の探索ノイズを小データで濾すが、①大データでは生徒が自己平均化してノイズが消え denoise の余地が無く、②AZ 平滑ラベルの不可逆圧縮損失(tanh 飽和・有限容量)が顕在化 → 反転。**AZ value は「より良い教師」ではなく「小データ用の正則化ラベル」**。
- 084 の「教師天井が動いた」示唆は**スケールで消えた**。champion(生技巧2 32M)は依然として (c)/教師軸の両方に対する天井。

## 申し送り
- **平滑化ルートは打ち止め**(スケールで反転を実測)。教師を本当に良くするには**新情報**が要る=AZ value(技巧2 蒸留物)では原理的に不可。→ AZ 側を**技巧2 非依存で強くする**(ゼロ知識性の高い warm-start/より深い自己対局/Transformer で棋力自体を上げ、その value を教師に)しかないが、M7-a で「AZ は αβ と別レジーム・内部優越は外部非転写」も判明済=険しい。
- **代替の安い一手(BACKLOG)**: 小データ域で +237 が出た事実は「正則化」として有効=**限られた新規教師(例: より深い技巧2 少量)を AZ 平滑化で薄めて足す**ハイブリッドは小〜中データ域で効く可能性(ただし champion は既に 32M で飽和側)。優先度中。
- **(c) の総括は不変**: 「単機自己対局 RL が正 elo(内部)」は M4/080 で達成、champion 級 αβ 超えは第2章(Transformer+規模)。085 は「AZ value を NNUE 教師に転用して天井を破る」バイパス路が**スケールで閉じた**ことを確定=第2章(AZ 自体の棋力向上)の必要性を補強。

## 規律・メタ
一変数=教師 train ラベルの出所のみ(cache/net/学習設定/mirror は task064 同一)。WS2-128 推論読取のみ、champion 不触。az_rescore.py 隔離・非コミット、既存 prepare_data/train_nnue/eval_vs_eval/binbo_to_yane 無変更流用。caches/models/eval_dirs/TSV は `~/binbo-shogi-data` 永続・非コミット。**Rust 無改変**。**push は人間承認後。**
