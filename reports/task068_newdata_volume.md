# Task 068 — 新規自己対局データ量の増量(10万→100万)、新規信号を一変数に

**ゴール**: Task 067 で replay 軸は決着(15.6% で champion とパリティ=gen4、超えず)。パリティ止まりの最有力原因 = **新規自己対局信号 10万が replay に対して非力**という仮説。
**一変数 = 新規自己対局量(10万→100万, 10×)**。replay を gen4 と同じ 5M(15.6%)に固定したまま新規だけを 10× にし、新規信号の力で champion を超えられるかを確かめる。

**クリーンな A/B**: gen4(100k new + 5M replay)が「新規10万」点、gen5(1M new + **同じ 5M replay**)が「新規100万」点。**差は新規量のみ**(replay/深さ/λ/lr/epoch/arch すべて同一)。

## 方法
- **新規 1M 生成(並列)**: `selfplay_label.py --workers 14`(新規・中=`multiprocessing.Pool`、各 worker が自前 Yane + 独立 seed で生成→concatenate→trim。**1局面ごとのラベル処理は直列と同一**)。
  champion@YaneuraOu@**depth12** 自己対局, cp勝率 + WDL, seed 20260630。→ 1,000,000 局面, **WDL 0.47/0.06/0.47**(066 と同一分布)。**14 workers で ~68 分**(≈256 pos/s, 直列 28 pos/s の ~9×。1M 直列なら ~10h)。
- **mix**: 新規 1M + replay 5M(`--replay-n 5000000`)→ train 6M, val = shared_32m pinned 50k。新規:replay = **1:5**(gen4 は 1:50)。
- **warm-start 学習**(`--init-weights` champion `nnue_best.pt`, λ0.5, tied QAT fd256/h32, lr1e-5, 3ep)→ export(nnue_final.pt)→ writer → **0cp ゲート 32/32 GREEN** → `eval_dirs/gen5/nn.bin`(sha `57f8053e…`)。
  **全ハイパラが gen4/066 と一致、差分は `n_new`(100k→1M)と `n_train`(replay 5M は同一)**(`model_config`/`history` で確認)。
- **判定**: gen-vs-gen SPRT(`yane-vs-yane` 対 champion, 0cp 両側, movetime 0.1s, 200 局, seed 20260701)+ 技巧2アンカー(`yane-vs-gikou` gikou@6, 60 局, seed 20260702 = 強さ + 忘却チェック)。i7-14700KF + RTX 4060 Ti。

## 結果 — 仮説と逆: 新規量を増やすと**弱くなる**(量は主因でない)
| 世代 | 新規(@5M replay 固定) | gen-vs-gen vs champion | 技巧2 gikou@6 | gen sha |
|---|---|---|---|---|
| gen4 | 100k(1:50) | 46.0% (−28) [39.1,52.9] = パリティ | 83.3% (+280) | `1f619938…` |
| **gen5** | **1M(1:5)** | **32.8% (−125) [26.3,39.2]** | **72.5% (+168)** | `57f8053e…` |
| champion(参照) | — | — | 83.3% (+280) | `3c25685e…` |

- 記録: gen5 vs champion = **65-134-1 / 200**(draw 0.5%)。gen5 vs gikou = 43-16-1 / 60。探索拮抗(gen5 14.94 ≈ champion 15.34 avg_depth, ~118k nodes, ~94ms=差は eval)。
- **新規量 10× で gen5 は gen4 より大きく後退**: gen-vs-gen 46%→**32.8%**(−28→**−125 elo**, ~−97 elo)、技巧2 83.3%→**72.5%**(~−112 elo)。**固定 replay 下で新規量 vs elo は単調減少**(新規が増えるほど悪化)。
- **内外(gen-vs-gen と技巧2)が一致して後退**=自家中毒(乖離)でなく純粋な弱化。066/067/068 を通じ**全条件で内外一致=中毒は一度も観測されず**。

## 解釈 — ボトルネックは新規信号の「量」でなく「質」
- 結論: **新規 champion@d12 自己対局ラベルは、置き換える技巧2@d8 replay より低品質**。希釈されている間(gen4 1:50 = 新規 2%)は無害(パリティ)、相当量入る(gen5 1:5 = 新規 1/6)と **−125 elo まで悪化**。
- なぜ低品質か(仮説): ① **WDL が近コイントス**(0.47/0.06/0.47)= champion 同士対局の勝敗ラベルは雑音が大きく、λ=0.5 で半分の重み。② champion@d12 cp は **champion 自身の eval(技巧2蒸留物)+ 12手探索補正**=技巧2@d8 を明確に上回る信号とは限らず、champion の盲点が自己対局で強化される恐れ。
- **067 のパリティ天井を後付けで説明**: replay 増で「回復」したのは、replay が **この(軽度に有害な)新規信号をほぼゼロまで希釈**したから=新規信号は伸びしろでなく現状は重し。
- **忘却チェック**: gikou 72.5%(<83%)に低下。ただし原因は replay 量不足でなく(replay は 5M 固定)**新規信号が能動的に悪い**ため。→ replay 量再考でなく**ラベル品質**へ。

## 結論と次の一手
- **新規「量」は前進の主因でない**(むしろ固定 replay 下では逆効果)。decision rule に従い **次の一変数 = ラベル品質**へ切り分けを移す。
- **次候補(本走の第一手, 個別タスク・一変数)**: ① **ラベル深さ d12→d14**(065: champion@d cp は飽和せず=深い探索ほど cp が静的 eval を書き換える=質ダイヤル)。② **λ↓**(noisy WDL の重みを下げ cp 寄りに)。③ 多様性(champion 自己対局の盲点強化を緩和=対局中ランダム手/次善手)。④ 将来 replay の自己ラベル上書き(自ラベルが技巧2品質を超えてから)。
- **champion 据え置き**(`task064_32m_lr1e-5`, sha `3c25685e…`)。gen5 は棄却(−125 elo)、昇格なし(reversibility)。gen4(100k)が依然この replay 帯の最良(パリティ)。
- **副産物**: `selfplay_label.py --workers`(並列自己対局, ~9× 速)を獲得=今後のラベル生成の足場(深さ/λ 探索でも再利用)。

## 再現
champion sha `3c25685e…` / depth12 / 新規 1M(seed 20260630, workers 14)/ replay-n 5M(shared_32m, seed 20260629 系)/ λ0.5・lr1e-5・3ep・fd256/h32(tied QAT)/
gen5 sha `57f8053e…` / movetime 0.1s / gen-vs-gen SPRT(elo0=0, elo1=10, seed 20260701)/ gikou@6 seed 20260702 / 生成 ~68min@14w / i7-14700KF + RTX 4060 Ti。
実装 = `selfplay_label.py --workers`(新規・並列生成, スモーク済)、他は 066/067 パイプライン流用。

## 規律
一変数(新規自己対局量のみ。replay比率/深さ/λ/cp表現/ラベル出所/057-059 bridge/Rust 探索は不触)。GPL 遵守(技巧2/tanuki=合法アンカー、dlshogi 封印)。
長時間=`scripts/detached.sh`(marker)+`env PYTHONPATH=<repo>`+`nice`。`cargo test --release` 24 green(Rust 無改変)。commit 可・**push は人間承認後**。
