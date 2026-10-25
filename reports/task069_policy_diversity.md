# Task 069 — 自己対局に方策多様性を導入(温度サンプリング)、ラベル品質を一変数に

**ゴール**: 068 で「parity 止まりの原因は新規量でなく品質」が確定。品質低下の構造的原因の仮説 = 現状の自己対局は **局面集合の多様性
(開幕8手ランダム)は高いが方策の多様性ゼロ**(対局本体は champion@d12 の最善手を決定的追従)→ champion が低評価する手は一度も探索されず
**盲点・系統誤差が強化**される。**一変数 = 対局本体の方策多様性(温度サンプリング)**で champion の盲点に光を当て、超えられるかを確かめる。

**クリーンな A/B**: gen4(決定的, 新規100k, replay 5M)が基準。gen6(温度サンプリング, **同じ新規100k・replay 5M・深さ12・λ0.5**)との差は
**対局本体の手の選び方(決定的→温度)のみ**。gen4 値(gikou 83.3% / gen-vs-gen 46.0% −28)を再利用。

## 方法
- **新規コード = `selfplay_label.py` の方策多様性**(他は 066/067/068 流用):
  - `Yane` に `MultiPV=N` 対応 + `go_depth_multi`(`info … multipv K … score cp … pv M1` を解析、`lowerbound/upperbound` 除外、K 昇順で `[(move,cp,is_mate)]`、`[0]`=最善)。
  - `play_one_game`: **ラベル cp = multipv-1(champion の決定的評価, 不変)**。**指す手**は `random_plies < ply ≤ sample_until_ply` かつ T>0 のとき、上位 N 手を **勝率空間 `w=sigmoid(cp/600)` の softmax/T** でサンプル、それ以外は最善(終盤=決定的)。
  - フラグ `--temperature 0.10 --top-n 4 --sample-until-ply 40`。`--temperature 0/未指定` = 従来の決定的経路(回帰)。`--workers` 並列はそのまま。
- **スモーク(実機)**: MultiPV が startpos で上位4手 `2g2f88/7g7f74/6i7h64/3i3h38` を返す、サンプルが T で挙動変化(T=0.02 で最善 56%、T=0.10 で top4 ほぼ一様)、temp=0 が決定的経路と同挙動、を確認。勝率空間 softmax は **鋭い局面では最善に集中・互角局面で散る**自己調整性。
- champion `task064_32m_lr1e-5`(sha `3c25685e…`)→ 100k 生成(temp, 14 workers, ~29分; MultiPV=4 は単PV の ~3–4×)WDL 0.49/0.02/0.49 → mix 5M → warm-start(λ0.5, lr1e-5, 3ep)→ 0cp ゲート 32/32 GREEN → `eval_dirs/gen6`(sha `72ff9494…`)。
- 判定 = gen-vs-gen SPRT(対 champion, 0cp 両側, 0.1s, 200局, seed 20260701)+ 技巧2 gikou@6(60局, seed 20260702)。i7-14700KF + RTX 4060 Ti。

## 結果 — 仮説と逆: 温度多様性は**悪化**(単純版では効かない)
| 世代 | 手の選び方(新規100k, replay 5M) | gen-vs-gen vs champion | 技巧2 gikou@6 | gen sha |
|---|---|---|---|---|
| gen4 | 決定的(最善) | 46.0% (−28) [39.1,52.9] = パリティ | 83.3% (+280) | `1f619938…` |
| **gen6** | **温度 T=0.10, top-4, ≤40ply** | **38.0% (−85) [31.3,44.7]** | **66.7% (+120)** | `72ff9494…` |
| champion(参照) | — | — | 83.3% (+280) | `3c25685e…` |

- 記録: gen6 vs champion = **75-123-2 / 200**(draw 1.0%)。gen6 vs gikou = 40-20-0 / 60。
- **温度多様性で gen6 は gen4 より後退**: gen-vs-gen 46%→**38%**(−28→**−85 elo**)、技巧2 83.3%→**66.7%**(~−160 elo)。**内外(gen-vs-gen と技巧2)が一致して後退**=自家中毒でなく純粋な弱化(066–069 全条件で内外一致=中毒なし)。
- val はほぼ不変(gen6 best_val_loss 0.01701 ≈ gen4 0.01697)。temp は局面分布を変えるがラベルは champion@d12 最善 cp のままなので当然。**val ≒ 同じなのに対局は悪化** = ここでも val は採用基準にならない。

## 解釈 — 「champion の知識の再サンプリング」では超えられない(067/068/069 を貫く結論)
温度多様性が効かない構造的理由: サンプリングは champion が**普段指さない off-distribution な局面**へ誘導するが、その局面のラベルは
**champion@d12 自身の評価**=champion が最も**信頼できない**領域の eval を教師にする → ノイズを増やす(068 の低品質ラベルの罠を、量でなく局面選択で再現)。
**盲点は、盲点を持つ評価器自身のラベルでは直せない**。

067(replay→パリティ天井)・068(量↑→悪化)・069(多様性→悪化)を貫く統一見解:
**champion の d12 知識を「再サンプリング」するだけの操作(量・多様性)は champion を超えられず、しばしば悪化させる**(champion の誤差を信頼性の低い領域で増やすため)。
champion を**超える**には、champion の静的 eval に**新情報を足す**必要がある。それができる唯一のレバーは **より深い探索**(065: champion@d cp は
depth16 まで飽和せず=深い探索が静的 eval を実質書き換え続ける=新情報)。temp/量は既存の限られた知識を混ぜ直すだけ。

## 結論と次の一手
- **方策多様性(単純版 T=0.10)は前進せず悪化**。decision rule に従い棄却(gen6 不採用・champion 不動, reversibility)。
- **次の本命 = ラベル深さ d12→d14**(Task 070 候補)。065 で唯一「新情報を足す」と実証済のレバー。新規量は gen4 帯(100k/replay 5M)固定、深さだけ一変数で。
- 副次の枝(温存): **温度 T↓ sweep**(T=0.10 は開幕で top4 ほぼ一様=高すぎた可能性。T=0.02–0.05 でマイルドな散らし)、**quiet position フィルタ**(off-distribution の戦術不安定局面を除外)。ただし上記の構造的理由から、深さ(新情報)が本命で、温度は補助。
- 副産物: `selfplay_label.py` の MultiPV + 温度サンプリング基盤を獲得(深さ sweep / 将来の T sweep で再利用)。

## 再現
champion sha `3c25685e…` / depth12 / 新規100k(seed 20260701, workers 14, **temperature 0.10, top_n 4, sample_until_ply 40**)/ replay-n 5M /
λ0.5・lr1e-5・3ep・fd256/h32(tied QAT)/ gen6 sha `72ff9494…` / movetime 0.1s / gen-vs-gen SPRT(elo0=0, elo1=10, seed 20260701)/ gikou@6 seed 20260702 / i7-14700KF + RTX 4060 Ti。
実装 = `selfplay_label.py`(MultiPV + 勝率空間温度サンプリング, スモーク済, `--temperature 0` 回帰)、他は 068 パイプライン流用。

## 規律
一変数(対局本体の方策多様性のみ。開幕散らし/replay/量/深さ/λ/cp表現/ラベル出所/057-059 bridge/Rust 探索は不触)。GPL 遵守(技巧2/tanuki=合法アンカー、dlshogi 封印)。
長時間=`scripts/detached.sh`(marker)+`env PYTHONPATH=<repo>`+`nice`。`cargo test --release` 24 green(Rust 無改変)。commit 可・**push は人間承認後**。
