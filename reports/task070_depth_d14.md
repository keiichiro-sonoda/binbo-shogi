# Task 070 — ラベル深さ d12→d14(clean A/B)+ gold subset で教師品質を直接測る

**ゴール**: 067/068/069 で「champion の d12 知識を**再サンプリング**するだけ(replay 比率・新規量・方策多様性)では超えられず悪化」と確定。
唯一「新情報を足す」と実証済のレバーが**深い探索**(065: champion@d cp は d16 まで飽和せず)。本タスクは一変数 = **ラベル深さ d12→d14** を
gen4(100k new・d12・決定的・replay 5M・λ0.5)と全変数一致の **clean A/B**(新規量も 100k に揃え 068 の量交絡を排除)で検証。
同時に **gold subset**(固定 3000 局面を d12/d14/d16 でラベル)で「d14 が d16 に近づくか=教師に新情報が乗るか」を学習と独立に直接測る。

## 方法
- **gold(成果物B, 既存 `label_cost.py` 再利用 + 新規 `gold_analyze.py`)**: teacher run1m から 3000 局面(seed 20260702)、単一インスタンス・Threads=1・直列・cold-TT(honest timing)で d12/d14/d16 をラベル → TSV。`gold_analyze.py` で d16 を gold 基準に closeness/sign/time を算出。
- **clean A/B(成果物A, 新規コード不要)**: `selfplay_label.py --depth 14 --temperature 0 --top-n 1`(決定的=gen4 と同じ方策)で新規 100k 生成(seed 20260702, workers 14)→ mix 5M replay → warm-start(λ0.5, lr1e-5, 3ep, tied QAT)→ 0cp ゲート 32/32 GREEN → `eval_dirs/gen7`(sha `9868a3fd…`)。
  **全ハイパラ gen4 と一致、差分は生成深さ d12→d14 のみ**(meta: depth=14, temperature=0, top_n=1, n_new=100000, replay 5M)。
- 判定 = gen-vs-gen SPRT(対 champion, 0cp 両側, 0.1s, 200局, seed 20260701=gen4 とペア)+ 技巧2 gikou@6(60局, seed 20260702)。i7-14700KF + RTX 4060 Ti。

## 結果① gold — d14 ラベルは確かに d16 に近い(教師品質は上がる)
| 指標(gold=d16, n=2882) | d12 | d14 |
|---|---|---|
| median &#124;cp − cp_d16&#124; | 80.0 | **47.0** |
| mean / p90 | 796 / 401 | 547 / 234 |
| d16 に d12 より近い局面の割合 | 30.8% | **67.7%**(tie 1.4%) |
| 符号反転率 vs d16 | 6.0% | **3.8%** |
| near-zero(&#124;cp_d16&#124;<50)符号反転 | 33.6% | **25.5%** |
| label-time p50/p90/p99 (ms) | 30.5 / 88.8 / 226 | 74.9 / 257 / 1077 |

→ **d14 は d16 への距離中央値を半減(80→47)、2/3 の局面で d12 より d16 に近く、符号一致も改善**。065 の「cp は飽和せず」を局面単位で裏付け。
**067-069 で初めて、レバーが教師品質を実際に上げた**。コスト d14≈d12 の ~2.4×(p50)、capped 0%。

## 結果② loop — だが d14 で学習した gen7 は gen4 より**悪化**(better labels → worse model)
| 世代 | 深さ(100k/5M 固定) | gen-vs-gen vs champion | 技巧2 gikou@6 | gen sha |
|---|---|---|---|---|
| gen4 | d12 | 46.0% (−28) [39.1,52.9] = パリティ | 83.3% (+280) | `1f619938…` |
| **gen7** | **d14** | **37.0% (−92.5) [30.3,43.7]** | **76.7% (+207)** | `9868a3fd…` |
| champion(参照) | — | — | 83.3% | `3c25685e…` |

- 記録: gen7 vs champion = **74-126-0 / 200**(draw 0)。gen7 vs gikou = 46-14-0 / 60。
- **d14 で gen7 は gen4 より後退**: gen-vs-gen 46%→**37%**(−28→−92.5 elo)、技巧2 83.3%→**76.7%**。内外一致(両判定が後退)=自家中毒でなく純粋な弱化(066-070 全条件で内外一致)。
- val は gen7 0.0177 > gen4 0.01697(d14 は技巧2-val からより離れる)=val 不採用を再確認。

## 解釈 — ボトルネックは「ラベルの質」でなく「新ラベル↔replay アンカーの不一致」
**最重要**: gold で**ラベルの質は上がった**のに loop は**悪化**=「ラベル情報量が足りない」説は棄却。質と結果が**デカップル**。
統一機構(067-070 を1つで説明):
- 系全体が **技巧2@d8 にアンカー**されている: warm-start champion(d8 蒸留)+ replay 5M(技巧2@d8)= 学習データの **98%**。
- 新規 champion 自己対局ラベル(d12 でも d14 でも)は **d8 アンカーと食い違う**。**深いほど食い違いが大きい**(gold: d14 は d12 から median 57cp 離れ、d8 からはさらに遠い)。
- d8 アンカーに固定された loop は、矛盾する少数(2%)の新ラベルを**ノイズ**として扱う → 矛盾が大きいほど(d14)悪化。
- これは **068(新規量↑=矛盾の総量↑→悪化)と 070(新規が深い=1件あたりの矛盾↑→悪化)を同一機構で説明**。再サンプリング系が全滅した理由も同じ(既存 d8 知識の混ぜ直しに留まり、矛盾だけ増やす)。
- gen4(d12, パリティ)が依然この族の最良=「新規信号をいじる」family は **gen4 でサチっており、深さでも超えない**。

**交絡(明記)**: gen7 の局面軌跡は gen4 と異なる(d14 は着手が違う)。ただし gold は固定局面で深さ効果を純粋分離済で、A/B は「champion 自己対局・探索深さだけ変えた」比較=結論(質↑≠棋力↑)は交絡に頑健。

## 結論と次の一手
- **深さ(d12→d14)はラベル品質を上げるが loop では悪化**。decision rule の「gold 陽性 + A 陰性 → ラベルの質は上がるが loop が使えていない」に該当。**gen7 棄却・champion 不動**(reversibility)。**d16 full は走らせない**(計画どおり)。
- **方向転換**: 067-070 で「新規信号(replay比/量/多様性/深さ)」family は出し尽くし、いずれも gen4 のパリティ天井を超えない。**真の律速 = 技巧2@d8 replay アンカーとの不一致**。
- **次の本命(Task 071 候補)= replay 出所/アンカーの一致化**(長く温存してきた BACKLOG 項目が筆頭に昇格):
  - ① **replay を新ラベルと同じ teacher で付け直す**(自己ラベル上書き=replay 局面を champion@d で再ラベル)。gold が「champion@d は search-refined」を示した今、上書きの判断材料が揃った(ただし d8 を捨てるリスクは要監視)。
  - ② **d8 アンカーを緩める**(replay 比率↓ + 忘却を別手段で防ぐ / warm-start を弱める)。067 は replay↓で崩壊したが、それは「新ラベルが d8 と矛盾」かつ「アンカー無し」の二重苦。一致化すれば replay↓が効く可能性。
  - 一変数原則で ①→② を個別タスク化。主アンカー=gikou@6。
- 副産物: `gold_analyze.py`(教師品質を学習と独立に測る診断)。今後の replay 一致化でも「上書きラベルが d16 にどれだけ近いか」を同手法で監視できる。

## 再現
champion sha `3c25685e…` / gold: teacher run1m 3000局, d12/14/16, seed 20260702, 単一インスタンス, cap 60s / A: 新規100k d14(seed 20260702, workers 14, 決定的)・replay 5M・λ0.5・lr1e-5・3ep・fd256/h32(tied QAT) /
gen7 sha `9868a3fd…` / movetime 0.1s / gen-vs-gen SPRT(elo0=0, elo1=10, seed 20260701)/ gikou@6 seed 20260702 / i7-14700KF + RTX 4060 Ti。
実装 = `gold_analyze.py`(新規・分析のみ)+ `label_cost.py`/`selfplay_label.py --depth`(既存流用)。

## 規律
一変数(ラベル深さのみ。新規量/replay/λ/方策/cp表現/ラベル出所/057-059 bridge/Rust 探索は不触)。GPL 遵守(技巧2/tanuki=合法アンカー、dlshogi 封印)。
長時間=`scripts/detached.sh`(marker)+`env PYTHONPATH=<repo>`+`nice`(`pkill` は sandbox 不可=不使用)。`cargo test --release` 24 green(Rust 無改変)。commit 可・**push は人間承認後**。
