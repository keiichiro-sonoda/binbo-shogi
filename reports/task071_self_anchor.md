# Task 071 — replay を技巧2@d8 から champion@d12 自己ラベルへ置換(技巧2ラベルからの離脱・自己アンカー化)

**ゴール**: 067-070 の統一律速「新ラベル↔技巧2@d8 replay の不一致」に対し、**d8 を学習信号から完全に外し、系のアンカーを技巧2から champion 自身へ移す**(パターンA=局面もラベルも champion 自己対局)。066 以来の本丸=自己強化が原理的に機能するかの直接検証。

## 方法(一変数 = replay/ラベルの出所)
- **gen8 = 純 champion@d12 自己対局 ~5M、技巧2@d8 replay 0%**。gen4(d12 new + d8 replay)に対し **同一深さ d12 で出所だけ d8→自分** を入れ替えた最 clean な一変数(当初計画の d14 は実測コスト ~200k/h=5M≈20-30h と判明しユーザ決定で d12 に変更=~2倍速 + gen4 と同深さ)。
- **生成(chunked + resumable)**: `t071_gen.sh` が 10 shard × 505k = 5.05M を champion@d12 決定的(温度0/top_n1)self-play で生成(各 shard 別 seed 20260801..810、`--workers 14`)→ concat → 末尾 50k を val に hold out → `caches/gen8_d12_self5m`(train **5,000,000** / val 50,000)。**実測 ~5h51m**(11:53→17:44、10 shard 各 ~35min、attempt 1 で全完走・retry 不要)。chunk 化理由=selfplay_label は一括書き(no-checkpoint)で長単発 run はクラッシュ全損。
- **唯一のコード変更**: `selfplay_label.py` に `--val-frac`(replay 無し時に new 行末尾を val に hold out)。reuse-new スモーク(95k/5k split・meta 整合・tail-split)+ 実エンジン micro で検証。`cargo test --release` **24 green**(Rust 無改変)。
- **学習・ゲート**: 070 verbatim(`--qat --qat-tie --feature-dim 256 --hidden-dim 32 --objective wdl --wdl-lambda 0.5 --lr 1e-5 --epochs 3 --augment-mirror --mirror-cache shared_8m_mirror_perm.npy --init-weights champion nnue_best.pt`)。cache guard で `depth=12, mix_replay=None, replay_n=0` を assert(一変数性を機械保証)。**0cp ゲート 32/32 GREEN**(engine e == binbo Mode::Quant == int-faithful、bit-exact)→ `eval_dirs/gen8`、sha256 `aae0847c…`。
- **判定**: gen-vs-gen SPRT 対 champion(両側0cp, 0.1s, 200局, **paired seed 20260701**)+ 技巧2 gikou@6(60局, **paired seed 20260702** = gen4/gen7 と同一=外部アンカーが直接比較可能)。i7-14700KF + RTX 4060 Ti。

## 結果 — d8 を外すと**最悪化**(内外そろって後退=自家中毒でなく忘却/弱化)
| 世代 | データ(label source) | gen-vs-gen vs champion | 技巧2 gikou@6 | gen sha |
|---|---|---|---|---|
| gen4 | d12 new + **技巧2@d8** replay | 46.0% (−28) = パリティ | 83.3% (+280) | `1f619938…` |
| gen7 | d14 new + **技巧2@d8** replay | 37.0% (−92.5) | 76.7% | `9868a3fd…` |
| **gen8** | **d12 champion 自己ラベルのみ(d8 0%)** | **25.5% (−186.2) [19.5,31.5]** | **71.7% [60.3,83.1]** | `aae0847c…` |
| champion(参照) | 技巧2@d8 蒸留 | — | 83.3% | `3c25685e…` |

- 記録: gen8 vs champion = **51-149-0 / 200**(draw 0)、sprt_llr −2.913(bounds ±2.944、verdict continue だが score CI 全域<50%=明確に劣後)。gen8 vs gikou = 43-17-0 / 60。
- **gen8 は 3 世代で最悪**(gen-vs-gen 46→37→**25.5%**、技巧2 83.3→76.7→**71.7%**)。**内部(gen-vs-gen)と外部(技巧2)が揃って後退** = 内外一致 = **自家中毒(内↑外↓)ではなく、純粋な弱化/忘却**。
- val(self-play holdout, 世代間 **非**比較): val_corr 0.732→0.722→**0.718**、val_loss は epoch1 以降 上昇(0.0595→0.0626)。export は他世代同様 final epoch。

## 切り分け診断 — 「狭い self-play 分布」説を**棄却**(=律速はラベルの知識質)
収集の素直な疑い=「決定的 self-play は局面が冗長/狭い」→ unique-position 比率を実測:
| プール(2M サンプル) | unique 局面 | unique 比率 |
|---|---|---|
| gen8 self-play d12 | 1,999,995 | **1.0000** |
| 技巧2@d8 replay(shared_32m) | 1,950,658 | 0.9753 |

- **gen8 self-play は 100% ユニーク**(replay より高い)= 局面被覆は狭くない。**冗長/狭さ説は棄却**。
- → 弱化の原因は**被覆**でなく**ラベルの知識質**: **champion の d12 自己 eval は技巧2@d8 ラベルより弱い教師信号**。champion は技巧2@d8 から蒸留された器=自己 eval を学習し直すと技巧2の知識から離れ、**より弱い自己整合的不動点**へ退行(技巧2-蒸留の強さを忘却)。**探索深さ d12(>d8)は教師の局面知識を代替しない**。

## 解釈 — 統一機構(067-071 を1つで)と本丸の結論
- **技巧2@d8 replay は load-bearing**: 忘却防止だけでなく、**ラベルの知識質そのものが champion 自己 eval より高い**(被覆は同等以上に良いのに質で負ける)。
- 067(replay↓=教師信号↓→崩壊)/ 070(深い自己ラベル=技巧2と矛盾↑→悪化)/ **071(自己ラベル100%=教師を弱い自己 eval で全置換→最悪)** を **同一機構**で説明。
- **本丸の結論**: **champion 自己対局ラベルは(どの深さでも)技巧2-蒸留 champion を超えられず、維持すらできない**。再サンプリング系(067-070)に続き **アンカー離脱(071)も否定**。自己強化ループ(champion self-play → warm-start 再学習)は **このアーキでは教師(技巧2@d8)の天井を超えない**。
- champion を超えるには **技巧2 より強い教師信号** が必要(champion 自身の self-eval では不可)。

## 結論と次の一手
- **gen8 棄却・champion 不動**(reversibility、SPRT 勝利時のみ昇格)。自己アンカー化は **net 負**。
- **戦略的転換点**: 067-071 で「自己対局ラベルをいじる」族 + 「アンカー離脱」を出し尽くし、いずれも champion を超えない。次は **教師信号の強化**が必須:
  - ① **より深い/強い教師ラベル**で replay(アンカー)を付け直す(技巧2@d8→より深い技巧2)。**ただし gikou は depth≥8 でクラッシュ**(既知=065/066)→ 技巧2 を深くする経路は要回避策。
  - ② **探索ベースの方策改善**(AlphaZero 流: 静的 eval 回帰でなく、探索の visit/対局結果を方策ターゲットに)= 教師を超える理論的に健全な道だがアーキ大改修。
  - ③ **技巧2@d8 蒸留 champion を本アーキの天井**と受け入れ、プロジェクト目標を再設定。
  - → 一変数原則で個別タスク化。主アンカー=gikou@6 は**ラベルから外しても監視として常設**(GPL 合法、封印対象でない)。
- 副産物: `--val-frac`(純 self-play の val hold-out)、chunked+resumable な大規模 self-play 生成 driver、unique-frac 診断手法。

## 再現
champion sha `3c25685e…` / gen: 5M champion@d12 決定的 self-play, 10 shard seeds 20260801..810, `--workers 14`, **technique2@d8 replay 無し**, val holdout 50k / 学習 070 verbatim(λ0.5, lr1e-5, 3ep, fd256/h32 tied QAT, warm-start champion, mirror=shared_8m_mirror_perm.npy)/ gen8 sha `aae0847c…` / movetime 0.1s / gen-vs-gen SPRT(elo0=0, elo1=10, seed 20260701)/ gikou@6 seed 20260702 / i7-14700KF + RTX 4060 Ti。
実装 = `selfplay_label.py --val-frac`(新規・infra)+ `t071_gen.sh`(chunked/resumable)/ 学習・ゲート・対局ハーネスは既存流用。

## 規律
一変数(ラベル/replay 出所のみ。深さは d12=gen4 同一に固定、温度/λ/量/bridge/Rust 探索は不触)。GPL 遵守(技巧2/tanuki=合法、ラベルから外しても監視アンカーは存置、dlshogi 封印)。長時間=`scripts/detached.sh`(marker)+`env PYTHONPATH=<repo>`+`nice`(`pkill` は sandbox 不可=不使用)。`cargo test --release` 24 green(Rust 無改変)。commit 可・**push は人間承認後**。
