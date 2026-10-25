# Task 066 — 自己強化ループの設計確定 + 1サイクル・リハーサル(機構検証 + 偏り観察)

**ゴール**: 自己強化ループを **(A) 設計し docs に確定記述**(`docs/self_reinforcement_design.md`)し、**(B) 1サイクルだけ回して機構を検証**する。
狙いは ①機構が end-to-end で回ること、②外部アンカーで自家中毒/偏りを観察できること。**本走の量産はしない**(一変数原則=本リハーサルは「機構を回す」が主目的、cp表現/λ/深さ/多様性の最適化は後続タスクへ分離)。
初期化点 = champion `task064_32m_lr1e-5`(0cp 済, sha256 `3c25685e…`)。

## 機構(1サイクルを実際に回した経路)
```
champion(gen0) ─[1]ラベル生成 selfplay_label.py(champion@YaneuraOu@depth12 自己対局, cp+WDL)
              ─[2]混合 新規100k + 32M replay 100k = train200k(val=shared_32m pinned 50k)
              ─[3]warm-start 学習 train_nnue --init-weights champion --objective wdl --wdl-lambda 0.5(tied QAT fd256/h32)
              ─[4]export→writer→0cp ゲート(32/32)→ eval_dirs/gen1/nn.bin
              ─[5]評価 gen-vs-gen SPRT + 技巧2/tanuki アンカー
```
**全段が通った**(新規コード=`selfplay_label.py` / `train_nnue --init-weights` / Rust `yane-vs-gikou` の3点 + 既存パイプライン流用)。`cargo test --release` = **24 passed / 0 failed**。

## 実装したパラメータ(再現用)
- **ラベル生成**: champion `task064_32m_lr1e-5`(sha `3c25685e…`), depth **12**, random_plies **8**, seed **20260629**。
  → games **754**, positions **100,012**, WDL(side-to-move win/draw/loss) **0.47/0.06/0.47**, eval[min/med/max]=−32767/6/32767, **28–29 pos/s**(単一インスタンス, Threads=1, nice)。
- **混合**: 新規 100,012 + replay 100,012(`shared_32m`)→ **train 200,024**, val **50,000**(shared_32m の pinned をコピー=比較可能性)。
- **warm-start 学習**: `--init-weights`=champion `nnue_best.pt`, objective=wdl, **λ=0.5**(`target=λ·sigmoid(cp/600)+(1−λ)·gameResult`, MSE on sigmoid),
  tied QAT(qat+qat_tie, fd256/h32), lr **1e-5**, **3 epochs**, batch 1024, augment_mirror, ~6 s/epoch。export=**nnue_final.pt(epoch3=最も fine-tune された方)**。
- **gen1 eval**: `eval_dirs/gen1/nn.bin` sha256 **`3d65a1d0…`**, 0cp ゲート **32/32 PASS**。
- **対局**: YaneuraOu 9.41git 64AVX2, movetime **0.1s**, 探索固定・eval のみ変数・両側 0cp。i7-14700KF。トリノミアル SPRT(elo0=0, elo1=10)。

## 結果① — 採用判定(gen-vs-gen SPRT, yane-vs-yane, gen1 vs champion)
| candidate=gen1 | record(W-L-D) | score | elo | SPRT |
|---|---|---|---|---|
| vs champion(gen0) | **35-133-2** / 170局 | **21.2%** [15.1,27.3] | **−228** [−300,−171] | **verdict=accept_h0**(170局で早期停止) |

→ **gen1 は champion に明確に負ける。SPRT は H0(=gen1 は強くない)を採択し昇格を棄却。** 1サイクルの自己対局蒸留は **大きく後退**した。
**reversibility が働き champion は不動**(後退しない設計が機能)。draw_rate 1.2%(0.1s 探索の決着率は高い)。

## 結果② — 外部アンカー(偏り検知の本体)
| 対戦相手 | champion(gen0) | gen1 | gen0→gen1 の変化 |
|---|---|---|---|
| **技巧2 = yane-vs-gikou@DepthLimit6**(主アンカー) | 50-10-0, **83.3%** (elo +280) | 36-24-0, **60.0%** (elo +70) | **−23.3pt / −210 elo(明確に低下)** |
| **tanuki = yane-vs-yane `$YANE_EVAL`**(副アンカー) | 1-59-0, **1.7%** (elo −708) | 0-60-0, **0.0%** (elo −2400=floor) | 両者 floor 付近(飽和) |

gikou アンカーは avg_depth=**6.00, errors=0**(DepthLimit6 が安定参照として機能、depth≥8 クラッシュ問題を回避)。base(我々の net)は avg_depth ~15.7–16.2。

## 結果③ — 学習側の傍証(val は採用基準にならない)
gen1 の val 軌跡(`history.json`): val_corr **0.847→0.846→0.841**(技巧2ラベルから**離れていく**), best_val_loss **0.0242**(champion baseline **0.0209 を一度も下回らない**)。
→ **自己対局 WDL ラベルで学習すると技巧2-val の当てはめは悪化する**=設計どおり **val は採用基準にならず、アンカー/世代間対戦が判定軸**であることを実データで確認。

## 偏り観察(本タスクの主眼)
- **観測されたのは「自家中毒」ではなく「一様な後退」**。自家中毒のシグネチャは *「前世代に勝つがアンカーに負ける/落ちる」*(内部審判と外部審判の**乖離**)。
  本リハーサルでは **内部審判(gen-vs-gen: gen1 負け)と外部審判(gikou: gen1 −210 elo)が一致**=乖離なし。gen1 はどこでも弱く、判定は整合。
- **これは偏り検知装置の健全性を逆向きに確認した**: 本当に弱くなった世代を、gen-vs-gen SPRT も技巧2アンカーも**揃って正しく棄却**できた(検出力の sanity check)。
- **方法論的所見**: **tanuki は強すぎて champion 自身が floor 付近(1.7%)=世代差の分解能ゼロ(飽和)**。一方 **gikou@6 は champion が 83.3% に座り、gen1 の 60.0% への低下を明瞭に測れる**=この棋力帯では **gikou@6 が分解能のあるアンカー、tanuki は飽和して低分解能**。本走では主アンカーを gikou@6 に置くのが妥当(tanuki は天井監視の参考)。

## なぜ gen1 は後退したか(後続の一変数候補)
1サイクルが弱くなった原因は複数あり、**どれが主因かは未分離**(本リハーサルは機構検証が目的)。後続で一変数ずつ切り分ける:
- **replay 比 1:1 = 32M champion 知識の 0.3% しか mix に入らない**(100k/32M)。新規ノイズ半分の 200k fine-tune が champion 最適から引き剥がした可能性が最大の容疑。→ replay 比を上げる(例 新規:replay = 1:N)を**一変数**で。
- **自己対局ラベルの質**: WDL 0.47/0.06/0.47=ほぼコイントス・draw 6%。現棋力の自己対局 WDL は雑音が多い。→ λ を下げる(cp 寄り)/対局の質改善を**一変数**で。
- **規模が機構検証用(新規 100k)**で小さい。→ 本走規模(数百k–1M)を**一変数**で。
- λ=0.5・3 epoch・lr1e-5 で val_corr が 0.847→0.841 へ実ドリフト=学習は確かに動いた(過学習ではなく方向の問題)。

## 結論
- **機構は完全に回った**(ラベル生成→混合→warm-start→export→0cp ゲート→3種対戦)。新規3コンポーネントは健全(cargo test 24/24)。
- **採用ゲート(gen-vs-gen SPRT)と外部アンカーは整合して正しく機能**: gen1 を棄却、champion 不動、reversibility 維持。
- **偏りの所見**: 今回は自家中毒(内外乖離)ではなく一様後退。アンカー検出力は確認できた。**主アンカー=gikou@6(分解能あり)、tanuki は飽和**。
- **次の一変数**: 第一候補は **replay 比の引き上げ**(forgetting 抑制)。続いて λ・規模・深さを個別に。**機構が回ることは実証済み**なので、本走は「強くする一変数探索」に進める。

## 規律
一変数原則(機構検証が主目的、最適化は後続個別タスク)。GPL 遵守(技巧2/tanuki=合法アンカー、dlshogi 封印)。
長時間=`scripts/detached.sh`(セッション境界耐性・marker)+`env PYTHONPATH=<repo>`+`nice`。057-059 bridge 不触。破壊的操作・host 設定は人間。commit 可・**push は人間承認後**。
非コミット永続(`~/binbo-shogi-data/`): `caches/gen1_mix`, `models/gen1/`, `eval_dirs/gen1/`, 対局 TSV(`logs/t066_*.tsv`), logs。
