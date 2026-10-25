# Task 045: 教師 depth 10 の実測・生成・学習 — depth 8 より改善するか

## 結論

**教師 depth を 8→10 に上げても生徒は改善せず、むしろ弱くなった**(同規模 1M・同レシピの対照実験)。
しかも depth10 は **wall で depth8 の ~16.6倍**のコスト。**教師 depth↑の軸は技巧2 のクラッシュ挙動で塞がれている**
と判定し、次段は**表現力の軸(正準 HalfKP / KPP)**へ切り替える。推奨 EvalFile は **`model_cp_fd8_mirror_8m`** のまま。

- **コスト**: depth10 = **11.96 ms/局面**(12並列 wall)、depth8 の **~16.6×**、並列効率 **0.34**(depth8 は 5.91)。1M 生成は **3.32時間**。depth12(~22×, 効率0.25)よりマシだが重い。
- **クラッシュ**: depth10 はゲームの **82.4%(8020/9730)が暖機状態 EOF で late-midgame で死ぬ**。crash recovery で生成自体は回る(1M 完走・KP可逆 200k/0)が、**終盤・決定的局面が採れない**。
- **データ品質の confound(本質的発見)**: クラッシュで **decisive 局面が約1/3に枯渇**(|cp|≥1500: 23.72%→**7.43%**、≥5000: 5.30%→**0.41%**、near-equal: 28.34%→**41.12%**)。Task 035/041 で「生徒は decisive の度合いが律速」と判明している。**depth10 はまさにその decisive をデータから奪う**ため逆効果。
- **棋力(本命, Lv1 8局1秒, idle 同条件)**: 両者 0-8(1M は系列でも勝ち星ゼロ)。**だが粘り = depth8 99.2手 vs depth10 68.0手**。系列は良いモデルほど粘りが増える(64→79→105)のに depth10 は後退 = **depth10 の生徒は弱い**。
- **評価品質(nnue-diag, 技巧2 depth8 基準=depth 非依存)**: 符号一致 depth8 79%(34/43) vs depth10 70%(28/40)。代表局面 depth8 6/6 vs depth10 5/6(**決定的な mid_tactical を符号ごと誤る**)。

「depth だけを変数」は**クラッシュ起因の分布シフトで成立しない**こと自体が結論:**この技巧2 から depth8 品質の深い教師は採れない**。

## 背景

データ量は逓減(Task 041, 倍化Δ +270→+197→+77)、器拡大は無効(Task 042)、正則化は度合いを潰す(打ち止め)。
残る「質を上げる」軸の一つが教師の探索深さ↑。depth12 は非現実的と判明済み(Task 043: ~22倍・クラッシュ激増・効率0.25)。
本タスクは未検証の **depth10** を、実測→生成→学習→depth8 比較まで一気通貫で確かめた。

## ステップ A: depth10 生成コストの実測

単一スレッド `hcpe-gikou`(crash recovery 無)で depth8 は **4.676 ms/局面**(クリーン基準)。
depth10 は **game1 ply99 で即クラッシュ**(93局面で死亡)= 単一スレッドでは測れない。
crash recovery を持つ `gen-teacher` の bounded probe(shard 1000・12並列・timeout 30s, 12k局面)で測定:

| 指標 | depth 8 | **depth 10** | depth 12 (T043) |
|---|---:|---:|---:|
| wall ms/局面 (12並列) | 0.688 | **11.40** | 16.4 |
| depth8 比 | 1× | **~16.6×** | ~22× |
| 並列効率 (4.07基準) | 5.91 | **0.36** | 0.25 |
| ゲームcrash率 | 低 | **86/107 ≈ 80%** | — |
| timeouts(ハング) | 0 | **0** | 0 |

クラッシュは**ハングではなく EOF**(timeouts=0)。小 shard(1000)なら各 shard はクラッシュ間に完走する
(10k shard では 21分で 0/12 = 大きすぎて非完走)。

## ステップ B–C: 生成量の決定と depth10 1M 生成

実測から 1M ≈ 3.2h と予測。既存 `run1m`(depth8 1M)と**公平な同規模対照**になる **1M** を選択(ユーザー承認)。
別 run・別 seed(20300501)・永続領域・resumable ラッパー(sleep でも再開=データ損失なし)で生成。

- **1,000,000 局面 / 1000 shard / 失敗 0**、wall **199.4分(3.32h)**、avg **11.96 ms/局面**、効率 0.34。
- **total_games 9730 / crash 8020 = 82.4%**、timeouts 0、eval −30500..+31592(詰みは生存20%ゲーム由来)。
- 検証: `verify_shards` KP可逆性 **200k/0 不一致**・手番均衡・count/size ok。

### データ分布(全 1M, depth10 vs depth8 run1m)

| 指標 | depth8 run1m | **depth10 run_d10** |
|---|---:|---:|
| decisive \|cp\|≥1500 | 23.72% | **7.43%**(1/3.2) |
| ≥2600 | 14.10% | **2.73%** |
| ≥5000 | 5.30% | **0.41%**(1/13) |
| near-equal <200 | 28.34% | **41.12%** |
| median \|cp\| | 460 | **265** |
| p90 \|cp\| | 3428 | **1228** |

大きい局面ほど枯渇(≥5000 で 1/13)。クラッシュが「勝敗の付いた局面」を構造的に奪う。

## ステップ D: 学習と比較(対照実験)

両者まったく同じレシピ: fd8 + mirror + cp回帰(clamp±5000, MSE), GPU, 20ep, 同 seed。
val は各 run の末尾 5万(depth8=last5shard, depth10=last50shard)。train 95万 / val 5万で**規模一致**。

| 指標(各自 val) | depth8-1M 対照 | depth10-1M |
|---|---:|---:|
| best val_loss | 1,121,423 @ep3 | 447,669 @ep6 |
| baseline(定数)val_loss | 3,156,021 | **755,474**(1/4) |
| val_corr | **0.80** | 0.61–0.64 |
| decisive_sign | 0.83 | 0.79 |
| pred_dispersion val decisive PRED | 2184 / 正解3915 (56%) | 1211 / 正解**2638** (46%) |
| val decisive_frac | 0.237 | **0.072** |

- 対照(depth8-1M)の pred_dispersion **val 2184/3915 (56%)・gap 810** は系列 1M 値(2185/56%/809)と**完全一致**=対照が正しい。
- depth10 は定数 val_loss が 1/4(ターゲット自体が小さい)、val_corr も低い。**各自 val は分布が違うため直接比較できない**ので、以下の depth 非依存指標で判定。

### 棋力(本命, Lv1 8局1秒, idle 背中合わせ同条件)

| | record | score | avg_plies | binbo_avg_depth |
|---|---|---:|---:|---:|
| depth8-1M 対照 | **0-8-0** | 0.0% | **99.2** | 5.15 |
| depth10-1M | **0-8-0** | 0.0% | **68.0** | 5.16 |

両者 Lv1 に負け越し(1M は系列でも 0-8、勝ち星は 8M で初出)。**識別シグナル = 粘り**: depth10 は
**68.0手で潰れる**(depth8 の 99.2手より早い)。系列は良いモデルほど粘り増(64→79→105→94)。
**depth10 はその逆 = より弱い生徒**。

### 評価品質(nnue-diag, 技巧2 depth8 基準)

- 代表6局面 符号一致: depth8 **6/6**、depth10 **5/6**。
- **mid_tactical(決定的, 技巧2 d8 = +1556)**: depth8 static 1641/search **1509**(正確)、depth10 static 407/search **−141**(**符号ごと誤読**)。
- black_down_rook(技巧2 −2405): depth8 search −868、depth10 search **−449**(さらに圧縮)。
- 対局全体の符号一致: depth8 **79%(34/43)** vs depth10 **70%(28/40)**。

depth10 生徒は**決定的局面の認識が一貫して悪化**(decisive 枯渇の機序と整合)。

## ステップ E: 判定と次段

- depth10 の**生成コスト**: ~16.6倍・効率0.34・82%クラッシュ = 重い。1M で 3.3h(depth12 ほどではないが現実的とは言い難い)。
- depth10 学習 vs depth8 学習(同規模)の**度合い・棋力**: depth10 が全指標で悪化(粘り 68<99、符号 70%<79%、決定的局面を誤読)。
- **判定: 教師 depth↑の質の軸は、現実的コストで生徒を改善しない(むしろ悪化)**。機序は「深い読みが悪い」のではなく、**技巧2 が depth10 で 82% クラッシュし、生徒に最も必要な decisive 局面をデータから奪う**こと。Task 035/041 の「decisive の度合いが律速」という診断と完全に整合。
- **次段: 表現力の軸(正準 HalfKP / KPP)へ切り替える**。幅(Task 042)は効かず、データ量(041)は逓減、教師 depth(本タスク)は塞がれた。残るは**入力構造そのもの**を richer にする方向。

## 数値一致・回帰(green)

- Rust 数値一致(`verify_nnue_rust`): depth8-1M **2.852e-06**、depth10-1M **1.412e-06**(tol 1e-3)。スケール `forward*600=cp` は Rust 不変。
- `cargo test --release --locked --offline` **16/16**、`verify` perft **1〜5 完全一致**。
- **コード変更なし**(`gen-teacher`/`train_nnue.py`/`hcpe-gikou` は既存)。本タスクは計測・生成・学習・比較のみ。

## 成果物

- 非コミット(永続データ): `~/binbo-shogi-data/teacher/run_d10`(生100万, depth10, 別seed)、
  `~/binbo-shogi-data/models/model_d8_1m_task045`(対照)、`~/binbo-shogi-data/models/model_d10_1m_task045`、
  `~/binbo-shogi-data/caches/{d8_1m,d10_1m}`(KP cache, 再生成可)、`run_d10.gen.sh`(resumable ラッパー)。
- 推奨 EvalFile は **変更なし** = `~/binbo-shogi-data/models/model_cp_fd8_mirror_8m/nnue_f32.bin`。
