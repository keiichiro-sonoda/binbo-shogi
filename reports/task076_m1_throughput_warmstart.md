# Task 076 — M1: フル将棋 self-play スループット/VRAM 実測 + warm-start 戦略の見極め

**目的**: (c) の規模決めの土台。075 M0(どうぶつ将棋で単機 AZ ループが回り前進)を受け、①Pgx **フル将棋** env の self-play スループット/VRAM を RTX 4060 Ti 16GB で実測し単機動作点を数字で特定、②「ゼロ知識 / warm-start / 中間題材」の a/b/c 比較+推奨を出す。065(本走前コスト実測)と同位置づけ、**計測と小規模試走のみ・フル本走なし**。実測 2026-07-02、隔離 venv `~/task074_frameworks/venv_pgx`(jax 0.10.2 / pgx 2.6.0 / mctx 0.0.71)、計測コード=`train_m0.py` の隔離コピー `train_m1.py`(計時分離+有効サンプル率を追加、非コミット)。**最終判断は Keiichiro**。

## 前提(コード精読で確定)
- Pgx shogi: observation **(9,9,119) bool**(dlshogi 流: 駒配置+**利き**+持ち駒閾値+王手、channels-last=AZNet の NHWC conv と整合)、行動空間 **2187=27方向×81**(cshogi/dlshogi と同一エンコード、`_from_dlshogi_action`)、`MAX_TERMINATION_STEPS=512`。どうぶつ将棋: obs (4,3,22)・132 手。
- **shogi の baseline model は Pgx に無い**(`_src/baseline.py`)→ M1 は throughput 専念(強さ評価は M2 の USI 橋+SPRT で)。
- train.py の値ターゲットは `max_num_steps` 窓内に終局しない対局分を mask(train.py:179,211)→ **「窓内有効サンプル率」が速度/VRAM と並ぶ第3の軸**(policy ターゲットは全フレーム有効、mask は value のみ)。

## パート1 — 実測表(4 iter 中 iter2-4 の平均。iter1=JIT 込は別列。VRAM は 1s ポーリング max、idle 941MiB 込)
| 点 | sims | net(params) | batch | steps | **VRAM MiB** | **selfplay s/iter** | train+prep s | **states/s** | **value有効率** | 終局数/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| **どうぶつ将棋(M0 設定)** | 32 | 64ch×5L (0.49M) | 512 | 64 | 4,658 | **8.9** | 0.5 | **3,867** | 0.67 | 912 |
| **center(将棋)** | 32 | 64ch×5L (0.80M) | 256 | 256 | 9,328 | **136.3** | 3.5 | **482** | 0.47 | 195 |
| sims16 | 16 | 64ch×5L | 256 | 256 | 9,513 | 43.4 | 3.7 | **1,525** | 0.33 | 128 |
| sims64 | 64 | 64ch×5L | 256 | 256 | 9,952 | **672.4** | 3.6 | **98** | 0.41 | 157 |
| net128×10 | 32 | 128ch×10L (3.46M) | 256 | 256 | 10,241 | 185.5 | 12.2 | 355 | 0.35 | 134 |
| **net256×20(dlshogi 級)**※ | 32 | 256ch×20L (24.3M) | 256 | 256 | 9,641 | 601.9 | 58.5 | **109** | 0.28 | 101 |
| batch128 | 32 | 64ch×5L | 128 | 256 | **5,775** | 67.7 | 1.7 | 490 | 0.45 | 89 |
| batch512 | 32 | 64ch×5L | 512 | 256 | 12,992 | 337.4 | 8.1 | **392** | 0.35 | 276 |
| steps128 | 32 | 64ch×5L | 256 | 128 | 5,196 | 63.4 | 2.2 | 517 | **0.057** | 17 |
| **steps512(AZ 論文値)** | 32 | 64ch×5L | 256 | 512 | 12,617 | 283.1 | 7.5 | 466 | **0.74** | 371 |

※ net256×20 は `training_batch_size=1024` だと**学習フェーズで真の OOM**(+7.16GiB 要求、selfplay は完走)→ **256 に縮小して完走**(3 iter, iter2-3 平均)。

**軸ごとの所見(実測)**:
1. **sims が最も高価で超線形**: 16→32→64 で 1,525→482→98 states/s(**倍化ごとに ~3-5× のコスト**、線形の 2× を大きく超える)。【推測】mctx は木の全ノードに env state(盤面+合法手 mask+cache)を embedding として保持し、sim ごとに木を traversal する=sims↑ で木 buffer への gather/scatter と traversal 段数が同時に伸びるため。→ **Gumbel の「少 sim でも方策改善」が単機ではそのまま throughput の生命線**(073 の期待を実測で裏書き)。
2. **net は FLOPs 比で大幅に割安**: 64ch×5L→128ch×10L は FLOPs ~8× なのに selfplay は 1.36×(482→355 states/s)、**dlshogi 級 256ch×20L(FLOPs ~64×)でも 4.4×(482→109 states/s)で回る**(VRAM 9.6GiB)。**selfplay の律速は net forward でなく env.step(将棋の合法手生成)**が大きく、大ネット化の限界コストは見かけより小さい。学習側(train s/iter)は FLOPs 通りに伸びる(2.3→11.1→57s)。
3. **batch は 128 で飽和**: 128/256/512 で 490/482/392 states/s=**GPU 並列はすでに使い切り**、batch↑ は VRAM(5.8→9.3→13.0 GiB)だけ食って逆に遅い。VRAM の主配分は batch×steps の軌跡バッファ+木バッファ。
4. **steps(窓長)は品質の軸**: steps128 は速いが**窓内終局率 5.7%=value 学習が事実上不能**(将棋の対局が窓に収まらない)。steps512(AZ 論文値)は 12.6GiB に収まり、有効率 0.74。**value 有効サンプル/秒では steps512(466×0.74≈345/s)が steps256(482×0.47≈224/s)を上回る**=フル将棋は窓 512 が正解。
5. **OOM の実態**: `XLA_PYTHON_CLIENT_PREALLOCATE=false` 下で「4GiB chunk 確保失敗」の E ログは**良性**(allocator が小 chunk で再試行し実行は継続、074 と同じ noise)。**真の OOM は net256×20 の学習フェーズ**(training_batch_size=1024 の fwd/bwd で +7.16GiB 要求→死亡)→ **training_batch_size=256 に落として解消**(=16GB では大ネットは selfplay より学習 minibatch が先に詰まる)。

**単機動作点(合格条件①)**: **フル将棋 1 iter ≈ 4.8 分(selfplay 283s + train 7.5s)、VRAM 12.6GiB、有効サンプル率 74%、466 states/s** @ sims32 / 64ch×5L / batch256 / steps512。= **1 日 ~300 iter、自己対局 ~11 万局/日、value 有効局面 ~2,900 万/日**。net128×10 に上げても ~1.4× 減で済む(env 律速のため)。

**どうぶつ将棋比(JAX 静的形状の実地検証、075 の学習の続き)**: 同 sims・同 net で per-state **~8-10×**(3,867 vs 482/392)、加えて窓長が 64→512 で **8×** → 1 iter は 8.9s→290s の **~33×**。内訳は形状スケール(obs 36.5×・行動 16.6×・対局長)通りで、「JIT は最悪ケースへ padding する」の実測。なお VRAM は steps・batch にほぼ線形(5.2/9.3/12.6 GiB @ steps 128/256/512)。

## パート2 — warm-start 戦略 a/b/c

### a. ゼロ知識フル将棋(実測外挿)= 非現実
- AlphaZero 将棋の自己対局は **2,400 万局**(5,000 TPUv1)。実測から: おもちゃ設定(0.8M net+32sim, ~11 万局/日)ですら **~220 日**、**dlshogi 級 net(256ch×20L)+32sim だと ~1.3 万局/日 = ~5 年**、AZ の 800sim に近づけると sims は実測で超線形(倍化ごと 3-5×)ゆえ **数十 GPU 年〜**=AobaZero の「単機 ~100 年」(073)と整合。**単機ゼロ知識フル将棋は自前実測の外挿でも非現実と確定**。
- ただし (c) の目標は AZ 再現でなく「**単機で self-play RL が正 elo を足す**」実証。M3 の short RL(数十〜数百 iter=数時間〜1日)は上の動作点で**十分回る**。ゼロ知識が非現実なのは「ゼロから強くなるまで」であって「warm-start に上積みする」ではない。

### b. champion 蒸留 warm-start(推奨)— 小実験+設計論点
- **SFEN 橋の実測(m1_sfen_bridge.py, 隔離)= 完全動作**: `pgx.experimental.shogi.from_sfen/to_sfen` で ①startpos の合法手 **30/30** ✓・observation (9,9,119) bool ✓、②**ランダム対局から採取した 600 局面(持駒あり 475・成駒あり 377)で roundtrip 完全一致**(盤・持駒・手番・合法手 mask・observation の全 assert 通過)、③スループット **from_sfen ~1,800 局面/秒**(単プロセス、1M ≈ 0.2h)・**バッチ `_observe` ~48-68k 局面/秒**(1M ≈ 15-21s)。→ **蒸留セット構築の律速は橋でなく champion ラベル生成**(065 実測: d12 で 1M ≈ 0.44h/16 workers)。
- **データ経路は既存資産で完結**: champion ラベル(cp)は既存 gen-teacher/label_cost 系(065: d12 で 1M ≈ 0.44h/16w)、cp→勝率→[-1,1] は既存変換、policy ターゲットは同探索の bestmove(one-hot)or MultiPV soft(069 の `selfplay_label.py --multipv` 資産)。**USI 手→Pgx action id(27×81, cshogi 流 direction×81+to)の写像**は手番視点 flip 込みで ~100 行+テスト規模【推測、M2 で実装】。
- **天井再持込みの論点(066-072 の教訓)**: 067-072 の天井は「**αβ 評価の蒸留それ自体を強化ループにした**」ことによる(教師=champion 自身の eval が上限)。(c) では蒸留は**初期化のみ**に使い、以降の学習信号は **MCTS の方策改善(探索が作る新しい教師)**=天井の機構が異なる。M3 の合否がまさに「MCTS 方策改善は蒸留天井を破るか」の検証になる=(c) の意義は薄まらない。**注意**: RL 中も蒸留データを replay し続けると 067 の「アンカー=天井」構図が再現しうる→ M3 では蒸留 replay の比率を段階的に下げる設計を推奨。
- **ゼロ知識哲学との整合**: 教師=**自前 champion**(技巧2 の知識は既に champion 経由で内在)=外部知識は増えない。「ゼロから」は放棄するが「自前で閉じる」は維持(大会ルール上も既存の合法ルートと同じ建て付け)。

### c. 中間題材 curriculum = 不採用を推奨
- **Pgx に将棋系の中間 env(5五将棋等)は無い**(確定)。選択肢は (i) ミニ将棋 env を JAX 自作=Pgx shogi 本体 ~800 行級の JAX 実装+駒テーブル+検証で**週単位の実装コスト**、(ii) `gardner_chess`(5×5 チェス, baseline 有)で手法予行=**別ゲームゆえ重み転移不可**(手法学習は M0 で済んだ)、(iii) 「フル将棋×縮小 config」で段階化=**追加実装ゼロ**でパート1の動作点がそのまま使える。
- (iii) は b と直交でなく包含される(小 net・少 sim で始めて上げる)→ **独立の「中間題材」は作らない**。

### 比較表と推奨
| 案 | 単機現実性 | 実装コスト | (c) 目標整合 | ゼロ知識哲学 |
|---|---|---|---|---|
| a. ゼロ知識フル将棋 | ×(数十 GPU 年) | 低 | ×(実証に届かない) | ○ |
| **b. champion 蒸留 warm-start** | **○(蒸留=時間規模、RL=日規模)** | 中(USI 手→action 写像+蒸留学習) | **◎(M2/M3 の設計そのもの)** | △(自前で閉じる) |
| c. 中間題材 | △ | 大(env 自作)or 転移不可 | △(遠回り) | ○ |

**推奨 = b**(最終判断は Keiichiro)。M2 へ: ①蒸留セット(1-5M 局面: SFEN+cp+bestmove)→ ②AZ ネット(まず 128ch×10L)を supervised 事前学習 → ③net→USI 橋で champion(αβ)と SPRT(=「同じ知識で MCTS は αβ に勝るか」の直接検証)→ M3 で short RL が warm-start に elo を足すか。
**M2/M3 の動作点メモ**: RL 用は steps512+batch256+sims16-32 を推奨(value 有効率 0.74)。**net128×10 × steps512 の組は未計測**(steps512 で +3.3GiB, net128×10 で +0.9GiB の実測から ~13-14GiB と外挿=収まる見込みだが M2 冒頭で 1 点確認を)。蒸留(supervised)は selfplay 不要ゆえ VRAM 余裕、大 minibatch 可。

## 詰まりどころと回避(実測)
- **net256×20 + training_batch_size=1024 は学習フェーズで OOM**(+7.16GiB 要求)。selfplay は完走しており、**training_batch_size=256 で解消**。allocator の fragmentation 警告には `TF_GPU_ALLOCATOR=cuda_malloc_async` の示唆もあり(未検証)。
- 「Failed to allocate 4.00GiB」E ログは prealloc 無効時の**良性 noise**(074 と同じ)。真の失敗は Python 例外(`RESOURCE_EXHAUSTED` の JaxRuntimeError)で判別。
- 計時は **`jax.block_until_ready`** を挟まないと async dispatch で selfplay 時間が train 側に紛れる(train_m1.py で対策)。
- 橋検証の初回、startpos が合法手 50 と出て Pgx のバグを疑ったが、**真因は自作テスト定数の SFEN が先手歩の段を欠いていた**(Pgx は正しく、`to_sfen(env.init)` は正準 SFEN と完全一致)。既知の正解(合法手 30)を持つ検証だったから即検出できた=ground truth 付き検証の効用。

## M1 判定 = **PASS**
- 単機動作点を数字で特定(上記)。a/b/c 比較+推奨 b を提示=M2 の設計判断が可能。ゼロ知識非現実を実測外挿で提示し、目標は「warm-start+short RL で正 elo」に再設定可能。

## 規律・メタ
(c) 作業は隔離環境(`~/task074_frameworks/pgx/examples/alphazero/`: train_m1.py / m1_run_point.sh / m1_grid.sh / m1_parse.py / m1_sfen_bridge.py / m1_logs、全て非コミット)。binbo リポジトリには本 report+STATUS/BACKLOG のみ。**champion・Rust 無改変**(`cargo test` 無関係=実行不要)。長時間ランは `scripts/detached.sh`+marker。**push は人間承認後。**
