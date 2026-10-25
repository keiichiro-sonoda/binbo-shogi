# PROJECT MAP — Task 057-085 全体地図(「技巧2を超える」挑戦の総括)

> **この文書の役割**: 057-085 の29タスクを一枚にまとめた正本。半年後の自分／新セッションのエージェントが**この文書だけで**「どの経路が実証済みで閉じたか・なぜ・唯一何が開いているか」を把握できる状態にする。全主張に根拠タスク番号(`reports/taskNNN…`)を付す。数字は各 report が正。外部主張は web 一次情報で裏取り(末尾 References)。作成 = Task 086(2026-07-11、文書のみ)。

---

## 1. エグゼクティブサマリ

- **到達点(現 champion)** = `task064_32m_lr1e-5`: 技巧2@d8 の探索cp を 32M 局面蒸留した HalfKP NNUE(256x2-32-32, tied QAT)。061→064 で **+188〜240(lr 修正)→ +125(8→16M)→ +141(16→32M)** を積み上げ、**この情報系の実質上限**に到達(val_corr 0.886 ≈ 技巧2@d8 ラベルノイズ天井 ~0.90、過学習 gap 底 ~26k が 16→32M で不動 = 蒸留の情報が枯れた [064])。
- **中心的結論**: **「系に入っている情報が技巧2@d8 で閉じている限り、いかなる加工でも技巧2@d8 が上限」**。これを**3つの独立経路**で実証した:
  1. **自己ラベルの再サンプリング**(replay 比/量/多様性/深さ、067-071)→ champion 自己ラベルは技巧2@d8 を超えない。
  2. **より深い教師**(技巧2@d12, 072)→ クラッシュ + d8 で飽和(二重に死)。
  3. **AZ value での教師平滑化**(084-085)→ 2M で +237 も **32M で −36 に反転**(小データ正則化であって教師改善でない)。
- **唯一開いている道**: **AZ 自己対局 RL の内部改善は本物**(単機で warm-start を +133 有意超え [080]、探索で +241 に増幅 [083])= 系に**新情報**を入れうる唯一の源泉。ただし ①内部→外部(αβ)非転写、②内部強さ→教師品質 非転写、の二重の壁があり、**規模・アーキ(第2章 = Transformer + ゼロ知識で真に強い AZ)**が要る。§5 参照。

**一行で**: 「技巧2@d8 を蒸留・加工して超えることは全経路で閉じた。超えるには技巧2@d8 に**無い情報**が要り、それを単機で作る唯一の候補が AZ 自己対局 RL(だが現状は別レジームの壁)。」

---

## 2. 経路の地図(何を試し、何が閉じ、なぜ閉じたか)

各経路: **【試したこと / 結果(数字+根拠) / なぜ閉じたか】**。

### (A) 蒸留の最適化 — 有効だったが天井に到達(056-065)
- **bridge 整備(056-060)**: 自前 NNUE を YaneuraOu nn.bin に **bit-exact(0cp)**で載せる経路を確立。付番 remap の silent 故障を negative control 付きで潰し(1620 局面 mismatch 0 [057])、量子化残差を 536cp→~25cp(~20×)に(QAT [058])、逆経路 writer で 0cp 書出(tying コスト≈0 [059])。**判定基盤 = 探索固定・eval のみ変数・0cp ゲート**の確立([060] `yane-vs-yane`)。
- **eval 強化(061)**: 教師据置で**学習 lr だけ**を下げると **+188〜240 elo**(lr1e-4 +208 / lr3e-5 +240.8 / lr1e-5 +188.5、val_corr 0.833→0.858)。=Task 060 本命は lr 過大で過学習だった。
- **データ量(062/064)**: 8→16M **+125**(val 0.858→0.875)、16→32M **+141**(val 0.875→0.886)。**棋力は逓減せず**だが **val は逓減**(天井 0.90 接近、gap 底 ~26k 不動)。
- **なぜ閉じたか**: 064 で **gap 底 = 技巧2@d8 ラベルの不可避ノイズ = 教師の天井**が可視化。lr・量・量子化は「生徒側」の最適化で、**教師の情報量そのものは増えない**。→ 次は教師を上げるしかない(=B/C/D)。

### (B) 自己強化(αβ, champion 自己ラベル)— 全経路パリティ以下(066-071)
- **機構(066)**: champion self-play → champion@YaneuraOu@depth で cp+WDL 自己ラベル → warm-start 再学習 → SPRT 採用ゲート + 技巧2@6 アンカー。機構は end-to-end で回るが 1-cycle(replay 1:1)は後退(gen1 棄却)。
- **一変数掃討**: replay 比(067: 0.3%→15.6% で **パリティ回復・超えず**)/ 新規量(068: 増やすと**逆効果**)/ 方策多様性(069: 温度で**悪化**)/ 深いラベル d14(070: **質↑なのに結果↓**=デカップル)/ d8 replay 全除去(071: **3世代で最悪化**)。
- **なぜ閉じたか(統一機構)**: **champion の自己ラベルは技巧2@d8 を超えない**(生徒が自分の盲点を強化)。技巧2@d8 replay は **load-bearing**(忘却防止+知識質)で、d8 を薄める/深くする/多様化する操作は全て d8 との矛盾で悪化。=**再サンプリングでは新情報は生まれない**。

### (C) より深い教師(技巧2@d12)— 二重に死(072)
- 技巧2 は **depth≥~8 で warm-state EOF クラッシュ**(gen 生成で ~80% crash)。加えて **cp が d8 で実質飽和**(d12≈d8, median 差 ~71cp、生徒棋力は上がらず)。無偏 cold 再ラベルは ~46h/1M + crash 14% で非現実。
- **なぜ閉じたか**: 技巧2 という教師自体が d8 付近で頭打ち。深くしても**新情報が乏しく、かつ生成不能**。→ 技巧2 より根本的に別の教師 = DL 系(D)へ。

### (D) AlphaZero アーク — 内部改善は本物、だが外部/教師に非転写(073-083)
- **調査→選定→疎通(073-075)**: 単機ゼロ知識将棋 AZ は非現実(AobaZero ~100 マシン年)だが Pgx(JAX GPU ベクトル化)+ Gumbel 少 sim + warm-start なら射程 [073]。Pgx 選定 [074]。**M0**: どうぶつ将棋で単機 AZ ループが回り学習済 baseline 超(win 0.008→0.5-0.65)[075]。
- **単機動作点(M1, 076)**: フル将棋 = sims32/64ch×5L/batch256/steps512 で **1 iter ≈ 4.8 分・466 states/s・VRAM 12.6GiB**。sims は超線形に高価、net は割安(env.step 律速)。
- **warm-start + USI 橋(M2, 077)**: 技巧2@d8 cp を AZ value に蒸留(val_corr **0.90**)+ USI エンジン化(SFEN↔Pgx action 写像を cshogi と 2000 局面照合)。vs champion **0-0-200**(sims128 ≈ 101ms)。
- **★M4 = (c) の中心的成果(080)**: 価値のみ warm-start(WS1)はループで**パリティに吸着**して止まる(gen300/400 とも ±0 [078/079])。**方策も warm-start(WS2 = 技巧2 bestMove16 蒸留, top-1 34%)**すると、ループが **gen150 vs WS2 = +133 elo(SPRT accept_h1)**。=**単機の自己対局 RL が正 elo を生む**を実証(WS2 vs WS1 = 89-0-0 で方策温めの寄与 ≥+586)。
- **限界(M5-M7a, 081-083)**:
  - **M5**: 延長は頭打ち(gen200 +109 ≈ gen150 +133)[081]。
  - **M6**: 容量 128ch は **warm-start の質を上げる**(WS2-128 corr 0.905/top-1 0.400、実棋力 +235)が **loop の増分は容量不変(+126 ≈ +130)**。外部 champion は 0-6-194(**アーク初の draw**)[082]。
  - **M7-a**: 対局 sims↑ で外部床の正体切り分け → **net 品質の床**(sims 128→2048 で champion 相手フラット)。だが**探索は内部を増幅**(gen100 vs WS2-128 = +126@128 → +241@512)。=**探索は効くが AZ の内部優越は外部 αβ に非転写** [083]。
- **なぜ(champion 超えに関して)閉じたか**: 内部 elo(WS 系列内)は本物だが、**外部 αβ champion に転写しない**(規模・レジームが違う)。champion 超えは第2章。**ただし内部改善そのものは新情報の唯一の実在源泉**(A/B/C には無い性質)。

### (E) AZ value を NNUE 教師に転用(nodchip 式)— 2M で +237 も 32M で反転(084-085)
- **2M 相対(084)**: 教師 cp を技巧2@d8 → AZ value(WS2-128)に上書き、同一 cache/設定で NNUE 学習 → **WS2-128 value 教師は技巧2@d8 探索cp より +237 elo 良い**(accept_h1, 79.7%)。gen100 value は +47。**★逆説**: ループ後の gen100 は value が圧縮(std 1471→698)し**教師として −145 劣**(WS2-128 > gen100)=**強いプレイヤー ≠ 良い教師**。
- **32M 本走(085)**: 全量 32M を WS2-128 value 上書き → champion 構成で学習 → **vs champion −36 elo(有意に負け)**。**2M +237 → 32M −36 の反転**。
- **なぜ閉じたか**: **平滑化(denoise)は小データ正則化であって教師改善ではない**。①小データは技巧2 探索cp のノイズを生徒が濾せず AZ の平滑化が大勝、②大データは**生徒が自己平均化**してノイズ消失 + **AZ 平滑ラベルの不可逆圧縮損失(tanh 飽和・有限容量)が支配** → 反転。**★nodchip との差(§5, References)**: nodchip は **DL 水匠(独立に強い top 級 DL エンジン)の value** で上書き = **新情報の注入**ゆえスケールで +R100。我々の AZ value は**技巧2@d8 の蒸留物 = 新情報なし**ゆえ大データで飽和・反転。=**教師上書きが効くのは上書き元が真に強い時であって、平滑化そのものではない**。

---

## 3. 確立された原理(自前実験で得た法則)

| 原理 | 根拠 |
|---|---|
| **教師の天井は情報源に付随し、加工(lr/量/量子化/再サンプリング/深さ/平滑化)では動かない**。動かすには情報源そのものを替える(=新情報)。 | 064(gap 底不動)/ 067-072(再サンプリング全否定)/ 084-085(平滑化は正則化) |
| **「対局の強さ」「教師としての良さ」「内部 elo」は互いに独立で、一方から他方へ転写しない**。 | 083(内部優越は外部 αβ に非転写)/ 085(強いプレイヤー gen100 は悪い教師) |
| **探索cp のノイズは小データでは平滑化で正則化緩和できるが、大データでは生徒の自然平均化に吸収され、平滑化の圧縮損失だけが残る**。 | 084(+237@2M)→ 085(−36@32M) |
| **val(教師ラベルへの近さ)は棋力の代理にならない。判定は常に対局(SPRT)**。 | 061(val 小改善が桁違い elo)/ 070(質↑結果↓)/ 084(gen100 val 高だが弱い) |
| **探索は net の内部優越を増幅するが、別レジーム(αβ)には転写しない**。53× wall-clock でも外部床不動。 | 083(sims 128→2048 フラット vs champion, 内部 +126→+241) |
| **判定インフラの規律 = 探索固定・eval のみ変数・両側 0cp ゲート(RED で abort)・一変数**。全 SPRT がこの型。 | 060-085 全域 |
| **long-run は detached.sh(setsid)+ watchdog + marker で境界耐性**。fresh Adam × default lr は蒸留初期化を 1 iter で全壊(監視必須)。 | 063 / 078(lr wipe を val_corr 監視が検出) |

---

## 4. 資産台帳(残っていて再利用可能なもの)

**モデル(全て `~/binbo-shogi-data`, 永続・非コミット)**
- `models/task064_32m_lr1e-5` = **現 champion**(技巧2@d8 32M 蒸留, val_corr 0.886)。`eval_dirs/task064_32m_lr1e-5/nn.bin` = SPRT baseline。
- `~/task074_frameworks/pgx/examples/alphazero/m6_ws2/ws2.ckpt` = **WS2-128**(価値+方策 warm-start, corr 0.905/top-1 0.400)= 最良の「教師用/warm-start 用」AZ ckpt。
- `m6_ckpt/000100.ckpt` = gen100(128ch, 自己対局 100 iter)。`m4_ws2/ws2.ckpt` = WS2-64。自前探索の到達点は Task 055 時点の `src/`。

**基盤(コード)**
- 判定: `yane-vs-yane`(eval-only SPRT)/ `yane-vs-gikou`(技巧2@6 アンカー)/ `src/sprt.rs`(trinomial)。driver `scripts/eval_vs_eval.sh`。
- 橋: NNUE↔nn.bin(`trainer/binbo_to_yane.py` + `scripts/binbo_to_yane.sh` 0cp ゲート)/ AZ↔USI(`m2_usi.py` mctx)/ SFEN↔Pgx obs(`m4_distill_data.py`)。
- 教師/学習: `trainer/prepare_data.py`(canonical-shared cache)/ `trainer/train_nnue.py`(tied QAT, `--eval-npy` 教師上書き)/ `trainer/dlshogi_rescore.py` + `az_rescore.py`(隔離)= value 上書き rescore。
- 運用: `scripts/detached.sh`(setsid+marker)/ watchdog / corr+top-1 監視 / `trainer/gold_analyze.py`(教師品質を学習と独立に測る)。
- (c) 隔離資産の在り処 = `~/task074_frameworks/pgx/examples/alphazero/`(m2_*/m3_*/m4_*/m6_* 全て非コミット、作者のメモ)。

**データ**
- `~/binbo-shogi-data/teacher/run1m…run16m_f` = 技巧2@d8 hcpe **32M**(cp+bestMove16+WDL)。
- `caches/shared_32m`(生技巧2 32M cache)+ `az_ws2_32m`(WS2-128 平滑化上書き, 085)。各 2M/8M/16M cache は /tmp 由来で再生成要(~15-58s)。

**知見文書**
- `docs/nnue_yaneura_export.md`(bridge 設計・§8-12)/ `docs/long_running_jobs.md`(detached 運用)/ `docs/self_reinforcement_design.md`(自己強化設計)/ `docs/USED_COMPONENTS.md`(ライセンス帳簿)/ `reports/task057-085`(各タスク正本)。

---

## 5. 開いている道と、その正直なコスト

### 第2章: Transformer + 規模で「真に強い AZ」を作る(→ 良い教師 or 直接強い)
- **🟢 T1 実測(Task 087, `reports/task087_t1_transformer_warmstart.md`)= GO**: 等パラメータ(3.585M vs CNN 3.462M)で Transformer が warm-start 品質(arch-only val corr +0.020/top1 +0.056)・内部棋力(vs CNN-b256 34-4-2 ~90%)・外部転写(champion +140 / gikou +300、ともにアーク最良)の**全軸で CNN 超え**。しかも **推論は CNN より速い**(sims128 110 vs 145ms)=**アーキは登山を重くしない**(大食いは学習 backward のみ=batch 1/4、自己対局生成は逆に速い)。→ 第2章の土台は Transformer で確定。**留保**: champion は warm-start のみでは依然不倒(37/40 敗)、床破りは loop+規模(T2+)。
- **🟡 T2-scale 実測(Task 089, `reports/task089_t2_scale.md`)= 規模は「同一 recipe/予算」では逆効果**: params を 3.6M→7.6M→14.4M(2M/8ep/lr1e-3 固定=一変数)で振ると **warm-start 品質も棋力も全軸で単調悪化**(val 0.936→0.918→0.829、champion −436→−511→−2400、gikou −301→−449→−727)。機構=**3.6M 用 recipe が大 net に転写しない**(7.6M=undertrained[gap~0=枯渇でない]、14.4M=lr 過大で発散)。標準ネット=3.6M(d240/L5)。**留保**: equal-budget の結論。
- **🟢 T2-scale-recipe(Task 090, `reports/task090_scale_recipe.md`)= 089 の留保を潰した: 内部の scale-hurt は undertraining、外部は非転写**: 7.6M に epoch を与える(8→~14)と **内部 vs 3.6M が −158→−26(互角)**=089 の「規模が内部を悪化」は **undertraining artifact**(大 net に 8ep 不足)、適正 epoch で規模は内部無害。**だが外部は非転写**: champion **−511 のまま**(epoch 倍でも不動)<3.6M −436、gikou −372<−301。→ **内部互角化しても外部 αβ では 7.6M<3.6M**(M6/M7/T1 の非転写の壁)。**∴ scale は単機の外部目標のレバーでない(内部を害しもしない)。標準=3.6M 据え置き、新情報の loop(T2-loop)を先に**。Phase2(lr ladder)未実施=外部非転写ゆえ低決定価値。
- **🔴 T2-loop 完全二塔(Task 091→094)= 「value 凍結・policy 自己対局ループ」は外部を伸ばさない(3タスク連続)**: WS-T を完全二塔化し value 塔を byte-for-byte 凍結、policy 塔だけ自己対局ループで更新する構成を、判定器を替えながら検証した。**091**(`reports/task091_t2loop_frozen.md`)= 凍結は GREEN だが最良教師は 400局 0 elo で WS-T 超えず FAIL。**092-0**(`reports/task092_0_nnue_bridge_calibration.md`)= 旧 NNUE 補助 head 橋は argmax vs random の最大級方策差すら検出できず bridge null =判定器を **gikou@6 直接対局**へ刷新。**093**(`reports/task093_t2loop_external_anchor.md`)= その直接対局で iter25/50/75 の外部が **単調悪化**(paired ΔElo −74/−108/−148、傾き −47.7/25iter)。**094**(`reports/task094_policy_anchor.md`)= WS-T supervised replay を policy update に混ぜる(r=0.25/0.50)と 093 の悪化は **反転・停止**(ΔElo −74→+21/+79)したが、両arm CI が 0 を含み **WS-T 超えの正の転写なし**(FORGETTING_STOPPED_NO_POSITIVE_TRANSFER)。**∴ sims32・訪問分布教師の完全二塔 policy-only ループは、アンカーを足しても外部 αβ 棋力を伸ばさない**=§5 が「唯一開いている道」と呼んだ内部改善の新情報源は、この構成では外部へ転写しない。**まだ試していない一変数**=policy 教師の質(sims↑48/64・soft 訪問分布)、value 側に新情報を入れる経路(凍結解除+WS-T value 蒸留正則化=共有幹案)。
- **🟡 r=1.0 対照(Task 095, `reports/task095_selfplay_control.md`)= 「+79 は追加 supervised の回収」説を棄却、ただし混合の優位は未確立**: anchor 率 r を 1.00(512update 全部 supervised=自己対局ゼロ)まで振り、4点用量反応を同一 opening で引いた。ΔElo vs WS-T = **r0 −74.0 / r0.25 +21.3 / r0.50 +79.1 / r1.0 −7.6** = **単調でなく r=0.50 に内点最適**。r=1.0 は supervised update が2倍なのに 87 Elo 低いので、094 の +79 を「自己対局を捨てた分の supervised 回収」で説明することはできない。**残る説明は「混合そのものが効く」だが、選択汚染ゼロの主判定 Δ_selfplay(block2+3)= +37.8 [−27.0,+103.5] は 0 を跨ぎ未確立**(block2 は +80.9 で再現、block3 は −3.4 で非再現)。機構面では r=1.0 が**学習側 top-1↑/未学習 val_top1↓** の過学習を示し、**同じ 2M を回す段が枯れている**ことを直接示した(§3-C・085 の「新情報が要る」と整合)。∴ **第2章の残る唯一の未解決点は「錨と自己対局の混合が本当に効くか」**で、決着には 2,430局/系列(~11h)。
- **🟢 決着測定(Task 096, `reports/task096_transfer_decision.md`)= POSITIVE_TRANSFER_CONFIRMED(限界的)= §5 が「唯一開いている道」と呼んだ内部改善が、初めて外部 αβ に転写した**: 事前登録(commit を実行前に置き optional stopping を封印)で r=0.50 − WS-T を汚染 block1 除外の **1,224 clean ペア(2,448局、ABBA 順10ブロック)** で測り **+28.65 [+3.79, +53.51]**。**アーク初の外部正転写**であり、§3-D/§4 の「内部改善は本物だが外部 αβ に非転写」という壁に**初めて穴が空いた**。ただし ①CI 下限 +3.79 と限界的 ②層別 bootstrap は 0 を跨ぐ(I²=0.0 で異質性の証拠は無く、k=10 クラスタ再標本化の保守性と読める)③検出力61%の限界的有意ゆえ点推定は上方偏り ④**各アームの学習は単一走**ゆえ主張は「このネットが自身の初期値を超えた」に限定。選択汚染の実測値も得られた(block1 単独 +79.12 → clean +28.65 = 上方バイアス約+50)。5点用量反応は `NON_MONOTONE`(−74.0/+21.3/+79.1/−15.4/−7.6)で **r=0.50 での尖り**、r>0.50 側の形は ±93/点の解像度では決定不能。**∴ 次の本丸は「この利得を champion(NNUE)へ運べるか」**=value 凍結の緩和(WS-T value 蒸留正則化の λ ラダー)+ 084方式の value 相対判定で運搬路を測る。**085 の 2M→32M 反転ゆえ 2M 勝ちは 32M 確認の資格まで**、かつ n=1 の解消(別シード再走)が前提。
- **🟢 再現走(Task 097, `reports/task097_seed_replication.md`)= REPLICATED。096 の正転写は単一学習軌跡の産物ではなかった**: 学習系シードだけを変えた第2走で、**2走統合(2,504 clean ペア)= +27.42 [+10.05, +44.79]**、しかも**要求されていなかった run2 単独でも +26.19 [+1.86, +50.42]** と 0 を除外。**走間差 −2.45 [−37.4,+32.4] で2走は区別不能**、異質性は走水準・ブロック水準とも I²=0.0。∴ §3-D/§4 の「内部改善は本物だが外部 αβ に非転写」という壁の破れは**再現性のある現象**として確立した(アーク7タスク目)。**ただし留保は定性/定量で分かれる**: 「単一軌跡の産物でない」は解消、「r=0.50 レシピ一般の効果量に区間を与える」は未解消(k=2 では τ² が 1df で推定不能。k≥5 が必要)。効果量として引くべきは選択汚染なしの **run2 +26.19**。**残る唯一の本丸は運搬路**(ループ内のネットが強いことと、そこから製品 NNUE が強くなることは別問題)。
- **🔴 運搬路スクリーニング(Task 098, `reports/task098_value_carriage.md`)= NULL_CARRIAGE_WALL。その本丸が閉じた**: value 塔を共有幹化し `λ·MSE(v,v_WS-T)` で拘束したところ、**value は動き(m4_data val で 59.2cp、事前登録の30cp基準の2倍)、しかも 082 の gen100 型圧縮は起きなかった**(std比 1.0014、corr_teacher 0.9357→0.9368)。だがその value を教師にした NNUE 学生は WS-T value の学生と**区別できなかった**(400ペア800局で **−0.87 [−23.92,+22.18]**)。**機構はラベル統計が示す**: 候補と WS-T は技巧2 との corr が 0.958/0.957、cp std 1502/1500、|Δ|median 201/202 と**実質同一**で、59.2cp のドリフトは「技巧2 を同じ精度で近似する別の点」への移動=**同じ関数まわりのノイズであって新情報ではない**。∴ §5 の「★本質的動機(085 + nodchip [R1])= 真の教師改善には新情報が要る」が、value チャネルで直接再確認された。**方策チャネルは 092-0 で bridge null、value チャネルは本タスクで壁 ⇒ 既知の運搬路は両方とも閉じている**。単機ループの外部利得(+26〜+27、097 で再現)は実在するが製品へ運べない。残る道は「新情報を作る」(value に独立教師信号+λ錨=082 に鎖を付ける案。本タスクで崩壊させずに動かす λ 帯が実測できた/より強い教師)か、BACKLOG の既知正方向への資源復帰。
- **🔴 探索裏付き value(Task 099, `reports/task099_search_label.md`)= REGRESSION。最後の未踏経路 (a') も閉じた**: 098 が「生 value ヘッドに新情報なし」を示した後、残る運搬路は「探索が方策の賢さを value の数字へ翻訳する」経路だけだった。**097 run2 の value 塔は WS-T と bit-identical** なので、②WS-T+探索 と ③097run2+探索 は **value ヘッド同一・違いは探索を導く policy だけ**という理想的な対照が組めた。結果、**バイアスフリーの主量 d32 = +2.70 [−3.66,+7.79]**(②③は gikou からの距離まで同一 — |Δ|vs gikou 242.653 vs 242.672 — なので gold の錨バイアスが完全に相殺する)、②③の直接距離は median 16.6cp・**corr 0.998**、しかも **sims 32/128/512 の全てで CI が 0 を含む**。⇒ **外部 +26 Elo 強い policy を積んでも、探索が到達する value の数字はほぼ同じ**。探索そのものは value を大きく動かす(①から 188.7cp、098 の 59.2cp「殻の内側」とは質的に違う)が、動いた先は gikou 多様体から離れる方向(corr 0.958→0.952)で、gikou 蒸留物を参照とする gold ではその良否を**原理的に測れない**。∴ **091→099 で、ループの外部利得は実在し再現するが、方策(092-0)・生value(098)・探索裏付きvalue(099)の既知3経路すべてで製品 NNUE へ運べないことが確定した**。残るのは「探索ラベル一般が gikou 由来 gold より良いか」を学生試験で直接測る道(②で足りる=ループ路線の問いではない)か、資源復帰。
- **🟢 探索裏付き value の学生試験(Task 100, `reports/task100_search_student.md`)= SEARCH_LABEL_TEACHES。099 が gold で測れなかった問いに gold を介さない計測器が答えた**: 099 の ②(WS-T+Gumbel MCTS S=32 の root value)を教師にした NNUE 学生が、生 value 教師の学生(同じ 1M 局面・同じレシピ・両腕新規学習)に **604-2-194、+196.7 Elo [+170.0, +225.0]**。hard gate 全通過、4 worker・両色すべてで 70% 超。∴ 099 の REGRESSION は gold の錨バイアスの産物で、**探索が値を gikou 多様体から離す方向は学生にとって良い**。ただし効果量は 084 の +237@2M と同じ桁で、085 で 32M 反転した前例があり、探索ラベルは分散がさらに小さい(1503→1446)ため**正則化説と情報説は未分離**。事前登録どおり 1M の勝ちは **2M/32M 確認の資格のみ**(Task 100b = 2M 確認を同夜起動、)。099 の d32≈0 は不変=これは「AZ 探索一般」の性質であり、ループ路線の運搬路が開いたわけではない。
- **🟢 探索ラベルの規模曲線と機構(Task 100b/102, `reports/task102_scale_curve.md`)= 2M +155.0 [+130.4, +180.6]、4M +124.0 [+99.5, +149.8]。スケール同変対照(生ラベル ×0.9622)は +8.3 [−15.2, +31.8] で null**: 利得は eval スケール/木の形ではなくラベルの中身にあるが、**倍加ごとに −36 Elo 縮む**(log2 線形外挿 8M +86 / 16M +50 / 32M +13)。084→085 型の「小データ正則化」成分を含むのは確実だが減衰は 085 より緩く、32M で残るかは 8M(≈6h GPU)以降の実測待ち。探索ラベラーは単一プロセス ~190 局面/s(GPU 並列化は無効)、rescore は part 再利用で規模点ごとに差分だけ計算できる。
- **🟢 8M 点・sims ラダー・絶対値(Task 103, `reports/task103_scale_8m_sims_absolute.md`)= 8M でも +107.1 [+83.2, +131.9](減衰 42→31→17 と鈍化、漸近 ≈ +70)。S=8 でも +222 [+194, +252]@1M = S=32 と同等以上(SIMS_INSENSITIVE)**: 利得は深い探索でなく**一手先の混合(max_visits 1.25)**で生じ、データで縮むが消えない形。絶対値では相性成分を差し引き +40〜+90(champion フレーム +88、技巧2@6 フレーム +42。4M 探索学生は技巧2@6 と互角、champion には −248)。**S=8 なら 32M の rescore ≈ 9.3h** = 32M 決戦(champion 構成で SPRT)が 1 日で回せる。技巧2@d8 に無い情報を単機で作る候補として、AZ ループでなく AZ 網の一手先探索ラベルが最も安価に実在を示した。
- **効く根拠**: ①**AZ 内部改善は本物**(080 +133、083 探索で +241 に増幅)= 新情報の実在源泉。②**warm-start の質は外部を動かした唯一の軸**(082: WS2-128 が champion 相手にアーク初の draw/勝ち、**087 で Transformer が +140 に拡大**)。③**Transformer の容量効率**: ChessFormer は AlphaZero 比 **8× 少 FLOPS で凌駕・GM 級に 30× 少 FLOPS で匹敵**、Lc0 の transformer BT4 は最良 CNN 比 raw policy **+~300 elo**(少パラメータ)[R2]。**087 で自環境でも sims128 推論 1.3× 速を実測**=容量効率の方向が実在。
- **向かい風(正直に)**: ①単機ゼロ知識将棋 AZ は **AobaZero で ~100 マシン年**、dlshogi 作者でも **warm-start + 自宅 PC 2年+** [R3, reports/task073]。②**内部→外部 非転写**(083)= 内部 elo を上げても αβ champion に効く保証がない(**087: 内部 +436 が champion では +140 に圧縮=正だが弱い転写**)。③**gen100 の教師劣化パラドックス**(085)= ループで強くしても教師品質は上がらない可能性。④実装規模は **087 で解消**(TransformerNet を dm-haiku 内で定義、既存パイプライン無改変で arch 分岐 1 箇所=作り替え不要と判明)。
- **★本質的動機(085 + nodchip [R1] の統合)**: 真の教師改善には**新情報**が要る。nodchip の +R100 は **DL 水匠(独立に強い top 級エンジン)** の value = 新情報だから効いた。我々の技巧2@d8 蒸留 AZ は新情報でないゆえスケールで反転した(085)。→ **技巧2@d8 に無い情報を単機・ゼロ知識で作る**のが第2章の唯一の意味。それが成れば「AZ value で NNUE 教師を上書き」が今度は**スケールで効く**(nodchip の再現)可能性がある。ただし上記向かい風ゆえ高コスト・不確実。

### 電竜戦準備(現 champion で出場)
- 現装備(champion + bridge + USI)で**出場は可能**。TODO = `docs/USED_COMPONENTS.md` の大会ルール原文を提出前に再確認(事前生成の自作教師→事前学習の自作 nn.bin が合法ルート、封印は学習済みモデルのみ)。

### BACKLOG 温存項目(低優先・格納)
- 64M 蒸留(逓減だが正の見込み、val 天井近い)/ Rust 探索の差分更新+SIMD(060 の 0-19-0 の主因=探索速度、大工事)/ 中間題材 curriculum(Pgx にミニ将棋なし=自作要)/ nnue-pytorch 学習側改善調査。

---

## 6. 年表(057-085、タスク→一行、`reports/` 参照)

| # | 一行 | report |
|---|---|---|
| 056 | 方向転換: 自前探索→YaneuraOu 土台、自作 NNUE で勝負 | task056… |
| 057 | forward parity: 付番 remap の silent 故障を潰す(0cp 手順確立) | task057… |
| 058 | QAT: float↔int 残差 536→~25cp(bit-exact 化) | task058_qat |
| 059 | writer: 自前 eval が実エンジンに 0cp で乗る(tying コスト≈0) | task059_writer |
| 060 | 探索器比較: 同一 eval で binbo 0-19-0 vs YaneuraOu(探索が桁違い) | task060_searchmatch |
| 061 | eval 強化: lr 修正だけで +188〜240 elo | task061_eval_tuning |
| 062 | 8→16M: +125 elo(データ律速を実証) | task062_data16m |
| 063 | detached.sh(setsid)で long-run 境界耐性 | (long_running_jobs.md) |
| 064 | 16→32M: +141 elo、**gap 底=技巧2@d8 天井が可視化** → champion | task064_data32m |
| 065 | ラベル生成コスト計測(cp は d16 まで飽和せず=蒸留の燃料あり) | task065_label_cost |
| 066 | 自己強化 1-cycle リハーサル(機構 OK, gen1 棄却) | task066_loop_rehearsal |
| 067 | replay 比: 15.6% でパリティ回復・**超えず** | task067_replay_ratio |
| 068 | 新規量↑: 逆効果(律速は量でなく質) | task068_newdata_volume |
| 069 | 方策多様性(温度): 悪化 | task069_policy_diversity |
| 070 | 深ラベル d14: 質↑なのに結果↓(デカップル) | task070_depth_d14 |
| 071 | d8 replay 全除去: 最悪化(d8 は load-bearing) | task071_self_anchor |
| 072 | 技巧2@d12: クラッシュ+d8 飽和=二重に死 → 技巧2天井 受容 | task072pre_gikou_d12 |
| 073 | (c) AlphaZero 転換の徹底調査(単機可行性) | task073_alphazero_survey |
| 074 | フレームワーク選定 = Pgx(+mctx) | task074_framework_selection |
| 075 | M0: どうぶつ将棋で単機 AZ ループ PASS(0.008→0.5-0.65) | task075_m0_dobutsu |
| 076 | M1: フル将棋動作点(4.8分/iter, 466 states/s) | task076_m1_throughput_warmstart |
| 077 | M2: 価値 warm-start(corr 0.90)+ USI 橋、vs champion 0-0-200 | task077_m2_warmstart_usi |
| 078 | M3: 価値のみループは warm-start 超えず(−110)、lr wipe 検出 | task078_m3_loop |
| 079 | M3延長: gen300 −11(パリティ吸引点)/ gen400 −15(確定) | task079_m3_extension |
| 080 | **M4 PASS: 方策 warm-start で +133 = 単機自己対局 RL が正 elo** | task080_m4_policy_warmstart |
| 081 | M5: 延長で頭打ち(gen200 +109 ≈ gen150) | task081_m5_extension |
| 082 | M6: 128ch で warm-start +235、loop 増分は容量不変(+126)、champion 初 draw | task082_m6_net128 |
| 083 | M7-a: 外部床=net 品質(sims 不効)、探索は内部を +241 に増幅 | task083_m7a_sims_floor |
| 084 | AZ value 教師上書き 2M: **+237**(WS2-128 > 技巧2@d8) | task084_az_value_teacher |
| 085 | 32M スケール: **−36 に反転**(平滑化は小データ正則化)= 天井堅持 | task085_32m_smoothed_champion |

---

## References(外部一次情報)

- **[R1] nodchip 教師 value 上書き +R100**: やねうら王公式「将棋AIの強化学習の基礎が根底から変わりそうな件」(2024-12-30) — NNUE 生成教師の探索cp を **DL 水匠(top 級 DL エンジン)** の value で上書き→ +R100。「DL 水匠の value のほうが NNUE で探索した評価値より優れている(学習に適している)」。続報「NNUE系でDL系の教師データを生成する」(2025-12)。https://yaneuraou.yaneu.com/2024/12/30/shogi-ai-may-transform-fundamentally/ ← **084/085 の直接の先行事例。上書き元が真に強い(=新情報)がゆえに効いた点が、我々の平滑化(085 反転)との差**。
- **[R2] Transformer 容量効率**: ChessFormer "Mastering Chess with a Transformer Model"(arXiv 2409.12272, ICLR 2026)= AlphaZero 比 8× 少 FLOPS で凌駕・GM 級 transformer に 30× 少 FLOPS で匹敵。Lc0 "Transformer Progress"(lczero.org, 2024-02)= BT4 が最良 CNN(T78)比 raw policy +~300 elo。https://arxiv.org/abs/2409.12272 / https://lczero.org/blog/2024/02/transformer-progress/
- **[R3] 単機ゼロ知識 AZ のコスト**(`reports/task073_alphazero_survey.md` に出典付き): AobaZero=単機ゼロ知識将棋 AZ ~100 マシン年 / dlshogi=warm-start+自宅 PC 2年+ / KataGo=AlphaZero 比 ~50× 効率化(囲碁)/ Gumbel AlphaZero=少 sim で方策改善保証 / Pgx=JAX GPU ベクトル化 10-100× / AlphaGo=SL 方策初期化→RL が SL に勝率80%(080 の先行構造)。
