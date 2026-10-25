# 最適化・将来課題 バックログ(正本)

> **📍 大局と「開いている道」の総括は `docs/PROJECT_MAP_057-085.md` §5**(Task 086)。第2章(Transformer+規模)/ 電竜戦準備 / 温存項目の**判断材料**はそこに一本化。本ファイルは個別課題の詳細台帳。

このファイルは binbo-shogi の **「実装方針は決まっているが後回しにしている最適化・将来課題」の
唯一の正本**です。何を次にやるか迷ったら、まずここを見て、各課題の**判断基準(いつやるか)**に
照らして選びます。「ドキュメントが真実の源、忘れることは前提に外部化する」というプロジェクトの
原則の実践です。

## 使い方と他ドキュメントとの役割分担(二重管理しない)

- **このファイル = 後回しの最適化・将来課題の一覧(正本)**。各項目は `何を / なぜ / いつ(判断基準) / 優先度感`。
- `STATUS.md` = **進捗と現在地**(完了したこと・現在のフェーズ・直近の次の一手)。網羅的な将来課題は持たず、ここへポインタ。
- `DESIGN.md`「未決の問い」 = **設計上の未決定事項(どう作るか未定)**。性質が違う(やること自体が未確定)。本バックログとは別管理。
- `docs/kp_spec.md` = KP 仕様の正本。正準 HalfKP 化など個別課題は §7 に詳細。本バックログからはポインタ参照(再記述しない)。
- `TODO.md` = 本ファイルと STATUS への入口ポインタ。

新しい「後でやる」が出たらここに追記。着手・完了したら、その行を消すか STATUS の完了履歴へ移す(生きた文書)。

## 進め方の原則(なんとなく最適化しない)

- **律速を見極めてから手を打つ**。各項目の「いつ」は、原則「何が律速になったらやるか」で書く。
- 効果は `nnue-bench`(同一局面・同一深さでノード数/時間)と自己対戦(`nnue-*match`)で**数字で**測る。各改善を独立に。
- 既存の動くもの(perft・テスト・selfplay・既存エンジン)を壊さない。

参考(現状の律速の実測):Rust 側の1局面コストは **SFEN解析+KP変換が支配的**、**NNUE forward はほぼ無料**(Task 020)。
学習は **1エポック約24秒(CPU)で GPU 不要**(Task 017)。→ この前提が判断基準のベースになる。

---

## ★ 方向転換後の次段(やねうら王土台, Task 056〜)= 現在の本線

> Task 056 で方向転換(自前探索→やねうら王土台、自作 NNUE 評価で勝負)。地固め(ビルド/動作確認/eval 差し替え
> 経路の理解/ライセンス帳簿)は完了。**以下が今後の本線**。下記 A–D(自前探索・自前評価の最適化課題)は
> **自前実装の歴史として保全**(Task 055 時点の `src/`)= 対局時の探索はやねうら王に委ねるため、原則として再着手しない
> (理解の記録)。設計 `docs/nnue_yaneura_export.md`、方向転換の理由 `DESIGN.md`「方向転換」/`STATUS.md`。

- **雑巾絞りの 2 周目以降: 強くなった eval でもう一度自己対局して絞る ★★★(Task 154 の後, 2026-10-21 追加)**
  - なぜ: Task 154 で, 採用 eval どうしの自己対局 15 万局 (depth 9, 詰みまで, 勝敗つき) で仕上げると +28.6 (外部の教師なし)。雑巾絞りは繰り返して強くするもの。
  - 何を(案): ① 新しい採用 eval `task154_R_result30` で同じ設定の自己対局をもう一度 ② 局数を増やす (30〜60 万局) ③ 勝敗の割合 (3 割) を振る ④ 局面の分布と勝敗のどちらが効いたかの切り分け (技巧2 の局面に勝敗だけ足す等)。
  - コスト: 15 万局で自己対局 2.7 時間 + 学習 6 分 + 判定 2.3 時間。
  - (完了: 3 本目の物差し つようら王11micro は Task 151 で採用前の確認用として導入済み。)

- **教師から生徒への橋渡し: 自玉の危険を生徒が受け取れていない ★(Task 133/134 後の候補、2026-10-18 追加。同日 Task 136: 静止局面では ±2000 で切った教師の目標に生徒はほぼ合う (+2.7cp/枚) が, 切らないと +178 (敵が多い局面の教師は ±2000 の外)。Task 133 の教師と binbo の差 +77/枚 は評価のやり方 (浅い MCTS か深い探索か) の違いとして残る。Task 139: 静止局面でも教師 −42 / binbo +11)**
  - なぜ: Task 133 で, 自玉の 2 マス以内の敵が多い局面を教師 (WS-T S=8) は手番側に −56cp/枚と辛く見るのに, 生徒 (採用 eval) は +20 と楽観。
    教師は危険を分かっているのに生徒に伝わっていない = ラベルではなく学習・表現の側の問題。Task 134 (該当局面の S=32 付け直し) は教師の側を直しただけで差なし (+12 [−13, +37])。
  - 何を(案): ① 学習データ内で該当局面の生徒の誤差 (教師 − 生徒) を特徴別に測り, 表現できていないのか (訓練誤差も大きい) / 汎化しないのか (val だけ大きい) を切り分ける。
    ② 表現の不足なら, KP 特徴で「玉の近くの敵の数」が表しにくい可能性 → 玉の周りの利きを表す小さな特徴の追加や損失の重み付けを検討。
  - 安い確かめ: Task 134 の生徒で Task 133 の回帰をやり直し, 係数がどれだけ動いたか。

- **qsearch 対照: 「一手先の戦術解決」だけで探索ラベルの利得が出るか ★★(Task 104 後の候補、2026-10-04 追加)**
  - なぜ: 採用 eval `task104_student_search` の利得(対 champion 直接 +134 / 技巧2@6 フレーム +56)は、WS-T の生の値ではなく
    「一手先を見て平均した値」から来ている(生ラベル学生は探索学生に −100、085 の生 value 学生は champion に −36。
    103b で S=8 でも S=32 と同等 = 深さに鈍感)。やねうら王公式の `qsearch_psv`(教師局面を qsearch の PV 末端 = 静止局面へ置換)や
    tanuki- の qsearch 効果と同じ現象と読める(`docs/web_scan_2026-09-11.md` §6 示唆 2)。もし qsearch だけで同じ利得が出るなら、
    深層学習網も MCTS も要らず、ラベル作りがずっと安くなる(= 64M 以上へのデータ拡大が楽になる)。出なければ「WS-T の方策で選んだ一手先」に固有の情報がある。
  - 何を(案、事前登録時に確定): 32M 局面 P それぞれで駒取り・王手回避の qsearch を回して PV 末端の静止局面 Q を得る →
    (b) **ラベルだけ静止探索**: 局面は P のまま、正解を「Q の WS-T 生 value」にした学生(MCTS を qsearch で置き換えた対照。
    生学生 → 探索学生の +100 が qsearch でも出るか)。
    (a') **局面の置き換え(やねうら王 `qsearch_psv` 型)**: 学習局面を Q に置き換え、Q とその評価値で学習した学生
    (NNUE に「取り合いの途中」の局面を学ばせない効果)。champion レシピで学習し、探索学生・champion と比べる。物差しは技巧2@7。
  - 注意(2026-10-04 訂正): 「静止局面を技巧2 で評価してラベルにする」案は意味が薄い。技巧2@d8 のラベルはもともと 8 手読み +
    末端の静止探索を済ませた値なので、ラベル側に qsearch を足しても変わらない。同じ理由で、champion の教師も戦術は解消済みだったのに
    探索学生が勝っている ⇒ 利得は「戦術の解消」だけでは説明しきれない(WS-T の値の滑らかさ・学びやすさ等も候補)。(b) と (a') はこの切り分けにも使う。
  - **マイクロベンチ(2026-10-04、`experiments/alphazero/bench_qsearch.py`、run32m_g から 2,000 局面 / WS-T 誘導は 400 局面、結果 `~/binbo-shogi-data/logs/bench_qsearch.json`)**:
    - **qsearch はほとんどの局面でラベルを動かさない**: 取り合いが無く Q = P の局面が 54%(駒得誘導)/ 68%(WS-T 誘導)。生の値から 50cp 超動くのは 40% / 27%、中央値は 0cp。
      対して **S=8 探索ラベルは 84% の局面で 50cp 超動く(中央値 182cp)**。静かな局面も含めて全体的に動いている。
    - qsearch ラベルは探索ラベルに近づかない: 探索ラベルとの差の中央値は 駒得誘導 202cp / WS-T 誘導 201cp で、生の値(182cp)より近くならない。
    - 2 方式の一致: 同じ Q に行くのは 58%。違うときの差は大きい(平均 160cp)= 駒得だけで止め所を決めると別の取り合いを選ぶ。
    - 速度: 駒得誘導は CPU 1 コアで 733 局面/s(32M を 1 コア 12h、20 コアなら 1h 未満)+ Q の WS-T 評価 1 回(生ラベルと同じ費用)。
      WS-T 誘導は 1 局面あたり固有の評価が平均 18 回 = 生ラベルの約 18 倍の GPU 評価が要り、batch 1 では非現実的(32M で 1,000h 超)。
      (このベンチの GPU 側の速度は前処理込みで本番より遅く出る。本番の S=8 rescore は ≈980 局面/s。)
    - **予測**: (b) の学生は生学生(探索学生に −100)とほぼ同じになる見込みが高い。⇒ 探索ラベルの利得の主因は「取り合いの解消」ではなく、
      静かな局面も含めて全局面で効いている「方策で重み付けした一手先の平均」の側にある、と読める。精度そのものはベンチでは測れない(099 の gold の教訓)。
  - 未決: qsearch の実装元(やねうら王の qsearch を流用 / cshogi で駒取り生成の自前 qsearch)、静止局面の評価に技巧2 を何深さで使うか、
    手番・符号の扱い(PV 末端の手番が元局面と変わる)、局面置換で手番偏りが出ないか。
  - いつ: 平均ラベル腕(技巧2@d8 と探索ラベルの平均、Task 104 決定表の予定腕)と並ぶ次の候補。機構の確認とコスト削減の両方に効く。
  - 優先度感: 中〜高。

- **Phase 2: 自前 NNUE v0(やねうら王形式 nn.bin の本実装)★★★ 次の本命**
  - なぜ: 方向転換の技術的核心。自作評価をやねうら王に載せて初めて「自作 eval で勝負」が成立する。
  - 何を: ①round-trip(配布 nn.bin を parse→同一バイト re-emit, sha 一致)で読み書きを保証 → ②自作 trainer を
    **halfkp_256x2-32-32 で学習**(fd8→256, 付番を BonaPiece 空間へ)→ 量子化(FT i16 / affine i8, `>>6`, FV_SCALE16)
    + 直列化で nn.bin 出力 → ③やねうら王に load → forward parity(評価値突合せ)→ ④旧版(やねうら王標準 or 自前)と 1000 局。
  - いつ: **今**(地固め完了後の第一歩)。`docs/nnue_yaneura_export.md` の §6 検証計画に沿う。
  - **進捗(Task 057)**: parity 手順を確立=GREEN(`docs/nnue_yaneura_export.md` §8)。付番 remap correspondence GREEN(1620局面・negative control 検出力あり)、
    int-faithful 参照 forward が実エンジン `e` と **bit-exact**(32/32)、round-trip byte一致。driver `scripts/yane_parity.sh`。
    **判明した制約**: binbo 現 net は plain ReLU=YaneuraOu の ClippedReLU+`>>6` と非一致 → 本命 forward は **trainer を量子化対応(clamp+整数スケール)**にする要あり(本命 eval の parity 再走は driver の eval 差し替えのみ)。
  - **進捗(Task 058)= 上記②の量子化対応が DONE**(`docs/nnue_yaneura_export.md` §9, `reports/task058_qat.md`)。binbo を量子化対応に:
    Rust `Mode::Quant`(BNNQ, 整数 forward)+ `QATSharedNNUE`(ClippedReLU+明示 ft_bias+fake-quant, `train_nnue.py --qat`)+ BNNQ export。
    **Part1 engine-grounded**: converted tanuki256 で binbo quant `nnue-eval` == `e` が **32/32 bit-exact(536cp→0cp)**。
    **Part2 QAT**: deploy==Python int 参照 bit-exact、量子化残差 **536→~25cp**(出力 i8 が下限)、品質 deploy≈理想連続(歪まない)。回帰 green。
  - **進捗(Task 059)= 上記③の writer + 実エンジン load が DONE**(`docs/nnue_yaneura_export.md` §10, `reports/task059_writer.md`)。
    **writer `trainer/binbo_to_yane.py`**(純 Python, `R` 逆 scatter + `yane_nnue.write` 再利用, Rust 無改変)で BNNQ→nn.bin。
    `R` の many-to-one(金/+P/+L/+N/+S を1スロットへ=52488 畳む)+ side 2 特徴を **tying**(`--qat-tie`)で吸収 → binbo-学習 eval も bit-exact 書出可能。
    **engine bit-exact**: tanuki round-trip + tied binbo eval とも **engine `e` == binbo Mode::Quant == int-faithful、32/32 0cp**(自前 eval が実エンジンに乗った)。
    negative control: untied は writer が拒否。tying コスト≈0(val_loss 160k≈164k)。→ **読込057+forward058+書出059 が全て engine bit-exact=経路完成**。
    **残り = 本命移行**: 256 で **新規 self-eval 量産**(自前教師→tied QAT→writer→engine)→ ④旧版と 1000局。
  - **進捗(Task 060)= bridge を本走で使用(探索器比較対局)DONE**(`docs/nnue_yaneura_export.md` §11, `reports/task060_searchmatch.md`)。
    本命 256(full 8M, tied QAT, val_corr 0.833)eval を**両側 bit-exact 同一**に固定し、自前 Rust 探索 vs YaneuraOu 探索 を対戦(変数=探索器のみ)。
    Rust: `Mode::Quant` cp 直返しの mode-aware leaf eval(`Searcher::leaf_eval`)+ ハーネス `nnue-vs-yane`(USI driver `src/usi_engine.rs`・trinomial SPRT `src/sprt.rs`)+ driver `scripts/selfeval_vs_yane.sh`(export→writer→0cp ゲート→対局)。
    **結果**: 対局前ゲート 32/32 0cp、SPRT(H1=binbo ≥50 elo 弱い)が **19 局で accept_h1**、binbo **0-19-0** → YaneuraOu の探索が同一 eval で ≥50 elo 強い(p<0.05)。
    速度差: 同一 movetime で yane depth 15.9/120.6k node vs binbo 3.8/34.6k node(約 4× 深さ)=**実装速度込みの総合探索力**。回帰 24 green、057-059 bridge 不触。
  - **進捗(Task 061)= eval 強化(踏み台を高くする)DONE**(`docs/nnue_yaneura_export.md` §12, `reports/task061_eval_tuning.md`)。
    教師据え置きで**学習側だけ**調整。検証ループ: 一次=val、確証=**eval-vs-eval SPRT**(`yane-vs-yane` 新ハーネス, 探索器固定で eval だけ差替=060 の逆軸, `eval_vs_eval.sh`)。
    **lever は lr の水準**: 正則化/cosine schedule は無効、**lr を下げると val 単調改善**(corr 0.833→0.858)=**Task060 本命は lr 高すぎで過学習**だった。
    確証: 低 lr 3 候補が baseline に **+188〜+241 elo**(同一探索, draw ほぼ 0)。**採用 `task061_lr1e-5_e18`**(val_corr 0.858)= 新 baseline 世代。回帰 24 green。
  - **進捗(Task 062)= 教師データ増量 8M→16M DONE**(`reports/task062_data16m.md`)。一変数=データ量(lr/net/teacher質 据え置き)。
    `gen-teacher`(技巧2 depth8, 新seed)で +8M を ~70分(24w, 1.38ms/pos)→ 16M cache(val は shared_8m と同一=公平)→ 061 champion 設定で学習。
    **val_corr 0.8577→0.8751**、train-val gap 49480→26884 に半減(**data律速を実証**: train↑ val↓)。確証 SPRT: 16M vs champion **+125 elo**(67.2%, CI +76–179)。
    **採用 `task062_16m_lr1e-5`**= 新 baseline。**頭打ちでない**(データまだ productive)が val_corr 0.875 が技巧2天井(~0.90)に接近。**Rust 無改変**。
  - **進捗(Task 064)= 教師データ増量 16M→32M DONE**(`reports/task064_data32m.md`)。一変数=データ量。無人放置(`detached.sh`, WSL RAM 15→22GB)。
    +16M(技巧2 depth8, 新seed 20290629, 16w, ~2h49m)→ 32M cache(val 同一=公平)→ 062 champion 設定で 12 epoch 学習(best ep6)。
    **val_corr 0.8751→0.8862**(+0.0111=**逓減**: 8→16 は +0.0174)、**gap は plateau**(26884→25655=教師ノイズ底 ~26k が不動)。確証 SPRT: 32M vs 16M champion **+141 elo**(69.2%, CI +91.5–196.7)。
    **採用 `task064_32m_lr1e-5`**= 新 baseline。**核心=逓減の食い違い**: val は逓減(天井接近・gap 底不動=蒸留の情報量が枯れつつある)が **棋力は逓減せず**(+125→+141, CI 重複)。**Rust 無改変**。
  - **進捗(Task 065)= 自己強化ループのラベル生成コスト計測 DONE**(`reports/task065_label_cost.md`)。計測のみ(champion を読むだけ, Rust/eval 不触)。
    新規 `trainer/label_cost.py`(verify_mirror の sampler+USI driver 流用, watchdog で固定深さ探索を hard-bound)で champion を YaneuraOu に載せ depth 8-16 × 500 局面を cold 計測。
    **per-局面 cold time**(median): d8 3.9 / d10 8.4 / d12 25.4 / d14 71.5 / d16 211.7 ms(+2depth で ~2.5-3×, 重い右テール)。
    **cp は depth16 まで飽和せず**(median&#124;Δcp(d→d+2)&#124; 73→48, 閾値 12 未到達=明確な knee なし)、**符号は depth10 以降安定**(反転 ≤5%)=深い探索は静的 eval を書き換え続ける=**蒸留の燃料あり**。
    **1サイクル外挿(16w)**: 1M を d12 0.44h / d14 1.24h / d16 3.67h。**提案(確定は066)= ラベル深さ第一候補 depth 12-14 / サイクル 1-5M**。
  - 優先度感: 最高。
- **自己強化ループ 設計確定 + 1サイクル・リハーサル(Task 066)= DONE**(`reports/task066_loop_rehearsal.md` / 設計 `docs/self_reinforcement_design.md`)。
  - 設計確定: ラベル深さ12開始・cp勝率変換+WDL ハイブリッド λ=0.5・多様性は軽い random_plies 開始・採用=gen-vs-gen SPRT(yane-vs-yane 0cp)・主アンカー=技巧2(新規 `yane-vs-gikou` DepthLimit6)・副=tanuki・warm-start+32M replay 混合・戻れる設計・tie 累積監視。
  - **リハーサル結果(機構検証 GREEN, 棋力は後退)**: 機構が end-to-end で回った(selfplay→mix→warm-train `--init-weights`→0cp ゲート 32/32→3種対戦, `cargo test` 24 green)。
    gen1 vs champion=**21.2%/−228 elo/accept_h0(棄却・champion 不動)**、技巧2 anchor で **83.3%→60.0%(−210 elo)** 明確低下。**自家中毒(内外乖離)でなく一様後退**=gate もアンカーも整合して正しく棄却(検出力 sanity 確認)。
    方法論: **gikou@6=分解能あり / tanuki=飽和(両者 floor)**。val は採用基準にならないと実データ確認(val_corr 0.847→0.841)。
- **自己強化ループ replay 比率の探索(Task 067)= DONE**(`reports/task067_replay_ratio.md`)。一変数=replay 比率。新規=`selfplay_label.py --reuse-new`(066 の新規自己対局を bit-完全再利用し replay 量だけ変える純 numpy パス)。
  - **結果(単調回復)**: replay-n=32M比 / 技巧2 / gen-vs-gen vs champion = gen1 0.3%: 60.0% / 21.2% −228(崩壊)‖ gen3 6.3%: 70.0% / 42.5% −52 ‖ **gen4 15.6%: 83.3%(=champion)/ 46.0% −28=パリティ**。
    → **066 後退の主因=replay 不足と確定**(0.3%→15.6% で −228→−28 elo に回復)。内外(gen-vs-gen と技巧2)が終始一致=自家中毒なし。
  - **ただし champion を超える世代は未達**(gen4 はパリティ止まり=昇格なし・champion 不動)。replay はパリティ天井で頭打ち=前進には**新規信号側**が要。
- **自己強化ループ 新規自己対局データ量の増量(Task 068)= DONE**(`reports/task068_newdata_volume.md`)。一変数=新規量(100k→1M, 10×, replay 5M 固定=gen4 と A/B)。新規=`selfplay_label.py --workers`(multiprocessing 並列自己対局, ~9× 速)。
  - **結果(仮説と逆=量を増やすと弱化)**: @5M replay 固定で gen4 100k(1:50): 46.0% −28 パリティ / 技巧2 83.3% ‖ **gen5 1M(1:5): 32.8% −125(65-134-1)/ 技巧2 72.5%**。**固定 replay 下で新規量 vs elo は単調減少**。
  - **切り分け確定**: ボトルネックは新規信号の**量でなく質**。新規 champion@d12 自己対局ラベル(noisy WDL 0.47/0.06/0.47・champion 同士=盲点強化)は技巧2@d8 replay より低品質=希釈中は無害、相当量で能動的に悪化。**067 パリティ天井を後付け説明**(replay 増=有害な新規信号をゼロ希釈)。内外一致=自家中毒なし。
- **自己強化ループ 方策多様性(温度サンプリング)(Task 069)= DONE**(`reports/task069_policy_diversity.md`)。一変数=対局本体の手の選び方(決定的→温度)。新規=`selfplay_label.py` MultiPV + 勝率空間温度サンプリング(ラベルは multipv-1=champion 決定的評価のまま)。
  - **結果(仮説と逆=悪化)**: 100k/5M 固定で gen4 決定的 46.0% −28 パリティ / 技巧2 83.3% ‖ **gen6 温度0.10: 38.0% −85(75-123-2)/ 技巧2 66.7%**。内外一致=中毒でなく純粋な弱化。val ほぼ不変(0.01701≈gen4)なのに悪化=val 不採用再確認。
  - **統一見解(067/068/069)**: 温度は off-distribution 局面へ誘導するがそのラベルは champion 自身の最も信頼できない eval=ノイズ増(盲点は盲点持ちのラベルでは直せない)。**champion の d12 知識を再サンプリングするだけの操作(replay/量/多様性)は超えられない**。超えるには新情報を足す=深い探索。
- **自己強化ループ ラベル深さ d12→d14 + gold 診断(Task 070)= DONE**(`reports/task070_depth_d14.md`)。一変数=ラベル深さ(gen4 と clean A/B)。新規=`gold_analyze.py`(教師品質を学習と独立に測る診断)。
  - **gold(教師品質は上がる)**: d16 を gold に median &#124;cp−d16&#124; が d12 80→d14 47、d14 が d12 より d16 に近い局面 67.7%、符号反転 6.0→3.8%。065 の「飽和せず」を局面単位で裏付け=067-069 で初めてレバーが教師品質を改善。
  - **loop(だが悪化)**: 100k/5M 固定・決定的・深さのみ変更で gen4 d12 46.0% −28 / 技巧2 83.3% ‖ **gen7 d14: 37.0% −92.5(74-126-0)/ 技巧2 76.7%**。**better labels → worse model**(内外一致の後退)。
  - **最重要(統一機構)**: 質と結果がデカップル=「情報量不足」説棄却。系は技巧2@d8 にアンカー(warm-start + replay 5M = 98%)、新規 champion 自己対局ラベルは d8 と食い違い深いほど矛盾大=ノイズ。**068(量↑)と 070(深さ↑)を同一機構で説明**。真の律速=**新ラベル↔d8 replay の不一致**(新規信号 family は gen4 のパリティ天井でサチ)。**d16 full は走らせない**。
- **自己強化ループ replay を技巧2@d8→champion 自己ラベルへ置換(自己アンカー化)(Task 071)= DONE**(`reports/task071_self_anchor.md`)。一変数=replay/ラベル出所(技巧2@d8→champion@d12 自己ラベル、深さは gen4 同一 d12 に固定)。新規=`selfplay_label.py --val-frac`(純 self-play の val hold-out)+ chunked/resumable 大規模生成 driver(`t071_gen.sh`)。
  - **結果(d8 を完全除去=自己アンカー)**: gen8 d12 self-play 5M(d8 0%)= gen-vs-gen **25.5% −186.2(51-149-0)** / 技巧2 **71.7%(43-17-0)**。**3 世代で最悪**(gen4 46%/83.3% ‖ gen7 37%/76.7% ‖ gen8 25.5%/71.7%)。**内外そろって後退=自家中毒でなく純粋な弱化/忘却**。0cp 32/32, gen8 sha `aae0847c…`。生成 ~5h51m(10 shard×505k, seeds 20260801..810)。
  - **切り分け(diagnostic)**: self-play は **100% ユニーク局面**(replay 97.5% より高い)=「狭い/冗長な分布」説を棄却。弱化は被覆でなく**ラベルの知識質**=**champion 自己 eval(d12)は技巧2@d8 ラベルより弱い教師信号**。技巧2-蒸留 champion を自己 eval で学習し直すと、より弱い自己整合的不動点へ退行(忘却)。深さ d12(>d8)は教師の局面知識を代替しない。
  - **本丸の結論**: **champion 自己対局ラベルは(どの深さでも)技巧2-蒸留 champion を超えず、維持すらできない**。067-070(再サンプリング系)に続き **アンカー離脱(071)も否定**=このアーキの自己強化ループは技巧2@d8 の天井を超えない。champion 不動(reversibility)。
- **自己強化ループ 教師信号の強化(より強い教師アンカー)(Task 072)= DONE/行き止まり**(`reports/task072pre_gikou_d12.md`)。067-071 で「自己対局ラベルをいじる」族+「アンカー離脱」が全滅→唯一残る道=technique2 より強い教師。GPL 内 in-family 候補=技巧2 を d8 より深く(d12)。
  - **① より深い技巧2 = 棄却(行き止まり)**: (a) gikou は depth≥~8 で **warm-state EOF crash**(局面非依存、cwd 未設定だと zero-eval=score cp 0 の罠も判明)。(b) **gen-teacher@d12 fresh batch は decisive 局面を crash→abandon** で喪失(≥5000cp 5.37%→0.03%)=d8 より弱いラベル。(c) **無偏 cold 再ラベル**(`trainer/gikou_relabel.py` 実装)は decisive 局面が cold でも crash(14%/hard-fail 9%)+ **~46h/1M** + (d) **技巧2 は d8 で実質飽和**(d12≈d8, median 71cp)。→ より強い in-family 教師は存在しない。
  - **採用 = ③ 技巧2@d8 を本アーキの天井と受容**(ユーザ決定)。**自己強化アーク 066-072 の結論=この NNUE 蒸留+自己対局アーキでは技巧2@d8 蒸留 champion(`task064_32m_lr1e-5`)を超えられない**。champion 据え置き。
  - 副産物(committed): `trainer/gikou_relabel.py`(cache-aligned cold gikou re-score, **cwd=gikou dir 必須**, watchdog, d8 で高速健全=将来の同一教師 re-score 基盤)。
- **(c) AlphaZero/MCTS 転換の徹底調査(Task 073)= DONE**(`reports/task073_alphazero_survey.md`)。実装なし・web 一次情報で単機(1×4060Ti)での (c) 可行性を調査。
  - **要点**: ゼロ知識将棋 AZ は単機 ~100年(AobaZero)=非現実。dlshogi は warm-start(supervised→RL)でも自宅 PC 2年+。効率化=KataGo 50×/Pgx(将棋 env+AZ 同梱, GPU 10-100×)/Gumbel 少 sim/最小 OSS(MiniZero 等)。dlshogi/Apery コード=GPL(借用可)、封印は学習済みモデルのみ。champion(HalfKP)は AZ の CNN と別アーキで直接初期化不可。
  - **構成候補**: A フルスクラッチ Rust MCTS / B 既存 GPU 枠組み借用(Pgx/MiniZero)/ C ハイブリッド warm-start。→ **B+C** が小資源で有望、到達目標=「dlshogi 超え」でなく「単機 self-play RL が正の elo」実証。
  - **第1マイルストーン**: M0 極小ゲーム器疎通→M1 将棋スループット実測→M2 warm-start 単体で MCTS が αβ 超え(SPRT)→M3 short RL で gen1>warm-start。未決 8 点は report 末尾。
- **(c) AlphaZero フレームワーク選定(Task 074)= DONE**(`reports/task074_framework_selection.md`)。Pgx/MiniZero/kZero を環境構築+最小サンプル起動まで実測比較。
  - **Pgx(+mctx)推奨**: 唯一「本環境で今すぐ GPU 認識+どうぶつ将棋で Gumbel AZ 内ループ実走」達成。Apache-2.0、animal_shogi+将棋 標準搭載、baseline model で評価即。MiniZero=Docker 不可 here+shogi 無+LICENSE 不明、kZero=CUDA build 不可+shogi 無+無ライセンス。
  - 主コスト=JAX 習得 / 学習ネット→USI glue(M2 以降)/ まずどうぶつ将棋で M0。作業物 `~/task074_frameworks/`(非コミット)。
- **(c) M0 = Pgx でどうぶつ将棋 AZ ループ(Task 075)= DONE/PASS**(`reports/task075_m0_dobutsu.md`)。単機小設定で animal_shogi AZ を 60 iter(~10分/4060 Ti)。
  - **win_rate 0.008→~0.5-0.65**(vs 学習済 animal_shogi_v0)=ランダム初期→baseline パリティ超。**単機 self-play RL が正方向=(c) の前提成立**(αβ アーク 067-072 の天井と対比)。留保=題材小/eval は raw-policy 代理。
  - JAX 3クセ(関数型+lax.scan / PRNGKey 分割 / 静的形状)を report にコード引用で解説。詰まり=`device_put_replicated` 削除→1行 patch(隔離 train_m0.py)。作業物 `~/task074_frameworks/`(非コミット)。
- **(c) M1 = フル将棋スループット実測+warm-start 見極め(Task 076)= DONE/PASS**(`reports/task076_m1_throughput_warmstart.md`)。10+1 点格子実測(隔離 train_m1.py)。
  - **単機動作点**: 1 iter ≈ 4.8 分・VRAM 12.6GiB・466 states/s・value 有効率 74% @ sims32/64ch×5L/batch256/steps512 = ~11 万局/日。**sims が超線形に高価**(Gumbel 少 sim が生命線)、**net は割安**(dlshogi 級でも 4.4×、律速は env.step)、窓は 512 が正解(128 は有効率 5.7%)。
  - **a/b/c**: a ゼロ知識=非現実を実測外挿で確定(dlshogi 級で AZ 24M 局 ≈ 5 年〜)/ **b champion 蒸留 warm-start=推奨**(SFEN 橋完全動作 600 局面 roundtrip、蒸留は初期化のみ=天井機構が異なる)/ c 中間題材=不採用(Pgx にミニ将棋無し)。
- **(c) M2 = 価値のみ蒸留 warm-start → USI 橋 → 立ち位置測定(Task 077)= DONE/PASS**(`reports/task077_m2_warmstart_usi.md`)。
  - **蒸留成立**: 32M hcpe 技巧2@d8 ラベル(再ラベルなし)で AZ 64ch×5L の value のみ蒸留 → val_corr **0.005→0.900**(教師天井 ~0.90、2M/4ep/~5分)。方策は乱数のまま(一変数)。
  - **橋**: USI↔Pgx action 写像を cshogi と 2,000 局面完全一致+negative control RED。**322 局 非合法手 0**。sims=128 ≈ 101ms/move(champion 100ms と等時間)。
  - **立ち位置**: vs champion **0-0-200**(elo ≤ −730)/ vs gikou@6 **0-1-99**(≈ −920)。value 知識だけでは乱数方策×考慮手16 の探索を補えない=M3 が持ち上げる高さを定量化。
- **(c) M3 = 方策改善ループで warm-start 超え(Task 078)= DONE/FAIL(本予算)**(`reports/task078_m3_loop.md`)。
  - 一晩 150 iter(19.66M frames, lr1e-4=ladder 実測選定)。**主判定 SPRT accept_h0**: gen150 = **−110 elo(CI −151..−72)vs warm-start**。カーブ= V 字(−108/−168/−269/−110)=谷から +160 elo 反発するも出発点未満。
  - **切り分け**: ループは学習している(066-072 型でない)。主因= fresh adam×共通 lr で**蒸留 value の質を手放す**(lr1e-3 は 1 iter で全壊=val_corr_m2 監視が検出)。方策の成長は方向正・量不足。副次: champion/gikou とも床(0 勝)。
- **(c) M3延長 = ckpt150→300(Task 079)= DONE(互角まで回復・有意超え未達)**(`reports/task079_m3_extension.md`)。
  - gen300 vs warm-start = **−11 elo(CI −44..+21, 400局)=統計的に互角**。gen150 −110 から **+100 elo(有意)**。回復は iter~215 から単調ランプ、250 以降パリティ帯。**corr は 0.25 底→0.41 に自己回復**(=案2 降格)。
  - 判定=(a)(b) 中間: 「出発点未満で頭打ち」棄却、「パリティ=吸引点」(教師相当 value の再導出止まり)仮説が残存。ckpt300→400 続行中(080 先行データ)。
- **(c) M4 = 方策 warm-start → ループで初期化点超え(Task 080)= DONE/PASS**(`reports/task080_m4_policy_warmstart.md`)。
  - WS2(価値 corr 0.873+top-1 34.1%、蒸留 ~5分)vs WS1 = **89-0-0(≥+586 elo)**=方策温め単体で 079 の吸引点を突破。**主判定: gen150 vs WS2 = SPRT accept_h1、+133 elo(CI +92..+177)= (c) 目標『単機自己対局 RL が正 elo』達成**。
  - カーブ谷なし単調上昇(+17/+53/+127/+133)、価値 corr 0.72-0.79 安定・top-1 保持=方策があると自己対局の質が学習信号の質を支える。副次: vs champion **0-4-196(初の非ゼロ)**。
- **(c) M5 = ループ延長 gen150→gen200(Task 081)= DONE/頭打ち**(`reports/task081_m5_extension.md`)。
  - gen200 vs WS2 = **+109 elo(CI +72..+149, SPRT accept_h1 @328局)= gen150 の +133 と統計的に同一**=延長で正味の上乗せ無し。カーブは +110〜+255 のバンドで振動(中間 n=40 は広 CI)。価値 corr 0.74-0.78・top-1 0.283 安定=**崩壊でなく飽和**。
  - 副次: gen200 vs champion 0-2-198(≈ gen150 の 0-4-196=床不変)/ gikou@6 0-0-100。内部 +130 は外部到達に転写されず。**「延長=時間」の安い一変数は使い切り**。
- **(c) M6 = ネット容量 64ch→128ch(Task 082)= DONE/部分 PASS**(`reports/task082_m6_net128.md`)。
  - (A) 容量↑ → warm-start 質↑: WS2-128 corr 0.905/top1 0.400(旧 0.873/0.341)、WS2-128 vs WS2-64 = +235 elo。(B) loop 増分は容量不変: gen100(128) vs WS2-128 = +126 ≈ 64ch +130。(C) 外部両アンカー限界改善: champion −798→−727(WS2-128 が初の 2 勝)、gikou 0-0-100→0-6-94(初床割れ)。だが 0 勝床。
  - ★内部/外部非整合: gen100(128) 内部 +360 over gen-64 が外部 +70。loop +126 は外部非転写(gen100 ≈ WS2-128 vs champion)。★外部床は探索予算(sims128 ~128node vs αβ ~1M node)の壁の疑い。詰まり= 蒸留 batch 4096→1024(128ch OOM)、init NaN は BN 推論初期化アーティファクト(無害)。
- **(c) M7-a = play-time sims↑ で外部床の正体切り分け(Task 083)= DONE/(b)net 品質の床**(`reports/task083_m7a_sims_floor.md`)。
  - gen100 vs champion = @128 0.015/@512 0.010/@800 0.050/@2048 0.008 = **フラットなノイズ帯**(5-53× wall-clock で床不変)→ **外部床は探索予算でなく net 品質/レジーム**。gikou も改善なし。
  - ★決定的対比: gen100 vs WS2-128 = +126(@128)→ **+241(@512)**=**探索は内部を増幅するが外部に非転写**。=「探索は効くが AZ net が αβ と別レジーム、内部優越が外部に転写せず」。080 の +133 は本物(net 実優越)だが champion 超えとは別問題。
- **AZ value 教師上書き(Task 084)= DONE/決定的 PASS**(`reports/task084_az_value_teacher.md`)。教師の cp 出所だけ技巧2@d8→AZ value に替え 2M で 3系統学習→SPRT。
  - **WS2-128 value 教師は技巧2@d8 探索cp より +237 elo 良い**(accept_h1, 79.7%)/ gen100 は +47(有意)/ WS2-128 > gen100(−145)。=**技巧2@d8 の noisy 探索cp を AZ で denoise すると NNUE が大幅に強くなる**(nodchip 一致、055 と逆=同信号の平滑化ゆえ分布一致)。**066-072 の教師天井は替えれば動くと実証**。
  - ★逆説: WS2-128(ループ前)が最良教師、gen100(ループ後)は圧縮(std 698)で悪化=内部棋力↑≠教師品質↑。alignment assert 全通過、az_rescore ~11k/s。
- **32M 平滑化教師 vs champion(Task 085)= DONE/RED**(`reports/task085_32m_smoothed_champion.md`)。084 の 2M +237 を 32M 全量へスケール(一変数=train 教師ラベル、cache/net/設定/mirror は task064 同一)。
  - **t085 vs champion = −36 elo(CI −70..−1.9, 有意に負け)**。**2M +237 → 32M −36 の反転**=平滑化は小データ正則化(大データは生徒が自己平均化しノイズ消失、AZ 平滑ラベルの不可逆圧縮損失が顕在化)。**技巧2@d8 champion 天井は堅持、AZ value 転用のバイパス路はスケールで閉じた**。rescore std 1462/corr 0.941(2M と同特性)、alignment 全通過、gikou 81%≈champion。
- **平滑化ルート打ち止め**: denoise は新情報でないゆえスケールで原理的に飽和。教師を真に良くするには**新情報**=AZ 自体を技巧2 非依存で強くする(第2章)しかない。**代替(優先度中)**: 小〜中データ域では +237 の正則化が効く=限定的な新規教師を AZ 平滑化で薄めるハイブリッド(champion は 32M で既に飽和側ゆえ効果は限定的見込み)。
- **(c)/教師軸の総括**: champion(生技巧2 32M)は (c) 内部優越の外部非転写(M7-a)でも、AZ value 教師転用(085)でも超えられない=**技巧2@d8 champion は現アーキの確固たる天井**。突破は第2章(Transformer+規模で AZ 棋力自体↑→良い教師 or 直接対局)へ。(c) 目標「単機自己対局 RL が正 elo(内部)」は M4/080 で達成済。
- **【第2章 = T1 GO / T2-loop FAIL】Transformer アーキ(ChessFormer 系)**: **Task 087(T1)で warm-start 段は GO**(`reports/task087_t1_transformer_warmstart.md`)。等パラメータで CNN を全軸で超え(arch-only val +0.020corr/+0.056top1、対局 34-4-2 ~90%、外部 champion +140/gikou +300)、推論も速い。**T2-scale(Task 089)+recipe(Task 090)**では7.6Mの内部悪化がundertrainingと判明した一方、適正epochでも外部へ非転写。標準ネットは3.6M(d240/L5)据え置き。**Task 091(T2-loop, `reports/task091_t2loop_frozen.md`)**は完全二塔でvalueをbyte-for-byte保護し方策だけ75 iter更新したが、最良世代の教師品質は400局 **198-198-4、0 elo(CI -34..+34)**でWS-Tを超えずFAIL。内部も `-52/+61/-26 elo` と安定上昇なし。**凍結は成功したが、完全二塔方策ループは教師改善レバーにならなかった**。フル32M還流/champion再挑戦は行わない。Phase2(lr ladder)は外部非転写ゆえ低決定価値(~7h)の任意follow-up。
- **NNUE方策橋の較正(Task 092-0)= DONE/FAIL(bridge null)** (`reports/task092_0_nnue_bridge_calibration.md`)。
  - 固定2M+同一WS-T valueで value-only / linear32 / MLP32 / FT-MLP512を比較。現行linearのvalue正則化はMAE/corr改善・std微悪化の混合。policy top-1は2.15→3.80→5.26%で、FT幹にはh32より多い指し手情報がある。
  - 陽性対照=最良FT-MLPの方策ラベルだけ **WS-T合法argmax vs 合法一様random**。補助headはtop-1 5.26% vs 1.65%と分離したが、捨てた後のevaluatorは400局 **193-204-3、−9.6 elo(CI −43.7..+24.4)**。既知の最大級の方策差も現行橋で検出不能。
  - 含意: 091の教師0 eloはループ方策の教師価値nullと装置nullを分離できない。事前ゲートどおりhard/soft比較は未実施。softは指し手の確率化でなく教師ラベル分布だが、現endpointを直す前に比較しても判定不能。
- **T2-loop外部アンカー(Task 093)= DONE/NEGATIVE** (`reports/task093_t2loop_external_anchor.md`)。
  - WS-T/iter25/50/75をgikou@6へ同一100 openingペア×先後反転=各200局。score **13.50→9.25→7.75→6.25%**、絶対Elo **−323→−397→−430→−470**。非合法/engine failure 0。
  - 主CI非重複はiter75で境界的negative。paired ΔEloは `−74/−108/−148`、全4点傾き **−47.7 Elo/25iter(CI −85.5..−14.1)**。=完全二塔policy-onlyループは外部を動かさないのでなく**単調悪化**。
  - iter50は内部 +61だが外部 −108=内部選抜と外部が逆向き。valueはdigest固定なので、この構成ではpolicy更新が外部悪化を生んだ範囲まで切り分けた。技巧2top-1低下とは同方向。
- **方策アンカー付きT2-loop(Task 094)= DONE/FORGETTING_STOPPED_NO_POSITIVE_TRANSFER** (`reports/task094_policy_anchor.md`)。
  - 一変数=アンカー率(091完全二塔・value凍結・sims32・512updates/iter固定、512中の固定割合をWS-T supervised best-move replayへ置換)。`r=0.25/0.50`の2arm、iter25×gikou@6 200局(093と同一opening/seed)。
  - paired ΔElo vs WS-T = r0.25 **+21.3** [−72.8,+118.6] / r0.50 **+79.1** [−11.1,+174.2]。**093 anchor無し(iter25 −74)の外部悪化を反転・停止**したが、両arm CIが0を含む=**正arm不在**ゆえ50iter延長せず。
  - 機構: アンカーが技巧2手top-1を0.4513→**~0.478-0.480**へ繋留(argmax変更 36%→~21%)。効果は「WS-Tへ引き戻す」で飽和、MCTS訪問分布側は外部強度に非寄与。=091(教師0)/093(負傾斜)に続き**完全二塔policy-onlyループはアンカーを足しても外部レバーでない**を3タスク確定。value digest全iter一致でGREEN、非合法0。
- **r=1.0 supervised-only 対照(Task 095)= DONE/NO_ARM_ABOVE_WST_SELFPLAY_UNRESOLVED** (`reports/task095_selfplay_control.md`)。
  - 一変数=anchor率 r を 1.00 へ(512update 全部 supervised=自己対局データを勾配に触れさせない)。他は094 byte-identical、判定器も093/094と同一。**事前登録済み(ブロックseed・判定表・許容誤差)= optional stopping を封じた**。
  - **★用量反応は r=0.50 に内点最適**: ΔElo vs WS-T = **−74.0 / +21.3 / +79.1 / −7.6**(r=0/0.25/0.50/1.0)。r=1.0 は supervised update が2倍なのに 87 Elo 低い ⇒ **「094の+79は追加supervisedの回収」説を棄却**。純visit分布でも純supervisedでもなく混合だけが上向く。
  - **主判定(選択汚染ゼロ block2+3, 200ペア)= Δ_selfplay +37.8 [−27.0,+103.5] = NULL**。block別 +86.7/+80.9/−3.4(block2は+79を独立再現、block3は再現せず)。pooled は +54.8 [+4.1,+106.8] だが block1 汚染ゆえ主判定に採らない。決着に 2,430局/系列(~11h)。
  - **機構**: r=1.0 は学習側 top-1↑ / 未学習 val_top1↓ / val_ce↑ = **同じ2Mを回す段は枯れて暗記へ**(085と整合)。内部 top-1 は r=0.50 とほぼ同値なのに外部 87 Elo 差=内外乖離の再現。
  - **計測系の宿題**: 等価性判定を「ノイズ床に対する比」に直す(現行は厳密不等号=偽陽性50%、今回フォールバックで生成ON本走 +3.7h)。`BLOCK_HETEROGENEITY` フラグ未実装(block2 +80.9 vs block3 −3.4 を機械検出できず)。回帰ゲートは再試行方式へ修正済。
- **決着測定 r=0.50 vs WS-T(Task 096)= DONE/POSITIVE_TRANSFER_CONFIRMED(限界的)** (`reports/task096_transfer_decision.md`)。
  - 事前登録済み を実行前に置き optional stopping を封印。主推定=汚染 block1 除外・1,224 clean ペア(2,448局)・ABBA 順の10ブロック。**+28.65 [+3.79, +53.51]** ⇒ CI 下限>0 で**アーク初の外部正転写**。同一セッション部分集合(1,024ペア)は +36.93 [+10.21,+64.17]。
  - **留保4点**: 限界的CI(下限+3.79)/ **層別 bootstrap は NULL**([−6.25,+61.94]、ただし Q=8.98 df=9 **I²=0.0** で異質性の証拠なし=k=10 クラスタ再標本化の保守性)/ 検出力61%の限界的有意ゆえ点推定は上方偏り(80%検出力には1,881ペア要)/ **n=1 学習走**ゆえ主張は「このネット」限定でレシピ一般に及ばない。
  - **選択汚染の実測**: block1 単独 +79.12 → clean +28.65 = 上方バイアス約 +50 Elo。095 で block1 を主判定から外した設計判断がこの規模で正当化された。
  - **5点用量反応(記述専用)**: −74.0/+21.3/+79.1/**−15.4**/−7.6。`NON_MONOTONE`。r=0.75 の2単ブロックが大きく食い違い(block1 で r=1.0 同等 / block2 で r=1.0 比 +112.1)= **±93/点では r>0.50 側の形は決定不能**。r=0.75 も過学習し「同じ2M は枯れる」機構を再現。内部 2M top-1 は3アーム同値なのに外部は +28.65 対 ≈0=内外乖離の再現。
  - `SESSION_RATE_SKEW` 全8ブロック発火は**効果そのもの**(resign=gikou投了=AZ勝ち、勝ちが多い⇒対局が長い⇒局/分低下。ABBA の先攻/後攻どちらでも同符号)。15%閾値が厳しすぎた=次タスクで閾値を勝率調整済みの指標に。
  - **終端**(事前登録どおり延長せず)。次は「利得の champion への運搬路」= value 凍結緩和(λラダー)+084方式 value 相対判定、085 の 2M→32M 反転ゆえ 2M 勝ちは 32M 確認資格まで、n=1 解消を先に予算化。
- **シード再現走(Task 097)= DONE/REPLICATED** (`reports/task097_seed_replication.md`)。
  - 一変数=学習系シードのみ(`seed`/`anchor_seed` 20260714→20269700)。事前登録済み を実行前にローカル commit。
  - **主判定(2走統合2,504 clean ペア)= +27.42 [+10.05, +44.79] ⇒ REPLICATED(終端)**。半幅17.37=事前予測±17.4。delta法(解析解)が bootstrap と小数第2位まで一致。
  - **run2 単独でも +26.19 [+1.86, +50.42]**(事前登録は要求していなかった)。**走間差 −2.45 [−37.4,+32.4]=区別不能**、異質性は走水準・両run のブロック水準とも **I²=0.0**、20ブロック中16が正。
  - **留保④: 定性解消(単一軌跡の産物でない)/ 定量未解消**(pooled は実現2ネットに条件付きの固定効果量、τ² は k=2/1df ゆえ推定不能。レシピ一般の区間には k≥5)。効果量の見出しは選択汚染なしの **run2 +26.19**。
  - 健全性: 新規5,120局で非合法手0、value 凍結 GREEN、軌跡は run1 の4sd帯内、手検算3種一致、cargo test 24 passed。
  - **★計測系の再発防止(重要)**: 今セッション**閾値/推定量/恒等式の転用ミス4件**(095 回帰ゲート閾値・096 層別推定量・097 シードガード閾値・097 帳簿式)。**今後は帰無対照でノイズ床を実測してから定数を決め、対照を self-test に同梱**(`null_control_selftest`・`assert_anchor_stream_changed` が実例)。小さな宿題: `SESSION_RATE_SKEW` の帳簿式を千日手系(`rep_win`/`rep_lose`)込みへ一般化、15%閾値も勝率調整済み指標へ。
- **共有幹+value蒸留正則化の運搬路スクリーニング(Task 098)= DONE/NULL_CARRIAGE_WALL** (`reports/task098_value_carriage.md`)。**本項目(「T2-loop 凍結方式の比較(共有幹+WS-T value正則化)= READY FOR DESIGN」)はこれで実施済み**。
  - 一変数=value 塔の扱い(完全二塔凍結→共有幹+λ正則化)。ブリーフの「二塔のまま凍結解除+λ·MSE」は `clone_as_dual` が独立2複製ゆえ**証明可能な no-op** だったので共有幹化(人間承認済)。
  - **主判定(084方式・学生同士400ペア800局)= −0.87 [−23.92,+22.18]**、半幅23.1。CI が0を跨ぎ value MAE 59.2cp>30cp ⇒ `NULL_CARRIAGE_WALL`。**2学生は区別できない**。
  - **★機構が判明**: 候補と WS-T のラベル素性が実質同一(std 1502/1500、corr_gikou 0.958/0.957、|Δ|median 201/202)。59.2cp のドリフトは「技巧2 を同じ精度で近似する別の点」への移動=**同じ関数まわりのノイズで新情報でない**。085/nodchip の「上書き元が真に強い時だけ効く」の value チャネルでの再確認。
  - **λ は設計どおり機能**: {3,30,300} で mae 44.3/28.0/19.8 と単調拘束、健全性全点維持。λ=3 本走で std比 1.0014・corr 0.9368、**082 の gen100 事故(0.475)は起きず**。「崩壊させずに value を動かす λ 帯」が実測できた=選択肢B(自己対局リターン+λ錨)の設計に直接使える。
  - gikou@6 tripwire +68.72 [+17.13,+122.74] は**事前登録どおり tripwire 扱い**(256ペアで粗く、アーキが二塔と違うので 096/097 の継続値でない)。
  - **∴ 既知の運搬路は方策(092-0 bridge null)・value(本タスク)とも閉じた**。次は「新情報を作る」(value に独立教師信号+λ錨 / より強い教師)か資源復帰。
- **探索裏付き value ラベルの診断(Task 099)= DONE/REGRESSION、(a')経路も閉** (`reports/task099_search_label.md`)。
  - **構造的にクリーンな対照**: 097 run2 の value 塔は WS-T と bit-identical(実測 0.000e+00)ゆえ **②WS-T+探索 と ③097run2+探索 は value ヘッド同一・違いは policy だけ**。
  - **主判定 REGRESSION**: median|cp−gold| = ①219.3 / ②264.6 / ③267.3、**d31=+48.07 [+33.3,+60.8]**。両 gold(既存 n=2,882 / champion holdout n=2,888)一致。
  - **★本題はバイアスフリーの d32=+2.70 [−3.66,+7.79]**: ②③は gikou からの距離が同一(|Δ|vs gikou 242.653 vs 242.672)ゆえ錨バイアス相殺。②③の直接距離 median 16.6cp・corr 0.998、**sims 32/128/512 全てで CI が0を含む** ⇒ **+26 Elo 強い policy を積んでも探索が到達する value はほぼ同じ**。
  - 「探索が悪化」は額面不可: 探索は gikou 多様体から離れる(corr 0.958→0.952)が gold は gikou 蒸留物の深探索なので機械的に罰する=事前登録した錨バイアスの領域。d31 は sims を上げると +48→+13.7→+10.1 と縮み CI が0を含む=**浅い木の産物**。
  - Stage2: ③は①から median 188.7cp(098 の 59.2cp「殻の内側」と質的に違う=関数は大きく動いた)。
  - **申し送り**: gold では「離れた方向が良いか」を測れない。測るなら**②のラベルでミニ学生試験**(2M rescore S=32 実測 5.65h、計~8.7h)。d32≈0 ゆえ ③でなく ② で足りる=ループ路線の問いではなく「AZ探索ラベル一般」の問い。
  - **計測系**: `m2_usi.new_state` に `current_player` の符号バグ(sfen 直接指定時)。対局は全て startpos+moves 経由ゆえ既存測定は無傷、修正は将来 sfen を使う時。
- **T2-loop 教師品質endpoint再設計= 部分解決**
  - 旧training-only補助head橋は引退。Task 093で **gikou@6直接対局**がpolicy変化の外部負傾斜を検出した。value教師性は感度実証済みのTask 084方式(候補自身value→2M rescore→学生対決)を使う。
  - 推論時policy head学生や手順位直結損失は将来候補だが、092-Bの25iterゲートには必須でない。
- **T2-loop 凍結方式の比較(共有幹+WS-T value正則化)= READY FOR DESIGN / 要 Keiichiro判断**
  - 何を: Task 091完全二塔に対し、共有幹を更新しながらWS-T valueへの蒸留/replay正則化で出力を拘束する一変数比較。λ対数ラダー→25iterゲート。
  - 判定: 内部は健全性、教師=value候補自身の084方式、外部=gikou@6直接対局。092-0のMAE約10cp改善は棋力差にならなかったため、value drift/改善はその検出床を明確に超えること。2M PASSは085を踏まえ32M確認資格まで。
  - なぜ: 完全二塔はvalueを守ったがpolicy-only更新が外部 `−48 Elo/25iter` で悪化。共有表現を許しつつvalueを保護すればこの副作用を避けられるかが次の一変数。
  - いつ: λ、停止条件、value許容幅を別タスクで事前固定し、人間承認後。追加実行方向なので自動着手しない。
- **T2-loop soft教師化 / Gumbel・sims比較(優先度: endpoint修復後)**
  - 何を: (a) hard argmaxではなくtop-k/訪問分布soft target、(b)低simsで改善を作る探索target(Gumbel等)またはsims予算、を各一変数で比較する。
  - なぜ: 091のargmax情報損失とsims32の改善幅不足は未決。ただし092-0で現NNUE橋が最大方策差にも無反応だったため、先に計器を直す。
  - いつ: endpoint陽性対照PASS後。現状でのhard/soft A/Bは行わない。
- **目標再設定 — 別軸(要 Keiichiro)**:
  - (a) **天井内最大化**: 32M→64M 蒸留 follow-up(逓減だが正、`task064` 比 +elo 見込み)。最小コスト・既存パイプライン流用。(c) と並行/代替。
  - (b) **探索裏付きラベルの規模曲線(Task 100 → 100b → 8M → 32M)★ 2026-09-11 に新規開通の候補**: Task 100 で「WS-T+Gumbel MCTS S=32 の root value」を教師にした NNUE 学生が、生 value 教師の学生に **+196.7 Elo [+170, +225]@1M**(`reports/task100_search_student.md`)。084 の +237@2M が 085 で 32M 反転した前例があるため、**2M 確認(Task 100b)= 保持: 566-3-231、+155.0 Elo [+130.4, +180.6]**→ **4M 確認(Task 102a)= +124.0 [+99.5, +149.8]、スケール対照(102b)は null(+8.3 [−15.2, +31.8])** = 利得は eval スケールでなくラベルの中身、ただし −36 Elo/倍加で縮む(log2 線形外挿: 8M +86 / 16M +50 / 32M +13、`reports/task102_scale_curve.md`)→ **8M(Task 103a)= +107.1 [+83.2, +131.9]、減衰 42→31→17 と鈍化、漸近モデル ≈ +70**。**S=8 でも S=32 と同等以上(103b: +222@1M)= 利得は一手先の混合、S=8 なら 32M rescore ≈ 9.3h**。絶対値(103c)は相性成分を差し引き +40〜+90(`reports/task103_scale_8m_sims_absolute.md`)→ **次 = 32M 決戦(S=8、champion 構成で SPRT、1 日)**(~48h GPU、champion 構成で学習し `task064_32m_lr1e-5` と SPRT)** の順で規模曲線を引く。反転すれば正則化(084/085 型)、残れば技巧2@d8 に無い「使える情報」の初の実証。安価な機構切り分け=生ラベルを std 合わせ(1503→1446)した対照学生。
  - 次 /plan で M2 or (a) を具体化。
- **【棄却済(自己ラベル族=枯渇)】温度 T↓ / quiet フィルタ / 深さ d14→d16 / より深い技巧2** — 067-072 で「ラベルをいじる/教師を深く」は単独で天井超えしないと確定。再訪は別アーキ(②)と組む場合のみ。基盤(`--temperature`/`--top-n`/`--depth`/`--val-frac`/`gikou_relabel.py`)は実装済で保全。
- **fixed-nodes 探索器比較(純探索アルゴリズム分離)** ★ Task 060 の follow-up
  - なぜ: Task 060 は movetime 固定=**実装速度込み**の総合力(yane SIMD+差分更新 vs binbo full 再計算=約 4× 深さ)。
    純粋な探索アルゴリズムの優劣を測るには nodes を揃える必要がある。
  - 何を: 同一 eval × 同一 fixed-nodes で 自前探索 vs YaneuraOu 探索(`nnue-vs-yane` を `go nodes` 対応に拡張)。SPRT は流用。
  - いつ: movetime 結果が「意外に競った」場合や、探索アルゴリズム自体を強化したくなったら(現状 0-19-0 で優先度は低い)。
  - 優先度感: 低〜中(任意 follow-up)。
- **教師データ量のさらなる増量(32M→64M)** ★ Task 064 の follow-up(優先度低下)
  - なぜ: 32M は 16M に **+141 elo(棋力は逓減せず)**だが、**val は逓減**(8→16 +0.0174 / 16→32 +0.0111)で **train-val gap 底(~26k=技巧2 depth8 ラベルノイズ)が不動**=蒸留の情報量は枯れつつある。64M は val 天井(0.90)で頭打ち濃厚、棋力は残るが gap 底は動かない。
  - 何を: さらに +32M 生成(~5.6h@16w, ~6.4GB disk)→ 64M cache(val 固定)→ 32M champion(`task064_32m_lr1e-5`)と SPRT。手順は 062/064 と同一。
  - いつ: 自己強化ループより棋力 ROI が高いと判断した場合のみ。**現状は自己強化ループ(教師自体を上げる)が ROI 上**=64M は任意 follow-up。
  - **進捗(2026-09-11, Task 101 stage 1)**: `teacher/run32m_g`(base_seed 20310101、他 run と非衝突)の生成を空き CPU で有界起動(12 worker、22:57 停止、resumable)→ **3200/3200 シャード完走(9/12 09:22)**。生成段は終了、次は `prepare_data` で 64M cache(RAM 壁 → memmap 化が先)。再開: `MAX_SECONDS=<秒> WORKERS=16 scripts/detached.sh gen64m bash ~/binbo-shogi-data/teacher/run32m_g.gen.sh`(省略で無人完走)。**前提課題**: 64M の cache/学習は配列 ~19.7GB vs WSL RAM 21.5GB で RAM 壁 → `prepare_data`/`train_nnue` の memmap 化(小改修)か `.wslconfig` 増量が先。
  - 優先度感: 低〜中(自己強化ループに価値が移った)。
- **ローカル自動対局基盤(engine A vs B)★★** — **実質 DONE(Task 060/061)**: `nnue-vs-yane`(内部探索 vs 外部)・`yane-vs-yane`(eval-vs-eval)
  + trinomial SPRT(`src/sprt.rs`)+ 勝率/Elo/CI + 色交替・ペアopening・per-game TSV。残: 大規模 1000 局の常用運用(必要なら)。
  - なぜ: 改善を数字で測る(「測ってから走る」流儀)。eval 比較・棋力回帰の土台。
  - 何を: A vs B、先後入替、1000 局、勝率・Elo・信頼区間、棋譜保存。USI 駆動(本タスクで判明した
    **read-write FIFO で stdin を保持し探索中断を防ぐ**運用を流用)。技巧2 Lv 対戦(`nnue-vs-gikou` の知見)も統合。
  - いつ: Phase 2 で nn.bin が載ったら即(1000 局の前提基盤)。
  - 優先度感: 高。
- **大会用の合法教師ルート(電竜戦を見据える)**
  - なぜ: dlshogi は私的使用のみ=**大会で封印**(`docs/USED_COMPONENTS.md`)。大会教師は別の合法ルートが要る。
  - 何を: 許諾明確な CPU エンジン(技巧2=GPL 等)で深探索ラベル、or 自己対戦生成。事前生成・事前学習で
    「大会期間中の他者ソフト取り込み禁止」に抵触しない形にする。
  - いつ: 自前 NNUE v0 が一定の強さに達し、提出が現実味を帯びたら。
  - 優先度感: 中(遠い目標だが帳簿は最初から)。

---

## A. 探索(search)

> 時間管理(go の持ち時間 → 反復深化を予算内で打ち切り)+ 最低限の info 出力は **Task 027 で実装済み**
> (`docs/...` ではなく STATUS 完了履歴 / `reports/task027_time_management.md`)。以下はその残課題を含む。

- **`stop` コマンド対応(GUI からの探索中断)**
  - なぜ: GUI の停止ボタンや ponder 解決で即座に bestmove を返せる。現状 `stop` は no-op(時間で切るのみ)。
  - いつ: 並行処理(探索スレッドと USI スレッドの分離 or 共有 AtomicBool)が必要。ponder や対局運用で必要になったら。
  - 優先度感: 中。

- **高度な時間配分(勝負所で多く使う等)**
  - なぜ: 現状は素朴(byoyomi ほぼ全部 − マージン、持ち時間は `/30` の粗い分配)。局面の難しさ・残り手数推定で配分を改善すると、同じ持ち時間で強くできる。
  - いつ: 素朴な配分が棋力/時間切れの律速になったら。実対局の時間運用で問題が出たら。
  - 優先度感: 中〜先。

- **pv(読み筋)出力** — **実装済み(Task 031)**。三角 PV テーブル(探索中に PV を持ち回る)+ TT exact カット箇所の
  TT 補完で、反復深化の各深さで `info ... pv <手列>` を出力・思考ログにも記録。詰み発見時は pv が詰み手順で
  `score mate N` と長さ一致。探索の最善手・スコア・ノード数は不変(`reports/task031_pv_and_eval_diagnosis.md`)。

- **MVV-LVA(取る手を価値順に)** — **実装済み(Task 028)**。中盤 d4 で 89%・d5 で 50% のノード削減
  (`reports/task028_mvv_lva.md`)。駒価値は探索順序ヒント専用(評価には不使用、`DESIGN.md`)。

- **キラームーブ / ヒストリーヒューリスティック** — **実装済み(Task 049)**。`Opts.use_killers/use_history`。
  beta-cutoff した静かな手を ply ごと killer 2つ + history 配列(`from*128+to`)で優先。探索バンドルの一部として棋力 +(下記 Task 049)。

- **固定サイズ配列 TT(+ 置換方式、世代管理)**
  - なぜ: 現状の `HashMap` は肥大とハッシュのオーバーヘッドがある。配列化でメモリを固定・高速化し、衝突時はキー照合。
  - いつ: **メモリ/HashMap オーバーヘッドが律速になったら**。時間管理で深く長く読むようになると `tt_size` が膨らみ近づく(Task 023 で深さ5に数百万エントリを観測)。
  - 優先度感: 中(時間管理導入後に再評価)。

- **PVS(NegaScout)** — **実装済み(Task 049)**。`Opts.use_pvs`。最初の手は全窓、以降ゼロ窓 scout + 改善時のみ全窓再探索。
  αβ 値不変で深さ +(同 seed 24局 37.5%→41.7%, 深さ 4.95→5.01)。MTD(f) は未(必要なら今後)。

- **静止探索(quiescence)** — **実装済み・棋力の決定打(Task 049)**。`Opts.use_quiescence`。葉(depth0)で取り合い延長
  (王手中は全回避手, それ以外は取る手のみ)+ stand-pat, ply cap 32。**Task 031 の「主症状でない」診断は外れ**だった:
  水平線効果(戦術的取りこぼし)が棋力の真の律速で、静止探索 + killer/history + 王手延長 + PVS のバンドルで
  **技巧2 Lv1 20.8% → 47%(50局)= 30% 目標突破**。詳細 `reports/task049_campaign_log.md`。
  - 教訓: 評価(KP→正準HalfKP, 度合い ~70%)を散々いじっても棋力が動かなかったのは、律速が探索側だったため。

- **さらなる探索(LMR / null-move / aspiration)** ★ Task 049 の上積み候補
  - なぜ: 同じ1秒でさらに深く。LMR(遅い静かな手を浅く)・null-move(手抜きで cutoff 証明)・aspiration 窓。
  - いつ: 棋力をさらに上げたいとき(Lv1 を超えて Lv2+ を狙う等)。Task 049 で探索が効くと実証済み。
  - 優先度感: 中(探索が棋力に直結すると判明したので有力)。

- **千日手の正式対応(sennichite, GHI 問題)**
  - なぜ: 正しさ。引き分けの千日手、連続王手の千日手(反則=負け)を正しく扱う。TT との相互作用(GHI)に注意。
  - いつ: 実対局で千日手が問題化したとき、または TT 高度化(配列化/世代管理)と並行して。詰み値の ply 補正(Task 023 で対処済み)の延長線。
  - 方針候補: `pos.key()` の出現回数で同一局面4回を引き分け扱い。連続王手は別判定。
  - 優先度感: 中。

## B. 評価(eval)

- **評価品質の診断(技巧2評価との局面別比較 / 負け棋譜解析)** — **実装済み(Task 031)**。診断 CLI `nnue-diag`
  で代表局面 + 実戦を生徒 NNUE(静的+探索+pv)と技巧2 depth1/depth8 で1手ずつ突合。
  - **結論: 符号(有利の向き)はほぼ正しいが、有利/不利の度合いを判別できず0付近に圧縮**(飛車得=生徒+603/技巧2+2635、
    飛車損=生徒−129/技巧2−2405、符号6/6一致)。実戦では形勢が傾く局面で「微有利」と誤認し数百cp 負けに気づかない。
  - これが「深く読む生徒が depth1 技巧2 に負ける」機序(葉がどこも互角に見え勝ち/負けへの勾配が立たない)。
  - 詳細: `reports/task031_pv_and_eval_diagnosis.md`。次の手は下記「評価の再学習・キャリブレーション」(C)。

- **評価の再学習・キャリブレーション(圧縮の解消)** — **②目標設計の変更は実装・効果確認済み(Task 033)**。
  cp 直接回帰(MSE, clamp±5000)で度合いの圧縮が緩和: decisive 評価 約3倍、静的分解能 1.86×→3.95×、
  実戦の符号一致 70%→100%、val_corr 0.62→0.75。スケールは `forward*600=cp` で Rust 無改修(`trainer/train_nnue.py --objective cp`)。
  採用 `model_cp_mse_c5000`。詳細: `reports/task033_cp_regression.md`。
  - **残課題**: 技巧2 Lv1 にはまだ 0-8。decisive の度合いが技巧2の ~0.4倍に**頭打ち**=目標設計だけでは限界。
  - **容量は犯人でない(Task 034)**、**decisive 不足でもない(Task 035: decisive |cp|≥1500=23.7%)**。
    真因は**汎化ギャップ(過学習)**: train では度合いを出せる(予測|cp| 3202≈正解4010)が val で平均回帰(1942)。
    → 次は下記「汎化を上げる(データ拡張・過学習対策・volume)」(C)。decisive 重点化・容量拡大は的外れ。
  - 優先度感: 高(継続。次は C の汎化系=左右対称データ拡張が最安)。
  - **【決着】cp vs 標準WDL を棋力で比較(Task 051)— cp 維持**。標準 WDL(勝率空間 MSE + scaling
    400/600/800 + lambda 0.7/0.5, `--objective wdl`/`--wdl-lambda`)を cp と同一データ(2M)・同一構造で
    棋力比較 → **cp が明確に勝つ**(Lv1 cp 29.7% vs 最良WDL 18.2% = +11.5pp、Lv2 +5.2pp。lambda 実結果混合は最弱)。
    「度合い↓でも棋力↑」の逆転は起きず(WDL は度合いも棋力も cp 以下)= 度合い⇔棋力の乖離は**学習目標でなく
    別要因**(KP 表現力 or 先生の質)と切り分け。詳細 `reports/task051_objective_log.md`。
  - **再訪条件(学習目標を WDL に再評価すべき時)**: ①表現力を上げた後(HalfKA/KPP=0付近精度が効く土俵)、
    ②先生を DL 系に変えた後(WDL ネイティブ教師)、③データを桁で増やした後(勝率信号のロバスト性)。
    実装(`--objective wdl`/`--wdl-lambda`、Rust 変更不要)は保持済みなので即再評価できる。

- **差分更新(accumulator の増分更新)**
  - なぜ: KP 変換+特徴変換を毎回フルで回さず、着手の差分だけ更新して高速化。**量子化より効果が大きい可能性**(現状 Rust 側は KP 変換が支配的なため)。
  - いつ: **探索高速化が進み、Rust 側時間で KP 変換が律速になったら**。探索が速くなるほど相対的な重要度が上がる(今後上がっていく筆頭)。
  - 優先度感: 中(探索最適化が一段落したら最有力の評価系最適化)。

- **量子化(int8/int16)**
  - なぜ: NNUE forward を整数化して高速化・省メモリ。
  - いつ: **NNUE forward 計算が律速になったら**。現状 forward は「ほぼ無料」(Task 020)なのでまだ先。差分更新の後。
  - 優先度感: 先。

- **SIMD(整数 SIMD で forward/accumulate)**
  - なぜ: forward/accumulate のベクトル化で高速化。
  - いつ: 量子化と併せ、forward が律速になったら。
  - 優先度感: 先。

- **正準 HalfKP 化** — **実装・検証済み(Task 046=偽 / Task 047=本物)**。正準の核心は **W₁ を両視点で共有**(原典 那須2018 式15〜17)。
  - **Task 046(two-band, BNN2)= 偽の正準だった**: (a)相手ハーフ回転反転・(b)2アキュムレータ連結 は正しかったが、view0/view1 を別帯にして両視点が**独立パラメータ=W₁ 非共有**。利得(共有=パラメータ半減・対称性)を持たず実質「容量2倍」で効かず(val 2735=naive と同一、gap 148→175 悪化、forward −17%)。比較用に残す。
  - **Task 047(shared, BNN3)= 本物の正準**: 両ハーフが**単一帯 `[0,176742)` を共有**、埋め込み **176744行=naive の約半分**。三者比較(同一 8M・同一 val): val 度合い naive 2729 / two-band 2735 / **shared 2769(70.7%, 初の70%超)**、best val_loss **shared 583170(最低)**、epoch1 過学習 gap **+15815(naive +262855 の 1/17)**、val_corr peak **0.9032(最高)**、**forward ~11% 速**(半分の表でキャッシュ局所性↑)。数値一致 shared KP 30000/30000・forward 2.6e-06、magic BNN1/BNN2/BNN3。
  - 判断: 本物の正準は**アーキテクチャとして明確に優秀**(パラメータ半減・過学習激減・val 最高・速い)= 原典設計の正しさを実証。ただし改善幅は小(+1pp)で Lv1 の壁は明確には破れない見込み。
  - 優先度感: 完(本物を検証済み)。→ 下記 KPP へ。詳細 `reports/task046_canonical_halfkp.md`(偽)・`reports/task047_canonical_shared.md`(本物)。

- **評価値の cp 厳密化(技巧2 cp との対応づけ)**
  - なぜ: 現状は順序保存の整数化(`logit*600`)。探索の閾値設計や `info score cp` 表示、人間可読性のために cp の意味を厳密化したくなる場面がある。
  - いつ: 探索の閾値設計や評価値表示で厳密な cp が必要になったら。
  - 優先度感: 必要時(低)。

- **器の拡大(feature_dim↑)の再訪** — **検証済み・採用せず(Task 042)**。
  - 結果: 生800万+mirror で fd8→16→32→64 を再訪したが、主指標は **fd8=2729 / fd16=2716 / fd32=2732 / fd64=2659**。
    fd32 は +3cp の誤差級、fd64 は悪化。技巧2 Lv1 同一 seed 8局も **fd8=2-6-0、fd32=2-6-0**。
  - 判断: 過学習がほぼ消えた後でも、単純な embedding 幅増は val 70% の壁を越えない。fd32 はサイズ 4倍・平均深さ低下で採用価値なし。
  - 採用: **fd8+mirror+生800万を維持**(`model_cp_fd8_mirror_8m`)。
  - 次の示唆: 律速は fd8 の幅ではなく、KP 表現そのものの粗さ。教師 depth↑(045)は技巧2 クラッシュで塞がれたため、次は**正準 HalfKP/KPP**(入力構造)。
  - 優先度感: 完。

- **HalfKA(玉を駒特徴に含める)** — **実装・棋力検証済み、plain は不採用(Task 052)**。HalfKP が原理的に欠く
  king-king 関係(玉安全)を、玉を board piece(id13)として足す。採用 shared 上に **BNN4/Mode::SharedKA** で実装
  (既存 HalfKP 完全保存、Python↔Rust 完全一致、`docs/kp_spec.md` §10、`reports/task052_halfka.md`)。
  - 結果(8M head-to-head, 同一レシピ, 3seed×12=36局/Lv): **Lv3 +18pp(38.9 vs 20.8, 3seed一貫)= 表現で初めて棋力が動いた**
    (047/050/051 の評価変更は全て不変だった)。**だが Lv1 −29pp(負け越し)/ Lv2 −15pp と退化**、Lv4 互角。非単調(Lv1 40/Lv2 29/Lv3 39 横ばい)=棋風の歪み。
  - 判断: **net 非改善 → 不採用**(最弱 Lv1 に勝てない eval は頑健でない)。採用は HalfKP shared 8M のまま。インフラ(BNN4)は保持。
  - **HalfKAv2(自玉-as-piece の冗長除去)— 検証済み・不採用(Task 053)**: BNN5/Mode::SharedKAv2 で実装。Lv1 45.8/Lv2 29.2/Lv3 23.6%
    = 両方の悪いとこ取り(Lv1/Lv2 退化は直らず、HalfKA の Lv3 利得を喪失)。**自玉-as-piece は冗長でなく Lv3 利得に寄与**していた(仮説は誤り)。

- **KPP 拡張(King-Piece-Piece)** — **king-zone を実装・棋力検証済み、不採用(Task 053)**。`docs/kp_spec.md` §9, `reports/task053_kav2_kpp.md`。
  - **king-zone KPP 本実装(BNN6/Mode::KppZone, 4アキュムレータ)**: Python↔Rust 完全一致(numbering 30000/30000・forward 2.5e-06)、
    KPP mirror perm、ragged CSR キャッシュ、8M cp+mirror。**val_loss −8.5%(プロトタイプ −8.7% 再現)= 駒ペアは val を改善**。
  - **だが棋力 ≤ HalfKP**(8M, 同一seed 36局: Lv1 68.1 維持・Lv2 26.4 −18pp・Lv3 16.7 −4pp)= **駒ペアは val を上げるが棋力を動かさない**
    (047/050/051/HalfKA と同じ「度合い⇔棋力の乖離」)。→ 不採用。インフラ(BNN6)は保持。
  - **full KPP(81玉×C(fe_end,2), 差分計算前提)は着手しない**: king-zone(その軽量版)が棋力を動かさない以上、18× 重い full KPP +
    差分更新機構への投資は割に合わない。表現リッチ化の路線(HalfKA/HalfKAv2/KPP)は**棋力天井を破らないと結論**。
  - 優先度感: 完(表現 richer 化は出し尽くした)。真の律速は表現でなく**探索深さ(対 Gikou)or 先生の質**。

- **差分更新(incremental accumulator)** ★ 独立インフラ課題(KPP の効果とは切り離す)。NNUE 本来の do_move/undo_move で
  アキュムレータを増分更新する高速化。**全特徴セットを高速化**(full KPP 等の重い表現を可能にする基盤)。
  - いつ: **速度が律速になった時**(現状 forward はほぼ無料=Task 020、KP/HalfKA/king-zone KPP は全再計算で十分)。
    full KPP など active が桁増える表現を採る時に必須。検証は「アキュムレータ ≡ 全再計算を毎手 assert」(静かに壊れる最大箇所)。
  - 優先度感: 低(今は速度が律速でない。表現リッチ化が棋力に効かないと分かったので、これを急ぐ動機も薄い)。

## C. 学習・データ(training / data)

- **DL系教師への移行(再蒸留)= 現在の本命 ★★★**。`reports/task054_dlshogi.md`。
  - **なぜ**: 評価品質が律速と実測確定(生徒は技巧Lv3/Lv4 を約1000倍ノード読んでも負ける=1ノードの評価が粗い、val_corr~0.90)。
    探索も KP系表現(HalfKA/KPP)も尽きた(049-053)。残る本線は**先生の質**=技巧2(従来型)→ dlshogi(DL系, より真理に忠実)。
  - **Task 054 地固め完了**: dlshogi `model-dr2_exhi.onnx` を 4060 Ti で動作(cshogi.dlshogi 特徴 + onnxruntime-gpu cu12, **5287 pos/sec,
    VRAM 2.8GB, 8M ~25-30分**)。勝率→cp 変換 `cp=600·logit(v) clamp5000`。re-score 5000局面で dlshogi cp vs Gikou cp =
    符号一致 decisive 0.975/相関 0.927/中央差 456cp(dlshogi が鋭い)。スクリプト `trainer/dlshogi_rescore.py`。
    ライセンス: 私的使用無償(山岡氏著作, 追加学習/転用/再配布 禁止)、ユーザー同意済み、生徒重みは非コミット。
  - **Task 055 実施 = 教師のみ差し替え再蒸留 → 全レベルで悪化、不採用**。`reports/task055_dlshogi_distill.md`。
    8M を整列付きで全 re-score(`--cache-dir`, gikou-eval==cache assert)→ `train_nnue --eval-npy` で教師ラベルだけ dlshogi に。
    棋力(同一探索部, 同一seed 36局/Lv): dlshogi 教師版 Lv1 54.2/Lv2 22.2/Lv3 12.5/Lv4 9.7% = **技巧2 教師版(69.4/44.4/20.8/13.9)
    より全レベルで悪化**(−4〜−22pp)。ラベルは正しい(整列・符号一致0.845・parity・詰み)。
  - **解釈**: ① 生徒が dlshogi ラベルを当てはめられない(val_corr 0.85 vs 技巧2 0.90)= **小 KP 生徒は深い CNN 評価を表現しきれない**
    (技巧2 KPP の方が KP 生徒に近い)。② 技巧2 ラベルは depth8 探索済み, dlshogi は静的 1-ply。**「エンジンとして強い」≠「良い蒸留先」**。
  - 優先度感: 素朴な value 差し替えは**打ち止め**(悪化)。残課題 = searched dlshogi(探索焼込ラベルで①②切り分け, dlshogi MCTS engine 要)
    と、より本質的には**生徒の根本強化**(下記)。

- **生徒の根本強化(真の壁=小 KP 生徒の表現容量?)★★ 次の本命候補**。
  - なぜ: Lv3/Lv4 の壁は探索でも(049-050)・KP系表現でも(052-053)・強い先生(055)でも破れず。技巧2 教師でも val_corr ~0.90
    (decisive の約10%で符号誤り)が天井 = **小さい KP fd8 生徒の表現容量そのものが律速**と強く示唆。
  - 候補: HalfKA を活かす生徒(052 で Lv3 +18pp=唯一の生徒側成功, ただし Lv1/Lv2 退化)/より大きい・深い NN(量子化で深さ確保)/
    別パラダイム。fd↑(042)単体は効かなかったので、表現+容量+教師マッチを同時に要する可能性。
  - いつ: dlshogi 素朴差し替えが効かなかった今、最有力の残り筋。

- **過学習対策の本格化(より強い weight decay / 容量調整 / dropout / データ増)**
  - なぜ: 検証ロス改善 → 評価器の質 → 強さ。Task 017 で標準設定は早期過学習、wd=1e-3 で改善を確認済み。
  - いつ: **今**(Task 031 で評価が律速=評価圧縮と判明)。ただし注意: wd を**強める**と val ロスは下がるが、Task 031 では
    その wd=1e-3 重みが decisive 局面で出力を縮めている可能性 → **val ロス最小化と「圧縮を解く(decisive で logit を伸ばす)」は
    トレードオフになりうる**。検証ロスだけでなく `nnue-diag` の符号/度合いも指標にする。上記「評価の再学習・キャリブレーション」(B)と一体。
  - 優先度感: 高(評価再学習の本体)。

- **教師データのスケール拡大 / decisive 局面の重点化(500万〜1000万、~~depth 上げ~~、decisive 比率↑)**
  - なぜ: より多く・多様な教師で**汎化**を上げる。**Task 039〜041 で生100/200/400/800万の系列が汎化を伸ばすと実証**:
    val 度合い 2185(56%)→2455(63%)→2652(68%)→2729(70%)、gap 809→541→285→148、val_corr 0.80→0.84→0.87→0.894。**棋力 0-8→…→1-7-0(初勝利)**。
  - **Task 041 で逓減が明確化**: 倍化Δ +270→+197→**+77**、過学習はほぼ消滅(gap148, epoch1 loss差 −32k)。収集時間は量に比例(4M=46分)で費用対効果↓。
  - **Task 042 で feature_dim↑ も不採用**: 過学習が消えた 8M 上でも fd 幅増は効かなかった。単純増量・単純幅増の両軸はいったん打ち止め。
  - **教師 depth 上げは Task 043/045 で塞がれた(不採用)**: depth12 ~22倍遅+効率0.25(043)、depth10 ~16.6倍遅+効率0.34+**ゲーム82%クラッシュ**(045)。クラッシュはゲームを late-midgame で殺すため**decisive 局面が約1/3に枯渇**(≥1500 23.7%→7.4%, ≥5000 5.3%→0.4%)。同規模対照(045)で depth10 生徒は**むしろ弱い**(Lv1 粘り 99→68手、符号一致 79%→70%)。技巧2 のクラッシュ挙動が「最も学ばせたい decisive」を奪う=depth は使えるレバーでない。詳細 `reports/task045_teacher_depth10.md`。
  - いつ: 正準 HalfKP/KPP が効かなかった場合の継続候補(decisive 比率↑は depth ではなく**サンプリング/フィルタ**で。クラッシュは全復旧)。ディスク試算は Task 016 メモ参照。
  - 優先度感: 中〜低(逓減一段落 + depth 軸は閉。まず構造を試す)。

- **汎化策(度合いを潰さない過学習対策)** ★ 評価系の本命(mirror の上に積む)
  - なぜ: Task 035 で度合いの圧縮 = 汎化ギャップ(train で度合いを出せるが val で平均回帰)と判明。汎化を上げれば val の度合いが伸びる。
  - (1) **左右反転の対称データ拡張** — **実装・採用済み(Task 036)**。`train_nnue.py --augment-mirror`(`trainer/mirror.py`)。
    train/val ギャップ −32%(val 度合い 1942→2163)、val_corr 0.75→0.80、過学習半減、推論コストゼロ。採用モデル `model_cp_mse_c5000_mirror`。
    左右反転の正しさは二経路一致+involution+詰み保存で検証(`trainer/verify_mirror.py`)。※技巧2 cp は鏡映で非対称(技巧2 評価が左右非対称)だが真の値は不変、対称化で除ける。
  - (2) **埋め込み次元↓** — **検証済み・効かない(Task 037)**。feature_dim 32→16→8(mirror 維持)で gap・val 度合いは横ばい
    (Task 034 の 32→256 横ばいと合わせ、**feature_dim は汎化の律速でない**=上げても下げても効かない)。ただし fd8 は同品質で 4倍小・2.3倍速のため
    **効率採用**(ベースを `model_cp_fd8_mirror` に)。
  - (3) **dropout** — **検証済み・採用せず(Task 038)**。アキュムレータ dropout を p=0.1〜0.5 で試したが、過学習は減る(loss差↓)ものの
    **val 度合いを下げてギャップを縮める**(val 2185→1475 単調低下、train が落ちるだけ)=weight decay 同様の度合い圧縮。実装は推論に漏れない(parity ok)が採用基準を満たさず。
  - **メタ結論(正則化は打ち止め)**: 容量/分散削減(wd 032・次元↓ 037・dropout 038)は系統的に**度合いを圧縮**し val を上げない。
    **val を上げたのは情報を足す左右対称拡張(036)だけ** → 次段は正則化でなく**データ(量・多様性)**。
  - 優先度感: 完(正則化系は確定)。次は下記「教師データのスケール拡大」。
    depth を上げる際は**技巧2 の暖機状態依存クラッシュ頻度を再計測**(Task 016、binbo は復旧で吸収済みだがバイト再現性が要る用途は crash=0 shard 限定 or 別 seed 再生成の運用)。サイズ見積りは Task 016 のディスク試算メモ参照。
  - 優先度感: 高(Task 033 の直接の続き)。並行候補=NNUE 構造拡大(feature_dim/hidden_dim、容量律速の検証)。

- **自己対戦学習(蒸留の先、知識ゼロにより近い方式)**
  - なぜ: 技巧2 蒸留の先で、自己生成データで強くする。`DESIGN.md`「AlphaZero 系は将来の差し替え候補」とも関連。
  - いつ: 蒸留 NNUE が一定の強さに達してから。
  - 優先度感: 先。

- **df-pn 詰み探索の援用**
  - なぜ: 終盤の詰み発見、コールドスタート援用。作者の df-pn 実装経験を活かせる。
  - いつ: 終盤の詰み逃しが棋力の律速になったら。`DESIGN.md`「df-pn 詰み探索をコールドスタートに援用する」に記録。
  - 優先度感: 中〜先。

- **指し手の多様性(softmax / 温度サンプリング)**
  - なぜ: 決定的すぎる対局を多様化(自己対戦の棋譜多様性、人間対局の幅)。現状は乱数オープニングで多様化している(`nnue-selfmatch` 等)。
  - いつ: 自己対戦学習や棋譜生成で着手レベルの多様性が必要になったら。評価値→勝率変換の温度(`DESIGN.md` 未決)とも関連。
  - 優先度感: 先。

## D. 環境・運用(infra / ops)

- **GPU / PyTorch CUDA 環境の整備** — **対応済み(Task 039)**。
  - 整備: WSL の RTX 4060 Ti に CUDA torch(`python -m venv venv-gpu && venv-gpu/bin/pip install torch==2.12.1 numpy cshogi` = `+cu130`, cuda True)。
    `train_nnue.py --device {auto,cpu,cuda}`(mirror は CPU→device 転送、推論は常に CPU、export 後数値一致 ok)。
  - 速度: 1.95M batch1024 で **CPU ~18s → GPU ~3.6s/ep(約5倍)**、CPU/GPU で結果同等。データ増量・再学習サイクルを高速化。
  - `nvidia-smi` は `/usr/lib/wsl/lib/nvidia-smi`(SSH では PATH 外)。GPU venv は /tmp(ephemeral, 上記 pip で再現)。

- **永続/一時の分類と環境の永続化** — **対応済み+拡張(Task 044)**(`docs/data_storage.md`)。
  - 重み・教師データに加え、**技巧2(`engines/gikou/`)・venv・CARGO_HOME も永続化**(Task 044 で /tmp 消失=技巧2/venv 喪失を機に)。
    判断基準=再取得コスト。`/tmp` は KP キャッシュ等(分単位で再生成可)だけ。正準パスは `scripts/env.sh`、規律は `AGENTS.md`。
  - ShogiHome 用重み = `~/binbo-shogi-data/models/model_cp_fd8_mirror_8m/nnue_f32.bin`。
  - 残(作者): **マシンのスリープ無効化**(Task 044 の直接原因=夜間に /tmp 消失。長時間無人ジョブの前提)。WSL からは制御不可。
    **※ Task 063 で確認**: ホスト AC standby は既に無効(`powercfg`)、Task 060–062 の長時間ジョブ喪失は**スリープではなく別原因**(プロセス道連れ + WSL2 自動停止)= `detached.sh` で対策済み(下記)。

- **長時間ジョブの安全起動(セッション越え)** — **対応済み(Task 063)**(`docs/long_running_jobs.md`)。
  - 真因: harness の background ジョブは **Claude プロセスの子→Claude 終了で道連れ kill→distro に残プロセス無→WSL2 が VM ごと自動停止**(/tmp wipe)。ホストスリープは無関係(AC standby=0=既に無効、Kernel-Power 42/1074/6008 なし)。Task 060/061/062 で学習(~90分)・SPRT(~27分)を反復喪失。
  - 対策: `scripts/detached.sh <tag> <cmd...>`(`setsid+nohup+</dev/null+ログ化`、ログ=`$LOGS_DIR/<tag>_<ts>.log`、完了 marker `=== DETACHED_DONE rc=N ===`)。新セッション/`/init` 再親付けで境界に耐え、実行中はプロセス常駐で WSL も保持。スモーク検証 PASS(SID=自身・PPID=`/init`・marker 付与・grep 完了判定)。
  - 任意の上積み(文書化のみ): WSL 常時起動=Windows 側 keepalive(`wsl.exe -d <distro> --exec sleep infinity` を常駐)推奨。`.wslconfig vmIdleTimeout` はバージョン差で不確実。

- **学習(`train_nnue.py`)への resume 機構追加** ★ 温存(Task 063 で課題提起)
  - なぜ: `gen-teacher`/`prepare_data` は manifest+atomic shard で resume 済だが、**学習は resume なし**。`detached.sh` でセッション境界には耐えるが、電源断/OOM 等の別死因では best-val 以外は頭からやり直し(Task 062 で実際に ~90分の学習を喪失)。
  - 何を: optimizer/epoch/scheduler 状態を周期チェックポイント保存し、`--resume <ckpt>` で再開。best-val とは別に latest を持つ。
  - いつ: 自己強化ループ(世代ごとに長時間学習を反復)で中断コストが顕在化したら。現状は best-val 部分救済 + `detached.sh` で許容。
  - 優先度感: 中(自己強化ループ着手時に上がる)。

## 横断・マイルストーン

- **粗い棋力当たり測定(技巧2 レベルとの対戦プロトコル)** — **実施済み(Task 029)**。
  - 結果: 現行生徒は 1秒/手・depth上限12で技巧2 Lv1 に 0勝8敗、2秒/手でも 0勝2敗。互角ゾーンは Lv1 より下。
  - 詳細: `reports/task029_gikou_strength.md`。

- **精密な棋力測定(多局数・統計つき)**
  - なぜ: 粗い当たりではなく、改善後の強さを誤差つきで測る。Lv1 に届いた後、どのレベル帯で互角かを安定して見る。
  - いつ: binbo が技巧2 Lv1 に勝ち越す兆候が出たら。現状は Lv1未満なので、精密測定より改善が先。
  - 方針候補: `nnue-vs-gikou` を使い、同一時間条件・乱数オープニング・先後入替・逐次実行。互角候補レベルだけ 20局以上に増やす。
  - 優先度感: 改善後。
