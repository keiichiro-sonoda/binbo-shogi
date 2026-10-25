# Task 077 — M2: 価値のみ蒸留 warm-start → net→USI 橋 → champion との対戦で立ち位置測定

**目的**: (c) の第2マイルストーン。①champion と同一の教師信号(32M hcpe の技巧2@d8 ラベル、**再ラベルなし=ユーザ決定**)で AZ ネットの**価値のみ**を温め(方策はランダム初期化のまま=一変数)、②AZ ネット+少 sim MCTS を USI エンジン化、③既存ハーネス条件を踏襲した対局で champion(αβ)との立ち位置を測る。**合格=測定の成立**(勝敗は問わない。勝つのは M3 の課題)。実測 2026-07-02、RTX 4060 Ti 16GB、隔離 venv `~/task074_frameworks/venv_pgx`。実験コード(m2_*.py)は隔離・非コミット、**リポジトリは文書のみ・Rust 無改変**。

## ステップ0 — 蒸留データ(m2_distill_data.py)
- **供給源**: `$TEACHER_DIR` の hcpe shards(run1m〜run16m_f=32M 総体、3,200 shards)。cshogi で hcp→SFEN+cp(手番 POV)をデコードし、**train 2M**(全 shard から seed 付き比例サンプル)+ **val 50k**(**末尾 5 shard を丸ごと hold-out**=shard 単位で分離、train と対局重複なし)。
- **表現**: SFEN→盤面/持駒の numpy パース(from_sfen 等価、後述の高速化)→ `_observe(flip=True)` バッチ(jit+vmap)→ **(9,9,119) bool を np.packbits**(1205 B/局面、2M ≈ 2.4GB memmap)。**flip=True が必須**: 対局時の `state.observation` は `_observe(_x, flip=True)` と一致(全 ply で実測確認。flip=False は相手視点=サイレント向き事故の芽)。
- **値ターゲット(式)**: `v = tanh(cp/1200) ≡ 2·sigmoid(cp/600) − 1` ∈ [−1,1](既存の `winrate=sigmoid(cp/600)`、scale_cp=600 と整合。詰み ±30000 → ≈±1)。
- **検証ゲート**: numpy パーサは `from_sfen` と **2,000 局面で完全一致**(盤・持駒・手番・observation)。ビルドは **~21k 局面/s**(2M+50k 全体 ~10 分)。

## ステップ1 — 価値のみ蒸留(m2_distill_train.py)
- AZNet 64ch×5L(M1 動作点の構成、0.80M params、`network.py` 流用)。**loss = MSE(value) のみ** → **policy head のパラメータは計算グラフ上 value に寄与せず勾配が構造的にゼロ=未学習のまま**(【JAX 学習ノート】明示 freeze(`optax.multi_transform`/`masked`)を書かなくても、loss の形だけで部分学習になる。ただし **policy head の BatchNorm 統計は is_training=True の forward で更新される**=パラメータでなく統計のみ。挙動は無害だが「完全に手つかず」ではない点を明記)。
- adam lr 1e-3、minibatch 4096、4 epochs(75s/epoch=全 4 epoch で ~5 分)。
- **結果(held-out val 50k)**:

| epoch | 0(init) | 1 | 2 | 3 | 4(best) |
|---|---|---|---|---|---|
| val_corr | 0.005 | 0.866 | 0.884 | 0.897 | **0.900** |
| val_mse | 1.292 | 0.087 | 0.078 | 0.076 | **0.073** |

- **蒸留成立は明確**: 乱数(corr 0.005)→ **0.900**。この 0.90 は 052/062 以来知られる**技巧2@d8 ラベルの教師ノイズ天井 ~0.90 とほぼ同水準**(champion の val_corr 0.886 は cp 空間の別指標=直接比較不可だが、両者とも同じ教師の天井に張り付く点は示唆的)。**AZ 盤面 CNN は champion(HalfKP)と同じ教師信号を ~5 分の学習で同水準まで吸収した。**

## ステップ2 — net→USI 橋(m2_moves.py / m2_usi.py)
- **写像**: Pgx 升番号は cshogi/YaneuraOu と同一(`(file−1)·9+(rank−1)`)、手番視点は `Inv=80−sq` 鏡映。方向 27 種は**ハードコードせず** Pgx 自身の `LEGAL_FROM_IDX` から `(from,to)→direction` 表を導出、from 解決も `Action._from_dlshogi_action` の numpy 忠実再現。
- **検証(057 の教訓=付番のサイレント故障を測定力ある形で潰す)**: ①startpos 30 手一致 ②**ランダム対局 2,000 局面で「Pgx 合法 action→USI」集合 == cshogi `legal_moves`(ground truth)が完全一致**(持駒あり 1,579・成駒あり 1,231 局面を含む)+ 逆方向 roundtrip 恒等 ③**negative control**: 写像 1 エントリ入替→startpos で即 RED(検出力確認。当初 argwhere で選んだ稀エントリは 100 局面で検出されず=**「絶対に踏まれる」エントリを壊すよう修正**した上で RED)。
- **USI エンジン(m2_usi.py)**: usi/isready(JIT warmup)/position(startpos+moves, prefix cache)/go→bestmove の最小実装。探索= mctx `gumbel_muzero_policy` **batch=1・gumbel_scale=0**(決定的、探索ノイズなし)、詰み(合法手ゼロ)→ resign。
- **sims↔時間対応(実測, batch=1, warm)**:

| sims | 4 | 8 | 16 | 32 | 64 | 96 | **128** | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| ms/move (mean) | 7 | 27 | 38 | 29 | 53 | 78 | **101** | 248 | 207 |

  → **sims=128 ≈ 101ms/move を採用**(champion の byoyomi 100ms とほぼ等時間)。batch=1 の latency は dispatch オーバーヘッド支配で sims にサブリニア(M1 のバッチ selfplay の超線形とは別の領域)。
- **健全性(20 局サニティ)**: 全局正常完走・**非合法手 0・クラッシュ 0**(全局 mate 決着、plies 中央値 33)。

## ステップ3 — 立ち位置測定(m2_match.py)
- **条件(既存 Rust ハーネス完全踏襲を Python で再実装**、generic usi-vs-usi は Rust に無くリポジトリは文書のみ制約のため): YaneuraOu champion(`task064_32m_lr1e-5`)options=`USI_OwnBook:false,Threads:1,MinimumThinkingTime:0,NetworkDelay:0,NetworkDelay2:0`、`go btime 0 wtime 0 byoyomi 100`。開幕=乱数合法手 6 plies(seed 20260702)を**同一開幕×先後入替ペア**、max_plies=320=draw。**審判= cshogi.Board**(全手合法性・詰み・千日手 `is_draw`)。elo は per-game score {0,½,1} の正規近似 95%CI を `elo=−400·log10(1/s−1)` に変換。
- **結果 vs champion(200 局, AZ=蒸留値+乱数方策+sims128)**: **0勝 0分 200敗**(全局 mate、plies 中央値 ~33)。score=0 のため elo は点でなく**上界**で報告: score < 1.49%(片側 95%, 0/200)→ **elo ≤ ≈ −730**。
- **参考 vs 技巧2@DepthLimit6(gikou anchor 条件: OwnBook:false,Threads:1,DepthLimit:6,MinThinkingTime:10, cwd=gikou dir=072 の教訓)(100 局)**: **0勝 1分 99敗**(mate 99・千日手 1)、score 0.005 → **elo ≈ −920(95%CI 上側 ≈ −730)**。外部アンカーに対しても大差。
- 全 322 局(smoke 2+サニティ 20+本測定 300)で**非合法手 0・エンジン失敗 0**=橋と審判は完全動作。

## M2 判定 = **PASS(測定の成立)**
- 合格条件どおり「対局が正常完走し、立ち位置が数字(CI 付き)で得られた」: **champion から ≤ −730 elo、gikou@6 からも ≈ −920(上界 −730)**。勝敗は問わない設計(勝つのは M3 の課題)であり、大差の敗北自体が想定内の測定結果。
- **解釈(なぜ value corr 0.90 でもこの弱さか)**: 方策がランダムだと Gumbel MCTS は root で**プライアからの考慮手 16 手(mctx 既定 `max_num_considered_actions=16`)**を実質乱数で選び、その先を 1-2 ply の value 読みで順位付けするだけ=**探索の狭さ・浅さと方策の欠如を、盤面 value の知識では補えない**(champion は同じ教師知識+αβ depth~15)。この「value だけでは登れない出発点」を数字で確定できたことが M2 の成果 — M3 の方策改善ループが持ち上げるべき高さが定量化された。

## 詰まりどころと回避(実測)
- **cshogi 1.0.2 は numpy<1.27 を pin**し pip が numpy を 1.26 に下げる(jax 0.10.2 は numpy≥2 要求)→ **numpy 2.5 に戻して共存確認**(pip の依存警告のみ、cshogi は実行時 numpy2 で全機能正常)。
- **un-jitted `_observe` は ~0.5s/局面**(eager の逐次 dispatch)— 検証ループで jit を忘れると 2,000 局面が ~15 分に。jit+vmap なら 27k/s。
- **`jnp.stack` を数千個の小 device 配列に使うと ~15× 遅い**(147/s)→ 各局面を numpy 化して `np.stack`→1 回の device_put(21k/s)。M1 の「形状・転送はまとめる」の続き。
- 上流 `from_sfen` に**持駒 2 桁枚数("18P")を桁ごとに読む潜在バグ**(自作パーサは正しく処理、上流へは影響範囲小=歩 10 枚以上の持駒のみ)。
- 負の対照は「壊しても踏まれない箇所」を壊すと**通ってしまう**→ 必ず踏まれるエントリ(startpos の 7g7f)を壊して RED を確認(検出力の検証自体が要検証)。

## M3 への申し送り(検証計画素案)
- **出発点**: 本 ckpt(value corr 0.90 / policy 乱数)= champion ≤−730 / gikou@6 ≈−920。M3 = **M1 動作点(steps512/batch256/sims16-32)で AZ 自己対局ループを短期回し、gen1 が本 warm-start を USI 対局(本ドライバ・sims128)で上回るか**を測る=「MCTS 方策改善は蒸留天井を破るか」(066-072 の機構との違い)の直接検証。
- 方策はループ内の MCTS 方策改善(action_weights)から初めて学習される(M2 で方策を温めなかったのはこの切り分けのため)。M2 の弱さの主因が「乱数プライア×考慮手 16」にある以上、**方策が少しでも形になれば大きく動くはず**=感度の高い検証点。`max_num_considered_actions`(既定 16)も明示ノブとして持つ。蒸留 value の忘却対策に蒸留 replay を混ぜるなら**比率を段階的に下げる**(067 の「アンカー=天井」構図を再現しない)。
- 評価軸: **gen-vs-warmstart(本ドライバ, sims128)を主判定**、gikou@6 の score>0 化を第2目標、champion への score 推移は長期指標。1 iter ≈ 4.8 分 → 1 日 ~300 iter が予算。

## 規律・メタ
隔離環境: `m2_distill_data.py / m2_distill_train.py / m2_moves.py / m2_verify_moves.py / m2_usi.py / m2_match.py`+データ/ckpt/TSV は `~/task074_frameworks/pgx/examples/alphazero/`(非コミット)。binbo リポジトリは本 report+STATUS/BACKLOG のみ。**champion・Rust 無改変**(`cargo test` 無関係=実行不要)。venv_pgx に cshogi 1.0.2 追加(隔離内)。長時間ランは `scripts/detached.sh`+marker。**push は人間承認後。**
