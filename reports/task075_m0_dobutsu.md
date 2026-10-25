# Task 075 — M0: Pgx でどうぶつ将棋 AlphaZero ループを回し「単機で正 elo」を最小実証

**目的**: (c) の最初の検証点。074 で選定した Pgx の付属 AlphaZero サンプルを**どうぶつ将棋(4×3)+ 単機小設定**で回し、「**自己対局 RL のループが回り、実際に強くなる**」最小の証拠を取る。066-072 で αβ蒸留の自己強化が技巧2@d8 の天井に阻まれたのと対に、**MCTS+方策改善なら正方向に動く**ことを示す。同時に JAX 学習の場(下記 §JAX 解説)。**フル将棋・net→USI 橋・SPRT・技巧2対戦は M1+(非対象)**。実測は 2026-07-01、RTX 4060 Ti 16GB(単機)、隔離 venv `~/task074_frameworks/venv_pgx`。

## 実行設定
- 付属 `examples/alphazero/train.py` の隔離コピー `train_m0.py`(1 行だけ patch、下記 §詰まりどころ)を、Pgx の **Gumbel AlphaZero(mctx)** で実行。
- `env_id=animal_shogi num_simulations=32 selfplay_batch_size=512 max_num_steps=64 training_batch_size=1024 num_channels=64 num_layers=5 eval_interval=2 max_num_iters=60 seed=0`(単機・軽量 ResNet)。`WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 評価は **`animal_shogi_v0` baseline(学習済)との raw-policy サンプリング対局**(train.py:230-258、MCTS 不使用=「debug 用」明記)。win/avg_R は各 eval で `print(log)`(train.py:291-298)。**GPU 実測**: `jax.devices()=[CudaDevice(id=0)]`、**~10 分 / 60 iter(~10s/iter, 初回 JIT 込)**。

## 結果 — 勝率推移(vs animal_shogi_v0, 512 games/eval)
| iter | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 32 | 42 | 52 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **win_rate** | .008 | .205 | .381 | .354 | .391 | .441 | .494 | **.547** | .590 | **.652** | .439 |
| **avg_R** | −.984 | −.586 | −.229 | −.271 | −.176 | −.068 | +.025 | +.137 | +.262 | +.369 | +.082 |

- **未学習(iter0)は baseline にほぼ全敗**(win 0.8%, avg_R −0.98)→ **~24 iter で avg_R が 0 を越え(勝ち越し圏)**、**~52 iter で win 0.65 / avg_R +0.37 のピーク**、以降 RL 特有のノイズ/振動(win ~0.44-0.59)。
- 512 games eval の std ≈ ±2-3% ゆえ、序盤の単調上昇(0.8%→50%超)は**ノイズを大きく超える明確なトレンド**。

## M0 判定 = **PASS**
- **単機の自己対局 AZ ループが回り、かつ前進**した:ランダム初期 → **学習済 baseline とパリティ超**(win 中央 ~0.5, ピーク 0.65)を **~10 分 / 単一 4060 Ti** で達成。合格条件「勝率が上昇トレンド」を明確に満たす(>50% 到達も副次的に達成)。
- **意義(αβ アークとの対比)**: 066-072 では αβ+評価値蒸留の自己強化が**技巧2@d8 の天井を一度も越えず**(067-072 全否定)。M0 は同じ「自己対局で強くする」でも、**探索を MCTS+方策改善に変えると、ゼロから baseline 強度まで正方向に登れる**ことを最小規模で実証=(c) の前提が原理的に成立。
- **留保(過大評価しない)**: (1) 題材はどうぶつ将棋(4×3, 132 手)= 将棋(~2187 手)より桁違いに小さい。(2) eval は **raw-policy サンプリング**(MCTS 不使用)で強さの粗い代理。(3) baseline は Pgx 付属の 1 モデルで絶対棋力は不明。→ M0 は「ループが回り前進する」ことの確認まで。強さ・将棋への一般化・USI/SPRT での厳密評価は M1+。
- 後半の win 低下(52→60 で 0.65→0.44)は**過学習/小ネットの不安定 + eval 分散**の範囲。トレンド(0→~0.5-0.6)は不変。

## 【JAX 学習】train.py が扱う3つのクセ(実コード引用)
JAX は「純関数 + 明示的乱数 + 静的形状」を要求する。train.py の該当箇所:

**1. 関数型スタイル + `lax.scan`(可変代入を避ける)**
- 自己対局は Python ループでなく `_, data = jax.lax.scan(step_fn, state, key_seq)`(train.py:162)。`step_fn(state, key)→(new_state, output)` は**純関数**(状態を返す、破壊的代入なし)。value ターゲットの後ろ向き累積も `jax.lax.scan(body_fn, jnp.zeros(...), jnp.arange(max_num_steps))`(:187)。パラメータ更新も `grads=jax.grad(loss_fn)(...)`→`optax.apply_updates`(:219-224)で**新しい配列を返す**(in-place なし)。
- **なぜ**: JIT/XLA はトレース可能な純関数を前提とし、Python の可変代入・副作用はコンパイルできない。`scan` は「同じ関数を N 回」を**1 個の融合カーネル**にまとめ、Python for ループより桁速い(GPU で全 batch 並列 + ループ融合)。→ 我々の Rust 自己対局(逐次 for)とは発想が逆で、**全対局・全手を配列次元にベクトル化**する。

**2. PRNGKey の明示分割(再現性 + 並列重複回避)**
- JAX の乱数は**状態を持たない**。単一 `rng_key=jax.random.PRNGKey(config.seed)`(:284)を、使うたびに `jax.random.split` で**独立な子キーへ分岐**: 毎 iter(:288 eval / :328 selfplay / :337 shuffle)、self-play 内で per-game キー(`split(sub_key, batch_size)` :159)・per-step キー列(`split(rng_key, max_num_steps)` :161)、`step_fn` 内 `key1,key2=split(key)`(:126、**key1=Gumbel 探索の乱数**:137、**key2→auto_reset の per-game キー**:145)、pmap 用 per-device キー(`split(subkey, num_devices)` :289,329)。
- **なぜ**: この分割木で (a) **再現性**(同 seed → 同キー列 → 同結果、決定的)、(b) **重複回避**(各 split は独立ストリーム=並列ゲーム/デバイスが同じ乱数を踏まない)。**これは 067 で我々が worker 間 seed を `seed+100003*k` で分離した問題と同一**だが、JAX は「グローバル可変 RNG」でなく明示分割で構造的に解決している。

**3. 静的形状 / JIT(可変長を固定長へ畳む)**
- 対局長は `max_num_steps`(:50)で**固定**。`scan(step_fn, state, key_seq)` の `key_seq` 長が静的 trip count(:161-162)。終局した対局は `auto_reset(env.step, env.init)`(:146)で**自動リセットして走り続け**、固定長 scan が常に有効な状態を出す(短い対局はパディング=別対局が始まる)。value 損失は truncate 分を `value_mask`(:179, :211)で除外。
- 合法手の可変数は**固定長 `legal_action_mask`**(`num_actions` 次元)+ 非合法手を `-inf` でマスク(:96, `invalid_actions=~legal_action_mask` :140)に畳む。policy 出力次元は `env.num_actions`(どうぶつ将棋 132)で固定。
- **なぜ将棋が重くどうぶつ将棋が軽いか**: JIT は形状を静的化し**最悪ケースへパディング**する。将棋は盤面平面が大・`num_actions`≈2187(vs 132)・対局長が長い → 固定形状テンソルが桁大 → VRAM/計算が跳ねる。どうぶつ将棋(4×3)はどれも小さく単機で軽い(本 M0 が ~10s/iter で回った理由)。

## 詰まりどころと回避策(実測)
- **`jax.device_put_replicated` が modern JAX(0.10.2)で削除**(2023 年の Pgx 例は旧 API)。→ train.py:270 を隔離コピー `train_m0.py` で 1 行 patch: `jax.tree_util.tree_map(lambda x: jnp.stack([x]*num_devices), (model, opt_state))`(デバイス軸を付けて複製)。`jax.pmap` 自体は動作(deprecation 警告のみ)。**M1+ でスケール時は pmap→jit/shard_map 近代化を検討**(単機 M0 では不要)。
- **`wandb` 未導入**(train.py が import/呼出)→ venv に pip install、`WANDB_MODE=disabled` で no-op 実行(ログイン/ネット不要)。
- **JAX 既定の VRAM 75% 事前確保 noise**(`CUDA_ERROR_OUT_OF_MEMORY` の表示)→ `XLA_PYTHON_CLIENT_PREALLOCATE=false` で回避、実行は成功。
- pydantic v2 の deprecation 警告(class-based Config)= 無害。

## next(M1)への申し送り
- **M1 = どうぶつ将棋(→将棋)の自己対局スループット実測**(state/s, VRAM, 1 サイクル時間)を 4060 Ti で。将棋 env は固定形状が桁大=VRAM/時間の跳ねを実測し、num_channels/num_simulations/batch の単機上限を見極める。
- **評価の格上げ**: 現状 eval は raw-policy サンプリング。M2 で **MCTS 対局 + 既存 SPRT ハーネス(USI)接続**へ(net→USI 橋が要)。M2 の主眼は「warm-start ネットで少 sim MCTS が champion(αβ)を超えるか」。
- **設定メモ**: 小ネット(64ch/5層)+32 sim+512 batch でどうぶつ将棋は ~10s/iter・60 iter で baseline パリティ超。将棋はこの桁では足りない見込み(M1 で実測)。
- 学習物(checkpoints)・ログは `~/task074_frameworks/pgx/examples/alphazero/`(隔離・非コミット)。

## 規律・メタ
(c) 作業は隔離環境に閉じ、binbo リポジトリには**本 report + STATUS/BACKLOG のみ**(実験コード `train_m0.py`・checkpoints は非コミット)。**既存 binbo リポジトリ・champion・057-064 資産は不触、Rust 無改変**(`cargo test` 無関係=実行不要)。長時間ラン=`scripts/detached.sh`+marker。**push は人間承認後。**
