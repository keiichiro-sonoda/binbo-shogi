# Task 074 — (c) AlphaZero フレームワーク選定: Pgx / MiniZero / kZero の実測比較

**目的**: 073 で確定した (c)(単機の自己対局 RL が正 elo を生むか実証)の土台フレームワークを3候補から選ぶ。**環境構築 + 付属最小サンプルの起動確認まで**を実測(フル学習は組まない)。最終選定は Keiichiro、本レポートは推奨+根拠まで。**表記**: 【実測】と【推測】を分離。作業は `~/task074_frameworks/`(repo 外・非コミット)、既存 binbo リポジトリ/champion 不触。

## 実測環境(2026-07-01)
GPU **RTX 4060 Ti 16GB**(サンドボックス内で `nvidia-smi -L` 可視)/ `torch 2.12.1+cu130` CUDA True / **CUDA toolkit 12.4**(`/usr/local/cuda-12.4`, nvcc あり)/ **system cuDNN dev(headers)は無し**(pip の nvidia-cudnn が別 venv 内にあるのみ)/ `cargo 1.96.0` / network 到達可(github/pypi 200)/ **Docker は distro 内で不可**(Docker Desktop Linux engine 未起動: `docker.exe info` → `dockerDesktopLinuxEngine: cannot find the file`)。

## 比較表(3候補 × 7軸、実測 ○△×)
| 軸 | **Pgx (JAX)** | MiniZero (C++/Docker) | kZero (Rust+Py) |
|---|---|---|---|
| **1. 環境構築** | ○ 【実測】`pip install "jax[cuda12]" pgx mctx` がクリーン成功(CUDA/cuDNN を pip が同梱=system 不要)。**GPU 認識**(`jax.devices()=[CudaDevice(id=0)]`)、AZ 内ループ実走。 | × ここでは【実測】**Docker 不可**(Desktop Linux engine 未起動=host 対応要)。非 Docker 手動再現は Dockerfile 参照で可だが C++ ビルド負担。 | × 【実測】Kyanite `kn-cuda-sys 0.7.3` の build が **bindgen で panic**(`cudnn` リンク要も system cuDNN dev headers 無し)。Rust toolchain・非CUDA crate(kz-util)は成功。 |
| **2. 将棋/どうぶつ将棋** | ○ 【実測】`animal_shogi`(4×3, 132 手)+ `shogi` を**標準搭載**、`pgx.make("animal_shogi")` で batched init/step 実走。 | × 【実測】環境は go/gomoku/othello/atari 等 22種、**shogi/どうぶつ将棋なし**→ C++ env 実装が要。 | × 【実測】`board_game` crate は chess/ataxx 等、**shogi なし**→ Rust で盤面+入出力 mapping 実装が要。 |
| **3. スタック親和性** | △ Python だが **JAX 新規習得**。既存 SPRT(USI)接続は「学習済ネット→USI エンジン」の glue が要(M2+)。 | △ C++/CMake、**GTP** インタフェース(USI でない)。config 駆動。shogi は C++ 実装要。 | ○寄り Rust+PyTorch で Keiichiro スタックに最近。ただし推論は **Kyanite(自作 CUDA lib、PyTorch でない)**=別習得。 |
| **4. 効率化(少sim)** | ○ 【実測】**mctx = Gumbel AlphaZero** 同梱(`gumbel_muzero_policy` 実走)。 | ○ **Gumbel AZ/MuZero + progressive simulation**(論文の効率化, batch GPU 推論)。 | △ AZ/MuZero 実装。Gumbel は不明確(要確認)。 |
| **5. 小資源実績** | ○ Pgx 論文=単一 A100 で AZ、GPU ベクトル化 10-100×。 | ○ IEEE ToG 論文=小規模 AZ/MuZero 比較、GTX 1080 8GB min。 | ○【推測寄り】作者はチェス/囲碁を 2×GPU で(一次確認は未)。 |
| **6. M0 適性** | ○ 【実測】`animal_shogi` + **baseline model `animal_shogi_v0` 取得可** + `examples/alphazero/train.py`(mctx)即用。 | △ 内蔵ゲームなら Docker 前提で容易だが、shogi/どうぶつは C++ env 追加要。 | × shogi 追加 + Kyanite 稼働が前提=遠い。 |
| **7. ライセンス** | ○ **Apache-2.0**(pgx リポLICENSE / mctx=DeepMind Apache-2.0)。permissive=大会派生に好適、封印重みなし。 | ? **repo に LICENSE ファイルなし**(要確認)。学術リポ。 | × **LICENSE なし**(= all rights reserved 既定)=派生・大会利用の法的ブロッカー。 |

## 各候補の詰まりどころ(実測)
- **kZero**: `cd rust && CUDA_PATH=/usr/local/cuda-12.4 cargo build -p kz-core` → `kn-cuda-sys 0.7.3` build.rs が bindgen(0.68.1)で panic。`cudnn` を要するが system cuDNN dev(headers)不在。回避には cuDNN dev(CUDA 12.4 対応版)の system インストール+版合わせが要(=host 作業/版ガチャ、本タスクでは**強行せず記録**)。加えて **無ライセンス**と **shogi 無**が本質的に不利。
- **MiniZero**: Docker 必須だが Docker Desktop の Linux engine が本環境で未起動(`docker.exe info` 失敗)。**Docker 有効化=host 判断**。非 Docker 手動ビルド(Dockerfile 追随)は可能だが C++ 依存の再現負担。**shogi 無**(C++ env 追加要)、**LICENSE 不在**も要確認。
- **Pgx**: **詰まりなし**。唯一の注意点=JAX 既定の VRAM 75% 事前確保で `CUDA_ERROR_OUT_OF_MEMORY` の noise が出るが実行は成功(fallback)。**`XLA_PYTHON_CLIENT_PREALLOCATE=false`**(or `XLA_PYTHON_CLIENT_MEM_FRACTION`)で解消。

## 総合推奨 — **Pgx(+ mctx)**(最終決定は Keiichiro)
**根拠**: 3候補で唯一、**本環境で今すぐ**「GPU 認識 + どうぶつ将棋で Gumbel AlphaZero 内ループ実走」を【実測】達成。**Apache-2.0**(大会派生に最も安全、封印重みなし)、**どうぶつ将棋+将棋を標準搭載**、**mctx=Gumbel AZ**(単機で効く少sim)、**baseline model で棋力評価も即可**。MiniZero(効率化・実績は最良だが Docker 不可 here + shogi 無 + LICENSE 不明)、kZero(無ライセンス + CUDA build 不可 + shogi 無)は本プロジェクトの制約(単機・大会・在庫環境)で不利。
- **主コスト/リスク**: (1) **JAX 習得**(numpy 風だが関数型/jit/pmap の作法)。(2) **USI 連携**: Pgx-学習ネット→USI エンジン化の glue が要(既存 SPRT ハーネスは USI 前提)。ただし M0/M1 段階では Pgx 内の baseline 対戦で評価でき、USI 化は M2 以降。(3) **フル将棋**は状態空間大=まず**どうぶつ将棋で M0**が定石。

## 推奨での第1マイルストーン(M0)具体手順案
既存の隔離 venv(`~/task074_frameworks/venv_pgx`, jax[cuda12]+pgx+mctx 導入済)を利用。
1. `examples/alphazero/train.py` を **`env_id=animal_shogi`** + 小 config(`selfplay_batch_size` 縮小、`num_simulations` 8–32、数 iteration)、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`WANDB_MODE=disabled` で起動。
2. **合格条件**: 自己対局→学習→**`animal_shogi_v0` baseline への勝率が数 iter で上昇**(=単機で AZ ループが回り強くなる)。VRAM/スループット/1 iter 時間を実測記録。
3. 続く M1=どうぶつ将棋(→将棋)の自己対局スループット(state/s)を 4060 Ti で実測 → 1サイクル現実コスト見積。M2=warm-start ネットで少sim MCTS が αβ champion 超えか(USI glue + SPRT)。M3=short RL で gen1>warm-start(073 のマイルストーン)。

## ライセンス/大会 要確認(軸7・`docs/USED_COMPONENTS.md` の論点)
- Pgx/mctx = **Apache-2.0**(コード借用は表示義務のみ、コピーレフト伝播なし)。→ 学習は**自前ゼロ知識/自前教師**、封印重みは使わない前提を維持すれば YaneuraOu 同様に合法ルート。
- **要確認**: (1) Pgx 同梱の baseline model(例 `animal_shogi_v0`)の**重みライセンス**(Google Drive 配布)= 大会には使わない/監視・sanity 用に留めるか。(2) mctx の Apache 表示義務(NOTICE 同梱)。(3) 将来 dlshogi の入出力設計(104ch/81×27)を**参考にする**場合の線引き(コード借用=GPL 表示 vs 単なる設計思想の踏襲)。(4) 電竜戦=事前生成教師→事前学習自作ネットの合法ルート維持。結論は次段で。

## 規律・メタ
実装は最小限(install + 最小サンプル起動のみ、フル学習・shogi 実装・長時間ランなし)。作業物は `~/task074_frameworks/`(**非コミット**、binbo リポジトリに実測ログを貼らず要約のみ)。**Docker 有効化・cuDNN system 導入等の host 作業は未実施(=Keiichiro 判断)**。**既存 binbo リポジトリ/champion 不触・Rust 無改変**(`cargo test` 無関係)。web 一次情報: [pgx](https://github.com/sotetsuk/pgx)(Apache-2.0)/[mctx](https://github.com/google-deepmind/mctx)/[minizero](https://github.com/rlglab/minizero)/[kZero](https://github.com/KarelPeeters/kZero)。バージョン: jax 0.10.2 / pgx 2.6.0 / mctx 0.0.71 / kn-cuda-sys 0.7.3 / CUDA 12.4。**push は人間承認後。**
