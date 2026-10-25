# データ・環境の保存方針(正本) — 永続/一時の分類

「再起動(WSL の `/tmp` 消去)で失うと痛いもの」を永続領域に置き、「一瞬で再生成できるもの」だけ
一時領域に置く、という分類の正本。**判断基準は『消えたときの再取得/再生成コスト』**。

> ⚠️ Task 044 の教訓: 夜間にマシンがスリープ/再起動して **WSL の `/tmp` が全消去**された。
> 教師データ・モデルは `~/binbo-shogi-data/`(永続)にあって無事だったが、**技巧2 本体・venv・KP
> キャッシュを `/tmp` に置いていたため消失**。特に技巧2 は再ダウンロード+ビルドが必要で、教師生成が
> 止まった。→ **再取得コストの高いもの(技巧2・venv・cargo registry)は必ず永続領域へ**。

## 分類(再取得コストで決める)

| 種類 | 置き場 | git 追跡 | コスト/理由 |
|---|---|---|---|
| 教師データ(hcpe + manifest) | `~/binbo-shogi-data/teacher/` | ❌ | 技巧2 で再生成=超高コスト(1M/12分〜)。最優先で永続 |
| 学習済みモデル/重み | `~/binbo-shogi-data/models/` | ❌ | 学習で再生成可だがデータ+時間が要る。永続 |
| **技巧2(本体+param 4ファイル+ソース+zip)** | `~/binbo-shogi-data/engines/gikou/` | ❌ | **clone+make+175MB DL が必要=高コスト。Task 044 の最大教訓。永続必須** |
| **やねうら王(クローン+g++ビルド+smoke eval)** | `~/binbo-shogi-data/engines/yaneuraou/` | ❌ | clone+make が必要=中コスト。Task 056 方向転換の土台(GPL-3.0)。`eval/nn.bin`(64MB)も永続 |
| **Python venv(torch+cu130, numpy, cshogi)** | `~/binbo-shogi-data/venv/` | ❌ | torch CUDA ~2.5GB DL=高コスト。永続(旧方針「必要時再構築」は誤り) |
| **cargo registry(CARGO_HOME)** | `~/binbo-shogi-data/cargo/` | ❌ | クレート再取得を避ける。永続 |
| スクリプト(env / 起動 glue) | リポジトリ `scripts/` | ✅ | 版管理+永続。`scripts/env.sh` が正準パスの一元定義 |
| 残したい**数値・結論** | リポジトリ `reports/*.md` | ✅ | 蒸留済みで小さい。比較の正本 |
| 比較用の**生ログ** | `~/binbo-shogi-data/logs/` | ❌ | 大きく揮発的。結論は reports |
| **KP キャッシュ `.npy`** | `/tmp`(一時で可) | ❌ | 教師データから `prepare_data` で再生成=確定的・分単位(1M ~40秒, 8M ~5分)。消えても痛くない |
| ビルド中間物・使い捨て小実験 | `/tmp` | ❌ | 一瞬で作れる |

- **永続データはどの git リポジトリにも含めない**(`~/binbo-shogi-data` は repo の外。`git clean` の巻き添えも無い)。
- **`/tmp` に置いてよいのは「消えても数分で確定的に再生成できるもの」だけ**(KP キャッシュ等)。数時間の再計算や
  ネットワーク再取得を要するものは絶対に `/tmp` に置かない。

## `~/binbo-shogi-data/` のレイアウト

```
~/binbo-shogi-data/
  README.md
  teacher/run1m run1m_b run2m_c run4m_d   # 教師データ(depth8。shard+manifest+expected.tsv)
  models/model_shared_fd8_mirror_8m/      # 現行の推奨重み(BNN3 本物正準, Task 048。nnue_f32.bin 5.7MB)
  models/model_cp_fd8_mirror_8m/          # 旧推奨(naive BNN1。比較用に保持)
  models/...                              # 過去の比較用モデル
  engines/gikou/                          # 技巧2 v2.0.2: gikou(bin) + params/book/probability/progress.bin
                                          # (10/21 Xubuntu で取り直し。g++ 13 では softmax の未定義動作で落ちるので 1 行修正してビルド, reports/task158_gikou_full.md)
  engines/gikou-src/                      # 技巧2 ソース(make release で再ビルド可)
  engines/gikou2_win.zip                  # 配布 zip(param 再抽出用に保持)
  engines/yaneuraou/                      # やねうら王(Task 056): clone(upstream) + source/YaneuraOu-by-gcc + eval/nn.bin(smoke)
  engines/dlshogi/                        # dlshogi model-dr2_exhi.onnx(私的使用のみ・封印, Task 054)
  venv/                                   # Python venv(torch 2.12.1+cu130, numpy, cshogi)
  cargo/                                  # CARGO_HOME(クレートキャッシュ)
  logs/                                   # 生ログ(任意・非追跡)
```

## 正準パス(スクリプトはここを参照。`/tmp` をハードコードしない)

`scripts/env.sh` が一元定義。新しい生成/学習スクリプトは必ずこれを `source` するか同じパスを使う。

| 変数 | パス |
|---|---|
| `GIKOU` | `~/binbo-shogi-data/engines/gikou/gikou` |
| `VENV` (python) | `~/binbo-shogi-data/venv/bin/python` |
| `DATA` | `~/binbo-shogi-data/teacher` |
| `MODELS` | `~/binbo-shogi-data/models` |
| `CARGO_HOME` | `~/binbo-shogi-data/cargo` |
| `BINBO` | `<repo>/target/release/binbo-shogi` |

- ShogiHome に渡す重み = `~/binbo-shogi-data/models/<推奨モデル>/nnue_f32.bin`(現行は **`model_shared_fd8_mirror_8m`**, BNN3 本物正準, Task 048)。`EvalFile` に絶対パス。
- `reports/task0NN_*.md` は実施当時の記録なので旧 `/tmp` 表記のまま残す(履歴)。**現在の場所の正本はこのファイル**。

## 万一失ったときの再取得/再生成(コスト順)

1. **技巧2**(高): `git clone --branch v2.0.2 https://github.com/gikou-official/Gikou` → `make release`(g++ で `bin/release` 生成、~数分)。
   param は `gikou2_win.zip`(v2.0.2 リリース, 175MB)から `params.bin book.bin probability.bin progress.bin` を `engines/gikou/` へ展開。
   **検証**: `id name Gikou 2 (v2.0.2)` と、既存教師データの高 |eval| 局面で再評価し符号一致(同一エンジン=同一 param の証拠)。
2. **venv**(中): `python3 -m venv ~/binbo-shogi-data/venv && venv/bin/pip install torch==2.12.1 numpy cshogi`(torch CUDA ~2.5GB)。`torch.cuda.is_available()`=True を確認。
3. **教師データ**(超高): `gen-teacher`(Task 016 基盤、技巧2 が要る)。最優先で保持(失わない)。
4. **KP キャッシュ**(低): `python -m trainer.prepare_data <teacher_run> <cache>`(確定的、分単位)。一時で可。
5. **重み**(中): `trainer/train_nnue.py`(seed 固定で決定的)→ `trainer/export_nnue.py`。
6. **やねうら王**(中, Task 056): `git clone https://github.com/yaneurao/YaneuraOu engines/yaneuraou` → `cd source && make -j normal COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVX2`(g++ 11.4 でビルド可、`source/YaneuraOu-by-gcc` 生成)。
   smoke eval: tanuki- Háo `halfkp_256x2-32-32` の `nn.bin`(GPL-3.0)を `eval/nn.bin` に展開。**検証**: `id name YaneuraOu NNUE ... 64AVX2`、`isready`→`readyok`(eval ロード)、`go depth 18`→ `bestmove`+`score cp`、mate-in-1 で `score mate 1`。詳細 `reports/task056_yaneuraou_pivot.md`。

## 再発防止
- 重要物を `/tmp` に新規に置かない(上表の分類に従う)。スクリプトは `scripts/env.sh` の正準パスを使う。
- マシンのスリープ/再起動で `/tmp` は消える。**長時間の無人ジョブを回す前に Windows 側のスリープ無効化を確認**(WSL 内からは制御不可=作者対応)。

## 片付けの記録
- 2026-10-22 (作者承認): 結果を報告済みの中間データを削除 (約 21.6GB, 空き 6.1GB → 28GB)。
  `work/task163/{cache,cache_compact,novel}` (付け直したラベル `novel_labels` と道具の記録 `relabel` は残す), `work/task160` (task154 と task159 から作り直せる),
  `work/task162`, `work/task146_full`, `work/task149_dryrun` (広いネットの専用ビルドのエンジンを含む, Task 149 の手順で作り直せる), `work/task144_dryrun`。
  モデル (`models/`, `eval_dirs/`) と自己対局の本体 (`work/task154`, `work/task159`) は残した。
- 2026-10-23 (作者承認): 水匠5 の生徒の学習用データ (`caches/public_suisho5_train64m` 19GB と `_compact` 9.3GB, 特徴の番号を展開したもの) を削除 (空き 8.2GB → 36GB)。
  元データ `public/nodchip_suisho5_depth9/train_first64m.bin` (2.56GB, PackedSfenValue 6,395 万局面) は残したので, Task 144 の道具 (静止局面への置き換え → 学習用へ変換 → compact) で作り直せる (ダウンロード不要, CPU で 1 時間前後の見込み)。
  軽い仕上げに使う `caches/public_suisho5_val7m` と compact は残した。
- 2026-10-24 (作者承認, 土台を水匠5 の生徒にする決定のあと): 技巧2 の系統の学習用の素材を **Xeon の HDD へ退避** (消していない, 約 46GB, 手元の空き 22GB → 65GB)。
  退避先 `<archive>/binbo-shogi-data/` (`ssh xeon-worker`, 作者が binbo 用に作った場所)。手元と同じ相対パスで置いた。
  移したもの: `caches/shared_64m_qs129/train_*` (技巧2 の局面 6,400 万 + WS-T のラベル), `caches/compact_qs129/train_*`, `work/task129/{parts,changed.npy}`, `teacher/` 全部 (技巧2 の自己対局の hcpe)。
  コピー → 両側の sha256 照合 (15,789 ファイル全部一致, 一覧 `work/xeon/archive_gikou_sha256.txt`) → 手元から削除。各ディレクトリに `ARCHIVED.txt` (戻し方と sha256)。
  **手元に残したもの**: 検証用 5 万局面 (`caches/shared_64m/val_*` と各キャッシュの val のリンク, `compact_qs129/val_*`, `work/task107/rescore_search/val_dlshogi_evals.npy`) と `meta.json` / `compact.json`, `work/task129/labels`。
  生徒の系統の学習 (Task 181/182/184 の束ね) が読むのは検証用の部分だけで, 退避後も読めることを確かめた。技巧2 の系統を学習し直すときは `ARCHIVED.txt` の rsync で戻す。
  道具 `experiments/alphazero/archive_gikou.sh` (このときの手順の記録)。
