# Task 044: 環境復旧(技巧2・venv)+ 永続/一時の分類確立

## 結論

- **インシデント**: 夜間にマシンがスリープ/再起動し、**WSL の `/tmp` が全消去**。`/tmp` に置いていた
  **技巧2 本体・param・venv・KP キャッシュ・CARGO_HOME が消失**。`~/binbo-shogi-data/`(永続)の
  **教師データ・モデル・repo は無事**。失ったのは環境と一晩の時間のみ(成果は無傷)。
- **パート1(規律)**: 「再取得/再生成コスト」で永続/一時を分類し `docs/data_storage.md` に正本化。
  正準パスを `scripts/env.sh` に一元化し、`AGENTS.md` に規律を追記(重要物を `/tmp` に置かない)。
- **パート2(復旧)**: 技巧2 v2.0.2 を配布元から再ビルド+param 再取得して**永続領域に固定**、venv 再作成、
  CARGO_HOME 永続化。**全て検証 green**(下記)。教師生成パイプラインは小規模で動作確認済み。
- 再発防止: 重要物は永続領域、スクリプトは `scripts/env.sh` 参照。**マシンのスリープ無効化は作者対応**(直接原因)。

## 何が消えて何が残ったか

| 消失(`/tmp`、再起動でワイプ) | 無事(`~/binbo-shogi-data` 永続 / repo) |
|---|---|
| 技巧2 本体 `release` + param 4ファイル + ソース | 教師データ run1m/_b, run2m_c, run4m_d(depth8, 1.6GB) |
| venv(CPU/GPU torch, cshogi) | モデル(1.1GB, `model_cp_fd8_mirror_8m` 等) |
| KP キャッシュ `.npy`、CARGO_HOME | repo + `target/release/binbo-shogi`(バイナリ無事)、git(Task 042 まで) |

原因: `uptime` が再開時 3 分 = マシンが ~09:01 に起動(夜間に落ちた)。Task 043 で警告した
「マシンが寝ると止まる」リスクが現実化。加えてセッションは ~00:58(夜間生成の起動直前)に切断 →
**夜間 depth12 生成は一度も起動されず**(教師データはゼロ生成)。

## パート1: 永続/一時の分類(`docs/data_storage.md` に正本化)

- 判断基準=**消えたときの再取得/再生成コスト**。高コスト=永続、一瞬=一時。
- 永続(`~/binbo-shogi-data/`): 教師データ・モデル・**技巧2(`engines/gikou/`)・venv・CARGO_HOME(`cargo/`)**。
- 一時(`/tmp`): KP キャッシュ `.npy`(教師データから `prepare_data` で確定的・分単位で再生成。1M ~40秒, 8M ~5分)、ビルド中間物。
- 正準パス: `scripts/env.sh`(`GIKOU` / `VENV_PY` / `TEACHER_DIR` / `MODELS_DIR` / `CARGO_HOME` / `BINBO`)。
  既存の `run_d12_a.overnight.sh` を `/tmp` ハードコードから `source scripts/env.sh` に修正。`AGENTS.md` に規律を追記。

## パート2: 復旧と検証

### 技巧2 v2.0.2(永続 `~/binbo-shogi-data/engines/gikou/`)
- ソース: `git clone --branch v2.0.2 gikou-official/Gikou` → `make release`(g++ 11.4)→ `bin/release`(**967752 bytes = 旧本体と同一サイズ**)。
- param: v2.0.2 リリース `gikou2_win.zip`(175MB)から `params.bin(259MB)/book.bin/probability.bin/progress.bin` を抽出。zip も永続保持(再抽出用)。
- **検証(同一エンジンの証拠)**:
  - USI: `id name Gikou 2 (v2.0.2)`。
  - eval 同一性: 既存 run4m_d の**高 |eval|(≥1500)局面 15個を depth8 で再評価 → 符号 15/15 一致**、複数が**完全一致**
    (−2069/−2069, 2256/2256, −5375/−5375, 30500/30500, 6087/6087)。決定的エンジン=正しい param。
  - ※ `gikou-recheck` は 60標本で decisive 28/29・status=VIEWPOINT_MISMATCH だったが、これは**±500 境界の冷間再評価ノイズ**
    (Task 016 既知: 冷間単発評価は生成時の暖機状態と差が出て境界で稀に符号が揺れる)。高 |eval| 15/15 一致が決定的に正しさを示す。

### venv(永続 `~/binbo-shogi-data/venv/`)
- `python3 -m venv` + `pip install torch==2.12.1 numpy cshogi`。**torch 2.12.1+cu130 / cuda True / RTX 4060 Ti / cshogi ok / numpy 2.2.6**(Python 3.10.12)。

### CARGO_HOME(永続 `~/binbo-shogi-data/cargo/`)
- `/tmp` の cargo cache 消失に対応。`CARGO_HOME=~/binbo-shogi-data/cargo` で `cargo test --locked` がクレート取得+ビルド成功(以後永続キャッシュ)。

### KP キャッシュ
- 一時(`/tmp`)で可と決定(`prepare_data` で確定的・分単位再生成)。8M でも ~5分なので永続不要。必要時に再生成。

## 回帰(全 green)
- `cargo test`(CARGO_HOME 永続): **16/16**。
- `verify`(perft 1〜5): 完全一致(生存バイナリ)。
- **小規模 depth8 生成**(復元 gikou, 5000局面): status=ok・crash0・0.47分、cshogi 読み戻し ok・eval ±30500 = **パイプライン健全**。
- **モデル整合**: `verify_nnue_rust`(`model_cp_fd8_mirror_8m`, 新 venv)= max_abs_err **3.275e-06**, ok = 生存モデル+バイナリ+venv が一致(データ・モデル無傷)。

## 付録: Task 043 で得た唯一のデータ(depth 方針の次段判断材料)

セッション切断前に depth12 のコスト実測だけは取れていた(同条件 120k局面, 12並列, shard1万):

| depth | avg_ms/局面 | depth8 比 | crash/12万 | parallel_speedup | 備考 |
|---|---:|---:|---:|---:|---|
| 8 | 0.7336 | 1× | 7 | 5.55 | 既存実績と一致 |
| 12 | **16.399** | **~22×** | **1218** | **0.25** | クラッシュ激増で**直列見積より遅い**=バルク生成に非現実的 |

→ depth12 は ~22倍遅 + クラッシュ地獄で、6時間でも ~1.2M + 不安定。**depth 方針(depth10 実測 or 質の軸見送り)は復旧後の別段で判断**(本タスクのスコープ外)。

## 成果物
- コミット: `docs/data_storage.md`(分類正本)、`scripts/env.sh`(正準パス)、`AGENTS.md`(規律)、`STATUS.md`/`docs/BACKLOG.md`、本レポート。
- 非コミット(永続データ): `~/binbo-shogi-data/engines/gikou/`(技巧2)、`~/binbo-shogi-data/venv/`、`~/binbo-shogi-data/cargo/`。
