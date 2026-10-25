# Task 172 道具と CPU 検証

2026-10-23。自前の系統をゼロから作り直す C / R / W と、それぞれの軽い仕上げ C1 / R1 / W1 を実装した。**この作業は道具の実装まで。実データ全件の監査、GPU 学習、VRAM pilot、書き出しの実エンジン照合、強さ測定は実行していない。** 実験結果を得たという意味の `audit.json` / `task172_results.json` はまだない。

A の勝敗既知率は約78%。0.6%ではない。C は A の純評価値、R は A の純評価値＋B1/B2 の勝敗3割、W は A の勝敗3割。raw を先に clamp せず `600*logit(0.7*sigmoid(raw/600)+0.3*result)` を計算し、±2000 に制限、最近接整数（ties-to-even）の int16 にする。引き分け・不明（WDL=0）のラベルは元の int16 を維持し、trainer 内で通常の clamp を行う。全腕の CLI は `--cp-result-lambda 1`。val に勝敗は混ぜない。

**実行（GPU を使える作者の環境）**

この worktree のルートで実行する。ソースを別の場所へコピーしない。既存ファイルを変更できる範囲は `trainer/train_nnue.py` だけとし、その他はすべて新規ファイルにした。STATUS は変更せず、コミットもしていない。

```bash
source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

# 先に全件の監査とラベル作成。GPU は使わない。
"$VENV_PY" -B experiments/alphazero/t172_build.py

# 実サイズの R 全体を常駐させて 2,000 更新。行数は縮小しない。
"$VENV_PY" -B experiments/alphazero/t172_run.py --pilot 2000

# pilot の VRAM を確認した後、本実行。同じコマンドで再開する。
"$VENV_PY" -B experiments/alphazero/t172_run.py
```

長時間実行前のスリープ設定は `docs/data_storage.md` に従う。pilot は本実行と別モデル・別 checkpoint。R の全入力を読み、同じ 256x2-32-32、batch 1024、Factorizer、mirror、QAT/tie、fast、Adam の状態を使う。`CUDA_PEAK allocated=... reserved=...` と、2秒間隔の `nvidia-smi` による対象プロセスの最大 MiB を記録する。後者が取得できない場合は0。`work/task172/pilot_2000.json` に結果、失敗時はモデル出力先の `last_failure.json` とログを残す。OOM 後に自動で別の学習経路に切り替えない。

標準出力先は `scripts/env.sh` から読み込む。監査・manifest・ラベル・mask・pilot は `$DATA_ROOT/work/task172/`、本学習は `$MODELS_DIR/task172_{C,R,W,C1,R1,W1}/`、ログは `$LOGS_DIR/`、最終結果は `$LOGS_DIR/task172_results.json`、書き出しは `$DATA_ROOT/eval_dirs/task172_{C1,R1,W1}/`。この sandbox では、実データ側への書き込みは `work/task172/` 以外行っていない。上記の本実行は作者の書き込み権限がある環境で行う。

`--work DIR` で監査先を変更できる。runner は `--models-root DIR` / `--logs-dir DIR` も受け取る（eval_dirs の場所は既存 export / std_frame に従う）。manifest のパスは絶対パスとして記録する。

**監査とデータの扱い**

- `t172_build.py` は A/B1/B2 をチャンク走査し、ハッシュ一致候補の全77列を bytes で照合する。衝突した別局面は除外しない。A 内の既存重複を数えて残し、B 内・B1–B2・A–B は A、B1、B2、行番号の順に最初のものを残す。同局面のラベル差も集計する。
- 各 source の val 特徴が共有 val と同一であることを確認する。val 50k と鏡像の両方について全 train 行を調べ、一致行を mask で除外する。鏡像は own 39列 / opp 38列をそれぞれ再ソートする。全件で compact の復号結果と raw の全77列も一致させる。
- `audit.json` に source stamp、raw/keep/exclude 行数、重複の組、A 内重複、val/鏡像への raw overlap と除外後0、WDL 分布、残った行の自然比率、各腕の1周の更新数を記録する。既報の B=20,112,062 行は参照値で、追加の A–B 重複や漏れがあれば減る。比率調整や再サンプリングはしない。
- `A_result30.npy`、`B1_result30.npy`、`B2_result30.npy` は新規 int16。C/R の A ラベルは元ファイルを参照する。位置キャッシュの複製は作らない。mask とラベルは元行数のままで、manifest の offset は除外前の物理行番号を表す。
- source のサイズ・mtime・先頭末尾 SHA256、ラベル/WDL/mask の全ファイル SHA256、compact の検証情報・形式・行数を照合する。manifest は shared HalfKP の77列・escapes=0 に限定する。複数 val を GPU に積まない。
- 監査成功の最後に `audit.json` を作る。ラベル等は排他的に公開し、既存ファイルは上書きしない。完了済み build は manifest と stamp を再検証して再利用する。途中で落ちて `.building-*` や一部成果物だけ残った build は受け付けない。その場合は元データを残し、新しい `--work` で作る。

B1 は task142、B2 は task154 による自己対局。その eval は水匠5 の公開データによる軽い仕上げを含む。R の土台にもその知識が間接的に入る。ラベルの尺度補正は追加しない。

**trainer と更新数**

```bash
# 任意の検証済み manifest を単独で使う例
"$VENV_PY" -B -m trainer.train_nnue \
  "$DATA_ROOT/caches/shared_64m_qs129" \
  --manifest "$DATA_ROOT/work/task172/R.json" \
  --out "$MODELS_DIR/task172_manual_example" \
  --qat --qat-tie --factorize --feature-dim 256 --hidden-dim 32 \
  --batch-size 1024 --seed 20260621 --augment-mirror \
  --mirror-cache "$DATA_ROOT/work/task172/mirror.npy" \
  --fast --lr 6e-5 --epochs 12 --max-steps 749424 \
  --cp-result-lambda 1 --checkpoint-minutes 10 --resume
```

`--manifest` は `--compact-cache` / `--eval-npy` / `--limit-train` と排他。位置引数は共有 val の cache。kings/codes は結合後の領域を GPU に1回だけ確保し、各 part を小分けで直接コピーする。全 GPU 入力の cat による二重保持はしない。全行を同じ shuffle 空間で扱い、mask はその行番号に対応する。all-true mask は従来の乱数消費を変えない。

`--max-steps N` は optimizer 更新の累計上限。epoch の途中でも validation と `nnue_final.pt` を保存する。必要な epoch 数は runner が除外後の行数から計算する。checkpoint に累計更新数・manifest 識別情報・引数を保存し、同じ条件で再開する。上限を引き上げて途中から追加学習する用途は受け付けない。上限未指定・manifest 未使用の既存経路は維持した。

C/R/W は全てゼロから lr 6e-5、seed 20260621、749,424 更新、最終重み。C1/R1/W1 は各土台の `nnue_final.pt` から、公開 val 7,240,673 行、lr 1e-5、2周、最終重み。best による周選択はしない。

既存 `task135_student_fast` は、履歴・model_config・学習ログでは指定レシピと整合する。一方、当時の seed と入力 stamp を結ぶ実行時 provenance がないため、**現状の runner は C を再学習する**。歴史的な入力同一性を現在のファイルから推定して再利用しない。`verify_control()` は元の実行時に記録した provenance がある場合だけ再利用を許す。現在の入力から後付けで証拠ファイルを作らないこと。

**再開・書き出し・判定**

runner はコマンド、コード SHA256、manifest、初期重み等を `stage.json` に固定する。成功した最終重みと履歴の stamp を `done.json` に記録し、完成済みだけ skip。途中 stage は同じコマンドの `train_state.pt` から再開する。履歴ファイルが存在するだけでは skip しない。コードや入力が変わっていれば停止するので、新しい出力先を使う。`--work` ごとのロックで runner の二重起動を防ぐ。

仕上げ3腕を `t147_self_relabel_train.export()` で書き出し、0cp GREEN を確認してから `std_frame.measure_sets()` に進む。export も入力・出力・使用エンジンの stamp を保存する。完了 marker のない書き出しは nn.bin と gate ログを作り直し、途中ファイルを成功扱いしない。s280/s281/s282 で R1−C1、W1−C1、各腕−task154_R_result30 を出す。best-arm は仕上げ3腕の点推定最大。採用比較の CI 下限>0なら `spaced_set(20295000)` / `conf172` で routine 2本と micro を測り直す。各物差しの点推定>0、失敗0で `ADOPT_CANDIDATE`。自動で採用 eval を切り替えない。各段階で `T172_*` を出し、最終結果 JSON に比較・訓練・監査の識別情報を記録する。

**CPU で確認したこと**

```bash
source scripts/env.sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$VENV_PY" -B experiments/alphazero/t172_test.py
```

人工データは各 cache 5〜6行、結合16行、val 2行。trainer の幅は2、GPU は使わない。テスト内容：

- 現 HEAD の旧 trainer と新 trainer を通常 / fast の両方で実行し、`nnue_final.pt`、`nnue_best.pt`、model_config をバイト一致。履歴も所要時間以外一致。
- 3 part の境界を往復する非連続 batch、重複した row 指定、全77列、ラベル、WDL、False を含む mask を照合。小さな実体結合 cache との学習最終重みもバイト一致。
- C の all-true manifest と単一 cache の最終重みがバイト一致。更新上限10（2周目の途中）で停止。更新3の後に中断した再開、上限到達直後・validation 前に中断した再開、完了後の再実行が同じ最終重み。
- 全ハッシュを意図的に衝突させ、A 内重複の保持、A優先、B内/B間重複の除外、val/鏡像の除外を独立の期待 mask と一致させる。混合式は math によるスカラー基準（±32000を含む）と比較。鏡像は20局面を盤面の左右反転から作り直す経路と照合。
- offset、行数、source/compact stamp、ラベル変更を拒否。runner の完了 marker による skip と改変拒否、pilot/full の生成コマンド、s280/s281/s282・新しい確認組・micro への分岐は人工プロセス / mock で確認。mock の Elo は実験結果ではない。

本番サイズの監査は未実施なので、上記の人工 audit を実データの `work/task172/audit.json` として置かない。CPU テスト9件が成功した。検証ログは `work/task172/tools_cpu_tests.log` に保存する。

**未確認事項・費用**

- 16GB GPU の実際のピークは pilot が必要。R の compact 約13.11GB（12.21GiB）に target、モデル、Adam、作業領域が加わる。pilot では全 R データを保持するため、小さいデータの pilot より現実の判定になる。
- 監査の特徴・compact 読み出しは全件。hash と安定 sort の配列だけで約1.3GB、scratch・mask・mmap のページも必要。新しい巨大 cache は作らず、恒久追加は主に混合ラベル約168MBと mask 約84MB。監査用 hash 約673MBは成功時削除。
- C の再学習分を含む学習時間は GPU 未測定。再開時のコード・入力変更は受け付けない。既存 TSV の再利用は std_frame の既存規則に従う。
- 勝率の較正は A/B で同一と証明していない。R は B の局面・評価値・勝敗を一緒に導入し、A の提示回数も減る。R1−C1 はそれらをまとめた効果、W1−C1 は A の勝敗混合の効果。学習 seed は1つで、対局 CI は学習の揺れを含まない。3腕の最大値を選ぶ採用比較は探索的なので、新しい組と micro の確認を必須にした。
