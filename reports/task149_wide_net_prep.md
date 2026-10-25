# Task 149: HalfKP 512x2-8-96 の接続準備 (2026-10-20)

**ランダム重みを trainer → BNQ2 → nn.bin → 専用 YaneuraOu に通し、32/32 局面で 0cp 一致。既定 256 の採用重みは BNNQ・nn.bin とも既存ファイルと完全一致。** 学習・GPU 使用・対局による棋力判定・コミット・push は行っていない。変更はこの worktree、生成物と YaneuraOu のビルドは `$DATA_ROOT/work/task149_dryrun/` のみ。既存エンジンの source は読取のみ。

## 変更

- `trainer/nnue.py`: QAT の `hidden_dim2` を追加。省略時は従来の `hidden_dim` と同じ。FT 幅、tie map、Factorizer、sparse FT は従来の経路を使う。
- `trainer/train_nnue.py`: `--hidden-dim2` と config 保存。wide は `--feature-dim 512 --hidden-dim 8 --hidden-dim2 96`。resume 時も同じ寸法を指定する (違えば state_dict の shape 検査で拒否)。`--epochs 0` は初期検証と final checkpoint/config の保存だけで正常終了するようにした。従来は終了時の `history[-1]` で落ちていた。
- `trainer/quant_format.py` / `export_nnue.py`: 対称な隠れ層の旧 **BNNQ は同一レイアウト**。非対称層は **BNQ2** (拡張子 `.bnnq`)。未知 magic/feature id・不正尺度は拒否する。
- `yane_nnue.py` / `binbo_to_yane.py`: 対象 2 構造のヘッダ、hash、FT 幅、affine 寸法と padding を扱う。`--template` は任意。同構造の template があれば従来どおり保持し、省略時は upstream 定義から生成する。金・成駒の tie と side 行 0 を検査する。
- `yane_to_binbo.py`: wide の整数 forward と逆変換にも対応。H1=8 の次層入力を 32 に pad する。旧 float BNN3 への wide 変換は明示的に拒否。
- `qat_eval.py`: H2 と Factorizer のロード、fold → tie → quantize に対応。
- `scripts/nnue_gate.sh` / `trainer/nnue_gate.py`: `--engine` を明示する新ゲート。元 BNNQ (binbo の特徴番号) と書出し nn.bin (YaneuraOu の特徴番号) の各整数 forward を、実エンジン `e` と比較する。全件 0cp・件数一致・正常 load を要求し、違えば非ゼロ終了。従来の `binbo_to_yane.sh` も RED/件数不一致を失敗終了にし、Threads=1 にした。

Rust (`src/eval/nnue.rs`) は変更していない。**BNQ2 は Rust 推論の対象外**。wide の配備検査には新しい Python/engine ゲートを使う。特徴・tie map・77 列 compact cache は共通で、作り直す必要はない。

### BNQ2 の形式

little endian、32-byte header:

```text
magic[4] = "BNQ2"
total(u32), FT(u32), H1(u32), H2(u32), FV_SCALE(i32), shift(u32), feature_id(u32=1: HalfKP)
emb[total,FT](i16), ft_bias[FT](i16),
w0[H1,2*FT](i8), b0[H1](i32), w1[H2,H1](i8), b1[H2](i32),
w2[1,H2](i8), b2[1](i32)
```

旧 BNNQ は 24-byte header のまま、H2=H1 と解釈する。旧 Rust reader が新ファイルを黙って旧構造として読むことはない。

## YaneuraOu の一次資料とテンプレート不要の生成

upstream HEAD: `820940ee31762ad598a9156a615b4b896ab8b321`。生成された `source/eval/nnue/architectures/halfkp_512x2_8_96.h` は以下を定義する。

```text
RawFeatures = FeatureSet<HalfKP<Side::kFriend>>
FT = 512
InputSlice<1024>
ClippedReLU<AffineTransformSparseInput<InputLayer,8>>
ClippedReLU<AffineTransform<HiddenLayer1,96>>
AffineTransform<HiddenLayer2,1>
```

`nnue_common.h` の `kWeightScaleBits=6`、`nnue_feature_transformer.h` の FT weight/bias=int16、`layers/affine_transform*.h` の weight=int8 / bias=int32、`clipped_relu.h` の clamp[0,127] と >>6 を確認。出力は FV_SCALE=16 のゼロ方向整数除算。従来の QAT 尺度 (FT=127、hidden weight=64、hidden bias=127×64、output weight=16×scale_cp/127、output bias=16×scale_cp) を変更する必要はない。

ファイルの後段は出力行順で `8×1024`, `96×32` (実入力8、残り24列は0), `1×96`。FT は `125388×512`。生成した nn.bin は **128,410,325 bytes**。学習済み重みの template は不要で、`yane_nnue.halfkp_architecture` が InputSlice / Affine / ClippedReLU / FT の hash 算式と構造文字列を組み立て、writer が全重みを埋める。

```text
version  = 0x7af32f16
khash    = 0x3e5aa70c
ft_hash  = 0x5d69d1b8
net_hash = 0x633376b4
Features=HalfKP(Friend)[125388->512x2],Network=AffineTransform[1<-96](ClippedReLU[96](AffineTransform[96<-8](ClippedReLU[8](AffineTransformSparseInput[8<-1024](InputSlice[1024(0:1024)])))))
```

`tools/task149_arch_probe.cpp` を同じ生成ヘッダからコンパイルし、上記 hash 3 個と構造文字列が Python 生成と完全一致することも確認した。エンジンの読込ログに hash mismatch なし。旧 template の dense affine と生成ヘッダの SparseInput は hash/保存順が同じで、既存 256 template も引き続き使える。

## ビルドと再実行

共通設定 (以下の `DRY` は永続領域、`WIDE_MODEL` は学習結果を置く場所):

```bash
source scripts/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
DRY="$DATA_ROOT/work/task149_dryrun"
WIDE_MODEL="$DRY/random512"
WIDE_ENGINE="$DRY/source/YaneuraOu-by-gcc"
```

実行したビルドコマンド。コピー先が未作成のときに実行する。obj もコピーの親である `$DRY/obj/` にできる。

```bash
cp -a "$YANE_DIR/source" "$DRY/source"
(
  cd "$DRY/source"
  timeout 880 env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    make -j2 normal COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3 \
    YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_halfkp_512x2_8_96 \
    > ../build_wide.log 2>&1
)
g++ -std=c++17 -O1 -D_LINUX -DYANEURAOU_ENGINE_NNUE \
  '-DNNUE_ARCHITECTURE_HEADER="architectures/halfkp_512x2_8_96.h"' \
  -I "$DRY/source" tools/task149_arch_probe.cpp -o "$DRY/arch_probe"
"$DRY/arch_probe"
printf 'usi\nquit\n' | "$WIDE_ENGINE"
```

g++ は Ubuntu **13.3.0-6ubuntu2~24.04.1**。ビルドした binary:

```text
sha256: bb22a792148959b81a53956b47c97b05d02595829182e06caf29469baa55bdb4
usi: id name YaneuraOu NNUE 9.41git 64AVX2
     id author yaneurao
```

USI id は現行と同じなので、構造の識別には binary SHA と生成ヘッダを使う。

ランダムモデルと小さな CPU 検査を再生成 (今回実行した経路):

```bash
PYTHONPATH=. "$VENV_PY" experiments/alphazero/t149_wide_prep.py --work-dir "$DRY"
# 実 cache の train 16 / val 8 行と Task129 ラベルから tiny/compact fixture を生成。
# --checks-only ならランダム重みと採用重みの再exportを省略。
"$VENV_PY" -m trainer.train_nnue "$DRY/tiny_cache" --out "$DRY/fast_init" \
  --qat --qat-tie --factorize --feature-dim 512 --hidden-dim 8 --hidden-dim2 96 \
  --fast --compact-cache "$DRY/compact" --device cpu --epochs 0 --batch-size 4
```

将来の学習では、従来の学習コマンドの寸法を次のように指定する。**これは今回実行していない学習例**。データ・周数・ラベル・seed・mirror 等の実験条件は別途固定する。

```bash
"$VENV_PY" -m trainer.train_nnue "$DATA_ROOT/caches/shared_64m_qs129" \
  --out "$MODELS_DIR/task149_wide" --eval-npy "$DATA_ROOT/work/task129/labels" \
  --qat --qat-tie --factorize --feature-dim 512 --hidden-dim 8 --hidden-dim2 96 \
  --fast --compact-cache "$DATA_ROOT/caches/compact_qs129" --device cuda \
  --objective cp --scale-cp 600 --epochs 12 --lr 6e-5 --batch-size 1024
```

書出しと 0cp gate (`WIDE_MODEL` を実際の学習結果に置き換える):

```bash
"$VENV_PY" -m trainer.export_nnue "$WIDE_MODEL" \
  --weights nnue_final.pt --out nnue_final.bnnq
# 単独writer (templateなし)
"$VENV_PY" -m trainer.binbo_to_yane --from-bnnq "$WIDE_MODEL/nnue_final.bnnq" \
  --out "$DRY/nn.bin"
# writer + 元BNQ2/書出しnn.bin/実エンジンの三者照合、ログは永続保存
scripts/nnue_gate.sh --from-bnnq "$WIDE_MODEL/nnue_final.bnnq" \
  --engine "$WIDE_ENGINE" --work-dir "$DRY/gate512"
```

既定 256 は従来の `--feature-dim 256 --hidden-dim 32` のまま。旧 writer/gate の引数も維持。新ゲートでも `--engine "$YANE" --template "$YANE_EVAL/nn.bin"` で検査できる。

## 検証結果

| 検査 | 結果 |
|---|---|
| seed149、FT512/H1=8/H2=96、tie + 非ゼロ Factorizer のランダム重み | checkpoint/config、BNQ2、nn.bin を作成。学習0ステップ |
| 既存 `tools/parity_positions.sfen` 32局面 | source BNQ2 == Python nn.bin == engine、各 **32/32、max差0cp、GREEN**。出力は −210〜−52cp (全件同値でない) |
| 元checkpointの `qat_eval` 整数出力 | source BNQ2 の32行と完全一致 |
| nn.bin → BNQ2 → nn.bin | 元 nn.bin と byte 完全一致 |
| wide の dense/sparse FT | 小さい特徴表で FT512/8/96、tie/Factorizer/QAT を有効にし forward と全 parameter の勾配を照合。atol=2e-6、rtol=2e-5 内 |
| 実 cache の tiny compact | encode/decode 全行一致、CPU resident decode 一致 |
| `--fast --compact-cache` | CPU、epochs=0、初期検証と checkpoint/config 保存が正常終了。sparse_ft/fused_adam/async_loss の設定を確認 |
| 既存 Task135 CPU 回帰 | **5 tests OK** (resume/compact/sparse/async を含む) |
| 既定モデル初期化 | H2省略と明示32で state_dict 全tensor完全一致、RNG順も維持 |
| Task142 final再export | 既存 BNNQと同一SHA、書出しnn.binも採用済みと同一SHA。現行エンジンとの32局面も全件0cp |
| 異なる構造の engine を指定 | wide重みを現行256 engineへ渡すとhash/loadエラー、ゲートexit=1。誤ってGREENにならない |
| 不明な BNQ2 feature id | 拒否 |
| `git diff --check`, shell構文 | 合格 |

主な生成物の SHA-256:

```text
Task142 BNNQ 0978120c14a36ba8f4f7fa760c94e31fa8b74e93cc21cc6d2a90648869ce4c4a
Task142 nn.bin 651ff9275a2fa416f195697270a1191967349d1601950f991b9bf522b9c8cece
random512 BNQ2 5dc3c115ff686993bbc872e335c0c0b26c3d622ea5dbb3d2e50677f189951ea5
random512 nn.bin 6bf7ab669e0b1e819ab0f13578f7b4685c8604bee988a25f3d6db5350b641048
```

保存先は `$DRY/` の `prepare.log`, `cpu_checks.log`, `fast_init.log`, `task135_regression.log`, `qat_eval.log`, `gate512.log`, `gate256.log`, `gate_wrong_engine.log`, `arch_probe.txt`, `roundtrip512*.log`。各 gate directory に engine 生ログと三経路の cp 行列がある。

## 速度 (負荷下の参考値)

i7-14700KF、g++/AVX2、Threads=1、Hash64MiB、FV16、定跡なし。既存32局面の先頭20局面を同じ順で各20万nodes、current→wide→wide→current。current は現行 binary + Task142、wide は今回の binary + ランダム重み。別の作業の時間つき対局が全コアで動作中。実行は逐次、各回数秒。

| 順 | 構成 | nodes | 探索ms | NPS |
|---|---|---:|---:|---:|
| 1 | current | 3,803,420 | 2,421 | 1,571,012 |
| 2 | wide | 3,826,066 | 2,851 | 1,342,008 |
| 3 | wide | 3,826,066 | 2,724 | 1,404,576 |
| 4 | current | 3,803,420 | 2,950 | 1,289,295 |
| 合算 | current | 7,606,840 | 5,371 | **1,416,280** |
| 合算 | wide | 7,652,132 | 5,575 | **1,372,580** |

**合算 nodes/time の wide/current = 0.9691**。現行側だけでも約22%揺れている。重み・探索木・CPU割当の違いが混ざるので、構造だけの速度差とも時間制棋力とも読めない。途中2回だけの比 (約0.85〜0.87) は最終値に使わない。current binary SHA は `0831b1c1925c4e7577fdd3cecc2ac3234f724686b44c4d495a678c969fd186b7`。

```bash
PYTHONPATH=. "$VENV_PY" experiments/alphazero/t149_wide_bench.py \
  --current "$YANE" --wide "$WIDE_ENGINE" \
  --current-eval "$DATA_ROOT/eval_dirs/task142_finetune" \
  --wide-eval "$DRY/gate512/eval" --out "$DRY/bench"
```

`bench/result.json` に数値・binary/eval SHA、同じ directory に20局面と4回の生ログを保存。

## 残ること

- GPU は使用不可のため、wide での CUDA fused Adam 更新、VRAM、実データの学習速度は未測定。CPU 勾配照合と fast 初期化は合格したが、GPU 学習全体の実証ではない。
- ランダムネットの0cp確認は配備の整合性だけを示す。学習済みwideの精度・棋力・時間制の優劣は未検証。学習後も出力ごとにゲートを通す。
- 負荷が空いた時の速度再測定と、同じデータ/seed/学習条件の対照が必要。
- Rust や旧 Rust 前提の residual/auxiliary trainer は BNQ2 に拡張していない。今回の対象は標準 QATSharedNNUE 学習・export・YaneuraOu 配備と新ゲート。通常のルール/探索コードは変更していないため cargo test/perft は実行していない。
