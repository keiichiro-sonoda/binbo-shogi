# 自作 NNUE を やねうら王形式(nn.bin)で出力する — 構想と調査(Task 056)

> **位置づけ**: 方向転換(自前探索→やねうら王土台, 評価で勝負)の**技術的核心=接続点**。
> 本ドキュメントは **構想/調査まで**(実装は次段=Phase 2)。決定方針 = **binbo の自作 trainer
> (`trainer/train_nnue.py` 系)を拡張して やねうら王 `nn.bin` を直接出力する**(最大限「自作」)。
> nnue-pytorch 採用や C++ 学習部は採らない(セッションで確定)。
> 一次情報 = upstream クローン `~/binbo-shogi-data/engines/yaneuraou/source/eval/nnue/`(無改変)。記憶でなくソースを正本にした。

## 1. 評価関数の差し替え機構(確認済み)

- ビルド edition = `YANEURAOU_ENGINE_NNUE`(既定)= **halfkp_256x2-32-32**(`nnue_architecture.h:63` の active include)。
- やねうら王は **`EvalDir`(USI option, 既定 `eval`)配下の `nn.bin`** を読む(本タスクの smoke で実証: `setoption name EvalDir value <abs>` → `isready` で `loading eval file .../nn.bin` → `readyok`)。`EvalFile` 単体オプションはこの版には無く、ディレクトリ+固定名 `nn.bin`。
- **自作 eval で勝負する = この `nn.bin` のバイト列を自分の重みに差し替えるだけ**(エンジン本体は無改変)。
- **重要: ビルド edition ↔ eval アーキは厳密一致が必須**(下記ハッシュ)。256x2-32-32 のエンジンには 256x2-32-32 の nn.bin しか載らない。

## 2. nn.bin バイト構造(ソース精読・`evaluate_nnue.cpp` / `nnue_common.h` / `nnue_feature_transformer.h` / `layers/affine_transform.h`)

全て **little-endian**。`read_little_endian`(`nnue_common.h:135`)。

```
ヘッダ (ReadHeader, evaluate_nnue.cpp:235):
  u32  version        = kVersion = 0x7AF32F16        (不一致は Warning のみ・継続)
  u32  hash_value     = kHashValue (全体ハッシュ)     (不一致は Warning のみ・継続)
  u32  arch_str_size
  byte arch_str[size] = GetArchitectureString()(可読の構造文字列)

本体 (Detail::ReadParameters, evaluate_nnue.cpp:173 — 各コンポーネント先頭に u32 ハッシュ):
  FeatureTransformer:
    u32  component_hash = FeatureTransformer::GetHashValue()  ← ★不一致なら load 失敗
    i16  biases[kHalfDimensions=256]
    i16  weights[kHalfDimensions * kInputDimensions]          (= 256 * 81*fe_end)
  Network = 3 affine 層を入力側から順に(各層先頭に u32 ハッシュ):
    AffineTransform 512→32 :
       u32 component_hash                                     ← ★要一致
       i32 biases[32]
       i8  weights[32 * kPaddedInputDimensions(512→512)]      (plain 出力major・padded順)
    AffineTransform 32→32 : u32 hash / i32 biases[32] / i8 weights[32*32]
    AffineTransform 32→1  : u32 hash / i32 biases[1]  / i8 weights[1*32]
```

- `kInputDimensions`(HalfKP)= `SQ_NB(81) * fe_end`(`features/half_kp.h:27`)= 玉位置81 × BonaPiece数。`kMaxActiveDimensions = PIECE_NUMBER_KING`(立つ特徴=駒数)。
- **LEB128 圧縮**: FT は `read_leb_128`(magic `"COMPRESSED_LEB128"`)版と plain int 版の両対応(`nnue_feature_transformer.h:179`、`USE_ELEMENT_WISE_MULTIPLY` で分岐)。配布 nn.bin の多くは LEB128。**自作出力は plain int で良い**(エンジンは両方読める。tanuki- smoke eval は 64,217,066 B = plain 相当)。
- **affine 重みの並び替え注意(GetWeightIndex, `affine_transform.h:218-239`)**: SSSE3/AVX2 ビルドでは内部配置がスクランブルされるが、**ファイル側は plain 順**(エンジンが `weights_[GetWeightIndex(i)] = read(i)` で展開)。自作出力は **plain 出力major + padded**(`i = out*kPaddedIn + in`、`in≥kInputDimensions` は 0 埋め)で書く(= nnue-pytorch serialize.py と同じ規約)。やねうら王内部 `WriteParameters` は内部スクランブル順を吐くので**そのまま参考にしない**。

## 3. 量子化スキーム(エンジンが前提とする整数化)

| 層 | weight 型 | bias 型 | 出力処理 |
|---|---|---|---|
| FeatureTransformer | int16 | int16 | accumulator(int16)を clamp `[0, 127*2]` → uint8(`nnue_feature_transformer.h:283`) |
| Affine(各層) | int8 | int32 | `(W·x+b) >> kWeightScaleBits(=6)`(÷64)→ ClippedReLU `[0,127]`→ uint8。出力層は int32 |
| 最終評価値 | — | — | `output / FV_SCALE`(既定16, 水匠5は24。USI option で可変, `nnue_common.h:70`) |

- 定数: `kWeightScaleBits=6`(affine 出力 ÷64)、`FV_SCALE=16`。これは**エンジン固定**。trainer 側は float 重みをこのスケールに合わせて量子化する(= 次段で書く新規コード):
  - 標準 Stockfish/nnue-pytorch 規約に一致させる想定: affine `w_i8=round(w_f*64)`(clamp±127)、`b_i32=round(b_f*64*127)`、FT `w_i16=round(w_f*127)`、`b_i16=round(b_f*127)`。最終 scale は学習時の出力スケールと FV_SCALE の積で cp に対応。**正確な係数は次段で round-trip 実測して確定**(§6)。

## 4. ハッシュ(なぜ「次元厳密一致」が要るか)

- **全体 `kHashValue`**: 不一致でも **Warning のみで継続**(`evaluate_nnue.cpp:203`)。
- **各コンポーネント `GetHashValue()`**: `Detail::ReadParameters`(:178)で**不一致なら load 失敗**。
  - FT: `RawFeatures::kHashValue ^ kOutputDimensions`(`:166`)。
  - Affine: `0xCC03DAE4 + kOutputDimensions ^ (prevHash>>1) ^ (prevHash<<31)`(`:195`)。
  - → **アーキ(特徴セット + 各層次元)が完全一致したときのみ load 可**。dims が1つでも違えば弾かれる。

## 5. binbo(自作)→ やねうら王 HalfKP 対応とギャップ

| 観点 | binbo 現状(BNN3 shared, `docs/kp_spec.md`) | やねうら王 halfkp_256x2-32-32 | ギャップ/対応 |
|---|---|---|---|
| 特徴セット | HalfKP(手番側視点・正準 shared W₁) | HalfKP<kFriend>(BonaPiece) | 同族。**付番が別系**(下記) |
| 変換次元 fd | **8**(fd8) | **256**(L1) | ① trainer を 256x2-32-32 で学習 ②カスタム edition |
| 隠れ層 | 8→hidden→1(自作) | 512→32→32→1 | 層構成を 32-32 に合わせる(①) |
| 量子化 | float32(Rust 推論も f32) | int16/int8 + kWeightScaleBits/FV_SCALE | **新規=量子化+直列化コード**(§3) |
| 付番 | cshogi 由来(駒種順 `P,L,N,S,B,R,G`・持駒順が別=`kp_spec` の罠) | BonaPiece(`fe_end`) | **index 再マップ必須**(§5.2) |

### 5.1 次元の二択(★要決定 in 次段)
- **A. trainer を halfkp_256x2-32-32 で学習**(推奨): 既定 edition にそのまま載る・配布 eval と互換・将来の電竜戦提出が楽。コスト = fd8→256 で器が桁増(学習・量子化の作法を新規に整える)。**binbo の Lv3+ 天井(fd8 の表現容量)を破る方向とも合致**(STATUS Task 055 の結論)。
- **B. binbo 次元に合うカスタム YANEURAOU_EDITION を作る**: `architectures/` に halfkp_8x2-… を足して再ビルド。エンジンが自分のハッシュを計算するので trainer はそのハッシュを吐けば載る。だが配布 eval と非互換・大会で特殊ビルド前提=筋が悪い。**非推奨**(切り分け検証用に限る)。

### 5.2 付番の突合せ(必須)
- やねうら王 HalfKP の index = `king_sq(81) × fe_end + BonaPiece`。binbo の KP 付番(`kp_spec.md`)とは**駒種順・持駒オフセットが異なる**。
- 対応 = ①binbo を**やねうら王の BonaPiece index 空間で直接学習**(教師生成時に index を BonaPiece で付ける)か、②学習後に **embedding 行を index 置換**で並べ替える。**①が安全**(検証は kp_spec の罠=盤上 `P,L,N,S,B,R,G` と持駒順の差を、cshogi/やねうら王の双方で 1 局面突合せ)。

### 5.3 PSV 再訪メモ
- DESIGN.md「PSV は採用しない」は本路線では**そのまま維持で良い**(自作 trainer は hcpe + 自作 KP を継続、PSV を経由しない)。将来 nnue-pytorch と相互検証する場合のみ PSV を再訪(その時に DESIGN を更新)。

## 6. 次段(Phase 2)の検証計画 ※本タスクでは実装しない

1. **round-trip(バイト一致)**: 配布 nn.bin(tanuki- smoke)を §2 の規約で parse → 同一バイトで re-emit → `sha256` 一致を確認。これで「読めて書ける」を保証してから自作重みを差す(NOCCA 流=オラクル一致)。
2. **forward parity(評価値一致)**: 自作 nn.bin を やねうら王に load → N 局面で **やねうら王の静的評価**(`go depth 0` 相当 or eval コマンド)と **binbo Python 推論**(量子化後)を突合せ。誤差を量子化丸めの範囲内に収める。
3. **棋力**: 旧版(やねうら王標準 or 自前)と 1000 局(先後入替・勝率/Elo/信頼区間)。詳細は `docs/BACKLOG.md`。

## 7. 一次情報(再検証用ポインタ, クローン内 相対パス)

`source/eval/nnue/` 配下: `nnue_common.h`(version/scale/LE/LEB128)、`evaluate_nnue.cpp`(ヘッダ・ReadParameters・hash)、
`nnue_feature_transformer.h`(FT 量子化・GetHashValue)、`layers/affine_transform.h`(int8/int32・GetWeightIndex・GetHashValue)、
`architectures/halfkp_256x2-32-32.h`(層構成)、`features/half_kp.h`(kDimensions=81×fe_end)。

---

## 8. forward parity 手順と許容窓(Task 057, 確立済み)

方向転換の最大の落とし穴 = **HalfKP 付番ズレ(binbo cshogi系 ≠ YaneuraOu BonaPiece)のサイレント故障**
(load は通るのに評価値が無相関=静かに壊れる)。標準256ビルド + tanuki256(確実に load する既知 eval=不偏な
ground truth)で、これを「評価値一致」で潰す手順を確立した。**本命256本走でもそのまま使える**。

### 8.1 実装(`trainer/` 3 モジュール + driver)
- `yane_nnue.py` — nn.bin の parser/serializer(§2 の plain layout)。round-trip **byte一致**(rebuild from arrays も load 可)。
- `yane_index_map.py` — **付番 remap** `R: binbo shared idx → YaneuraOu HalfKP idx(=1548·king + BonaPiece)`。
  - 対応の鍵(全て source 確認): 升番号は cshogi==YaneuraOu(`(file-1)*9+(rank-1)`, `Inv=80-sq`)で一致。
    binbo の side-to-move 正規化フレーム == YaneuraOu の perspective(`piece_list_fb/fw`)。
    **fe_end=1548 ⇒ DISTINGUISH_GOLDS OFF**: `+P/+L/+N/+S` は GOLD に畳む(binbo 13駒種→YaneuraOu 9駒種)。
    BonaPiece base は `source/evaluate.h` から転記。
- `yane_to_binbo.py` — ① **bit-exact 参照 forward** `yane_forward_int`(remap + YaneuraOu の整数演算を厳密再現:
  FT clamp[0,127]→affine→ClippedReLU(`clamp(x>>6,0,127)`)→affine→`out/FV_SCALE`、C切り捨て除算)。
  ② **converter** tanuki256 → binbo BNN3(de-quant + remap、ft_bias は network.0.bias に畳む)。
- driver `scripts/yane_parity.sh` — 位置集合(`tools/parity_positions.sfen`)で YaneuraOu `e`(真値)・int forward・
  binbo `nnue-eval` を集めて突合せ。**YaneuraOu の静的 eval は USI 拡張 `e`(`engine.evaluate()`)**で取り出す。

### 8.2 結果(GREEN)と許容窓
- **付番 correspondence = GREEN(これが本命)**: binbo の active 特徴(`canonical_shared_halves`)を R で写したものが、
  盤から独立計算した YaneuraOu active index 集合と **完全一致**(own/opp 両 perspective)。random self-play **1620局面
  (うち promotions 460)で mismatch 0**。**negative control**(remap を1行入替)は RED 検出 ✓ = 検出力あり。
- **forward parity(int-faithful)= GREEN・bit-exact**: 多様32局面で `yane_forward_int` == YaneuraOu `e` が **32/32 完全一致**
  (max|diff|=0)。→ remap + de-quant 解釈 + forward 全体が実エンジンを**厳密再現**。**許容窓 = 0cp(exact)**。
- **binbo Rust float forward(converted eval)= 緩い相関のみ**: Pearson r≈**0.87**、resid std≈**536cp**。
  特徴は正しく追える(付番が誤りなら r≈0)が、**binbo は plain ReLU、YaneuraOu は ClippedReLU+`>>6`** のため tight でない。

### 8.3 重要な含意(Phase 2 への設計制約)
- **bridge(serialize/remap/forward 解釈)は厳密正**(int-faithful が bit-exact で実証)。付番ズレは撲滅済み。
- だが **binbo の現 net(plain ReLU, 非量子化)を YaneuraOu に載せても評価値は厳密一致しない**。本命 forward 路線
  (binbo→nn.bin)では、binbo trainer を **quantization-aware(clamp[0,127]+整数スケール)** にするか、出力を
  YaneuraOu の整数演算に合わせて serialize する必要がある。許容窓は「int-faithful=0cp / 非量子化 float=相関のみ」。
- 本命256 eval の parity 再走は、この driver の eval を差し替えるだけで動く(手順確立済み)。

---

## 9. binbo の量子化対応(QAT)— 536cp を量子化誤差まで詰める(Task 058, 完了)

§8.3 の制約(plain ReLU の binbo float は YaneuraOu の ClippedReLU+整数演算と厳密一致しない=536cp ズレ)を、
**binbo を量子化対応**にして潰した。256・**engine-grounded**(§8 の int-faithful=実エンジンを真値)。**加法的**で
既存 float BNN1〜6 は不変。

### 9.1 量子化スキーム(`yane_forward_int` と一致)
1局面 forward: FT `acc = ft_bias(i16) + Σ ft_w(i16)` → **clamp[0,127]**(own/opp 連結)→ affine 各層
`bias(i32)+W(i8)·x` → 中間は **ClippedReLU `clamp(x>>6,0,127)`**、出力は生 i32 → `out/FV_SCALE(16)`(C切捨て除算)。
学習時スケール(`trainer/nnue.py` の定数, fake-quant と export で共有):活性 [0,1]↔[0,127](`ACT_SCALE=127`)、
FT→i16(`*127`)、隠れ affine→i8(`*64=2^6`)・bias→i32(`*127·64`)、出力→i8(`*FV_SCALE·scale_cp/127`)・
bias→i32(`*FV_SCALE·scale_cp`)。出力スケールに `scale_cp`(=600)を畳むので **deploy int の `o2/FV_SCALE` が
`forward_logit·scale_cp`(=cp)に一致**。`>>6` は **floor**(算術右シフト)なので活性 fake-quant も floor で学習(deploy 忠実)。

### 9.2 実装(加法的・1変数)
- **重み形式 BNNQ**(新 magic `b"BNNQ"`): `magic|total|fd|hidden|fv_scale(i32)|weight_scale_bits(u32)|
  emb(i16) ft_bias(i16) w0(i8) b0(i32) w1(i8) b1(i32) w2(i8) b2(i32)`。**ft_bias は別持ち**(clamp 前に
  perspective 毎へ加算するため、float 路線のように network.0.bias へ畳めない)。
- **Rust `Mode::Quant`**(`src/eval/nnue.rs`): BNNQ reader + 整数 forward。特徴抽出は `Mode::Shared` と同一
  (`kp_indices_canonical_shared`)。`forward()` は Quant のとき **cp(整数)** を返す(`e` と直接比較可)。既存 float 路線不変。
- **converter**(`trainer/yane_to_binbo.py::convert_quant`, `--convert-quant`): tanuki256 の **整数のまま** remap して
  BNNQ 出力(de-quant しない)。→ §9.3 の engine-grounded 検証用。
- **QAT trainer**(`trainer/nnue.py::QATSharedNNUE`, `train_nnue.py --qat`): shared EmbeddingBag + 明示 ft_bias +
  ClippedReLU + fake-quant(straight-through、`--qat-no-fakequant` で連続=PTQ観測)。出力は logit、`pred=logit·scale_cp`。
  既存 `run_batches` の `shared` 路にそのまま乗る。**export**(`export_nnue.py`, config.qat)で BNNQ 書き出し。
- **計測**(`trainer/qat_eval.py`): 連続 float eval と deploy int eval(binbo 付番)を出し残差。Rust deploy との突合せも。

### 9.3 結果(GREEN, before→after を並記)
- **Part1 engine-grounded(本命数値, 536→0)**: converted tanuki256 を BNNQ 化 → binbo `Mode::Quant` の `nnue-eval` が
  `tools/parity_positions.sfen` 32局面で **YaneuraOu `e` と 32/32 bit-exact(max|diff|=0, std 0.00cp)**。
  **before: binbo float(BNN3) vs `e` = r0.87 / resid std 536cp → after: binbo quant(BNNQ) = 0cp**。
  (`scripts/yane_parity.sh` が両方を並記。int-faithful==`e` も 32/32 で同時確認。)
- **Part2 QAT 学習(fd256/hidden32, 2M・5ep, economical)**: deploy(Rust BNNQ)== Python int 参照 **32/32 bit-exact**。
  量子化残差(deploy int − 連続 float): **QAT mean −3.5cp / std 26.8cp**、PTQ(連続学習→素朴量子化)mean +8cp / std 20.4cp。
  → **536cp → ~25cp(約20×)**。残差の主項は**出力層 i8**(scale `16·600/127`)= 実 NNUE エンジン共通の量子化下限。
  floor-aware 学習で QAT の系統バイアスは ~0(−3.5cp)。
- **quality(歪まないか)**: QAT の **deploy 時** val_loss 164k / corr 0.72 / decisive 0.82 ≈ PTQ の **連続(理想)** 165k /
  0.70 / 0.80。→ **量子化は品質を落とさない**(deploy int が理想連続と同等)。
- **回帰**: `cargo test` 22 green(quant_forward 手計算一致を追加)、perft5=19,861,490 不変、既存 BNN3 float は load+forward 不変。

### 9.4 含意・残タスク(本命移行=次タスク)
- **bridge は厳密正・量子化 forward も engine 一致**。これで「学習した eval が歪まず YaneuraOu に載る」前段が揃った。
- **次**: binbo→**nn.bin writer**(BNNQ → YaneuraOu plain layout へ逆remap+serialize)で QAT eval を**実エンジンに load**して
  `e` 直接照合 / 新規256 self-eval 量産 / 対局基盤。残差下限(出力i8 ~20cp)は実エンジンと同等なので実用上問題なし。

---

## 10. binbo QAT eval → nn.bin writer(逆経路)— 自前 eval が実エンジンに bit-exact で乗る(Task 059, 完了)

§9 の量子化 forward(読込方向)に続き、**逆向き=binbo BNNQ → YaneuraOu nn.bin writer** を実装。書き出した nn.bin を
**実エンジンに load** して `e` を取り、**bit-exact(0cp)** を確認した(**本命移行の心臓部**)。**Rust 無改変**(writer は Python)。

### 10.1 writer(`trainer/binbo_to_yane.py`, 純 Python, 057 再利用)
BNNQ を読み、`R`(057 `build_remap`)の**逆 scatter**で `yane_ftw2d[R[b]] = emb[b]`、ft_bias 直コピー、affine を `[out,pin]` へ。
hash/arch/version は同 arch の tanuki を流用(重み非依存)。直列化は **`yane_nnue.write`** を再利用。
**同一 yane スロットへ畳まれる binbo 行が同値かを assert**(非tied は loud に失敗=writer バグ/未tie 検出)。driver `scripts/binbo_to_yane.sh`。

### 10.2 collapse(many-to-one)と tying
- **実測**: `R` は many-to-one。YaneuraOu fe_end=1548 が **金/+P/+L/+N/+S を1スロット**へ畳む(**13122 金スロット×5、計 52488**)、
  binbo の **side 2 特徴は yane に無し**。→ converted tanuki は5特徴同値ゆえロスレスだが binbo-学習 eval は素では書けない。
- **tying(`build_collapse_map` + `QATSharedNNUE` tie + `--qat-tie`)**: 畳まれる5特徴を代表行へ束ね、side を pad(凍結0)へ。
  export で同値化 → **Rust `Mode::Quant` は不変**(emb 行が同値なら collapse 挙動が自動)。これで binbo-学習 eval が **bit-exact 書出可能**。

### 10.3 結果(GREEN, 0cp)
- **tanuki round-trip(engine-grounded)**: tanuki→BNNQ→nn.bin(writer)→engine `e` == 元 tanuki `e` == int-faithful == binbo `nnue-eval`、**32/32 0cp**。
- **tied binbo-学習 eval(核心)**: `model_qat_tie_fd256_2m`→writer→nn.bin→**engine `e` == binbo `nnue-eval`(Mode::Quant)== int-faithful、32/32 0cp**。
  parity 32/32 が畳まれるスロットを活性化、writer の**全13122金グループ同値 assert pass**。
- **negative control**: untied 058 eval は writer が `AssertionError(52488 conflicting)` で**拒否**(検出力あり)。
- **tying コスト ≈ 0**: tied val_loss 160055/corr 0.716/decisive 0.822 ≈ untied(058)164353/0.724/0.818 →
  金/成駒区別+side は棋力にほぼ寄与せず、**yane 形式へ厳密適合しても品質維持**。tied 量子化残差 −12.6/19.7cp(量子化誤差レベル)。
- **回帰**: `cargo test` 22 green(**Rust 無改変**)、perft 不変、untied QAT(058)・float 不変。詳細 `reports/task059_writer.md`。

### 10.4 含意・残タスク(本命移行)
- **読込(057)・量子化 forward(058)・書出(059)が全て engine と bit-exact** → 自前 eval を実エンジンに歪みなく載せる経路が**完成**。
- **次**: 256x2-32-32 で**新規 self-eval 量産**(自前教師→tied QAT 学習→writer→engine)+ 旧版と 1000局・対局基盤。

---

## 11. 探索器比較対局 — 同一 eval × 自前 Rust 探索 vs YaneuraOu 探索(Task 060, 完了)

§9/§10 の bridge を**初めて本走で使用**。本命 256(full 8M, tied QAT)eval を**両側 bit-exact 同一**に固定し、
**探索器のみを変数**に対戦。eval が両側で同一(0cp ゲート通過)なので、結果の差は**探索器に帰属**する。

### 11.1 Rust 配線(eval scaling + ハーネス)
- **mode-aware leaf eval**: `Mode::Quant` の `forward()` は **cp 直返し**(float BNN1〜6 は logit×600)。`Nnue::is_quant()` +
  `Searcher::leaf_eval`(`src/search.rs`)を追加し leaf 3 箇所を置換。**float modes 数値不変**(`eval_to_int` 温存)。
- **`nnue-vs-yane`**(`src/main.rs`, `run_nnue_vs_gikou` 雛形)+ 汎用 USI driver `src/usi_engine.rs` + trinomial SPRT `src/sprt.rs`。
  YaneuraOu options: EvalDir / USI_OwnBook=false / Threads=1 / MinimumThinkingTime=0 / NetworkDelay=0 / NetworkDelay2=0(0.1s 尊重)。
- driver `scripts/selfeval_vs_yane.sh`: export→writer→**0cp ゲート(RED なら abort)**→SPRT 対局を一発。

### 11.2 結果(0-19-0, SPRT accept_h1)
- **対局前ゲート**: 本命 BNNQ→nn.bin→engine `e` == binbo Mode::Quant == int-faithful、**32/32 0cp**(両側 eval 完全同一)。
- **SPRT 対局**(movetime 100ms, H1=binbo ≥50 elo 弱い, α=β=0.05): **19 局で accept_h1**(LLR 3.027 ≥ 2.944)、binbo **0-19-0**(全局詰まされ負け)。
  → **YaneuraOu の探索は同一 eval で binbo の探索より ≥50 elo 強い(p<0.05)**。
- **速度差(総合探索力)**: 同一 movetime で YaneuraOu avg depth **15.9(max 39)/ 120.6k node** vs binbo **3.8(max 7)/ 34.6k node**
  = 約 4.2× 深さ・約 3.5× node。これが 0-19-0 の正体(SIMD+差分更新 vs full 再計算・非SIMD)。
- **回帰**: `cargo test` **24 green**(新規 `sprt` 2 本含む)、perft 1-5 不変、float/既存 `nnue-vs-gikou` 不変。**057-059 bridge 不触**。詳細 `reports/task060_searchmatch.md`。

### 11.3 含意・残タスク
- **bridge は本走で機能**: 自前 eval を実エンジンに 0cp で載せ、探索器を差し替えて定量比較できる基盤が動いた。
- 本結果は**実装速度込みの「総合探索力」**。純探索アルゴリズム分離は **fixed-nodes** 比較(任意 follow-up)。本命 eval は過学習傾向
  (best_epoch=1)=次回 epoch/正則化の調整余地。

---

## 12. 世代比較で eval を強化(同一探索 × eval-vs-eval, Task 061, 完了)

§11 の逆軸: **探索器を固定**(YaneuraOu 両側)し **eval だけ**を差し替えて世代の棋力を測る。Task 060 本命(lr1e-3)を baseline に、
教師(技巧2蒸留8M)据え置きで**学習側だけ**を調整し、踏み台を高くした。

### 12.1 ハーネス(`yane-vs-yane`, 060 infra 再利用)
- `src/main.rs::run_yane_vs_yane` + `play_yane_vs_yane_game`: YaneuraOu を 2 起動(別 EvalDir, 同一探索)、両手とも外部エンジン、
  色交替・ペアopening・千日手・trinomial SPRT(`src/sprt.rs`, H1=候補が強い elo1>0)。`src/usi_engine.rs`/`src/sprt.rs` は **060 のまま再利用**。
- driver `scripts/eval_vs_eval.sh`: 候補 model dir → export(BNNQ)→ writer(nn.bin)→ **0cp ゲート**(RED で abort)→ `yane-vs-yane`。
- trainer 最小追加 `--lr-schedule {none,cosine,step}`(default none = 従来数値不変)。回帰: `cargo test` **24 green**、perft 1-5 不変、既存 `nnue-vs-yane`/`nnue-vs-gikou` 不変。

### 12.2 結果(踏み台 +約190〜240 elo, 学習 lr の修正だけで)
- **lever は lr の水準**(一変数 val 探索): 正則化(weight_decay/dropout)・cosine schedule は無効、**lr を下げると val 単調改善**
  (corr 0.8334→0.8577, lr1e-3→1e-5)。**Task 060 本命は lr が高すぎて過学習**(epoch 1 で val 最適を overshoot)だった。
- **eval-vs-eval SPRT(探索器固定, movetime 100ms, 両側 0cp ゲート)**: 低 lr 3 候補が baseline に圧勝 — lr1e-4 **+208**、lr3e-5 **+240.8**、
  lr1e-5 **+188.5**(74.8%, CI +137–249)。3 者は CI 重複=互いに区別不可。**val の小改善が桁違いの elo 差**=baseline の過学習失点(corr に出にくい)。
- **採用 = `task061_lr1e-5_e18`**(val_corr 0.8577, nn.bin sha256 `559cebd1…`)。新 baseline 世代に昇格。詳細 `reports/task061_eval_tuning.md`。

### 12.3 含意・残タスク
- **eval-vs-eval 基盤が機能**: eval の世代を 0cp で載せ替えて棋力を SPRT で定量比較・採用できる(自己強化ループの採用判定に直結)。
- **次 lever = データ量増量**(8M→16M+, BACKLOG)。学習 lr は floor 到達。**自己強化ループ**は踏み台が高くなった今着手可。
