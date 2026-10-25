# Task 059 — binbo QAT eval → nn.bin writer(逆経路): 自前 eval を実エンジンに bit-exact で載せる

**ゴール**: binbo の QAT eval を YaneuraOu の `nn.bin` 形式に書き出す **writer**(inverse remap + serialize)を実装し、
書き出した nn.bin を**実エンジン YaneuraOu に load** して「自前 eval が本物の探索器に正しく載る」ことを **bit-exact(0cp)** で確認。
057(reader)・058(量子化対応 BNNQ)の逆向き=**本命移行の心臓部**。

## 何をしたか(1変数。順方向 forward・探索・付番 remap・057 bridge・Rust は不触)
- **writer `trainer/binbo_to_yane.py`**(純 Python): BNNQ を読み、`R`(057 `build_remap`)の**逆 scatter**で FT を YaneuraOu
  `ft_w` へ、ft_bias 直コピー、affine を `[out,pin]` へ詰める。hash/arch は同 arch の tanuki nn.bin から流用(重み非依存)。
  直列化は **057 の `yane_nnue.write` を再利用**。**同一スロットに畳まれる binbo 行が同値かを assert**(非tied は loud に失敗)。
- **QAT tying**(binbo-学習 eval を bit-exact 書出可能に): `R` の many-to-one(YaneuraOu fe_end=1548 が 金/+P/+L/+N/+S を1スロットへ=
  **13122 金スロット×5、計 52488 を畳む**)+ binbo の **side 2 特徴は yane に無し**。`build_collapse_map` で畳まれる5特徴を代表へ束ね・
  side を pad(凍結0)へ → `QATSharedNNUE`(tie, `train_nnue.py --qat-tie`)+ export で同値化。**Rust `Mode::Quant` は不変**
  (emb 行が同値なら collapse 挙動は自動)。
- **driver `scripts/binbo_to_yane.sh`**: BNNQ→writer→nn.bin→engine `e`/int-faithful/binbo `nnue-eval` を一発突合せ。

## 結果(0cp が green、整数→整数)

### writer 正当性 + tanuki round-trip(engine-grounded, B/A)
`tools/parity_positions.sfen`(32局面)、tanuki nn.bin → BNNQ(058 `convert_quant`)→ nn.bin(writer)→ 実エンジン:

| 判定 | 内容 | 結果 |
|---|---|---|
| A | engine `e`(written) == int-faithful(`yane_forward_int`, written) | **32/32 exact, 0cp** |
| fid/B | binbo `nnue-eval`(Mode::Quant, src)== engine `e`(written)= 元 tanuki `e` | **32/32 exact, 0cp** |

→ エンジンは writer 出力を**受理**(hash/arch/layout 正)し、我々の参照 reader と**同一解釈**。round-trip bit-exact。

### tied binbo-学習 eval が実エンジンに bit-exact で乗る(C, 本タスクの核心)
`model_qat_tie_fd256_2m`(fd256/h32, 2M・5ep, economical)→ BNNQ → writer → nn.bin → 実エンジン:

| 判定 | 結果 |
|---|---|
| engine `e`(written) == binbo `nnue-eval`(Mode::Quant)== int-faithful | **32/32 exact, 0cp** |

→ **自前 eval が実エンジンに bit-exact で乗った**。tying が効いている証拠: parity 32/32 が畳まれるスロットを活性化、
writer の**全 13122 金グループ同値 assert が pass**。

### negative control(writer が非tied を弾く)
untied 058 eval(`model_qat_fd256_2m`)を writer に渡すと **`AssertionError: 52488 binbo features collapse to a
YaneuraOu slot with DIFFERENT embeddings`** で**拒否** → writer は壊れた slot を黙って書かない(検出力あり)。

### tying の品質コスト ≈ 0
| | val_loss | corr | decisive |
|---|---|---|---|
| **tied**(059, yane形式に厳密適合) | **160055** | 0.716 | 0.822 |
| untied(058) | 164353 | 0.724 | 0.818 |

→ 金/成駒の区別 + side 特徴は棋力にほぼ寄与せず、**yane 形式へ厳密適合しても品質は落ちない**(むしろ僅かに良い=正則化)。
tied の量子化残差(int−連続float)mean −12.6 / std 19.7cp(058 untied と同等=量子化誤差レベル)。

## 回帰
- `cargo test --release` = **22 green**(**Rust 無改変** → 再ビルド不要)。perft 不変。
- 既存 **untied QAT(058)** path 不変(qat_eval 残差 −3.5/26.8cp で同一)、float BNN1〜6 不変。

## 非対象 / 次タスク(本命移行)
- **256x2-32-32 本命移行**(writer green 達成済): 新規 256 self-eval 量産 + 旧版と 1000局・対局基盤。
- 「同じ eval を自前 Rust 探索 vs YaneuraOu 探索」基準点対局。探索差し替え。付番 remap・057 bridge・Rust forward 改変。

## 再現
- `bash scripts/binbo_to_yane.sh <bnnq>`(tanuki BNNQ でも tied BNNQ でも)。
- 学習: `train_nnue.py --qat --qat-tie --feature-dim 256 --hidden-dim 32`。export: `export_nnue.py`(config.qat_tie)。
- 重み/nn.bin は `~/binbo-shogi-data/models/model_qat_tie_fd256_2m`(非コミット・永続)。
