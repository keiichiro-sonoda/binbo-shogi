# Task 058 — binbo の量子化対応(QAT): 536cp → 量子化誤差

**ゴール**: binbo の forward を YaneuraOu の整数 NNUE 演算に合わせ、binbo float forward と int-faithful 参照
(=実エンジン `e`)の残差を **現状 ~536cp → 量子化誤差レベル**まで詰める。これが「学習した eval が歪まず
YaneuraOu に載る」前段。256・**engine-grounded**、Rust 推論にも反映、加法的(既存 float 不変)。

## 何をしたか(1変数=eval/trainer のみ。探索・付番 remap・057 bridge は不触)
- **量子化スキーム**: FT `ft_bias(i16)+Σ ft_w(i16)`→clamp[0,127]、affine `bias(i32)+W(i8)·x`→ClippedReLU
  `clamp(x>>6,0,127)`、出力生 i32→`out/FV_SCALE(16)`(C切捨て)。`yane_forward_int`(057, bit-exact)と同一。
- **BNNQ**(新 magic, 既存 BNN1〜6 不変): int16 FT + 別持ち ft_bias(clamp 前に加算=畳めない)+ i8 affine + i32 bias。
- **Rust `Mode::Quant`**(`src/eval/nnue.rs`): BNNQ reader + 整数 forward。`nnue-eval` が cp を直接出力。
- **converter**(`trainer/yane_to_binbo.py::convert_quant`): tanuki256 を整数のまま remap→BNNQ(engine-grounded 用)。
- **QAT trainer**(`trainer/nnue.py::QATSharedNNUE` + `train_nnue.py --qat`): ClippedReLU + 明示 ft_bias +
  fake-quant(straight-through, floor で `>>6` 忠実)。`--qat-no-fakequant` で連続=PTQ 観測。`export_nnue.py` で BNNQ。
- **計測**: `trainer/qat_eval.py`, `scripts/qat_residual.sh`(再現一発), `scripts/yane_parity.sh`(Part1 before/after)。

## 結果(before → after)

### Part 1 — engine-grounded(本命数値, 536→0)
`tools/parity_positions.sfen`(32局面)、converted tanuki256:

| forward | vs YaneuraOu `e` | 残差 |
|---|---|---|
| int-faithful(057参照) | 32/32 exact | 0cp |
| **binbo float (BNN3, plain ReLU)** [BEFORE] | r=0.871 | **std 536cp** |
| **binbo quant (BNNQ, Mode::Quant)** [AFTER] | **32/32 exact** | **0.00cp (max\|diff\|=0)** |

→ binbo の量子化 forward は実エンジンを **bit-exact 再現**。**536cp → 0cp**。

### Part 2 — QAT 学習(fd256/hidden32, shared_8m から 2M・5ep, economical)
deploy(Rust BNNQ)== Python int 参照 = **32/32 bit-exact**(両モデルとも)。残差 = deploy int − 連続 float:

| 学習 | 残差 mean | 残差 std | max\|.\| | deploy/理想 val_loss | corr | decisive |
|---|---|---|---|---|---|---|
| **QAT**(fake-quant on, floor) | **−3.5cp** | **26.8cp** | 82cp | 164k(deploy) | 0.72 | 0.82 |
| PTQ(連続学習→素朴量子化) | +8.0cp | 20.4cp | 60cp | 165k(理想) | 0.70 | 0.80 |

- **536cp → ~25cp(約20×)**。残差の主項は**出力層 i8**(scale `16·600/127`)で、実 NNUE エンジン共通の量子化下限。
- **品質は落ちない**: QAT の **deploy 時** val_loss/corr が PTQ の **連続(理想)** と同等 → 量子化が eval を歪めない。
- floor-aware 学習で QAT の系統バイアス ~0(−3.5cp; round 学習だと −21cp だった)。

## 回帰
- `cargo test --release` = **22 green**(`quant_forward_matches_hand_computation` 追加)。
- `perft 5` = **19,861,490**(不変)。
- 既存 BNN3 float eval(`model_shared_fd8_mirror_8m`)= load+forward 不変(logits ±3.5)。

## 非対象 / 次タスク(本命移行)
binbo→**nn.bin writer**(BNNQ→YaneuraOu plain layout 逆remap+serialize)で QAT eval を**実エンジンへ load** して `e`
直接照合 / 新規256 self-eval 量産 / 対局基盤。探索差し替え・付番 remap・057 bridge 改変は本タスク対象外。

## 再現
- Part1: `scripts/yane_parity.sh`(before/after を並記、int-faithful==`e` 同時確認)。
- Part2: `bash scripts/qat_residual.sh <model_dir>`。学習: `train_nnue.py --qat --feature-dim 256 --hidden-dim 32`。
- モデル/重みは `~/binbo-shogi-data/models/model_qat_fd256_2m`, `model_ptq_fd256_2m`(非コミット・永続)。
