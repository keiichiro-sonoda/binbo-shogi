# Task 038: dropout — 過学習は減るが度合いを圧縮する(採用せず)

## 結論

- fd8+mirror+cp回帰(clamp±5000, MSE)の上に **dropout** を積んで数水準(0.1/0.2/0.3/0.5)試した。
- **dropout は汎化ギャップ(train/val 差)を縮めるが、val の度合いを下げる方向**で縮める = **求めていた汎化ではない**:
  - val decisive 予測が **2185(なし)→2143→1987→1788→1475(p=0.5)** と**単調に低下**(正解 3915 から遠ざかる)。
  - ギャップが縮むのは **train が val 側へ落ちる**ため(train 2994→1868)。**val が train・正解側へ伸びてはいない。**
  - これは Task 037 で警告した「両方下がってギャップだけ縮む」失敗モードそのもの。**dropout は weight decay 同様に出力を圧縮**する(Task 032 と同じ罠)。
- **採用せず**。`model_cp_fd8_mirror`(Task 037, dropout なし)を据え置き。
- **メタ結論(重要)**: 容量/分散を削る正則化(weight decay=032・次元↓=037・dropout=038)は**度合いを圧縮するだけで val の度合いを上げない**。
  汎化ギャップを「正しく」縮めた(val 度合いを上げた)のは**情報を足す mirror(036)だけ**。→ **次段はデータ(量・多様性)**であり、これ以上の正則化ではない。
- dropout 実装は**推論に漏れない**ことを検証(学習専用、eval 時 identity、export 重み不変、Rust 無改修)。

## 1. dropout の配置と理由(エージェント裁量)

- **配置 = アキュムレータ(EmbeddingBag 出力)に `nn.Dropout`**、ネットワークの前。
- 理由: (1) 過学習の本体は巨大な埋め込み/表現(Task 035。35万×fd が局面の特徴組合せを暗記)なので、
  評価を駆動する**表現**に dropout をかけて次元の共適応を壊すのが筋。(2) `network` Sequential(Linear=network.0/2/4)を
  **構造ごと不変**に保てる → export(`export_nnue.py` の TENSOR_ORDER)と Rust `nnue.rs` forward が無改修。
  `nn.Dropout` はパラメータを持たず eval 時 identity なので、**学習専用で推論に一切現れない**。
- 中間層 dropout は network の層インデックスをずらし export を壊すため不採用。feature dropout は将来の別案(本タスクは標準 dropout)。

## 2. dropout 率スイープ(fd8+mirror, cp/MSE/clamp5000, 16 epochs)

主指標 = `pred_dispersion` の train/val decisive(|target|≥1500)予測 + 絶対値。

| dropout | train 予測\|cp\| | **val 予測\|cp\|** | gap | val/train | val_corr | sign | epoch1 loss差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0(fd8+mirror) | 2994 | **2185** | 809 | 0.73 | 0.80 | 0.83 | ~309k |
| 0.1 | 2917 | 2143 | 774 | 0.735 | 0.80 | 0.84 | 214k |
| 0.2 | 2682 | 1987 | 695 | 0.74 | 0.79 | 0.84 | 165k |
| 0.3 | 2442 | 1788 | 654 | 0.73 | 0.79 | 0.85 | 119k |
| 0.5 | 1868 | **1475** | 393 | 0.79 | 0.77 | 0.80 | 56k |

- **過学習補助指標(epoch1 の train/val loss 差)は dropout を強めるほど減る**(309k→56k)= dropout は確かに過学習を抑えている。
- **だが val の度合い(絶対値)は単調に低下**(2185→1475)。gap が縮むのは train が落ちるためで、**val は伸びていない**。
- val/train 比(相対ギャップ)も 0.73→0.74 とほぼ不変(p=0.5 で 0.79 だが val 絶対値が 1475 まで激減=逆効果)。
- 符号一致は微増(0.83→0.85, p≤0.3)するが、度合いを犠牲にした分で、評価品質の向上ではない。
- 代表局面の静的分解能は curated ノイズが大きく(p=0.2 で 9.12× は marginal が偶々小さいため)、頑健な指標は val 度合い=低下。

→ **「ギャップ縮小 かつ val 度合い維持/向上」を満たす率は存在せず**。dropout(アキュムレータ)は採用基準を満たさない。

## 3. 確認(採用しないが実装の健全性は検証)

- **推論に漏れない**: dropout=0.2 で学習したモデルを export → Rust `nnue-eval` と Python **eval モード** forward が一致
  (`verify_nnue_rust` 最大誤差 **2.666e-06** < 1e-3, ok)。学習専用 dropout が推論に現れないことを実証。
- 採用モデル `fd8_mirror` も nnue.py 変更後に Rust と一致(2.130e-06, ok)= 後方互換(dropout 既定0、Dropout はパラメータ無し)。
- 棋力(参考, drop0.1 = baseline に最も近い率, Lv1 8局): **0-7-1**(初の引分1だが val 度合いは baseline より低く、ノイズ範囲。avg_plies 69.4, depth 5.37)。
- 回帰: `cargo test` 16/16、perft 1〜5 完全一致、`selfplay` 完走、対 Random 20勝0敗。

## 4. 切り分けの結論と次段

- **dropout は採用せず**(過学習は減るが度合いを圧縮、val が伸びない)。`model_cp_fd8_mirror` 据え置き。
- **正則化の系統的な切り分けが完了**: weight decay(032)・feature_dim↓(037)・dropout(038)はいずれも**度合いを圧縮**し val を上げない。
  **val の度合いを上げた=本当の意味で汎化したのは、情報を足す左右対称拡張(036)だけ**。
- **次段 = 教師データの量・多様性↑**(`docs/BACKLOG.md` C)。decisive は十分(035)なので比率は弄らず、**volume を増やして暗記を物理的に難しくする**。
  正則化はもう打ち止め(系統的に効かないと確定)。主指標は引き続き `pred_dispersion` の train/val ギャップ + val 度合い。

## 成果物
- コード(コミット): `trainer/nnue.py`(MinimalNNUE に `dropout` パラメータ=アキュムレータ dropout、学習専用)、`train_nnue.py`(`--dropout`)。
  既定 0 で既存挙動・推論は不変。採用はしないが**再現と将来利用のため knob は残す**(inference-safe を検証済)。
- 重み(非コミット): 採用据え置き `~/binbo-shogi-data/models/model_cp_fd8_mirror/`。dropout 各モデルは破棄(度合い低下=劣る)。

## スコープ外(次段)
- 教師データの量↑・再生成(次の本命)。feature dropout・中間層 dropout(export 都合で見送り)。目標設計・構造・mirror は据え置き。
