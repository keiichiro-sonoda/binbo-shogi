# Task 042: 器拡大の再訪(8M+mirror) — feature_dim↑ は採用しない

## 結論

- Task 041 の転換点(生800万+mirrorで過学習ほぼ消滅)を受け、**feature_dim を fd8→16→32→64**へ再拡大した。
- **結論: feature_dim↑ は実質伸びない。採用しない。**
  - 主指標 `pred_dispersion` の val decisive 予測は **fd8=2729 / fd16=2716 / fd32=2732 / fd64=2659**。
  - fd32 は fd8 より +3cp だけで、誤差級。fd64 は悪化。
  - 技巧2 Lv1 同一 seed 8局は **fd8=2-6-0、fd32=2-6-0**。勝率も改善しない。
  - fd32 はモデルサイズ 11MB→44MB、探索平均深さ 5.36→5.17 とコスト増。
- **採用モデルは据え置き: `model_cp_fd8_mirror_8m`**。
- 切り分け: 8M で過学習が消えても、単純な embedding 幅増では val 70% の壁を越えない。律速は「fd8 の幅」ではなく、**KP 表現そのものの粗さ / 教師ラベル品質 / 探索側の葉評価の使い方**のいずれか。

## 実験条件

- データ: `/tmp/binbo-task041/cache_8m`
  - train 7,950,000、val 50,000。
  - val は run1m held-out 5万で、Task 036〜041 と同一。
- 目標: cp 回帰、`--loss mse --clamp-cp 5000 --augment-mirror`。
- 比較軸:
  - fd8/h16: Task 041 採用ベース。
  - fd16/h16: 20 epochs。
  - fd32/h16: 8 epochs。best が早期に出たため、過学習確認まで。
  - fd64/h32: 6 epochs。上限側確認。重いため短期。
- GPU: RTX 4060 Ti。Codex サンドボックス内では GPU が見えないため、学習コマンドのみサンドボックス外で実行。

## 学習指標

`train_nnue.py` の best epoch 指標:

| model | fd | hidden | best epoch | best val_loss | val_corr | decisive sign | train sec/epoch(best) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fd8 baseline | 8 | 16 | 3 | 594,891.9 | 0.9009 | 0.8943 | 15.0 |
| fd16 | 16 | 16 | 3 | 602,307.7 | 0.8996 | 0.8939 | 23.6 |
| fd32 | 32 | 16 | 3 | 593,767.8 | 0.9012 | 0.8944 | 43.9 |
| fd64_h32 | 64 | 32 | 3 | 610,735.9 | 0.8981 | 0.8920 | 68.4 |

- fd32 は val_loss だけ見ると fd8 より約 1,124 改善するが、全体スケールでは約 0.2%。
- fd16/fd64 は fd8 より悪い。

## 主指標: pred_dispersion

`trainer.pred_dispersion --decisive-cp 1500 --limit 60000`:

| model | train decisive pred | val decisive pred | target val | gap(train-val) |
|---|---:|---:|---:|---:|
| fd8 baseline | 2877 | **2729** | 3915 | 148 |
| fd16 | 2892 | 2716 | 3915 | 176 |
| fd32 | 2925 | **2732** | 3915 | 193 |
| fd64_h32 | 2915 | 2659 | 3915 | 256 |

- fd32 の val は fd8 +3cp。これは採用判断に足る差ではない。
- fd32/fd64 は train 側だけわずかに伸び、gap はむしろ広がる。容量が汎化表現ではなく訓練側へ寄る傾向。

## 棋力スモーク(技巧2 Lv1)

同一 opening seed `20290623`、1秒/手、8局:

| model | record | score | avg depth | avg nodes | model size |
|---|---:|---:|---:|---:|---:|
| fd8 baseline | 2-6-0 | 25.0% | **5.36** | 887,517 | 11MB |
| fd32 | 2-6-0 | 25.0% | 5.17 | 730,821 | 44MB |

- fd32 は同じ勝率。平均深さは fd8 より浅い。
- Task 041 の 1-7-0 から 2-6-0 に見える改善は seed 差で起きる範囲。fd8 同一 seed でも 2-6-0。

## Rust 数値一致

fd32 は export して Rust 推論と照合:

- `trainer.export_nnue model_cp_fd32_mirror_8m_task042 --out nnue_f32.bin`
- `trainer.verify_nnue_rust` 3,000局面:
  - max_abs_err `3.115e-06`
  - count_abs_err > `1e-3`: 0
  - 判定: ok

Rust の動的次元ロードは fd32 でも問題なし。

## 成果物

- 非コミットモデル:
  - `~/binbo-shogi-data/models/model_cp_fd16_mirror_8m_task042`
  - `~/binbo-shogi-data/models/model_cp_fd32_mirror_8m_task042`
  - `~/binbo-shogi-data/models/model_cp_fd64_h32_mirror_8m_task042`
- 非コミットログ:
  - `~/binbo-shogi-data/logs/task042/train_fd16.log`
  - `~/binbo-shogi-data/logs/task042/train_fd32.log`
  - `~/binbo-shogi-data/logs/task042/train_fd64_h32.log`
  - `/tmp/binbo-task042/nnue_vs_gikou_fd32.log`
  - `/tmp/binbo-task042/nnue_vs_gikou_fd8_same_seed.log`

## 次の判断

- `feature_dim↑` は 1M でも 8M でも効かなかったため、単純な幅拡大は打ち止め。
- 次に切るなら、幅ではなく構造:
  - 正準 HalfKP 化(2視点の扱いを標準形に近づける)。
  - KPP など入力特徴そのものを richer にする。
  - 教師ラベルの質(depth 上げ・局面分布の見直し)。
- fd8 は小さく速く、棋力も同等なので継続採用。
