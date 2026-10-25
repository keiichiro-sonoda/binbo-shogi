# Task 039: 教師データ倍増(生100万→200万)で汎化が伸びた + GPU 学習整備

## 結論

- **情報を足す手の本命=生データ量↑を実施**。技巧2 教師を新規に生100万生成(14700KF, 12並列, depth8, seed 20270101)、
  既存100万と合わせ**生200万 + 左右対称拡張**で再学習(構成=fd8+mirror+cp回帰 clamp±5000 MSE 据え置き、量だけ増)。
- **val の度合いが「正しく」伸びた**(主指標, val=run1m の同一 held-out 5万で直接比較):
  - val decisive 予測 **2185(生100万)→ 2455(生200万)**(+12%, 正解 3915 へ前進)。**train はほぼ不変(2994→2996)**。
  - **gap 809→541(−33%)、val が train 側へ上がって縮んだ良い縮み方**(dropout の「両方下がる」=度合い圧縮とは質が逆)。
  - val_corr 0.80→0.84、decisive 符号一致 0.83→0.85。情報追加(mirror と同系統)が効いた。
- **GPU 学習を整備**(torch 2.12.1+cu130, RTX 4060 Ti, CUDA 利用可)。**GPU は CPU の約5倍速**(1.95M, batch1024 で 18s→3.6s/epoch)、
  CPU/GPU で結果は同等(val_corr 0.845 一致)。推論は従来通り CPU、export 後の数値一致も確認(GPU 学習重みが Rust と一致)。
- **棋力は技巧2 Lv1 に依然 0-8**(avg_plies 76.8, depth 5.24)。度合いは着実に伸びる(val 2455/正解 3915=63%、生100万 56%→)が、Lv1 を超えるには未達。
  → **段階的にさらにデータを増やす**(効くと確認できたので次段で増量)。
- **採用モデル更新 = `model_cp_fd8_mirror_2m`**(生200万+mirror, GPU 学習)。

## 1. GPU 学習の整備 + CPU/GPU 速度

- 環境: WSL2, RTX 4060 Ti(16GB), driver 591.86。`/usr/lib/wsl/lib/nvidia-smi` で確認。
- セットアップ(再現): `python3 -m venv venv-gpu && venv-gpu/bin/pip install torch==2.12.1 numpy cshogi`(PyPI 既定 torch は CUDA 同梱=`+cu130`)。
  `torch.cuda.is_available()=True`, device=RTX 4060 Ti。
- `train_nnue.py` に `--device {auto,cpu,cuda}`(既定 auto=cuda があれば cuda)。モデル・バッチを device へ。**mirror 反転は CPU で(安いgather)→ device 転送**。推論(Rust)は常に CPU。
- **速度実測**(cache_2m 1.95M, batch1024, fd8+mirror+cp, 同一構成):

  | device | epoch 時間 | 備考 |
  |---|---:|---|
  | CPU(14スレッド) | ~18s | 14700KF |
  | **GPU(4060 Ti)** | **~3.6s** | **約5倍速** |

  val 指標は CPU/GPU 一致(val_corr 0.845, |pred|_dec 1571 vs 1566 = FP 差)。GPU は学習を 5倍に高速化し、結果は同等。
  → データ増量・再学習サイクルを GPU で速く回せる(fd8 でも 5倍。大きいモデル/バッチなら更に効く見込み)。

## 2. 新規教師データ生成(14700KF, CPU, 並列)

- `gen-teacher`(Task 016 基盤)で新規生100万: `--workers 12 --depth 8 --random-plies 6 --max-plies 256 --seed 20270101`(既存と depth/条件は同一、**seed だけ変え多様性確保**)。
- 結果: **1,000,000 局面 / 100 shard / 失敗0 / 12.57分**(5.40× 並列速度向上)。crash 93(全て restart で復旧, timeouts 0)。eval ±31794。
- 検証: cshogi 読み戻し ok(sample sfen is_ok)、shard/manifest 健全、eval 範囲妥当。出力 `~/binbo-shogi-data/teacher/run1m_b`(非コミット, 199MB)。
- 収集は CPU 仕事(技巧2 は CPU エンジン、GPU は収集に無効)。並列で加速。Xeon は nocca 本走中のため 14700KF 単独。

## 3. 計400万(生200万+mirror)での学習(GPU)

- データ: cache_2m = run1m の train(95万)+ run1m_b 全部(100万)= **train 195万**、**val = run1m の held-out 5万(Tasks 035〜038 と同一=直接比較可)**。
  ペア漏れなし(mirror は train 内のみ・val 非反転、run1m_b は run1m val と独立対局)。
- 構成は fd8+mirror+cp回帰(clamp±5000, MSE)据え置き。GPU で 16 epochs(best @ep5)。**量だけ増やす変数管理**。

## 4. 効果(主指標 = train/val ギャップ + val 度合い)

`pred_dispersion`(decisive |target|≥1500、val=run1m val):

| モデル | train 予測\|cp\| | **val 予測\|cp\|** | gap | val_corr | decisive_sign |
|---|---:|---:|---:|---:|---:|
| fd8+mirror(生100万) | 2994 | 2185 | 809 | 0.80 | 0.83 |
| **fd8+mirror(生200万)** | 2996 | **2455** | **541(−33%)** | **0.84** | **0.85** |

- **val が 2185→2455 と上がって gap が縮んだ**(train 不変)。これは「より多いデータで暗記が難しくなり汎化が上がった」教科書的な良い縮み方で、
  **dropout(038)の「両方下がる=度合い圧縮」とは質が逆**。情報追加(データ量)は mirror(036)と同様に効く、が裏付けられた。
- val 度合いは正解 3915 の **63%**(生100万 56% → 生200万 63%)。段階的に前進。
- 代表局面(`nnue-diag`): 飛車得 静的 1703/探索 1253(技巧2 2635)、中盤 1388。静的分解能 12.5×(curated, marginal が小さく出たぶん高め)。符号は維持。

## 5. 棋力・確認

- 棋力(`nnue-vs-gikou` Lv1, 1秒, 8局): **0-8**(avg_plies 76.8, depth 5.24)。度合いは伸びたが Lv1 超えには未達。
- スケール/数値一致: GPU 学習重みを export → Rust `nnue-eval` と Python eval が一致(`verify_nnue_rust` **2.167e-06** < 1e-3, ok)。GPU 学習でも推論一致。fd8 は Rust 無改修。
- 回帰: `cargo test` 16/16、perft 1〜5 完全一致、`selfplay` 完走、対 Random **20勝0敗**。コード変更は `train_nnue.py` の `--device` のみ(推論不変)。

## 所見と次段

- **情報追加(データ量↑)は汎化を正しく伸ばす**(val 度合い +12%, gap −33%, train 不変)。正則化(圧縮)とは違う良い効き方。
  これで「効くのは情報を足す手(mirror, データ量)」が二例で確定。
- **棋力 Lv1 はまだ 0-8** だが度合いは正解の 63% まで来た。効くと分かったので**次段は段階的にさらに増量**(生400万〜・mirror で計800万〜)。
  GPU 整備済みで再学習は速い(5倍)。データ収集は 14700KF 並列で 1M/12分。
- 主指標は引き続き `pred_dispersion` の val 度合い + gap(良い縮み方=val が上がるか)。質(教師 depth↑)は別変数で将来。

## 成果物
- コード(コミット): `trainer/train_nnue.py`(`--device` GPU 対応、mirror は CPU→device 転送)。GPU venv は `pip install torch numpy cshogi`(再現手順を本レポートに記載)。
- データ(非コミット): `~/binbo-shogi-data/teacher/run1m_b`(新規生100万)。採用重み `~/binbo-shogi-data/models/model_cp_fd8_mirror_2m`。
- 速度・分布の生値は本レポート。cache_2m/cache_b は /tmp(再生成可)。

## スコープ外(次段)
- 教師の質向上(depth↑)= 別変数。さらなる大規模化(まず計400万で効果確認、効けば次段)。構造・目標・正則化は据え置き(038 で正則化は打ち止め)。
