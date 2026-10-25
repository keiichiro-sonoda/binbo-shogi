# Task 054: DL系教師(dlshogi)への第一歩 — 動作確認・re-score・コスト

目的: 評価品質が律速(速度実測: 生徒は技巧Lv3/Lv4 を約1000倍ノード読んでも負ける=1ノードの評価が粗い)。
本線は**先生の質**。技巧2 を DL系(dlshogi)に変えて再蒸留する、その**第一歩=地固め**(動作・コスト・小規模 re-score・計画)。

## Step 0 — モデルとライセンス(人間判断ゲート, 完了)

- **ソース GPL-3.0。モデル `model-dr2_exhi.onnx` は著作権=山岡忠夫氏、私的使用無償**(`LICENSE.txt`)。
  禁止: 定跡生成 / 直接対局 / **本モデルへの追加学習** / **パラメータ転用** / 改変・再配布。大会向けは「学習用棋譜生成」「勝率測定」のみ許可。
- **ユーザー(Keiichiro)が dr2_exhi の使用に同意**(私的使用・非コミット運用)。本タスクの利用 = **モデルを動かして cp ラベルを生成し、
  別アーキの生徒(KP NNUE)を蒸留** = clause 2(私的使用)+ clause 3「学習用棋譜生成」に該当。禁止事項(追加学習/パラメータ転用/
  直接対局/定跡/再配布)に当たらない(生徒は別モデル・出力から学ぶ・重みは非コミット)。既存の技巧2(GPL)蒸留と同じ規律。
- 配置: `~/binbo-shogi-data/engines/dlshogi/model-dr2_exhi.onnx`(57MB, 永続・**非コミット**)。

## Step 1 — 4060 Ti で動かす(最軽量パス, 完了)

- **cshogi.dlshogi(FEATURES1=62 + FEATURES2=57 平面)+ onnxruntime-gpu 1.23.2** で推論(dlshogi 本体ビルド/TensorRT 不要)。
- モデル I/O: `input1(B,62,9,9) + input2(B,57,9,9)` → `output_value(B,1) + output_policy(B,2187)`。
  **value は既に勝率**(手番側, ∈[0,1]。sigmoid 不要): startpos 0.542(先手微有利=妥当)、優勢=1.0、劣勢=0.0。
- **GPU 化**: onnxruntime は CUDA 12 + cuDNN 9 を要求。系は CUDA 13(torch cu130)で当初 CPU fallback →
  `pip install nvidia-{cuda-runtime,cublas,cudnn,cufft,curand}-cu12` + `LD_LIBRARY_PATH` で **CUDAExecutionProvider 起動**。
- **速度**: batch256 で **5287 pos/sec**(batch1024/2048 はむしろ遅い=**ボトルネックは Python の特徴抽出ループ**、GPU は余裕)。
  **VRAM 2.8GB / 16GB**(余裕)。CPU は 382 pos/sec(参考)。

## Step 2 — 勝率→cp 変換 + 小規模 re-score(完了)

- **変換**: dlshogi value v(手番側勝率)→ `cp = 600·logit(v) = 600·ln(v/(1−v))`、**clamp ±5000**(生徒スケール: cp/600≈logit(勝率), Task 051)。
  既存 cp pipeline と整合(forward×600, clamp5000)。Gikou cp も手番側 → 直接比較可。
- **re-score(run_wdl_2m 5000局面)**: dlshogi cp vs Gikou depth8 cp(同一局面):

| 指標 | 値 | 解釈 |
|---|---|---|
| 符号一致(全) | **0.845** | 誰が良いかは 85% 一致(向き正しい=逆なら0.15) |
| 符号一致(decisive |gikou|≥500) | **0.975** | 決着局面はほぼ一致 |
| 相関(cp, clamp) | **0.927** | 強い相関=暴れていない |
| dlshogi cp std | **2510**(Gikou 1963) | **dlshogi の方が鋭い**(自信を持って差をつける) |
| |dlshogi−gikou| cp | 中央値 **456**, p90 1852 | ~半分の局面で半香車級の差=意味ある違い |

- **示唆**: dlshogi は Gikou と整合しつつ**より鋭く・別の判断**をする。決着局面は一致、**不明瞭な局面で判断が分かれる**
  (そこが「より真理に忠実な先生」の効きどころ候補)。蒸留教師として妥当に機能。

## Step 3 — コスト見積もり + KD-001 計画

- **8M re-score コスト ≈ 25-30 分**(GPU, ~4400-5300 pos/sec)。VRAM 2.8GB=余裕。**自宅マシンで完全に実用的**。
  - 更なる高速化: 特徴抽出が律速 → multiprocessing で並列抽出すれば GPU を埋められる(KD-001 で検討、必須でない)。
- **KD-001(次タスク, 学習は別)= 教師だけ dlshogi に差し替える純粋切り分け**:
  1. 既存 8M 教師(run1m + extras)を**prepare_data と同じ順**で全 re-score → `dlshogi_evals.npy`(int16, cp, 整列)。~25-30分。
  2. `train_nnue` に cp ターゲット差し替えフラグ(`--eval-npy <path>`, 小改修)を足し、**既存キャッシュ・生徒・探索・cp+mirror は固定**で再学習。
  3. 棋力比較(技巧2 Lv1-Lv4, 経済的に Lv3/Lv4 中心): dlshogi 教師の生徒 vs 現 HalfKP 8M。**「先生の質が律速か」を純粋に切り分け**。
  - 静的 value(探索なし)で測る(速い・生徒も静的eval)。効かなければ searched dlshogi(MCTS value)が次段候補。

## 成果物・スクリプト
- `trainer/dlshogi_rescore.py`(cshogi 特徴 → onnxruntime value → cp、Gikou cp と比較、`--out` で cp 保存)。
- 実行例: `LD_LIBRARY_PATH=<venv nvidia cu12 libs> python -m trainer.dlshogi_rescore <hcpe> --limit 5000 --batch 256`。
- モデル(.onnx)・cu12 ライブラリは**非コミット**(ライセンス・サイズ)。永続領域に配置。

## 結論(第一歩)
dlshogi(dr2_exhi)は **4060 Ti で動く**(GPU 推論 5000 pos/sec, VRAM 2.8GB)。勝率→cp 変換は生徒スケールに整合。
小規模 re-score で **dlshogi は Gikou と整合しつつ鋭い別判断**(符号一致 decisive 0.975/相関 0.927/中央差 456cp)。
**8M re-score ~25-30分=実用的**。次は **KD-001(教師のみ差し替え再蒸留)** で、評価品質=先生の質が Lv3/Lv4 天井の真因かを切り分ける。
