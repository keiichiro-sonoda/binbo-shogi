# Task 142 点検 (Codex, 2026-10-19)

**総合: 採用を支持。ただし、尺度の断定と開局の独立性に留保が必要。** 変換・実行条件・最終重みの出力は整合し、既存 TSV の再集計でも改善は残る。「ラベルの質が律速」は有力な仮説にとどまり、この比較だけでは原因を分離できない。

既存ファイルは変更せず、本報告だけを追加。永続データは読み取りのみ。GPU・エンジン・学習・対局は実行していない。Python は `scripts/env.sh` の `VENV_PY` を使い、OMP/OpenBLAS/MKL/NumExpr を各 2 スレッド以下、torch intra-op=2 / inter-op=1、CUDA 非表示、bytecode 保存なし。最大の走査も数秒程度。

## 1. 変換・視点・尺度

**OK: PSV の読み取り、手番、77 列の順序、保存ラベル。**

- 実ファイルは 289,626,920 B / 40 = **7,240,673 行**。cshogi の dtype は `sfen:uint8[32], score:int16, move:uint16, gamePly:uint16, game_result:int8, padding:uint8`。PSV と hcp は異なる符号なので、`Board.set_psfen()` の使用が正しい。SHA256 は `7590e12897e82633a2be727045df60ad8e34b0115d410804d0bbf55e1351b26a` で事前指定と一致。
- 元の gate と同じ `default_rng(20271420).choice(n, 2000, replace=False)` で再検査。保存行と再計算した **77 列全体、復元 SFEN の盤面・手番・持駒、own39 / opp38 の配置、すべて 2,000/2,000 一致**。先手番 1,012 / 後手番 988。`own[38] = VIEW_DIM + turn`、opp は逆向き・所有者反転。元の SFEN 復元 gate は主に own と相手玉を検査するため、それ単独では opp 全列の正しさまでは保証しない。今回の全列比較で補った。
- `labels_public/train_dlshogi_evals.npy` と PSV `score` は **全行一致**。val ラベルは Task 129 の val と完全一致。
- 符号の経験的検査: 各データから seed=142 で無作為 12,000 行。盤上・持駒の材得を手番側に揃えた。歩100、香300、桂320、銀500、金550、角800、飛1000、と金等550、馬1100、龍1300、玉0。持駒の順 `P,L,N,S,G,B,R` に注意。

| データ | 先手番 n / 材得と生ラベルの相関 | 後手番 n / 同相関 | 全体: 生 / ±2000 clamp 後の相関 |
|---|---|---|---|
| 公開 PSV | 5,899 / **+0.625** | 6,101 / **+0.602** | +0.614 / +0.690 |
| 自前 qs129 + Task129 labels | 5,928 / **+0.859** | 6,072 / **+0.858** | +0.859 / +0.779 |

両手番で正相関となり、先手固定視点や全体の符号逆転は支持されない。局面・教師が異なるので、相関の大小は教師の質や単位の一致の証明にはならない。

**unclear: 「同じ尺度の cp」という厳密な断定。** 自前は `t099_search_label.value_to_cp()` の `1200*atanh(v)` (保存時 ±5000)。`sigmoid(cp/600)=(1+tanh(cp/1200))/2` は代数的には一致するが、公開ファイルの生成時の数値較正を保証しない。[公開カード](https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9) は水匠5 depth9・qsearch PV 末端への置換を明記する一方、score の較正係数は明記しない。[やねうら王の生成コードの例](https://github.com/yaneurao/YaneuraOu/blob/cf6bb36f/source/learn/learner.cpp) は内部評価値を保存する。ローカルの `source/usi.cpp:1106` は内部値を `100/90` 倍して表示 cp にするので、PSV の生値と USI 表示 cp も区別すべき。生成版を固定した変換仕様、または同じ局面での較正が未確認であり、今回これを確かめるエンジン実行は行わない。**符号は実測で OK、600 の慣用尺度との互換性は妥当、厳密な同一較正は未証明**と読む。

**OK (微小な異常あり): 極値と clamp。** 全件の範囲は **−32001〜+31999**。`|score|>=30000` は **614,525 行 (8.49%)**、`|score|>=2000` は **2,097,187 行 (28.96%)**。trainer は float32 化後に **両符号を ±2000 に clamp** し、Huber(delta=600) で cp 回帰するため、詰み距離は捨て、極端な値が損失を支配しない。−32001 が 1 行 (index 2,805,901、王手なし・合法手49) あり、現行やねうら王の `−VALUE_INFINITE` 相当なので通常の詰み値と同一視しない。1/724万行の異常で、同じ clamp を受ける。

## 2. 実行した学習と出力

**OK: 事前コードと実ログ・JSON・成果物が整合。** 実コマンドは `logs/task142_20261019-101849.log:95` (dirty=0)。重要引数は次のとおり。

```text
trainer.train_nnue caches/public_suisho5_val7m
--qat --qat-tie --feature-dim 256 --hidden-dim 32 --batch 1024 --scale-cp 600
--factorize --augment-mirror --mirror-cache caches/shared_8m_mirror_perm.npy
--fast --checkpoint-minutes 10 --resume --lr 1e-5 --epochs 2
--init-weights models/task129_student_64m_qsrel_e12/nnue_best.pt
--eval-npy work/task142/labels_public --limit-train 7240673
--compact-cache caches/public_suisho5_val7m_compact
```

上記は `$DATA_ROOT/` を省略した表記。`task142_finetune.log` は warm-start を記録し、resume による state 復帰の記録はない。Adam、weight_decay=0、一定 lr、公開データだけの 2 周、各 **64.64 / 64.36 秒**。mirror は学習時に確率 0.5、ラベルの符号は変えない。実際の permutation は shared 用の `shared_8m_mirror_perm_shared.npy`。`model_config.json` の `canonical_shared:false` は CLI フラグ値で、`--qat` が内部で shared=True にするためレイアウト不一致ではない。

- 自前 val loss は **41,086.07 (初期) → 97,857.91 (ep1) → 100,352.48 (ep2)**。best は ep1 だが、実際の export コマンド (`同ログ:98`) は事前指定どおり **`nnue_final.pt` (ep2)**。trainer 末尾の `weights=...nnue_best.pt` 表示だけで判断すると誤る。
- final checkpoint を量子化した全 dense パラメータと、無作為 2,000 行の embedding を保存済み final BNNQ と比較し、**不一致0** (Factorizer の fold と tie を含む)。既存 0cp gate も二系統とも **32/32 完全一致**。実成果物の SHA256 は nn.bin `651ff9275a2fa416f195697270a1191967349d1601950f991b9bf522b9c8cece`、BNNQ `0978120c14a36ba8f4f7fa760c94e31fa8b74e93cc21cc6d2a90648869ce4c4a` で採用記録と一致。
- compact manifest の train/val は **7,240,673 / 50,000 行、不一致0、source/decoded SHA 一致**。今回 source_stamp の resolved path・size・mtime_ns・先頭/末尾64MiB hash を両 split とも再計算し一致。compact の無作為 2,000 行再復号も全77列一致。全 compact を今回再検査したわけではない。val の参照先は `caches/shared_64m/val_indices.npy`。学習ラベル全件照合も通るため、古い compact と別ラベルの取り違えを示す証拠はない。

初回 の処理は mix キャッシュの末尾コピーで失敗したが、公開変換 gate 通過後・学習前。修正版で compact を構築し学習した記録と整合する。

## 3. 漏洩・物差しへの適応

**problem (実害は微小): 自前 val との重複が1局面。開局の直接漏洩は検出なし。** 公開 train **全7,240,673行**を、自前 val **全50,000行**および4組の **228種類の開局**と照合。各77列の uint64 hash で候補を絞り、候補は全列の完全一致で確認した。左右反転も half ごとに再ソートして照合した。

| 対象 | 完全一致した公開行数 |
|---|---:|
| 自前 val | **1** (valの0.002%) |
| 自前 val の左右反転 | 0 |
| std/conf135/conf138/conf142 の6手後の開局 | 0 |
| 同開局の左右反転 | 0 |

一致は public index **709,164** / val index **20,450** (0始まり)、公開 score **+136** / 自前 label **−31**。SFEN は `lr5nl/3gk1g2/2nsppsp1/p1pp2p1p/7P1/PPP2PP1P/1SNPPSN2/2G1KG3/L6RL w Bbp 1`。val の完全独立性は失われているが、1行で対局 Elo の改善を説明する規模ではなく、finetune は val 最良周の選択もしていない。公開側の「validation」を学習に使うこと自体は、その validation の損失を独立評価に使わなければ問題ない。

ここで除外できたのは**保存された盤面・手番・持駒の一致とその左右反転**。対局途中の全局面、類似局面、公開データとアンカー自身の学習集合の重複は未調査。6手ランダム開局は学習局面と全く別の分布を保証しない。

**unclear: tanuki- との系統的な相性はありうる。ただし「tanuki- だけへの試験適応」は支持が弱い。** 既存の `work/task139/labels.tsv` 2,985局面に Task129 best / Task142 final の QAT forward を CPU で実行。両アンカーがcpで `|tanuki|<=1500` の **2,742局面・216対局**を使用 (既存アンカー値は20万ノード探索、生徒は静的値なので記述的比較)。

| 相手の保存評価値 | 相関 Task129→142 | MAE Task129→142 (cp) | RMSE Task129→142 (cp) |
|---|---|---|---|
| tanuki- | **0.413→0.515** | 313.6→283.7 | 446.2→432.7 |
| AobaNNUE | **0.378→0.462** | 451.2→397.9 | 617.0→567.6 |

生徒の平均変化は **+161.4cp**、平均絶対変化208.4cp。相関は両者で改善し、MAE/RMSE の改善は AobaNNUE にもある。3組の対tanuki +48.5 / 対Aoba +12.2 は相性の可能性を残すが、fresh conf142 では Aoba も **+41.4 [+8.8,+73.9]**。特定アンカーの未知局面への汎化と普遍的棋力は同義ではなく、教師・アンカーの学習系譜が独立だとも断言しない。公開データの局面分布・ラベル・最適化経路を同時に変えた比較なので、改善原因を「高品質ラベルだけ」とは確定できない。

## 4. 統計と `std_frame.delta_sets`

**OK: 候補と基準の対応、点推定と報告された区間の再現。** 4組×2アンカー×両モデルの **12,800局**を読み取り、各ファイル100局・先後対・開局seed・開局手列の対応を確認。失敗理由0。`delta_sets` の連結順 (set内の行順を保ち、tanuki群→aoba群) は両モデルで同じ。setごとに game 番号が戻るが、`paired_series_delta` は番号で join せず隣接2行の score 平均を使うため、この番号重複によるペアの誤結合はない。再計算は **+29.4821 [+15.4398,+43.4603] / fresh +47.3403 [+23.4943,+71.3823]** で完全再現した。

**problem: チャンク間の開局seedの重なりと、アンカー間で共通の開局を独立に扱う bootstrap。** `src/main.rs:2195` は `opening_seed = seed + game/2 + 定数`。チャンクseedを1ずつ増やして各50開局を使うので、8チャンク合計400組のうち**異なる開局は57種類だけ**。43開局は8回ずつ、残り14開局は1〜7回ずつ各2種類。両アンカーも同じ57種類を共有する。したがって pooled の `n_pairs=2400` は **171種類の開局**、fresh の800は **57種類**。4組間の開局重複は0 (計228手列・228盤面)。重複した開局でも勝敗は変動するため、単純に局数を約1/7に割って誤差を推定するのも適切でない。

感度分析として、**同じ6手の手列の全反復・両アンカー・両モデルを一つの cluster** として再抽出し、各clusterの score 合計と行数から元の重複重みを保つ勝率→Elo差を計算した (percentile bootstrap 200,000回、seed=20270930)。

| 比較 | 元の点推定 | 開局clusterで再計算した95%区間 |
|---|---:|---|
| 3組 pooled (171開局) | +29.5 | **[+15.4,+43.7]** |
| 同 tanuki- | +48.5 | [+30.2,+67.0] |
| 同 AobaNNUE | +12.2 | [−8.2,+32.6] |
| fresh pooled (57開局) | +47.3 | **[+24.3,+70.1]** |
| fresh AobaNNUE | +41.4 | [+6.1,+76.7] |

さらに各開局の反復を平均して開局を等重みにすると、pooled **+34.3 [+18.1,+50.5]**、fresh **+55.8 [+31.5,+80.1]**。今回の結論はこの扱いにも頑健。元CIの一般的な妥当性や「2400個の独立開局」は主張できないが、**重複を考慮しても今回の有意な改善は消えない**。今後はチャンクseedの範囲を分離し、実開局の重複をgateに加えるべき (本レビューでは修正・対局なし)。

**最終判定: 採用判断を覆す不整合は検出せず。** 変換・学習・成果物はOK。val重複1局面と開局seed設計は問題として記録し、厳密な尺度一致・教師品質が改善原因という断定は保留する。裏付けられる結論は「この公開データで仕上げた最終evalは、保存済み2アンカー・新旧4組のランダム6手開局において前任より強い」。
