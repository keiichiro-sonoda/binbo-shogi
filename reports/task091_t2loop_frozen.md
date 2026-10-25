# Task 091 — T2-loop: value 凍結 Transformer 自己対局ループ

**目的**: 標準 WS-T(3.6M Transformer warm-start)から自己対局で方策へ新情報を入れ、内部棋力ではなく **NNUE 教師品質が WS-T 教師版を超えるか**で判定する。gen100 の value 圧縮(std 1471→698、教師 −145 elo)を、value の構造的凍結と教師品質ゲートの並走で防ぐ。

## 実装と事前固定した設計

- **凍結粒度 = 完全二塔**: WS-T を2コピーし、policy 塔だけを MCTS 訪問分布の CE で更新する。value 塔は optimizer/JIT 学習グラフから除外し、MCTS は policy 塔の logits と value 塔の値を読む。共有幹を方策勾配で動かす構成は採らない。
- **凍結オラクル**: value 塔の全 params/state の SHA-256 digest を checkpoint ごとに完全一致確認し、同一実行中の iter0 value 出力との差を監視する。GPU の別 XLA グラフ/別プロセス間では同一重みでも preflight で最大 ±1cp、本走2Mで最大4cpの丸め差が出た。rescore は候補/基準の value digest 完全一致を独立に確認し、数値差を監査値として記録したうえで WS-T 基準 value 配列をそのまま保存する。これによりミニ教師比較の value ラベルは byte-for-byte 同一になる。
- **方策教師化 = 合法 raw-policy argmax**: hcpe 各局面をネットで再採点し、合法手 mask 後の方策 argmax を 2187-class target とする。`bestMove16` へも復元して合法性を全件監査する。訪問分布 soft target は本タスクでは扱わない。
- **ミニ NNUE 教師試験**: Task 084 と同じ 2M cache、tied QAT HalfKP fd256/h32、cp MSE、scale 600、lr 1e-5、12 epoch、mirror なし。学習時だけ h32 から 2187-class policy head を分岐し、損失を `cp_MSE + 0.1 * 600^2 * policy_CE` とする。配布 evaluator は従来と同じ value 部だけで、両系統の差は policy target の出所だけ。
- **対照について正直に**: 080 比では Transformer 化と value 凍結の二変数変更。非凍結対照は gen100/084 が「内部上昇・value 圧縮・教師 −145」の機構を既に示したため省略する。

## preflight と本走条件

| 項目 | 実測 | 決定 |
|---|---:|---|
| dual checkpoint CPU smoke | policy 更新あり、value max delta **0**、digest 不変 | PASS |
| 極小 GPU loop / resume | 2 iter、checkpoint 再開、value monitor delta **0** | PASS |
| rescore 2,000局面 | 全手合法、value digest一致、候補/基準 value 保存配列 **完全一致**、policy change 検出 | PASS |
| policy-aux NNUE smoke | forward/backward/export 成功、有限 loss、重み更新 | PASS |
| sims32 / 48(dual、4局面 bench) | **35 / 55 ms per move = 1.57×** | 閾値1.45超のため **32** |
| batch256、131,072 states | 1269s/iter、110 states/s、VRAM観測7.57GiB、学習時4GiB確保警告 | 不採用 |
| batch128、65,536 states | **614s/iter、117 states/s**、OOMなし(selfplay) | selfplay **128** |
| training batch256 / 128 | 256 は非致命4GiB確保警告、128 は警告なし | training **128** |

本走は **sims32 / selfplay batch128 / training batch128 / 65,536 states per iter / lr1e-4**。基準停止点は **75 iter**、hard cap 150。旧 M4 の1 iter(131,072 states)の半量なので、iter 数の直接比較はしない。

## 並走ゲートと停止基準

- **監視**: 5 iter ごとに frozen digest、value corr/std、iter0 value 差、技巧2 top-1。毎 iter に policy CE、訪問分布 entropy、`CE-entropy`、訪問 argmax 一致率、states/s。
- **25 iter ごとの内部ゲート**: WS-T と 40局、両者 sims128、0.1秒相当、同一乱数 opening。内部 elo は健全性計器であり主判定にしない。
- **25 iter ごとの教師ゲート**: 2M rescore → policy-aux NNUE → 0cp gate → WS-T 教師版と 80局。value ラベルは完全同一、方策教師だけが変数。
- **早期停止**: 75 iter 以降、教師 elo と内部 elo がともに3ゲート連続で +10 elo の新高値を作らなければ停止。
- **延長**: 基準75 iter 後は教師 elo が直近ゲートで +10 の新高値を作り、次25 iterを含む予測総時間が24時間以内の場合だけ延長。hard cap 150。
- **最終判定**: 教師 elo が最大の世代を fresh seed・最大400局で WS-T 教師版と比較。`accept_h1` または 95% CI 下限 > 0 なら **PASS**。それ以外は **FAIL**。

## 実走

逐次パイプラインは `experiments/alphazero/t091_pipeline.py`、起動ラッパは `experiments/alphazero/t091_pipeline.sh`。`scripts/detached.sh` 経由で clean HEAD の `RUN_META` を記録する。checkpoint/log/model はクローンまたは `~/binbo-shogi-data` に置き、非コミット。

### 実走中の安全停止

- iter25 の2M rescoreで、別XLAグラフ間の同一value重みが最大4cpずれ、事前の1cp監査閾値で安全停止した。checkpointの frozen digest とループ内出力差0は維持されており、凍結漏れではない。
- 出力差の恣意的な閾値ではなく、候補/基準の value params/state digest 完全一致を主オラクルへ変更。教師用valueは従来どおり基準配列を厳密コピーする。追加実験ではなく監査の根拠強化であり、iter25 checkpointから再開する。

## 結果

### 凍結と方策学習の監視

75 iter、合計 **4,915,200 states** を生成した。安全停止前の初回実行と clean HEAD からの再開を合わせた実計算時間は約 **16.8時間**、再開パイプライン自身の計測は **12.14時間**。基準75 iter時点で教師新高値が無かったため、事前固定した延長条件に従い150 iterへは延長しなかった。

| iter | policy CE | visit entropy | visit argmax一致 | 技巧2 top-1(監視集合) | value corr | value cp std | value差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.2700 | 0.9784 | 0.7344 | — | — | — | — |
| 5 | 0.8396 | 0.6309 | 0.8183 | 0.4652 | 0.9357 | 1525.2 | 0 |
| 10 | 0.7017 | 0.5202 | 0.8418 | 0.4585 | 0.9357 | 1525.2 | 0 |
| 15 | 0.6310 | 0.4598 | 0.8582 | 0.4556 | 0.9357 | 1525.2 | 0 |
| 20 | 0.5722 | 0.4147 | 0.8675 | 0.4504 | 0.9357 | 1525.2 | 0 |
| 25 | 0.5074 | 0.3651 | 0.8828 | 0.4489 | 0.9357 | 1525.2 | 0 |
| 30 | 0.3910 | 0.2888 | 0.9225 | 0.4447 | 0.9357 | 1525.2 | 0 |
| 35 | 0.3852 | 0.2816 | 0.9223 | 0.4407 | 0.9357 | 1525.2 | 0 |
| 40 | 0.4531 | 0.3209 | 0.8996 | 0.4431 | 0.9357 | 1525.2 | 0 |
| 45 | 0.4562 | 0.3236 | 0.8950 | 0.4374 | 0.9357 | 1525.2 | 0 |
| 50 | 0.5321 | 0.3800 | 0.8717 | 0.4414 | 0.9357 | 1525.2 | 0 |
| 55 | 0.5491 | 0.3976 | 0.8721 | 0.4447 | 0.9357 | 1525.2 | 0 |
| 60 | 0.5530 | 0.3943 | 0.8708 | 0.4428 | 0.9357 | 1525.2 | 0 |
| 65 | 0.5014 | 0.3590 | 0.8863 | 0.4436 | 0.9357 | 1525.2 | 0 |
| 70 | 0.4919 | 0.3480 | 0.8860 | 0.4439 | 0.9357 | 1525.2 | 0 |
| 75 | 0.4348 | 0.3157 | 0.9044 | 0.4390 | 0.9357 | 1525.2 | 0 |

- value params/state digest は全 checkpoint で warm-start と完全一致し、同一ループ内の value 出力差も常に **0**。2M rescoreで保存した教師 value 配列も全世代で基準と完全一致した。**gen100 の value 圧縮は構造的に防げた**。
- 方策 CE は下がり、訪問 argmax 一致率は 0.734→0.904。したがって optimizer が何もしていない失敗ではない。
- 一方、固定教師局面での技巧2手 top-1 は WS-T の **0.4788** から iter25/50/75 で **0.4570/0.4536/0.4513**へ低下。方策 entropy は **1.544→0.643/0.642/0.599**、WS-T からの合法 argmax 変更率は **32.6%→34.4%→35.6%**。ループは方策を明確に変えたが、技巧2方策への一致を改善する方向ではなかった。

### 内部棋力と教師品質の並走カーブ

| iter | 内部 vs WS-T(W-D-L、40局) | 内部 elo(95% CI) | 教師 vs WS-T教師(W-L-D、80局) | 教師 elo(95% CI) |
|---:|---:|---:|---:|---:|
| 25 | 14-6-20 | -52.5 [-160.6, +46.1] | 48-32-0 | +70.4 [-5.1, +153.3] |
| 50 | 20-7-13 | +61.4 [-35.3, +168.9] | 40-40-0 | 0.0 [-77.4, +77.4] |
| 75 | 16-5-19 | -26.1 [-132.5, +75.5] | 40-40-0 | 0.0 [-77.4, +77.4] |

内部 elo は正負に振れ、全点の CI が0を含んだ。gen150/080 のような単調な内部上昇は再現せず、安定した内部改善の証拠は得られなかった。教師80局の iter25 +70.4 も CI が0を含む暫定値で、後続世代では消えた。内部カーブと教師カーブは同じ世代で連動せず、教師品質を最初から並走させる計器は機能した。

## 最終判定

教師80局で最大だった **iter25** を fresh seed で400局再検証した。

| 候補 | W-L-D | score | elo(95% CI) | SPRT |
|---|---:|---:|---:|---:|
| iter25教師 vs WS-T教師 | **198-198-4** | **50.0%** | **0.0 [-34.0, +34.0]** | continue、LLR -0.167 |

全400局でエラー0、候補/基準の平均 depth・nodes・time は同等、事前0cpゲートも32/32 bit-exact。対局条件や探索量の差ではなく、方策教師を置き換えた効果そのものが **0 elo** だった。

**判定: FAIL。ループ後ネットは NNUE 教師として WS-T を超えなかった。** iter25の80局 +70 eloは400局で完全に消え、主成功条件を満たさない。フル32M還流・champion再挑戦には進まない。

### 再現ログ

- 初回: `~/binbo-shogi-data/logs/t091_t2loop_20260714-165613.log`、`RUN_META dirty=0 exp_dirty=0`。
- 監査修正後の再開〜完了: `~/binbo-shogi-data/logs/t091_t2loop_resume1_20260715-002640.log`、`RUN_META dirty=0 exp_dirty=0`、末尾 `DETACHED_DONE rc=0`。
- ゲート機械可読値: `~/binbo-shogi-data/logs/t091_s32_b128_gates.json`。最終400局: `~/binbo-shogi-data/logs/t091_s32_b128_teacher_final.console.log`。
- checkpoint、2M rescore、NNUEモデル、対局TSVは規律どおりクローン側または `~/binbo-shogi-data` に置き、リポジトリへコミットしない。

## 解釈と申し送り

1. **value凍結は成功**: gen100 の教師劣化原因だった value 圧縮を完全に除去できた。二塔化とdigest監査は、value資産を守る方法として成立した。
2. **方策は変わったが、安定して強くならなかった**: CE、訪問一致、entropy、argmax変更率は学習を示す一方、内部 elo は `-52→+61→-26` と非単調で全CIが0を含む。したがって本結果だけから「内部だけ登ったのに教師へ転写しない第3の壁」とは断定しない。まずループ自体の安定改善が未達である。
3. **教師品質への転写は少なくとも確認できない**: valueを完全同一にした厳しいA/Bで最終0 elo。合法 policy argmaxを補助教師にした方式では、自己対局方策の変化はNNUE evaluatorの棋力向上へ転写されなかった。
4. **完全二塔の副作用候補**: value出力を厳密固定する代わりにpolicy幹をvalue表現から切り離したため、080の共有幹ループより改善が鈍った可能性がある。ただし080比は Transformer化も同時に違うため、原因を凍結だけに帰属しない。
5. **次の比較候補**: ループ路線を続けるなら、完全二塔に対し「共有幹を許しつつWS-T valueへの蒸留/replay正則化で出力を拘束する方式」を一変数比較する。別候補は、argmaxで捨てた情報を保持する訪問分布soft targetの教師化、またはsims48へ探索予算を振る比較。いずれも本タスクでは追加実行しない。

対照上の制約は事前記載どおりで、080比では **Transformer + 完全凍結** の二変数変更である。非凍結Transformer対照を省略したため、失敗を凍結単独の因果効果とは主張しない。
