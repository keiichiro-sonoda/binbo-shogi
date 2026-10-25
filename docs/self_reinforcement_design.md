# 自己強化ループ 設計(self-reinforcement loop)

技巧2蒸留の踏み台(champion = `task064_32m_lr1e-5`)を初期化点に、教師ラベルを **「自分の eval + 深い探索」** へ差し替えて
世代を回し、世代ごとに外部アンカーで健全性を確認しながら強くする。Task 064 で蒸留の情報限界(train-val gap 底 ~26k=技巧2 depth8
ラベルノイズが不動)が見え、Task 065 で champion@YaneuraOu の固定深さ cost を実測(depth12=1M/~0.44h、cp は飽和せず=深い探索が
静的 eval を書き換え続ける=蒸留すべき信号が残る)→ 本ループが次の本命。本書は Task 066 で確定した設計。**実装は最小限の新規
コードと既存パイプライン流用**(`selfplay_label.py` / `train_nnue --objective wdl` / `export→writer→0cp ゲート` / `yane-vs-yane` / `yane-vs-gikou`)。

## 全体像(1 世代の流れ)
```
champion(gen N) ──┐
                  ├─► [1] ラベル生成: champion@YaneuraOu@depth で自己対局 → cp(勝率)+WDL
                  │
                  ├─► [3] 学習: warm-start(gen N 重み)+ 新ラベル(+replay) → gen N+1 eval
                  │        (tied QAT fd256/h32, objective=wdl, λ 配合)
                  │
                  ├─► export→writer→[0cp ゲート] → gen N+1 を YaneuraOu eval に
                  │
                  └─► [4] 評価: gen-vs-gen SPRT(採用判定)+ [5] 外部アンカー(偏り監視)
                           勝てば昇格(gen N+1 が新 champion)、負け/偏りなら [6] 健全世代へ戻る
```

## 1. ラベル生成
- **champion eval を YaneuraOu に載せ**(EvalDir, 0cp ゲート済)、**自己対局**で各局面に **cp(side-to-move)+ WDL(side-to-move +1/0/−1)** を付与。
- 実装=`trainer/selfplay_label.py`(`kp.canonical_shared_row` で KP 変換=prepare_data と同一、`label_cost` 流の watchdog 付き USI driver で
  固定深さ探索を hard-bound)。出力は prepare_data 互換 cache(`train_indices/evals/wdl.npy`)。
- **ラベル深さ = depth 12 で開始**(065: 飽和点なし=コスト/品質のダイヤル、d12 で 1M≈0.44h@16w)。信号が薄ければ後で depth14 へ(**一変数で**)。
- 「数字の depth8」より **champion 探索での信号の質**が本質(エンジンが違えば depth の中身も違う)。champion@d12 が技巧2 depth8 を超えるかは
  **学習+SPRT で実証**する(val では測れない)。

## 2. 局面の多様性
- **最初は軽い散らし(`random_plies` のみ)**で開始。**作り込みすぎると偏りが出ず観察できない**ため(偏り検知が本タスクの主眼)。
- 偏りが出てから多様性を増やす(対局中のランダム手/次善手混ぜ、開幕局面の拡張)。各案は **一変数で**追加・評価する。

## 3. cp 勝率 / WDL のスケールと配合(ハイブリッド必須)
- **両成分とも勝率空間に揃える**: loss = λ·(cp→勝率回帰) + (1−λ)·(WDL回帰)。λ が比率を意味する。
- 既存実装そのまま流用: `train_nnue --objective wdl --wdl-lambda λ`(`target = λ·sigmoid(eval/scale) + (1−λ)·gameResult`, MSE on sigmoid)。
- **λ=0.5 で開始**(確定値でない)。**cp 直接 vs 勝率変換**の優劣、**λ の本格探索**は後続の一変数比較に分離(足場として勝率変換+λ0.5 を採用)。
- WDL の性質: 個々の局面ラベルは 1 局の勝敗(0/0.5/1)だが、**大量サンプルで似た局面をまたいで平均され勝率に収束**=自分の cp 偏りの錨。
  ただし WDL の質は **自己対局の質=今の棋力依存**(初期は質低)。

## 4. 学習(ウォームスタート + replay 混合)
- **warm-start**: 前世代の重みから継続(`train_nnue --init-weights <prev nnue_best.pt>`)。速く、癖を引き継ぎやすい。
- **データ = 新規自己対局 + 32M champion キャッシュ replay の混合**(`selfplay_label.py --mix-replay shared_32m --replay-n`)。
  → **catastrophic forgetting を防ぎ「新ラベルの効果」を分離**(新規のみだと 32M を忘れて弱くなり、偏りと忘却が混ざる)。混合比は当面 1:1。
- arch 固定(tied QAT fd256/h32)=世代間で writer/engine が一貫。lr は低め(champion=1e-5 系)、epoch は少なめ(warm-start fine-tune)。
- **判定は val ではなくアンカー対戦**(val は技巧2ラベルへの当てはめ=cp 寄りを不当に勝たせる)。val は参考のみ。

## 5. 採用・棄却・停止 + 外部アンカー監視(偏り検知の本体)
- **採用判定 = 世代間 SPRT**: `yane-vs-yane`(探索固定・eval のみ変数・**両側 0cp ゲート**, movetime 0.1s)で gen N+1 が gen N に勝てば**昇格**。
- **棄却**: 負け世代は捨て、**健全世代から別設定で再挑戦**(深さ/λ/多様性を一変数で変える)。
- **外部アンカー(固定参照)で自家中毒/偏りを検知**:
  - **主アンカー = 技巧2**: 新規 `yane-vs-gikou`(champion を YaneuraOu フル強度で技巧2と対戦=運用一致・最鋭敏)。
    ※ 技巧2 は探索 depth ≥ ~8 で warm-state EOF クラッシュが頻発(gen-teacher 既知, ~80%@depth10)。**安定する DepthLimit 6 を固定参照**に使う
    (depth6 は数 ms で完了・0 エラー)。深い技巧2を使いたければ restart-and-continue の頑健化が要(follow-up)。
  - **副アンカー = tanuki**(GPL smoke eval)を `yane-vs-yane` で(両側 YaneuraOu フル強度=eval のみ)。自作旧版 (Task 055 時点の自前探索) は要ビルド=将来。
  - **偏りのシグネチャ = 前世代に勝つがアンカーに負ける/落ちる**(自分にだけ都合よく見える=自家中毒)。
- **監視頻度**: リハーサル/初期は**毎世代 密に**(偏りの立ち上がりを捉える)。本走移行後は**間引き + 異常時エスカレーション**。
- **停止**: アンカー勝率が頭打ち/低下したら停止(蒸留の天井 or 自家中毒)。

## 6. 戻れる設計(reversibility)
- 各世代の **eval(.pt/.bnnq/nn.bin)・ラベル cache・設定**を保存(`~/binbo-shogi-data/` 永続・非コミット)。
- 偏り検知時に**健全世代へ戻れる**ようにする(NOCCA のレジューム思想と同様)。
- champion(現行最良)は常に明示(STATUS に sha256)。昇格は SPRT 勝利時のみ=後退しない。

## 7. tie 累積の監視
- 毎世代 学習→**tied QAT**→writer→engine を反復する。tie は YaneuraOu 形式へ畳む際に **金/成駒(52488 特徴)と side を代表へ束ねる**=
  情報を落とす。**単発の代償は ≈0**(Task 059/062 で確認)だが、**世代をまたぐ累積影響は未確認**。
- 監視項目として置く: 世代ごとに tie 前後の val/engine 0cp を確認、累積で品質が劣化しないか(アンカー勝率の長期トレンドで監視)。

## 確定パラメータ(リハーサル=Task 066, 確定値でないものは「開始値」)
| 項目 | 値 | 備考 |
|---|---|---|
| 初期化点 | `task064_32m_lr1e-5` | 0cp 済, sha `3c25685e…` |
| ラベル深さ | depth 12 | 開始値(065)。飽和なし=ダイヤル |
| cp 表現 / 配合 | 勝率変換, λ=0.5 | 開始値。`--objective wdl --wdl-lambda 0.5` |
| 学習 | warm-start + 32M replay 1:1 | forgetting 回避 |
| 採用判定 | gen-vs-gen SPRT(yane-vs-yane, 0cp, 0.1s) | val でない |
| 主アンカー | 技巧2 = `yane-vs-gikou` DepthLimit 6 | depth≥8 はクラッシュ |
| 副アンカー | tanuki = `yane-vs-yane`(`$YANE_EVAL`) | GPL, 合法 |

## 規律
- 一度に一変数(深さ/λ/cp表現/多様性の最適化は機構が回ってから個別タスクで)。GPL 遵守(技巧2/tanuki=合法アンカー、dlshogi 封印)。
- 長時間ジョブは `scripts/detached.sh`(セッション境界耐性)+ `env PYTHONPATH=<repo>` + `nice`。**057-059 bridge 不触**。
- 破壊的操作・host 設定は人間。STATUS/BACKLOG を区切りで更新、push は人間承認後。
