# Task 073 — (c) AlphaZero/MCTS 系への転換: 徹底調査(実装なし・方向性の探索)

**目的**: 自己強化アーク 066-072 で「NNUE 蒸留 + αβ評価値蒸留 + 技巧2出発点」は技巧2@d8 が天井と実証。天井を破る唯一の理論的経路 = 探索の種類を **MCTS + 方策改善(AlphaZero/dlshogi 流)= (c)** へ変える。本レポートは **実装せず**、web 一次情報で「**単一 RTX 4060 Ti 16GB + i9-14700KF + WSL2** で現実的に回る (c) 構成」を洗い出す。**表記規約**: 【事実(出典)】と【推測】を明示分離。

---
## Item 1 — 将棋の AlphaZero 流実装事例

- 【事実】**dlshogi**(TadaoYamaoka)= 将棋界最強クラスの deep RL 将棋 AI。構成: `cppshogi`(盤面/合法手生成, **Apery 由来**)+ `dlshogi`(Python NN 学習)+ `selfplay`(MCTS 自己対局)+ USI エンジン + OnnxRuntime 版。[github](https://github.com/TadaoYamaoka/DeepLearningShogi)
- 【事実】**ネット構成**: 入力 = **104 チャネル × 9×9 の盤面平面**(位置依存特徴)、**ResNet ブロック + policy ヘッド + value ヘッド**(AlphaZero 構造の小型版)。**行動空間 = 81×27 = 2187**(移動先81マス × 27方向/駒種)。policy net の特徴抽出層を value net 初期化に転移可。[Pgx paper](https://arxiv.org/abs/2303.17503) が dlshogi の観測/行動設計を踏襲と明記。
- 【事実】**MCTS**: policy+value を GPU でバッチ計算。CPU 論理コア数を超えるスレッドで並列探索し、計算要求をキューに積んで GPU がバッチ処理(GPU 時間が支配的)。[Yamaoka blog その45](https://tadaoyamaoka.hatenablog.com/entry/2017/11/14/002305)
- 【事実】**AobaZero**(kobanium)= AlphaZero 将棋実験の**追試(ゼロ知識)**プロジェクト。ゲームルール以外の知識をほぼ持たず自己対局で elmo 超えを目指す。分散ボランティア計算方式。[github](https://github.com/kobanium/aobazero)
- 【事実】**python-dlshogi / pydlshogi**(Yamaoka の書籍「将棋AIで学ぶディープラーニング」コード)= **教育規模の最小 policy/value/MCTS 将棋実装**(個人開発規模の一次事例)。[github](https://github.com/TadaoYamaoka/python-dlshogi)

## Item 2 — 小計算資源での AlphaZero 再現・効率化

- 【事実・スケールの現実】AlphaZero 原典 = **5000 TPUv1(自己対局)+ 64 TPUv2(学習)、700k step**、ランダム初期化。[AlphaZero paper](https://arxiv.org/abs/1712.01815)
- 【事実・将棋のゼロ知識コスト】AobaZero: **高性能デスクトップ1台でゼロ知識将棋 AZ 追試は ~100年の計算時間**を要する(だから分散)。[aobazero](https://github.com/kobanium/aobazero) → **単機ゼロ知識将棋 AZ はスクラッチでは非現実的**。
- 【事実・dlshogi の実 RL コスト】マルチ GPU 自己対局で **~17.9 局面/秒、5M 局面/サイクル = 3.2日/サイクル**。開発者は**自宅 PC を 2年以上**回して RL。[Yamaoka その14](https://tadaoyamaoka.hatenablog.com/entry/2018/03/13/215608) [その26](https://tadaoyamaoka.hatenablog.com/entry/2019/02/04/215958)
- 【事実・効率化①: KataGo】非ドメイン+ドメイン改良で **AlphaZero 比 ~50× の計算削減**(囲碁)。「**数個の強い GPU で数日、個人でもゼロから高段アマ強度まで訓練可能**(19×19)」と明言。[Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) → 効率化の中核ツールボックス(playout cap randomization, 補助ターゲット等)。囲碁だが多くは非ドメインで移植可。
- 【事実・効率化②: Pgx】**JAX で GPU ベクトル化した将棋を含むゲーム環境**。既存 Python RL ライブラリ比 **10-100× 高速**、単一 A100 で AZ 学習(9×9 囲碁)を実演。**将棋環境 + AlphaZero 例が同梱**。[Pgx paper](https://arxiv.org/abs/2303.17503) [github sotetsuk/pgx](https://github.com/sotetsuk/pgx) → 自己対局スループットの律速を GPU 並列化で桁で改善しうる。
- 【事実・効率化③: Gumbel AlphaZero/MuZero】**少シミュレーション(2回でも)で方策改善を保証**。標準 AZ は小 sim 予算で方策改善が保証されない弱点があり、Gumbel が root の行動サンプリングを工夫して解決。同計算で少 sim でも AZ 相当に到達。[Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) → **MCTS の1手あたり sim 数を大幅削減=自己対局コスト直撃の削減**。
- 【事実・最小 OSS 枠組み】
  - **MiniZero**(rlglab, IEEE ToG)= AlphaZero/MuZero/**Gumbel AZ/MuZero** 統合訓練枠組み(囲碁/オセロ/Atari、拡張可)。[github](https://github.com/rlglab/minizero) [paper](https://arxiv.org/abs/2310.11305)
  - **AlphaZero-Edu** = **単一 RTX 3090** で回る軽量教育実装、並列自己対局で 8プロセス 3.2×。[arxiv 2504.14636](https://arxiv.org/pdf/2504.14636)
  - **turbozero**(PyTorch, fast+parallel GPU)[github](https://github.com/lowrollr/turbozero_torch)、**AlphaZero.jl**(Julia, generic)[github](https://github.com/jonathan-laurent/AlphaZero.jl)、**alpha-zero-general**(Thakoor 2017, Python 汎用小規模)。
- 【推測】4060 Ti 16GB(A100 の数分の一の性能, VRAM 16GB)なら、Pgx/Gumbel/小型ネット/縮小目標を組めば **「囲碁で個人が数日で高段アマ」相当の到達は将棋でも原理的に射程**。ただし**将棋トップ(dlshogi)超えは非現実的**で、狙いは「小資源で self-play RL が elo を足すことの実証」に限定すべき。

## Item 3 — 既存資産の再利用可能性(自前 vs 借用)

| 資産 | (c) での可否 | 理由 |
|---|---|---|
| **SPRT/対局ハーネス**(`src/sprt.rs`, `yane-vs-yane`, `src/usi_engine.rs`) | ✅ **流用可** | USI を話す任意エンジンの棋力測定=MCTS エンジンが USI 対応なら評価基盤そのまま(champion/技巧2 との対局も既存)。 |
| **技巧2(gikou, GPL)** | ✅ 教師/アンカー存置可 | 事前生成の supervised ラベル源、外部アンカー監視に継続利用。 |
| **NNUE bridge**(057-059, `binbo_to_yane.py` 等) | △ **value 初期化のみ余地** | **スカラー評価(value 相当)を nn.bin に乗せる経路**。AZ の **policy ヘッド/MCTS には非対応**。 |
| **champion `task064_32m_lr1e-5`** | △ 直接は不可 | **HalfKP(玉相対スパース, 差分更新向き)= AZ の CNN 盤面平面と別アーキ**。AZ ネットを直接初期化できない(warm-start には「AZ 形ネットを技巧2/champion ラベルで supervised 再学習」の橋渡しが要る)。 |
| **YaneuraOu(GPL, αβ)** | ✗ 探索は作り直し | αβ であり MCTS でない。土台 OSS としては存置。 |
| **盤面/合法手生成** | 借用推奨 | **`apery_rust`(GPL, Rust USI エンジン, magic bitboard)**= 我々の Rust リポと相性可。自前 movegen(`src/`)もあるが MCTS/平面特徴向けに要拡張。[apery_rust](https://github.com/HiraokaTakuya/apery_rust) |
| **MCTS 探索本体** | **借用が現実的** | 自前実装は大コスト。Pgx(JAX, 将棋同梱)/ MiniZero / dlshogi コード(GPL)/ turbozero を借用候補に。 |

## Item 4 — 方策・価値ネットの設計選択肢

- 【事実】dlshogi/AZ は **単一 ResNet トランク + policy ヘッド + value ヘッド**(統合トランク・2ヘッド)。分離型も可だが統合が主流(特徴共有で効率的)。[Item 1 出典]
- 【事実】**入力表現**: AZ 系は **密な盤面平面(CNN 前提, dlshogi=104ch×9×9)**。現行 **HalfKP は玉相対スパース特徴で NNUE(全結合小型)向け=CNN 方策には不適**。→ (c) では**盤面平面表現を新規採用**が既定路線(dlshogi の 104ch 設計を踏襲可)。
- 【事実】**方策の出力表現**: dlshogi = **81×27=2187**(移動先マス × 移動方向/駒種)。[Item 1 出典] AZ 原論文の将棋も同種の平面エンコード。
- 【推測】小資源では **小型 ResNet(ブロック数↓/チャネル数↓)** + Gumbel 少 sim が要。VRAM 16GB は学習・推論とも小型なら余裕。

## Item 5 — 出発点(ゼロ知識 vs warm-start)

- 【事実】**ゼロ知識・単機はスクラッチ非現実的**(AobaZero ~100年/1台)。
- 【事実】**dlshogi は warm-start**: 棋譜/elmo 教師で **supervised 事前学習 → AlphaZero 流 self-play RL** のサイクルで成長。「elmo 教師収束モデルより self-play RL で明確に強化」。[logmi](https://logmi.jp/main/technology/324191) [Yamaoka その14/その26]
- 【事実】**RL は supervised より遥かに計算資源を食う**(同上)。
- 【推測】小資源での現実解 = **warm-start**。ただし champion は HalfKP で AZ ネットを直接初期化できない → **「AZ 形 policy+value ネットを、技巧2/champion のラベル(cp→value, bestmove→policy)で supervised 事前学習」する橋渡し**が要る。これは我々の既存教師生成基盤(gen-teacher, hcpe, 技巧2 の bestmove)を再利用できる可能性が高い【要検証】。「ゼロ知識哲学」を厳密に守るなら分散/長期計算が要り、単機では哲学より **warm-start の実利**を取るのが妥当。

## Item 6 — ライセンス/大会(電竜戦)整合(結論でなく要確認事項の列挙)

- 【事実】**dlshogi コード = GPL-3.0**(setup.py「GNU General Public License (GPL)」/ PyPI GPL)。[dlshogi](https://github.com/TadaoYamaoka/DeepLearningShogi) → **コードは YaneuraOu 同様に借用可(GPL 派生物として公開前提)**。
- 【事実】**dlshogi の学習済みモデル(`model-dr2_exhi.onnx` 等)= 私的使用のみ・封印**(`docs/USED_COMPONENTS.md` §3、著作権 山岡氏、禁止=追加学習/転用/直接対局/再配布)。
- 【事実】**cppshogi ← Apery(GPL-3.0)**, **apery_rust も GPL-3.0**。[Apery](https://www.chessprogramming.org/Apery) [apery_rust](https://github.com/HiraokaTakuya/apery_rust)
- 【事実】**電竜戦ルール**(`USED_COMPONENTS.md` §6): 大会期間中の他者ソフト取込禁止。**事前生成の自作教師 → 事前学習の自作 nn.bin** が合法ルート。
- **要確認事項リスト(次 /plan で詰める)**:
  1. 「dlshogi/MiniZero/apery_rust の**コード**を借用 + **重みは自前ゼロ知識/自前教師で学習**」が、YaneuraOu 借用時の建て付け(実装は借用可・評価の中身は自作)と**同一に成立するか**(=GPL コード借用は OK、封印は「学習済みモデル」のみ、で線引き一貫か)。
  2. Pgx(JAX)/ MiniZero / turbozero の**各ライセンス**を個別確認(GPL でない MIT/Apache 等なら派生物の扱いが変わる)。
  3. 借用フレームワークに**他者の学習済み重みが同梱されていないか**(同梱なら封印対象に該当しないか確認)。
  4. warm-start に**技巧2ラベル**を使う場合は §3 どおり「事前生成教師=合法」を維持(技巧2 自体は対局に使わない)。
  5. **HEROZ 版 dlshogi の非公開部分**は借用対象外(公開 OSS 部分のみ)。
  6. 大会が近づいたら最新ルール PDF を再確認(§6 の注記どおり)。

---
## (c) の現実的構成候補(比較表)

| 構成 | 実装コスト | 計算コスト(1×4060Ti) | 期待到達点 | 主リスク |
|---|---|---|---|---|
| **A. フルスクラッチ軽量 MCTS(Rust)** apery_rust movegen + 自前 policy/value(小 ResNet)+ 自前 MCTS(Gumbel) | **大**(MCTS/学習ループ/平面特徴/GPU 推論を新規) | 中(全部自作で最適化余地大だが実装律速) | 最も自作純度高・ライセンス最クリーン | 実装量が膨大、バグ源多、成果まで遠い |
| **B. 既存 GPU フレームワーク借用**(Pgx[将棋同梱] or MiniZero[Gumbel])+ 自前ネット学習 | **小〜中**(将棋 env は Pgx にあり/MiniZero は要追加、ネット/報酬を差す) | **小〜中**(Pgx の GPU ベクトル化 10-100×, Gumbel 少 sim) | **小資源で最速に self-play ループを回せる**=実証向き | JAX/フレームワーク学習コスト、ライセンス個別確認、将棋 env の成熟度 |
| **C. ハイブリッド warm-start**(A or B の器 + 技巧2/champion ラベルで AZ 形ネットを supervised 事前学習 → Gumbel 少 sim で短 self-play RL) | 中(橋渡し supervised + RL) | 中 | **「self-play RL が warm-start に elo を足すか」を最短で検証**=(c) の核心仮説を直接テスト | warm-start が天井(067-072 の再来)を持ち込む恐れ、橋渡し実装 |

【推測・方向性】**B の器 + C の warm-start** が小資源では有望。まず B で自己対局ループを回せる状態を作り、C で「warm-start + 短 RL が elo を足すか」を測る。到達目標は **dlshogi 超えでなく「単機で self-play 方策改善が正の elo を生む」実証**に置く。

## 第1マイルストーン案(最小で前進を確認できる検証点 = NOCCA 5×5 相当)
- **M0(器の疎通)**: 既存フレームワーク(Pgx or MiniZero or turbozero)の AZ を **極小ゲーム(tic-tac-toe / 5x5 mini-go 等)**で 4060 Ti 上で回し、**self-play→学習→強くなる**ループの疎通と VRAM/スループットを実測。
- **M1(将棋スループット)**: Pgx-shogi(or 将棋 env)で **1×4060 Ti の自己対局スループット(局面/秒)**を実測 → dlshogi の 17.9/s(マルチ GPU)と比較し **1サイクルの現実コストを見積る**。Gumbel 少 sim の効果も測る。
- **M2(warm-start 単体 = 探索が効くかの直接テスト)**: **技巧2/champion ラベルで AZ 形 policy+value を supervised 学習**し、**MCTS(少 sim)で champion(αβ)と対局**。self-play RL 抜きで「**MCTS 探索が同じ知識で αβ より elo を出すか**」を測る(既存 SPRT ハーネス流用)。← 最小コストで (c) の探索仮説を検証。
- **M3(self-play 1サイクル)**: M2 の warm-start を起点に **Gumbel 少 sim で短い self-play RL を1サイクル**回し、**gen1 が warm-start を超えるか**を SPRT で判定(自己強化アークと同じ gen-vs-gen + 技巧2 アンカー枠組み)。← これが正なら (c) は原理的に機能。

## 未解決の設計判断リスト(次 /plan で Keiichiro が決める)
1. **器**: フルスクラッチ(A) / Pgx / MiniZero / turbozero(B)のどれか。
2. **出発点**: ゼロ知識(哲学重視・単機非現実)vs warm-start(実利・天井リスク)。
3. **ネット**: 入力平面(dlshogi 104ch 踏襲?)/ ResNet サイズ / policy 出力(81×27)/ 統合 vs 分離。
4. **MCTS**: Gumbel 少 sim の採否・sim 予算。
5. **warm-start 橋渡し**: 技巧2 bestmove→policy, cp→value の supervised 事前学習を作るか。
6. **到達目標の再定義**: 「dlshogi 超え」でなく「単機で self-play RL が正の elo」= どの水準を成功とするか。
7. **ライセンス**: Item 6 の要確認 6 点をクリアするか。
8. **既存資産の去就**: SPRT ハーネス/技巧2 アンカー/champion(warm-start 橋渡しの教師)の使い方。

## 規律・メタ
本タスクは**調査のみ**。**コード変更・ビルド・学習・破壊的操作ゼロ**、既存 champion/リポジトリ不触、**Rust 無改変**(`cargo test` は無関係=実行不要)。一次情報優先・出典付き・事実/推測分離。**push は人間承認後。**
