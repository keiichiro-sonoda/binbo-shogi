# 使用コンポーネントとライセンス帳簿(正本)

このファイルは binbo-shogi が依存・利用する**外部成果物のライセンスと由来**の唯一の正本です。
方向転換(Task 056: 自前探索→やねうら王土台, 自作 NNUE 評価で勝負)を機に作成。
**目的**: GPL 遵守、教師の由来を説明可能に、いずれ大会(電竜戦等)に「出せる形」を最初から保つ
(「忘れることは前提に外部化する」原則)。新しい依存が増えたらここに追記する。

> 配置の規律: やねうら王本体・eval・教師エンジン・モデルは **`~/binbo-shogi-data/`(永続・git 非追跡)**。
> リポジトリには本帳簿・スクリプト・glue・ドキュメントのみ。詳細 `docs/data_storage.md`。

## 1. 探索エンジン(土台)

| 項目 | 内容 |
|---|---|
| 名称 | **やねうら王 (YaneuraOu)** NNUE 通常版 |
| バージョン | `YaneuraOu NNUE 9.41git 64AVX2`(clone HEAD) |
| 取得元 | https://github.com/yaneurao/YaneuraOu |
| ライセンス | **GPL-3.0**(`LICENSE`)。改変・再配布可、派生物も GPL。配布する場合は無償でもソース提供などの義務がある (GPLv3 §§4–6)。このリポジトリはやねうら王を同梱しない |
| 改変ファイル | **なし**(pristine upstream。`git status` で source 無改変を確認。`eval/` と build 成果物 `source/YaneuraOu-by-gcc` のみ untracked 追加) |
| ビルド | `make -j normal COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVX2`(g++ 11.4, WSL2 14700KF) |
| 配置 | `~/binbo-shogi-data/engines/yaneuraou/`(非コミット・永続) |

**GPL 遵守メモ**: 将来 Public 化・配布する場合、やねうら王を含む派生物は GPL-3.0 で配布し、
ソース提供義務を満たす(改変点・ビルド手順を明示。現状 改変なし)。

- **大会のビルド (2026-10-24 〜)**: 同じやねうら王のソース (無改変) を `make pgo COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVXVNNI` で作ったもの (`engines/yaneuraou_fast/YaneuraOu-avxvnni-pgo`, sha256 `38f89c23…`, Task 178)。AVX-VNNI と PGO はどちらもやねうら王の Makefile に元からある選択肢。読みは AVX2 版と同じ (bench・静的評価・対局 100 局で一致), 28 スレッドで NPS +8.1%。

## 2. 評価関数(nn.bin)

### 2.1 採用中の評価関数 (2026-10-24 〜) = 水匠5 の公開データで学習した生徒 + binbo の自己対局での絞り
- **採用中の eval(2026-10-25 〜, 作者承認)= `task188_S2_MR`**(`eval_dirs/task188_S2_MR/nn.bin` sha256 `6f0fffd7f7e974200b6cbe40fa369ec261d6b1d501dea80a0c3ccaa0f16d7e6e`)。土台の生徒を, 公開データ nodchip/shogi_suisho5_depth9 (MIT) の学習用 `shuffled.bin` の先頭 127,900,000 局面 (下の 1. の 2 倍, 前半は同じ局面) でゼロから 12 周 (Task 188, 2 つのキャッシュを載せ替える学習器), 仕上げは下の S_MR と同じ。S_MR に +24.4 [+10.5, +38.3], 確認の組で物差し 2 本 +20.0 / micro +33.8。
- 10/24 午後〜10/25 の採用 eval = `task184_S_MR`(`eval_dirs/task184_S_MR/nn.bin` sha256 `059368143f05ef4deca064e83121574c97b00d4a82fe9e36f0dc98b6f5402326`)。下の 1. の生徒から, 2. の自己対局 (1 周目 task142 どうし 1,028 万 + 2 周目 task154 どうし 983 万, 目標は binbo の depth 9 の評価値 7 割 + 勝敗 3 割) と, nodchip/shogi_suisho5_depth9 のうち配布元が検証用に分けた 724 万局面 (MIT) を混ぜて 2 周 lr 1e-5 (Task 184)。S_R2 に +19.7 [+2.6, +36.9], 確認の組で物差し 2 本 +28.0 / micro +3.6。
- 同じ日の午前の採用 eval = `task166_S_R2`(`eval_dirs/task166_S_R2/nn.bin` sha256 `ed39dd6626bfbbf7a03de4c74525e2aaf4427ab359f583f0165ea773a9b1abe3`)。
  形は今までと同じ HalfKP 256x2-32-32 (やねうら王の NNUE), 学習・書き出しは自作の `trainer/` (書き出しの 0cp 一致 GREEN)。決定 `docs/decision_2026-10-24_student_base.md`。
  1. **土台 = 水匠5 の生徒** (`task145_suisho5_student`, Task 145/145b): 公開教師データ **nodchip/shogi_suisho5_depth9**(Hugging Face, **MIT**, 水匠5 depth 9 の評価値, 静止局面に置き換え済み)の**学習用データ**の先頭 63,950,000 局面 (シャッフル済みの `shuffled.bin` の先頭 = 無作為の抜き出し, 7z の 1 本目 sha256 `c7dd5b39…`) で, 自作の学習器でゼロから 12 周 (lr 6e-5, Factorizer, 鏡像, QAT)。**大会では使用ライブラリとして申告する**。取り出しの手順は `reports/task145_suisho5_student.md`。
  2. **絞り** (Task 166, 確認 Task 168): 1. の重みから, binbo の自己対局 (2026-10-21〜22, 前の採用 eval `task154_R_result30` どうしの 15 万局, やねうら王 depth 9, 詰みまで) の 983 万局面で 2 周 lr 1e-5。目標 = その対局で binbo が読んだ depth 9 の評価値 7 割 + 勝敗 3 割。自己対局に使った eval (task154) は §2.1 の下の旧系統 (技巧2 の局面 + 自作の教師 WS-T + 水匠5 の検証用 724 万での仕上げ)。
  3. 強さ: 前の採用 eval に新しい 3 組で +69.1 [+53.6, +84.7], 確認の組で +76.0, micro +45.1 (Task 166/168)。
- 比較用の旧系統 (2026-10-24 にいったん止めた): 以下の自作 NNUE の系譜。

### 2.1.1 旧系統 = 自作 NNUE (技巧2 の局面 + 自作の教師)
- **self-trained NNUE**(HalfKP, 自作教師=技巧2 蒸留, 自作学習パイプライン `trainer/`)。
- やねうら王形式 `nn.bin` での出力は **Task 056 で構想・調査**(`docs/nnue_yaneura_export.md`)、実装は次段。
- 学習データ・重みは**非コミット**(`~/binbo-shogi-data/models/`, `teacher/`)。由来は本帳簿 §3 + `docs/kp_spec.md`。
- **前の採用 eval(2026-10-21 〜10-24)= `task154_R_result30`**(`eval_dirs/task154_R_result30/nn.bin` sha256 `d1a1cb49…`)。下の Task 129 の eval を 2 段仕上げたもの。
  1. **Task 142**(`task142_finetune`, `651ff927…`): 公開教師データ **nodchip/shogi_suisho5_depth9** の検証用 724 万局面(**MIT**, 水匠5 depth 9 の評価値, 静止局面に置き換え済み)で 2 周 lr 1e-5。**大会では使用ライブラリとして申告する**。データは `~/binbo-shogi-data/public/`(非コミット), 取得と出典は `docs/public_data_survey_2026-10-19.md`。
  2. **Task 154**: 1. の eval どうしの自己対局 15 万局(やねうら王 depth 9, 詰みを読み切るまで指す)の 1,028 万局面で 2 周 lr 1e-5, 目標 = 自分の探索の評価値 7 割 + 勝敗 3 割。新たな外部要素はない(自己対局・静止局面への置き換えとも, 既に使っているやねうら王 GPL-3.0)。
- 2 段の仕上げの元 = `task129_student_64m_qsrel_e12`(2026-10-12〜10-19 の採用 eval, `eval_dirs/task129_student_64m_qsrel_e12/nn.bin` sha256 `6bdd0a7c…`、BNNQ `34714340…`)。64M の全局面(除外なし)で、41% の局面をやねうら王の `qsearch_psv` で静止局面に置き換え(静止探索に使った eval は自作の `task109_student_64m_e24`)、置き換えた局面には同じ教師(WS-T + MCTS S=8)でラベルを付け直した。学習率 6e-5・Factorizer・12 epoch(Task 128/129)。その前任 = `task126_student_64m_factor_e24`(`3f436825…`)← `task121_student_64m_lr3e5_e24`(`709191c4…`)← `task109_student_64m_e24`(`df68cecf…`)← `task107_student_64m`(`a1c70adc…`)← `task106_student_filter`(`7aaa0e27…`)← `task104_student_search`(`61875707…`)← `task064_32m_lr1e-5`(`3c25685e…`)。
  - 置き換えで新たに使った外部要素は、やねうら王(既に対局・書き出しで使っている、GPL-3.0)の `qsearch_psv` コマンドだけ。置き換え後の局面は技巧2 で自己生成した局面から静止探索で数手進めたもので、ラベルは自作の WS-T による。
  - 局面: 技巧2 で自己生成した hcpe(32M + Task 101 で追加生成した `run32m_g` 32M)。
  - 教師の系譜は `task104_student_search` と同一(ラベルも同じファイル)。違いは学習局面の選別だけで、除外の判定には局面そのもの(王手か)と hcpe に入っている技巧2 の最善手(駒取りか)を使った。新たな外部要素はない。
  - 教師の系譜: 局面 = 技巧2 で自己生成した hcpe(§5)。評価ラベル = **WS-T**(自作の Transformer 方策・価値網。技巧2 蒸留データ `m4_data` で事前学習、Task 087)+ Gumbel MCTS S=8 の root value(Task 099–104)。**dlshogi 由来の重み・ラベルは含まない**(rescore 出力のファイル名 `*_dlshogi_evals.npy` は Task 054 由来の形式名で、中身は WS-T の値)。
  - ラベル生成に使ったライブラリ: Pgx(Apache-2.0、将棋 env)、mctx(Apache-2.0、Gumbel MCTS)、JAX / dm-haiku(Apache-2.0)。いずれも事前の教師生成にだけ使い、対局には使わない。Pgx 同梱の学習済み重みは使っていない。

### 2.2 動作確認用 smoke eval(**大会では封印**)
| 項目 | 内容 |
|---|---|
| 名称 | **tanuki- Háo `halfkp_256x2-32-32` (2023-05-08)** の `nn.bin` |
| 取得元 | https://github.com/nodchip/tanuki-/releases/tag/tanuki-.halfkp_256x2-32-32.2023-05-08 |
| ライセンス | **GPL-3.0**(`nodchip/tanuki-` リポジトリ。配布 7z にも `gpl-3.0.txt` 同梱) |
| サイズ/sha256 | 64,217,066 B / `1141d275bceec911156801f27303dc9ff5beb24f4f59144cc069306c59e80782` |
| 用途 | **やねうら王の動作確認のみ**(Task 056 smoke: eval ロード→指し手→詰み)。**自作 eval で勝負するため大会用は封印** |
| 配置 | `~/binbo-shogi-data/engines/yaneuraou/eval/nn.bin`(非コミット。license も `eval/LICENSE-tanuki-eval-gpl-3.0.txt` に同梱保存) |

### 2.3 測定用の対局相手(物差し、2026-10-05 決定)— **測定専用。学習・教師・配布・大会には使わない**
| 名称 | ライセンス | 用途 | 選定理由 |
|---|---|---|---|
| **tanuki- Háo** `halfkp_256x2-32-32`(§2.2 と同じファイル) | GPL-3.0 | やねうら王 + `NodesLimit` で強さを絞った外部アンカー | 強さを連続調整でき、エラー 0 で安定、ライセンス明確 |
| **AobaNNUE**(作者 山下宏氏、https://github.com/yssaya/AobaNNUE) | **GPL-3.0**(作者リポジトリに明示、nn.bin も GPLv3 と作者が明言 2026-09-27) | AobaNNUE 用やねうら王(halfkp_768x2-16-64、作者の GPL ソースからビルド)+ `NodesLimit` の外部アンカー | AobaZero 由来で tanuki- とも本プロジェクトの教師とも系統が違い、相性の偏りを減らせる。ライセンス明確 |
- AobaNNUE の配置(非コミット・永続): `~/binbo-shogi-data/engines/aobannue/`。配布 zip `20260103AobaNNUE.zip`(v1.1、作者リリース記載の Google Drive、sha256 `83d4798d…`)を展開、`AobaNNUE/eval/nn.bin`(192,624,720 B、sha256 `f8ee839a…`)。エンジンは同梱ソース(やねうら王 V9.00 ベース、`YANEURAOU_ENGINE_NNUE_HALFKP_768X2_16_64`)を g++ / AVX2 でビルドした `AobaNNUE/source/YaneuraOu-by-gcc`(`id name AobaNNUE NNUE 9.00git 64AVX2`、FV_SCALE 既定 40)。GPL 本文は `COPYING-AobaNNUE-GPL-3.0.txt`(作者リポジトリの `Copying.txt`)を同じ場所に保存。
- **採用前の確認だけに使う 3 本目(2026-10-21)**: **つようら王11micro**(評価関数, **MIT**, 作者 やねうら王氏の 2026-10-14 公開)を GitHub `32165a8` のやねうら王 9.80git(GPL-3.0, `YANEURAOU_ENGINE_SFNN_ka2_2048_8_64_progress8` を g++ / AVX2 でビルド)で @11,000 ノード。配置 `~/binbo-shogi-data/public/tsuyouraou11/`, `engines/yaneuraou980/`。経緯 `reports/task151_tsuyou_ladder.md`。
- 不採用: **水匠5**(一次情報にライセンス表記なし。Wikipedia の「GPL v3」は出典なし)、**技巧2@8**(対局中に落ちる)。詳細と出典は `docs/yardstick_survey_2026-10-05.md`。

## 3. 教師エンジン(評価ラベル生成)

| エンジン | ライセンス/由来 | 用途 | 大会可否 |
|---|---|---|---|
| **技巧2 (Gikou v2.0.2)** | **GPL**(https://github.com/gikou-official/Gikou) | depth8 cp ラベル生成(蒸留教師)。`src/teacher/gikou.rs` 経由 | **評価ラベル生成は可**(大会"期間中"のソフト取り込みに非該当=事前生成)。要・由来記録 |
| **dlshogi (`model-dr2_exhi.onnx`)** | **私的使用無償**(著作権 山岡氏)。禁止=追加学習/パラメータ転用/直接対局/定跡化/再配布 | Task 054/055 の蒸留先実験(value→cp)。**ユーザー同意済み**。重み非コミット | **大会用は封印**(私的使用のみ)。大会教師は別の合法ルートで(下記 §6) |

- 技巧2 蒸留の cp ラベルから自作 NNUE を学習(=自分の学習データ生成)。技巧2 自体は対局に使わない。
- dlshogi は素朴 value 差し替えが棋力で悪化し不採用(Task 055)。大会には用いない。

## 4. 定跡(opening book)

- **現状なし**(smoke で `standard_book.db` 読み込みエラーは無害=book 未設置で続行)。
- 将来 定跡を使う場合は出典・ライセンスをここに追記(自作 or 許諾明確なもの)。

## 5. 棋譜(game records)

- 教師データ = **自作**(技巧2 で自己生成した hcpe, `~/binbo-shogi-data/teacher/`)。外部棋譜は不使用。
- 外部棋譜を取り込む場合は出典・ライセンスをここに追記。
- **公開教師データ(2026-10-19 〜)**: nodchip/shogi_suisho5_depth9(MIT)。2026-10-24 から**採用中の eval の土台**(学習用データの 6,395 万局面でゼロから学習した生徒, §2.1)。それ以前は検証用 724 万局面を旧系統の仕上げに使っていた。
- **自己対局(2026-10-21 〜)**: binbo 自身の自己対局 15 万局(Task 154, `~/binbo-shogi-data/work/task154/`)。

## 6. 大会(電竜戦)に向けた注記 ※遠い目標・軽い確認

- **電竜戦ルール**(出典 = denryu-sen.jp 大会ルール PDF): **大会期間中**(委員長の開会〜閉会宣言)に
  **他者作成の将棋ソフト(評価関数・探索・定跡 等)を取り込むことを禁止**。また**大会中の棋譜**を
  定跡・学習に取り込むことも禁止(自戦は除く)。
- **本プロジェクトの合法ルート**: **事前生成**の自作教師(技巧2=GPL, 大会前)から**事前学習**した
  **自作 nn.bin** を載せて出す。やねうら王本体は GPL で取り込み可(土台 OSS)。
  **2026-10-24 から**: 大会前に公開教師データ (nodchip 氏, MIT) と自分の自己対局で学習した nn.bin (自作の学習器で作成) を載せる。公開データは申込の使用ライブラリ欄で申告する。
- **封印**: dlshogi 由来の教師/ラベルは**大会に用いない**(私的使用のみ)。smoke eval(tanuki-)も大会では使わず自作 eval に差し替える。
- 提出が近づいたら最新の大会ルール PDF を再確認し、この節を更新する。
