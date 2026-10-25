# Task 056: 方向転換 — やねうら王土台 + 自作 NNUE 評価(地固め)

## 目的(Phase 1 地固め)
Tasks 029–055 で全軸を切り分けた結論 = **Lv3+ の壁の真因は小さい KP fd8 生徒の表現容量**
(探索 049-050・KP系表現 052-053・強い先生 055 のどれも破れず)。自前探索は基礎を辿り尽くした。
→ **自前探索を手放し、やねうら王(GPL-3.0)を土台に、自作 NNUE 評価 + 自作教師 + 学習パイプライン +
実験ログで勝負する**。本タスクは「壊れずに動く土台」を作る(棋力は次段)。「測ってから走る」。

セッション確定方針: **NNUE 経路 = 自作 trainer 拡張で nn.bin 直接出力**(構想のみ今回)、
**リポジトリ = main 据え置き + tag 堅牢化**。

## ステップ1: 自前実装の保全 + 資産棚卸し
- Task 055 時点の自前実装に印を付けて保全した (コードは `src/` に残っている)。
  - 注: SSH agent が `communication with agent failed` を出すが push 自体は成功(別キーで通る)。`ls-remote` は BatchMode だと publickey 拒否で hang/失敗するため、push の `[new tag]` 確認を正とした。
- main は据え置き(自前 Rust を消さない)。引き継ぐ資産(新方向の中核): `trainer/`(PyTorch)、
  `src/teacher/` + 技巧2 教師生成の知見、`trainer/dlshogi_rescore.py`、`docs/{kp_spec,data_storage,BACKLOG}.md`、
  `scripts/env.sh` 規律、`reports/` ログ規律。

## ステップ2: やねうら王ビルド + USI 動作確認(WSL2)
- 配置 `~/binbo-shogi-data/engines/yaneuraou/`(永続・非コミット)。clone は **pristine upstream**(source 無改変)。
- ビルド済みバイナリ `source/YaneuraOu-by-gcc` = **`YaneuraOu NNUE 9.41git 64AVX2`**(g++, AVX2, halfkp_256x2-32-32)。
  正準ビルド: `make -j normal COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVX2`。
- **smoke eval**: tanuki- Háo `halfkp_256x2-32-32` (2023-05-08) の `nn.bin`(**GPL-3.0**, 64,217,066 B,
  sha256 `1141d275…e80782`)を GitHub release から取得→ `eval/nn.bin` 配置。**大会では封印**(`docs/USED_COMPONENTS.md`)。
- **USI smoke(全 green)**:
  - `usi` → `usiok`(id name 上記)。
  - `setoption EvalDir` → `isready` → `info string loading eval file …/nn.bin` → **`readyok`**(eval ロード成功)。
  - `position startpos` + `go depth 18` → **`score cp 83`, nodes 816308, 6.4 Mnps, `bestmove 2g2f` ponder 3c3d**(妥当な序盤 PV)。
  - 詰み: `position sfen 4k4/9/4G4/9/9/9/9/9/K8 b G 1`(自作 mate-in-1, 玉頭金=`G*5b`, 5c金で支え) + `go mate` →
    **`score mate 1`, `bestmove G*5b`**(手検証どおり)。
- **観測した罠(記録)**: 全コマンドを一括 pipe すると **`quit`/stdin EOF が探索スレッドを即中断**し
  `info depth 1 nodes 0` で返る。**read-write FIFO(`exec 3<>fifo`)で stdin を開いたまま保持**し、
  `timeout` で idle エンジンを終了させて正常な探索を観測した。最初の mate テストの segfault は
  **SFEN に先手玉が無い不正局面**(両玉必須)が原因→修正で解消。
- `scripts/env.sh` に `YANE_DIR`/`YANE`/`YANE_EVAL` 追加、`docs/data_storage.md` に やねうら王パス+再ビルド手順を追記。

## ステップ3: eval 差し替え経路の理解 + nn.bin 出力構想
- `docs/nnue_yaneura_export.md` 作成(**技術的核心**)。upstream C++ を一次情報に精読:
  - **差し替え = `EvalDir` 配下の `nn.bin` を差すだけ**(本体無改変)。edition ↔ eval アーキは厳密一致必須。
  - **nn.bin 構造**(全 LE): `[version 0x7AF32F16][kHashValue][arch_str]` + FT(`[hash][i16 bias×256][i16 weight×256×(81·fe_end)]`)
    + affine 3 層(`[hash][i32 bias][i8 weight(plain 出力major padded)]`)。
  - **量子化**: FT=int16, affine=int8/int32, 出力 `>>kWeightScaleBits(6)`, 最終 `/FV_SCALE(16)`。
  - **要一致**: 各コンポーネント `GetHashValue()` 不一致は **load 失敗**(全体 hash は Warning のみ)→ dims 厳密一致が必須。
  - **binbo→やね ギャップ**: ①fd8→256x2-32-32(**推奨=trainer を 256 dims で学習**, 天井破りと合致)、
    ②HalfKP 付番が cshogi 系 ≠ BonaPiece(**index 再マップ必須**)、③量子化+直列化=次段の新規コード、
    ④affine 重みの GetWeightIndex スクランブル(ファイルは plain 順)、⑤PSV は本路線では不要(DESIGN 維持)。
  - **次段検証計画**: round-trip(既知 nn.bin を parse→同一バイト re-emit, sha 一致)→ forward parity(N 局面で評価値突合せ)→ 1000 局棋力。

## ステップ4: ライセンス帳簿 `docs/USED_COMPONENTS.md`
- Engine やねうら王(GPL-3.0, 無改変)/ smoke eval tanuki-(GPL-3.0, 封印)/ 目標 eval=自作 NNUE /
  教師=技巧2(GPL, 生成のみ)・dlshogi(私的使用のみ・封印)/ 定跡なし / 棋譜=自作。
- **電竜戦注記**(出典 denryu-sen.jp ルール PDF): 大会期間中の他者ソフト/大会棋譜の取り込み禁止 →
  **事前生成の自作教師から事前学習した自作 eval が合法ルート**。dlshogi/smoke eval は大会で封印。

## ステップ5: 方向転換のドキュメント化
- `STATUS.md` 現在のフェーズを方向転換に更新(Task 055 の結論=器が天井 を理由に明記、保全に言及)。
- `DESIGN.md` に方向転換の確定判断を追加(探索系判断に「自前は保全・土台はやねうら王」を相互参照、PSV 再訪注記)。
- `docs/BACKLOG.md` に次段(Phase 2 自前 NNUE v0 / ローカル自動対局基盤 / 大会用合法教師ルート)。

## Definition of Done(達成状況)
- [x] 自前実装版を保全+ 引き継ぐ資産を整理(本報告 §1)
- [x] やねうら王を WSL2 でビルド(済バイナリ検証)+ USI 動作確認(評価 cp83・指し手・詰み mate1)
- [x] 評価差し替え経路を理解 + 自作 eval 出力の構想(`docs/nnue_yaneura_export.md`)
- [x] `docs/USED_COMPONENTS.md`(エンジン・評価・教師・定跡・棋譜 + 電竜戦注記)
- [x] STATUS/DESIGN に方向転換、BACKLOG に次段
- [x] やねうら王本体・eval を永続領域に配置(/tmp 回避)、GPL 遵守
- [x] レポート(本ファイル)、コミット(次に実施)

## スコープ外(次段)
自前 NNUE v0 の学習・対局(Phase 2)、棋力追求、dlshogi 大会利用、自前 Rust 消去、Public 化実行、Xeon 使用。
