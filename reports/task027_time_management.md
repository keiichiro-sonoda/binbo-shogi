# Task 027: 時間管理 + 最低限の info 出力

## 結論

- `go` の持ち時間を見て**反復深化を時間予算内で打ち切る**ようにした(`src/search.rs`,
  `src/main.rs`)。各深さ完了ごとに **USI `info`(depth/score/nodes/time)を stdout に流す**ので、
  ShogiHome の思考欄で「今何手まで読んだか」が見える。
- 反復深化の相性を活かし、**時間内に終わらない深さは中断・破棄し、完了した深さの最善手を返す**。
- 弱くなっていない: 対 Random 30勝0敗(時計なし=深さ上限で従来どおり決定的)、時間管理下でも合法手を返し
  破綻なし、詰みを `score mate` で正しく出す。perft/テスト/selfplay 無回帰。
- スコープ厳守: `stop`、高度な時間配分、`pv` 出力は未実装(バックログ送り)。

## A. 時間管理

### go のパース
`GoLimits`(既存)が `btime/wtime/binc/winc/byoyomi/movetime/infinite` 等をパース済み。今回それを使う。

### 時間予算の計算(`budget_from_limits`, 素朴)
- `infinite` → 予算なし(`None`。深さ上限まで)。
- `movetime <ms>` があれば最優先 → `movetime − マージン`。
- それ以外 → 手番側の `main_time/30 + inc + byoyomi − マージン`(`MOVES_LEFT=30`、想定残り手数)。
  - 秒読みのみ(持ち時間0)なら実質 `byoyomi − マージン`。持ち時間があれば粗く分配。
- **安全マージン = 500ms**(固定)。`bestmove` 送信の遅延で時間切れしないよう保守的に。下限 10ms。
- **時計情報が全く無い `go`(CLI/自己対戦/スモーク)→ 予算なし(`None`)= 深さ上限まで**(従来どおり決定的)。
  - これにより `nnue-match` 等の自己対戦は時計の影響を受けず再現可能なまま。

### 反復深化への打ち切り(`search_iddfs_timed`)
- 開始時刻 `Instant` を記録、`hard_deadline = start + budget`。
- 各深さ前にソフト判定: **`elapsed >= budget × 0.5` なら次の深さを始めない**(次の深さは通常それまでの合計より重いため)。
- ハード中断: `nega_alpha` が **2048 ノードごとに時計を確認**し、`hard_deadline` を過ぎたら `aborted` を立てて即巻き戻し
  (TT に中途結果は保存しない)。中断された深さはルートで破棄。
- **深さ1は必ず完了**(中断対象は深さ2以上)→ どんなに予算が小さくても合法手が必ず手に入る。
- 返す手: **最後に完了した深さの最善手**(中途深さの結果は使わない=反復深化と時間管理の相性の核)。
- `Depth` オプションは**安全上限**として残す(時計があれば時計が主、`Depth` は上限)。

### 実装構造
探索状態(net/tt/opts/nodes/deadline/aborted)を `Searcher` 構造体にまとめ、再帰の引数地獄を解消。
時計なしの `search_iddfs`/`search_root`(ベンチ・自己対戦用、決定的)は薄いラッパとして維持。

## B. 最低限の info 出力

- 各深さ完了時に1行: `info depth N score <cp|mate> nodes <累計> time <経過ms>`。
- `score`: 評価帯は `cp <整数>`(Task 022 の順序保存整数=手番側視点)、詰みは **`mate <手数>`**
  (詰ます)/`mate -<手数>`(詰まされる)。`usi_score()` が `MATE−n` から手数に変換。
- **stdout に出し、各行ごとに flush**(パイプはブロックバッファなので flush しないと GUI に届かない)。
  `info string ...`(ロード通知等)は従来どおり stderr(stdout は USI 専用、`DESIGN.md`)。
- `pv`(読み筋)は出さない(置換表を辿る復元が要る → バックログ)。depth/score/nodes/time だけでも
  ShogiHome は表示する。

## 動作確認

| 条件 | 結果 |
|---|---|
| byoyomi 2000ms(予算1500), Depth上限10, 中盤 | info depth 1→2→3 が流れ、**実時間 ~1.5s**。深さ4は着手したが hard_deadline で中断・破棄 → 深さ3の手 `7e6d` を返す |
| movetime 800ms(予算300) | depth 1,2 完了、depth 3(338ms)は中断 → depth 2 の手を返す |
| 終盤 詰みあり, byoyomi 3000 | `info depth 3 score mate 3` を出し `bestmove 5g5c+`(mate-break と USI mate 表記) |
| byoyomi 50ms(予算は下限10msに退化) | depth 1,2 完了で**合法手を返す**(深さ1は必ず完了) |
| 対 Random 30局(時計なし=深さ上限3) | **30勝0敗**(決定的、無回帰) |

- info は stdout(stderr を捨てても info 行が出ることを確認)。`info string` は stderr。
- `cargo test` 16/16、`perft` 1〜5 完全一致、`selfplay`(Random)完走、option 無指定 USI は Random、警告なし。

## 作者向け(ShogiHome での確認)

- 通常対局(持ち時間0・秒読み30秒)なら ShogiHome は `go btime 0 wtime 0 byoyomi 30000` を送るはず。
  → 予算 ≈ 29.5s(byoyomi − 0.5s マージン)で打ち切り、時間切れしない想定。
- **実機の `go` ログを確認**して、想定どおりのトークン(byoyomi 等)が来ているか見てほしい(推測で書いたパースの裏取り)。
- ShogiHome の思考欄に `depth / 評価値 / nodes / 時間` が深さ 1→2→3… と更新表示されることを確認。

## 所見・正直な注記

- 時間配分は素朴(byoyomi ほぼ全部 − マージン、持ち時間は `/30` の粗い分配)。勝負所配分などは未実装(バックログ)。
- マージンは固定 500ms。秒読みがマージン未満(極小)だと予算が下限 10ms に退化するが、深さ1は完了するので
  合法手は必ず返る(実用的な秒読み ≥ 1s では問題なし)。
- ハード中断は中途深さを**破棄**するため、その深さ分の計算は捨てる(時間切れは防げるが計算は無駄になる)。
  ムーブオーダリングで EBF が下がっているので、ソフト 0.5 と併せ無駄は限定的。
- 時間管理下の自己対戦は局面・タイミングで到達深さが変わり**非決定的**(正常)。決定的な回帰は時計なし(深さ上限)で取る。

## 成果物の場所

- コード(コミット): `src/search.rs`(`Searcher` 構造体、`search_iddfs_timed`、`IterInfo`、`usi_score`、時間中断)、
  `src/main.rs`(`budget_from_limits`、`NnueEngine` の info 出力、USI engine の `emit_info`)。
- 重み(非コミット): `/tmp/binbo-task017/model_wd1e-3/nnue_f32.bin`。

## スコープ外(バックログ送り)

- `stop` コマンド対応(並行処理)。高度な時間配分(勝負所で多く使う等)。`pv`(読み筋)出力(TT 復元)。
- → `docs/BACKLOG.md` に残課題として追記。
