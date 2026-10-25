# Task 021: NNUE 評価器を1手探索で ThinkingEngine に接続(対局できる骨格)

## 結論

- **技巧2 の評価を学んだ生徒 NNUE が、初めて盤上で手を指せるようになった。**
- `NnueEngine`(`ThinkingEngine` 実装、1手探索)を追加し、USI で着手を返す。エンジン選択は
  USI option(`EvalFile` / `Eval`)で行い、`ThinkingEngine` trait と USI ループは無改修。
- **視点・符号は正しい**(段階B): タダで駒が取れる局面で取る手を最上位に選ぶ。
- **NNUE 対 Random で 300 戦 300 勝 0 敗**(2 シード、両手番)。学習した評価がランダムより明確に
  強い手を選べている。
- 既存(RandomEngine・selfplay・perft・全テスト)は無回帰。

## 1. 評価値の視点・符号(最重要)

- NNUE `forward` は **手番側視点の logit**(大きいほど手番側に有利。技巧2 の手番側 score cp を蒸留)。
- 1手探索は「自分が指した後の局面」を評価する。指すと手番が相手に移るので、子局面の logit は
  **相手視点の良さ**。よって **自分視点スコア = `-forward(child)`**(negamax 深さ1)。これを最大化。
- 符号を間違えると「最悪手を選ぶエンジン」になる(静かなバグ)。段階Bで明示的に確認した。
- 唯一の終端特例: 自分の手の後で相手に合法手が無ければ詰み(将棋にステイルメイト無し)→ 我々の勝ち
  = `+∞`。それ以外は全て NNUE で評価(「まず素直に」)。

## 2. 実装

- `src/main.rs`:
  - `NnueEngine`(`ThinkingEngine`): `Nnue::load` で重みをロード。`think` は全合法手を
    `do_move → -forward(child) → undo_move` で評価し、自分視点最大の手を返す。合法手0なら resign。
    局面は `pos.clone()`(yasai `Position: Clone`)した作業盤で進める。
  - エンジン選択: USI option `Eval`(combo: `nnue`/`random`、default `nnue`)と
    `EvalFile`(string、重み `nnue_f32.bin` のパス)。`isready` で `rebuild_thinker`:
    `Eval=nnue` かつ `EvalFile` があり読めれば `NnueEngine`、そうでなければ `RandomEngine` に
    フォールバック(info string で通知)。**option 無指定では従来どおり Random**(無回帰)。
  - 確認用 CLI: `nnue-bestmove <weights> <sfen>`(全合法手を自分視点スコア順表示)、
    `nnue-match <weights> <games> [seed] [max_plies]`(NNUE 対 Random 勝率)。
- `ThinkingEngine` trait・USI ループ(`go → think → bestmove`)は無改修(Task 005 の殻分離の狙い)。

## 3. 動作確認(段階A〜C)

### 段階A: USI スモーク
`usi / setoption EvalFile / isready / position / go` で NNUE がロードされ、合法手を bestmove で返す。
no-arg 起動でも USI を話す(ShogiHome 互換)。非合法手は返さない。

### 段階B: 視点・符号 / 筋の確認
タダで取れる金がある局面 `4k4/9/4g4/9/9/9/4R4/9/4K4 b - 1`(黒 ▲5七飛 が無防備な △5三金 を取れる)で:

```
top moves (our-view score):
   1. 5g5c+   +0.9292   ← 金を取って成る(最善)
   2. 5g5c    +0.9083   ← 金を取る
   3. 5g2g    +0.3795   ← 以下、駒を取らない手
   ...
  worst: 5i6i    -0.4991  ← 玉が無意味に動く手
bestmove 5g5c+
```

→ **取る手を最上位に選ぶ = 視点・符号は正しい**(符号が逆なら最悪手を選ぶ)。材料得・成りも妥当に評価。

### 段階C: NNUE 対 Random(自己対戦完走 + 勝率)
- 両手番を交互、最大320手、複数シード:

| シード | 局数 | NNUE 勝 | 負 | 分 |
|---|---|---|---|---|
| 20260621 | 100 | 100 | 0 | 0 |
| 99 | 200 | 200 | 0 | 0 |
| **合計** | **300** | **300** | **0** | **0** |

→ **NNUE 対 Random で 100% 勝利**。1手読みでも、学習した評価がランダムな手選択より圧倒的に強い。
これが「知識ゼロから技巧2 蒸留で学んだ評価が、実際の手選択を改善した」最初の傍証。

## 4. ShogiHome 登録手順(段階D、作者向け)

1. リリースビルド: `CARGO_HOME=/tmp/binbo-task011-cargo cargo build --release --locked --offline`
   → 実行ファイル `target/release/binbo-shogi`(no-arg で USI を話す)。
2. **重みファイルを安定した場所に置く**: 現状 `nnue_f32.bin` は `/tmp/binbo-task017/model_wd1e-3/`
   (再起動で消える)。永続利用するなら任意の安定パスにコピー(例 `~/binbo/nnue_f32.bin`)。
   重みが無い/再学習したら `python -m trainer.export_nnue <model_dir>` で再生成。
3. ShogiHome: エンジン管理 → 追加 → `target/release/binbo-shogi` を選択。
4. そのエンジンの USI オプションで **`EvalFile`** に手順2の `nnue_f32.bin` の絶対パスを設定
   (`Eval` は既定 `nnue` なので EvalFile 設定だけで NNUE が有効。Random にしたいなら `Eval=random`)。
5. 対局開始。`info string loaded NNUE eval from ...` が出れば NNUE で指している。

## 5. 所見

- 1手読み・f32・差分更新なしでも、NNUE 対 Random は完勝。評価器が「材料・成り・玉の安全度らしきもの」を
  捉えている(段階Bの手の並びと整合)。
- ただしこれは「ランダムより強い」段階。**本格的な棋力(技巧2 レベルとの対戦)は未測定**で、
  1手読みは浅いので人間相手には弱いはず。深さは次段の αβ で。
- 終端(詰み)以外は NNUE 任せ。深い読み・千日手・詰み探索は今回スコープ外。

## 回帰

- `cargo test --release --locked` 13/13。`cargo run -- verify` perft 1〜5 完全一致。
- `selfplay`(RandomEngine)完走。option 無指定の USI は従来どおり Random にフォールバック。
- ビルド警告なし。

## 成果物の場所

- コード(コミット): `src/main.rs`(`NnueEngine`、エンジン選択、`nnue-bestmove`/`nnue-match`)。
- 重み(非コミット・再生成可): `/tmp/binbo-task017/model_wd1e-3/nnue_f32.bin`。

## スコープ外(次段)

- 深い αβ探索(数手先読み・枝刈り)。差分更新・SIMD・量子化。本格棋力測定(技巧2 レベル戦)。
  df-pn 詰み探索。ponder。
