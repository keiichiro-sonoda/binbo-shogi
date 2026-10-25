# Task 022: 固定深さ αβ(nega-alpha)探索

## 結論

- 1手読みを **固定深さの negamax + αβ枝刈り探索**(`src/search.rs`)に置き換えた。深さは USI option
  `Depth`(default 3)で可変。
- **αβ枝刈りは明確に効いている**(中盤 depth3 でノード 96.8% 削減)。
- **depth3 は depth1 に 50戦50勝**(乱数オープニング25種×色入替)、depth3 対 Random も 50戦50勝。
  → **読みを深めて強くなった**(このタスクの主目的を達成)。
- 視点・符号は多段でも正しい(タダ取り/詰みの局面で取る手・詰ます手を選ぶ)。
- 既存(perft・全テスト・RandomEngine・selfplay)は無回帰。
- スコープ厳守: 置換表・反復深化・ムーブオーダリング・MTD(f)/PVS・静止探索・時間管理は**入れていない**
  (純粋な αβ だけを固め、次段の各改善の効果をくっきり測れるようにした)。

## 1. logit → 整数化(順序保存)

- `eval_to_int(logit) = round(logit * 600)`、`±29000` にクランプ(`search.rs`)。
- 要件は順序保存のみ(cp の厳密な意味づけは後段)。スケール600は学習時 sigmoid スケールで、
  結果は「cp 風」の整数になるが、**比較に使うのは大小関係だけ**。

## 2. 詰み・終端

- 合法手0 = 手番側の詰み(将棋にステイルメイト無し)→ `-(VALUE_MATE - ply)` を返す。
  `ply` は root からの距離で、**速い詰まされほど低評価/速い詰ましほど高評価**(negamax で符号整合)。
- 値域は干渉しないよう設計: 評価 `±29000` < `VALUE_MATE=30000` < `VALUE_INF=1,000,000`(αβ窓の初期値)。
  単体テストで「評価は必ず詰み帯の内側」「詰みは必ず評価より上」を固定。
- Task 021 の「詰ます手に +∞」は、この `-(MATE-ply)` 終端へ統合(多段で筋が通る形)。

## 3. nega-alpha 実装(`src/search.rs`)

- `search_root(net, pos, depth, prune) -> SearchResult{best_move, score, nodes, depth}`。
  ルートで `pos.clone()` した作業盤を使い、内部は `do_move/undo_move` で進める(子盤面コピーしない)。
- `nega_alpha(net, pos, depth, ply, alpha, beta, prune, nodes)`:
  - 合法手0 → 詰み値。`depth==0` → 葉(NNUE評価を整数化)。
  - 各手 `do_move → -nega_alpha(child, depth-1, ply+1, -beta, -alpha) → undo_move`、
    `alpha` 更新、`alpha>=beta` で β cutoff。
  - `prune=false` で枝刈り無効(全幅ミニマックス)= ベンチでαβ効果を測るため。
- ノード数(全 nega_alpha 呼び出し)をカウント。
- ムーブオーダリングは無し(スコープ外)。今は素直な合法手順。葉でのみ NNUE 評価。

## 4. ベンチ(測定基盤)— `nnue-bench <weights> [max_depth]`

3局面(序盤/中盤/終盤)で深さ別に node 数・時間・手・score、および αβ vs 全幅の削減率を出力:

| 局面 | depth | αβ nodes | time | 全幅 nodes | αβ削減 | best / score |
|---|---|---|---|---|---|---|
| startpos | 2 | 279 | 0.3ms | 931 | 70.0% | 3i3h / 135 |
| startpos | 3 | 2,110 | 2.5ms | 26,401 | **92.0%** | 7g7f / 167 |
| startpos | 4 | 32,249 | 34ms | — | — | 7g7f / 72 |
| midgame | 2 | 1,398 | 2.3ms | 37,463 | 96.3% | 7e6d / 1690 |
| midgame | 3 | 248,348 | 392ms | 7,694,536 | **96.8%** | 7e6d / 1520 |
| midgame | 4 | 5,710,942 | 9.7s | — | — | G*1g / 1038 |
| endgame | 3 | 1,049 | 0.9ms | 4,066 | 74.2% | 5g5c+ / **29997** |

- **αβ枝刈りが効いている**(深さが増すほど削減率が上がる)。
- 終盤 depth3 の score=29997 = `MATE-3` → **3手詰めを発見**して `5g5c+` を選ぶ。
- 性能の所見: ムーブオーダリング無しのため中盤 depth4 は 5.7M node/9.7s と重い。**置換表・反復深化・
  ムーブオーダリングを入れたとき、この同一局面・同一深さで node/time がどれだけ減るかが次段の評価基準**。

## 5. 動作確認(段階A〜C)

- **段階A(USIスモーク)**: `EvalFile`+`Depth 3` で NNUE をロードし合法 bestmove(`7g7f`)。
- **段階B(視点・符号)**: 終盤局面で `5g5c+`(詰ます手)を選ぶ。`Depth` で手が変わる
  (中盤で d1=`N*8e` 貪欲、d3=`7e6d` 深い読み)→ 符号正しく深さが効いている。
- **段階C(深さ別自己対戦)**:
  - **depth3 対 depth1: 50戦50勝(100%)**。両者決定的なので**乱数オープニング(25種×色入替)**で
    多様化した上での結果 = 読みの深さが効いている確かな証拠。
  - depth3 対 Random: 50戦50勝(回帰確認)。

> 注意点として記録: NNUE エンジンは決定的なので、startpos からの自己対戦は全局面が同一棋譜になる。
> 意味のある「複数局」にするため、同一オープニングを色入替で2局ずつ当てる乱数オープニングを selfmatch に入れた。

## 6. ShogiHome で深さ違いエンジンを2つ登録(段階D、作者向け)

同じ `target/release/binbo-shogi` を **表示名を変えて2回登録**し、それぞれ深さを変える:

1. エンジン管理 → 追加 → `target/release/binbo-shogi` を選択 → 表示名 `binbo-d1`。
   USI option: `EvalFile`=`nnue_f32.bin` の絶対パス、`Depth`=`1`。
2. もう一度 追加 → 同じバイナリ → 表示名 `binbo-d3`。`EvalFile`=同じパス、`Depth`=`3`。
3. `binbo-d1` 対 `binbo-d3` で対局させる(深さの効果を体感)。作者と指す場合は `binbo-d3`。
4. 重みは現状 `/tmp/binbo-task017/model_wd1e-3/nnue_f32.bin`(再起動で消える)。永続利用は安定パスへコピー。

(`Eval` は既定 `nnue`。`Depth` は `Eval=nnue` のとき有効。`Eval=random` で従来の Random に戻せる。)

## 回帰

- `cargo test --release --locked` 15/15(search の単体テスト2件を含む)。`cargo run -- verify` perft 1〜5 完全一致。
- RandomEngine `selfplay` 完走、option 無指定の USI は従来どおり Random フォールバック。ビルド警告なし。

## 成果物の場所

- コード(コミット): `src/search.rs`(新規)、`src/main.rs`(NnueEngine の αβ化、`Depth` option、
  `nnue-bench`/`nnue-selfmatch`、`nnue-match` の深さ対応)。
- 重み(非コミット・再生成可): `/tmp/binbo-task017/model_wd1e-3/nnue_f32.bin`。

## スコープ外(次段で1つずつ、効果を測りながら)

- 置換表(+ GHI/千日手)、反復深化、ムーブオーダリング(TT手/MVV-LVA)、MTD(f)/PVS、静止探索、時間管理。
- cp の厳密な意味づけ、差分更新・SIMD・量子化。
