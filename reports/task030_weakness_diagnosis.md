# Task 030: 弱さの原因切り分け — 非対称ハンデ + 読み深さ比較

## 結論

- `nnue-vs-gikou` を拡張し、生徒と技巧2に**別々の持ち時間**を指定できるようにした。
  併せて、各手・各局・全体の **到達深さ / ノード数 / 実思考時間**を集計する。
- 技巧2 Lv1 の `info` は実物から取得できた。今回の全条件で **技巧2 Lv1 は depth=1**、
  平均ノードは約100〜140、実思考時間は1ms未満だった。
- 生徒は同じ1秒条件で平均 depth≈5、約63万ノードまで読んでいる。それでも技巧2 Lv1(depth1)に全敗。
- 生徒の時間を 1秒→3秒→10秒に増やすと、到達深さは約5.0→5.5→6.4へ伸びるが、勝ちは出ない。
  10秒条件では1局が最大手数引分になっただけで、勝ち切れていない。

**見立て: 主因は第二層(評価の質)。**

探索効率で「技巧2の方が深く読んでいる」わけではない。むしろ生徒の方が何十万〜何百万ノード読んでいる。
それでも depth1 の技巧2に負けるため、現行 NNUE 評価の質が技巧2の評価より大きく粗い、という示唆が強い。
読み量を増やすと多少粘るが、10秒でも勝ちに転じないため、第三層(読みの量)だけでは埋まらない。

## 実装

- `src/teacher/gikou.rs`
  - 対局用 `GikouMove` に `elapsed`, `reported_depth`, `nodes` を追加。
  - `bestmove_sfen` で Gikou の `info depth ... nodes ...` をパースして返す。
- `src/main.rs`
  - `nnue-vs-gikou` の引数を拡張:
    - `... [depth] [binbo_think_ms] [gikou_think_ms]`
    - `gikou_think_ms` 省略時は従来互換で `binbo_think_ms` と同じ。
  - 生徒側は `ThinkResult` に search result の `score/depth/nodes` を詰める。
  - 対戦集計に、双方の `avg_depth`, `max_depth`, `avg_nodes`, `max_nodes`, `avg_time_ms` を追加。

## 実験条件

- binbo weights: `~/binbo-shogi-data/models/model_wd1e-3/nnue_f32.bin`
- 技巧2 binary: `/tmp/binbo-task007/gikou-run/release` (`Gikou 2 v2.0.2`)
- 技巧2: Lv1 (`DepthLimit=1`), `OwnBook=false`, `Threads=1`, `MinThinkingTime=10`
- binbo: depth上限12、TT/反復深化/ハッシュムーブ/MVV-LVA有効
- ランダムオープニング: 6手、同一オープニングで先後入替
- 逐次実行。並列なし。

注: 10秒条件は実測で1局5分超になったため、先後1ペア・最大80手に縮小した。これは精密測定ではなく原因切り分けなので、
長時間化を避けて傾向確認を優先した。

## 非対称ハンデ: 技巧2 Lv1 1秒固定、生徒だけ時間増

| 条件 | games | binbo成績 | score rate | binbo avg depth | binbo avg nodes | binbo avg ms | Gikou avg depth | Gikou avg nodes | Gikou avg ms | 備考 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 生徒1秒 / 技巧2 1秒 | 4 | 0-4-0 | 0.0% | 4.98 | 635,674 | 839.03 | 1.00 | 136 | 0.60 | 全局 `no_legal_moves` 負け |
| 生徒3秒 / 技巧2 1秒 | 4 | 0-4-0 | 0.0% | 5.47 | 1,925,534 | 2,624.42 | 1.00 | 143 | 0.53 | 全局 `no_legal_moves` 負け |
| 生徒10秒 / 技巧2 1秒 | 2 | 0-1-1 | 25.0% | 6.41 | 6,217,595 | 8,434.06 | 1.00 | 106 | 0.60 | 1敗 + 最大80手引分 |

### 読み量の伸び

- 1秒→3秒: 生徒平均深さは 4.98→5.47。ノードは約3.0倍。勝率は 0% のまま。
- 3秒→10秒: 生徒平均深さは 5.47→6.41。ノードは約3.2倍。1局は最大80手まで粘ったが勝ちは無し。
- Gikou Lv1 は全条件で depth1。`gikou_think_ms=1000` を渡しても、`DepthLimit=1` のため実時間はほぼ使わない。

## 対称条件の軽い確認

| 条件 | games | binbo成績 | score rate | binbo avg depth | Gikou avg depth | 備考 |
|---|---:|---:|---:|---:|---:|---|
| 1秒同士 | 4 | 0-4-0 | 0.0% | 4.98 | 1.00 | 非対称表の基準条件と同じ |
| 5秒同士 | 2 | 0-2-0 | 0.0% | 5.81 | 1.00 | Gikouは `DepthLimit=1` なので5秒を使わない |

対称に時間を増やしても順位は不変。実際には Gikou Lv1 が depth1 制限で時間を使わないため、5秒同士は
「生徒5秒 / Gikou depth1」の確認に近い。

## 切り分け

### 第一層: 探索効率

- 「同じ時間で技巧2の方が深く読める」という仮説は、この Lv1 条件では否定的。
- 技巧2 Lv1 は常に depth1、約100ノード、1ms未満。
- 生徒は1秒条件で depth約5、約63万ノード、10秒条件では depth約6.4、約622万ノード。
- それでも勝てない。探索効率以前に、評価の質で大きく負けている可能性が高い。

### 第三層: 読みの量

- 時間を増やすと深さとノードは素直に増える。
- 10秒では1局が80手まで持ちこたえ、読み量で粘りは増える。
- ただし勝ちは出ていない。読み量だけで Lv1 を超える兆候は弱い。

### 第二層: 評価の質

- 最も強い示唆はここ。
- depth1 の技巧2が、depth5〜6の生徒に勝つ。これは技巧2の静的評価/浅い探索の手選択が、
  現行 NNUE の深読みを上回っているということ。
- 生徒が技巧2 Lv1より深く読んでいるのに負けるため、次の主戦場は探索速度ではなく評価品質。

## 次に注力すべきこと

優先度は以下。

1. **評価品質の診断**
   - 生徒 NNUE の局面評価と技巧2 depth1/8 の評価を同一局面で比較する。
   - 負け棋譜の局面で、どの手をなぜ悪く選ぶかを見る。
   - `pv` 出力や候補手上位の評価一覧があると診断しやすい。
2. **再学習・データ改善**
   - 100万局面モデルの容量・正則化・データ分布・勝率変換スケールを見直す。
   - 必要なら教師データを増やす/深くする前に、まず現データでの overfit/validation と実戦弱さの関係を確認する。
3. **探索品質の改善**
   - 静止探索は評価の粗さを直接緩和できる可能性がある。
   - キラー/ヒストリーは速度改善として有効だが、今回の主因ではなさそう。

## 実行ログ抜粋

```text
# 生徒1秒 / 技巧2 1秒
binbo_record=0-4-0 errors=0
binbo_avg_depth=4.98
binbo_avg_nodes=635674.05
binbo_avg_time_ms=839.03
gikou_avg_depth=1.00
gikou_avg_nodes=136.08
gikou_avg_time_ms=0.60

# 生徒3秒 / 技巧2 1秒
binbo_record=0-4-0 errors=0
binbo_avg_depth=5.47
binbo_avg_nodes=1925533.90
binbo_avg_time_ms=2624.42
gikou_avg_depth=1.00
gikou_avg_nodes=143.31
gikou_avg_time_ms=0.53

# 生徒10秒 / 技巧2 1秒
binbo_record=0-1-1 errors=0
binbo_avg_depth=6.41
binbo_avg_nodes=6217595.14
binbo_avg_time_ms=8434.06
gikou_avg_depth=1.00
gikou_avg_nodes=106.00
gikou_avg_time_ms=0.60

# 5秒同士
binbo_record=0-2-0 errors=0
binbo_avg_depth=5.81
binbo_avg_nodes=2971051.89
binbo_avg_time_ms=4011.94
gikou_avg_depth=1.00
gikou_avg_nodes=112.74
gikou_avg_time_ms=0.56
```

## 回帰

- `CARGO_HOME=/tmp/binbo-task011-cargo cargo test --release --locked --offline` → 16/16 passed
- `CARGO_HOME=/tmp/binbo-task011-cargo cargo run --release --locked --offline -- verify` → perft 1〜5 完全一致
- `target/release/binbo-shogi selfplay 512 1` → 211手目 `resign`、非合法手/クラッシュなし
