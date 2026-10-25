# Task 029: 技巧2レベルを物差しにした棋力当たり測定

## 結論

- `nnue-vs-gikou` CLI を追加し、binbo NNUE と技巧2を **逐次・同一時間条件・初手ランダム・先後入替**で対戦できるようにした。
- 技巧2のレベル設定は一次情報で確認:
  - 実機 `usi` 応答: `option name DepthLimit type spin default 100 min 1 max 100`
  - `/tmp/binbo-task007/gikou-run/gikou_ja.txt`: `DepthLimit = 読みの深さ(強さのレベル調節用)`
  - 技巧2 README: Lv1≈将棋倶楽部24 R800(8級)、Lv2≈R1100、Lv3≈R1400、Lv4≈R1600、Lv5≈R1800。短時間ではレベルを上げても強さが上がらないことがある。
- 測定結果: **現行の生徒 NNUE は、1秒/手・depth上限12でも技巧2 Lv1に 0勝8敗**。2秒/手の確認でも 0勝2敗。
  - したがって今回の当たりは **技巧2 Lv1未満、R800未満相当の可能性が高い**。
  - 少数局なので精密レートではない。ただし下限 Lv1 に全敗しており、二分探索は Lv1 で打ち切りが妥当。

## 実装

- `src/teacher/gikou.rs`
  - `GikouOptions.depth_limit` と `min_thinking_time_ms` を追加。
  - 対局用 `bestmove_sfen(sfen, go_command, timeout)` を追加。`bestmove` を必ず読んでから次の指示へ進む。
- `src/main.rs`
  - `nnue-vs-gikou <weights_bin> <gikou_path> <level> <games> [seed] [max_plies] [opening_plies] [depth] [think_ms]`
  - 技巧2は各局 fresh process、`OwnBook=false`, `Threads=1`, `DepthLimit=<level>`, `MinThinkingTime=10`。
  - binbo は `movetime=think_ms+500ms` を渡し、内部の 500ms 安全マージン後に概ね `think_ms` を使う。
  - 技巧2は `go btime 0 wtime 0 byoyomi <think_ms>`。
  - 決定性対策: games `2k/2k+1` は同じランダムオープニング、binbo先後を入替。
  - 終了処理: 合法手なし=負け、`resign`、同一 `pos.key()` 4回=引分、`max_plies`=引分、技巧2異常は error として勝敗に混ぜない。

## 測定条件

- binbo weights: `~/binbo-shogi-data/models/model_wd1e-3/nnue_f32.bin`
- 技巧2 binary: `/tmp/binbo-task007/gikou-run/release` (`Gikou 2 v2.0.2`)
- 技巧2: `OwnBook=false`, `Threads=1`, `DepthLimit=1`, `MinThinkingTime=10`
- binbo: depth上限12、TT/反復深化/ハッシュムーブ/MVV-LVA有効
- ランダムオープニング: 6手、先後入替
- 最大手数: 180手

## 結果

### 当たりつけ: Lv1, 1秒/手

| seed | games | binbo成績 | score rate | 平均手数 | elapsed |
|---|---:|---:|---:|---:|---:|
| 20260622 | 4 | 0-4-0 | 0.0% | 80.0 | 125.1s |
| 20260623 | 4 | 0-4-0 | 0.0% | 76.0 | 119.1s |
| **合計** | **8** | **0-8-0** | **0.0%** | **78.0** | **244.1s** |

全局 `reason=no_legal_moves` で binbo 負け。技巧2の illegal bestmove、USI同期エラー、timeout は無し。

### 下限確認: Lv1, 2秒/手

| seed | games | binbo成績 | score rate | 平均手数 | elapsed |
|---|---:|---:|---:|---:|---:|
| 20260624 | 2 | 0-2-0 | 0.0% | 61.5 | 94.9s |

時間を2秒/手へ増やしても 0-2。互角ゾーンは Lv1 より下と判断。

### 最大手数打ち切りスモーク

| 条件 | 結果 |
|---|---|
| Lv1, 50ms/手, max_plies=8, games=2 | 2局とも `draw`, `reason=max_plies` |

千日手(`repetition_key4`)は今回の実測では発生なし。最大手数打ち切りは動作確認済み。

## 二分探索の経路

本来は Lv1→2→4→8... で上げ、負けたら下げる予定だったが、**下限の Lv1 で 0-8** だった。
よって上位レベルの探索は不要。二分探索は「範囲外: Lv1未満」として終了。

## 棋力の当たり

- 条件: 1秒/手、depth上限12、ランダムオープニング6手、先後入替、技巧2定跡オフ。
- 結論: **binbo は技巧2 Lv1未満**。
- 技巧2 README の目安では Lv1 は将棋倶楽部24 **R800程度 / 8級相当**。
- したがって現在の生徒は **R800未満相当の可能性が高い**。
- これは少数局の粗い測定であり、精密な Elo ではない。ただし Lv1 全敗なので「互角は Lv1より下」という判断は十分強い。

## 所見

- 今回の測定は30分以内どころか、主測定8局+2秒確認2局で約5.7分。逐次実行でも十分現実的。
- Gikou Lv1未満なので、現時点では精密棋力測定より先に、評価・探索・学習データの改善が必要。
- 次に Lv1 に勝ち越せる兆候が出たら、Lv1/Lv2 の境界で 20局程度に増やして再測定する。
- 千日手の正式実装は未対応のまま。今回の測定では発生しなかったが、長局・同型反復が増えるなら正式対応が必要。

## 実行ログ抜粋

```text
nnue-vs-gikou ... level=1 games=4 seed=20260622 max_plies=180 opening_plies=6 depth=12 think_ms=1000
binbo_record=0-4-0 errors=0
binbo_score_rate=0.0% (wins + 0.5*draws over 4 scored games)
avg_plies=80.0
elapsed_ms=125055.4
status=ok

nnue-vs-gikou ... level=1 games=4 seed=20260623 max_plies=180 opening_plies=6 depth=12 think_ms=1000
binbo_record=0-4-0 errors=0
binbo_score_rate=0.0% (wins + 0.5*draws over 4 scored games)
avg_plies=76.0
elapsed_ms=119085.4
status=ok

nnue-vs-gikou ... level=1 games=2 seed=20260624 max_plies=160 opening_plies=6 depth=12 think_ms=2000
binbo_record=0-2-0 errors=0
binbo_score_rate=0.0% (wins + 0.5*draws over 2 scored games)
avg_plies=61.5
elapsed_ms=94867.9
status=ok
```

## 回帰

- `CARGO_HOME=/tmp/binbo-task011-cargo cargo test --release --locked --offline` → 16/16 passed
- `CARGO_HOME=/tmp/binbo-task011-cargo cargo run --release --locked --offline -- verify` → perft 1〜5 完全一致
- `target/release/binbo-shogi selfplay 512 1` → 211手目 `resign`、非合法手/クラッシュなし
