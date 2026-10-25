# Task 072-pre — gikou(技巧2)depth≥8 crash の原因究明 + d12 ラベル生成の可否

**目的**: 「より強い教師アンカー」= 技巧2@d12 ラベルを得たい。だが gikou は depth≥~8 で warm-state EOF crash(`main.rs:2242`、Task045: d10 で games 82.4% crash)。d12 で **安定かつ無偏(decisive 局面を失わない)** なラベル生成レシピがあるかを実証する(無ければ 072 を止める)。

## 方法(調査のみ・小規模)
- **raw-USI 制御プローブ**(`scratchpad/gikou_probe.py`): gikou を直接 USI 駆動し、warm(1プロセスで連続 `go depth D`)/ cold(局面ごとに fresh プロセス)で d8/d12 を比較。
- **gen-teacher 実走**(`gen-teacher@d12`, shard-size 200, workers 12, random-plies 6)で 2k/50k を生成し、crash/restart/throughput と **eval 分布**を実測。
- **crash 局面の cold 再評価**: gen-teacher が「abandoning game」した実際の sfen を fresh gikou で d8/d12/d16 検証。
- **d8 baseline**: `shared_32m`(=gen-teacher@d8 出力、31.95M)の eval 分布(3M sample)。

## 結果① — crash は「局面」でなく「warm-state 累積」
- **warm d8**: 1プロセスで 80 連続 `go depth 8` = **crash なし**(対照)。
- **warm d12**: gikou 自身の最善手で自己対局 → 80 連続 d12 = **crash なし(×3 game)**。Hash256/Threads4 でも crash なし。
- **cold d12/d16**: gen-teacher が **abandoning game した 4 局面**を fresh gikou に与えると、**d8/d12/d16 すべて 0.0–0.2s で bestmove を返し crash しない**。
- → **crash 局面そのものは健全**。crash は **その局面に至るまでの warm-state 累積**(連続探索でエンジン内部状態が壊れる)で起き、abandon メッセージの sfen は「壊れた状態で次に来た局面」にすぎない。`usinewgame` では戻らず、**fresh プロセスだけが reset**(`gen.rs:252` のコメントと整合)。
- gen-teacher が warm session で crash するのは `--random-plies` 由来の自己対局を ~50–100 手連続探索して warm-state が育つため(私の「最善手自己対局」プローブは育ちにくく 80 手生存した)。

## 結果② — gen-teacher@d12(fresh バッチ=承認済の機構)は**致命的に偏る**
50k @ d12(shard-size 200, workers 12): total 50000, **crash_cutoffs=441(0.88%/局面)/ restarts=441 / timeouts=0 / shards_failed=0**(全回復)。throughput **~61 pos/s @12w → 1M ≈ 4.5h**。

| 指標 | **d12(gen-teacher 50k)** | d8 baseline(shared_32m) | |
|---|---|---|---|
| mean &#124;cp&#124; | **310.9** | 1261.3 | 4× フラット |
| decisive &#124;cp&#124;≥1000 | 4.84% | 31.53% | 6.5× 低 |
| decisive &#124;cp&#124;≥2000 | 1.21% | 18.35% | 15× 低 |
| decisive &#124;cp&#124;≥5000 | **0.03%** | 5.37% | **180× 低** |
| cp==0 | 7.30% | 3.79% | より「互角」 |

- **decisive 局面がほぼ消滅**(≥5000 が 5.37%→0.03%)。gikou は鋭い/決着的なラインに入ると warm-state crash → gen-teacher がそのゲームを **abandon** → 生き残るのは**静かな互角局面ばかり**(mean&#124;cp&#124; 311)。Task045 の d10 偏り(5.3%→0.41%)が d12 で更に悪化。
- 補足(交絡): d12 自己対局は d8 より**強い均衡対局**になり自然に eval が平坦化する寄与もありうる。**どちらにせよ** fresh バッチは「strong/sharp なラベル」を与えるどころか **d8 より弱く平坦なラベル**=「より強い教師」の狙いと真逆。**→ option(ii) fresh d12 バッチは不可**。

## 結果③ — 無偏な d12 ラベルは「cold 再ラベル」でのみ得られる
- cold gikou@d12 は**全局面を crash せず評価**(結果①)。よって **既存の決着豊富な局面集合**(shared_32m=d8 自己対局, decisive≥5000 が 5.37%)を **cold gikou@d12 で付け直せば**、(a) 局面分布は据え置き(交絡なし=純粋なラベル深さ d8→d12 の一変数)、(b) crash でゲームを捨てない=**decisive 局面を失わない(無偏)**。
- これは当初ユーザが選ばなかった **「同一局面の厳密再ラベル」**(option i)。072-pre は **fresh batch と re-label の優劣を反転**させた:fresh batch=偏る/交絡、cold re-label=無偏/clean。
- 必要な実装: gikou 再ラベル driver(`dlshogi_rescore.py` の cache-aligned パターン + cold gikou: **K 局面ごとに proc 再起動**で warm-state を crash 前に reset、crash 時は同一局面を cold で **再探索(abandon しない)**)。throughput は warm gen-teacher と同等 ~数十 pos/s/worker 級 → 1M ≈ 数時間。新規コード=中程度。

## GATE 判定
- **承認済の fresh d12 バッチ(gen-teacher@d12)= 不採用**(decisive 消滅で d8 より弱いラベル)。
- **無偏 d12 は cold 再ラベルで実現可能**(技巧2@d12 自体は cold で健全、d16 も可)。ただし新ツールが必要で、**機構がユーザ承認(fresh batch)と異なる**ため、072 本走前に方針を確認する。
- 代替: ② 探索ベース方策改善 / ③ 技巧2@d8 を天井と受容(BACKLOG)。

## Part 2 試行 — cold 再ラベルの実装と実測(行き止まり)
GATE で選んだ cold 再ラベル(`trainer/gikou_relabel.py`、dlshogi_rescore の cache-aligned パターン + cold gikou + restart-every-K + EOF cold-retry + watchdog stop)を実装し実測:

- **重要な実装上の落とし穴(修正済)**: gikou は eval(`params.bin`/`progress.bin`)を **CWD から読む**。CWD を gikou ディレクトリに設定しないと **zero-eval で起動**し、`go depth` が**全局面 score cp 0** を返す(探索はする=bestmove は妥当なので対局では露見しない)。Rust `GikouEngine` は `gikou.rs:419 .current_dir(parent)` で対処済。Python driver に `cwd=dirname(gikou)` を追加して解決(startpos→cp160、決着局面→正しい大cp)。**move 番号は eval に無影響**(mv 1/67/131 で同値=progress.bin は局面から進行度推定)、gikou は決定的(Threads1)。現 binary は shared_32m 生成 binary(消失)と微差(decisive 局面で median ~24cp、~2%=build 差)で実用上一致。
- **致命的なコスト/技術問題(行き止まり)**:
  - **throughput**: 序盤(静か)は速い(13–41/s/proc)が、**中終盤局面は ~1/s/proc**(d12 探索が秒オーダー、TT 連続性のない disjoint 再ラベルで gen-teacher の自己対局より遅い)→ **1M ≈ ~46h**(16 workers でも ~17h)。
  - **crash/contamination**: 中終盤 decisive 局面は **crash 14% + hard-fail 9%**(cold-retry でも EOF=fresh でも落ちる局面が存在)。hard-fail は d8 へ fallback → **「最も欲しい decisive 局面」ほど d12 化に失敗**(= cold でも decisive を無偏に d12 化できない)。watchdog stop(cap-s 2)でも中終盤は cap に張り付く。
  - **そもそも d12 ≈ d8(技巧2 は早期飽和)**: 現 build で d8 vs d12 の差は **median ~71cp**、決着局面でも **+100–400cp 程度**(4920→5041, −4729→−4963, 5005→5416)=方向は d12 がわずかに鋭いが**僅差**。065 の「champion@YaneuraOu は d16 まで cp 改善継続」は**自前 NNUE の話**で、**技巧2(成熟 KPP エンジン)は d8 で実質飽和**。
- **Part 2 の結論**: **gikou@d12 は「より強い教師」になり得ない**。(1) 無偏 d12 ラベルは decisive 局面で原理的に取りにくい(cold でも crash)、(2) コストが ~46h/1M + contamination、(3) そもそも d8 と僅差。067-071(より大きなラベル変更でも champion 未超)を踏まえると、僅差の 20% d12-replay が champion を超える見込みは低く、投資対効果が著しく悪い。**stronger-teacher-via-deeper-gikou(option ①)は行き止まり**。

## 総合 GATE 判定(072-pre + Part 2)
- option ①(より深い技巧2 で replay 強化)= **棄却**(技巧2 は d8 飽和 + 深い labeling が crash/コストで非現実的)。
- 副産物(コミット可): `trainer/gikou_relabel.py`(cache-aligned cold gikou 再ラベル、cwd 修正済、d8 では高速・健全に動作=将来の同一教師 re-score 基盤)。`reports/task072pre_gikou_d12.md`。
- 次は人間の方針判断: ② 探索ベース方策改善(AlphaZero 流, アーキ大改修)/ ③ 技巧2@d8 蒸留を本アーキの天井と受容し目標再設定。

## 再現
gikou=`$GIKOU`(~/binbo-shogi-data/engines/gikou/gikou、**CWD=同ディレクトリ必須**)。プローブ=`scratchpad/gikou_probe.py`/`gikou_sfen_crash.py`、hcpe 分布=`scratchpad/hcpe_dist.py`(eval=i16 LE @offset32, 38B/entry)。cold 再ラベル=`trainer/gikou_relabel.py`(cwd 修正 + watchdog)。d8 baseline=shared_32m train_evals。i7-14700KF。crash は EOF(timeouts=0)。
