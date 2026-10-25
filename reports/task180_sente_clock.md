# Task 180: 先手3分の使い方と, 時計を変えて比べる道具 (2026-10-24)

**最初の候補は, ブイ後の自分の15手まで `go` の `btime` を30秒以下にする中継**。やねうら王は変更しない。初手の配分は最適6.171秒・最大17.880秒から, 4.016秒・8.880秒になる。30秒は「1手30秒」でも「序盤合計30秒」でもなく, エンジンに見せる残り時間の上限。効くかは未確認で, 採用を勧める段階ではない。

既存の定跡比較から, 定跡なし先手の序盤15手 **90.5秒 [83.25, 96.5]**, 最小残り **2秒 [2, 3.75]** を再現した。以下の角括弧は, 特記しなければ25%点・75%点。計測・時計付き対局・先手への介入・単体USI中継・対応付き集計を用意した。実エンジンの短時間10局は完走。**大きな実験は実施していない**。

成果物の正本は `~/binbo-shogi-data/work/task180_clock/`。この文書だけ `reports/task180_sente_clock.md` にコピーした。既存コード・STATUS.md・大会設定は変更していない。コミットなし。指定資料は AGENTS.md → STATUS.md → DESIGN.md → 大会準備・定跡調査・10/23方針C・当日手順書の順で読んだ。

## 時間管理の式

対象はローカルのやねうら王ソース `820940ee31762ad598a9156a615b4b896ab8b321`。`source/timeman.cpp`, `.h`, `search.h`, `usi.cpp`, `engine/yaneuraou-engine/yaneuraou-search.cpp` を読んだ。実行ファイルは USIで `YaneuraOu NNUE 9.41git 64AVX2` と名乗る。各ソース・実行ファイル・採用evalのSHA256は `summary.json` に記録した。ソースの追跡ファイルに差分なし。

以下はすべてms, 整数除算は切り捨て。`B=btime`, `I=binc=2000`, `p=局面のSFEN手数` (平手は1, ブイ4手後は5), `D=MaxMovesToDraw=512`, `M=MinimumThinkingTime=2000`, `d=NetworkDelay=120`, `d2=NetworkDelay2=1120`, `s=SlowMover=100`。後手にはwtime/wincを同じ式へ入れる。相手の残り時間はこの式に直接入らない。

```text
H = 180 - min(p, 80)                       # 加算または秒読みがあるとき
G = floor(min(D - p + 2, H) / 2)           # 残りの自分の手番の見積もり
R = max(B + I - d2, 100)                   # 今回使える残り
m = max(M - d, 1000)
E = max(B + I*G - (G+1)*1000, 0)
t1 = m + floor(E/G)
t2 = min(m + int(float32(E)*5.0f/G), int(E*0.3))
O = floor(min(t1,R)*s/100)
O = O + floor(O/4)                        # USI_Ponder=true && Stochastic_Ponder=false のとき
minimum = min(Q(m),R)
optimum = min(O,R)
maximum = min(Q(min(t2,R)),R)
```

`Q(t)` は, `u=max(ceil(t/1000)*1000,M)-d` として, `u<t` ならさらに1000を足し, `min(u,R)` を返す。`optimum` 自体は秒へ切り上げない。`G=1` はminimum/optimum/maximumをすべてRにして即return, `G<=0` は診断を出して500にする。序盤で512の上限が効くことはなく, pが415以上になってから終局までの手数がHより短くなる。

秒読み `Y=byoyomi` があればRにYを足し, Eにも `Y*G` を足す。`I=Y=0` の切れ負けでは `H=200-min(p,40)`, 倍率5を `min(5,max(B/60000,1))` に抑える。`RoundUpToFullSecond=false` ではRの下限100とmの下限1000を1にし, Eの `(G+1)*1000` を引かず, Qを `min(max(t,M)-d,R)` にする。秒読みありで `B<int(1.2*Y)` なら, 最終の丸め・Rへの制限の前に3種の時間をB+Yにそろえる。`rtime` と `movetime` には別の固定時間経路がある。depth/nodes/mate/infinite/perft指定時は通常の時間管理を使わない。Stockfish側のnodestimeはこのビルドの時間管理には使われない。

本番の最初は `go btime 180000 wtime 600000 binc 2000 winc 2000`, p=5, G=87, E=266000, R=180880, m=1880。t1=4937, ponder補正後6171。t2=17167 → Qで17880。最低は1880。

| 初手でエンジンに見せる残り | ponder補正なしoptimum | 本番のoptimum | maximum |
|---|---:|---:|---:|
| 180秒 (既定) | 4.937秒 | 6.171秒 | 17.880秒 |
| 60秒 | 3.558秒 | 4.447秒 | 10.880秒 |
| 30秒 | 3.213秒 | 4.016秒 | 8.880秒 |
| 10秒 | 2.983秒 | 3.728秒 | 7.880秒 |
| 2秒 | 2.880秒 | 2.880秒 | 2.880秒 |
| 0秒 | 0.880秒 | 0.880秒 | 0.880秒 |

このmaximumは壁時計の厳密な停止上限ではない。探索終了予定を決める `set_search_end(elapsed)` が, **maximumを越えた時点からさらに秒へ切り上げる**。512ノードごとの時間チェック, 最初の反復, OSの待ち時間もある。通常の無ponderではmaximumよりほぼ1秒長くなることがあり, 最後はRで制限される。

探索側は反復ごとに `totalTime = optimum * F * reduction * instability * effort` として使う。

```text
F = clamp((11.325 + 2.115*(前手の平均評価-bestValue)
                   + 0.987*(iterValue[iterIdx]-bestValue))/100, .5688, 1.5698)
tr = .723 + .79/(1.104 + exp(-.5189*(completedDepth-(lastBestMoveDepth+11.57))))
reduction = (1.455 + previousTimeReduction)/(2.2375*tr)
instability = 1.04 + 1.8956*totBestMoveChanges/Threads
effort = .666 if depth>=10 and 最善手のnodesEffort>=92425 else 1
```

合法手が1つならtotalTimeを502ms以下にする。`elapsed > min(totalTime,maximum)` で終了予定を立てる。したがってoptimumは平均消費時間をそのまま予測する式ではない。最善手の変更・評価の悪化・反復の深さとponderの当たり外れで動く。

Ponder中の時間チェックは停止を保留し, 条件を越えれば `stopOnPonderhit` を立てる。通常のponderhitは時刻を保存してponderフラグを下げ, **時間配分のinitをやり直さない**。goからのelapsedで探索量を見つつ, hit以後の時間で秒の切り上げを行う。`USI_Ponder=true` にしただけでもoptimumの25%増しは働く。`parse_ponderhit()` の拡張パーサは定義されているが, このソースでは呼び出されていない。ponderhitに新しい時計を付けて再配分できるとは扱わない。

## 現存ログで何が起きているか

`analyze_logs.py` はCSAのTを読み, ブイの手を含め `残り += increment - T` を再生する。`T422` と他のブイの `T2` により600秒が180秒になる。序盤・最低残りの集計からブイ4手を除く。サーバ保存を優先し, 同じ `$EVENT` のbridgeコピーを重ねない。宣言直前・512手直前の長いブイは除外した。`denryusen/stage3/` は現存しないため, 準備文書のWSL試験の時間切れ5局を新たに検証したとはしていない。

| 先手の集合 | 局数 | 最初15手の消費 秒 (中央値 [25%,75%]) | 局ごとの最低残り 秒 |
|---|---:|---:|---:|
| Xubuntuの通常ブイ (`xub/srv`, 10/15) | 30 | 50.5 [45,56] | 2 [2,46.5] |
| 定跡比較・定跡なし (`xub/book1`) | 14 | 90.5 [83.25,96.5] | 2 [2,3.75] |
| 定跡比較・定跡あり | 26 | 12 [0.25,26.75] | 38 [2,84.5] |
| floodgate・定跡なし | 12 | 263.5 [207.25,332.5] | 76.5 [2,197] |
| floodgate・定跡あり | 2 | 80 [40,120] | 276.5 [259.75,293.25] |

floodgateは**300秒+10秒, ブイなし**。大会の180秒+2秒と直接比較しない。eval・対戦相手・時期も混ざる。通常ブイと定跡比較にもスレッド数や相手の違いがある。熱による速度低下については定跡調査の留保を引き継ぐ。ほかに熱試験4局・本番通し2局・別の作業の時間つき対局4局があり, `summary.json` では分けて残した。

定跡なし14局の自分の手ごとの内訳 (ブイ後の自分の手を1から数える)。消費はCSAの秒未満切り捨て, 残りは指した直後。

| 自分の何手目 | n | 消費 秒 中央値 [25%,75%] | 残り 秒 中央値 | 入力時計から計算したoptimum 秒 中央値 |
|---|---:|---:|---:|---:|
| 1 | 14 | 7.5 [6,14.75] | 174.5 | 6.171 |
| 5 | 14 | 4 [3,9] | 156.5 | 6.023 |
| 10 | 14 | 7 [2.5,11] | 135.5 | 5.8425 |
| 15 | 14 | 2 [1,4.75] | 119.5 | 5.679 |
| 20 | 14 | 1.5 [1,3] | 114 | 5.6855 |
| 30 | 14 | 2 [1,5.75] | 96 | 5.732 |
| 40 | 14 | 3 [1,6.75] | 77.5 | 5.625 |
| 60 | 13 | 3 [1,6] | 45 | 4.8 |
| 80 | 11 | 1 [1,5.5] | 10 | 3.925 |

残りのIQRと全ての自分の手番については `analysis_complete/by_own_move.tsv`。同じ表に通常ブイとfloodgateも入れた。通常ブイの1/15/40/80手目の消費中央値は10/2/1/1秒, 残りは172/159.5/98.5/8秒。floodgateの同じ位置では62.5/6/13/5.5秒, 残り247.5/186.5/25/8秒 (後半は生存局だけ)。長い局だけが後半に残る偏りにも注意する。

「加算だけ」を区別した。**着手前の残りが厳密に0秒の手は全集合で0**。実用上の枯渇に近い `着手前残り <= 加算` なら, 定跡なし14局で37/1362手 (2.72%, 8/14局), 通常ブイで71/2898手 (2.45%, 16/30局), floodgate定跡なしで113/653手 (17.30%, 5/12局)。残り5秒以下になった局は定跡なし11/14, 定跡あり10/26, 通常ブイ18/30。定跡なしで着手前残り10秒以下の手は220/1362手 (16.15%)。一方, **消費が2秒以下だった**手は791/1362手 (58.08%)あり, ponderhitや短い思考も含むので, これを「加算だけで指した率」とはしない。

USIログからも照合した (`audit_usi.py`, `usi_audit_complete.json`)。定跡なし先手の非ponderのgo 342回では, 壁時間/計算optimumの中央値1.000 [0.545,1.673]。ponderhitした手はこの比率から除いた。エンジン申告時間との差は0ms [0,1]。maximumより50ms以上長かった34回の超過は最大1001msで, 二度目の秒切り上げに整合する。序盤に長く読み, その後ponderhitと短い手で貯金を戻す局もあり, 単純にoptimumを足しただけでは時計を説明できない。

## 改変せずに動かせるもの

| 手段 | 予想する変化・留保 |
|---|---|
| SlowMover 100→70 | optimumを約0.7倍。minimum/maximumを直接下げない。最初の15手だけでなく全局に効く。序盤の節約候補だが勝負どころも短くする |
| MinimumThinkingTime 2000→1000 | 下駄を減らす。丸めありではmの下限1000, QにもMが入るため比例にはならない。短い手の節約候補。秒未満切り捨ての得も減る |
| NetworkDelay2を増す | Rを小さくし, 枯渇時により早く返して余裕を残す。序盤の配分がRより十分小さければほとんど効かない。対局終盤の時計の衛生向け |
| NetworkDelayを増す | 最低時間とQから引く分が増す。丸め境界でさらに1秒足されることもある。通信遅延用の余裕と強さの調整を混同しない |
| RoundUpToFullSecond=false | 秒への切り上げをやめる。同時にEから1手約1秒の予約を引かなくなるので, 常に短くなるとは限らない。本番の切り捨てルールで安易に変えない |
| Ponder / Stochastic_Ponder | 相手の時間で読める。通常ponderはoptimum25%増しとhitでの節約が一体。stochasticはinitや局面の流れも違う。今回は固定する |
| MaxMovesToDraw | 早い終局を設定すればGが減り, 逆に序盤を使いやすくなるほか裁定も変わる。大会の512を維持 |
| DepthLimit / NodesLimit, go depth/nodes/movetime | 通常の配分を外せる。局面の難しさへの適応を失い, 固定ノードはOS待ちによる時間切れを別に監視する必要がある |
| Threads / Hash / MultiPV / PvInterval / ResignValue | NPS・探索の反復や終了時点・出力負荷・対局の長さへの間接効果。配分比較では固定。DrawValueBlack/Whiteも今回は変更しない |
| 中継: 最初K手のbtimeをcap | 本当の時計はそのまま。短い時計で探索の時間管理を使い, Kを過ぎたら貯めた実時計をそのまま渡す。cap30/K15を主候補にする |
| 中継: 最初K手だけ `btime 0 binc 0 byoyomi Y` | 初期の思考量を強く固定できる。ただし局面への適応を失う。本番のd2=1120ならY=3000でも使えるのは概ね1880msで, 実時間3秒にはならない |

中継の本体は `clock_proxy.py` と `clock_rule.py`。ハーネスも**同じrewrite関数**を呼ぶ。入力のpositionから手番とSFEN手数を読み, `自分の手番=(p-5)//2+1` が1〜Kの先手のgoだけ変える。positionの再送・ponder外れによる呼び直しで回数がずれない。通常のgoとgo ponderの両方に適用し, bare ponderhitはそのまま送る。hitでの再initは期待しない。Stochastic_Ponder=falseを前提に検討する。infinite/mate/nodes/depth/movetime/rtimeは介入対象外。

**420秒を引くのはbridge側で一度だけ**。中継は引かず, 既に180秒になったbtimeのminを取る。600秒を受け取ったままcapだけでしのぐ設計にはしない。K後に600秒へ戻って時間切れになるからである。玉の屈伸4手は手数へ残す。

本番時間で中継を置くときの起動形 (まだ大会設定へは導入していない):

```bash
source scripts/env.sh
W="$DATA_ROOT/work/task180_clock"
python3 -B "$W/clock_proxy.py" --rule cap --cap-ms 30000 --k 15 \
  --cwd "$YANE_DIR/source" -- "$YANE"
```

bridgeがこのプロセスに通常どおりsetoption/isready/position/goを送る。proxyのstdoutは子のUSI出力だけで毎回flush, 診断はstderrへ送る。

## 時計付き対局の仕様と検証

`clock_match.py`: A/BはJSONのargv/cwd/optionsで別のUSIエンジンにもできる。`make_config.py` が採用eval同士・定跡なしの設定を作る。Aの先後は指定可能で, 実験ではA=先手固定。Aだけ追加optionまたはrewriteを使える。CPU制限を `parallel*Threads <= cpu-budget` で確認する (既定4)。資源用optionsの上書きはCLIのthreads/hash引数へまとめた。**ponderはoff固定**。任意のponder実装は今回入れていない。

- 初期局面からcshogiでブイ4手を適用し, その後にseed付きの合法手4手を置く。開局用の手には時計を使わず, 実時計は指定baseから始める。手数と反復履歴には含める。介入のKもブイ後から数えるため, 4手を事前に置く既定実験では初めの自分2手は介入の手数にも含まれ, 実際に検索するのは3手目から。
- `--opening-file` は「ブイの後のUSI手列を1行1本」。空行・#行を除き, seedで行を選び, `--opening-plies` 分を使う。全局を起動前に合法性検査する。ファイルを省くとソートした合法手から一様に選ぶ。今回の事前登録では公開定跡ファイルを使わない。
- 時計は `monotonic_ns()`。position送信前から, 読み取り専用スレッドがbestmove行を受けた時刻までを測る。親の集計処理の遅れを加算しない。初期化は時計の外。各手で `残り += increment - charged`。実時計とrewrite後の時計を分離し, `go btime wtime binc winc` を出す。
- `charged = elapsed + network_margin_ms`。marginは追加猶予ではなく, 1手の往復・橋渡しに使う時間として差し引く。残り+加算を越えたら切れ負け。量子 `--charge-quantum-ms q` を指定した場合だけchargedをq単位で切り捨てる。本番のCSA消費の再現にはq=1000, 縮小実験の既定はq=0。
- ルール・合法手は**cshogi 1.0.9**を使う。bestmoveは合法手一覧のUSI文字列と照合してから適用する。無手は負け, resignは負け, winは残り時間>0かつ27点宣言が成立すれば勝ち, 不成立なら宣言側の負け。
- 同一盤面・両者の持駒・手番をSFENで数えて4回目に千日手。最後の4出現の区間で, 一方の全着手が王手ならその側の負け。cshogiの `is_draw()` は探索用で4回を数えないため直接の裁定に使わない。これは[上流の実装](https://github.com/TadaoYamaoka/cshogi/blob/master/src/position.cpp)と[API文書](https://tadaoyamaoka.github.io/cshogi/cshogi.html)を照合した。
- ブイ込み512手で引き分け。大会準備文書どおり512手目の詰みも引き分けを優先し, 手を指す前の合法な入玉宣言は認める。短いスモークのために上限を下げられる。
- 1局1TSVに先後・seed・序盤・結果・理由・手数・全手列・両者の各手実消費ms(JSON配列)・各手後の残り・最終残りを保存する。終局応答の時間も配列に含み, 応答を取れない切れ負けは期限までの待ち時間を含む。各エンジンのUSI送受信ログも残す。
- run.jsonには条件・ソース・実行ファイル/evalのhashを記録する。既存出力先への上書きは拒否。初期化失敗・プロセス異常終了はINVALIDとして強さの得点に含めず, 全体の集計を止める。通常の時間切れは負けとして含む。

実機の通常スモーク: Threads 1・Hash 256MB, 先手1.8秒/後手6秒+各50ms, MinimumThinkingTime 50, NetworkDelay 3 / NetworkDelay2 28, 丸めoff, 通信の消費2ms, 2局並行。**10/10局完了, 無手9・正しい27点宣言1, 時間切れ/非合法手/異常0**。手数中央値134, 経過84.29秒。勝敗は先手1勝9敗だが, 極端に短い時計による動作確認であり強さの資料にしない。NPS中央値1.348M [1.073M,1.477M], 1482着手の最終infoから測定。

固定手順と偽USIエンジンの8テストは全通過 (`tests_release.log`)。1500ms待つ偽エンジンに300msの時計を渡して先手切れ負け, 玉の屈伸で4回目の千日手 (ブイを含む12手), 合法な飛車の連続王手12手で王手側負け, 27点宣言の境界7局面, 不正宣言・非合法手・投了・最大手数, 中継のgo ponderとponderhit, 開局の再現性を検査した。偽エンジンの待ち時間と期限は十分離し, 空いているマシンを待つ方式は使っていない。

さらに2序盤×3条件の48手打切りを2通りの起動方法で12局実行し, option/capと対応付き集計まで通した。全て打切り引き分けなので効果の評価ではない。集計器は別の固定得点データで **対応差0.35, 序盤clusterの95%幅[0.2,0.5]** を照合し, 1行欠けた腕を拒否した (`summary_tests.log`)。本体のRust局面更新・合法手生成には触れておらず, cargo test/verifyは対象外。

再現 (出力先は毎回新しい名前にする):

```bash
source scripts/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="$DATA_ROOT/work/task180_clock"
"$VENV_PY" -B "$W/analyze_logs.py" --data-root "$DATA_ROOT" --out "$W/reanalysis"
"$VENV_PY" -B "$W/test_clock.py" --out "$W/retest"
"$VENV_PY" -B "$W/clock_match.py" --config "$W/smoke.json" --out "$W/resmoke10" \
  --games 10 --parallel 2 --cpu-budget 4 --threads 1 --hash-mb 256 \
  --black-base-ms 1800 --white-base-ms 6000 --black-inc-ms 50 --white-inc-ms 50 \
  --network-margin-ms 2 --opening-plies 4 --seed 180001 --arm smoke
```

## 縮小する時計と計算費用

主実験は**先手18秒・後手60秒・各+0.2秒, Threads 1, Hash 256MB, ponderなし**。NetworkDelay 12 / NetworkDelay2 112 / MinimumThinkingTime 200, RoundUpToFullSecond=false。他の設定は本番どおり, evalは両側task154_R_result30。実ネットワークを模擬しないので外側marginは0。時間切れを時計の不等式で裁定することと, 対局のNPSが負荷で揺れることを分ける。タイミングの誤差を避けるためにマシンが静かになるのは待たない。

**名目1/10であって本番の完全な相似ではない**。やねうら王の1000ms定数はオプションで縮められず, 丸めoffではEの1秒予約も消える。初手のoptimumはponderなしで594ms (本番ponderなし4937msの12.0%), maximum 2210ms。ponder有無・探索の深さ・最善手変更の頻度も異なる。eventの約11M npsに対し, 今回測れた1スレッド1.348M npsなら, 同じ思考秒数比1/10でノード数は**約1/82**。初手のoptimum同士なら0.594×1.348M / (6.171×11M) = **約1/85**。ponderで無料に読めるノードはこの比に入らない。XeonのNPSは測っていないため, この比をそのままXeonの実測値とはしない。

134手前後なら両者の総時計の上限は約104秒, 512手まで行っても約180秒 (初期化を除く)。通常局は残りを使い切る前に終わるので, **1コアあたり毎時35〜60局**を計画値にする。40局並行なら毎時1400〜2400局, 3万局で約12.5〜21.5時間。最大手数まで使い切る局ばかりなら毎時約800局, 約37.5時間。短時間10局では1局あたり壁時間の合計165.56/10=16.56秒で, その別時計では約217局/コア時だった。これは主実験の直接の速度測定ではない。既存の判断対局は常に動いているので, 時間やNPSを断定せず実行ログから更新する。

Hash 256MB・Threads 1でisready後のエンジンRSSは444592KiB (約434MiB)。40局は常駐80本で**約33.9GiB**に相当し, 62GBのXeonに収まる見込み。ponderなしなら検索中は1局1本で40スレッド, 56のうち16を制御・OS等へ残す。ローカルは2局並行に抑えた。別マシンでは本タスクの試験を一切実行していない。

## 事前登録と必要局数

結果を見る前の登録は `preregistration.json`。以下の2介入を既定と比較する。後手は全て既定。公開定跡なし, 同じeval, seed=2026180000から**10000序盤**, 1序盤につき3局=計30000局。開局はブイ+ランダム合法手4手。各序盤で3条件の投入順をseedで混ぜ, 単一の並行枠で実行する。既定側の1万局を両比較で共有する。

| 腕 | 縮小実験での変更 | 本番で対応する変更 |
|---|---|---|
| default | 変更なし | 変更なし |
| cap30 (主候補) | 先手の自分15手目までbtimeを3000ms以下 | 同じ手数まで30000ms以下 |
| slow70 | AのSlowMover=70だけ | 先手用のエンジンだけSlowMover=70 |

主評価は `先手勝ち=1, 分=0.4, 負け=0` の **同じ序盤の腕−既定の平均**。後手は勝ち1・分0.6・負け0なので合計は常に1。rawのW/D/Lと切れ負け率・最小残り・序盤消費も併記する。時間切れを取り除いて勝率をよく見せない。初期化や読み取り異常・欠測はINVALIDとし, 欠けたペアを無視した判定はしない。

`paired_summary.py` はseedと手列の完全一致・全局数・時計・資源・実行ファイル/eval・ハーネスのhashを照合する。同じ序盤を複数seedで引いた場合は同じclusterへ束ね, **序盤cluster単位の対応付きbootstrap 20000回, seed=180**で区間を出す。通常95%幅と, 2比較Bonferroni用の各97.5%幅を両方出す。後者の下端>0を一次の通過条件にする。途中の有意判定・良さそうなところでの延長はしない。実装が変われば登録し直す。

必要数の計画は, 勝敗が決まる場合の先手勝率pと固定引分率dを置き, `S=(1-d)*p+0.4*d`, `V=(1-d)*p+0.16*d-S²` とする。+e Eloは勝敗が決まる部分の `p'=1/(1+((1-p)/p)*10^(-e/400))`, 得点差 `δ=(1-d)*(p'-p)` へ写す。対応差の分散を相関0で `2V` と近似して, `n ≈ (z(.9875)+z(.8))²*2V/δ²` (両側family α=.05, 2比較, power80%)。

| 勝敗が決まるときのp | 引分率 | +20 Elo: 腕ごとのペア数 | +40 Elo: 腕ごとのペア数 |
|---:|---:|---:|---:|
| .5 | 0 | 5750 | 1447 |
| .5 | .1 | 6414 | 1615 |
| .2 | 0 | 8375 | 1961 |
| .2 | .1 | 9538 | 2233 |

1ペア=同じ序盤の既定1局+腕1局。2介入なら既定を共有するので総局数はペア数の3倍。これは帰無分散を使う正規近似で, 実際の共分散・引分率・時計による先手不利は未測定。対応が正に相関すれば少なくて済むが, 相関0より悪いこともある。分散が2倍なら必要数も2倍。**10000ペアは上表の範囲を目安にした計画で, +20の検出力を保証しない**。0.4引分を普通の0.5得点のEloへそのまま変換しない。

通過しても縮小・ponderなしでの証拠に限る。別seed (2026190000以後) と本番の3分/10分+2秒, Threads 28・Hash 8GB・ponderありで, 残した候補を改めて確認する必要がある。本タスクではその長い確認も実行していない。

## そのまま渡せる起動コマンド

Xeon用。スクリプト一式を同じDATA_ROOT相対位置へコピーした後に実行する。venvは使わず, 指定のpylibでcshogi 1.0.9とnumpy 1.26.4を読む。パスを含むJSONは**Xeon上で生成し直す**。以下は実行していない。長い無人実行の前には, スリープ無効を作者側で確認する。

```bash
export DATA_ROOT=<worker>/binbo-work/data
export PYTHONPATH=<worker>/binbo-work/pylib${PYTHONPATH:+:$PYTHONPATH}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="$DATA_ROOT/work/task180_clock"
python3 -B "$W/make_config.py" --data-root "$DATA_ROOT" --scale .1 --out "$W/xeon_scaled_v1.json"
python3 -B "$W/test_clock.py" --out "$W/xeon_preflight_v1"
python3 -B "$W/run_experiment.py" --config "$W/xeon_scaled_v1.json" --out "$W/xeon_prereg_v1" \
  --games 10000 --parallel 40 --cpu-budget 40 --threads 1 --hash-mb 256 \
  --black-base-ms 18000 --white-base-ms 60000 --black-inc-ms 200 --white-inc-ms 200 \
  --network-margin-ms 0 --charge-quantum-ms 0 --opening-plies 4 --max-plies 512 \
  --cap-ms 3000 --k 15 --seed 2026180000
python3 -B "$W/paired_summary.py" "$W/xeon_prereg_v1/default" \
  "$W/xeon_prereg_v1/cap30" "$W/xeon_prereg_v1/slow70" \
  --reps 20000 --seed 180 --out "$W/xeon_prereg_v1/paired.json"
```

ローカルで同じ登録の実験を走らせる形。こちらも実行していない。2局並行なので検索は2スレッド, 上限4以内。短い事前点検だけならgamesを10, seedを180101へ替え, 本登録と違う出力先を使う。

```bash
source scripts/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W="$DATA_ROOT/work/task180_clock"
"$VENV_PY" -B "$W/run_experiment.py" --config "$W/scaled.json" --out "$W/local_prereg_v1" \
  --games 10000 --parallel 2 --cpu-budget 4 --threads 1 --hash-mb 256 \
  --black-base-ms 18000 --white-base-ms 60000 --black-inc-ms 200 --white-inc-ms 200 \
  --network-margin-ms 0 --charge-quantum-ms 0 --opening-plies 4 --max-plies 512 \
  --cap-ms 3000 --k 15 --seed 2026180000
"$VENV_PY" -B "$W/paired_summary.py" "$W/local_prereg_v1/default" \
  "$W/local_prereg_v1/cap30" "$W/local_prereg_v1/slow70" \
  --out "$W/local_prereg_v1/paired.json"
```

同じseedでも時間付きの探索はマシン・負荷で手が変わるので, 両マシンの結果が一致することは要件にしない。検証するのは同じ開局, 同じ裁定, 同じ介入であること。`summary.json` は今回の解析・小試験・計画をまとめたもので, 未実行の3万局の結果ではない。

## 事前登録の改訂 (Claude, 10/24 02:35, 対局を始める前)
- 局数を 1 万序盤 × 3 条件 (3 万局) から **4,000 序盤 × 3 条件 (12,000 局)** に減らす。今夜の Xeon の空き時間に合わせるため。腕・seed・時計・指標・推定は変えない。
- 4,000 局そろってから 1 回だけ集計する (途中で見ない)。Xeon で 40 並列, Threads 1, Hash 256MB。同時に判定の対局は Xeon で指さない。
- 検出力は上の表から, 腕ごとに 4,000 ペアで +40 Elo なら十分, +20 Elo には足りない。
- 道具は `experiments/clock/` に入れた (Codex が作った版のまま)。
- **改訂 2 (10/24 03:00, 対局を始める前)**: 2 台に分ける。14700KF が seed 2026180000 から 2,000 序盤 (20 並列), Xeon が seed 2026182000 から 2,000 序盤 (40 並列)。
  局 i の序盤は seed + i なので, 合わせて登録どおりの 4,000 序盤になる。1 つの序盤の 3 条件は必ず同じマシンで指す。
  集計は両方そろってから 1 回: マシンごとの paired_summary (記述) と, 両マシンの序盤ごとの対応差をマシンで層別して序盤単位で bootstrap (20,000 回, seed 180, 97.5% 区間の下限 > 0 で合格)。

## Task 180b: 2台のプール解析を結果閲覧前に固定 (Codex, 2026-10-24)

`experiments/clock/pooled_summary.py` と合成データ専用の `test_pooled_summary.py` を追加。実験の run.json・対局TSV・途中集計は一切読まずに実装・検証した。コミットなし。解析スクリプトは永続領域の `work/task180_clock/` にもコピーした。

- 各機械の default/cap30/slow70 を既存 `paired_summary.summarize` で検証し、その出力を機械別の記述としてJSONにそのまま格納する。機械間では時計・Threads/Hash・介入引数・パス以外の設定値・ハーネスhash・記録されたエンジン/evalの役割別hashを一致させる。argvの実行ファイル/絶対パス・cwd・EvalDirは機械ごとの差を許す。投入並列数とCPU budgetは20/40の分担なので機械間一致を要求しない。matrix.jsonの実行ハーネスhashも、記録があれば照合する。実機パスへアクセスせず、持ち帰ったメタデータのhashで確認する。
- seedは機械内で宣言した連続範囲と一致、機械間で重複なし、和集合が **2026180000〜2026183999の4000個に完全一致**することを必須にする。本番CLIにこの制約や反復数を変えるオプションは付けない。
- 推定量は全seedの先手得点差 (腕−default、勝1/分0.4/負0) の平均。機械内で同一opening_idは既存解析と同じclusterに束ねる。各機械の元のcluster数だけ復元抽出し、対応差の合計/seed数で機械内のbootstrap平均を計算する。それを元の機械別seed数で固定加重してプールする (登録の2台なら各1/2)。同じ手列が複数seedにある場合もclusterを分割せず、clusterサイズの揺れで機械の寄与率を変えない。各反復で2介入には同じ抽出添字を使う。入力機械の指定順・各機械のseed昇順でclusterを作り、numpy `default_rng(180)`、100反復ずつ、計 **20000反復**で固定。
- 通常95%のpercentile区間 (.025/.975) と2比較Bonferroni用97.5%区間 (.0125/.9875) を出し、後者の下限が厳密に0より大きい腕だけpass。機械別の通過判定はしない。raw W/D/L、両側を含む時間切れ局数、先手の局ごとの最小着手応答後残りms、開局後の先手の最初15検索応答の実消費ms合計を各腕・機械別/プール別に併記する。短い局は得られた応答までを合計し、観測応答数も出す。投了・宣言・期限切れの最終応答もハーネス配列どおり含める。時間分布は平均・最小・25%点・中央値・75%点・最大を保存する。
- 合成の各機械6序盤だけで、設定/時計/hashの機械間不一致、欠測、機械内不一致、seed重複、登録集合不足の拒否、固定機械比率の層別抽出、不均等サイズの重複clusterを検証。JSON・短表出力も含め9テスト通過。数値ライブラリは1スレッドに固定し、マシンが静かになるのは待っていない。

全局がそろった後に一度だけ実行するコマンド (このタスクでは実行していない):

```bash
source scripts/env.sh
W="$DATA_ROOT/work/task180_clock"
"$VENV_PY" -B experiments/clock/pooled_summary.py \
  "$W/local_prereg_v1" "$W/xeon_prereg_v1" --out "$W/pooled_prereg_v1.json"
```
- **改訂 3 (10/24 03:15, 結果は見ていない。見たのは最初の試行の局数 67 だけ)**: 時計で指す対局は CPU に関係なく 1 局の壁時間がほぼ同じ (1 枠あたり 1 時間 約 32 局) なので, 2,000 / 2,000 だと 14700KF 側が昼までかかる。枠の数に合わせて配り直す: 14700KF は seed 2026180000 から 1,333 序盤 (24 並列, `local_prereg_v2`), Xeon は seed 2026181333 から 2,667 序盤 (48 並列, `xeon_prereg_v2`)。合わせて登録どおりの 4,000 序盤。最初の試行 (`local_prereg_v1`, 67 局) は集計せずに捨てる。集計の道具・腕・時計は変えない。
- **改訂 4 (10/24 03:23, 結果は見ていない)**: 独立の点検 (Sonnet 3 観点 + 反証役) で, 待機中のエンジンが落ちると `close()` の BrokenPipeError で実験全体が止まる不具合が見つかった (再開できない)。
  後始末の例外を飲み込み, 起動時の待ち時間切れ・OS の例外も 1 局の INVALID にした。修正前は失敗し修正後は通る試験を足した (`test_idle_engine_death_is_one_invalid_game`)。
  あわせて cap30 の K を 15 → **17** にした。開局のランダム 4 手に先手の 2 手が含まれるので, 自分の 3〜17 手目 = **実際に読む最初の 15 手**に上限がかかる (本番の中継はブイの直後から K 15 で読む 15 手)。`rewrite()` で 15 手ちょうどと確かめた。
  出力は `local_prereg_v3` / `xeon_prereg_v3` に替える。`local_prereg_v2` (140 局) は集計せずに捨てる。

## 結果 (10/24 08:56, 事前登録どおり両方がそろってから 1 回だけ集計) = 2 つとも合格

縮小した時計 (先手 18 秒・後手 60 秒, 各 +0.2 秒, Threads 1, ponder なし, 両側とも採用 eval), 4,000 序盤 × 3 条件 = 12,000 局, INVALID 0。
14700KF 1,333 序盤 (24 並列) + Xeon 2,667 序盤 (48 並列), マシンで層別した序盤単位の bootstrap (`experiments/clock/pooled_summary.py`, 数値は `reports/task180_pooled_prereg_v3.json`)。

| 腕 | 先手の得点の差 (腕 − 既定) | 95% 区間 | 97.5% 区間 (2 比較の Bonferroni) | 合格 |
|---|---|---|---|---|
| cap30 (読む最初の 15 手は残りを 3 秒に見せる) | **+0.019** | [+0.004, +0.034] | [+0.002, +0.036] | 合格 |
| slow70 (先手だけ SlowMover 70) | **+0.036** | [+0.021, +0.052] | [+0.018, +0.054] | 合格 |

| 腕 | 先手の勝ち / 引き分け / 負け | 先手の得点率 | 時間切れ (両側) | 序盤 15 手の先手の消費 (平均) |
|---|---|---|---|---|
| 既定 | 567 / 73 / 3,360 | 0.149 | 39 | 10.3 秒 |
| cap30 | 640 / 82 / 3,278 | 0.168 | 31 | 8.0 秒 |
| slow70 | 698 / 107 / 3,195 | 0.185 | 34 | 8.2 秒 |

マシンごと (記述): 14700KF cap30 +0.005 [−0.022, +0.031] / slow70 +0.032 [+0.003, +0.060], Xeon cap30 +0.026 [+0.009, +0.045] / slow70 +0.038 [+0.020, +0.057]。
14700KF の区間には 04:02〜04:05 ごろのスワップ (Task 181 の確認の対局が重なった) が入っている。条件は序盤ごとに混ぜてあるので偏りではなく雑音として扱う。

### 読み
- 縮小した時計では, 先手の時間を序盤で使いすぎないほうが良い。序盤だけ絞る cap30 より, 1 局を通して控えめに使う slow70 のほうが大きい。どちらも序盤 15 手で約 2 秒 (18 秒の 1 割強) を残し, 時間切れは増えない。
- 得点率 0.149 の付近で Elo に直すと cap30 約 +25, slow70 約 +45 (先手の対局だけ, 引き分けを 0.4 で数えた得点からの粗い換算)。**本番の大きさではない**: ノード数は本番の約 1/80, ponder なし, 相手は同じ binbo (時計の差がそのまま効きやすい)。
- 本番で使うなら, slow70 は「先手のときだけ SlowMover 70」が要る。bridge は対局ごとに先後で設定を変えられないので, 中継 (`clock_proxy.py`) が先手と分かった最初の `go` の前に `setoption name SlowMover value 70` を送る形が考えられる (未実装・未検証)。cap30 は中継の今の規則のまま使える (本番は K 15 = 読む最初の 15 手)。
- 次に確かめること (作者の判断): ponder ありでの縮小実験 (中継を通す), SlowMover の値の掃引 (50 / 70 / 85), 後手 (10 分) に効くか, 本番の時計での少数の通し対局で時間切れと異常がないこと。
