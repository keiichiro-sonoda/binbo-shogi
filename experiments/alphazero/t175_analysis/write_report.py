#!/usr/bin/env python3
"""Write final Japanese report and machine-readable decisions (no source mutation)."""
import json,math,hashlib,os
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text())
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    a=read('draw_value_refined.json');b=read('duplicates.json');bf=read('duplicates_followup.json');plan=read('fixed_position_test_plan.json')
    experiment=dict(task=175,executed=False,question='Does equal weighting of distinct QS leaves improve validation after cheap warm-start adaptation?',
       pool=dict(rows=1280000,unique=984354,source='same 128 uniformly sampled shards listed in duplicates.json; preserve global row order',
                 exclusions='Before grouping, remove every Task172 A_mask=false position; keep shared validation and its mirrors entirely out. Recompute group count after exclusions.'),
       arms=dict(C='sample a leaf group proportional to its multiplicity m; one shared target per group',
                 D='sample the same leaf groups uniformly; exactly the same shared target per group'),
       targets='pure WS-T labels: mean(sigmoid(clip(raw,-2000,2000)/600)) per group. Keep mean probabilities directly if possible; if using int16 labels, apply identical logit/rounding to both arms. WDL mixing disabled to isolate position weighting.',
       mechanism='For winrate MSE, grouping targets by their mean changes the original expected loss only by within-group target variance, a constant with respect to prediction; multiplicity-weighted C preserves the expected gradient.',
       warm_start='same Task129 checkpoint (already compatible with this exact cache) in every arm; reset optimizer identically. This tests adaptation, not scratch-training superiority.',
       training=dict(batch=1024,steps_per_arm=4096,seeds=[20261750,20261751,20261752],arms=2,total_updates=24576,
          checkpoints=[0,1024,2048,4096],learning_rate=1e-5,recipe='existing fast QAT/factorizer/mirror recipe; identical LR schedule, optimizer initialization, augmentation and RNG seeds per pair; no early stopping',
          primary='fixed 4096 updates; keep data group set, init and labels fixed so only sampling weights differ',
          reason='2% pool already exposes 295,646 redundant rows and the same phase gradient. About 4.2M row presentations per arm is a small warm-start weighting pilot; no guaranteed minimum can be inferred from a data audit alone.'),
       evaluation=dict(primary='shared held-out per-position winrate MSE at predeclared final checkpoint, with game/shard-cluster paired bootstrap and 3 training seeds reported separately',
          secondary=['unweighted unique validation and original validation weighting, both reported','per-ply/activity strata and pure-label target variance','held-out selfplay score calibration only as secondary: different label source'],
          validation_caveat='Do not pick an epoch or LR on this validation; if continuing after the pilot, reserve a fresh position/game holdout.'),
       progression='If validation/phase effects are consistent across seeds, expand once to ~8M rows and 16,384 fixed updates per arm; only then consider match testing. A negative tiny warm-start pilot is not proof that scratch de-duplication cannot help.',
       elo='No Elo claim from this pilot. For a true +18.8 Elo effect at p~0.5, an idealized independent-game calculation needs about 2,700 decisive-equivalent games for 80% power at two-sided 5%; paired openings/draws/yardsticks require the repository evaluation protocol and actual variance.',
       cheaper_label_policy='First-occurrence retention is operationally simplest and nearly identical on pure WS-T labels; grouped mean is used here only to isolate frequency weighting. Outcome-mixed data need a separate trial.'
    )
    dump('dedup_experiment_plan.json',experiment)
    summary=dict(task=175,complete=True,engine_runs=0,training_runs=0,commits=0,max_concurrent_analysis_threads=2,
       report='reports/task175_analysis.md',authoritative_A='draw_value_refined.json',authoritative_B=['duplicates.json','duplicates_followup.json'],
       draw_proposal=dict(tournament_initial=dict(DrawValueBlack=-60,DrawValueWhite=60,
          black_cp_point=-62.54899590438166,black_sampling_ci95=[-213.5718865999841,88.4738947912208],
          white_cp_point=54.640602578504044,white_sampling_ci95=[-109.07085365131344,218.35205880832154],
          status='behavior-test candidate; 22 Floodgate games do not establish tournament optimality'),
          depth9_selfplay=dict(DrawValueBlack=-260,DrawValueWhite=290,ply=[40,80],
             black_ci95=[-294.1049547814432,-232.87148228304488],white_ci95=[257.55105783236394,320.7632653767998]),
          model_source_uncertainty='Not covered by sampling CIs; observed source/ply spread is roughly -450..-50 and +50..+550 before late-ply extrapolation.',
          max_moves='future behavior check must include MaxMovesToDraw=512 and =0; data cap was 320'),
       dedup=dict(raw_rows=63950000,duplicate_excess=15095305,unique=48854695,removed_fraction=15095305/63950000,
          sample_rows=1280000,qs_collapse_share_of_sample_duplicates=248780/295646,same_game_share=295365/295646,
          pure_ws_t_mean_abs_label_difference=b['labels']['absolute_difference']['mean'],
          pure_ws_t_identical_pair_fraction=286295/295646,raw_label_mse_floor=b['labels']['irreducible_label_mse_all_rows'],
          result30_label_mse_floor=bf['result30_labels']['irreducible_raw_label_mse_all_rows']),
       constraints=dict(writes='new artifacts under task175_analysis only; repository and existing source data unchanged',
          denryusen_reads='only denryusen/floodgate/*/me/records/*.csa (22 files)',
          notes='no repository STATUS update: explicit user write restriction overrides repository checkpoint rule'),
       verification=dict(wdl_mle_score_norm=a['models']['pooled']['draw']['score_norm'],
          duplicate_origin_partition=295646,validated_repetition_fixtures=len(plan['fixtures']),
          external_sources=[
            'https://github.com/yaneurao/YaneuraOu/wiki/思考エンジンオプション',
            'https://www.apply.computer-shogi.org/wcsc36/appeal/sojo/sojo_WCSC36_appeal.pdf']))
    dump('summary.json',summary)
    report='''# Task 175: 引き分け価値と技巧由来データの重複（2026-10-23）

結論: **大会向け DrawValue の初期試験候補は Black=-60 / White=+60。ただし最適値は未確定。重複削減は約24%の行を減らせるが、現在のラベルの雑音除去より、静止探索の葉の出現頻度を均す施策になる。** エンジン・学習・コミットなし。リポジトリと既存データは変更せず、この作業ディレクトリ内に新規保存した。

**A. 0.4 / 0.6 の引き分け価値**

Task154 / 159 の各15万局から各8,192局をseed固定で無作為抽出。計16,384局、引き分け289局（千日手142、320手上限147）。非詰みの `raw` を **USI cp・手番視点**で使用し、内部 `score` との換算を1,804,291行で照合した。±600cp内の866,153行で、先後・評価値・手数と交互作用を含む二段階ロジットを当てた。係数・共分散・確率表は `draw_value_refined.json`。誤差は**局単位のcluster sandwich 95% CI**で、同じ局の各手を独立標本には数えない。

`q=P(win | 決着), r=P(draw)` とすると、`(Pwin,Pdraw,Ploss)=((1-r)q,r,(1-r)(1-q))`。引き分けの得点をdとすると、継続の期待得点 `Pwin+d Pdraw=d` は **q=d** と同値。自然な引き分け確率を無視したのではなく、両辺から消える。勝敗別の局を丸ごと分けたholdout 3,184局ではWDL logloss=0.763、Brier=0.504。例:80手・0cp・先手で勝47.7%／分3.93%／負48.4%。holdoutの引き分け予測は評価値帯により実測より約0.2〜0.9pt高く、厳密な確率とは扱わない。

手数帯ごとの独立な先後別ロジットで境界を確認した（0始まりのply、単位cp）。

| ply | 先手: d=0.4 の境界 [95% CI] | 後手: d=0.6 の境界 [95% CI] |
|---|---:|---:|
| 16–39 | −225 [−251,−199] | +234 [+208,+260] |
| 40–79 | −263 [−294,−233] | +289 [+258,+321] |
| 80–119 | −341 [−401,−280] | +386 [+320,+452] |
| 120–199 | −367 [−496,−238] | +418 [+283,+554] |

世代を分けた連続モデルの60手ではTask154が−247/+272、Task159が−280/+303。遅い手数ほど不確かで、連続モデルは200手以降で±600cpの支持範囲を外れたり傾きが逆転するため外挿しない。320手上限の局と連続王手を除く感度分析でも、決着条件のqと境界は変わらない（該当局には決着行がない）。Pdraw自体は変わる。標本中の千日手はcshogiで再生し、連続王手による勝敗は0局だった。

指定のFloodgate CSA **22局だけ**も全件読み、先手視点のCSAコメントを手番視点へ換算した。全て投了、binboの評価値1,092手（±600内631手）。局数が少なく、先後・手数別の大会推定には足りない。先後共通の補助ロジットは **−63 [−214,+88] / +55 [−109,+218]cp**。ローカルTask158の400局・Task165の96局は勝敗集計のみで各手の評価値がなく、温度推定には使えない。

したがって提案は **大会の固定局面試験を−60/+60から開始、depth9自己対局の較正確認には−260/+290**。前者のCIは0を跨ぐ。自己対局との数百cpの差は探索条件・相手・局面分布の差を含み、局数を増やした標本CIでは消せない。観測された主な範囲はおよそ先手−450〜−50／後手+50〜+550cpで、これは大会用CIではない。既定−2で選択された棋譜からの観測推定であり、「継続」の反実仮想の勝率や設定変更後の最適性は識別できない。

[やねうら王の公式説明](https://github.com/yaneurao/YaneuraOu/wiki/思考エンジンオプション)とローカル `yaneuraou-search.cpp:1003` / `usi.cpp:1106` を確認: 指定単位はcp、内部は `int(option*90/100)`、rootの色の設定だけを選び、相手番の葉にはその負号を使う。確率0.4/0.6をそのまま指定する設定ではない。

**固定局面試験案（未実行）**: `fixed_position_test_plan.json` に合法な全履歴つき24局面（先後各12）、4回目になる手と他の合法手、USI入力を保存。cshogiで全履歴・同一盤面/持駒/手番4回・連続王手でないことを検証済み。まず各1万ノード・1スレッドで千日手の手を `searchmoves` から除き、非反復PVの継続値cを測る。**棋譜にある−2近辺の値は選択済み千日手の値であり、cの代用にしない。** 先後各4局面を選び、既定と候補をfresh processで比較・再現: 先手は `−60<c<−2` だけで「分→継続」、後手は `−2<c<+60` だけで「継続→分」を期待する。±10cpの余裕を置く内外の対照、強制千日手、履歴を消した同盤面、非root側オプション変更も比較。帯内外の局面が揃わなければ判定保留。最大約96探索・96万ノード、曖昧な局面だけ倍増する。

**512手例外**: 同じdrawValueを手数上限にも使う（通常探索2461行・qsearch4611行）。`MaxMovesToDraw=0` は上限無効相当、試験では512を明示する。履歴なしの非王手局面をSFEN手数511/512/513にした対照で、512設定時の上限分岐の値も変わることを確認する。実戦データは320手止めで512付近を較正できない。選択的探索・内部探索手数上限にも波及し得るため、全局面で「千日手の帯以外は絶対不変」とは保証せず、上記の対照付き挙動を合否とする。

**B. 重複の正体とラベル**

Task172全件監査の15,095,305はペア数でなく**最初の1行を除いた余剰行数**。63,950,000行→**48,854,695固有局面（23.605%減）**、val除外前の値。同じ更新数なら固有局面当たり平均提示回数は1.309倍、1周は62,452→47,710更新（batch1024）。独立な実効標本数そのものが4,885万になるわけではない。

原因は全6,395 shardから一様に128個、**128万行=2.002%**を抽出し、77列の完全bytes一致で調べた（ハッシュ近似なし）。元HCP・元ply・局番号と対応づけ、静止探索後の局面も往復変換で照合。前半71万／後半57万行を含む。局所重複率23.084% [shard bootstrap 22.726,23.440]、標本全体では295,646余剰行・984,354固有局面。

| 標本の重複余剰行の原因 | 行 | 割合 |
|---|---:|---:|
| 異なる元局面→同じqsearch葉 | 248,780 | 84.15% |
| 同じ元局面の再出現 | 46,866 | 15.85% |

295,365行（99.905%）が同じ対局内。qsearch合流の99.51%は元ply差4以内（中央値1）。元局面の反復は同局内46,586行、最初からのply差中央値92・最小4: **長い循環を生成器が256手まで保存する影響**と整合する（`src/teacher/gen.rs` は千日手終局判定を持たない）。別対局由来281行、異なるshard281行、32M境界を跨ぐもの67行。9,186局の先頭16保存局面に完全一致の対局は0、各runのseed領域も別なので、同一対局や32M半分の丸ごと再コピーを主因とする証拠はない。

サンプル内の別shard追加重複は175行だが、全体の別shard重複率の推定値にはできない。局内を保つサンプルの約23.1%だけで全件23.6%の大部分を説明できる一方、差約0.5ptには未捕捉の別shard一致と標本誤差がある。**全体の84.15%が合流、と全件確定したわけではない。**

| 元ply | 行数 | 重複余剰率 |
|---|---:|---:|
| 6–29 | 220,529 | 7.37% |
| 30–59 | 274,955 | 16.92% |
| 60–99 | 355,345 | 26.03% |
| 100–149 | 301,484 | 29.55% |
| 150–199 | 93,985 | 35.37% |
| 200–255 | 33,702 | 53.52% |

phaseの代理「成駒数+持駒枚数」が0なら重複3.37%、10以上なら30.00%。**序盤重複が主因ではない。** 標本の最初の行を残す除重複では6–29手の比率が17.23→20.75%、200手以降が2.63→1.59%になる。終盤を相対的に薄くする変更でもある。

現在使う **Task129 WS-Tラベル**は、重複とその最初の行の差が96.84%で完全0、絶対差の中央値0／平均0.0645／最大253（ラベル単位、USI cpとは別）。差100超は14/295,646行。重複内の不可避MSEは全行平均でラベル空間0.306、実際の `sigmoid(clip(label,±2000)/600)` では **1.85e−8**にすぎない。参考に同梱の旧技巧ラベルでは中央値66／平均174と大きいが、現在のWS-Tラベルの雑音と混同しない。

WDLが食い違う重複群は85/178,744群、うち既知の勝ち/負けの衝突41群（0は引き分け/不明を区別できない）。Task172の実物 `A_result30.npy` でも97.85%の重複ペアは同値だが、絶対差平均0.210／最大764、100超94行となる。不可避MSEはラベル空間12.215、sigmoid空間 **1.95e−6**。除重複だけでラベルの正しさが増すとはいえず、平均targetなら群内分散を除ける一方、単純に先頭を残すだけでは勝敗衝突を解消しない。

**最安の対照実験案（未実行）**: 上記128万行（約98万固有）を使い、valとその鏡像を先に全排除。同一Task129 checkpointから2腕×3 seed、各**4,096更新、batch1024、lr1e−5固定**で軽い継続学習を行う。各局面の共通ラベルを「clip後sigmoid targetの群平均」に統一し、Cは群の重複回数mに比例して抽選、Dは固有群を一様抽選する。winrate MSEではCの期待勾配は元の重複行学習と同じなので、ラベル変更と頻度変更を分離できる。更新数・初期値・optimizer・mirror等を揃え、最終checkpointの独立val損失とphase別変化を比較する。これは23%の頻度差を安く見るpilotで、**4,096歩で棋力差を検出できる保証はない**。

同じ方向が3 seedで出たら8M行・16,384更新/腕へ一度だけ拡大してから既存の対局評価手順へ進む。小pilotの陰性はゼロからの除重複学習を否定しない。+18.8 Eloを80% powerで検出する単純な独立局近似でも約2,700決着局相当が必要で、少数対局による判定はしない。手順・更新数・保留条件は `dedup_experiment_plan.json`。

[奏乗WCSC36アピール文書 p.11](https://www.apply.computer-shogi.org/wcsc36/appeal/sojo/sojo_WCSC36_appeal.pdf)の24B→14.5B、1660勝120分1484敗・R18.8を一次資料で確認した。別の教師・規模・重複率の結果であり、このデータへの+18.8の予測には使わない。

正本の数値: `summary.json`、`draw_value_refined.json`、`duplicates.json`、`duplicates_followup.json`。再現と検証は `README.md`。全件特徴20GBの再走査を避け、B本体11.6秒+ラベル追跡7.7秒、A標本処理6.4秒+最終再推定2.7秒（入出力cache・準備・失敗試行を除く計測値）。各解析は1スレッド、最大同時2、ピークRSS約1.7GiB。初期draw-logitの未収束を勾配チェックで検出して修正し、最終fitは平均予測=実測、勾配norm<1e−7を確認した。旧 `draw_value.json` は途中結果で採用しない。
'''
    (OUT/'reports').mkdir(exist_ok=True)
    with (OUT/'reports/task175_analysis.md').open('x') as f:f.write(report)
    readme='''Task175成果物（すべてこのディレクトリで新規作成）

最終報告: reports/task175_analysis.md
正本: summary.json / draw_value_refined.json / duplicates.json / duplicates_followup.json
未実行の次段階: fixed_position_test_plan.json / dedup_experiment_plan.json
再利用できる入力抽出: draw_sample.npy（int32、列定義はdraw_value_refined.json）, sample_game_ids.json

再現はリポジトリで source scripts/env.sh し、export TASK175_REPO="$PWD"、export PYTHONDONTWRITEBYTECODE=1 と各BLAS/OMP thread変数=1。すべてのスクリプトはデータをread-onlyで開き、出力をexclusive-createする。再実行時は既存ファイルを上書きしない。新しい許可済み作業ディレクトリへ必要なスクリプトをコピーして実行する。

B: analyze_duplicates.py -> duplicates.json。duplicate_followup.py -> duplicates_followup.json（同じ128 shard）。
A: analyze_draw_v2.py は初期抽出の draw_sample.npy / sample_game_ids.json が既存なら内容一致を確認し、統計とfixtureを作る。新規抽出には analyze_draw.py を用い、fixtures内の move_from_usi は Board のメソッドを使う必要がある。抽出後に refine_draw_v2.py で final WDL fit を作る。最終モデルはline searchとscore-gradientで検証済み。
make_test_plan.py はcshogiだけで24履歴を検証する。USIプロセスは起動しない。
write_report.py は最終数値から報告と試験案を書く。

途中試行も上書き・削除せず残した。analyze_draw.py はfixtureのAPI指定誤りで中断、v2は全JSON保存後のstdout ndarray表示でエラー。draw_value.jsonのdraw fitは未収束であり破棄扱い。refine_draw.pyは構文エラー、refine_draw_v2.pyが成功版。これらのエラーと補正はログに保存し、正本は上記のみ。入力データと開始時に存在したファイルは変更していない。

検証: validation.json（最終生成）、source_evidence.json（ソースstampと外部一次情報）。エンジン実行/学習/コミットは0。denryusen内は指定CSAのglobと22ファイルだけを読んだ。
'''
    with (OUT/'README.md').open('x') as f:f.write(readme)

if __name__=='__main__':main()
