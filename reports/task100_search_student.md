# Task 100 — 探索裏付き value ラベルのミニ学生試験(1M): SEARCH_LABEL_TEACHES — 探索裏付きラベルの学生が生ラベルの学生に +196.7 Elo(1M。事前登録どおり 2M/32M 確認の資格のみ)

**一変数 = 教師ラベル族**(WS-T の生 value vs WS-T + Gumbel MCTS S=32 の root value)。他は全て同一: 同じ 1,000,000 局面(run2m_c cache の先頭 100 シャード、cache 順)+ 同じ val 40k、同じ NNUE レシピ(098 `NNUE_ARGS`: tied QAT 256×2-32-32、lr 1e-5、12 epoch、batch 1024、scale 600)、同じ seed、同じ判定器(yane-vs-yane 100ms/Threads 1、paired bootstrap seed 20269800)。事前登録済み(実行前・ローカル)。実走 9/11 19:06 起動、21:29 完了(`rc=0`、2.37h)。

## なぜこの試験か(099 申し送り (a))
099 は「ループ利得は探索を通しても value の数字に翻訳されない」(d32≈0)を示したが、gold(champion NNUE@d16 = 技巧2 蒸留物)では「探索が値を gikou 多様体から引き離す方向(corr_gikou 0.958→0.952、|Δ|vs gikou 202→243)が学生にとって良いか悪いか」は原理的に測れない(錨バイアス)。gold を介さない唯一の計測器は 084 方式=ラベルで NNUE 学生を学習させて対局させること。d32≈0 ゆえラベラーは ③(ループ網)でなく ②(WS-T+探索)で足りる。∴ これは**「AZ 探索ラベル一般」の問い**であり、ループ路線の問いではない。

## ブリーフからの変更(事前登録に明記、いずれも結果非依存)
1. **2M 全量でなく先頭 1,000,000 局面**(`--limit-train`)。ラベラーは単一プロセス実測 ~187 局面/s(scratch 探針、shard 0)。GPU 上の 2 プロセス並列は合計 168/s で単一より遅く並列化は無効。本日の残り時間(19:00 起動→24:00)に学習・書出・800 局・報告を収めるための縮小。085 の 2M→32M 反転は 1M ではより強く効くので、**正の結果は 2M/32M 確認の資格を得るのみ**。
2. **生ラベル学生を 098 の `task098_student_wst`(1.96M 学習)から流用せず、同じ 1M で再学習**(流用するとラベル族とデータ量が交絡)。生ラベル npy は 098 の `work/task098/rescore_wst` をバイト単位で流用(value ヘッド不変)。
3. 800 局を **4 並列 worker × 200 局**(seed 20270911-14)で実行(098 は 1 本逐次)。各 worker は完全な colour-swap 系列でペア構造は無傷、両腕は各 worker 内で色を交互に持つ。時計規則: 22:00 JST 以降の対局開始なら 150 局/worker。

## hard gate(Elo を見る前に全て通過)
- 符号オラクル S=32: black/white mate-in-1 = +0.908 / +0.908、mirror_gap 0.000(099 と同値)。
- 整合 assert: 全 104 シャード(train 100 + val 4)で gikou eval == cache eval(不一致 0)。合法手なし局面 0。
- 生ヘッドの 098 再現: train exact 88.55%・最大 3cp、val 88.57%・最大 3cp(配線ゲート 50%/10cp を通過。差は探索グラフ batch 128 と az_rescore batch 4096 の TF32/XLA 丸め→int16 切捨て)。
- 探索ラベルの移動: median|cp_search − cp_raw| = 188cp(train)/193cp(val)≥ 100cp。
- 0cp ゲート: 両学生とも engine `e` == int-faithful == binbo Mode::Quant で 32/32 exact(GREEN, bit-exact)。
- 非合法手・engine_fail: 0(800 局中 798 局が詰み `no_legal_moves`、2 局が千日手 `repetition_key4`)。
- yane TSV アダプタ自己テスト(092-0 の 400 局を再導出): score 0.4863(file 0.4860)/ elo −9.56(file −9.60)。

## ★主判定 = `SEARCH_LABEL_TEACHES`(終端、`FLAGS ['SEARCH_ASYMMETRY']`)
- **探索学生 vs 生学生(400 opening ペア・800 局): 604-2-194、score 0.7562、Elo +196.69 [+170.04, +224.97]、半幅 27.5**(paired bootstrap 200k、seed 20269800)。CI 下限 > 0 ⇒ 決定表 行1。手検算: 400·log10(0.7562/0.2438) = +196.7 で一致。
- worker 別(各 200 局・独立 seed): w0 0.772 / w1 0.730 / w2 0.765 / w3 0.757。先手番 0.740–0.760、後手番 0.710–0.790 = **4 worker・両色すべてで探索学生が 70% 超**。効果は単一 worker や色の産物ではない。
- 半幅 27.5 は 098 実測 23.1 より広い(score が 0.5 から離れると分散は減るが Elo 変換の傾きが増す)が、1.5×23.1 = 34.7 未満で `UNDERPOWERED_VS_PLAN` は不発。
- 平均手数 108–111 手(098 の 122.6 より短い = 決着が早い)、1 局 9.1s、4 worker 計 32–33 分。

## ラベル統計(1M train / 40k val、099 stage2 と同軸)
| 分割 | ラベル | cp std | corr(gikou) | \|Δ\| vs gikou median | \|Δ\| vs raw median | corr(search, raw) |
|---|---|---|---|---|---|---|
| train 1,000,000 | search(S=32) | 1446 | 0.952 | 243 | 188 | 0.976 |
| train 1,000,000 | raw | 1503 | 0.958 | 202 | — | — |
| val 40,000 | search(S=32) | 1448 | 0.952 | 248 | 193 | 0.976 |
| val 40,000 | raw | 1503 | 0.957 | 203 | — | — |

099 stage2(run2m_c 20万、②WS-T+探索): 1447.8 / 0.952 / 242.65 / 189.2 と一致 = 同じラベラー。gikou clip std は 1780(train)/1772(val)。mean max_visits 5.11(099: 4.9)。探索ラベルは生ラベルより gikou から遠く(202→243cp)、分散が小さい(1503→1446)。

## 学生の学習(val は各腕の自分の教師ラベル)
| 学生 | 教師ラベル | n_train | best epoch | val_mae(自教師) | val_corr(自教師) | epoch 時間 |
|---|---|---|---|---|---|---|
| raw | WS-T 生 value(098 npy) | 1,000,000 | 12/12 | 385.2 | 0.798 | 50–73s(rescore と GPU 共有) |
| search | WS-T + 探索 S=32 | 1,000,000 | 12/12 | 347.1 | 0.830 | 27–28s(単独) |

留保: 両腕とも best epoch = 12(最終)で val がまだ改善中 = 2M レシピ(12 epoch)を 1M にそのまま当てたため学習不足気味。両腕対称なので一変数は保たれるが、絶対棋力は 098 の学生(1.96M、val_corr 0.847)より低い。val 指標は各腕の**自分の教師**に対する値で、探索ラベルは分散が小さい分だけ当てやすい=強さの比較ではない。

## 探索条件の対称性
| 側 | avg depth | avg nodes/手 |
|---|---|---|
| cand = 探索学生 | 14.20 | 133,332 |
| base = 生学生 | 14.90 | 133,480 |

nodes 比 0.999 = **持ち時間の対称性は保たれている**(098: 125.8k/127.4k)。一方 depth 差 0.70 は事前登録の閾値 0.30 を超え `SEARCH_ASYMMETRY` が立った。ただし同じ nodes で depth が違うのは、**評価関数が違えば αβ の枝刈り・aspiration の挙動が変わり木の形が変わる**ため(探索学生のラベルは std が 4% 小さく gikou から遠い)で、片側だけ CPU が遅かった証拠ではない。この flag は「棋力差の一部は eval の形状が探索効率に与える影響を含む」という注記として読む。製品 NNUE はまさにこの条件で使われるので、判定器の規約(探索固定・eval のみ変数)の範囲内。

## 判定の含意
- **099 が gold で測れなかった問いに、gold を介さない計測器が答えた**: 探索が値を gikou 多様体から引き離す方向(corr_gikou 0.958→0.952、|Δ|vs gikou 202→243cp)は、**1M の NNUE 学生にとっては明確に良い**。099 の REGRESSION(gold への距離が +48cp 悪化)は gold の錨バイアスの産物だったことが裏づけられた(事前登録で懸念したとおり)。
- **効果量が大きい**(+197)。084 の「AZ 生 value 教師 vs 技巧2@d8 探索cp 教師」= +237@2M と同じ桁。だが 084/085 の教訓は「小データでは平滑化=正則化が大勝し、32M で反転(−36)」。今回のラベルは生 value より**さらに分散が小さく**(1503→1446)、学生は自分の教師に対して val_corr 0.798→0.830 と当てやすくなっている。∴ **正則化説(データが増えると消える)と情報説(探索が 1〜2 手の戦術解決を静的評価に焼き込む=技巧2@d8 に無い情報ではないが HalfKP 学生にとって使える形の情報)は本タスクでは区別できない**。区別するのは規模曲線(2M → 8M → 32M)。
- 099 の d32≈0 と合わせると: **探索そのものはラベルを良くするが、ループで policy が強くなっても探索到達値は変わらない**。つまり「探索ラベル」は AZ 探索一般の性質であり、ループ路線の運搬路が開いたわけではない(099 の結論は不変)。
- 留保: (1) 1M・12 epoch は両腕とも学習不足気味(best epoch = 最終)。(2) 探索学生の絶対棋力は未測定(champion・gikou@6 との対局なし)。(3) 学生は各 1 本(学習 seed n=1)。

## 申し送り
- **即時(同夜起動済)**: **Task 100b = 2M 確認**(事前登録済み)。Task 100 の 104 シャード分の part を再利用し残り 96 シャードを rescore、探索学生を 1,960,000 行で学習し、**098 の `task098_student_wst`(同一 cache・同一生ラベル・同一レシピ・同一 seed)と 800 局**。判定器・決定表は Task 100 と同一。log `~/binbo-shogi-data/logs/task100b_20260911-213222.log`、結果 `task100b_results.json`。
- **次段 = 規模曲線**: 32M の探索 rescore は 185 局面/s で **~48h GPU**(32M cache は `caches/shared_32m`)。085 と同じ土俵(champion 構成で学習 → champion と SPRT)に乗せる前に、**8M(~12h)** の中間点で 2M→8M の傾きを見るのが安い。反転すれば 084/085 と同じ「正則化」、残れば技巧2@d8 に無い**使える情報**の初の実証になる。
- **機構の切り分け(安価、~1h)**: 生ラベルを線形に縮めて探索ラベルと同じ std(1446)にした対照学生を 1M で学習・対局。それで +197 の大半が出るなら「eval スケール/木の形」の効果、出なければ探索の情報。
- **絶対値**: 探索学生(2M)を champion(`task064_32m_lr1e-5`)と gikou@6 に当てて立ち位置を測る(084 の t084_ii と同軸)。
- 計測系: `SEARCH_ASYMMETRY` の閾値は「同じ eval 同士」を想定した 098 の値。eval が違う対局では depth ではなく **nodes 比**を主にすべき(申し送り)。

## 成果物
- master log: `~/binbo-shogi-data/logs/task100_20260911-190642.log`(`T100_PIPELINE_DONE verdict=SEARCH_LABEL_TEACHES` / `DETACHED_DONE rc=0`)
- machine-readable: `~/binbo-shogi-data/logs/task100_results.json`
- ラベル: `~/binbo-shogi-data/work/task100/rescore_search/{train,val}_dlshogi_evals.npy`(train は先頭 1,000,000 行のみ有効、`meta.json` に明記)、生腕は `work/task098/rescore_wst`
- 学生: `models/task100_student_{raw,search}`、`eval_dirs/task100_student_{raw,search}/nn.bin`(0cp GREEN)
- 対局: `logs/task100_h2h_w{0..3}_seed2027091{1..4}_n200.tsv`
- コード(SSOT): `experiments/alphazero/t100_search_student.py`。事前登録済み。

## 規律・メタ
一変数=教師ラベル族のみ。事前登録を実行前にローカル commit し、決定表・閾値・時計規則は結果を見る前に固定。自己テストで当初の配線ゲート(exact ≥99.9%/2cp)が別グラフの TF32 丸め(exact 84.5-96.8%、最大 2-3cp)で止まったため実行前に 50%/10cp へ緩めた(配線ミスなら exact≈0%・数百 cp なので検出力は保たれる)。`cargo test --release` 24 passed、Rust・champion・過去タスクの成果物-of-record は不触。push は人間承認後。
