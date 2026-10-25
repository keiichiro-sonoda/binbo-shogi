# Task 067 — replay 比率の探索(自己強化ループで前進する世代を作る)

**ゴール**: Task 066 の 1-cycle リハーサルで gen1 が一様に弱化した(gen-vs-gen −228 elo、技巧2 83.3%→60.0%)。
診断 = warm-start を champion から始めたのに replay が 32M champion キャッシュの **~0.3%(新規:replay=1:1, 局面数)** しか混ざらず、
fine-tune が champion 最適点から **引き剝がされた**疑い。**一変数 = replay 比率**を振り、「champion から引き剝がされず、かつ新規自己対局の信号で
前進する」世代が作れるかを確かめる。前進世代が1つでも出れば自己強化が原理的に機能する証明=本走の出発点。

## 方法(replay 比率以外は 066 と完全同一)
- **新規自己対局 = 066 のものを bit-完全再利用**: `selfplay_label.py --reuse-new caches/gen1_mix`(先頭 100,012 行 = champion@YaneuraOu@depth12 自己対局, cp勝率+WDL)。
  `--replay-n N` で **32M(技巧2 depth8)から N 行をサンプル連結**、val は shared_32m の pinned 50k をコピー。**検証: `--replay-n 100012 --seed 20260629` で出力 train+val が gen1_mix と完全一致**(新規再利用が忠実)。
- **warm-start 学習(`--init-weights` champion `nnue_best.pt`)**: objective=wdl, **λ=0.5**, tied QAT(fd256/h32), lr 1e-5, 3 epochs, batch 1024, augment-mirror。
  → **全ハイパラが 066/gen1 と一致、差分は `n_train`(replay 数)のみ**(`model_config`/`history` で確認)。export(nnue_final.pt)→ writer → **0cp ゲート 32/32 GREEN**(全 gen)。
- **判定**: 主 = gen-vs-gen SPRT(`yane-vs-yane`, 探索固定・eval のみ・両側 0cp, movetime 0.1s, 200 局, seed 20260701)。
  外部アンカー = `yane-vs-gikou`(gikou@6 フル強度, 60 局, seed 20260702 = 066 paired)。champion の技巧2 83.3% は 066 値を再利用。
- ハード i7-14700KF + RTX 4060 Ti, YaneuraOu 9.41git 64AVX2。champion `task064_32m_lr1e-5`(sha `3c25685e…`)。

## 結果 — replay 比率は後退の主因だった(単調回復)
| 世代 | replay-n(32M比 / 新規:replay) | 技巧2 gikou@6(外部) | gen-vs-gen vs champion(内部=採用判定) | gen sha256 |
|---|---|---|---|---|
| gen1(066) | 100k(0.3% / 1:1) | 60.0% (+70) [47.6,72.4] | **21.2% (−228) accept_h0 = 棄却(崩壊)** | `3d65a1d0…` |
| gen2 | 500k(1.6% / 1:5) | 56.7% (+47) [44.1,69.2] | (≈gen1=明確に弱・SPRT 省略) | `8ff7a9f4…` |
| gen3 | 2M(6.3% / 1:20) | 70.0% (+147) [58.4,81.6] | 42.5% (−52) [35.7,49.3] | `66a19f70…` |
| gen4 | 5M(15.6% / 1:50) | **83.3% (+280) [73.9,92.8]** | **46.0% (−28) [39.1,52.9] = 統計的パリティ** | `1f619938…` |
| champion(参照) | — | 83.3% (+280) | — | `3c25685e…` |

- 記録(W-L-D): gen4 vs gikou = **50-10-0**(= champion の 066 記録 50-10-0 と**完全一致**, paired seed)。gen4 vs champion = 91-107-2 / 200。gen3 vs champion = 84-114-2 / 200。draw 1.0%。探索は拮抗(gen4 vs champion とも avg_depth ~15.3, ~122k nodes, ~94ms=差は eval のみ)。
- **後退は replay 不足が主因と確定**: replay を 0.3%→15.6% に上げると **−228 elo の崩壊が −28 elo(champion と統計的パリティ)に回復**。技巧2でも 60.0%→83.3% と champion に到達。066 の謎は説明され、replay 比率で治癒可能。
- **内部(gen-vs-gen)と外部(技巧2)が終始一致**(replay↑で両方が単調改善: gen1 崩壊 → gen3 小敗 → gen4 パリティ)。**乖離なし=自家中毒は全 gen で観測されず**(066 の所見と整合)。

## 結果 — ただし champion を超える世代は出ていない
- **gen4 はパリティ止まり**(46.0%, score CI が 50% を含む, elo CI [−76.8,+20.0] が 0 を含む)。SPRT は 200 局で未決(LLR −0.550=continue)だが点推定は champion 微下=**昇格なし**(champion 不動, reversibility 維持)。
- これ以上対局を増やしても「明確な勝ち」は出ない見込み(CI が既にパリティを示す)ため経済的に打ち切り。**昇格は SPRT 勝利時のみ**=後退しない。
- 含意: **replay でない別のボトルネック**が前進を阻む。replay を増やすほど champion 分布への錨が強まり後退は消えるが、同時に **新規信号(100k)が希釈**され、champion を**上回る**には新規信号の量/質が足りない。

## val の傍証(採用基準ではないが診断に有用)
| 世代 | best_val_loss(champion baseline=0.02087) | val_corr(ep1→3) |
|---|---|---|
| gen1(066) | 0.02423(**baseline 未達**) | 0.847→0.841 |
| gen2 | 0.02042 | 0.856→0.851 |
| gen3 | 0.01785 | 0.856→0.850 |
| gen4 | 0.01697(**baseline 大幅下回る**) | 0.856→0.845 |

→ replay↑で val_loss は単調低下(技巧2-val への当てはめ改善=replay は技巧2ラベルそのもの)。だが **val 改善は棋力の前進を意味しない**(gen4 は val で champion を大きく上回るのに gen-vs-gen はパリティ)=**val が採用基準にならない**ことを再確認(技巧2-val は cp/技巧2 寄りを不当に勝たせる)。

## 結論と次の一手
- **replay 比率 = 066 後退の主因(確定)**。**~15.6%(新規:replay≈1:50)で champion とパリティに回復**。replay の効きカーブを取得(0.3%崩壊→1.6%→6.3%→15.6%パリティ)。
- **前進世代は未達**(decision rule: 「champion を超える世代が出れば本走 / 出なければ次の一変数へ」→ **後者**)。replay はパリティの天井で頭打ち=次は **新規信号側の一変数**へ切り分ける。
- **次の一変数候補(本走の第一手, 個別タスク)**: ① **新規自己対局の量↑**(066/067 の 100k は機構検証規模。新規:replay を保ったまま新規を数百k–1M に)。② **ラベル深さ d12→d14**(065: cp 飽和せず=信号の質ダイヤル)。③ λ(WDL は自己対局質依存=現状ほぼコイントス 0.47/0.06/0.47)。④ 将来 replay の自己ラベル上書き(技巧2脱却。質が技巧2 を超えてから)。
- **champion 据え置き**(`task064_32m_lr1e-5`, sha `3c25685e…`)。gen2-4 eval/cache は永続保存(reversibility・将来の足場)。

## 再現
champion sha `3c25685e…` / depth12 / 新規再利用元 `caches/gen1_mix`(n_new=100012) / replay-n {500k, 2M, 5M}(seed 20260629) /
λ0.5・lr1e-5・3ep・fd256/h32(tied QAT)/ gen sha256 上表 / movetime 0.1s / gen-vs-gen SPRT(elo0=0, elo1=10, seed 20260701)/
gikou@6 seed 20260702 / i7-14700KF + RTX 4060 Ti。実装 = `selfplay_label.py --reuse-new`(新規・純 numpy)、他は 066 パイプライン流用。

## 規律
一変数(replay 比率のみ。λ/深さ/cp表現/ラベル出所/057-059 bridge/Rust 探索は不触)。GPL 遵守(技巧2/tanuki=合法アンカー、dlshogi 封印)。
長時間=`scripts/detached.sh`(marker)+`env PYTHONPATH=<repo>`+`nice`。`cargo test --release` 24 green(Rust 無改変)。commit 可・**push は人間承認後**。
