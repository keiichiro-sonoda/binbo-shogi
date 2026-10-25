# Task 131 — 勝率空間の power loss(2.5 乗、32M): `POWER_LOSS_NEUTRAL`、power − 対照 = −20.3 [−43.7, +3.0]

調査(`docs/research_survey_2026-10-12.md` の提案 2)に基づき、損失を cp 空間の Huber から、勝率空間の |σ(予測) − σ(ラベル/600)|^2.5(Stockfish 方式)に替えた。換算係数 600 は固定。事前登録済み、ドライバ `experiments/alphazero/t131_power_loss.py`。実行は 10/13 00:34 → 05:03(`rc=0`、失敗なし)。

## 変更
- trainer に `--wdl-power` を追加した(`--objective wdl` 用)。既定値 2.0 は従来の勝率 MSE で、既存の動作は変わらない。
- 腕は 1 本: `--objective wdl --wdl-lambda 1.0 --wdl-power 2.5`。ほかは対照(Task 128 腕 C)と同じ: 32M、除外なし、lr 6e-5、Factorizer、12 epoch。
- best epoch は 8/12(対照は 11)。0cp ゲートは GREEN。

## 物差し(Task 127、各 800 局)
| 比較 | まとめ | tanuki- | AobaNNUE |
|---|---|---|---|
| **power − 対照(判定)** | **−20.3 [−43.7, +3.0]** | −34.0 [−67.6, −0.5] | −7.1 |

CI が 0 をまたぐので **`POWER_LOSS_NEUTRAL`**。どちらの物差しでも点推定は負で、tanuki- 単独では CI の上限も負。不採用。

## 解釈
- Task 111 の参考比較(勝率空間の 2 乗が cp の Huber に −22.6)とほぼ同じ大きさで、同じ向きだった。**勝率空間に移ること自体が、当方では −20 前後の損**と読める。2.5 乗は、それを取り返さなかった。
- 当方の cp 空間の Huber(差が 600 を超えたら 1 乗)と 2000 での打ち切りが、「大差の局面に力を使い過ぎない」役目をすでに果たしているらしい。そのうえで勝率空間に移ると、互角から離れた局面(±600〜2000cp)の重みが下がり過ぎる。探索はそこでも指し手を比べるので、損になる、という見方ができる。
- Task 130(分布の較正)と合わせ、**ラベルの尺度・損失の形は、今の設定(atanh の尺度、cp Huber、FV16)が当方の探索とよく噛み合っている**。この方向は閉じる。

## 次
調査の提案 3(HalfKP_vm)へ。

## 成果物
- log: `logs/task131_20261013-003401.log`、`task131_student.log`
- 結果: `logs/task131_results.json`
- 学生: `models/task131_student_power25`、`eval_dirs/task131_student_power25`
