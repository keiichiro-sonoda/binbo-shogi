# Task 178: 大会用やねうら王を速くする (2026-10-24)

**推奨は `B2-avxvnni-pgo`。28スレッドで現行比 +8.10%、95%区間 [+6.03, +10.17]%。** 検索・評価・USIの一致確認を通過した。元の大会用バイナリは置き換えていない。コミットなし。

## 条件

Intel Core i7-14700KF、20コア/28論理スレッド。`/proc/cpuinfo` の `avx_vnni` を確認した。現行の `.comment` と今回のコンパイラは、ともに GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)。現行の表示は `YaneuraOu NNUE 9.41git 64AVX2`。ソースは `820940ee31762ad598a9156a615b4b896ab8b321`。Makefileと設定は現行と同じ `YANEURAOU_ENGINE_NNUE`、halfkp_256x2-32-32、normal。normalは元から `-O3 -ffast-math -flto` を使う。

評価ファイルは `~/binbo-shogi-data/eval_dirs/task154_R_result30/nn.bin`。SHA256は `d1a1cb49b3075eb15b570f13e1aa974044552d44d8c78cb757d04348853d383f`。FV_SCALE 16、定跡なし、Hash 256MiBで統一した。本番の8GiBとは異なる。速度の結論はこのHashでの測定による。CPUの固定配置やクロック固定は行っていない。B3はclang++未導入のため省略した。

コピーしたソース 286 ファイルの内容は元と一致する。PGOが更新するのはmtimeと生成物だけ。元ソース・元バイナリ・評価ファイルを変更していない。作業ディレクトリ内に入力、バイナリ、コード、生ログを保存した。リポジトリにはこの報告だけを新規追加する。

## 検索と評価の一致

benchは `bench 256 1 13 default depth`。この版の無引数benchは4局面×15秒の時間固定なので、ノード数一致の比較には深さ13を明示した。4つの既定局面、Threads 1、同じHashで比較した。

| バイナリ | bench合計ノード | 追加4局面の手・ノード | 静的評価の不一致 | USI option行 |
|---|---:|---|---:|---|
| A-current | 529,024 | 基準 | 0 / 2,086 | 40行 |
| B0-avx2-normal | 529,024 | 4/4一致 | 0 / 2,086 | 40行完全一致 |
| B1-avxvnni-normal | 529,024 | 4/4一致 | 0 / 2,086 | 40行完全一致 |
| B2-avxvnni-pgo | 529,024 | 4/4一致 | 0 / 2,086 | 40行完全一致 |

静的評価は `position sfen ...` → `e` の整数値を全行照合した。`vm_mirror_check_a.sfen` 2,054局面と `parity_positions.sfen` 32局面。B0/B1/B2の不一致は、前者・後者とも0。追加の深さ13探索は後者の3〜6行目を使い、毎局 `usinewgame` と `isready` で状態を初期化した。全候補は以下の現行値と一致した。

| parity行 | bestmove | nodes |
|---:|---|---:|
| 3 | `bestmove 9b8b ponder 1f1e` | 50,097 |
| 4 | `bestmove 2c2d ponder 2h2d` | 101,428 |
| 5 | `bestmove 4f5e ponder 5g5f` | 92,316 |
| 6 | `bestmove 2g2f ponder 8c8d` | 27,764 |

`EnteringKingRule CSARule27`、`MaxMovesToDraw 512`、`NetworkDelay 120`、`NetworkDelay2 1120`、`USI_Ponder true` を設定して全検証を実行した。`getoption` の読戻しは `protocol_readback.json`。Ponderは深さ8の探索後もbestmoveを出さず、`ponderhit` と `stop` のそれぞれで返答した。4バイナリともエラー0。同一性の根拠は、ソース無改変と上記の再現試験。

## 速度

同じ追加4局面を使った。各スレッド数で5ラウンド、各候補20探索。毎回Hashを初期化し、1スレッドは `go movetime 2000`、28スレッドは `go movetime 10000`。最終infoのnodes/time由来のNPSを集計した。順序は A→B1→B2、B1→B2→A、B2→A→B1、A→B1→B2、B1→B2→A。

平均・中央値は20探索のNPS。比率と区間は、各ラウンドの4局面平均NPSを現行の同じラウンドで割り、5つの対応比の平均 ± t(4, .975) × SD/√5。t=2.77645。関門は対応比の平均が1.03以上、95%区間の下端が1より大きいこと。探索20本を独立標本として区間を狭めてはいない。

| Threads | バイナリ | 平均NPS | 中央値NPS | 現行比の平均 | 改善率95%区間 | 関門3 |
|---:|---|---:|---:|---:|---|---|
| 1 | A-current | 1,730,324 | 1,707,816 | 1.0000 | [+0.00, +0.00]% | 基準 |
| 1 | B1-avxvnni-normal | 1,718,220 | 1,724,438 | 0.9928 | [-6.70, +5.27]% | 参考 |
| 1 | B2-avxvnni-pgo | 1,813,686 | 1,785,782 | 1.0482 | [+4.00, +5.64]% | 参考 |
| 28 | A-current | 20,944,029 | 20,677,414 | 1.0000 | [+0.00, +0.00]% | 基準 |
| 28 | B1-avxvnni-normal | 22,170,712 | 21,815,068 | 1.0586 | [+5.10, +6.62]% | 合格 |
| 28 | B2-avxvnni-pgo | 22,637,672 | 22,450,778 | 1.0810 | [+6.03, +10.17]% | 合格 |

28スレッドのラウンド別平均NPS:

| round | A | B1 | B2 | B1/A | B2/A |
|---:|---:|---:|---:|---:|---:|
| 1 | 21,084,088 | 22,337,516 | 22,980,057 | 1.0594 | 1.0899 |
| 2 | 20,136,514 | 21,410,396 | 22,066,138 | 1.0633 | 1.0958 |
| 3 | 21,099,847 | 22,483,246 | 22,210,716 | 1.0656 | 1.0526 |
| 4 | 21,147,281 | 22,230,593 | 22,906,450 | 1.0512 | 1.0832 |
| 5 | 21,252,413 | 22,391,807 | 23,025,002 | 1.0536 | 1.0834 |

初期手順では1分loadが3を超える間、25秒ずつ待った。第4ラウンド前の待機中に中断されるまで、待機判定は15回。作者の途中指示により、再開後の第4・5ラウンドは待機せず連続実行した。第3ラウンドも全12探索が完了しており、捨てたデータ・やり直しはない。NPS工程全体は中断と待機込み20.11分、再開後は4.26分。28スレッドの指定探索時間は合計10分。

28スレッドの各ラウンド開始前のuptime:

| round | uptime |
|---:|---|
| 1 | `01:48:03 up 2 days, 16:13,  1 user,  load average: 1.38, 1.54, 2.41` |
| 2 | `01:52:41 up 2 days, 16:18,  1 user,  load average: 2.68, 6.52, 4.81` |
| 3 | `01:57:44 up 2 days, 16:23,  1 user,  load average: 2.66, 8.49, 6.73` |
| 4 | `02:01:38 up 2 days, 16:27,  1 user,  load average: 4.72, 10.70, 8.41` |
| 5 | `02:03:46 up 2 days, 16:29,  1 user,  load average: 24.16, 16.19, 10.73` |

上位CPUプロセスは第4・5ラウンド前に記録した。第1〜3ラウンドは追加指示より前で、取得していない。psの可視範囲はツールのsandbox内だけで、ブラウザ等のホストのプロセスは見えない。以下はその範囲の記録。

round 4:

```text
    PID COMMAND         %CPU NLWP
      9 python3         60.0    1
      1 codex            8.3    1
      2 bash             0.0    1
     11 ps               0.0    1
```

round 5:

```text
    PID COMMAND         %CPU NLWP
      9 python3          0.0    1
      1 codex            0.0    1
      2 bash             0.0    1
    538 ps               0.0    1
```

uptime全文・負荷履歴は `nps_results.json`。`logs/nps-progress.log` は再開後の進行記録。`nvidia-smi` はドライバへ接続できず、GPU学習の稼働は確認できなかった。速度の区間はこの5ラウンドにおけるばらつき。Hashは256MiB、デスクトップのある実行環境での測定。

## ビルドの再現

コピー・ビルドは作業ディレクトリ内。`prepare_source.py` にコピー手順、`build_candidates.py` に実行引数、`build_results.json` に所要時間を残した。以下はソースをコピー済みの状態からのコマンド。

```bash
source scripts/env.sh
TASK178_WORK="$DATA_ROOT/work/task178_fast_build"
cd "$TASK178_WORK/source"
make -j8 clean COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVX2
make -j8 normal COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVX2
cp YaneuraOu-by-gcc ../B0-avx2-normal
make -j8 clean COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVXVNNI
make -j8 normal COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVXVNNI
cp YaneuraOu-by-gcc ../B1-avxvnni-normal
make -j8 clean COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVXVNNI
make -j8 pgo COMPILER=g++ YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE TARGET_CPU=AVXVNNI .DEFAULT_GOAL=YaneuraOu-by-gcc
cp YaneuraOu-by-gcc ../B2-avxvnni-pgo
```

`../build/eval` は大会用evalディレクトリへのsymlink。Makefile自身の `pgo` は profgen → `./YaneuraOu-by-gcc EvalDir ../build/eval , bench , quit` → touch → profuse を実行する。無引数の再帰makeが既定の `all: clean ...` を呼ぶとgcdaも消すため、`.DEFAULT_GOAL=YaneuraOu-by-gcc` をコマンドラインで渡した。Makefileは無変更。48本のgcdaを保持し、profile-use時の欠落・不一致警告は0。PGO用benchはThreads 1、Hash 1024MiB、4局面合計60,001ms、64,089,922ノード。

| ビルド | 秒 (clean込み) |
|---|---:|
| B0-avx2-normal | 15.85 |
| B1-avxvnni-normal | 15.75 |
| B2-avxvnni-pgo | 83.13 |

AVXVNNIの既定フラグは `-DUSE_AVX2 -DUSE_BMI2 -DUSE_VNNI -DUSE_AVXVNNI -march=alderlake`。B1の逆アセンブルで `vpdpbusd` 命令を確認した。通常のビルドと検証は8並列以下。28スレッドは速度測定だけで使った。

## バイナリと保存物

| バイナリ | SHA256 |
|---|---|
| A-current | `0831b1c1925c4e7577fdd3cecc2ac3234f724686b44c4d495a678c969fd186b7` |
| B0-avx2-normal | `a1abaf627fc6a21353f76f53334acb4ab230a43145189c170ffc2d0e4bae2cd9` |
| B1-avxvnni-normal | `fc6e0f743e1f8a3594da472231abcda2b2ec59a330e4b39cd60ff35f72f12aed` |
| B2-avxvnni-pgo | `38f89c23c2117863ff832b16546d34ba435aeb607c589ebb36969e23f3d6940f` |

作業領域は `~/binbo-shogi-data/work/task178_fast_build/`。`report.md` と `summary.json` が結論。`gate12.json` は全静的評価値と深さ13の結果、`nps_results.json` は120探索と負荷記録、`environment.json` は環境、`source_manifest.json` はソース全ファイルのSHA256、`logs/` はUSI・ビルドの生ログ。`check_gates.py` と `measure_nps.py`、`run_nps.sh` で再現できる。

```bash
source scripts/env.sh
cd "$DATA_ROOT/work/task178_fast_build"
python3 check_gates.py
./run_nps.sh /path/to/binbo-shogi
python3 finalize_report.py
```

リポジトリの動作コードとルールは変更していないため、Rustの回帰テストとperftは対象外。STATUS.mdを含む既存ファイルには触れていない。

## 追加の確認 (Claude, 10/24 02:20): 実際の対局で同じになるか

候補側のエンジンだけを B2 に替え, 採用 eval の既存チャンク (s280 tanuki- c00, seed 20280000, 100 局, 候補 10 万ノード / tanuki- 2 万ノード) を指し直した。
時間の列を除いて 1 局ずつ比べると, 100 局すべてで勝敗・終わり方・手数・各側の平均ノード数・序盤が一致した。
違いは 3 局の「平均の深さ」の表示だけ (候補側 14.16 / 14.19, 14.27 / 14.30, **替えていない tanuki- 側** 10.78 / 10.79)。
替えていない側でも出るので, 探索の違いではなく, 対局の道具が深さを読む時機の差と読む (Xeon の確認でも同じ形の差が 1 局あった)。
比べた道具 `experiments/alphazero/xeon_cmp_tsv.py`, 結果 `~/binbo-shogi-data/work/task178_fast_build/e2e/`。

## 本番に入れるとき (作者の承認後に行う, まだ変えていない)
- 候補は永続の置き場にコピーした: `~/binbo-shogi-data/engines/yaneuraou_fast/YaneuraOu-avxvnni-pgo` (sha256 `38f89c23…`, `BUILD_INFO.txt` つき)。AVX-VNNI が要るので Xeon (待機機) では動かない。待機機は今の AVX2 版を使う。
- 変える所は `tools/denryusen/prod_start.sh` の `engine=` の 1 行 (上の置き場へ)。あわせてエンジンの sha256 の点検を 1 行足すとよい (eval と同じ形)。手順書 `docs/denryusen7_runbook.md` の冒頭の「やねうら王 (g++ ビルド)」の記述も直す。
- 入れたら, 本番の設定でローカルの試験用サーバのブイ対局と floodgate を少し指して, 時間切れ・異常がないことを確かめる (手順書 §6)。
