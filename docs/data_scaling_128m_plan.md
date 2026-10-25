# 64M → 128M 教師データ拡張の準備 (Task 140)

2026-10-19。**設計と小規模 CPU 試験のみ。本生成・GPU 処理・学習は実行していない。**
採用 eval は Task 129 のまま (書いた時点。10/21 現在の採用 eval は `task154_R_result30`, `STATUS.md` 参照)。学習の実装技術は Task 135 の速い経路を基準にする。
本書の GB は 10^9 B、GiB は 2^30 B。パスの起点は `source scripts/env.sh` の `DATA_ROOT`。

## 結論と未解決の前提

- 現行の「64M」は **63,950,000 train + 50,000 val**。64,000,000 train を足せば
  **127,950,000 train** になる。128,000,000 train を厳密に要求するなら追加は 64,050,000。
- 現行 64M は一意局面数ではない。新しい seed は棋譜・局面の独立性を保証しない。
  **追加分は静止化後に既存 train・固定 val・追加分自身と照合し、64M の新規一意局面を確保する**。
  元の 63.95M に重複があれば、追加後も「128M 一意局面」とは呼べない。
- 156 B/行の既存 compact を全行常駐させると、127.95M で **19.960 GB = 18.59 GiB**。
  RTX 4060 Ti 16 GB には索引だけで収まらない。`--limit-train` で半分に切ると目的を満たさない。
  第一案は **旧 63.95M を GPU 常駐、新しい 64M を CPU RAM から小バッチ転送**。
  既存 CLI はこの混在を未対応。後続タスクで実装・計測・一致確認が必要。
- この環境の空きは調査時点で **約 52 GiB** (`df -h . /tmp`、同一ファイルシステム)。
  中間ファイルを素直に残す計画には不足する。既存成果物を消さず、追加の空きを確保してから本走する。
- 永続領域を読み取り調査したところ、`engines/gikou/gikou`、静止化用の
  `eval_dirs/task109_student_64m_e24/nn.bin`、32M キャッシュ、Task 107 の一部中間物は現存しない。
  元 hcpe、`shared_64m{,_qs129}`、`compact_qs129`、Task 129 ラベルは残っている。
  **技巧2本体・パラメータと正確な Task 109 eval の復元が、本レシピで新規生成する前提**。
  保存方針文書の古い配置例より、実ファイルの検査を優先する。

## 既存 64M の来歴と実際の条件

参照した一次資料はリポジトリのソースと、永続領域の manifest / meta / 実行ログ。
履歴の説明と、現在そのまま再実行できる手順を分ける。

### 1. 技巧2 depth 8 の自己対局を hcpe に保存

生成器は [`src/teacher/gen.rs`](../src/teacher/gen.rs) の `gen-teacher`。
永続領域の `teacher/run{8m_e,16m_f,32m_g}.gen.sh` と全 run の `manifest.json` を確認した。
生成スクリプトは git 追跡外で、古い clone の絶対パスを持つ。**そのまま起動せず、現在の worktree の
`scripts/env.sh` を使う wrapper にする**。

| run (キャッシュ内の順序) | 元行数 | base_seed | shard 数 (10,000 行/個) |
|---|---:|---:|---:|
| `run1m` | 1,000,000 | 20260621 | 100 (95 train / 5 val) |
| `run1m_b` | 1,000,000 | 20270101 | 100 |
| `run2m_c` | 2,000,000 | 20280101 | 200 |
| `run4m_d` | 4,000,000 | 20290101 | 400 |
| `run8m_e` | 8,000,000 | 20270627 | 800 |
| `run16m_f` | 16,000,000 | 20290629 | 1,600 |
| `run32m_g` | 32,000,000 | 20310101 | 3,200 |

共通条件: 技巧2 v2.0.2、`--depth 8 --random-plies 6 --max-plies 256 --shard-size 10000
--timeout-ms 30000`、OwnBook=false、1 engine=1 thread。通常 16 workers。
各 shard は seed=`base_seed + index` の新しい技巧2プロセスで作る。各対局の最初の 6 手は合法手から
一様ランダム、その後は技巧2の最善手を指し、探索直前の局面と手番側 cp を保存する。
詰み・投了・手数上限・クラッシュを区別して manifest に集計する。クラッシュ/上限/途中打切りの
勝敗不明は draw/unknown として扱う。実際の run32m_g は crash_cutoffs=2,626、timeouts=0。

正本は **38 B/行 hcpe** (`hcp[32], eval:i16, bestMove16, gameResult, dummy`) と manifest。
`shard_XXXX.expected.tsv` は SFEN・cp・最善手・結果・hcp の照合用で、`gN_pM` から shard 内の
game ID も取れる。ゲームは shard をまたがず、未完了の最後のゲームは勝敗不明になりうる。
完了 shard の atomic write と manifest によって同条件で resume できる。行重複除去は行っていない。

### 2. raw HalfKP キャッシュと探索ラベル

[`trainer/prepare_data.py`](../trainer/prepare_data.py) は primary=`run1m` の末尾
shards **95–99 の 50,000 行を val** にし、extra_runs は全 done shard を train にする。
`--canonical-shared` の固定レイアウトは own 39 列 + opp 38 列、int32 × 77 = 308 B/行。
evals は int16、wdl は手番側 int8。左右反転は保存行を増やすのでなく学習時に確率 0.5 で適用。

32M までの `shared_32m` に、[`t107_scale64.py::build_cache`](../experiments/alphazero/t107_scale64.py)
が `run32m_g` を追加した。旧 31.95M をそのままコピーし、新しい 32M を Pool(12) で変換、val は固定。
最終 shard の再変換一致、新規 hcpe と cp の一致、特徴範囲を検査した。実測 **118 秒**。
`caches/shared_64m/meta.json` に順序・行数・元 run 一覧が残っている。

ラベラーは [`t099_search_label.py`](../experiments/alphazero/t099_search_label.py) の
`build_batched_search` / `score_sfens`。キャッシュ順への適用は
[`t100_search_student.py::rescore_cache_with_search`](../experiments/alphazero/t100_search_student.py)、
全 32M は Task 104、追加分は Task 107。

- WS-T checkpoint: `~/task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt`。
  実行 Python は `~/task074_frameworks/venv_pgx/bin/python` (torch 用 `VENV_PY` と別)。
- **Gumbel MCTS S=8、batch=128、gumbel_scale=0.0、PRNGKey(0)**。
  ラベルは `search_tree.summary().value`。生 value head や visit-weighted Q ではない。
- `cp = clip(1200 * atanh(clip(v, -1+1e-7, 1-1e-7)), ±5000)` を int16 にキャストする。
  `current_player=color` と合法手 mask の再構成は必須。先後 mate-in-1 の `sign_oracle` を通す。
- Task 104 の parts を再利用し、Task 107 が新しい 32M を約 9h でラベル化。
  `work/task107/rescore_search/{train,val}_dlshogi_evals.npy`。raw head は配線照合用で学習目標ではない。
- Task 107 の「王手/最善手が駒取り」を落とす mask は当時の学習用。**Task 129 では使わない**。

### 3. 静止探索の葉 Q に置換し、Q 自身をラベル化

部品は [`qs_replace.py`](../experiments/alphazero/qs_replace.py)。hcpe を **40 B/行 PSV** にし、
YaneuraOu の独自 `qsearch_psv in out 4` を使う。
**eval=`task109_student_64m_e24`、4 workers、hash 16 MB、各ファイルの前に `usinewgame`**。
score=+1 の目印を送り、戻り値 −1 で PV 奇数手の手番反転を検出する。レコード順は維持する。

- [`t119_qs_replace.py`](../experiments/alphazero/t119_qs_replace.py): 最初の 31.95M の train を
  静止化、約 9 分。変わった行の特徴を書き換え、eval/wdl/引継ラベルを奇数 PV で反転。
- [`t128_qs_relabel.py`](../experiments/alphazero/t128_qs_relabel.py): 同じ静止化を再現し、
  **変わった 13,140,149 行を Q の手番で WS-T S=8 により付け直した**。
- [`t129_qs_relabel_64m.py`](../experiments/alphazero/t129_qs_relabel_64m.py): 前半を再利用し、
  `run32m_g` 32M を 1M/part で静止化。変化 **13,164,704 行 (41.1397%)**、
  手番反転 **27.4419%**。Q に付け直す速度 686 changed rows/s、約 5.3h。
  変わらなかった行は Task 107 の探索ラベルをそのまま使う。**付け直した Q のラベルをさらに反転しない**。

最終成果物は `caches/shared_64m_qs129` と `work/task129/labels`。
全体の置換率 41.1335% (26,304,853 行)、特徴不一致=0、葉で王手=0、新旧ラベル相関 0.93546。
val は静止化せず、元の 50k とその探索ラベルを使う。詳細は
[`reports/task129_qs_relabel_64m.md`](../reports/task129_qs_relabel_64m.md)。

### 4. NNUE 学習と高速化

Task 129 は QAT shared HalfKP **256×2–32–32–1、`--qat --qat-tie --factorize`**、
lr=6e-5、12 epochs、batch=1024、`--scale-cp 600 --augment-mirror`、mask なし。
デフォルトも含めると cp 回帰、target clamp ±2000、Huber delta=600、Adam、weight_decay=0、
lr schedule=none、seed=20260621、random init。mirror permutation は
`caches/shared_8m_mirror_perm_shared.npy` (`--mirror-cache` 指定は `_shared` を付ける前の名前)。
旧経路は `--mmap`。best epoch=10、val loss=41,086。

Task 135 は同じデータ/レシピを **`--fast --compact-cache caches/compact_qs129`** で学習。
[`compact_cache.py`](../trainer/compact_cache.py) は king uint8×2 + code uint16×77=156 B/行。
全 train+val を変換・復号照合し **224.806 秒、不一致 0、escape 0** (compact.json)。
`--fast` は sparse FT + fused Adam + async loss、compact は GPU に全行常駐する。
実数の加算順序による差の検査は [`reports/task135_fast_trainer.md`](../reports/task135_fast_trainer.md)。

計測値に小差がある: 同 report の旧 2,454 秒/epoch に対し、保存済み Task 129 history の合計は
**29,832.55 秒 / 12 = 2,486.05 秒/epoch (8.287h)**。
Task 135 history は **6,520.79 秒 / 12 = 543.399 秒/epoch (1.811h)**。
本書の拡張見積もりは後者を使う。強さの採用は Task 129 のまま。
学習後は `trainer.export_nnue` → `trainer.binbo_to_yane` → 0cp gate → 外部物差し。

## 追加 64M の実行計画 (後続タスク。本タスクでは起動しない)

1. **入力固定・復元**: 現行 cache/labels/val の SHA256、教師 checkpoint、技巧2の本体とパラメータ、
   YaneuraOu 本体と Task 109 eval の hash、git HEAD/dirty diff、依存版を保存。
   Task 109 が復元できず採用 eval で静止化するなら、レシピ変更として別に記録・比較する。
   古い t107/t129 ドライバは固定 N32/N64・出力先・削除処理・commit 要求を持つので**無変更で再実行しない**。
2. **新しい raw run**: `teacher/run64m_h` を新設。base_seed 案 **20720000**、
   6,400 shards なら **20720000–20726399**。本走直前に `teacher/*/manifest.json` の
   done/pending 両方の全 seed と非衝突を検査して予約する。補充分は別 run/別予約範囲にする。
   生成条件は上記 depth8 レシピを保持する。起動の形は次のとおり (未実行):

   ```bash
   source scripts/env.sh
   "$BINBO" gen-teacher "$GIKOU" "$TEACHER_DIR/run64m_h" 64000000 \
     --shard-size 10000 --workers 16 --depth 8 --random-plies 6 \
     --max-plies 256 --seed 20720000 --timeout-ms 30000
   ```

   実運用 wrapper はこの worktree を起点にし、`scripts/detached.sh` と停止/再開を使う。
   長時間走行前のスリープ無効化は作者側の確認が必要 (`docs/data_storage.md`)。
3. **Q 作成 → 重複除去**: 1M 以下/part で hcpe→PSV→qsearch。エラー/非合法 PV=0、件数・順序・
   不変行の特徴一致・手番反転を検査。元行→Q→採否→最終 row ID の対応表を永続化する。
   下記の既存/val/新規重複を除き、不足分を補充して**採用 Q が 64M に達するまで**繰り返す。
4. **新規採用 Q のみ S=8 で一度ラベル化**: 新しい raw P に全部ラベルを付けてから 41% を
   付け直す方法も Task 129 と同じ目標になるが、余分な GPU 時間を要する。
   **Q を先に確定し、変化の有無を問わず全採用 Q を一度だけ採点**する案を推奨。
   既存 63.95M の Q とラベルは完全再利用。sign oracle、有限値、int16 変換、row hash を検査し、
   小標本で従来の「P を採点 → 変化行だけ Q で再採点」と一致することを GPU 上で先に確かめる。
5. **組立**: old Q 63.95M + new Q 64M の順で train を作り、val と探索ラベルは byte-identical に固定。
   `prepare_data.convert_shard` 相当を小分割 memmap で使う。汎用 `prepare_data.main` の全行
   `np.zeros((N,77),int32)` は約 40GB RAM を要求するので本番128M組立に直接使わない。
   t107 の並列 memmap 方式を一般化し、part ごとの入力 hash/行範囲/完了 marker を保存する。
   `meta.json` があるだけで再利用せず、全 part とラベルの長さ・由来を検査して atomic publish。
6. **compact → 128M 学習**: 下記の residency 拡張を先に検証。Task 129 の学習条件と
   Task 135 の高速化を固定し、12 epochs で比較する。同 epoch なら更新回数も約2倍になることを記録。
   元 64M の速い経路の対照も持つ。データ量だけの寄与を分ける追加比較なら 128M×6 epochs
   (約同更新数) だが、本タスクでは走らせない。
7. **評価**: 0cp gate、非合法手/失敗 0、固定物差し tanuki-@20k/AobaNNUE@10k、各100k nodes・
   各800局、標準 seeds20272001–08。採用判断は `docs/evaluation_protocol.md` に従い新 seed 確認まで。
   val は元の非静止化分布なので loss だけで棋力を判断しない。

## 重複・棋譜重なり・検証への混入

**比較キーは局面の board + hands + side-to-move (手数は除外)**。cp、結果、最善手はキーに含めない。
hcpe と PSV の32バイトは別の符号化なので直接比較しない。デコードして共通形式にする。
左右反転 augmentation があるため、val は `min(core(P), core(mirror(P)))` の同値類でも排除する。
新規64Mも同じ基準で一意化すると、水増しでない情報量を数えやすい。色交換/180度回転を同一視するかは
ラベル視点も絡むため別統計にし、勝手に混ぜない。

- **val**: run1m 95–99 (別名 `run1m_holdout99` は同じ内容)。元 val に加え、その mirror、必要なら
  同じ固定evalによる val の Q も除外集合にする。元 P が val でも Q が別なら許す運用はしない。
  新規 raw P と新規 Q の両方を検査する。教師の元学習用 holdout との重なりも由来台帳で確認する。
- **既存 train**: old raw P と old Q への重複を別々に報告。追加の最終 Q が old Q と同一なら捨てる。
  raw P が old raw と同一の行も採らない。Q は取り合いの隣接局面から同じ葉へ収束するため、
  raw のみの一意化では足りない。旧 cache の features から SFEN を復元できる
  (`kp.canonical_shared_row_to_sfen`) が、大規模スキャンは別途予算化する。
- **棋譜単位**: expected.tsv の `(run,shard,gN)` と局面列から game fingerprint を作る。
  完全一致棋譜、同一6手開始局面、共有 prefix の割合を報告。固定 val と同じ開始局面の新規ゲームは
  全体を除外し、val game の fingerprint が再登場した場合も全体を除外する。
  それでも別開始からの合流はあるので局面照合を併用する。
- shard 分割は同一**生成ゲーム**の train/val 分割を避けるが、別 seed の同じ棋譜や同じ局面は避けない。
  既存64Mの train/val overlap は未集計。発見しても旧データを無言で変更せず、数を報告し、
  必要なら「汚染を除いた64M対照」と128Mを同条件で比較する。既存を固定した比較の限界も残す。
- 全128Mの Python `set(str)` は RAM を使いすぎる。128-bit hash + source/row ID を prefix partition
  して disk sort/merge する案。hash一致時は元の正規化局面で最終照合し、衝突を誤除外にしない。
  Bloom filter は事前絞り込みのみ。確定的な行順で先着採用し、resume でも採否を変えない。

採択率を r とすれば必要な raw 候補数は概ね **64M/r**。r は既存ログからは分からない。
まず 10k〜100k の後続 pilot で測り、補充時間・空き容量を見直す。
今回の tiny replay は 128 raw → 100 unique Q だったが、同一棋譜の短い部分であり r の推定には使えない。
除去により頻度分布も変わるので、「データ量のみ変更」とは厳密には言えない点を記録する。

## 機械時間の予算

14700KF/RTX4060Ti16GB の過去実測からの外挿。CPU/GPU の表記は占有する主な資源で、
GPU処理にも Python変換などCPU時間を含む。今のCPUのみのsandboxではGPU速度は未計測。

| 工程 | 根拠 | 追加64M/合計127.95Mの予算 |
|---|---|---|
| raw自己対局 (CPU16 workers) | run32m_g 最終attempt: **新規18.05M / 11,071.425秒** | **10.9h/64M候補**。熱・負荷を見込み11–15h、重複補充は概ね1/r倍 |
| Q変換・特徴 (CPU4 workersほか) | Task119 31.95M約9分、t107追加32M変換118秒 | 64M約18分 + 特徴組立数分。I/O込み**20–40分の推定** |
| 重複/ゲーム照合 (CPU/disk) | 全64Mの完全照合の実測なし | **1–4hを仮置き**。SFEN復元・sortが律速なら超過。pilot必須 |
| WS-T S8, Qを全行一度 (GPU) | Task129 **13,164,704 / 686 rows/s** (Q生成等も含む複合速度) | **25.9h/64M採用Q**、概算24–30h。重複除去後だけ採点すれば補充で増えない |
| P全部採点+変更Q再採点 (GPU,代替) | Task107追加32M約537分→約993/s、変更率41.14% | P約17.9h + Q約10.7h = **約28.6h**。実行グラフ差で速度比較は概算 |
| 128Mキャッシュ組立 (CPU/disk) | t107 118秒、旧31.95Mコピー16秒、追加32M並列変換 | **3–10分の推定**。新旧を共に再変換しない |
| compact変換+全行復号照合 (CPU/disk) | 64M全体224.806秒 | **約450秒(7.5分)** の線形外挿、5–15分を予算化 |
| 128M×12 epochs (GPU) | Task135 6,520.79秒×127.95/63.95 | 下限目安 **3.62h**。混在転送案は**4–6hを仮置き**、未実測 |
| export/0cp・外部判定 (CPU) | 固定物差し、候補1,600局+必要な確認 | 別枠。負荷依存で未見積もり、学習時間に含めない |

**生成ログの読み方**: `teacher/run32m_g.gen.log` は再開を含む。
6.56M/約5,055秒 (12workers)、7.39M/約7,182秒 (8workers)、18.05M/11,071秒 (16workers)。
最後の `total_positions=32000000` と `elapsed_min=184.524` を割ると約2倍速く誤積算する。
実行区間合計は約6.47h/32M、待ち時間込み21:32→翌09:22は約11.8h。
また Task107 の表示 `rate=1981/s` は再利用31.95Mも分子に含むので、追加32M/約537分で再計算した。

採用r=1の理想で、生成→照合→一度ラベル→学習を直列に並べると **概ね42–57h + 外部判定**。
実際は raw生成の補充と復元作業が上乗せ。工程間でCPU/GPUを重ねる場合もGPUジョブは同時実行せず、
他の時間つき対局の負荷を避けて計測する。

## ディスクとホスト RAM

| データ | 追加64Mぶん | 合計127.95Mぶん/補足 |
|---|---:|---|
| raw hcpe (38 B) | 2.432 GB | 4.862 GB (旧val除外換算) |
| expected.tsv | **約10.87 GB** | run32m_gの実サイズ5,433,567,873 B/32Mから外挿 |
| int32索引+eval+wdl (311 B) | 19.904 GB | **39.792 GB**、元PとQを両方残すと倍 |
| compact (156 B、escapeなし) | 9.984 GB | **19.960 GB**、manifest/headerは微小 |
| S8 labels (int16) | 0.128 GB | 0.256 GB |
| raw/Q PSV一時ファイル | 1M chunkで各40 MB | **同時80 MB程度**。全量二重保存は不要 |
| Q再ラベルparts (従来形式) | 約8.44 GB | Task129 parts32個=4,218,423,616 Bから外挿。採点後のfeats二重保存を避ければ削減可能 |
| 除外key/index+sort scratch | 未実測 | hash128+row64だけでも128Mで約3.07 GB。複数集合/複製で**10–30GB枠**を仮置き |
| モデル/Adam/途中保存 | 数GB枠 | 45M程度の重み、best/final/stateを保存。旧モデルを保持 |

既存を保持し、新raw+TSV約13.3GB、新combinedQ39.8GB、compact20.0GB、parts8.4GB、
labels/一時/モデル/照合用領域を足すと **追加で概ね95–115GB**。
原Pの64M索引も別に残すならさらに19.9GB。重複補充・atomic publishの旧tmp共存も考え、
**空き120–150GB程度を目標**にする (厳密値はrと保存方式を決めて再計算)。
52GiBの現空きのまま走らせない。append fragments方式でcombinedコピーを避ける設計は可能だが、
現在のtrainer/compact source stampは単一の元int32ファイルを要求するので追加実装が必要。

RAMは full compact約20GB + labels + CPU randperm **1.024GB** + OS/page cacheが必要。
第一案は **追加半分だけ約10GBをRAMに保持**し、旧半分はGPUへ読み終わった後のCPUコピーを解放する。
pin memoryは全量でなく小さな二重bufferに限定。128MをPythonオブジェクト列へ展開しない。
valの索引は15.4MB、compact7.8MBで小さい。

原 hcpe/manifest/対応表/ラベル/高コストpartsは `DATA_ROOT` の永続領域へ。
128M索引の再生成が実測で数分を超えるなら、旧「KPキャッシュは/tmp可」を機械的に適用せず永続側に置く。
`/tmp` はこの試験のように数秒で作り直せるfixtureに限る。今回永続データへの書き込みはしていない。

## 16GB GPUでの学習方式と選択

以下は**未実装の案**。現在の `CompactCache._resident` は全行をdeviceに確保する。
`train_nnue --fast` にstreaming/chunk指定を加えるだけで動く現存オプションはない。
索引以外にも全target fp32 **0.512GB**、val、約0.18GBのFT重みと勾配/Adam2本 (計約0.72GB)、
量子化/sparse FTの一時テンソル、allocator余白が必要。実測peak reserved/allocatedでサイズを決める。

| 案 | GPUのデータ量 | 利点 | 負担・比較上の注意 |
|---|---|---|---|
| **A. 半分常駐+半分CPU転送 (推奨)** | 旧63.95M索引9.976GB + 全targets0.512GB + 小buffer | global randperm、mirror乱数、Adam更新順を保てる。追加halfだけCPU RAM約10GB | RAMからencoded行をgatherしpinned staging→非同期転送→GPU復号。SATAからのランダムmmapは避ける。実装はCPU/GPU行混在を扱う |
| B. 16–32Mずつ分割常駐 | 索引2.50–4.99GB/塊 | RAM不足でもSSD順読み。少ないVRAMで余裕が大きい | 塊順+塊内shuffleはglobal shuffleと異なり学習条件が変わる。元runと塊を一致させず、事前にold/new混合。epochを通してoptimizerを保持 |
| C. codeをbit単位で詰める | 12bitなら**118 B/行、15.098GB**。11bitなら108 B/行、13.819GB | 12bitなら現code0..2183を可逆に格納できる | 12bitでもtargets/Adam/作業域を含めると余裕不足。**11bit最大2047では現codeは収まらず**、別escape/符号設計が必須。単純なbit詰めだけで解決とは言えない |
| D. own側だけ保存しoppを復元 | anchors2 + own38×uint16 + turn1=**79 B/行**、約10.108GB | 128M全行常駐に十分な余裕。`canonical_shared_row_to_sfen`もownから復元可能なことを示す | GPUで盤上/持駒・相手視点変換・整列が必要。元77列の順序/重複/例外保持を検証し、新版形式とescapeを設計する。開発量最大 |

Aの具体案: `__getitem__(global_rows)` は元のバッチ行順を維持してold/newを振り分け、
newだけRAM上のcodes/kingsからgatherする。小さいpinned bufferとCUDA stream/eventで先読みし、
old GPU gatherとnew転送結果を元の位置にscatterして**同じ77列**を返す。
resident経路の`rows`は現在GPUに移されるので、CPU側元row IDも保存し、毎バッチGPU→CPU同期を作らない。
targetsは全量GPUでよい。復号後のmirror乱数、バッチ境界、Adam step、epoch loss集計は従来どおり。
escapeが出た場合はbases/escapesの配置も含める。0件と仮定して捨てない。

転送量は64M×156B ≈ **9.98GB/epoch**。仮に実効5–10GB/sなら転送そのものの下限は約1–2秒だが、
これは**仮定した帯域の算術**で実測予測ではない。batchあたり約512行のCPU gather・起動回数・同期が
支配しうる。8.9ms/stepの計算と重ねられるかを1–10k batchesで測る。ホスト全体をpinしてはいけない。

Bでは20GB/epochの順次H2Dが下限だが、二重bufferとモデルを含めても余裕あり。
chunk順序を変えた同一データの対照が必要。途中保存はepoch/chunk/row cursor、chunk順と局所shuffleの
generator状態、mirror状態、optimizer、schedulerを保存し、中断なしと再開後を照合する。
同じglobal randpermを保つため毎バッチchunkをswapする方式はI/O過多になるため避ける。

**検証順序**: CPUで全行encode/decode完全一致 → GPUで元経路と同じrow/batch/mirror入力の出力・勾配・
lossを比較 → resident/streamed混在と境界・最後の端数・escape・stop/resumeを検査 → VRAM peakと速度 →
64M対照再学習/0cp → 128M本走。加算順序を変えるDはTask135の加算順を変えた対照程度の差かも確認する。
本タスクでは学習器・評価関数・探索コードは変更しない。

## 小規模 dry run の使い方と今回の結果

[`t140_scale128_dryrun.py`](../experiments/alphazero/t140_scale128_dryrun.py) は最大512行、既定256行。
新規出力ディレクトリ以外には書かず、CUDAを隠し、JAXをimportしない。子プロセスの時間上限あり。
通常モードはこのworktreeの`BINBO`と技巧2を使い2 shardの新規対局→独立cshogi照合→
`prepare_data`→trainのみqsearch→特徴復元→compact全行照合→GPU用SFEN行リストまで進む。
本番valの代わりに独立shardの128行をfixture valにする。本番の固定valを作り直す手順ではない。

今回、正準の技巧2本体とTask109 evalが欠けていたため通常モードはpreflightで停止した。
明示的なfixtureモードで、既存`run32m_g`の先頭2 shardから各128行を読み取り、
**Task129 evalで**qsearchした。これは新規生成・元のQとの一致を検証したものではない。

```bash
source scripts/env.sh
PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" experiments/alphazero/t140_scale128_dryrun.py \
  --replay-run "$TEACHER_DIR/run32m_g" \
  --fixture-eval-dir "$DATA_ROOT/eval_dirs/task129_student_64m_qsrel_e12" \
  --out /tmp/task140_dryrun_check
```

結果: **train128 + val128、changed53、flip33、decode_errors=illegal_pv=0、葉で王手0**。
qsearch 1/4 workers のPSV出力完全一致、Rust expected.tsvとcshogiのSFEN/cp/合法最善手/結果一致、
HalfKP→SFEN往復一致、compactのtrain/val全列不一致0・escape0。
raw unique128→leaf unique100、mirror込みでも100、fixture valとの交差0。
別途 `CompactCache(device="cpu")` の実際のtorch復号経路でも256行×77列の完全一致を確認した。
train特徴SHA256=`8a20df4eb652f74118b9a838334a943bcf8d3f6553dc4a78a8435499a57dc46a`。
処理本体 **0.721秒** (Python/torch import時間は除く)。

`status.json` は `model_ready=false, recipe_exact=false, generation=replay_existing_hcpe`。
`pending_gpu.jsonl` のsplit/row/SFENを後段への契約とし、コードの **TODO GPU 1/2** に教師採点と学習を記載した。
fixtureキャッシュのeval/wdlはGikou診断値を手番に合わせただけで、**WS-Tラベルファイルは作っていない**。
GPU符号oracle・本番64Mとの全面重複照合・新規生成・正確なTask109静止化・128M学習/VRAMは未検証。
一時fixtureは削除されてもこのコマンドから数秒で再生成できる。

次の着手点は、欠けた正準依存の復元、空き容量の確保、重複率pilot、方式Aの小標本実装/計測。
本タスクはそれらの長時間ジョブを開始せず、commit/pushもしない。
