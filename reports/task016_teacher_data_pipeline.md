# Task 016: 教師データ大量生成基盤 + 100万局面本走

## 結論

- shard 分割・manifest(JSON)・レジューム・技巧2プロセス並列・タイムアウト/クラッシュ復旧・
  shard 検証を備えた教師データ生成基盤を Rust で実装した(`gen-teacher` コマンド)。
- レジュームは「shard を書き終えてから manifest に完了印」の write-ahead 順序を守る。
  SIGKILL→再開で、中断前に完了した shard はスキップし、最終的に目標局面数に到達することを実測で確認した。
- 並列(技巧2 を worker 数ぶん別プロセスで起動)でも、各 shard は seed で決まる。
  **クラッシュを伴わない shard は worker 数・完成順に依らずバイト一致**(再現可能)を実測確認した。
- depth 8 で技巧2が**特定局面(暖機状態依存)で異常終了**することを発見。該当 game を破棄して
  エンジンを再起動し shard を継続する復旧を実装。これにより 1 局面のクラッシュが shard 全体を壊さない。
- **100万局面を本走完走**(8 worker / 15.4 分 / 単一プロセス比 4.39×)。クラッシュ 83 件は全て自動復旧
  (失敗 shard 0)。検証: 全 shard の件数・サイズ一致、標本 30,000 局面の KP 可逆性不一致 0、
  cshogi 照合 ok、手番側視点(決定的局面の符号)100% 一致。

## 生成基盤の設計

### コマンド

```
gen-teacher <gikou_path> <out_dir> <target>
  [--shard-size N] [--workers N] [--depth N] [--random-plies N]
  [--max-plies N] [--seed N] [--timeout-ms N] [--own-book BOOL] [--max-retries N]
```

- 実装: `src/teacher/gen.rs`(オーケストレーション)、`src/teacher/manifest.rs`(JSON 目録)、
  `src/teacher/gikou.rs`(USI 子プロセス + reader スレッド + タイムアウト + 再起動)。
- 生成条件は Task 015 を踏襲: `OwnBook=false`, `Threads=1`, `ResignScore=-32600`, depth 8,
  出だしランダム 6 plies, 最大 256 手。eval は技巧2 `score cp` を手番側視点で無変換保存。

### shard と manifest

- shard 数 = `ceil(target / shard_size)`。shard i の seed = `base_seed + i`。
  shard i のファイル = `shard_{i:04}.hcpe`(+ cshogi 照合用 `shard_{i:04}.expected.tsv`)。
- `manifest.json`(人間可読 JSON。`serde_json` はオフライン不可のため自作の reader/writer):
  - meta: gikou_path, depth, random_plies, max_plies, base_seed, shard_size,
    target_positions, own_book, threads_per_engine, timeout_ms, hcpe_entry_size,
    binbo_git_commit, created_at。
  - 各 shard: index, file, seed, target_count, count, done, games,
    mate_cutoffs, resign_cutoffs, max_ply_cutoffs, crash_cutoffs, timeouts,
    engine_restarts, elapsed_ms, eval_min, eval_max。

### レジューム(write-ahead)

1. shard をメモリ上で生成 → `shard.hcpe.tmp` / `expected.tsv.tmp` を書く → atomic rename。
2. **その後**に manifest の当該 shard を `done=true` にして manifest を atomic 保存(tmp+rename)。

- (1)と(2)の間でクラッシュ → shard ファイルはあるが done でない → 再生成(上書き)。
- (1)途中でクラッシュ → `.tmp` だけ残る → 起動時に掃除して再生成。
- manifest 自体も tmp+rename なので半端書きにならない。
- 再開時、`done` かつ「ファイルが存在し byte 長 = count×38」の shard だけ信用し、
  それ以外は pending に戻して再生成する。

### 並列化

- worker を `--workers` 個 std::thread で起動。各 worker は共有キューから shard index を取り、
  **shard ごとに技巧2を新規プロセスで起動**して生成する。
- なぜ shard ごとに新規プロセスか: 技巧2は `usinewgame`/`isready` でも内部状態(TT 等)を
  リセットせず、同一プロセスで連続生成すると shard の内容が「その worker が直前に何を探索したか」に
  依存してしまう(下記「発見」)。新規プロセスにすれば shard の内容は seed だけで決まる。
  技巧2の起動は約 0.3 秒で、1 shard(1万局面=約40〜70秒の解析)に対し 1% 未満の overhead。

### タイムアウト・異常処理

- `gikou.rs` は子プロセスの stdout を専用 reader スレッドで読み、`mpsc` チャネル + `recv_timeout`
  で **bestmove 待ちにタイムアウト**を設ける(`--timeout-ms`、既定 30 秒)。
- 解析中にタイムアウト(ハング)/EOF(異常終了)を検知すると、その game を破棄してエンジンを
  新規プロセスで再起動し、shard を継続する。イベントは `timeouts` / `crash_cutoffs` /
  `engine_restarts` として manifest と stderr ログに記録。
- mate / resign / 最大手数はそれぞれ `mate_cutoffs` / `resign_cutoffs` / `max_ply_cutoffs` に記録。
- shard 生成が回復不能(エンジン再起動失敗・不正 bestmove 等)なら、その shard を `--max-retries`
  回まで丸ごとやり直し、それでも駄目なら failed として記録し、run は非0終了(無言で止まらない)。

### shard 検証

- Rust: 各 shard 生成後に「ファイル byte 長 = count×38」を確認。
- Python `trainer/verify_shards.py`: manifest を読み、**全 shard の done/サイズ/件数**を検査し、
  各 shard から標本(既定 200/shard、`--full` で全件)を取り cshogi 読み戻し + KP 可逆性
  (局面→KP→局面)+ 特徴数を検査。異常頻度(manifest)も集計。
- Python `tools/verify_hcpe.py`: shard 単位の cshogi 照合(eval/bestMove16/hcp_hex/sfen)。
- Rust `gikou-recheck`: 生成済み標本を技巧2で再解析し、保存 eval の**手番側視点(符号)**を再確認。

## 検証結果(基盤)

### レジューム(SIGKILL→再開)

- 2,000 局面 / 100 shard / 4 worker / seed 777 の無中断リファレンスを生成(concat md5
  `ae265c96431373edb3dc91e45cc68ec0`)。
- 同条件を別ディレクトリで起動し 3 秒で `SIGKILL`(`timeout -s KILL 3s`)。中断時点: **24 shard 完了 /
  24 ファイル / 半端 .tmp 0 / done 印が付いた未完成 shard 0**(write-ahead が守られている)。
- 同コマンドで再開 → 「24 already done, 76 pending」→ 100 shard / 2,000 局面で完走。
- **再開後の concat md5 = リファレンスと完全一致**(`ae265c...`)。レジューム + 原子性 + 決定性を同時に確認。

### 並列の再現性

- 同条件を workers=1 / workers=4(2回)で生成 → 全 shard が**バイト一致**(concat md5 一致)。
  クラッシュを伴わない shard は worker 数・完成順に依らず同一。

### タイムアウト検知

- `--timeout-ms 1`(depth 8 は数 ms 必要なので必ず超過)で実行 → 各 shard で
  「timeout → fresh Gikou(retry 1/2, 2/2)→ FAILED」を記録、`status=incomplete`、終了コード 2。
  ハングが無言の停止にならないことを確認。

### 技巧2の異常終了(発見)

- depth 8 で技巧2が**特定局面で異常終了(stdout EOF)**することを発見。
  例: `l1R+B3n1/6+P1l/ppn3p2/2kg1Ps1p/b3p4/3S5/PP+p1GKS1P/2PP3P1/L6+rL w 2GS3P2np 146`。
- この局面は**冷間(usinewgame 直後の新規プロセス)では `score mate -2` を返し落ちない**。
  暖機状態(同一プロセスで長い対局を続けた後)で落ちる = 暖機状態依存のクラッシュ。
- 当初は「クラッシュ→ shard 丸ごとやり直し」だったが、クラッシュ局面は同一プロセス履歴で再現するため
  毎回同じ所で落ち shard が永久に失敗した。→ **該当 game を破棄し新規プロセスで継続**する復旧に変更。
- 影響: クラッシュを伴った shard は 1 万局面を**必ず満たす(有効データ)**が、技巧2のクラッシュ自体が
  完全には決定的でない(暖機依存)ため、**バイト単位の再現性は保証されない**。各局面は本物の
  depth 8 eval であり、データの正しさには影響しない。再現性が要る用途では crash=0 の shard のみ対象。

## 100万局面 本走

### 実行条件

```
gen-teacher /tmp/binbo-task007/gikou-run/release /tmp/binbo-task016/run1m 1000000 \
  --shard-size 10000 --workers 8 --depth 8 --random-plies 6 --max-plies 256 \
  --seed 20260621 --timeout-ms 30000 --max-retries 3
```

- 環境: 14700KF / WSL2 / 28 論理コア / 15 GiB RAM。技巧2 1 プロセス当たり RSS ≈ 600 MB
  (params.bin 259 MB をロード)。8 worker でも RAM 余裕は約 9 GB。
- 保存先(リポジトリ外): `/tmp/binbo-task016/run1m/`(`manifest.json` + `shard_0000..0099.hcpe` +
  `shard_*.expected.tsv`)。hcpe 本体 38 MB(= 1,000,000 × 38)、expected.tsv 込みで約 199 MB。

### 結果(`run1m.out`)

- total_positions = **1,000,000**(target 達成)、shards_done = **100 / 100**、shards_failed = **0**。
- total_games = 7,163。
- 壁時計 = **926.6 s = 15.44 分**、avg = **0.9266 ms/局面**。
- 単一プロセス見積もり(Task 015 の 4.07 ms/局面)= 1.131 時間(67.9 分)。
  → **並列効率 4.39×**(8 worker)。8 倍に届かないのは shard ごとの新規プロセス起動 overhead と、
  8 並列時の CPU/メモリ帯域競合(14700KF は P+E コア混在)による 1 解析あたりの実時間増のため。
- eval 範囲 = [-31,795, +31,795](終盤の極端 cp を含む)。

### 異常イベント頻度

| 種別 | 件数 | 割合 |
|---|---|---|
| mate 打ち切り (mate_cutoffs) | 6,625 | 0.66% / 局面、対 game 92% |
| 最大手数打ち切り (max_ply_cutoffs) | 355 | game の 5.0% |
| 技巧2 クラッシュ復旧 (crash_cutoffs) | 83 | 8.3e-5 / 局面 |
| エンジン再起動 (engine_restarts) | 83 | = crash 件数(全て復旧) |
| タイムアウト (timeouts) | 0 | — |
| resign 打ち切り (resign_cutoffs) | 0 | — |

- mate 打ち切りが多いのは、ランダム出だし由来の対局が depth 8 で早期に詰みに至るため(終盤の正確な
  ラベルは将来 df-pn。今は mate 局面を cp 教師に混ぜず対局打ち切り)。
- クラッシュ 83 件は全て game 破棄 + 新規プロセスで復旧し、**失敗 shard 0**。

### 検証

- `trainer/verify_shards.py /tmp/binbo-task016/run1m --sample-per-shard 300`:
  - 構造: shards_done=100/100, size_errors=0, total_positions=1,000,000(target 一致)。
  - KP 可逆性(標本 30,000 局面): reversibility_mismatches=**0**, feature_count_mismatches=**0**。
  - 多様性(標本): 手番 先手 15,133 / 後手 14,867、成駒含む 16,710、持ち駒あり 26,135。
  - `verify_shards=ok`。
- `tools/verify_hcpe.py`(shard_0050 全 10,000 件): `cshogi_readback=ok`(eval/bestMove16/hcp_hex/sfen 一致)。
- `gikou-recheck`(手番側視点の再確認、shard_0007/0033/0077 × 各 120 標本):
  決定的局面(|eval|≥500cp)の符号一致 = **60/60, 57/57, 53/53 = 100%** → `ok_viewpoint`。
  非決定的(0 近傍)局面のみ稀に符号が揺れるが、これは暖機(保存時)と冷間(再解析)の探索差による
  ノイズで、視点誤りではない(視点誤りなら大きな eval も片手番の約半数も反転するはず)。

### 見積もりとの比較(外挿の裏取り)

- Task 015 の単一プロセス外挿: 100万 ≈ 1.13 時間。今回 8 worker 実測 15.44 分 = 0.257 時間。
- 1 局面あたり: 単一 4.07 ms → 8 並列実効 0.927 ms(4.39×)。並列の天井は起動 overhead + 帯域競合。
- 500万/1000万へは shard を増やすだけで同基盤で対応可(レジューム前提)。実効 4.39× なら 500万 ≈ 77 分、
  1000万 ≈ 2.6 時間の見込み(depth 8、同並列度)。

## やり残し / 申し送り

- 技巧2の暖機依存クラッシュは GPL の外部エンジン側の問題。binbo 側は game 破棄+再起動で吸収済み。
  将来 depth を上げる際はクラッシュ頻度を再計測する。
- 完全なバイト再現性が要るなら、shard ごとに「クラッシュ 0」を要求し、クラッシュした shard を
  別 seed で作り直す運用も可能(今回はスコープ外)。
- 生成データ本体(shard 群 hcpe + expected.tsv)はコミットしない(大きい・再生成可能)。
