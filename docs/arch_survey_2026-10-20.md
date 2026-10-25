# Task 148: NNUE 構造変更の実現性と費用 (2026-10-20)

**第一候補は HalfKP 512x2-8-96、次点は HalfKA_hm1 256x2-32-32。1024 幅と SFNN は後回し。** 前者は特徴キャッシュ・Factorizer・量子化演算を保った容量試験、後者は速度低下を小さく抑えた入力表現の試験になる。ただし改善 Elo は予測できず、64M で広いネットが勝つ根拠はまだ弱い。まず同じ後段の HalfKP 256x2-8-96 を対照に入れる。

調査のみ。ネット接続・GPU・学習・ビルド・エンジン起動なし。標準ライブラリによる小さな算術と eval ヘッダ読取のみ実施。既存ファイルは変更しない。以下の工数・時間・NPS 外挿は実測と区別した**計画値**。

## 1. 対象と参照方法

- `Y/` = `$DATA_ROOT/engines/yaneuraou/source/`。ローカル HEAD は `820940ee31762ad598a9156a615b4b896ab8b321`。`Y/eval/nnue/` を以下 `N/` と略す。外部ソース参照はこの版の `ファイル:行`。
- binbo の参照は本 worktree の `ファイル:行`。公開実験の数字は既存調査の再引用で、原記事を今回再確認したものではない。
- 本線は自前探索のラベルで再学習。公開水匠5は比較基準と軽い仕上げ、Hao 教師は使わないという決定を維持する ([教師方針](teacher_direction_2026-10-20.md):14–24)。大量公開データが存在することと、本線で大量使用することは別の判断。

## 2. この YaneuraOu が選択できるもの

表の `D×2-H1-H2-1` は片視点 FT 幅 D、手番/非手番を連結、隠れ2層、出力1。標準構造は ClippedReLU と整数 affine。各 `.h` の RawFeatures と層定義を直接確認した。

| 既製 architecture header (`N/architectures/`) | 特徴量 / 入力次元 | 層 | `YANEURAOU_EDITION` (`YANEURAOU_ENGINE_` 以下) |
|---|---|---|---|
| `halfkp_256x2-32-32.h:19–33` | HalfKP(Friend), 125,388 | 256×2-32-32-1 | `NNUE` (既定) |
| `kp_256x2-32-32.h:18–29` | **K + P** の加算的特徴, 162+1,548=1,710 | 同上 | `NNUE_KP256` |
| `halfkpe9_256x2-32-32.h:20–32` | HalfKP × 自他の利き数各0/1/2以上, 1,128,492 | 同上 | `NNUE_HALFKPE9` |
| `halfkpvm_256x2-32-32.h:20–32` | HalfKP_vm, 玉45マスに左右反転共有, 69,660 | 同上 | `NNUE_HALFKP_VM_256X2_32_32` |
| `halfkp_512x2-16-32.h:19–33` | HalfKP, 125,388 | 512×2-16-32-1 | `NNUE_HALFKP_512X2_16_32` |
| `halfkp_1024x2-8-32.h:19–33` | HalfKP, 125,388 | 1024×2-8-32-1 | `NNUE_HALFKP_1024X2_8_32` |
| `halfkp_1024x2-8-64.h:19–33` | HalfKP, 125,388 | 1024×2-8-64-1 | `NNUE_HALFKP_1024X2_8_64` |
| `sfnn-1536.h:21–47` | HalfKA_hm2, 73,305 | FT1536×2 → 積で1536 → 16 → 二乗/通常各15を連結30 → 32 → 1、16番目から短絡加算。後段9組 | `SFNN1536` |

**HalfKP の KP は玉×駒の組、`KP256` の K+P は別物**。標準時の `fe_end=1548`, `f_king=1548`, `e_king=1629`, `fe_end2=1710` による次元。金と成歩/成香/成桂/成銀は同じ駒スロット (`Y/evaluate.h:204–226`, `trainer/yane_index_map.py:31–49`)。

既製以外はヘッダ自動生成で選べる (`Y/Makefile:112–122,418–455,642–654`, `N/architectures/nnue_arch_gen.py:130–226,298–320`)。

| generator の特徴名 | 内容 / 次元 | 選択例 (`YANEURAOU_ENGINE_` 以下) |
|---|---|---|
| `halfkp`, `kp`, `halfkpe9`, `halfkpvm` | 上表と同じ。D/H1/H2 は指定可能 | `NNUE_halfkp_512x2_8_96`, `NNUE_halfkp_1024x2_8_96` |
| `halfka1` | 両玉も駒として区別して含む。81×1710=138,510 | `NNUE_halfka1_256x2_32_32` |
| `halfka2` | 両玉を**同じ玉 plane に合流**。81×1629=131,949 | `NNUE_halfka2_256x2_32_32` |
| `halfkahm1` | halfka1 + 視点の玉が6〜9筋なら左右反転。45×1710=76,950 | `NNUE_halfkahm1_256x2_32_32` |
| `halfkahm2` | halfka2 + 同上。45×1629=73,305 | `NNUE_halfkahm2_512x2_8_96` |
| `ka2` | K + A2 (玉 plane 合流済みの全駒)。162+1629=1,791 | `NNUE_ka2_256x2_32_32` |
| `SFNN_<feature>_D_H1_H2_k3k3` | 上記特徴名を使用可能、積FT・二乗活性・短絡・後段9組 | `SFNN_halfkahm2_1536_15_32_k3k3` |

ビルド指定例 (将来用、**今回は未実行**): 書込可能な隔離コピーの `source/` で `make -j1 normal COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3 YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_halfkp_512x2_8_96`。`normal`/`tournament` が生成器を呼び、`NNUE_ARCHITECTURE_HEADER` を定義する。既製では Makefile が対応マクロを定義し `N/nnue_architecture.h:14–63` が include を選ぶ。CPU対象は `Y/Makefile:550–569`。ネットは実行時の USI オプションでは切り替わらず、**構造ごとの実行ファイルが必要**。任意寸法には SIMD の整列制約があるので「任意の整数ならビルド保証」ではない (`N/nnue_architecture.h:70`, `N/nnue_feature_transformer.h:236,299`)。

**LayerStack は存在するが、進行度 buckets はない。** SFNN のみ、各玉をそれぞれ自身の色から見た段で3区分し、その直積9組から後段1組を選ぶ。9組すべてを推論するのではない。通常 NNUE は1組固定 (`N/evaluate_nnue.cpp:284–316`, `N/evaluate_nnue.h:24–38`)。generator は `k3k3` / `king3_by_king3` のみ受理し、旧 `ls9` / `SFNNWOP` は拒否する (`nnue_arch_gen.py:34–41,68–85`)。駒数/手数による phase 分岐や通常 HalfKP の多分岐は C++ の追加実装が必要。`half_relative_kp.*` はあるが generator の選択肢にはなく、Mobility の既製選択肢も見つからない。

## 3. 強い公開 eval の採用例: 確認範囲

| eval | ローカルで裏付けられる構造 | 留保 / 根拠 |
|---|---|---|
| tanuki- Háo / Hao (手元の物差し) | **HalfKP 256×2-32-32-1** | `engines/yaneuraou/eval/nn.bin` の先頭ヘッダを今回読取。`Features=HalfKP(Friend)[125388->256x2]`, affine 512→32→32→1, khash `0x3e5aa6ee`。出自対応は [evaluation_protocol.md](evaluation_protocol.md):36–38、[教師方針](teacher_direction_2026-10-20.md):14 |
| tanuki- の後年の実験 | **HalfKP 512×2-8-96** が比較基準、1024幅等も実験 | [10/07調査](research_survey_2026-10-07.md):85,121、[10/12調査](research_survey_2026-10-12.md):75,143–147。HalfKA_hm の1024幅実験もあるが、hm1/2の別と現在の最終配布構造はこの資料では不明 |
| 水匠5 (Suisho5) | **HalfKP 256×2-32-32** | [物差し調査](yardstick_survey_2026-10-05.md):18。水匠5 depth9 データから教師構造以上のものは推定しない |
| 水匠10/11、DL水匠 | 対象版の寸法は確定不能 | 既存調査には FV_SCALE や教師としての情報はあるが、今回確認できる構造の一次資料・対象重みがない。水匠5から類推しない |
| つようら王11 mini / micro | **確定不能** | ローカルの対象配布物・モデルヘッダ・対応する構造説明が見つからない。名称から幅や SFNN 採用を推測しない |
| 参考: AobaNNUE v1.1 | HalfKP 768×2-16-64 | [物差し調査](yardstick_survey_2026-10-05.md):41–42。大きい幅の成功例だが学習データ量・教師が異なる |

「エンジンが読める」「公開実験で勝った」「現在の強豪が採用中」は区別する。最新配布版の断定は今回のローカル限定調査ではできない。

## 4. binbo の変更箇所と工数

現状は `QATSharedNNUE` の FT 幅は可変だが **隠れ2層は同じ hidden_dim** (`trainer/nnue.py:251–288`)。BNNQ も hidden を1個だけ保存し、Rust はそれに従って読む。幅だけでなく 8→96 の非対称後段を扱う基盤が必要。

### A. HalfKP 256/512/1024×2-8-96 (512×2-16-32 等にも共通)

| 範囲 | 具体的変更 |
|---|---|
| 学習 | `trainer/nnue.py::QATSharedNNUE.__init__` に H1/H2 を分離、`l0/l1/l2` を `(2D,H1),(H1,H2),(H2,1)` にする。`forward_hidden_from_features` の演算はそのまま。`train_nnue.py::main` のCLI・モデル生成・保存config・resume互換を更新 (`:333–334,723–734,957–975`)。`fold_ft_factor`, `quantize_qat_state`, `_touched_features` は幅非依存のため再利用可能 (`nnue.py:64–101,345–377`) |
| 特徴 | `trainer/kp.py::canonical_shared_halves/row`、tie map、77列 cache は**変更不要**。`compact_cache.py` も幅非依存で再利用。左右反転増強・Factorizerも維持 |
| BNNQ | `export_nnue.py::export_qat` の header を版分けし H1/H2 と feature/arch 識別を保存。旧 BNNQ を黙って再解釈しない (`:110–141`)。`binbo_to_yane.py::parse_bnnq` と `yane_to_binbo.py::convert_quant` も対応 |
| nn.bin | `yane_nnue.py::parse` の固定 `FT_DIM/KIN/AFFINE` を構造記述にする (`:41–78`)。`binbo_to_yane.py::bnnq_to_yane` の固定256検査と reshape を更新 (`:101–108,131–145`)。`yane_to_binbo.py::_attach_int_views/yane_forward_int` はFT幅・パディングを対応 (`:46–78`)。**H1=8/16の入力を32へゼロ埋め**して後段へ渡す (現行 Python int forward は埋めずに行列積するためそのままでは不一致/shape error) |
| template/hash | 同じ特徴・D/H1/H2の template を新規用意。対象実行ファイルの構造文字列とhashを一次情報として生成/取得し、全重みを自作重みで置換。旧256 templateの流用不可。plain層の読込順・入力32倍パディングは `N/layers/affine_transform.h:180–182,233–239`、同 `_sparse_input.h:185–191` |
| Rust / gate | `src/eval/nnue.rs::load_quant/quant_forward` に H1/H2 分離 (`:214–243,374–427`)。特徴取得は従来のまま (`:317–320`)。`scripts/binbo_to_yane.sh` に構造メタデータと対象 `$YANE` を渡す。3者一致を維持 (後述) |

**見積: 共通基盤2〜4人日、以降の幅1点追加は半日程度** (学習・対局待ち別、bit-exact確認込みの目安)。幅だけを512へ上げて32-32のままなら BNNQ/Rust の既存動的幅を使えるが、writerの256固定は残る。公開例の8-96とは別実験になる。

### B. HalfKA1/2・HalfKA_hm1/2

A の writer汎用化・構造識別に加え、以下が必要。256-32-32なら後段の非対称化自体は不要だが、**BNNQ QuantがHalfKP固定**なので特徴識別は必要。

- `trainer/kp.py` に YaneuraOu と一致する全駒40個/視点の特徴経路、hmなら左右反転を追加。`trainer/yane_index_map.py::build_remap/yane_active`、`binbo_to_yane.py::build_collapse_map` を全駒・金合流・玉plane合流に対応。hmは視点ごとに反転する (`N/features/half_ka_hm1.cpp:15–29`, `half_ka_hm2.cpp:15–29,48–57`)。
- **既存 `canonical_shared_ka_*` は出発点にすぎない** (`kp.py:336–427`)。既存 `kav2` はアンカー玉を除去する方式 (`:433–470`) で、YaneuraOu の `HalfKA2` (両玉とも残してplane合流) と異なる。集合化による重複排除ではなく、ソースと同じ寄与の個数も照合する。
- `QATSharedNNUE.ft_weight/_touched_features` と `fold_ft_factor` の `kp.VIEW_DIM/FEATURES_PER_KING` 固定を特徴仕様へ移す。Factorizerは新特徴の駒番号とtie後の代表行に整合させる (`nnue.py:64–77,300–307,355–358`)。
- `prepare_data.py` と `train_nnue.py::run_batches/main` の半分の長さ・総次元を対応。`--qat` は現在 `CANONICAL_SHARED_TOTAL` を強制し、`--compact-cache` は HalfKP 77列しか認めない (`train_nnue.py:470–472,723–734`)。新cacheと `compact_cache.py` の40+40(+side)版を用意しないと Task135 の速度を失う。
- export・writer・Python int oracleの特徴選択/KIN、`src/eval/kp.rs` の対応特徴と `src/eval/nnue.rs::forward` の Quant 分岐を追加。現行 `Mode::SharedKA` は浮動推論用で、Quant経路はHalfKP (`nnue.rs:317–320,342–348`)。hmの直接スロット番号で学習する案ならtie scatterは簡単になるが、旧キャッシュの変換/再構築と独立した番号照合が要る。

**見積: A の共通基盤に追加3〜6人日** (新cacheの高速経路・Rust・数値一致込み)。hm1を先にする理由は、両玉の区別を保持してhm2のplane合流という追加変数を避けるため。ただし Task132 の独立視点反転の弱点を共有し、次点に留まる。

### C. SFNN / LayerStack / その他

- SFNNには別モデルクラスを用意。FTを2分して積、二乗/通常活性を連結、短絡出力、9個の後段と玉段による選択をQATにも実装。特徴抽出はB、bucketは元局面の両玉から取得してbatchに渡す (`N/architectures/sfnn-1536.h:91–105`, `N/evaluate_nnue.cpp:284–297`)。
- BNNQの形式/量子化とRust整数推論を新設。`N/nnue_feature_transformer.h:179–191,242–285,482–492` は **LEB128 FT読込 → 重み/biasを2倍 → 同一視点内の前半後半の積**。従来のclamp連結と違う。`N/layers/sqr_clipped_relu.h:63–96` の丸め/飽和とSIMD経路まで一致させる。
- `yane_nnue.py` は圧縮FTと9組を解読/書出し、writerは9組すべて埋める。templateのhash転用だけでは実現しない。現ローカルFTの `WriteParameters` はplain配列書出し (`nnue_feature_transformer.h:198–200`) で読込のLEB128と対称でもないため、エンジンのsaveをそのままtemplate生成法と仮定しない。hashもSFNNでは固定値部分がある (`:161–166`, `sfnn-1536.h:53–55`)。
- **見積: Bを含め2〜4人週以上**。9分岐は後段の保存量/学習データ分割を増やすが推論MACは9倍にならない。通常HalfKPにphase bucketsだけを足す案も既存スイッチではなくC++/形式/全oracle改修になるため、別タスク。
- HalfKPE9は利き数cache・差分更新oracle・Rust特徴から変更、FT表9倍、探索速度の既知の損失あり。KP(K+P)/KA2(K+A2)は組関係を減らす圧縮実験で、容量増の本命ではない。いずれも今回の推奨対象外。

## 5. 速度・メモリ・データの見込み

**ソースから分かる費用**: accumulator本体は `2×D×int16=4D bytes` (refresh trigger 1個の場合、管理領域/align別; `N/nnue_accumulator.h:18–23`)。通常の後段は `MAC=2D·H1+H1·H2+H2`。1 MACを2 FLOPと数えれば表の2倍だが、実体は整数SIMD・疎入力。ゼロ活性率/パディング/キャッシュ/玉移動時の全再計算で時間は変わり、MAC比をNPS比にできない。

| 候補 | accumulator | FT int16表 (MiB) | 後段の論理MAC/評価 | 通常の差分/全再計算の負荷 |
|---|---:|---:|---:|---|
| 現行HalfKP 256-32-32 | 1 KiB | 61.22 | 17,440 | 基準 |
| HalfKP 256-8-96 | 1 KiB | 61.22 | 4,960 | FT同じ、後段小。速度向上の余地 |
| HalfKP 512-8-96 | 2 KiB | 122.45 | 9,056 | FT更新/読出し約2倍、後段約0.52倍 |
| HalfKP 512-16-32 | 2 KiB | 122.45 | 16,928 | FT約2倍、後段ほぼ同じ |
| HalfKP 1024-8-96 | 4 KiB | 244.90 | 17,248 | FT約4倍、後段ほぼ同じ |
| HalfKP 1024-8-32 / -8-64 | 4 KiB | 244.90 | 16,672 / 16,960 | FT約4倍。後段削減は小さい |
| HalfKA_hm1 / hm2 256-32-32 | 1 KiB | 37.57 / 35.79 | 17,440 | 通常の駒移動は同幅、refreshで38→40駒。敵玉移動の寄与更新が追加。表は小さくなる |
| HalfKA_hm2 512-8-96 | 2 KiB | 71.59 | 9,056 | 広いFTと玉特徴更新。hm1なら75.15 MiB |
| SFNN HalfKA_hm2 1536 | 6 KiB | 214.76 (展開後) | 25,568 + FT積/二乗活性 | FT幅6倍、40駒、9組中1組のみ実行 |
| HalfKPE9 256-32-32 | 1 KiB | 551.02 | 17,440 | 利き変更に伴い動いていない駒の特徴も更新 |

表は学習用のbinbo冗長行数ではなく**エンジン配置時の表**。H1=8の次層入力は32までpadするので、512-8-96の保存affine重みは11,360個 (論理MAC9,056とは別)。この点でも後段が0.52倍なら速度も同率とは言えない。

**NPSの目安**: 現行に対する厳密な比は未計測。感度計算として、現行時間のFT部分を20〜50%、後段を20%、残りを探索等と仮置きすると、512-8-96は約**0.71〜0.91倍**、1024-8-96は**0.40〜0.63倍**になる (`時間比 = その他 + FT比×FT時間割合 + MAC比×後段割合`)。これは予測区間でも実測でもなく、負荷配分を変えると変わる例示。hm256は通常更新の幅が同じなので現行付近が出発予想、SFNNは6倍FTと非線形演算があり、分岐だけを見て「ほぼ無料」とは言えない。

既存調査の公開実測 (すべて別ネット/データ/CPUなので binbo への保証ではない):

- tanuki-: **256-8-96は512-8-96より25%速いが−27.2 Elo**。1024幅は512比 **NPS約2/3、+10.1 Elo** (80億局面)、1536は−24.5。1024-8-32は旧256-32-32比−74.4、過学習の報告 ([10/12](research_survey_2026-10-12.md):143–147、[10/07](research_survey_2026-10-07.md):121)。25%速いの比較対象は今回の現行32-32とは違う。
- Stockfish 256-32-32→512-16-32: 15〜20%遅い、+51 (固定20k nodes、160億局面)。将棋の時間制の利得と混同しない ([10/07](research_survey_2026-10-07.md):123)。
- HalfKA_hmは1024幅で+21.8±10.1、hm1/2や256幅での効果は不明。HalfKPE9のNPS約2/3、Mobilityは−47。LayerStack/SFNN等で時間制−28〜−40の例もある ([10/12](research_survey_2026-10-12.md):144–147,191–192)。
- Task034/042の幅拡大失敗は小データ・旧Rust float推論、Task052/053はfd8・旧探索の時代。現在のQAT256への反証にはならない ([task034](../reports/task034_capacity_scaling.md):5–15、[task042](../reports/task042_capacity_revisit_8m.md):5–12、[task052](../reports/task052_halfka.md):49–65)。一方 **Task132のvm −33.7** は現行に近く、hmへの警戒材料 ([STATUS.md](../STATUS.md) Task132、[task132](../reports/task132_halfkp_vm.md))。

**データ量と学習費**:

- 64M・12周の現行は **1.8 GPU時間**、平均543秒/周 ([task135](../reports/task135_fast_trainer.md):48–55)。同じ高速cache経路でFT幅に比例すると仮定した予算は、512で3.6h、1024で7.2h。後段/帯域/メモリで変わり、hm新cacheとSFNNにはこの速度を保証できない。
- 同じ12周を局面数にも比例させた256幅/512幅の予算は、128Mで3.6/7.2h、256Mで7.2/14.4h、1Bで28/56h、8Bで225/450h。転送経路の整備・前処理・評価は別で、全量GPU常駐の速さを維持できなければ増える。周回数と独立局面数を混同せず、旧データの周回だけで大データ実験を代用しない。
- 既存調査ではAobaNNUEは150M/300Mでvalが反転し、**1Bで反転しなくなった**。tanuki-の構造比較は8B級 ([10/07](research_survey_2026-10-07.md):121,129)。512は64Mで予備試験、可能なら独立局面128〜256M、その先に1B、1024/SFNNは1B級の準備後、という順が妥当。これは必要量の実証済み閾値ではない。9 bucketsは稀な玉配置のデータ不足も点検する。
- 128Mの156 B/行cacheだけで18.6 GiBとなり16GB GPUに入らない。**小分け転送/シャード学習が先** ([data_scaling_128m_plan.md](data_scaling_128m_plan.md))。現行の学習FTは176,745行を保持し、weight/grad/Adam2状態のfloat32だけで256/512/1024幅=約0.67/1.35/2.70 GiB、64M cache約9.3 GiBに加えて中間領域が要る。広いほどVRAMの余裕が減るので、1.8hの単純倍算は予算用に留める。
- ユーザー提示の再ラベル**合計500〜1000局面/s**なら64Mは17.8〜35.6h、128Mは35.6〜71.1h、1Bは278〜556h、8Bは2,222〜4,444h (28プロセス継続なら64Mは約498〜996 CPU-thread時間)。既存64Mの付け直しは局面数を増やさない。独立局面追加には生成/除重複の別費用もある。
- Task146の古い低速見積は使わない。末尾追記でpartごと初期化へ修正済み、8プロセス519局面/sの短時間実測がある ([task146](../reports/task146_self_relabel_prep.md):124–130)。今回の500〜1000/sはユーザー提供の運転情報として扱い、新たな負荷測定はしない。
- 公開水匠5約8Bのprefixは入手/再ラベル費を減らせるが、64M以降の利用は教師方針の変更を伴う。Task145は現在の64M比較用で、既存スクリプトの16 worker前処理も**本調査では実行しない** (`experiments/alphazero/t145_suisho5_student.py:9–14,49–78,82–104`)。

## 6. 推奨する実験順と0cp維持

**この表は将来の提案。現在のラベルジョブ中には実行しない。** 工数込みの利得/時間は未知なので、追加の教師生成を避ける順で進める。

| 順 | 実験 | 使うデータ / GPU予算 (12周、推定) | 次へ進む条件 |
|---|---|---|---|
| 先行対照 | 現行256-32-32を再学習 | 完成したTask146の同じ64Mラベル、1.8h | 自前教師更新の効果を分離。既存のTask145はその目的のまま保持 |
| **1** | **HalfKP 512-8-96** + 同後段256-8-96 | 同じ64M・split・ラベル・増強・Factorizer・seed。256対照約1.8hを仮置き、512約3.6h、現行対照と計約7.2h | valだけで選ばず時間制で比較。256対照で後段変更、512で幅変更を分離。良ければ新seed学習で再現し、独立128〜256Mを双方に同量追加 |
| **2** | **HalfKA_hm1 256-32-32** | 同じ64M局面から新特徴cache、同ラベル。高速経路移植後は現行付近の学習時間を期待するが未測定 | Python/C++/Rust一致を先に確保。vmの悪化があるため小規模で打切り可、好結果時だけ大データへ |
| 保留 | 1024-8-96 / hm2 / SFNN | 512の成功・データ規模曲線とVRAM/転送基盤を見てから | 大量ラベル生成や9分岐の実装を先行しない |

### 0cpゲートは緩めない

1. 構造ごとに **export元・nn.bin・template・エンジン・特徴仕様** を記録し、SHAとD/H1/H2/特徴名/量子化尺度を固定。halfkpの `>>6`・clamp[0,127]・出力のゼロ方向整数除算とFV16を維持。Factorizer合算 → tie gather → 量子化の順も維持 (`trainer/export_nnue.py:121–138`)。
2. **Rustの元量子化モデル == 独立Python整数forward == 対象エンジンの `e` を全行0cp**。既存32局面に両色、玉全筋/端、金・成駒、持駒、玉移動/取り/成/打を足す。hmは反転境界と玉plane、SFNNは9 buckets・段境界を覆う。連続手順で差分更新した `e` と同SFENから全再計算も照合する。将来の生成器/小さな読取補助は隔離コピーで作り、稼働中エンジンは触らない。
3. `scripts/binbo_to_yane.sh:49–52` は今はGREEN/REDを**印字するだけで失敗終了しない**。`scripts/selfeval_vs_yane.sh:54–63` のように「2系統全件一致・REDなし・件数一致」を必須終了コードにする。新経路はその検査を明示的に継承する。単にスクリプトがexit 0では合格ではない。
4. C++はhash不一致を警告だけで読み進める (`N/evaluate_nnue.cpp:173–180,201–208`)。**読めたことやhashだけは証明にならない**。templateは正確な構造用を使い、全重み置換後の整数oracleで証明する。

### 時間を合わせた主判定 (構造変更用の追加提案)

既定の物差し/seed/ペアbootstrapは [evaluation_protocol.md](evaluation_protocol.md):5–18,22–28 に従う。ただし固定10万nodesの既存成績だけでは重い構造を過大評価するので、**候補と再学習対照に同じ1手時間**を与えた別の比較を主にする。現行の判定規約ファイルは変更しない。

- まず負荷が空いた時に同じ局面集合・Threads1・Hash/CPU命令/コンパイラを揃えてNPSを測る。NPSは重みによる木の差も含むので、固定局面の評価費用と探索nodes/timeを併記する。
- 例えば候補/対照とも100ms/手、物差しは従来のtanuki- 20k / AobaNNUE 10k nodesのまま。同一条件の物差しへ両者を当てることで、候補間の速度損を含めつつ物差しの較正を維持できる。候補用と対照用の実行ファイルを各々指定できる (`src/main.rs:2309–2336`, `yane-vs-yane-ex --a-movetime 100 --b-nodes ...`)。100msで勝率が飽和する場合は**両者共通の時間を結果を見る前に較正して固定**。
- 時間制の対照側は新たに指す。既存の固定nodes成績を使い回さない。3組×2物差し×800局、離れたseed・先後入替で候補−対照のCIを出す。一次ふるい分けを800局×2本に絞るなら参考判定と事前に明記し、最終判断は3組で行う。新seed確認と新学習seedでも確認。固定nodesの差は容量効果の診断用に併記する。
- 時間制の合格候補のみ、両者同じ長めの時間 (例1秒/手) で再確認。対照との直接時間制対局は参考、物差しを主にする。時間制は負荷で揺れるためラベル28プロセスとの同時測定は避け、両腕を交互に実施する。
- 評価費用は `局数×平均手数×(候補の1手時間+物差しの1手時間)/2` 秒/腕が目安。仮に120手、100msと物差し平均20msなら4,800局で**約9.6単コア時間/腕**、候補+対照約19.2h (初期化等別)。固定nodesの過去の「約15分/1600局・8並列」を時間制へ流用しない。数時間の学習より最終対局費が大きくなりうるので、まず構造2種までに絞る。
