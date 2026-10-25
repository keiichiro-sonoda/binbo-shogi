# Task 047: 本物の正準 HalfKP(W₁ 共有・単一帯)

## 結論

**Task 046 の「正準」は NNUE 原典(那須2018)に照らすと偽物だった**(正準の核心 = W₁ を両視点で共有、
式15〜17・表1。だが Task 046 は view0/view1 を別帯にして両視点が独立パラメータ = 「2帯・非共有版」)。
本タスクで**本物の正準**(両ハーフが**同じ単一帯 `[0,176742)` を共有**、埋め込み **176744行 = naive の約半分**)を
実装し、naive / two-band(偽) / shared(本物)を同一 8M・同一 val で三者比較した。

**本物の正準はアーキテクチャとして明確に優秀**(パラメータ半減・過学習激減・val 度合い最高・forward が速い)
= 原典設計の正しさを実データで実証。**ただし改善幅は小さく(val 70%→70.7%、+1pp)、Lv1 の壁は明確には破れない。**
→ 次段はより richer な **KPP**。推奨 EvalFile は度合い・速度・パラメータで shared 優位だが改善が小さいため
据え置き寄り(本文末)。Python/Rust 数値一致は最優先で完全達成。

## 1. 何が偽物だったか(原典 §2.6〜2.7, 式15〜17, 表1)

正準 HalfKP の核心は **「アフィン変換 W₁ を手番側・非手番側の両視点で共有する」**(式15 が両 c に同じ W₁、
p4「W₁ は共有される」、表1 で W₁ は単一テーブル 81×1548=125388列)。各ハーフは自分の玉1つをアンカーにし、
**同じテーブルを共有**して両視点を作る。

Task 046 の実装は (a)相手ハーフ回転反転・(b)2アキュムレータ連結 は正しかったが、view0(自玉)を `[0,176742)`、
view1(相手玉)を `[176742,353484)` の**別帯**に持ち、両視点が**独立パラメータ = W₁ 非共有**。
→ コスト(2アキュムレータ)だけ払い、利得(共有=パラメータ半減・対称性の帰納バイアス)を持たず、
実質「容量2倍」で効かなかった(Task 046: val 度合い naive と同一、gap 148→175 悪化、forward −17%)。

## 2. 本物の正準(shared, BNN3)の実装

両ハーフとも**単一帯 `[0,VIEW_DIM)=[0,176742)`**(view=0 固定)を引く:
- 自ハーフ: `self_king*2182 + offset(rel, piece, S)`。
- 相手ハーフ: `rotate(opp_king)*2182 + offset(1−rel, piece, rotate(S))`(`rotate(X)=80−X`)。
- **「自玉=K と駒X」と「相手玉=K と駒X(回転反転後)」が同じ行を共有**(= 正準の核心)。
- `CANONICAL_SHARED_TOTAL = VIEW_DIM + 2 = 176744`(side 2行)。**埋め込みは naive 353486 の約半分**。
- 自/相手は index 値が衝突しうる(=共有の証拠)ので、**範囲分割でなく固定列** `[own(39)|opp(38)]=77` で保存・分割。
- モデルは `CanonicalNNUE`(同一・total_features を 176744 に)。構造(回転反転+2アキュムレータ連結)は Task 046 流用。

## 3. 検証(Python/Rust 数値一致 = 最優先・完全達成)

- **Python 共有 KP**: 可逆性 3336/3336、不変量 own39+opp38=77、**312局面で own∩opp の共有行**(= W₁ 共有が本物)。
  初期局面は180度+色対称なので**自ハーフ盤駒38個 = 相手ハーフ38個が完全一致**(共有の最小実証)。
- **Rust 共有 KP == Python: 30000/30000(100%)**(技巧2+Random 2分布、両手番、玉81/81マス)。`verify_kp_rust --canonical-shared`。
- **共有 forward Python==Rust(BNN3)**: probe 1.2e-06、**実 shared-8M モデル 2.6e-06**(tol 1e-3)。
- 共有 mirror perm(長さ176744): involution 成立、二経路 3640/3640(水平ミラーが180度回転と可換、二重回転なし)。
- 回帰: naive BNN1 3.3e-06 + two-band BNN2 3.2e-06 とも一致(無破壊)。`cargo test` 18/18、perft 1〜5 一致、selfplay 完走。
- Rust `Nnue::load` が magic `BNN1`/`BNN2`/`BNN3` を `Mode{Naive,TwoBand,Shared}` に分岐。

## 4. 三者比較(8M, fd8+mirror+cp clamp5000 MSE, GPU 20ep, 同一 val=run1m 末尾5万)

| 指標 | naive (BNN1) | two-band (BNN2, 偽) | **shared (BNN3, 本物)** |
|---|---:|---:|---:|
| 埋め込み行数 | 353486 | 353486 | **176744(約半分)** |
| best val_loss | 594891 @ep3 | 591669 @ep4 | **583170 @ep4(最低, −2.0%)** |
| epoch1 過学習 gap | +262855 | +115886 | **+15815(1/17)** |
| pred_dispersion val 度合い(/3915) | 2729 (69.7%) | 2735 (69.9%) | **2769 (70.7%)** |
| pred_dispersion gap(train−val) | 148 | 175 | 188 |
| val_corr(peak) | ~0.90 | 0.90 | **0.9032** |
| forward 速度(nnue-bench d5) | 1.14M nps | 0.66M(−17%) | **1.26M nps(+11%)** |
| 到達深さ(Lv1 実戦) | 5.23 | 5.05 | **5.39** |

- **本物の正準は初めて val 度合いを 70% 超に押し上げた**(naive/two-band は 2729〜2735 で頭打ち)。
- しかも**埋め込み半減 + forward が速い**(two-band は同じ2アキュムレータでも全表335万行で −17% 遅。
  shared は半分の表でキャッシュ局所性が上がり +11% 速)。過学習も epoch1 gap が naive の 1/17 に激減。
- これらは原典の主張(共有 W₁ で「同等情報をより少ない計算で表現」)を実データで裏づける。

## 5. 棋力(本命, 技巧2 Lv1, 1秒, idle 同条件)

| | 8局 | 16局 | avg_plies(16) | 深さ |
|---|---|---|---:|---:|
| naive-8M | 1-7-0 | **1-15-0**(6.2%) | 97.2 | 5.20 |
| shared-8M(本物正準) | 0-8-0 | **0-16-0**(0%) | 98.9 | 5.37 |

**16局でも区別不可**(naive 1勝 / shared 0勝、8+16 合計でも naive 2/24・shared 0/24 で有意差なし)。
avg_plies はほぼ同一(98.9 vs 97.2)、深さは shared が上(5.37>5.20=半分の表で速い)。
**val 度合いの +1pp は Lv1 棋力には届かない**(改善が小さすぎる)。

## 6. 評価品質(nnue-diag, 技巧2 depth8 基準)

- shared B-1 代表局面: **decisive 4/4 符号一致**(mid_tactical の best=`7e6d` は技巧2 depth8 の最善と一致)。
- 対局全体の符号一致 39/42(93%)。ただし生徒の着手が分岐し局面集合が異なるため直接比較は不可(参考)。

## 7. 結論と次段

- **本物の正準 HalfKP は naive・two-band より明確に良いアーキテクチャ**: パラメータ半減・過学習激減
  (epoch1 gap 1/17)・best val_loss 最低・val_corr 最高・val 度合い最高(初の70%超)・forward 最速。
  原典設計(共有 W₁ = 対称性の帰納バイアス + パラメータ効率)の正しさを実証した。
- **だが改善幅は小さい**(val 度合い +1pp、70.0→70.7%)。Lv1 の壁を明確に破るには不足。
  残り ~30% の主因は naive↔正準 HalfKP の違いでなお説明されず、**律速は KP 表現そのもの**
  (King-Piece のみ=駒ペア関係を持たない)に絞られた。
- **次段: KPP(King-Piece-Piece)** = 表現を質的に richer に(駒ペア)。正準実装の Python/Rust 一致基盤
  (`canonical_shared` 切替・固定列レイアウト・magic 分岐・shared mirror)と「埋め込み半減で過学習が減る」
  知見を再利用できる。
- Task 046 の「正準は効かない」という結論は**偽の正準での結論だったので無効**。本物では小さいが正方向の
  改善があり、HalfKP の枠内では本物正準が最良 = アーキテクチャの教訓として確定。

### 推奨 EvalFile

**据え置き = `model_cp_fd8_mirror_8m`(naive BNN1)**。理由: 本命の棋力が改善しない(16局でも区別不可、
合計 naive 2/24・shared 0/24)ため、差し替える積極的理由がない。ただし `model_shared_fd8_mirror_8m`(BNN3)は
**度合い・パラメータ・速度・過学習で優り、KPP の土台**として保持(非コミット)。KPP が効けばそちらを基盤にする。

## 数値一致・回帰(green)

- `cargo test` **18/18**、perft 1〜5 完全一致、selfplay 完走、正準各エンジン合法手。
- 数値一致: shared KP 30000/30000、shared forward 2.6e-06、naive/two-band 回帰一致。スケール `forward*600=cp` 不変。

## 成果物

- コード(切替式・naive 既定不変): `trainer/{kp,nnue,train_nnue,mirror,prepare_data,export_nnue,verify_kp_rust,verify_nnue_rust,pred_dispersion}.py`、
  `src/eval/{kp,nnue}.rs`、`src/main.rs`(`kp-from-sfen ... canonical-shared`)、`docs/kp_spec.md` §7 全面補正(原典準拠・偽訂正)。
- 非コミット(永続データ): `~/binbo-shogi-data/caches/shared_8m`、`~/binbo-shogi-data/models/model_shared_fd8_mirror_8m`。
