# Task 046: 正準 HalfKP 化 — 入力表現の質で残り30%が縮むか

## 結論

**正準 HalfKP 化は残り30%(val 度合い 70%・Lv1 ~1-7 の壁)を縮めなかった。不採用。**
同一 8M データ・同一レシピで「正準化だけを変数」にした対照実験で、正準版は val_loss/val_corr が
**僅かに改善する一方、val 度合いは同一(70%)・棋力は区別不可(0-7-1 vs 1-7-0)で、forward は ~17% 遅い**。
(a) 相手ハーフの向き + (b) 2アキュムレータ連結は残り30%のレバーではない。→ 次段は **KPP**(表現を richer に)。

実装の最重要要件 **Python/Rust 数値一致は完全達成**(正準 KP 30000/30000、正準 forward max_abs_err 3.0e-06)。
正準版は切替式(naive=既定・不変、採用済み `model_cp_fd8_mirror_8m`(BNN1)と全過去検証は有効)。

## 背景

効く軸(情報追加: mirror 036・データ量 039〜041)と効かない軸(正則化 032/037/038・feature_dim 幅増 042・
教師 depth↑ 043/045=技巧2 クラッシュで塞)が出尽くし、残る本命が**入力表現そのもの**。生徒は技巧2 depth8
評価(正解3915)の 70% しか再現できず、容量(幅)では伸びない=「情報の量」でなく「表現の質」が律速かを問う。

正本 `docs/kp_spec.md` §7 が定義する正準 HalfKP との2差異を実装した(推測せず正本準拠):
- (a) 相手ハーフを相手目線(手番側フレームを180度回転+相対所有者反転)で持つ。
- (b) 自/相手ハーフを別アキュムレータにして連結(入力 2×fd)、第1層が攻守を区別。

## 1. 正準 KP の付番(正本 §3 から導出・§7 に記録)

view 0(自ハーフ)は naive と同一。view 1(相手ハーフ)を `rotate(X)=80−X` で:
```
盤上: 1*VIEW_DIM + rotate(opp_king)*2182 + ((1−rel)*13 + id)*81 + rotate(S)
持駒: 1*VIEW_DIM + rotate(opp_king)*2182 + 2106 + (1−rel)*38 + hand_off + slot
```
- 次元・index 範囲・埋め込み表サイズは不変(`TOTAL_FEATURES=353486`)。view 1 帯を全単射に再配置するだけ。
- 検算(初期局面 ▲7七歩, opp_king=36, S=60, rel=0): 自 `96068`(naive と同一)、正準相手 **`273823`**(naive は `255354`)。

### 検証(Task 013/018 水準)
- Python 正準 KP: **可逆性 16000/16000**、77特徴不変量、ハーフ分割 39/38、検算値一致、naive 不変。
- **Rust 正準 KP == Python: 30000/30000(100%)**(技巧2+Random 2分布、両手番、玉81/81マス、持駒・成駒・端玉)。
  `verify_kp_rust --canonical`。
- 左右ミラー拡張は正準でも **同一 perm** で正しい(水平ミラーが180度回転と可換: `80−mirror(80−q)=mirror(q)`)。
  二経路一致 正準 3640/3640・naive 323/323。**二重回転は起きない**(DoD)。

## 2. 構造(2アキュムレータ)+ 3. Rust 推論 + 数値一致(最優先・完全達成)

- `CanonicalNNUE`: 同一埋め込み表で自(view0+side)・相手(view1)を別 bag → 連結(2fd) → `Linear(2fd,hidden)`。
  sorted 77 の固定列構造 `[0:38]自 [38:76]相手 [76]side` で安価に分割。
- Rust: 重み magic **`BNN1`(naive)/`BNN2`(canonical)** で `Nnue::load` が分岐(1/2アキュムレータ forward)。
- **正準 forward Python==Rust**: probe 5.9e-07、**実 canonical-8M モデル 3.0e-06**(tol 1e-3, 技巧2+Random)。
- naive BNN1 回帰: 既存 8m モデルが変更後も一致 3.1e-06(naive パス不変)。`cargo test` **17/17**(+正準分割テスト)。

## 4. 学習と比較(8M, fd8+mirror+cp clamp5000 MSE, GPU 20ep, **同一 val=run1m 末尾5万**)

正準化だけを変数にするため、同一キャッシュ・同一レシピ・同一 seed で naive 対照も新規学習。
対照(naive-8M)の best val_loss 594891・val 2729(70%)・gap148・val_corr0.894 は**文書化済み
`model_cp_fd8_mirror_8m` を完全再現**=対照の正しさを担保。

| 指標 | naive-8M 対照 | canonical-8M | 差 |
|---|---:|---:|---|
| best val_loss | 594891 @ep3 | **591669 @ep4** | −0.5%(僅差) |
| pred_dispersion val 度合い | 2729 (70%) | **2735 (70%)** | +6cp = 同一 |
| pred_dispersion gap | 148 | 175 | やや増 |
| val_corr | 0.894 | **0.900** | +0.006(僅差) |
| decisive_sign | 0.892 | 0.894 | 同等 |
| forward 速度(nnue-bench d5 中盤) | **789 knps** | 660 knps | **−17%** |
| 到達深さ(Lv1 実戦) | 5.22 | 5.05 | 遅い分浅い |
| 学習速度 | ~15s/ep | ~19s/ep | 2アキュムレータ分 |

**val 度合いは同一(両者70%)**。正準は val_loss/val_corr が僅かに良いが、誤差級かつ速度低下と相殺。

### 棋力(本命, Lv1 8局1秒, idle 背中合わせ同条件)

| | record | score | avg_plies | depth |
|---|---|---:|---:|---:|
| naive-8M 対照 | **1-7-0** | 12.5% | 97.6 | 5.22 |
| canonical-8M | **0-7-1** | 6.2% | 101.9 | 5.05 |

両者とも ~1-7 水準で**統計的に区別不可**(8局、勝ちが引分になっただけ=ノイズ)。avg_plies は僅かに増(101.9>97.6)。
**正準化は Lv1 の壁を動かさない**。

### 評価品質(nnue-diag, 技巧2 depth8 基準)

- B-1 代表局面(固定): 両者とも decisive 4/4 符号一致。canonical は飛車得/飛車損(black_up/down_rook)で
  naive より技巧2 に近い度合い(例 down_rook: canonical −1921 vs naive −1491, 技巧2 −2405)が、mid_tactical search は低い(725<1561)。総じて同等。
- 対局全体の符号一致は naive 35/36 vs canonical 29/43 だが、生徒の着手が分岐し**異なる局面集合**を診断するため直接比較不可(参考値)。

## 5. 結論と次段

- 正準化が残り30%を縮めるか: **縮めない**。度合い同一(70%)、棋力区別不可、val_loss/val_corr は誤差級の改善、
  forward ~17% 遅。(a)向き+(b)2アキュムレータは残り30%のレバーでない。
- 含意: 残り30%は **naive↔canonical HalfKP の違いでは説明されない**。律速はより深く、**KP 表現そのもの**
  (King-Piece のみ=駒ペア関係を持たない)の可能性。
- **次段: KPP(King-Piece-Piece)** = 表現を質的に richer に(駒ペア)。本タスクが「正準でも頭打ちなら KPP」と
  位置づけた通り。正準版の実装(切替式・Python/Rust 一致基盤)は KPP でも再利用可能。
- 推奨 EvalFile は **変更なし** = `model_cp_fd8_mirror_8m`(naive BNN1、同等度合いで ~17% 高速)。

## 回帰(green)

- `cargo test --release --locked --offline` **17/17**、`verify` perft **1〜5 完全一致**、`selfplay` 完走、
  正準エンジン合法手(errors=0)。
- Python/Rust 数値一致(最優先): 正準 KP 30000/30000、正準 forward 3.0e-06、naive BNN1 回帰 3.1e-06。

## 成果物

- コード: `trainer/{kp,nnue,train_nnue,mirror,prepare_data,export_nnue,verify_kp_rust,verify_nnue_rust,pred_dispersion}.py`、
  `src/eval/{kp,nnue}.rs`、`src/main.rs`(`kp-from-sfen [canonical]`)、`docs/kp_spec.md`(§7 正準実装を記録)。
  すべて切替式(naive=既定・不変)。
- 非コミット(永続データ): `~/binbo-shogi-data/caches/{canon_8m,naive_8m}`、
  `~/binbo-shogi-data/models/{model_canon_fd8_mirror_8m,model_naive_fd8_mirror_8m}`。
