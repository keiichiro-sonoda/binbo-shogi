# Task 052: 表現力UP — HalfKP → HalfKA(玉を駒特徴に含める)— 結果ログ

目標: **HalfKP が原理的に欠く「両玉の相対位置(玉安全)」を HalfKA で足し、Lv3/Lv4 の評価天井が動くか**を
**棋力**で検証。判断は棋力(度合いは参考)。測定 1秒/手固定。

## 背景

- HalfKP は view0=自玉アンカー+非玉駒、view1=敵玉アンカー+非玉駒で、**(自玉, 敵玉)の king-king 関係を一度も特徴化しない**。
- **HalfKA = 玉も駒特徴(id 13)に含める**。view0 に敵玉の位置が `(自玉, 敵玉sq, K)` として入る = KP では原理的に
  不可能だった玉安全情報。**Lv3/Lv4 の評価天井を破る最有力の表現変更**(Tasks 049–051 で Lv1/Lv2 は探索で達成、Lv3/Lv4 は評価天井と確定)。

## 実装(既存 HalfKP を完全保存、新モード BNN4 / Mode::SharedKA)

- **次元**: 駒種 13→14(K=id13)、`FEATURES_PER_KING_KA=2344`、`VIEW_DIM_KA=189864`、`CANONICAL_SHARED_KA_TOTAL=189866`
  (HalfKP shared 176744 の +7.4%)。active=**81**(own41+opp40。HalfKP は 77)。詳細 `docs/kp_spec.md` §10。
- 採用 shared(BNN3)機構を流用、玉を board piece として含めるだけ。naive/two-band/shared は不変。
- **検証(parity 最優先, すべて green)**:
  - Python↔Rust KP **完全一致 30000/30000**(玉マス81/81・持駒26303・成駒16329・端玉17778)。
  - KA mirror perm: involution + bijection + 盤レベル flip 一致(572局)。初期局面 81 列 pin。
  - NNUE forward Python↔Rust 一致 **2.3e-06**(KA cp-2M)。`cargo test` 19/0、KA 詰み発見 OK、
    **既存 HalfKP shared 8M(BNN3)はロード・指し手とも不変**。

## 学習(cp+mirror、Task 051 で cp 確定)

| モデル | データ | best_val_loss | val_corr | decisive_sign | 対 HalfKP |
|---|---|---|---|---|---|
| HalfKA cp-2M | 2M(run_wdl_2m) | 830,069 | 0.8575 | 0.8671 | HalfKP 836939/0.8504/0.8600 → KA 僅か良 |
| HalfKA cp-8M | 8M(run1m+extras) | 590,369 | 0.8993 | 0.8968 | HalfKP shared 8M 583170/**0.9032** → KA 僅か**劣**(同一val) |

- **度合い⇔棋力の乖離(再掲)**: 2M では KA の val 指標が僅かに良、8M では僅かに劣。いずれも**小差**で、
  「リッチな表現=高い度合い」は単調でない。判断は棋力。

## 棋力(主判断)

### Stage1 — 2M ablation(同一2Mデータ・同一seed、表現のみ違い、4seed×24=96局)

| | HalfKA cp-2M | HalfKP cp-2M | 差 |
|---|---|---|---|
| Lv1 | 30.7%(26-63-7) | 29.7% | +1.0pp(ほぼ同) |
| Lv2 | 10.9%(10-85-1) | 16.7% | **−5.8pp(悪化)** |

- **Lv1/Lv2 は探索律速の領域**(Tasks 049–050)。HalfKA は棋力を改善せず、Lv2 ではむしろ悪化。
  ただしこの regime は HalfKA の価値(評価天井)を測る場ではない → 天井での本評価(Stage2)へ。

### Stage2 — 8M head-to-head(採用 HalfKP 8M と同一レシピ・同一seed)

測定は**段階的・信号駆動**(ラフ探り→信号の出たレベルだけ確認、無駄局を避ける):
Lv3/Lv4 ラフ12局 → Lv3 に信号 → Lv3 確認 → Lv1/Lv2/Lv4 補完。各レベル **3 seed×12 = 36局**。
採用 HalfKP と HalfKA は **total_features 以外すべて同一**(fd8/h16/cp/clamp5000/mse/mirror、同一8Mデータ)。

| Level | HalfKA 8M | HalfKP 8M(採用) | Δ(KA−KP) | per-seed 一貫性 |
|---|---|---|---|---|
| Lv1 | 40.3%(14-21-1) | **69.4%(24-10-2)** | **−29pp** | KP が3 seed すべて上 |
| Lv2 | 29.2%(10-25-1) | **44.4%(16-20-0)** | **−15pp** | KP が3 seed すべて ≥ |
| **Lv3** | **38.9%(14-22-0)** | 20.8%(6-27-3) | **+18pp** | **KA が3 seed すべて上**(3>1,7>3,4>2) |
| Lv4 | 11.1%(3-31-2) | 13.9%(5-31-0) | −3pp | 互角(両者とも天井) |

- **異常な非単調性**: HalfKA は Lv1 40 / Lv2 29 / Lv3 39% と**ほぼ横ばい**(弱い Lv1 にも強い Lv3 にも同程度)。
  HalfKP は 69/44/21% と**単調**(弱い相手ほど勝つ=正常)。HalfKA が**最弱の Lv1 に勝てず(40%<50%)
  強い Lv3 に勝つ**のは逆説的 = 一様な強化でなく**棋風の歪み**を示唆。
- 2M ablation(Stage1)でも HalfKA は Lv1 互角・Lv2 劣後で、8M と整合(Lv1/Lv2 劣後は頑健)。

## 決着

**HalfKA は採用しない(net で改善でない)。ただし「表現で初めて棋力が動いた」点は重要な収穫。**

- **棋力(主判断)**: HalfKA は **Lv3 で +18pp(3 seed 一貫)= プロジェクトで初めて表現変更が棋力を明確に動かした**
  (Tasks 047/050/051 の評価変更は全て棋力不変だった)。玉安全(king-king 関係)は**評価律速の Lv3 で確かに効く** = 仮説は部分的に実証。
- **だが採用は不可**: Lv1 −29pp / Lv2 −15pp と**勝てていたレベルを大きく退化**(Lv1 で 50% を割って**負け越し**)。
  平均(Lv1–Lv4)では採用 HalfKP が上。**最弱 Lv1 に勝てない eval は頑健でない**(非単調=棋風の歪み)。
- **解釈**: HalfKA 8M は val_corr 0.8993 < HalfKP 0.9032 と**全体の当てはめが僅かに劣る**。plain HalfKA は
  自玉-as-piece が冗長(定数特徴=容量の無駄)で、玉特徴が**全体 fit を僅かに崩しつつ Lv3 の玉安全だけ底上げ**した可能性。
- **次の一手(BACKLOG)**: ① **HalfKAv2**(自玉-as-piece の冗長除去・特徴マージ=Stockfish 流)で Lv1/Lv2 を崩さず
  Lv3 の利得を残せるか。② Lv1/Lv2 退化の真因調査(玉特徴の eval 寄与・棋風)。③ 容量/正則化で全体 fit を HalfKP 並みに戻す。
- **インフラは保持**(BNN4/Mode::SharedKA、parity 検証済み)。採用 EvalFile は **HalfKP shared 8M のまま**。

### 効いた/効かなかった(要点)
- **効いた**: HalfKA の玉安全特徴が **Lv3(評価律速)で +18pp** = 表現で棋力を動かせる初の実証。
- **効かない/退化**: Lv1/Lv2 を大きく退化(plain HalfKA の冗長・全体 fit 微減)。**net 非改善 → 不採用**。
- **教訓**: 表現の richer 化は「効く土俵(評価律速 Lv3)」では効くが、plain 実装は副作用(Lv1/Lv2 退化)を伴う。
  精緻化(HalfKAv2)が次段。**度合い(val_corr)は僅差でも棋力はレベル依存で大きく動く**(度合いは相変わらず弱い指標)。
