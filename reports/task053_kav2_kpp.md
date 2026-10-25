# Task 053: HalfKAv2 → KPP — 結果ログ

方針(ユーザー指示): HalfKAv2 を軽く試す → 微妙なら KPP(king-zone→full+差分)へ。差分更新が重くても
**検証を厚くして正しさを保証**した上で進める。測定は経済的に(ラフ→信号レベルだけ確認)。

## Part 1 — HalfKAv2(自玉-as-piece の冗長除去)— 微妙(むしろ悪化)→ Part 2 へ

**仮説**: plain HalfKA(Task 052)の自玉-as-piece は `(anchor, anchor, K)` の定数特徴=容量の無駄で、Lv1/Lv2 退化の
原因かも。各ハーフから**そのハーフのアンカー玉を落とし**(own=自玉, opp=相手玉)、交差玉だけ残す(active 81→79)。

**実装**: 新モード BNN5/Mode::SharedKAv2(KA 抽出を2行変更、同一 KA テーブル)。Python↔Rust KP **完全一致 30000/30000**、
forward 2.37e-06、`cargo test` 20/0、詰み発見 OK、既存モデル不変。同一8Mデータ・同一レシピ(cp+mirror)で学習。

**棋力(8M, 同一seed=Task052, 3seed×12=36局/Lv)**:

| | Lv1 | Lv2 | Lv3 |
|---|---|---|---|
| HalfKP(採用) | 69.4% | 44.4% | 20.8% |
| HalfKA(Task052) | 40.3% | 29.2% | **38.9%** |
| **HalfKAv2** | 45.8% | 29.2% | 23.6% |

- **両方の悪いとこ取り**: Lv1/Lv2 退化は直らず(≪ HalfKP)、**HalfKA の Lv3 利得を失った**(23.6 ≈ HalfKP、HalfKA −15pp)。
- **示唆**: 自玉-as-piece は冗長でなく **Lv3 利得に寄与**していた(冗長除去仮説は誤り)。両玉を含む plain HalfKA の方が
  king 情報を豊かに使えていた。val 指標も KAv2≈HalfKA(best_val_loss 590619 vs 590369)で改善せず。
- **判断**: 微妙(悪化)→ 深追いせず **Part 2(king-zone KPP)へ**。HalfKAv2 不採用。インフラ(BNN5)は保持。

## Part 2 — king-zone KPP(駒ペア, 差分不要)— 実装中(正確性の核は検証済み)

**設計**: 各玉の 5×5 zone(Chebyshev≤2)内の非玉駒を zone_state(`zone_rel(0..24)×ptype(0..12)×rel_owner(0..1)=650`)で
表し、**無順序ペア** `C(650,2)=210925` を特徴とする。**側手番正規化フレーム**(KP と一致)で実装し side-to-move 不変に:
own=自玉zone のペア、opp=相手玉zone のペアを canonical 180度回転(`zone_rel→24-zone_rel`)+owner反転で**共有テーブル**。
active ~27/half(≤KP)=**差分更新不要**。HalfKP + KPP-zone の **4アキュムレータ**(KP own/opp + KPP own/opp → concat 4·fd)。

**ここまで完了(検証済み)**:
- `trainer/kp.py` / `src/eval/kp.rs`: `kpp_zone_halves`(`pair_index` 組合せ rank)。**Python↔Rust 完全一致 30000/30000**
  (玉マス 81/81)。初期局面 pin(own/opp 各36ペア)。`cargo test` 21/0、perft 1–5 green。
- `trainer/nnue.py`: `KppNNUE`(kp_emb + kpp_emb の2テーブル、4アキュムレータ、空ペアは EmbeddingBag-sum=0 でパディング不要)。
- `trainer/prepare_data.py`: `--kpp-zone`(KP 77 + KPP own/opp を **ragged CSR**(`*_kpp_{own,opp}_{flat,len}.npy`)で格納、
  パディング無し=RAM 効率)。2M で検証(mean 27.5/27 pairs/half, max 153)。

**実装完了(end-to-end 検証済み)**:
- `train_nnue.py --kpp-zone`(KP固定77 + KPP CSR の4アキュムレータ学習、`run_batches_kpp`/`do_pass`)、
  `mirror.py build_kpp_zone_mirror_perm`(involution+盤flip一致 715/715)、`export_nnue`(BNN6, 2埋め込み, 24Bヘッダ)、
  `nnue.rs Mode::KppZone`(KP+KPP 4アキュムレータ)。**KPP forward Python↔Rust 一致 2.5e-06/3.0e-06**、cargo test 21/0、
  詰み発見、既存モデル不変。OOM 修正(flat 蓄積を array.array)。

**学習(8M cp+mirror)**: best_val_loss **533617 vs HalfKP 583170 = −8.5%**(プロトタイプ −8.7% と一致)、val_corr 0.9085 vs 0.9032。
**駒ペアは val fit を改善**(プロトタイプ再現)。

**棋力(8M, 同一seed=Task052, 3seed×12=36局/Lv)**:

| | Lv1 | Lv2 | Lv3 |
|---|---|---|---|
| HalfKP(採用) | 69.4% | 44.4% | 20.8% |
| HalfKA(Task052) | 40.3% | 29.2% | 38.9% |
| **KPP-zone** | 68.1% | 26.4% | 16.7% |

- **king-zone KPP は棋力を動かさない(むしろ低下)**: Lv1 は HalfKP 並みを保つ(HalfKA と違い退化なし)が、Lv2 −18pp・Lv3 −4pp。
  **val −8.5% 改善でも棋力 ≤ HalfKP** = 表現リッチ化は val を改善するが棋力を robust に上げない(047/050/051 と同じ)。
- **判断: 不採用、Part 3(full KPP+差分)に進まない**。king-zone(安い)で駒ペアが棋力を動かさないと分かった以上、
  18× 重い full KPP + 差分更新機構への投資は割に合わない(経済性ゲート)。採用は HalfKP shared 8M のまま。インフラ(BNN6)は保持。

## Part 3 — full KPP + 差分更新 — 着手しない(ゲート不成立)

king-zone KPP が棋力を動かさなかったため、**full KPP + 差分更新は着手しない**(差分更新は重く・full KPP は king-zone の上位だが、
その king-zone が棋力に効かない以上、効果の見込みが立たない)。差分更新(全特徴を高速化する基盤)は、将来 **速度**が律速に
なった時の独立インフラ課題として BACKLOG に残す(KPP の効果とは切り離す)。

## 総括(Task 053)

- **HalfKAv2(冗長除去)**: 微妙(悪化、Lv3 利得喪失)→ 不採用。自玉-as-piece は冗長でなく Lv3 利得に寄与していた。
- **king-zone KPP(駒ペア)**: val −8.5% 改善も棋力 ≤ HalfKP(Lv2/Lv3 低下)→ 不採用。
- **結論**: **表現のリッチ化(HalfKA/HalfKAv2/KPP)は棋力天井を破らない**。HalfKA の Lv3 +18pp は唯一の例外だが
  Lv1/Lv2 退化付きで net 非改善。**棋力の真の律速は表現でなく探索深さ(対 Gikou)or 先生の質**と強く示唆。
  次の本丸は表現でなく**探索の更なる深化** or **先生を強い DL 系に**。採用 EvalFile は HalfKP shared 8M のまま。

## Part 3 — full KPP + 差分更新

<!-- PART3 -->
