# KP 特徴表現 仕様(正本)

このドキュメントは binbo-shogi の **KP 特徴表現の唯一の正本**です。学習(Python)と
推論(Rust)はこの同一の付番規則に従い、Task 018 で30万局面の完全一致を確認しています。
**付番が1つでもずれると、学習済み重みが前提とする特徴の意味が変わり、評価が静かに無意味
になります。** 付番を変える/再実装するときは必ずこの正本と、後述の検算値を照合してください。

- 実装(正解基準): `trainer/kp.py`(Python、学習用)
- 実装(これに一致させる側): `src/eval/kp.rs`(Rust、推論用)
- 関連設計判断: `DESIGN.md`「KP は可逆な特徴表現として扱う」「NNUE の入力と層構造は小さく始める」
- このドキュメントはコードと照合して書いています。食い違いを見つけたら、コードではなく
  まずこの不一致を報告してください(正本とコードはどちらも真実であるべき)。

---

## 1. KP とは何か(概要)

- **KP(King-Piece)** = 玉の位置と1つの駒(位置・駒種・相対所有者)の関係を1特徴とする入力表現。
  駒のペアを見る KPP(King-Piece-Piece)より軽い線形特徴。
- **2視点を持つ HalfKP 系**:
  - view 0 = **手番側の玉**を基準(アンカー)にする。
  - view 1 = **相手側の玉**を基準にする。
  - "Half" とは「片方の玉から見た KP 特徴ひと組」の意味で、入力全体は自玉ハーフと相手玉ハーフの2枚。
- **手番側視点に統一**する。盤・駒・玉位置は手番側から見た向きに正規化し、駒の所有者も
  絶対色ではなく「手番側/相手側」の相対所有者で扱う。生徒評価値も手番側視点(`DESIGN.md`)。
- **知識ゼロ縛りとの線引き**(`DESIGN.md` と整合): KP は「何を入力として見るか」という枠で
  あり、「その関係が良いか悪いか」という評価の中身は含まない。枠は先人の知恵から借り、
  重み・評価は技巧2の蒸留と学習で獲得する。

---

## 2. 特徴空間の構造(次元の内訳)

| 記号 | 値 | 内訳 |
|---|---|---|
| `SQUARE_NB` | 81 | 盤のマス数 |
| 盤上非玉駒種 | 13 | `P,L,N,S,B,R,G,+P,+L,+N,+S,+B,+R`(玉は除く) |
| 相対所有者 | 2 | 0=手番側 / 1=相手 |
| 持ち駒スロット/所有者 | 38 | `P18 + L4 + N4 + S4 + G4 + B2 + R2` |
| `BOARD_FEATURES_PER_KING` | 2,106 | `2 × 13 × 81` |
| `HAND_FEATURES_PER_KING` | 76 | `2 × 38` |
| `FEATURES_PER_KING` | 2,182 | `2106 + 76` |
| `VIEW_DIM` | 176,742 | `81 × 2182`(玉位置 × 玉ごとの特徴数) |
| `TOTAL_FEATURES` | 353,486 | `2 × 176742 + 2` |
| `SIDE_FEATURE_BASE` | 353,484 | `2 × VIEW_DIM` |

- view 範囲: view 0 = `[0, 176742)`、view 1 = `[176742, 353484)`。
- 末尾の `+2` は side feature 2スロット(`353484`=実手番が先手、`353485`=実手番が後手)。
  片方しか立たないので情報的には1bit。可逆性確認のため実手番を明示的に持つ設計。

---

## 3. 付番式(インデックスの計算)

### 座標系

- cshogi/Apery square index: `square = (file-1) × 9 + (rank-1)`(`1a=0 … 9i=80`)。
  - Rust(shogi_core)では `cshogi_square = Square::index() - 1`(同式。hcpe の玉エンコードで検証済み)。
- **手番側視点への正規化**: 手番が先手なら `square` をそのまま、後手なら **180度回転** して `80 - square`。
  玉位置・駒位置の両方に同じ正規化を適用する。
- 相対所有者: `rel_owner = 0`(手番側の駒/持ち駒)、`rel_owner = 1`(相手側)。

### view 内共通

```text
base = view × VIEW_DIM(176742) + king_square_norm × FEATURES_PER_KING(2182)
```

### 盤上の駒

```text
index = base + (rel_owner × 13 + board_piece_id) × 81 + piece_square_norm
```

### 持ち駒

```text
index = base + 2106 + rel_owner × 38 + hand_kind_offset + slot
```

`slot` は同一駒種の持ち駒を 0 始まりで並べる(歩2枚なら slot 0,1 が立つ)。

### side feature

```text
index = 353484 + turn        # turn: 先手=0, 後手=1
```

### 駒種ID の割り当て(食い違いやすい箇所 — 要注意)

**盤上駒 `board_piece_id`**(Python `BOARD_PIECE_TO_ID`、Rust `board_piece_id()`):

| id | 駒 | id | 駒 |
|---|---|---|---|
| 0 | P 歩 | 7 | +P と |
| 1 | L 香 | 8 | +L 成香 |
| 2 | N 桂 | 9 | +N 成桂 |
| 3 | S 銀 | 10 | +S 成銀 |
| 4 | B 角 | 11 | +B 馬 |
| 5 | R 飛 | 12 | +R 龍 |
| 6 | **G 金** | | |

- ⚠️ **盤上駒は金(G)が飛(R)の後 = id 6**。基本駒の並びは `P,L,N,S,B,R,G`(金が最後)。
- ⚠️ Rust の `shogi_core::PieceKind` 列挙順は Gold が Bishop **より前**で、この付番と異なる。
  `src/eval/kp.rs` は推測せず明示 match で対応付けている。Task 015(打つ手の駒種番号ズレ)・
  Task 018 で実際に罠になった箇所。

**持ち駒 `hand_kind_offset`**(Python `HAND_KIND_OFFSETS` / `MAX_PIECES_IN_HAND`、Rust `HAND_OFFSET`):

| 駒 | offset | 最大枚数 |
|---|---|---|
| P 歩 | 0 | 18 |
| L 香 | 18 | 4 |
| N 桂 | 22 | 4 |
| S 銀 | 26 | 4 |
| G 金 | 30 | 4 |
| B 角 | 34 | 2 |
| R 飛 | 36 | 2 |

- ⚠️ **持ち駒の並びは `P,L,N,S,G,B,R`(金が銀の後)で、盤上駒の並び `P,L,N,S,B,R,G` とは違う**。
  cshogi の `BOARD_PIECE_TYPES` と `HAND_PIECE_TYPES` が別系統の列挙のため。金の位置が
  盤と持ち駒で食い違う点に注意(再実装時に最も間違えやすい)。

---

## 4. 「77特徴」の由来(不変量)

1局面で立つ特徴数は **常に 77**。

```text
立つ特徴数 = 2 × (盤上の非玉駒数 + 持ち駒の総枚数) + 1(side feature)
```

- 将棋は駒数保存。盤上の非玉駒 + 全持ち駒 = 玉2枚を除いた **常に 38 枚**。
- 同じ 38 枚を2視点(自玉基準・相手玉基準)で別々に見るので `38 × 2 = 76`、最後に side feature 1。
- よって `76 + 1 = 77`。
- これは強い不変量チェック: **77 から外れたら駒が落ちた/重複した/玉を数え間違えたサイン**。
  Task 013(可逆性)・Task 018(Python/Rust 一致)で `len == 77` を assert に使った。

---

## 5. 具体例(検算値)

初期局面・手番=先手の **▲7七歩**(cshogi square = `(7-1)×9 + (7-1)` = 60、先手なので正規化後も 60):

- 先手玉 ▲5九 = cshogi square 44(正規化後 44)、後手玉 △5一 = cshogi square 36(正規化後 36)。
- view 0(自玉アンカー 44): `0×176742 + 44×2182 + (0×13 + 0)×81 + 60` = **96068**
- view 1(相手玉アンカー 36): `1×176742 + 36×2182 + (0×13 + 0)×81 + 60` = **255354**

→ **同じ1枚の歩が、2視点で 96068 と 255354 の2つのインデックスを生む**(= HalfKP の本質)。
両者とも初期局面の KP 集合に含まれ、`decode_feature_index` で「view, 自/相手玉, 盤上駒,
rel_owner=0, 駒種=PAWN, piece_square=60」に戻ることを確認済み(`trainer/kp.py` で実検算)。

**将来この値が変わったら付番が変わった合図。** 再実装・変更時の回帰基準として使うこと
(`src/eval/kp.rs` の単体テストは初期局面の全77インデックスを Python 値に pin している)。

---

## 6. 保存と展開(疎表現・on-the-fly)

- KP は **疎**(353,486 次元中、立つのは 77 個だけ)。
  - 密に持つと爆発する(bitset で約 44 KB/局面、float32 で約 1.4 MB/局面)。絶対に密保存しない。
  - **疎表現 = 立っているインデックスのみ保持**。int32 で `77 × 4 = 308 B/局面`。
- **ディスク上の保存正本は局面(hcp、32 B)**であって KP ではない(`DESIGN.md`「保存正本は
  圧縮局面、KP は読み込み時変換」)。KP は **学習/推論時に hcp(hcpe)から展開する派生物**。
  - 速度のため `.npy` などにキャッシュすることはある(Task 017 の前処理)が、それも派生物。
- KP は hcp と **同じ情報を持つ可逆な再表現**(Task 012/013 で実証)。「KP にすると情報が増える」
  のではなく、ハフマン圧縮を疎特徴に展開する表現変換にすぎない(疎表現で約 9.6× の容量)。

---

## 7. 正準 HalfKP — 3モード(naive / two-band / shared)

§1〜§6 が記述する既定の付番は **naive HalfKP**。NNUE 原典(那須2018「Efficiently Updatable
Neural-Network-based Evaluation Functions」, §2.6〜2.7, 式15〜17, 表1)の**正準 HalfKP の核心は
「W₁(埋め込みテーブル)を両視点で共有する」**こと(式15 が手番側・非手番側の両方に同じ W₁ を適用、
p4「アフィン変換のパラメータ b₁, W₁ は共有される」、表1 で W₁ は **単一テーブル** 81×fe_end=125388列)。
各ハーフは**自分の玉1つだけ**をアンカーにし、**同じテーブルを共有**して両視点を作る(§2.7
「敵玉を含む位置関係を使用せず、自玉を含む位置関係だけを使用する」)。

正準化の差異は2点((a)相手ハーフの向き、(b)2アキュムレータ連結)で、**核心は共有 W₁**。実装には
3モードがある(naive=既定、不変。採用済み `model_cp_fd8_mirror_8m`(BNN1)と全過去検証は有効):

| モード | magic | 帯 | W₁ 共有 | 埋め込み行数 | 位置づけ |
|---|---|---|---|---:|---|
| **naive** | BNN1 | 2帯を1本に合算 | — | 353486 | 既定・採用 |
| **two-band**(Task 046) | BNN2 | view0/view1 別帯 | **無**(視点ごと独立) | 353486 | **偽の正準**(下記) |
| **shared**(Task 047) | BNN3 | 単一帯を両視点が共有 | **有** | **176744**(約半分) | **本物の正準** |

### ⚠️ Task 046(two-band)は正準ではなかった(訂正記録)

Task 046 の「canonical」は (a)相手ハーフ回転+反転、(b)2アキュムレータ連結 は正しかったが、
view0(自玉)を `[0,176742)`、view1(相手玉)を `[176742,353484)` の**別々のインデックス帯**に持ち、
両視点が**独立パラメータ**だった = **W₁ が共有されていない「2帯・非共有版」**。原典の核心(共有 W₁)を
欠いており正準ではない。結果、コスト(2アキュムレータ)だけ払い利得(共有=パラメータ半減・対称性
帰納バイアス)を持たず、実質「容量2倍」で効かなかった(Task 046 で val 度合い同一・gap 148→175 悪化)。
two-band は比較用に残す(BNN2)。詳細 `reports/task046_canonical_halfkp.md`。

### (a) 相手ハーフの向き(two-band / shared 共通)

view 1(相手ハーフ)を**相手目線=手番側フレームを180度回転 + 相対所有者反転**で持つ
(`rotate(X)=80−X`, `rel→1−rel`)。view 0(自ハーフ)は naive と同一。

### (b) アキュムレータ + W₁ 共有(shared が本物)

- two-band(BNN2): 自ハーフ=view0帯 `[0,176742)`、相手ハーフ=view1帯 `[176742,353484)`。**別の行**を引く=非共有。
- **shared(BNN3, 本物)**: 両ハーフとも**単一帯 `[0,VIEW_DIM)=[0,176742)`** を引く(view=0 固定)。
  - 自ハーフ: `self_king*2182 + offset(rel, piece, S)`。
  - 相手ハーフ: `rotate(opp_king)*2182 + offset(1−rel, piece, rotate(S))`。
  - **「自玉=K と駒X」と「相手玉=K と駒X(回転反転後)」が同じ行を共有する**(これが核心)。
  - `CANONICAL_SHARED_TOTAL = VIEW_DIM + 2 = 176744`(side feature 2行)。**埋め込みは naive の約半分**。
  - 検算(初期局面 ▲7七歩, Black手番): 自ハーフに `96068`(naive と同一)、相手ハーフに `97081`
    (= `44*2182 + 13*81 + 20`, **`<176742`=共有帯内**)。初期局面は180度+色対称なので
    **自ハーフ盤駒38個 = 相手ハーフ38個が完全一致**(共有の実証)。
  - 自/相手は index 値が衝突しうる(=共有の証拠)ので、**範囲分割でなく固定列レイアウト**
    `[own(39=body38+side) | opp(38)]=77` でキャッシュ保存し、列で分割する。
  - 可逆性: 自ハーフだけから局面を復元可(相手ハーフは同局面の回転=冗長)。
  - 左右ミラー拡張は shared でも**共有版 perm**(`build_mirror_perm_shared`, 長さ176744)で正しい
    (水平ミラーが180度回転と可換 `80−mirror(80−q)=mirror(q)`、二経路一致+involution で実証)。

### 重み形式 / Rust ディスパッチ

- export magic: naive=**BNN1**(network.0 入力 = fd)、two-band=**BNN2**、shared=**BNN3**(共に 2×fd 入力、
  ただし shared は total_features=176744)。Rust `Nnue::load` が magic で `Mode{Naive,TwoBand,Shared}` に分岐。
- 数値一致(Task 018/020 の精神): **shared KP Python/Rust 30000/30000 完全一致**、**shared forward
  max_abs_err ~1.2e-06**(2分布)。`verify_kp_rust --canonical-shared` / `verify_nnue_rust`(config 自動判別)で固定。

### 実装の所在

| 役割 | naive(BNN1) | two-band(BNN2, Task046) | shared(BNN3, Task047 本物) |
|---|---|---|---|
| KP 抽出 Python | `board_to_kp_indices(b)` | `…(b, canonical=True)` | `canonical_shared_halves(b)` / `canonical_shared_row(b)` |
| KP 抽出 Rust | `kp::kp_indices` | `kp::kp_indices_canonical` | `kp::kp_indices_canonical_shared` |
| モデル Python | `MinimalNNUE` | `CanonicalNNUE`(353486) | `CanonicalNNUE`(176744) |
| 学習/前処理 | — | `--canonical` | `--canonical-shared` |
| mirror perm | `build_mirror_perm` | 同左(共有) | `build_mirror_perm_shared` |

### 副次差(正準性とは独立): 駒符号化の非標準性

現実装の per-king 特徴数は **2182**(=2×13×81 盤 + 2×38 持駒)で、原典標準の **fe_end=1548**
(表1: W₁ 列数 125388 = 81×1548)より**密・非標準**(成り/不成・全マスを素朴に展開、持駒はスロット式)。
これは naive/two-band/shared 共通で正準性の論点とは独立。将来 標準ツール(やねうら王等)の重みを
流用する場合は付番差に注意。今回は触らない(記録のみ)。

> (a) は特徴の向き、(b)+共有 はモデルの消費側。§3 の付番式は naive を記述。
> 評価結果は `reports/task046_canonical_halfkp.md`(two-band)・`reports/task047_canonical_shared.md`(shared)。

---

## 8. 実装の所在と検証

| 役割 | 実装 |
|---|---|
| 正解基準(Python、学習) | `trainer/kp.py`(`board_to_kp_indices`, `kp_indices_to_sfen`, `*_feature_index`) |
| これに一致させる側(Rust、推論) | `src/eval/kp.rs`(`kp_indices(&yasai::Position) -> Vec<u32>`) |
| クロスチェック用 CLI | `binbo-shogi kp-from-sfen <sfen_in> <indices_out>` |
| Python↔Rust 照合スクリプト | `trainer/verify_kp_rust.py` |

**Python が正解基準であり、Rust はそれに完全一致させる関係**(もし Python 側にバグが見つかれば
別途相談だが、基本は Python が真)。

検証の経緯:

- **Task 012**: 可逆性(局面 → KP → 局面)を4局面で確認。
- **Task 013**: 可逆性を約50万局面(持ち駒・成駒・終盤・端の玉を含む多様局面)で確認。
- **Task 018**: Python と Rust の KP が **30万局面・2分布(技巧2自己対局20万 + RandomEngine10万)で
  完全一致**(不一致ゼロ)。両手番・持ち駒あり・成駒あり・端の玉・玉位置81/81マスを網羅。
  `src/eval/kp.rs` の単体テストで初期局面の77インデックスと次元定数を Python 値に pin。

詳細レポート: `reports/task018_rust_kp_parity.md`。

---

## 9. KPP(King-Piece-Piece)— 規模見積もりと縮小設計(Task 048)

KP(玉と1駒)が律速と絞られた後(naive/two-band/shared 正準を尽くし、残り ~30%・Lv1 の壁は KP が
駒ペア関係を持たないことに帰着)、次の richer 候補が **KPP(玉と2駒の関係)**。ただし KPP は特徴空間が
桁違いに大きいため、**まず規模・コストを見積もってから**実装する(測ってから走る、Task 043/045 の教訓)。

### 9.1 full KPP の規模(16GB PC では非現実的)

共有正準を踏襲し、テーブル = 81玉 × C(fe_end, 2)、両視点で共有:

| fe_end | ペア/玉 | テーブル行 | 埋め込み(fd8) | +Adam |
|---:|---:|---:|---:|---:|
| 2182(現密版) | 2,379,471 | 192,737,151 | 5.74 GB | **17.2 GB** |
| 1548(原典標準) | 1,197,378 | 96,987,618 | 2.89 GB | 8.67 GB |

- 1局面の立つ特徴: 非玉駒38 → C(38,2)=703/視点 ×2 = **1406(KP の 18×)**。
- cache(8M): 1406×4×8M = **42 GB**(int32)= RAM 16GB 不可。
- forward: 差分なしで ~18× 遅(KP-shared 1.26M nps → ~70K nps)= 探索深さ低下=生徒の優位(深く読む)を削る。
- **結論: full KPP は GPU/RAM に乗らず、差分計算(原典の本質)なしでは速度的にも非現実的**。差分は別課題(BACKLOG)。

### 9.2 縮小設計: king-zone KPP(玉周りの駒ペアに限定)

KPP の価値の主因は**玉周りの駒関係(玉安全)**。pairs を**玉の 5×5 zone(Chebyshev ≤ 2)内の駒**に限定すると激減する
(実測, run1m): zone 内駒の both-in-zone pairs = 平均 **53/局面(0.69× KP!)**・max 123。

- テーブル(zone-relative): zone-state = `zone_rel_sq(0..24) × piece_type(0..12) × rel_owner(0..1) = 650`、
  unordered pair = `C(650,2) = 210,925` 行(king-square 次元を持たせる full 版でも 81×C(650,2) ≈ 17M 行 = **0.51 GB**)。
- 立つ特徴 ~53/局面 ≤ KP なので **forward は KP 並み(差分不要)・メモリ余裕** = 16GB PC で現実的。
- 原典流の無駄排除: zone-relative で盤外・遠方を最初から作らない。**正準共有を踏襲**(相手ハーフ = zone を180度回転 + 所有者反転、
  同じ KPP テーブルを共有)。`≥1-in-zone`(平均 411, 5.3×)は中間(玉×任意駒の文脈、テーブル 81×650×2182 ≈ 115M=3.4GB)。
- モデル: KP own/opp + KPP-zone own/opp = 4アキュムレータ連結 → `Linear(4*fd, hidden)`(KP-shared への加算拡張)。

### 9.3 効果(度合いが伸びるか)と次段

効果プロトタイプ(Python のみ・Rust parity 前・mirror なし・run1m 1M・ablation, `trainer/kpp_zone_probe.py`):
KP-shared own/opp に king-zone KPP own/opp を加えた4アキュムレータ vs KP-only(2アキュムレータ)を同条件で比較。

| | KP-only | KP + KPP-zone |
|---|---:|---:|
| best val_loss | 1,295,097 | **1,183,061(−8.7%)** |
| val_corr | 0.7683 | **0.7926(+0.024)** |
| val 度合い(/3915) | 1971(50.3%) | 1993(50.9%) |

- **king-zone の駒ペアは評価を改善する**(val_loss −8.7%・corr +0.024)。度合いは +0.6pp と小さいが、全体精度(loss/corr)は明確に向上。
  立つ特徴 ~27/half(KP 38 より少)・抽出 71µs/局面 = 軽い・差分不要 = 16GB PC で現実的。
- ※ プロトタイプは mirror 無しで即過学習(絶対値は低い)。ablation は内部統制ありなので delta は有効。
- **次段(本実装の候補)**: king-zone KPP を Python/Rust 数値一致・正準共有・mirror 込みで本実装し、生800万で
  度合い・**棋力**(本命)を測る。ただし Task 047(度合い +1pp でも Lv1 棋力動かず)を踏まえると、度合い +0.6pp の
  本 KPP も棋力は動きにくい見込み。full KPP(差分前提)へ広げるかは king-zone の棋力結果で判断。

### 9.4 本実装と決着(Task 053)

king-zone KPP を本実装(magic **BNN6** / Rust `Mode::KppZone`)。付番(`trainer/kp.py::kpp_zone_halves`):
- **zone_state** = `zone_rel(0..24) × ptype(0..12) × rel_owner(0..1) = 650`(`(zone_rel*13 + pid)*2 + rel_owner`)。
  `zone_rel = (df+2)*5 + (dr+2)`(df,dr = 駒 − 玉の file,rank, |·|≤2)。**側手番正規化フレーム**で side-to-move 不変。
- **特徴 = 無順序ペア** `_kpp_pair_index(a,b)`(組合せ rank), `KPP_ZONE_TOTAL = C(650,2) = 210925`。
  own = 自玉zone のペア、opp = 相手玉zone のペアを **canonical 180度回転(`zone_rel→24-zone_rel`)+owner反転**で共有テーブル。
- モデル = HalfKP shared + KPP-zone の **4アキュムレータ**(`KppNNUE`: kp_emb + kpp_emb, concat 4·fd)。cache は ragged CSR
  (`*_kpp_{own,opp}_{flat,len}.npy`, パディング無し)。mirror = `build_kpp_zone_mirror_perm`(zone file 反転)。
- 検証: Python↔Rust **numbering 30000/30000・forward 2.5e-06**、初期局面 pin(own/opp 各36)、KPP mirror perm(involution+盤flip一致)。

**決着(棋力)**: 8M cp+mirror で **val_loss −8.5%(プロトタイプ −8.7% 再現)も棋力 ≤ HalfKP**(Lv1 維持・Lv2/Lv3 低下)。
**駒ペアは val を上げるが棋力を robust に動かさない** → **不採用**。**full KPP(差分前提)は着手しない**(king-zone が効かない以上
18×重い投資は不当)。表現リッチ化(HalfKA/HalfKAv2/KPP)は棋力天井を破らず、真の律速は探索深さ/先生と結論。詳細 `reports/task053_kav2_kpp.md`。

---

## 10. HalfKA — 玉を駒特徴に含める拡張(Task 052)

§1〜§7 の HalfKP は**玉をアンカーにのみ使い、駒特徴には含めない**(§2 の盤上非玉駒種=13)。そのため
view0(自玉)は非玉駒しか見ず、**(自玉, 敵玉)という king-king 関係を一度も特徴化しない**=玉安全
(opposition・距離・囲い)が表現に無い構造的欠陥。**HalfKA("A"=All)= 玉も駒特徴に含める**ことで、各ハーフに
**敵玉の位置**が `(自玉アンカー, 敵玉sq, K)` として入る(KP では原理的に不可能だった情報)。

採用基盤 **shared(BNN3, §7)上の並行モード**として実装(magic **BNN4** / Rust **Mode::SharedKA**)。既存
naive/two-band/shared(HalfKP)の付番・モデル・検証は**一切不変**。

### 10.1 KA 次元(HalfKP の 13→14 駒種、King=id13)

| 記号 | 値 | 内訳 |
|---|---|---|
| 盤上駒種 KA | 14 | §3 の 13 種 + **K(玉)= id 13** |
| `BOARD_FEATURES_PER_KING_KA` | 2,268 | `2 × 14 × 81` |
| `FEATURES_PER_KING_KA` | 2,344 | `2268 + 76`(持駒は §2 と同じ) |
| `VIEW_DIM_KA` | 189,864 | `81 × 2344` |
| `CANONICAL_SHARED_KA_TOTAL` | 189,866 | `VIEW_DIM_KA + 2`(side 2行)。HalfKP shared 176744 の +7.4% |

- **立つ特徴 = 81(不変量)**: `2 × (非玉駒38 + 持駒) + 玉2×2 + side`… 正確には盤上駒=非玉38+玉2=**40/half**、
  両ハーフ ×2 + side = **2×40+1 = 81**(将棋は常に玉2枚=強い不変量。HalfKP の 77 → KA の 81)。
- 固定列レイアウト: **own(41=40body+side) ++ opp(40) = 81**。

### 10.2 付番(shared 帯、view=0)

§7(b) shared と同形。玉も他の盤上駒と同じ式で符号化(`board_piece_id_ka(King)=13`):
```text
own: self_king * 2344 + (rel*14 + piece_id)*81 + psq
opp: rotate(opp_king) * 2344 + ((1-rel)*14 + piece_id)*81 + rotate(psq)
```
自玉は `(anchor, anchor, K)`=玉位置ごとに固定オフセット(冗長だが無害=玉位置 bias に畳まれる)、
**敵玉が king-king 関係の本体**。plain HalfKA(両玉を含む)を採用(HalfKAv2 の冗長除去は将来)。

### 10.3 検算値(初期局面・先手)

`canonical_shared_ka_halves` の own/opp は対称局面で body 一致。玉特徴: **自玉 104233**(`44*2344+13*81+44`)、
**敵玉 105359**(`44*2344+27*81+36`)。`src/eval/kp.rs` の `canonical_shared_ka_startpos_matches_python` が
全 81 列を pin。

### 10.4 実装と検証

| 役割 | 実装 |
|---|---|
| Python 抽出 | `trainer/kp.py`: `canonical_shared_ka_halves` / `canonical_shared_ka_row` / `*_ka` 付番・decode |
| Rust 抽出 | `src/eval/kp.rs`: `kp_indices_canonical_shared_ka`(両玉を board_pieces に push, id13) |
| mirror | `trainer/mirror.py`: `build_mirror_perm_shared_ka`(len 189866) |
| CLI / 照合 | `kp-from-sfen … canonical-shared-ka`、`verify_kp_rust.py --canonical-shared-ka` |
| 学習/推論 | `prepare_data/train_nnue --canonical-shared-ka`、`export_nnue`(BNN4)、`nnue.rs Mode::SharedKA` |

- **Python↔Rust KP 完全一致**: 30000/30000(技巧2自己対局・持駒26303・成駒16329・端玉17778・玉マス81/81)。
- mirror perm: involution + bijection + 盤レベル flip 一致(572局)。初期局面 81 列 pin(`cargo test` 19/0)。
- NNUE forward Python↔Rust 一致: max_abs_err 2.3e-06(KA cp-2M モデル)。既存 HalfKP shared 8M(BNN3)はロード・指し手とも不変。
