# Task 048: 学習哲学のドキュメント化 + KPP の規模・コスト調査

## 結論

- **部A(哲学)**: `DESIGN.md` に学習哲学を明記。**評価の中身(局面の良し悪し)= 知識ゼロ縛りを維持**(人間戦略知識を
  注入せず技巧2 蒸留で学ぶ)、**実装の工夫・技術(符号化・正準化・差分・SIMD・量子化・NNUE 構造)= なぜ効くかを
  理解した上で先人の手法を取り込む**(縛りの対象外)。MVV-LVA(Task 028)・入力構造の既存線引きの一般化。
- **採用切替**: 推奨 EvalFile を **`model_shared_fd8_mirror_8m`(BNN3 本物正準, Task 047)**に変更。棋力は naive と
  誤差だが**パラメータ半減(5.7MB)・forward ~11% 速・過学習耐性・val 度合い最高**で全指標 naive 以上。
- **部B(KPP)**:
  - **full KPP は 16GB PC で非現実的**: テーブル 18.6GB(+Adam, fe_end 2182)・cache 8M 42GB・forward ~18×遅
    = GPU/RAM に乗らず、差分計算(原典の本質)なしでは速度も非現実的。
  - **king-zone KPP(玉5×5 zone 内の駒ペア)は現実的**: テーブル 0.51GB・立つ特徴 ~27/half(KP 38 より少)・差分不要。
  - **効果プロトタイプ(同条件 ablation)で king-zone KPP は評価を改善**: val_loss −8.7%・corr +0.024(度合い +0.6pp)。
    = **駒ペア関係は効く**。ただし度合いの伸びは小さく、Task 047(度合い +1pp でも棋力動かず)を踏まえると棋力は
    動きにくい見込み。→ 次段は king-zone KPP の本実装(Python/Rust 一致・mirror・8M・棋力)。

## 部A: 学習哲学(DESIGN.md に記録)

プロジェクトの姿勢を「純粋自前主義」から **「まず自分で考え手を動かし、調査してより良い既存手法が見つかれば、
なぜ効くかを理解した上で取り込む(将棋AIの歴史を体感する)」**へ更新。2軸を分ける:

- **評価の中身 = 知識ゼロ(維持)**: 定跡・囲い・大局観・手作り評価特徴・人間棋譜の模倣を入れない。重み・判断は蒸留で獲得。
- **実装技術 = 理解して取り込む(対象外)**: 符号化効率化・正準化(W₁共有)・差分・SIMD・量子化・NNUE 構造。

価値の実例(自分で手を動かしたから理解できた): ①素朴な駒符号化を作ったから「絶対1にならないビット ~500個」の無駄に
気づき原典標準(fe_end=1548)の意味を実感。②正準 HalfKP を 素朴→偽(046)→原典照合→本物(047)と辿り W₁ 共有の意味
(パラメータ半減・対称性・過学習減)を実データで理解。**理解なき流用も純粋自前も避け、その中間を行く**。

## 部B-1: KPP 規模見積もり

共有正準踏襲でテーブル = 81玉 × C(fe_end, 2):

| fe_end | テーブル行 | 埋め込み(fd8) | +Adam | 1局面の立つ特徴 | cache 8M |
|---:|---:|---:|---:|---:|---:|
| 2182(密) | 193M | 5.74 GB | **17.2 GB** | 1406(KP の 18×) | **42 GB** |
| 1548(標準) | 97M | 2.89 GB | 8.67 GB | 同上 | 同上 |

- GPU 16GB / RAM 16GB に乗らない。forward ~18×遅(KP-shared 1.26M nps → ~70K nps)= 探索深さ低下=生徒の優位を削る。
- **full KPP は差分計算(原典 §2.5, BACKLOG)が前提**。今回は素の forward でコストを測り、重い=差分の必要性を確認。

## 部B-2: king-zone KPP(縮小設計・実測)

KPP の価値の主因は玉周りの駒関係(玉安全)。pairs を玉の 5×5 zone(Chebyshev ≤ 2)内に限定:

| 設計 | 立つ特徴(実測) | テーブル | 位置づけ |
|---|---:|---:|---|
| full board-pair | 平均 962 / max 1406(~12×) | 193M | 非現実的 |
| **king-zone both-in-zone** | **平均 53(2視点)/ ~27 per half** | **0.51 GB**(zone-rel は 0.006GB) | **現実的** |
| king-zone ≥1-in-zone | 平均 411(5.3×) | 3.4 GB | 中間 |

- zone-relative 符号化(zone_rel_sq 0..24 × ptype × owner = 650, unordered pair = C(650,2))で**原典流に盤外・遠方を最初から排除**。
  **正準共有を踏襲**(相手ハーフ = zone 180度回転 + 所有者反転、同じ KPP テーブルを共有)。立つ特徴 ~27/half ≤ KP = **差分不要・軽い**。

## 部B-3: 効果プロトタイプ(駒ペアは効くか)

`trainer/kpp_zone_probe.py`(Python のみ・Rust parity 前・mirror なし・run1m 1M・12ep)。KP-shared own/opp に
king-zone KPP own/opp を加えた4アキュムレータ vs KP-only(2アキュムレータ)を同条件 ablation:

| | KP-only | KP + KPP-zone |
|---|---:|---:|
| best val_loss | 1,295,097 | **1,183,061(−8.7%)** |
| val_corr | 0.7683 | **0.7926(+0.024)** |
| val 度合い(/3915) | 1971(50.3%) | 1993(50.9%) |

- **king-zone の駒ペアは評価を改善**(val_loss −8.7%・corr +0.024)。全体精度が上がる。度合いは +0.6pp と小。
- 抽出 71µs/局面・active ~27/half = 軽い。※ mirror 無しで即過学習(絶対値低)、ablation は内部統制ありで delta 有効。

## 部B-4: 結論と次段

- **KPP は full では PC で非現実的(差分が前提)。king-zone KPP は現実的で、駒ペアは評価を改善する(小幅)。**
- **次段(本実装の候補)**: king-zone KPP を Python/Rust 数値一致・正準共有・mirror・生800万で本実装し、度合い・**棋力**を測る。
  ただし Task 047(度合い +1pp でも Lv1 棋力動かず)を踏まえ、度合い +0.6pp の本 KPP も**棋力は動きにくい見込み**。
  king-zone の棋力が動けば full KPP(差分実装)へ広げる根拠になる。動かなければ表現の軸も頭打ち=自己対戦学習等(DESIGN 将来構想)へ。

## 回帰(green)

- コード変更は `DESIGN.md`(哲学)・`docs/kp_spec.md` §9(KPP 設計)・各ドキュメントのみ(本体 Rust/学習は無改修)。
  `cargo test` 18/18・perft 1〜5 一致は Task 047 から不変(本タスクで Rust 無変更)。
- 採用切替は重みファイルの指定変更のみ(`model_shared_fd8_mirror_8m`、Task 047 で Rust 一致・合法手確認済み)。

## 成果物

- `DESIGN.md`(学習哲学)、`docs/kp_spec.md` §9(KPP 規模・縮小設計・効果)、`STATUS.md`/`docs/BACKLOG.md`/`docs/data_storage.md`(採用切替・KPP)。
- 非コミット: 採用重み `~/binbo-shogi-data/models/model_shared_fd8_mirror_8m`。プロトタイプ `trainer/kpp_zone_probe.py`(参考)。
