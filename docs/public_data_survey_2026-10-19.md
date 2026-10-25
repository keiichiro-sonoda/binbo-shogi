# 公開されている将棋 NNUE の教師データ (調査, 2026-10-19)

目的: 自前の 64M 局面 (技巧2 の自己対局 → 静止局面に置き換え → WS-T S=8) に足せる, 他の開発者が公開した教師データがあるか。
調査は Web のみ (ダウンロードはしていない)。【明記】= 出典に書いてある, 【推論】= こちらの解釈。

## 候補
| データ | 作り方 | 規模 | ライセンス | 形式 |
|---|---|---|---|---|
| **nodchip/shogi_suisho5_depth9** (Hugging Face) | 【明記】水匠5 depth 9, **静止探索の PV 末端に置き換え済み**, シャッフル済み | 学習用 7z 分割 計 185GB (展開後の局面数は記載なし), 検証用 `shuffled.bin` 289,626,920B (【推論】約 724 万局面) | **MIT** (タグ。本文の追加条件なし, 原文で確認) | やねうら王 PackedSfenValue (40B/局面) |
| nodchip/shogi_hao_depth9 | 【明記】Hao depth 9, 置き換えなし, シャッフルなし | 320GB, .bin 1 本 290〜340MB (【推論】約 750 万局面) | MIT | 同上 |
| nodchip/tanuki-.nnue-pytorch-2024-07-30.1 | 【明記】tanuki- depth 9, 置き換えなし | 320GB (10 億局面 × 3 組) | MIT | 同上 |
| nodchip/shogi_suisho5_depth9_entering_king | 入玉局面 | | MIT | 同上 |
| たややん氏の Suisho10Mn_psv 等 (Google Drive) | 水匠 1 手 1,000 万ノードの対局, 約 1 億局面 | 未確認 | **明記なし** (「共有財産として」の文言のみ) | psv |
| dlshogi の GCT 学習用データ (Google Drive) | 自己対局 hcpe3 等 | | 明記なし | hcpe / hcpe3 |

使わないもの: DL 水匠で付け直したデータと DL 水匠そのもの (支援者向けの頒布), 説明もライセンスもない再アップロード版, 書籍付属データの再配布。

## 尺度
【推論】やねうら王の慣例は 勝率 = sigmoid(cp/600) = (1 + tanh(cp/1200))/2 で, binbo のラベル cp = 1200·atanh(v) と同じ目盛り。ただし中身は depth 9 の αβ 探索値で, WS-T (DL の value + MCTS) とは分布が違う (Task 139: 教師と物差しの相関 0.47)。混ぜる比率の調整が要る。

## 大会での扱い
【明記】WCSC34 では複数チームが「nodchip 氏が公開している教師データ群」「たややん氏が公開している教師データ群」を使用ライブラリとして申告している。電竜戦の参加者一覧にも使用ライブラリの欄がある。【推論】使うなら申込とアピール文書に使用ライブラリとして書く。

## 次の一歩 (案)
1. `nodchip/shogi_suisho5_depth9` の `shuffled.bin` (290MB) だけを取り, 形式の確認と小さな試し (724 万局面で学習, または自前 64M に混ぜる) をする。作り方 (静止局面に置き換え済み) が binbo のパイプラインに最も近い。
2. 効くなら, 7z (185GB) のための空きを作るか, Hao / tanuki- の .bin を数本ずつ (1 本 300MB) 取って自前で静止局面に置き換える。

出典: https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9 , https://huggingface.co/datasets/nodchip/shogi_hao_depth9 , https://huggingface.co/datasets/nodchip/tanuki-.nnue-pytorch-2024-07-30.1 , https://nodchip.hatenablog.com/entry/2024/12/25/000000 , https://www.apply.computer-shogi.org/wcsc34/appeal/Suisho/WCSC34Suisho.pdf , https://yaneuraou.yaneu.com/2024/12/30/shogi-ai-may-transform-fundamentally/
