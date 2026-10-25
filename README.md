# binbo-shogi

binbo (びんぼう) は将棋 AI の個人開発です。名前は「貧乏」から: 大規模な GPU やクラウドを使わず, 自宅の PC で評価関数を一から学習して作っています。

今の binbo は, **やねうら王 (無改変) の探索 + 自作の学習器で作った NNUE 評価関数** です。評価関数の土台は公開教師データ (水匠5 depth 9, nodchip 氏, MIT) で学習し, その上で自分のソフトどうしの自己対局 (詰みまで指して勝敗を付けたもの) で仕上げています。
最初は「ルール以外の人間の知識を与えずに作る」構想で, 自前の αβ 探索と技巧2 の蒸留から始めました。そこから何をどう変えてきたかは [docs/history/README.md](docs/history/README.md) にまとめています。

## 中身

| 場所 | 内容 |
|---|---|
| `src/` | 自前の Rust 実装: USI エンジン, αβ 探索, NNUE の推論, 教師データの生成 |
| `trainer/` | 評価関数の学習器 (PyTorch, 量子化を意識した学習), やねうら王形式への書き出し |
| `experiments/` | 実験ごとのドライバ (AlphaZero / Transformer の寄り道, 判定の対局, 時計の実験など) |
| `reports/` | 実験ごとの報告。効かなかったものも含む |
| `docs/` | 設計, 判定の方法 (`docs/evaluation_protocol.md`), 使っている部品とライセンス (`docs/USED_COMPONENTS.md`), [経緯](docs/history/README.md), [年表](docs/history/timeline.md) |
| `STATUS.md` | 段階ごとの詳しい作業記録 |

学習済みの重み, 教師データ, 外部エンジンの実体は含みません。

## 公開前の検査

push の前に, 鍵・同定情報・非公開の情報が入っていないかを `tools/publish_lint.py` で機械的に確かめています。
クローンしたら `git config core.hooksPath .githooks` で pre-push フックを有効にしてください。詳細は [docs/publish_check.md](docs/publish_check.md)。

## ライセンス

作者が書いた部分は GPL-3.0-or-later (`LICENSE`)。他者の著作物とデータは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 自前の Rust 実装

合法手生成には `yasai` を使い, ローカルの perft オラクル検証との完全一致を確かめています。依存は git revision で固定しています。

```toml
yasai = { git = "https://github.com/sugyan/yasai.git", rev = "51c23780109a4ac7258c27d8965640d4f8b78e9e", features = ["simd"] }
```

### ビルド

```sh
cargo build --release
```

`.cargo/config.toml` で `target-cpu=native` を指定し、release profile では LTO を有効化しています。

WSL/Linux では `target/release/binbo-shogi` ができます。Windows ネイティブの `.exe` が必要な場合は、Windows 側に Rust MSVC toolchain と Visual Studio Build Tools を入れてビルドします。

```powershell
cargo build --release
```

Windows バイナリは `target\release\binbo-shogi.exe` です。

### 実行

USI エンジンとして起動:

```sh
cargo run --release
```

基本ハンドシェイク確認:

```sh
printf 'usi\nisready\nquit\n' | cargo run --release
```

perft オラクル検証:

```sh
cargo run --release -- verify
```

単発 perft / divide:

```sh
cargo run --release -- perft 5
cargo run --release -- divide 2
```

ベンチマーク:

```sh
cargo run --release -- bench 5 5
```

ランダム自己対戦スモークテスト:

```sh
cargo run --release -- selfplay 512 1
```

### ShogiHome

ShogiHome にはビルド済みバイナリを USI エンジンとして登録します。ShogiHome から実行できる環境でなければ、Windows ネイティブ `.exe` を MSVC toolchain で作って登録します。

USI の stdout は UTF-8 のプロトコル出力専用です。独自 `d` コマンドの盤面表示など、デバッグ出力は stderr に出します。
