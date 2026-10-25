#!/usr/bin/env python3
"""push で取り消せないものだけを止める (公開前の門番)。

公開リポジトリでは, 一度出したものは引っ込められない。参照を消しても SHA を知って
いれば取れることがあり, fork・キャッシュ・アーカイブにも渡る。鍵は失効させるしかなく,
機械や人の同定情報には取り消す手段そのものが無い。

一方で, 説明の誤り・数字の取り残し・命名は, あとから直せる。なので, この門番が見るのは
**追記で回復できない種類だけ**にする (直せるものを門番に積むと, 鳴っても止めない癖がつく)。

| ID | 内容 | 見る範囲 |
|----|------|----------|
| P1 | 鍵・トークンらしき文字列 | 各コミットの追加行とメッセージ |
| P2 | ユーザー名入りのパス・メールアドレス・IP アドレス | 各コミットの追加行とメッセージ |
| P3 | 非公開の語 (端末名・別の作業の名前など。一覧は追跡しない場所に置く) | 各コミットの追加行とメッセージ |
| P4 | 非公開の履歴のコミットへの参照 | 各コミットの追加行とメッセージ |
| P5 | 会話ログ・エージェントの作業場所へのポインタ | 各コミットの追加行とメッセージ |
| P6 | 公開しないと決めたパス・重みやデータの実体 | 各コミットで増えた・変わったファイル |
| P7 | 中身を検査できないもの (バイナリ) | 各コミットの `--numstat` |

すべて**コミットを1つずつ**見る。足して次のコミットで消しても blob は履歴に残り,
SHA を知っていれば取れるので, 最終差分だけでは足りない。

非公開の語の一覧 (P3) と非公開のリポジトリの場所 (P4) は, このリポジトリに入れない。
一覧を公開物に書けば, それ自体が漏洩になる。どちらも用意されていなければ,
「指摘なし」ではなく**検査できない (終了コード 2)** で止める。

    python3 tools/publish_lint.py                     # origin/main..HEAD を見る
    python3 tools/publish_lint.py --remote origin     # origin のどこからも辿れないコミット (pre-push 用)
    python3 tools/publish_lint.py --base <ref>        # 比較先を変える

設定 (どちらもクローンごと。追跡されない):

    <git の共通ディレクトリ>/info/publish-denylist    # P3。1 行 1 語, # から行末は注釈
    git config publish.privateRepo <非公開のリポジトリのパス>   # P4
    git config publish.privateRepo none               # P4 を意図して切る (非公開のリポジトリが無い機械)

`Co-Authored-By:` などの trailer 行のメールは通す (コミットの帰属の約束)。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (パス, 新しい側の行番号, 行の中身)
Added = tuple[str, int, str]
# (ルール, 場所, 何が)
Finding = tuple[str, str, str]

# --------------------------------------------------------------------------
# P1 鍵・トークン
#
# ⚠️ 検体を literal で書かない。長さを要求する形にしてあるので, この表そのものは
#    検体にならない (テスト側も実行時に組み立てる)。
# --------------------------------------------------------------------------
URL_CREDENTIALS = re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")

SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PEM 秘密鍵らしき文字列", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub トークンらしき文字列", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{36}\b")),
    ("GitHub PAT らしき文字列", re.compile(r"\bgithub_pat_[0-9A-Za-z_]{22,}")),
    ("AWS アクセスキーらしき文字列", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("`sk-` で始まる API キーらしき文字列", re.compile(r"\bsk-[0-9A-Za-z_-]{20,}")),
    ("Slack トークンらしき文字列", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}")),
    ("Authorization ヘッダらしき文字列", re.compile(r"[Aa]uthorization:\s*(?:Bearer|Basic)\s+\S")),
    ("認証情報つきの URL", URL_CREDENTIALS),
    # 値に英字と数字の両方を要求する。`password = {PASSWORD}` のような「言及」で鳴らないため
    (
        "秘密らしき代入",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|token|api[-_]?key)\s*[=:]\s*[\"']?"
            r"(?=[^\s\"']*[A-Za-z])(?=[^\s\"']*\d)[^\s\"'{}]{8,}"
        ),
    ),
)

# --------------------------------------------------------------------------
# P2 同定情報
# --------------------------------------------------------------------------
USER_PATH = (
    "ユーザー名を含む絶対パス",
    re.compile(r"/(?:home|Users|root)/[A-Za-z0-9._-]+|\b[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+"),
)
EMAIL = ("メールアドレス", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"))
IPV4 = ("IP アドレスらしき文字列", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"))
FILE_IDENTITY = (USER_PATH, EMAIL, IPV4)
MESSAGE_IDENTITY = (USER_PATH, IPV4)
TRAILER = re.compile(r"^[A-Za-z][A-Za-z-]*-[Bb]y:\s")
# 生成物なので中身を選べない。4 連の版番号が IPV4 と衝突しうる
GENERATED = ("Cargo.lock", "package-lock.json", "uv.lock")

# --------------------------------------------------------------------------
# P4 非公開の履歴
# --------------------------------------------------------------------------
HEX = re.compile(r"\b[0-9a-f]{7,40}\b")

# --------------------------------------------------------------------------
# P5 会話ログ・作業場所へのポインタ
#
# ⚠️ 使用と言及を分ける。規約を説明した行 (AGENTS.md など) まで鳴ると,
#    門番のことを書けなくなる。落とすのは URL か不透明な値が続くときだけ
# --------------------------------------------------------------------------
SESSION_LINE = re.compile(r"^\s*Claude-Session:\s*(?:\S+://|[0-9A-Fa-f][0-9A-Fa-f-]{15,})", re.MULTILINE)
POINTERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("会話ログの URL", re.compile(r"\b(?:claude\.ai/(?:code|chat)|chatgpt\.com/(?:c|codex))/[0-9A-Za-z_-]{8,}")),
    ("エージェントの一時領域のパス", re.compile(r"/tmp/claude-\d+/")),
    ("エージェントの作業ツリーのパス", re.compile(r"\.codex/worktrees/[0-9A-Za-z]+")),
    ("エージェントの記憶へのリンク", re.compile(r"\[\[[a-z0-9]+(?:-[a-z0-9]+){2,}\]\]")),
)

# --------------------------------------------------------------------------
# P6 公開しないと決めたもの
#
# パスは「無いことに意味がある」ものだけ (大会の本番用の道具, 手順書, アピール文書,
# 機械固有の監視, 私的使用のモデルの道具)。重みやデータは拡張子で止める。
# --------------------------------------------------------------------------
NEVER_PATHS = (
    "tools/denryusen/",
    "scripts/monitor/",
    "docs/denryusen7_",
    "trainer/dlshogi_rescore.py",
)
NEVER_SUFFIXES = (
    ".bin", ".onnx", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".hcpe", ".7z", ".zip",
    ".pem", ".key", ".log", ".csa",
)
NEVER_NAMES = re.compile(r"(?:^|/)\.env(?:\.|$)")

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ESCAPE = "\x1b["


class GitError(RuntimeError):
    """git が答えを返さなかった。

    ⚠️ 「見つからなかった」と混ぜない。混ぜると, 差分を取れなかったときに
    「指摘なし」と出てしまう。黙っているのと合格しているのを区別できないのが
    門番としていちばん悪い壊れ方になる。
    """


class SetupError(RuntimeError):
    """検査に要る設定が無い (非公開の語の一覧・非公開のリポジトリ)。"""


def git(root: pathlib.Path, *args: str) -> str:
    """git を呼んで標準出力を返す。失敗したら GitError。

    ⚠️ 色を止めるのは 3 重にしてある。差分に色が付くと行頭が `+` で始まらなくなり,
    **追加行が 1 つも見えないまま合格する**。
    """
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", "-c", "color.ui=false", "-c", "color.diff=false", *args],
        cwd=root,
        capture_output=True,
        # ⚠️ text=True はロケールで復号する。cp932 の環境では日本語の差分で落ちる
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        raise GitError(f"git {' '.join(args)} が終了コード {proc.returncode}: {detail[:1]}")
    return proc.stdout


def git_ok(root: pathlib.Path, *args: str) -> bool:
    """git の成否だけを答えとして使う (存在の確認)。"""
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    return proc.returncode == 0


def parse_added_lines(text: str) -> list[Added]:
    """`--unified=0` の出力から追加行だけを拾う。

    マージは `--cc` で見る。**全ての親に対して `+`** の行だけを, この範囲で入った行と数える
    (片方の親から来た行は既に履歴にある)。
    """
    if ESCAPE in text:
        raise GitError("差分に色が付いている。git の color 設定を確認すること")
    out: list[Added] = []
    path = ""
    lineno = 0
    width = 1
    for line in text.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = "" if target == "/dev/null" else target.removeprefix("b/")
        elif line.startswith("@@"):
            marks = len(line) - len(line.lstrip("@"))
            width = max(marks - 1, 1)
            m = re.search(r"\+(\d+)(?:,\d+)?\s@@", line)
            lineno = int(m.group(1)) if m else 0
        elif path and len(line) > width:
            prefix = line[:width]
            if set(prefix) == {"+"}:
                out.append((path, lineno, line[width:]))
                lineno += 1
            elif "-" not in prefix:
                lineno += 1
    return list(dict.fromkeys(out))


def load_denylist(root: pathlib.Path, override: pathlib.Path | None) -> list[str]:
    """P3 の一覧。無い・空なら SetupError (黙って素通りさせない)。"""
    if override is not None:
        path = override
    else:
        common = pathlib.Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
        path = common / "info" / "publish-denylist"
    if not path.is_file():
        raise SetupError(
            f"非公開の語の一覧が無い: {path}\n"
            "  1 行 1 語で書く (端末名・ユーザー名・別の作業の名前など)。このファイルは追跡されない"
        )
    words = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    words = [w for w in words if w]
    if not words:
        raise SetupError(f"非公開の語の一覧が空: {path}")
    return words


def private_repo(root: pathlib.Path, override: str | None) -> pathlib.Path | None:
    """P4 で照合する非公開のリポジトリ。`none` は意図して切ったもの。"""
    value = override
    if value is None:
        proc = subprocess.run(
            ["git", "config", "--get", "publish.privateRepo"], cwd=root, capture_output=True, encoding="utf-8", check=False
        )
        value = proc.stdout.strip() or None
    if value is None:
        raise SetupError(
            "非公開のリポジトリの場所が設定されていない (P4 を検査できない)\n"
            "  git config publish.privateRepo <パス>   (無い機械では none を明示する)"
        )
    if value == "none":
        return None
    path = pathlib.Path(value)
    if not git_ok(path, "rev-parse", "--git-dir"):
        raise SetupError(f"publish.privateRepo が git のリポジトリを指していない: {value}")
    return path


def private_commits(repo: pathlib.Path, tokens: Iterable[str]) -> set[str]:
    """`tokens` のうち, 非公開のリポジトリでコミットに解決できるもの。"""
    uniq = sorted(set(tokens))
    if not uniq:
        return set()
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(rest)"],
        cwd=repo,
        input="".join(f"{t} {t}\n" for t in uniq),
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(f"非公開のリポジトリで cat-file が終了コード {proc.returncode}")
    hits = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "commit":
            hits.add(parts[1])
    return hits


def line_findings(path_or_msg: str, at: str, text: str, denylist: list[str], is_message: bool) -> Iterator[Finding]:
    """1 行に P1・P2・P3・P5 を当てる (P4 はまとめて照合するので別)。"""
    for name, pattern in SECRETS:
        if pattern.search(text):
            yield ("P1", at, name)
    if is_message:
        rules = MESSAGE_IDENTITY if TRAILER.match(text) else (*MESSAGE_IDENTITY, EMAIL)
    else:
        rules = tuple(r for r in FILE_IDENTITY if not (r is IPV4 and path_or_msg.endswith(GENERATED)))
        if URL_CREDENTIALS.search(text):
            rules = tuple(r for r in rules if r is not EMAIL)
    for name, pattern in rules:
        if m := pattern.search(text):
            yield ("P2", at, f"{name} ({m.group(0)})")
    lowered = text.lower()
    for word in denylist:
        if word.lower() in lowered:
            # ⚠️ 語そのものは出さない。端末の出力やログに残ると, それが漏れ口になる
            yield ("P3", at, f"非公開の語 (一覧の {denylist.index(word) + 1} 行目)")
    for name, pattern in POINTERS:
        if pattern.search(text):
            yield ("P5", at, name)


def path_findings(name_status: str, where: str) -> Iterator[Finding]:
    """P6。増えた・変わったファイルのパスだけを見る (消したものは公開を減らすだけ)。"""
    for line in name_status.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[0][:1] == "D":
            continue
        path = parts[-1]
        if path.startswith(NEVER_PATHS):
            yield ("P6", f"{where}{path}", "公開しないと決めたパス")
        elif path.lower().endswith(NEVER_SUFFIXES) or NEVER_NAMES.search(path):
            yield ("P6", f"{where}{path}", "重み・データ・鍵・ログの実体らしきファイル")


def binary_findings(numstat: str, where: str) -> Iterator[Finding]:
    """P7。`git diff` はバイナリに `+` 行を出さない。検査できなかったものを合格に混ぜない。"""
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] == "-" and parts[1] == "-":
            yield ("P7", f"{where}{parts[2]}", "バイナリなので中身を検査できない")


def commits_to_publish(root: pathlib.Path, tip: str, base: str | None, remote: str | None) -> list[str]:
    """この push で新しく公開されるコミット (古い順)。

    `--remote` なら「そのリモートのどこからも辿れないコミット」。新しいブランチでも
    force push でも, これが「今から公開されるもの」そのもの。
    """
    if remote is None:
        assert base is not None
        return git(root, "rev-list", "--reverse", f"{base}..{tip}").split()
    return git(root, "rev-list", "--reverse", tip, "--not", f"--remotes={remote}").split()


def collect(
    root: pathlib.Path,
    base: str | None,
    tip: str,
    remote: str | None,
    denylist: list[str],
    private: pathlib.Path | None,
) -> tuple[list[Finding], int]:
    shas = commits_to_publish(root, tip, base, remote)
    found: list[Finding] = []
    hex_sites: list[tuple[str, str]] = []  # (場所, 16 進の並び)
    for sha in shas:
        where = f"{sha[:7]} "
        parents = git(root, "rev-list", "--parents", "-n", "1", sha).split()[1:]
        first = parents[0] if parents else EMPTY_TREE
        text = git(root, "show", "--format=", "--unified=0", "--cc", "--no-color", sha)
        if not parents:
            # 根のコミットは show が差分を出さない版があるので, 空のツリーとの差で取る
            text = git(root, "diff", "--unified=0", "--no-color", EMPTY_TREE, sha)
        for path, lineno, line in parse_added_lines(text):
            at = f"{where}{path}:{lineno}"
            found += line_findings(path, at, line, denylist, is_message=False)
            hex_sites += [(at, h) for h in HEX.findall(line)]
        found += path_findings(git(root, "diff", "--name-status", "--no-color", first, sha), where)
        found += binary_findings(git(root, "diff", "--numstat", "--no-color", first, sha), where)

        body = git(root, "log", "-1", "--format=%B", sha)
        for lineno, line in enumerate(body.splitlines(), 1):
            at = f"{sha[:7]} のメッセージ:{lineno}"
            found += line_findings("", at, line, denylist, is_message=True)
            hex_sites += [(at, h) for h in HEX.findall(line)]
        if SESSION_LINE.search(body):
            found.append(("P5", f"{sha[:7]} のメッセージ", "Claude-Session: の行がある"))

    if private is not None:
        hits = private_commits(private, (h for _, h in hex_sites))
        found += [("P4", at, f"非公開の履歴のコミット ({h})") for at, h in hex_sites if h in hits]
    return found, len(shas)


def main(argv: list[str]) -> int:
    # ⚠️ Windows では標準出力が cp932 になり, パイプ越しに読む側 (フック・試験) が化ける。
    #    -I で起動するので PYTHONIOENCODING も効かない。ここで utf-8 に固定する
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="push で取り消せないものを検査する")
    parser.add_argument("--base", default="origin/main", help="比較先 (既定: origin/main)")
    parser.add_argument("--to", default="HEAD", help="検査の終点 (既定: HEAD)")
    parser.add_argument("--remote", help="このリモートのどこからも辿れないコミットを見る (--base より優先。pre-push 用)")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT, help="検査するリポジトリ")
    parser.add_argument("--denylist", type=pathlib.Path, help="非公開の語の一覧 (既定: <git の共通ディレクトリ>/info/publish-denylist)")
    parser.add_argument("--private-repo", help="非公開のリポジトリ (既定: git config publish.privateRepo)")
    args = parser.parse_args(argv)

    base = None if args.remote else args.base
    if base is not None and not git_ok(args.root, "rev-parse", "--verify", f"{base}^{{commit}}"):
        print(f"publish_lint: {base} が見えない。git fetch してから回すこと (初回は --remote origin)")
        return 2
    if not git_ok(args.root, "rev-parse", "--verify", f"{args.to}^{{commit}}"):
        print(f"publish_lint: {args.to} が見えない")
        return 2

    try:
        denylist = load_denylist(args.root, args.denylist)
        private = private_repo(args.root, args.private_repo or os.environ.get("PUBLISH_PRIVATE_REPO"))
        found, n_commits = collect(args.root, base, args.to, args.remote, denylist, private)
    except (GitError, SetupError) as exc:
        # ⚠️ ここで 0 を返さない。検査できなかったことを合格と区別する
        print(f"publish_lint: 検査できなかった (終了コード 2)。{exc}")
        return 2

    if found:
        print(f"publish_lint: 公開すると取り消せない指摘が {len(found)} 件\n")
        for rule, where, what in found:
            print(f"  {rule} {where}: {what}")
        print("\n⚠️ push は point of no return。参照を消しても SHA から取れることがあり, 鍵は失効, 同定情報は取り消す手段が無い")
        return 1
    against = f"{args.remote} に無いもの" if args.remote else f"{args.base} との差"
    p4 = "非公開の履歴との照合なし (publish.privateRepo=none)" if private is None else "非公開の履歴とも照合"
    print(f"publish_lint: 取り消せない指摘なし ({args.to[:12]} まで {n_commits} コミット, {against}, {p4})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
