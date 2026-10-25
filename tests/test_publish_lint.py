"""tools/publish_lint.py と .githooks/pre-push の試験。

門番は「止まるべき時に止まる」だけでなく「通るべき時に通る」も見ないと判定にならない
(常に止まる壊れ方を「効いている」と読んでしまう)。両方を同じ形で並べる。

⚠️ 検体 (鍵・メール・非公開の語) は literal で書かず実行時に組み立てる。
   このファイル自身が公開物なので, literal で書くとこのファイルが門番に止められる。

    uv run --no-project --with pytest python -m pytest tests/test_publish_lint.py
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINT = ROOT / "tools" / "publish_lint.py"
HOOK = ROOT / ".githooks" / "pre-push"

# 実行時に組み立てる検体
SECRET_WORD = "zq" + "xv" + "private" + "word"  # 非公開の語の代わり
TOKEN = "gh" + "p_" + "A1" * 18  # GitHub トークンの形 (36 文字)
MAIL = "someone" + "@" + "example" + ".com"
USER_DIR = "/" + "home" + "/someone/data"
COMMITTER = "t" + "@" + "t.invalid"  # 試験用の作者。これも literal で書くと P2 で止まる


def run_git(repo: pathlib.Path, *args: str, stdin: str | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": COMMITTER,
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": COMMITTER}
    proc = subprocess.run(["git", *args], cwd=repo, input=stdin, capture_output=True,
                          encoding="utf-8", env=env, check=True)
    return proc.stdout.strip()


def commit(repo: pathlib.Path, files: dict[str, str | bytes], message: str = "add") -> str:
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


@pytest.fixture
def private_repo(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """非公開のリポジトリの代わり。そのコミットの短縮 SHA を返す。"""
    repo = tmp_path / "private"
    repo.mkdir()
    run_git(repo, "init", "-q", "-b", "main")
    sha = commit(repo, {"secret.md": "private\n"})
    return repo, sha[:7]


@pytest.fixture
def repo(tmp_path: pathlib.Path, private_repo: tuple[pathlib.Path, str]) -> pathlib.Path:
    """公開側の代わり。非公開の語の一覧と非公開のリポジトリを設定済み, origin は空。"""
    pub = tmp_path / "public"
    pub.mkdir()
    run_git(pub, "init", "-q", "-b", "main")
    run_git(pub, "config", "publish.privateRepo", str(private_repo[0]))
    common = pathlib.Path(run_git(pub, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "info").mkdir(exist_ok=True)
    (common / "info" / "publish-denylist").write_text(f"# test\n{SECRET_WORD}\n", encoding="utf-8")
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    run_git(pub, "remote", "add", "origin", str(bare))
    commit(pub, {"README.md": "# ok\n", "src/a.rs": "fn main() {}\n"}, "init")
    return pub


def lint(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", str(LINT), "--root", str(repo), "--remote", "origin", *args],
                          capture_output=True, encoding="utf-8", check=False)


# --- 通るべき時に通る ---------------------------------------------------------

def test_clean_repo_passes(repo: pathlib.Path) -> None:
    r = lint(repo)
    assert r.returncode == 0, r.stdout
    assert "指摘なし" in r.stdout


def test_trailer_email_passes(repo: pathlib.Path) -> None:
    commit(repo, {"b.md": "x\n"}, f"msg\n\nCo-Authored-By: Someone <{MAIL}>")
    assert lint(repo).returncode == 0


def test_already_pushed_commits_are_not_rechecked(repo: pathlib.Path) -> None:
    commit(repo, {"c.md": f"{SECRET_WORD}\n"})
    run_git(repo, "push", "-q", "origin", "main")
    commit(repo, {"d.md": "fine\n"})
    assert lint(repo).returncode == 0


def test_upstream_hash_that_is_not_private_passes(repo: pathlib.Path) -> None:
    commit(repo, {"e.md": "upstream rev `" + "0123abc" + "`\n"})
    assert lint(repo).returncode == 0


# --- 止まるべき時に止まる -------------------------------------------------------

@pytest.mark.parametrize(
    ("content", "rule"),
    [
        (f"key: {TOKEN}\n", "P1"),
        (f"contact {MAIL}\n", "P2"),
        (f"data at {USER_DIR}\n", "P2"),
        ("host " + ".".join(["192", "168", "1", "20"]) + "\n", "P2"),
        (f"on {SECRET_WORD.upper()} machine\n", "P3"),
        ("see https://" + "claude.ai/code/" + "session_" + "abc12345" + "\n", "P5"),
    ],
)
def test_bad_line_stops(repo: pathlib.Path, content: str, rule: str) -> None:
    commit(repo, {"notes.md": content})
    r = lint(repo)
    assert r.returncode == 1, r.stdout
    assert f" {rule} " in r.stdout


def test_denylisted_word_is_not_echoed(repo: pathlib.Path) -> None:
    commit(repo, {"notes.md": f"{SECRET_WORD}\n"})
    r = lint(repo)
    assert r.returncode == 1
    assert SECRET_WORD not in r.stdout.lower()


def test_private_commit_reference_stops(repo: pathlib.Path, private_repo: tuple[pathlib.Path, str]) -> None:
    commit(repo, {"report.md": f"pre-registered at `{private_repo[1]}`\n"})
    r = lint(repo)
    assert r.returncode == 1 and " P4 " in r.stdout, r.stdout


def test_added_then_removed_still_stops(repo: pathlib.Path) -> None:
    commit(repo, {"leak.md": f"{TOKEN}\n"})
    commit(repo, {"leak.md": "gone\n"})
    r = lint(repo)
    assert r.returncode == 1 and " P1 " in r.stdout


def test_secret_in_message_stops(repo: pathlib.Path) -> None:
    commit(repo, {"f.md": "x\n"}, f"oops {TOKEN}")
    assert lint(repo).returncode == 1


def test_email_in_message_body_stops(repo: pathlib.Path) -> None:
    commit(repo, {"f.md": "x\n"}, f"ask {MAIL}")
    assert lint(repo).returncode == 1


def test_session_trailer_stops(repo: pathlib.Path) -> None:
    commit(repo, {"f.md": "x\n"}, "msg\n\nClaude-Session: https://" + "example.invalid/s/" + "x" * 12)
    r = lint(repo)
    assert r.returncode == 1 and " P5 " in r.stdout


@pytest.mark.parametrize("path", ["tools/denryusen/run.sh", "docs/denryusen7_prep.md", "eval/nn.bin", ".env"])
def test_never_publish_path_stops(repo: pathlib.Path, path: str) -> None:
    commit(repo, {path: "x\n"})
    r = lint(repo)
    assert r.returncode == 1 and " P6 " in r.stdout, r.stdout


def test_binary_stops(repo: pathlib.Path) -> None:
    commit(repo, {"img/a.dat": b"\x00\x01\x02binary"})
    r = lint(repo)
    assert r.returncode == 1 and " P7 " in r.stdout


# --- 検査できない時は合格にしない -----------------------------------------------

def test_missing_denylist_is_exit_2(repo: pathlib.Path) -> None:
    common = pathlib.Path(run_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "info" / "publish-denylist").unlink()
    r = lint(repo)
    assert r.returncode == 2, r.stdout


def test_empty_denylist_is_exit_2(repo: pathlib.Path) -> None:
    common = pathlib.Path(run_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "info" / "publish-denylist").write_text("# only a comment\n", encoding="utf-8")
    assert lint(repo).returncode == 2


def test_unset_private_repo_is_exit_2(repo: pathlib.Path) -> None:
    run_git(repo, "config", "--unset", "publish.privateRepo")
    assert lint(repo).returncode == 2


def test_private_repo_none_is_explicit_opt_out(repo: pathlib.Path) -> None:
    run_git(repo, "config", "publish.privateRepo", "none")
    r = lint(repo)
    assert r.returncode == 0 and "照合なし" in r.stdout


def test_unknown_tip_is_exit_2(repo: pathlib.Path) -> None:
    assert lint(repo, "--to", "no-such-ref").returncode == 2


# --- フック: push 対象の ref を 1 つずつ見る -----------------------------------

@pytest.mark.skipif(shutil.which("sh") is None, reason="sh が無い")
def test_hook_checks_every_pushed_ref(repo: pathlib.Path) -> None:
    good = run_git(repo, "rev-parse", "HEAD")
    run_git(repo, "checkout", "-q", "-b", "leaky")
    bad = commit(repo, {"g.md": f"{TOKEN}\n"})
    run_git(repo, "checkout", "-q", "main")
    stdin = f"refs/heads/main {good} refs/heads/main {'0' * 40}\nrefs/heads/leaky {bad} refs/heads/leaky {'0' * 40}\n"
    # ⚠️ PUBLISH_LINT_CMD は単語に分割されるので, 空白を含むパスでは試験できない
    if " " in str(sys.executable) + str(LINT) + str(repo):
        pytest.skip("パスに空白がある")
    env = {**os.environ, "PUBLISH_LINT_CMD": f"{sys.executable} -I {LINT} --root {repo}"}
    r = subprocess.run(["sh", str(HOOK), "origin"], cwd=repo, input=stdin, capture_output=True,
                       encoding="utf-8", env=env, check=False)
    assert r.returncode == 1, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh が無い")
def test_hook_passes_clean_push_and_skips_deletion(repo: pathlib.Path) -> None:
    good = run_git(repo, "rev-parse", "HEAD")
    stdin = f"refs/heads/main {good} refs/heads/main {'0' * 40}\n(delete) {'0' * 40} refs/heads/old {'1' * 40}\n"
    env = {**os.environ, "PUBLISH_LINT_CMD": f"{sys.executable} -I {LINT} --root {repo}"}
    if " " in str(sys.executable) + str(LINT) + str(repo):
        pytest.skip("パスに空白がある")
    r = subprocess.run(["sh", str(HOOK), "origin"], cwd=repo, input=stdin, capture_output=True,
                       encoding="utf-8", env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
