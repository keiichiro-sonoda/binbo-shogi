#!/usr/bin/env python3
"""qsearch replacement helpers (Task 119): hcpe rows -> psv -> YaneuraOu `qsearch_psv` -> leaf rows.

YaneuraOu's `qsearch_psv in out [workers]` replaces every record's position by the
leaf of its qsearch PV (driven by the loaded eval) and keeps the score, negating it
when the PV has an odd number of plies (yaneuraou-search.cpp `qsearch_psv_record`).
We only need the leaf and that parity, so every exported record carries score = +1:
after the command, score == -1 marks "side to move flipped" and the caller negates
its own labels for that row. Records keep their order (records are processed in place
per chunk and written back in order).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import cshogi
import numpy as np

PSV = cshogi.PackedSfenValue


def hcpe_to_psv(entries: np.ndarray) -> np.ndarray:
    """hcpe rows -> psv records (score +1, move none, gamePly 1, result 0)."""
    out = np.zeros(len(entries), dtype=PSV)
    board = cshogi.Board()
    buf = np.zeros(32, dtype=np.uint8)
    for i in range(len(entries)):
        board.set_hcp(np.asarray(entries[i]["hcp"]))
        board.to_psfen(buf)
        out[i]["sfen"] = buf
    out["score"] = 1
    out["gamePly"] = 1
    return out


class QSearchEngine:
    """A YaneuraOu process that runs `qsearch_psv` jobs one at a time."""

    def __init__(self, yane_bin, eval_dir, threads=4, hash_mb=16):
        self.threads = threads
        self.proc = subprocess.Popen([str(yane_bin)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     cwd=os.path.dirname(str(yane_bin)))
        self.lines = []
        self.lock = threading.Condition()
        threading.Thread(target=self._reader, daemon=True).start()
        for name, value in (("EvalDir", eval_dir), ("Threads", threads), ("USI_Hash", hash_mb),
                            ("USI_OwnBook", "false")):
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait(lambda line: line == "readyok", 120)

    def _reader(self):
        for line in self.proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))
                self.lock.notify_all()

    def _send(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def _wait(self, pred, timeout):
        end = time.time() + timeout
        with self.lock:
            start = 0
            while True:
                for k in range(start, len(self.lines)):
                    if pred(self.lines[k]):
                        line = self.lines[k]
                        del self.lines[:k + 1]
                        return line
                start = len(self.lines)
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError(f"no expected line within {timeout}s; last={self.lines[-5:]}")
                self.lock.wait(min(left, 5))

    def run(self, in_path, out_path, timeout=6 * 3600):
        """Run one file; returns the parsed `qsearch_psv done:` stats (raises on failure)."""
        self._send("usinewgame")   # clears the hash between files
        self._send(f"qsearch_psv {in_path} {out_path} {self.threads}")
        line = self._wait(lambda l: l.startswith("info string qsearch_psv done") or "qsearch_psv failed" in l
                          or l.startswith("info string failed") or l.startswith("info string usage"), timeout)
        if "done" not in line:
            raise RuntimeError(line)
        stats = dict(kv.split("=") for kv in line.split("done:")[1].split())
        return {k: int(v) for k, v in stats.items()}

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def leaf_rows(inp: np.ndarray, out: np.ndarray):
    """-> (changed bool[n], flip bool[n]) comparing the exported and returned records."""
    changed = np.any(inp["sfen"] != out["sfen"], axis=1)
    flip = out["score"] == -1
    if np.any(flip & ~changed):
        raise AssertionError("score flipped on an unchanged record")
    bad = ~np.isin(out["score"], (1, -1))
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} records with unexpected score")
    return changed, flip
