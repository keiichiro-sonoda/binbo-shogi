#!/usr/bin/env python3
"""Task 138c: retrains B and C on the third seed set 20273601-08 (prefix conf138), where the adopted eval's games
were played by t138b. Gives every eval three 800-game seed sets per anchor (std, conf135, conf138), so the
retrain-vs-adopted difference can be read over 2,400 paired games per anchor instead of one frozen base sample.
Descriptive; adoption is decided by the author."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402


def main():
    repo, *_ = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(repo))
    bad = []
    for name in ("task138_seedC", "task138_seedB"):
        bad += std_frame.measure(name, env, seeds=tuple(20273601 + i for i in range(8)), prefix="conf138")[1]
    print(f"T138C_DONE failures={bad}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
