#!/usr/bin/env python3
"""Task 138b: is the adopted eval's standard-frame score unluckily low? (descriptive, games only)

Task 138: three retrains of the adopted recipe all scored +18..+34 vs the adopted eval on the standard seeds,
but -4 / +12 on fresh seeds 20273501-08. Node-fixed games repeat exactly, so the adopted eval's standard-frame
games are ONE frozen sample reused by every comparison; if it is low, every delta is inflated. Here:
  1. the adopted eval on a third seed set 20273601-08 (prefix conf138), and
  2. retrain seed B on 20273501-08 (prefix conf135; the adopted eval's games there already exist),
to see the adopted eval's spread across seed sets and the retrains' mean on fresh seeds. No verdict.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402

ADOPTED = "task129_student_64m_qsrel_e12"


def main():
    repo, *_ = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(repo))
    _, bad1 = std_frame.measure(ADOPTED, env, seeds=tuple(20273601 + i for i in range(8)), prefix="conf138")
    _, bad2 = std_frame.measure("task138_seedB", env, seeds=tuple(20273501 + i for i in range(8)), prefix="conf135")
    print(f"T138B_DONE failures={bad1 + bad2}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
