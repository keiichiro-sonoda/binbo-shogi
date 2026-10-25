#!/usr/bin/env python3
"""Task 135 confirmation on fresh seeds: is the fast-trainer retrain really stronger than the adopted eval?

The standard frame gave +24.2 [+0.5, +48.2] (FAST_BETTER) for `task135_student_fast` (same recipe as the
adopted `task129_student_64m_qsrel_e12`, only the trainer path differs). docs/evaluation_protocol.md:
a "stronger" result is confirmed on fresh seeds before adoption.

Decision (fixed before the run): seeds 20273501-08 (never used before; prefix conf135), same yardstick
(tanuki- @20k, AobaNNUE @10k, 100k nodes, 8 x 100 games per anchor), paired delta pooled:
  CI low > 0 -> ADOPT (task135_student_fast becomes the adopted eval)
  otherwise  -> NOT_CONFIRMED (keep task129; the first result is read as run-to-run variation)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402

CAND, ADOPTED = "task135_student_fast", "task129_student_64m_qsrel_e12"
SEEDS = tuple(20273501 + i for i in range(8))
PREFIX = "conf135"


def main():
    repo, data, logs, *_ = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(repo))
    cand, b1 = std_frame.measure(CAND, env, seeds=SEEDS, prefix=PREFIX)
    base, b2 = std_frame.measure(ADOPTED, env, seeds=SEEDS, prefix=PREFIX)
    pooled, per, b3 = std_frame.delta(cand, base, "confirm (fast retrain - adopted)")
    failures = b1 + b2 + b3
    lo = pooled["ci95"][0]
    verdict = "INVALID" if failures else "ADOPT" if lo > 0 else "NOT_CONFIRMED"
    print(f"T135C_DELTA pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
          + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    out = logs / "task135_confirm_results.json"
    out.write_text(json.dumps({"seeds": SEEDS, "pooled": pooled, "per_anchor": per, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T135C_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
