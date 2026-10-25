#!/usr/bin/env python3
"""Task 152: does Task 142's gain (task142_finetune over task129) also hold against つようら王11micro?

One-off check with the confirmation-only anchor (std_frame.CONFIRM_ANCHORS = micro @11,000 nodes, Task 151), on the new
seed sets s280/s281/s282 (2,400 games per eval). Descriptive: report the pooled paired delta and per set; on the routine
anchors the same pair gives +32.5 [+18.7, +46.4] (Task 145)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402

NEW, OLD = "task142_finetune", "task129_student_64m_qsrel_e12"


def main():
    repo, data, logs, *_ = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(repo))
    a, b1 = std_frame.measure_sets(NEW, env, which=std_frame.CONFIRM_ANCHORS)
    b, b2 = std_frame.measure_sets(OLD, env, which=std_frame.CONFIRM_ANCHORS)
    total, per, per_set, b3 = std_frame.delta_sets(a, b, f"({NEW} - {OLD}) vs micro")
    print(f"T152_DELTA pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
          f"sets={ {k: round(v['delta_elo'], 1) for k, v in per_set.items()} } failures={b1 + b2 + b3}", flush=True)
    (logs / "task152_results.json").write_text(json.dumps({"pooled": total, "per_set": per_set, "failures": b1 + b2 + b3},
                                                         indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
