#!/usr/bin/env python3
"""Task 181: the confirmation stage of t181_mixfinish.py, run on its own.

The runner passed its primary gate (M - adopted +22.4 [+7.6, +37.4]) and was stopped before its confirmation,
because its candidate side would have shared the 14700KF's memory with the Task 180 clock experiment. The base
side's routine chunks were already played on the Xeon and imported. This plays the rest here once that experiment
is done, with the same sets, anchors and rule as the runner: conf181 = spaced_set(20297000); routine anchors and
micro; ADOPT_CANDIDATE if both pooled point estimates are positive.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
from t181_mixfinish import ADOPTED, CONF, NAME, TAG  # noqa: E402


def main():
    repo, data, logs, *_ = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(repo))
    failures, conf = [], {}
    for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
        cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=CONF, which=which)
        cc, b2 = std_frame.measure_sets(NAME, env, sets=CONF, which=which)
        ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({NAME} - {ADOPTED})")
        failures += b1 + b2 + b3
        conf[label] = {"pooled": ct, "per_anchor": cper}
        print(f"T181_CONFIRM {label} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in cper.items()), flush=True)
    confirmed = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
    verdict = "INVALID" if failures else ("ADOPT_CANDIDATE" if confirmed else "NOT_CONFIRMED")
    path = logs / f"{TAG}_results.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record.update(verdict=verdict, confirm=conf, confirm_failures=failures)
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"T181_VERDICT {verdict} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
