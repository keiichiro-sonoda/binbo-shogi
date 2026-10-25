#!/usr/bin/env python3
"""Task 118: FV_SCALE sweep of the adopted eval (no training).

YaneuraOu divides the NNUE output by FV_SCALE before the search sees it. Our export
folds the cp scale so that FV_SCALE 16 (the default) gives exactly the trained cp
(600 * logit of the WS-T + MCTS win rate). The search's pruning margins assume some
eval magnitude, so the same weights can play differently at another FV_SCALE.
tanuki- found FV 16 -> 28 worth about +75 Elo for an eval trained on raw DL-teacher
values (lambda = 1), which is our setting (docs/research_survey_2026-10-07.md §3.1);
the best value moves with the training setup and cannot be read off the eval
distribution, so it is measured by games.

PRE-REGISTERED (committed before the run; no game of this task is seen):
  * eval = adopted `task109_student_64m_e24`; only the eval side's FV_SCALE changes
    (`--a-opt FV_SCALE=<fv>`); anchors unchanged (tanuki- @10,000 at its default,
    AobaNNUE @5,000 with its own build default).
  * stage 1 (standard frame, docs/evaluation_protocol.md: 100,000 nodes, seeds
    20272001-08, 800 games per anchor): FV 12, 20, 24. FV 16 = the Task 113 games of
    the same eval on the same seeds (reused, already seen; they are the baseline).
    delta(fv) = pooled (fv - 16) over the 1,600 shared openings.
  * stage 2 (extension, pre-registered): if the best stage-1 point estimate is at an
    edge of the grid (12 or 24), also measure the next value beyond it (9 or 32) on
    the standard frame. Best = argmax point estimate over all measured values.
  * decision:
      best != 16 and its standard-frame CI low > 0 -> confirm on FRESH seeds
        20272201-08: the best FV and FV 16 are both played (800 games per anchor each),
        pooled paired delta (best - 16):
          CI low > 0 -> FV_CONFIRMED (adoption candidate; the switch waits for approval)
          otherwise  -> FV_NOT_CONFIRMED (keep 16)
      otherwise -> FV16_HOLDS (no confirmation games).
  * gates: no illegal move / engine failure, openings identical between compared arms.
  * descriptive: every delta per anchor; the whole curve.

Interruptible (scripts/job.sh pause|resume task118).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402

TAG = "task118"
EVAL = "task109_student_64m_e24"
BASE_FV = std_frame.FV_DEFAULT
GRID = (12, 20, 24)
EXTEND = {12: 9, 24: 32}
CONFIRM_SEEDS = tuple(20272201 + i for i in range(8))
CONFIRM_PREFIX = "conf118"


def main():
    repo, data, logs, *_ = std_frame.paths()
    if not (data / "eval_dirs" / EVAL / "nn.bin").exists():
        raise FileNotFoundError(data / "eval_dirs" / EVAL / "nn.bin")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T118_PREFLIGHT_OK head={head} eval={EVAL} grid={GRID} extend={EXTEND}", flush=True)
    failures = []

    def frame(fv, seeds=std_frame.SEEDS, prefix="std"):
        rows, bad = std_frame.measure(EVAL, env, fv=fv, seeds=seeds, prefix=prefix)
        failures.extend(bad)
        return rows

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T118_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    print("=== T118 STAGE 1 (standard frame: FV 12, 20, 24 vs the FV 16 games of Task 113) ===", flush=True)
    base = frame(BASE_FV)
    curve = {}
    for fv in GRID:
        curve[fv] = compare(frame(fv), base, f"(fv{fv} - fv16)")
    best = max(curve, key=lambda f: curve[f]["pooled"]["delta_elo"])
    if best in EXTEND:
        nxt = EXTEND[best]
        print(f"=== T118 STAGE 2 (best {best} is a grid edge -> also FV {nxt}) ===", flush=True)
        curve[nxt] = compare(frame(nxt), base, f"(fv{nxt} - fv16)")
        best = max(curve, key=lambda f: curve[f]["pooled"]["delta_elo"])
    lo = curve[best]["pooled"]["ci95"][0]
    print(f"T118_BEST fv={best} delta={curve[best]['pooled']['delta_elo']:+.2f} ci_low={lo:+.2f}", flush=True)

    confirm = None
    if failures:
        verdict = "INVALID"
    elif curve[best]["pooled"]["delta_elo"] > 0 and lo > 0:
        print(f"=== T118 STAGE 3 (confirm FV {best} vs 16 on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]}) ===",
              flush=True)
        c_base = frame(BASE_FV, CONFIRM_SEEDS, CONFIRM_PREFIX)
        c_best = frame(best, CONFIRM_SEEDS, CONFIRM_PREFIX)
        confirm = compare(c_best, c_base, f"confirm (fv{best} - fv16)")
        verdict = ("INVALID" if failures else "FV_CONFIRMED" if confirm["pooled"]["ci95"][0] > 0
                   else "FV_NOT_CONFIRMED")
    else:
        verdict = "FV16_HOLDS"

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 118, "git": {"head": head, "pushed": False}, "eval": EVAL,
                               "curve": {str(k): v for k, v in sorted(curve.items())}, "best_fv": best,
                               "confirm": confirm, "failures": failures, "verdict": verdict}, indent=2),
                   encoding="utf-8")
    print(f"T118_VERDICT verdict={verdict} best_fv={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
