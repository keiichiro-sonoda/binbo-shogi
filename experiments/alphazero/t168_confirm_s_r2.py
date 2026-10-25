#!/usr/bin/env python3
"""Task 168 (pre-tournament strategy review): adoption-style confirmation of task166_S_R2 (Suisho5 student + round 2 wringing,
+69.1 [+53.6, +84.7] vs the adopted task154_R_result30 on s280/s281/s282). Fresh spaced set spaced_set(20294000)
prefix conf168, vs tanuki- + AobaNNUE and vs micro (CONFIRM_ANCHORS). Both point estimates > 0 -> CONFIRMED.
Adoption itself is the author's decision (it changes the concept: public data as the base, not only a light finish)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import std_frame  # noqa: E402

CAND, BASE = "task166_S_R2", "task154_R_result30"


def main():
    _repo, _data, logs, *_ = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(_repo))
    cs = {"conf168": std_frame.spaced_set(20294000)}
    out, failures = {}, []
    for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
        b, f1 = std_frame.measure_sets(BASE, env, sets=cs, which=which)
        c, f2 = std_frame.measure_sets(CAND, env, sets=cs, which=which)
        t, per, _ps, f3 = std_frame.delta_sets(c, b, f"confirm {label} ({CAND} - {BASE})")
        failures += f1 + f2 + f3
        out[label] = {"pooled": t, "per_anchor": per}
        print(f"T168_CONFIRM {label} {t['delta_elo']:+.2f} ci={[round(v, 2) for v in t['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    ok = not failures and all(out[k]["pooled"]["delta_elo"] > 0 for k in out)
    (logs / "task168_results.json").write_text(json.dumps({"task": 168, "results": out, "failures": failures,
                                                           "verdict": "CONFIRMED" if ok else "NOT_CONFIRMED"}, indent=2))
    print(f"T168_VERDICT {'CONFIRMED' if ok else 'NOT_CONFIRMED'} failures={failures}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
