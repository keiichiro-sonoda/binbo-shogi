#!/usr/bin/env python3
"""Task 171: local-only comparison of the real Suisho5 eval with our Suisho5-data student and the adopted eval.

Suisho5 nn.bin from the YaneuraOu GitHub release 'suisho5' (Suisho5.7z, sha256 6734e3a3...; nn.bin sha256 768068f0...),
no license text in the release; used ONLY as a local opponent measurement (not as teacher, yardstick, or in any
submission). Same frame as every candidate: s280/s281/s282 x tanuki-@20k + AobaNNUE@10k, candidate @100k nodes,
FV_SCALE 24 (the release's recommended value). Reported vs task154_R_result30, task145_suisho5_student_final, task166_S_R2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import std_frame  # noqa: E402

repo, *_ = std_frame.paths()
env = dict(os.environ, PYTHONPATH=str(repo))
s5, bad = std_frame.measure_sets("suisho5_local", env, fv=24)
for other in ("task154_R_result30", "task145_suisho5_student_final", "task166_S_R2"):
    o, _ = std_frame.measure_sets(other, env)
    t, per, ps, b2 = std_frame.delta_sets(s5, o, f"(suisho5 - {other})")
    print(f"T171_DELTA suisho5_local - {other}: {t['delta_elo']:+.2f} ci={[round(v, 2) for v in t['ci95']]} "
          + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
print(f"T171_DONE failures={bad}")
