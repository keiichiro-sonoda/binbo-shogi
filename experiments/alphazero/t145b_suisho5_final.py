#!/usr/bin/env python3
"""Task 145b: re-measure the Suisho5 student with its FINAL (epoch 12) weights.

Task 145 exported the best epoch by val loss, but val labels are our WS-T labels, which the Suisho5-labelled run
moves away from: the val loss rose every epoch (116,327 -> 124,591) while the train loss fell (111,140 -> 67,020),
so epoch 1 was exported -- an under-trained student (-90 vs task129). This is a design error of Task 145's
pre-registration (it said val loss is not comparable but still used it to pick the epoch). Here nnue_final.pt
(epoch 12, as for every fine-tune) is exported as task145_suisho5_student_final and compared on the same three new
seed sets with task129 and task142_finetune (descriptive, like Task 145).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402

OWN, ADOPTED = "task129_student_64m_qsrel_e12", "task142_finetune"
NAME = "task145_suisho5_student_final"


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    env = dict(os.environ, PYTHONPATH=str(repo))
    failures = export(NAME, data / "models/task145_suisho5_student", "nnue_final.pt", data, logs, torch_py, repo, env)
    frames = {}
    for ev in (OWN, ADOPTED, NAME):
        frames[ev], bad = std_frame.measure_sets(ev, env)
        failures += bad
    results = {}
    for b in (OWN, ADOPTED):
        total, per, per_set, bad = std_frame.delta_sets(frames[NAME], frames[b], f"({NAME} - {b})")
        failures += bad
        results[f"{NAME} - {b}"] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T145B_DELTA ({NAME} - {b}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    (logs / "task145b_results.json").write_text(json.dumps({"results": results, "failures": failures}, indent=2),
                                               encoding="utf-8")
    print(f"T145B_DONE failures={failures}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
