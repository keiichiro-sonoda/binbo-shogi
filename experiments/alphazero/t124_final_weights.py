#!/usr/bin/env python3
"""Task 124: play the last-epoch weights instead of the best-val-loss weights.

The trainer exports the epoch with the lowest val loss (`nnue_best.pt`). Val is 50,000
positions and moves +-500 between epochs, as much as the gains being selected for, and
val-loss differences of this size did not predict strength (Task 117). In Task 123 the
64M lr 6e-5 run never beat epoch 10's val loss (46,598) in epochs 13-24, so its export
was the epoch-10 weights (= Task 122 e12, -12.3 vs the adopted eval); its epoch-24
weights (`nnue_final.pt`, val 46,652) were never played.

PRE-REGISTERED (committed before the run; no result is seen):
  * candidate F6 = epoch-24 weights of `task123_student_64m_lr6e-5_e24` (lr 6e-5).
    PRIMARY = F6 - adopted (`task121_student_64m_lr3e5_e24`), standard anchor frame per
    docs/evaluation_protocol.md (adopted games reused):
      CI low > 0 -> confirm on FRESH seeds 20272301-08 (both evals, 800 games per anchor
        each): CI low > 0 -> ADOPT_CANDIDATE (the swap waits for approval) / otherwise
        NOT_CONFIRMED
      CI high < 0 -> FINAL6_WORSE / otherwise FINAL6_NEUTRAL
  * descriptive: F6 - best-val weights of the same run (epoch 10 = Task 122 e12), and
    F3 = epoch-24 weights of the adopted run (lr 3e-5; its export is epoch 22) - adopted.
  * gates: 0cp GREEN for each export, no illegal move / engine failure, openings identical.

Interruptible (scripts/job.sh pause|resume task124).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402

TAG = "task124"
ADOPTED = "task121_student_64m_lr3e5_e24"
BEST6 = "task122_student_64m_lr6e-5_e12"   # = the best-val (epoch 10) weights of the 6e-5 run
FINALS = {"F6": "task123_student_64m_lr6e-5_e24", "F3": ADOPTED}   # arm -> model dir whose nnue_final.pt is played
CONFIRM_SEEDS = tuple(20272301 + i for i in range(8))
CONFIRM_PREFIX = "conf124"


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    for model in FINALS.values():
        if not (data / "models" / model / "nnue_final.pt").exists():
            raise FileNotFoundError(data / "models" / model / "nnue_final.pt")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T124_PREFLIGHT_OK head={head} adopted={ADOPTED} finals={FINALS}", flush=True)
    failures = []

    def export_final(arm, model):
        name = f"{TAG}_{arm.lower()}_{model}_final"
        model_dir, eval_dir = data / "models" / model, data / "eval_dirs" / name
        bnnq = model_dir / "nnue_final.bnnq"
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_final.pt", "--out", bnnq],
                     logs / f"{name}_export.log", cwd=repo, env=env)
            t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", bnnq,
                      "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                     logs / f"{name}_yane.log", cwd=repo, env=env)
        gate = logs / f"{name}_0cp.log"
        if not gate.exists():
            t106.run(["bash", "scripts/binbo_to_yane.sh", bnnq], gate, cwd=repo, env=env)
        text = gate.read_text(encoding="utf-8")
        if "RED" in text or text.count("GREEN") < 2:
            failures.append(f"0cp gate RED {name}")
        return name

    def frame(name, seeds=std_frame.SEEDS, prefix="std"):
        rows, bad = std_frame.measure(name, env, seeds=seeds, prefix=prefix)
        failures.extend(bad)
        return rows

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T124_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    names = {arm: export_final(arm, model) for arm, model in FINALS.items()}
    adopted = frame(ADOPTED)
    f6 = frame(names["F6"])
    res = {"F6-adopted": compare(f6, adopted, "(F6 final 6e-5 - adopted)"),
           "F6-best6": compare(f6, frame(BEST6), "(F6 final - best-val epoch 10)")}
    lo, hi = res["F6-adopted"]["pooled"]["ci95"]
    confirm = None
    if failures:
        verdict = "INVALID"
    elif lo > 0:
        print(f"=== T124 STAGE confirm on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]} ===", flush=True)
        confirm = compare(frame(names["F6"], CONFIRM_SEEDS, CONFIRM_PREFIX),
                          frame(ADOPTED, CONFIRM_SEEDS, CONFIRM_PREFIX), "confirm (F6 - adopted)")
        verdict = ("INVALID" if failures else "ADOPT_CANDIDATE" if confirm["pooled"]["ci95"][0] > 0
                   else "NOT_CONFIRMED")
    elif hi < 0:
        verdict = "FINAL6_WORSE"
    else:
        verdict = "FINAL6_NEUTRAL"
    res["F3-adopted"] = compare(frame(names["F3"]), adopted, "(F3 final 3e-5 epoch 24 - adopted epoch 22)")

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 124, "git": {"head": head, "pushed": False}, "evals": names,
                               "frame": res, "confirm": confirm, "failures": failures, "verdict": verdict},
                              indent=2), encoding="utf-8")
    print(f"T124_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
