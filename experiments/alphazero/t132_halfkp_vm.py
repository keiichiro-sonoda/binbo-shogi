#!/usr/bin/env python3
"""Task 132: HalfKP_vm as weight tying (a king on files 6-9 shares its mirror's row), 32M.

docs/research_survey_2026-10-12.md proposal 3. YaneuraOu's HalfKP_vm mirrors a
perspective left-right when its king stands on files 6-9 (tanuki- 2021: +67.5 over HalfKP,
5,000 games, old trainer without factorizer / mirror augmentation). It is exactly HalfKP
with the tie W[k, p] = W[mirror(k), mirror(p)], so it is trained as a tie
(`trainer/kp.vm_mirror_map`, composed into the Task 059 collapse map) and exported as
plain HalfKP with mirrored copies: the YaneuraOu build, the Rust forward and the 0cp
gate are unchanged. Independent rows 124,254 -> 69,030 (YaneuraOu HalfKP_vm: 69,660).

Checks before pre-registration (300k rows, 1 epoch, --factorize --vm): export and 0cp
gate GREEN (32/32 exact); on 2,054 positions with neither king on the 5-file, the
exported net gives the SAME eval for a position and its left-right mirror (2,054/2,054
exact; the non-vm Task 128 net: 33/2,054, max |d| 1,032 cp).

PRE-REGISTERED (committed before the run; no result is seen):
  * one variable vs the control `task128_student_qsrelc` (Task 128 arm C; standard-frame
    games on the Task 127 yardstick reused): --vm. Same cache `caches/shared_32m_qs119`,
    labels `work/task128/labels`, no row mask, 098 NNUE_ARGS with lr 6e-5, --factorize,
    --augment-mirror (kept: one variable), 12 epochs.
  * yardstick = Task 127 (tanuki- @20,000 / AobaNNUE @10,000, 100,000 nodes, seeds
    20272001-08, 800 games per anchor), pooled.
  * decision (vm - control): CI low > 0 -> VM_HELPS (next: 64M vs the adopted eval) /
    CI high < 0 -> VM_HURTS / otherwise VM_NEUTRAL.
  * gates: 0cp GREEN; mirror symmetry exact on the 2,054 check positions; no illegal move
    / engine failure; openings identical.
  * descriptive: val loss (same labels as the control, comparable), best epoch.

Interruptible (scripts/job.sh pause|resume task132).
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
from t098_pipeline import NNUE_ARGS  # noqa: E402

TAG = "task132"
CONTROL = "task128_student_qsrelc"
CACHE, LABELS = "caches/shared_32m_qs119", "work/task128/labels"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
VM_ARGS = ["--vm"]


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    name = f"{TAG}_student_vm"
    model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / "eval_dirs" / CONTROL / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T132_PREFLIGHT_OK head={head} control={CONTROL} anchors={std_frame.ANCHORS} args={VM_ARGS}", flush=True)
    failures = []

    if not (model_dir / "history.json").exists():
        print("=== T132 STAGE train (32M, HalfKP_vm tie) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS, *VM_ARGS,
                  "--factorize", "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", LIMIT_TRAIN, "--checkpoint-minutes", "10",
                  "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                  "--out", model_dir / "nnue_best.bnnq"], logs / f"{TAG}_export.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                  "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 logs / f"{TAG}_yane.log", cwd=repo, env=env)
    gate = logs / f"{TAG}_0cp.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        failures.append("0cp gate RED")
    sym = {}
    for side in ("a", "m"):
        out_txt = logs / f"{TAG}_mirror_{side}.txt"
        subprocess.run([repo / "target/release/binbo-shogi", "nnue-eval", model_dir / "nnue_best.bnnq",
                        repo / f"tools/vm_mirror_check_{side}.sfen", out_txt],
                       cwd=repo, env=env, check=True, capture_output=True)
        sym[side] = [float(x) for x in out_txt.read_text(encoding="utf-8").split()]
    n_exact = sum(x == y for x, y in zip(sym["a"], sym["m"]))
    print(f"T132_MIRROR exact={n_exact}/{len(sym['a'])}", flush=True)
    if len(sym["a"]) != len(sym["m"]) or n_exact != len(sym["a"]):
        failures.append(f"mirror symmetry {n_exact}/{len(sym['a'])}")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    print(f"T132_TRAINED best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f}", flush=True)

    print("=== T132 STAGE standard frame ===", flush=True)
    ctrl, bad = std_frame.measure(CONTROL, env)
    failures += bad
    cand, bad = std_frame.measure(name, env)
    failures += bad
    pooled, per, bad = std_frame.delta(cand, ctrl, "(vm - task128 C)")
    failures += bad
    print(f"T132_FRAME pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
          + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    lo, hi = pooled["ci95"]
    verdict = ("INVALID" if failures else "VM_HELPS" if lo > 0 else "VM_HURTS" if hi < 0
               else "VM_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 132, "git": {"head": head, "pushed": False}, "args": VM_ARGS,
                               "anchors": std_frame.ANCHORS,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "frame": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T132_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
