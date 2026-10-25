#!/usr/bin/env python3
"""Task 122: learning rate above 3e-5 (32M screening of 6e-5 and 1e-4, then 64M at 12 epochs).

Task 120 (32M) and Task 121 (64M, adopted) showed lr 1e-5 -> 3e-5 helps (+30.9 at 32M;
+42.5 / confirmed +70.1 at 64M), and at 3e-5 twelve epochs matched the old 24-epoch
eval. This asks whether the step is still too small.

PRE-REGISTERED (committed before the run; no result is seen):
  * stage 1 (32M screening, one variable = lr): 6e-5 and 1e-4, everything else = Task
    120 (`caches/shared_32m`, search labels `work/task104/rescore_search`, Task 106
    mask, 098 NNUE_ARGS + --augment-mirror, 12 epochs). Control = `task120_student_lr3e5`
    (3e-5; standard-frame games reused). Standard anchor frame per
    docs/evaluation_protocol.md, pooled (lr - 3e-5).
      best = the larger point estimate. best CI low > 0 -> LR_HIGHER_HELPS(best) -> stage 2;
      both CI high < 0 -> LR_HIGHER_HURTS; otherwise LR_HIGHER_NEUTRAL (stop).
  * stage 2 (only after LR_HIGHER_HELPS): 64M at the best lr for 12 epochs (Task 121
    data / labels / mask / mmap). Compared with the adopted eval
    `task121_student_64m_lr3e5_e24` (3e-5, 24 epochs) on the standard frame:
      CI low > 0 -> confirm on FRESH seeds 20272301-08 (both evals, 800 games per
        anchor each): CI low > 0 -> ADOPT_CANDIDATE (swap waits for approval) /
        otherwise NOT_CONFIRMED
      CI high < 0 -> HIGHER_64M_E12_WORSE / otherwise HIGHER_64M_E12_NEUTRAL
    (12 vs 24 epochs is deliberate: a win would be stronger AND half the training time;
    a loss does not rule out the best lr at 24 epochs.)
  * gates: 0cp GREEN for each student, no illegal move / engine failure, openings identical.
  * descriptive: val losses, best epochs, first-layer weight size.

Interruptible (scripts/job.sh pause|resume task122).
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

TAG = "task122"
CONTROL = "task120_student_lr3e5"
ADOPTED = "task121_student_64m_lr3e5_e24"
LRS = ("6e-5", "1e-4")
S32 = {"cache": "caches/shared_32m", "labels": "work/task104/rescore_search",
       "mask": "work/task106/mask/train_keep.npy", "mmap": False}
S64 = {"cache": "caches/shared_64m", "labels": "work/task107/rescore_search",
       "mask": "work/task107/mask/train_keep.npy", "mmap": True}
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
CONFIRM_SEEDS = tuple(20272301 + i for i in range(8))
CONFIRM_PREFIX = "conf122"


def lr_args(lr):
    args = [lr if a == "1e-5" else a for a in NNUE_ARGS]
    assert args.count(lr) == 1 and "1e-5" not in args
    return args


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    for s in (S32, S64):
        for path in (data / s["cache"] / "meta.json", data / s["labels"] / "meta.json", data / s["mask"]):
            if not path.exists():
                raise FileNotFoundError(path)
    for name in (CONTROL, ADOPTED):
        if not (data / "eval_dirs" / name / "nn.bin").exists():
            raise FileNotFoundError(name)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T122_PREFLIGHT_OK head={head} control={CONTROL} adopted={ADOPTED} lrs={LRS}", flush=True)
    failures = []
    stop_args = ["--stop-file", t106.STOP] if t106.STOP else []

    def student(name, lr, setup, epochs):
        model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
        if not (model_dir / "history.json").exists():
            print(f"=== T122 STAGE train {name} (lr {lr}, {setup['cache']}, {epochs} epochs) ===", flush=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            n_train = json.loads((data / setup["cache"] / "meta.json").read_text(encoding="utf-8"))["n_train"]
            t106.run([torch_py, "-m", "trainer.train_nnue", data / setup["cache"], "--out", model_dir, *lr_args(lr),
                      "--epochs", epochs, "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                      "--eval-npy", data / setup["labels"], "--limit-train", n_train,
                      "--row-mask", data / setup["mask"], *(["--mmap"] if setup["mmap"] else []),
                      "--checkpoint-minutes", "10", "--resume", *stop_args],
                     logs / f"{name}.log", cwd=repo, env=env)
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                      "--out", model_dir / "nnue_best.bnnq"], logs / f"{name}_export.log", cwd=repo, env=env)
            t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                      "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                     logs / f"{name}_yane.log", cwd=repo, env=env)
        gate = logs / f"{name}_0cp.log"
        if not gate.exists():
            t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
        text = gate.read_text(encoding="utf-8")
        if "RED" in text or text.count("GREEN") < 2:
            failures.append(f"0cp gate RED {name}")
        hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
        print(f"T122_TRAINED {name} best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f}",
              flush=True)
        return {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs", "lr")}

    def frame(name, seeds=std_frame.SEEDS, prefix="std"):
        rows, bad = std_frame.measure(name, env, seeds=seeds, prefix=prefix)
        failures.extend(bad)
        return rows

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T122_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    # ---- stage 1: 32M screening ----
    ctrl = frame(CONTROL)
    stage1, training = {}, {}
    for lr in LRS:
        name = f"{TAG}_student_32m_lr{lr}"
        training[name] = student(name, lr, S32, 12)
        stage1[lr] = compare(frame(name), ctrl, f"(lr{lr} - lr3e-5) 32M")
    best = max(stage1, key=lambda k: stage1[k]["pooled"]["delta_elo"])
    lo = stage1[best]["pooled"]["ci95"][0]
    if failures:
        v1 = "INVALID"
    elif lo > 0:
        v1 = "LR_HIGHER_HELPS"
    elif all(stage1[k]["pooled"]["ci95"][1] < 0 for k in LRS):
        v1 = "LR_HIGHER_HURTS"
    else:
        v1 = "LR_HIGHER_NEUTRAL"
    print(f"T122_STAGE1 verdict={v1} best={best}", flush=True)

    # ---- stage 2: 64M at the best lr, 12 epochs ----
    stage2, confirm, v2 = None, None, None
    if v1 == "LR_HIGHER_HELPS":
        name = f"{TAG}_student_64m_lr{best}_e12"
        training[name] = student(name, best, S64, 12)
        stage2 = compare(frame(name), frame(ADOPTED), f"(64M lr{best} e12 - adopted)")
        lo2, hi2 = stage2["pooled"]["ci95"]
        if failures:
            v2 = "INVALID"
        elif lo2 > 0:
            print(f"=== T122 STAGE confirm on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]} ===", flush=True)
            confirm = compare(frame(name, CONFIRM_SEEDS, CONFIRM_PREFIX), frame(ADOPTED, CONFIRM_SEEDS, CONFIRM_PREFIX),
                              f"confirm (64M lr{best} e12 - adopted)")
            v2 = ("INVALID" if failures else "ADOPT_CANDIDATE" if confirm["pooled"]["ci95"][0] > 0
                  else "NOT_CONFIRMED")
        elif hi2 < 0:
            v2 = "HIGHER_64M_E12_WORSE"
        else:
            v2 = "HIGHER_64M_E12_NEUTRAL"

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 122, "git": {"head": head, "pushed": False}, "training": training,
                               "stage1": stage1, "stage1_verdict": v1, "best_lr": best,
                               "stage2": stage2, "confirm": confirm, "stage2_verdict": v2,
                               "failures": failures}, indent=2), encoding="utf-8")
    print(f"T122_VERDICT stage1={v1} best={best} stage2={v2} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
