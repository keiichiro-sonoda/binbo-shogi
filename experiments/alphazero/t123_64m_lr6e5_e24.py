#!/usr/bin/env python3
"""Task 123: continue the 64M lr 6e-5 student from 12 to 24 epochs.

Task 122: at 32M, lr 6e-5 beat 3e-5 by +46.3 [+20.6, +72.5]; at 64M, 6e-5 for 12 epochs
tied the adopted eval `task121_student_64m_lr3e5_e24` (3e-5, 24 epochs): -12.3
[-41.4, +16.5]. At 3e-5, going 12 -> 24 epochs added +37.8 (Task 121).

PRE-REGISTERED (committed before the run; no result is seen):
  * the Task 122 64M 6e-5 run (`task122_student_64m_lr6e-5_e12`, train_state.pt after
    epoch 12) is copied and resumed to 24 epochs (resume is bit-exact; constant LR, so
    this equals a 24-epoch run from scratch) -> `task123_student_64m_lr6e-5_e24`.
    Data / labels / mask / mmap as Task 121.
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, 100,000 nodes, seeds 20272001-08, 800 games per anchor); the adopted eval's
    games are reused.
  * PRIMARY = e24 - adopted, pooled:
      CI low > 0 -> confirm on FRESH seeds 20272301-08 (both evals, 800 games per anchor
        each): CI low > 0 -> ADOPT_CANDIDATE (the swap waits for approval) / otherwise
        NOT_CONFIRMED
      CI high < 0 -> LR6E5_E24_WORSE / otherwise LR6E5_E24_NEUTRAL
  * descriptive: e24 - e12 (Task 122 student), val losses, best epochs. If the best
    epoch is <= 12, e24 is the e12 model and is not replayed (verdict from e12's games).
  * gates: 0cp GREEN, no illegal move / engine failure, openings identical.

Interruptible (scripts/job.sh pause|resume task123).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS  # noqa: E402

TAG = "task123"
ADOPTED = "task121_student_64m_lr3e5_e24"
E12 = "task122_student_64m_lr6e-5_e12"
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
CACHE, LABELS, MASK = "caches/shared_64m", "work/task107/rescore_search", "work/task107/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
CONFIRM_SEEDS = tuple(20272301 + i for i in range(8))
CONFIRM_PREFIX = "conf123"


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    for path in (data / CACHE / "meta.json", data / LABELS / "meta.json", data / MASK,
                 data / "eval_dirs" / ADOPTED / "nn.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    n_train = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))["n_train"]
    print(f"T123_PREFLIGHT_OK head={head} adopted={ADOPTED} lr={LR} n_train={n_train}", flush=True)
    failures = []
    stop_args = ["--stop-file", t106.STOP] if t106.STOP else []

    def train(model_dir, epochs, log_name):
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS,
                  "--epochs", epochs, "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", n_train, "--row-mask", data / MASK, "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / log_name, cwd=repo, env=env)

    def export(name):
        model_dir, eval_dir = data / "models" / name, data / "eval_dirs" / name
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
        return json.loads((model_dir / "history.json").read_text(encoding="utf-8"))

    e12, e24 = E12, f"{TAG}_student_64m_lr6e-5_e24"
    d12, d24 = data / "models" / e12, data / "models" / e24
    if not (d12 / "train_state.pt").exists() or not (d12 / "history.json").exists():
        raise FileNotFoundError(d12 / "train_state.pt")
    if not (d24 / "history.json").exists():
        if not d24.exists():
            tmp = d24.with_name(d24.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(d12, tmp)
            for name in ("history.json", "nnue_final.pt", "nnue_best.bnnq"):   # written again by the resumed run
                (tmp / name).unlink(missing_ok=True)
            tmp.rename(d24)
            print(f"T123_COPIED {d12} -> {d24} (Task 122 model untouched)", flush=True)
        print("=== T123 STAGE train epochs 13..24 (resume) ===", flush=True)
        train(d24, 24, f"{TAG}_student_e24.log")

    h12, h24 = export(e12), export(e24)
    print(f"T123_TRAINED e12 best_epoch={h12['best_epoch']} val={h12['best_val_loss']:.1f} | "
          f"e24 best_epoch={h24['best_epoch']} val={h24['best_val_loss']:.1f}", flush=True)

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T123_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    print("=== T123 STAGE standard frame ===", flush=True)
    adopted, bad = std_frame.measure(ADOPTED, env)
    failures += bad
    r12, bad = std_frame.measure(e12, env)
    failures += bad
    same = h24["best_epoch"] <= 12
    if same:
        r24 = r12
    else:
        r24, bad = std_frame.measure(e24, env)
        failures += bad
    res = {"e24-adopted": compare(r24, adopted, "(e24 - adopted)"),
           "e12-adopted": compare(r12, adopted, "(e12 - adopted)")}
    if not same:
        res["e24-e12"] = compare(r24, r12, "(e24 - e12)")

    lo, hi = res["e24-adopted"]["pooled"]["ci95"]
    confirm = None
    if failures:
        verdict = "INVALID"
    elif lo > 0:
        print(f"=== T123 STAGE confirm on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]} ===", flush=True)
        c_adopted, bad = std_frame.measure(ADOPTED, env, seeds=CONFIRM_SEEDS, prefix=CONFIRM_PREFIX)
        failures += bad
        c_cand, bad = std_frame.measure(e12 if same else e24, env, seeds=CONFIRM_SEEDS, prefix=CONFIRM_PREFIX)
        failures += bad
        confirm = compare(c_cand, c_adopted, "confirm (e24 - adopted)")
        verdict = ("INVALID" if failures else "ADOPT_CANDIDATE" if confirm["pooled"]["ci95"][0] > 0
                   else "NOT_CONFIRMED")
    elif hi < 0:
        verdict = "LR6E5_E24_WORSE"
    else:
        verdict = "LR6E5_E24_NEUTRAL"

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 123, "git": {"head": head, "pushed": False}, "lr": LR,
                               "training": {"e12": {k: h12.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                                            "e24": {k: h24.get(k) for k in ("best_epoch", "best_val_loss", "epochs")}},
                               "frame": res, "confirm": confirm, "e24_is_e12": same,
                               "failures": failures, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T123_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
