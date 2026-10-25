#!/usr/bin/env python3
"""Task 126: the 64M student with the factorizer (lr 6e-5, 24 epochs, 12-epoch point kept).

Task 125: the factorizer (king-independent P table, folded at export) beat the best
32M student by +40.8 [+12.7, +69.3] and cut val loss 13% (49,479 -> 42,854), still
improving at epoch 12. Without it, 64M at 6e-5 plateaued around epoch 10 (Tasks
122-124) at the strength of the adopted eval.

PRE-REGISTERED (committed before the run; no result is seen):
  * candidate = 64M (`caches/shared_64m`, `work/task107/rescore_search`, Task 107 mask,
    --mmap) + --factorize, lr 6e-5, 098 NNUE_ARGS + --augment-mirror. Epochs 1..12 into
    `<tag>_student_64m_factor_e12`, then resumed to 24 in `<tag>_student_64m_factor_e24`
    (constant LR, bit-exact resume = one 24-epoch run).
  * standard anchor frame per docs/evaluation_protocol.md (tanuki- @10,000, AobaNNUE
    @5,000, 100,000 nodes, seeds 20272001-08, 800 games per anchor); the adopted eval
    `task121_student_64m_lr3e5_e24` games are reused.
  * PRIMARY = e24 - adopted, pooled:
      CI low > 0 -> confirm on FRESH seeds 20272301-08 (both evals, 800 games per anchor
        each): CI low > 0 -> ADOPT_CANDIDATE (the swap waits for approval) / otherwise
        NOT_CONFIRMED
      CI high < 0 -> FACTOR64_WORSE / otherwise FACTOR64_NEUTRAL
  * descriptive: e12 - adopted, e24 - e12, val losses, best epochs. If the 24-epoch
    run's best epoch is <= 12, e24 is the e12 model and is not replayed.
  * gates: 0cp GREEN for each export, no illegal move / engine failure, openings identical.

Interruptible (scripts/job.sh pause|resume task126).
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

TAG = "task126"
ADOPTED = "task121_student_64m_lr3e5_e24"
LR = "6e-5"
LR_ARGS = [LR if a == "1e-5" else a for a in NNUE_ARGS]
assert LR_ARGS.count(LR) == 1 and "1e-5" not in LR_ARGS
CACHE, LABELS, MASK = "caches/shared_64m", "work/task107/rescore_search", "work/task107/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
CONFIRM_SEEDS = tuple(20272301 + i for i in range(8))
CONFIRM_PREFIX = "conf126"


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
    print(f"T126_PREFLIGHT_OK head={head} adopted={ADOPTED} lr={LR} n_train={n_train}", flush=True)
    failures = []
    stop_args = ["--stop-file", t106.STOP] if t106.STOP else []

    def train(model_dir, epochs, log_name):
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *LR_ARGS, "--factorize",
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

    e12, e24 = f"{TAG}_student_64m_factor_e12", f"{TAG}_student_64m_factor_e24"
    d12, d24 = data / "models" / e12, data / "models" / e24
    if not (d12 / "history.json").exists():
        print("=== T126 STAGE train epochs 1..12 (64M, lr 6e-5, factorizer) ===", flush=True)
        d12.mkdir(parents=True, exist_ok=True)
        train(d12, 12, f"{TAG}_student_e12.log")
    if not (d24 / "history.json").exists():
        if not d24.exists():
            tmp = d24.with_name(d24.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(d12, tmp)
            for name in ("history.json", "nnue_final.pt", "nnue_best.bnnq"):   # written again by the resumed run
                (tmp / name).unlink(missing_ok=True)
            tmp.rename(d24)
        print("=== T126 STAGE train epochs 13..24 (resume) ===", flush=True)
        train(d24, 24, f"{TAG}_student_e24.log")

    h12, h24 = export(e12), export(e24)
    print(f"T126_TRAINED e12 best_epoch={h12['best_epoch']} val={h12['best_val_loss']:.1f} | "
          f"e24 best_epoch={h24['best_epoch']} val={h24['best_val_loss']:.1f}", flush=True)

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T126_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    print("=== T126 STAGE standard frame ===", flush=True)
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
        print(f"=== T126 STAGE confirm on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]} ===", flush=True)
        c_adopted, bad = std_frame.measure(ADOPTED, env, seeds=CONFIRM_SEEDS, prefix=CONFIRM_PREFIX)
        failures += bad
        c_cand, bad = std_frame.measure(e12 if same else e24, env, seeds=CONFIRM_SEEDS, prefix=CONFIRM_PREFIX)
        failures += bad
        confirm = compare(c_cand, c_adopted, "confirm (e24 - adopted)")
        verdict = ("INVALID" if failures else "ADOPT_CANDIDATE" if confirm["pooled"]["ci95"][0] > 0
                   else "NOT_CONFIRMED")
    elif hi < 0:
        verdict = "FACTOR64_WORSE"
    else:
        verdict = "FACTOR64_NEUTRAL"

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 126, "git": {"head": head, "pushed": False}, "lr": LR, "factorize": True,
                               "training": {"e12": {k: h12.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                                            "e24": {k: h24.get(k) for k in ("best_epoch", "best_val_loss", "epochs")}},
                               "frame": res, "confirm": confirm, "e24_is_e12": same,
                               "failures": failures, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T126_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
