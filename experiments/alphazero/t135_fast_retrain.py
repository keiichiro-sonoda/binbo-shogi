#!/usr/bin/env python3
"""Task 135 confirmation: retrain the adopted eval with the fast trainer path and check it plays the same.

The fast path (`train_nnue --fast --compact-cache`, Task 135) gives bit-exact results where it keeps
the float summation order and, where it does not (touched-rows FT, fused Adam), drifts no more than
reversing the feature order does (experiments/alphazero/t135_fast_equiv.py, 2,000 GPU steps). Drift of
that kind changes the trained weights like a different seed would, so the remaining question is only
whether the product is as strong. One variable: the trainer path. Everything else is the adopted
recipe (Task 129: caches/shared_64m_qs129 + compact_qs129, labels work/task129/labels, 63.95M rows,
lr 6e-5, Factorizer, mirror, 12 epochs, same seed; constants imported from t129_qs_relabel_64m).

Decision (fixed before the run): standard frame (tanuki- @20k, AobaNNUE @10k, 100k nodes, seeds
20272001-08), paired delta (fast retrain - adopted), pooled:
  CI contains 0 -> FAST_OK (switch the default recipe to --fast)
  CI high < 0   -> FAST_WORSE (do not switch; look for the cause)
  CI low > 0    -> FAST_BETTER (switch; note it, the expectation was no difference)
0cp export gate GREEN, failures -> INVALID. Also recorded: seconds per epoch (adopted run: 2,454 s).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402

TAG = "task135"
ADOPTED = "task129_student_64m_qsrel_e12"
CAND = f"{TAG}_student_fast"
LABELS = "work/task129/labels"
COMPACT = "caches/compact_qs129"


def main():
    repo, data, logs, _binbo, _yane, _anchors = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    for path in (data / t129.QS64 / "meta.json", data / LABELS / "meta.json", data / COMPACT / "compact.json",
                 data / "eval_dirs" / ADOPTED / "nn.bin", data / t129.MIRROR_CACHE.replace(".npy", "_shared.npy")):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ, PYTHONPATH=str(repo))
    print(f"T135_PREFLIGHT_OK head={head} adopted={ADOPTED}", flush=True)
    failures = []

    model_dir, eval_dir = data / "models" / CAND, data / "eval_dirs" / CAND
    student_log = logs / f"{TAG}_student.log"
    if not (model_dir / "history.json").exists():
        print("=== T135 STAGE train (Task 129 recipe, --fast) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / t129.QS64, "--out", model_dir, *t129.LR_ARGS,
                  "--factorize", "--epochs", t129.EPOCHS, "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", t129.N64,
                  "--fast", "--compact-cache", data / COMPACT,
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 student_log, cwd=repo, env=env)
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
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    ad_hist = json.loads((data / "models" / ADOPTED / "history.json").read_text(encoding="utf-8"))
    secs = [float(s) for s in re.findall(r"train_secs=([0-9.]+)", student_log.read_text(encoding="utf-8"))]
    print(f"T135_TRAINED best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f} "
          f"adopted={ad_hist['best_val_loss']:.1f} epoch_secs={[round(s) for s in secs]}", flush=True)

    print("=== T135 STAGE standard frame ===", flush=True)
    cand, bad1 = std_frame.measure(CAND, env)
    base, bad2 = std_frame.measure(ADOPTED, env)
    failures += bad1 + bad2
    pooled, per, bad = std_frame.delta(cand, base, "(fast retrain - adopted)")
    failures += bad
    lo, hi = pooled["ci95"]
    print(f"T135_DELTA pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
          + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    verdict = "INVALID" if failures else "FAST_WORSE" if hi < 0 else "FAST_BETTER" if lo > 0 else "FAST_OK"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 135, "git": {"head": head}, "anchors": std_frame.ANCHORS,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "epoch_secs": secs, "adopted_best_val_loss": ad_hist.get("best_val_loss"),
                               "primary": {"pooled": pooled, "per_anchor": per}, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T135_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
