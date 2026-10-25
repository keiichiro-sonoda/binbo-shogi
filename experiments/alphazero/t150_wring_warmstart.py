#!/usr/bin/env python3
"""Task 150: "wringing" by CONTINUING to train the adopted eval on binbo's own depth-9 labels (warm start).

Task 147 trained from scratch on the depth-9 labels (work/task146_full/labels) and was 100 Elo below the teacher
task142. Codex's review (reports/task147_review_codex.md) found no label/alignment/unit bug, the new labels match the
yardsticks better than the old WS-T labels, and supports a warm start from the adopted weights. Usual NNUE practice
continues training the current net. A planned game-result-mixing arm was dropped before the run: only 0.6% of the
train rows carry a known Gikou2 game result (99.4% are 0 = unknown/draw), so mixing would touch almost nothing.

Arms (fixed before the run; all start from models/task142_finetune/nnue_final.pt = the adopted eval, NOT nnue_best.pt
which differs; --fast; 1 epoch; final weights; fresh output dirs, no --resume across arms):
  A: all 63.95M rows, lr 5e-6        B: all 63.95M rows, lr 1e-5
  C: a uniform random 8M-row subset (seed 20271500, --row-mask), lr 1e-5  (fewer updates: a milder change)
Note: one 64M epoch is ~4.4x the updates of the Suisho5 finish (7.24M x 2 epochs).
Decision: new seed sets s280/s281/s282 pooled, paired, each arm - task142_finetune. The best arm by point estimate
with CI low > 0 is confirmed on a fresh spaced set (std_frame.spaced_set(20285000), prefix conf150, both evals):
point > 0 -> ADOPT_CANDIDATE else NOT_CONFIRMED. No arm with CI low > 0 -> NO_BETTER (descriptive deltas kept).
Gates: 0cp GREEN, no frame failures.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402

TAG = "task150"
ADOPTED = "task142_finetune"
ARMS = {"A_lr5e-6": ("5e-6", None), "B_lr1e-5": ("1e-5", None), "C_8m_lr1e-5": ("1e-5", 8_000_000)}
CONF_PREFIX = "conf150"


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    labels = data / "work/task146_full/labels"
    start = data / "models" / ADOPTED / "nnue_final.pt"
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    failures, names = [], []
    for key, (lr, subset) in ARMS.items():
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        extra = []
        if subset:
            mask_path = work / f"mask_{subset}.npy"
            if not mask_path.exists():
                mask = np.zeros(t129.N64, dtype=bool)
                mask[np.random.default_rng(20271500).choice(t129.N64, subset, replace=False)] = True
                np.save(mask_path, mask)
            extra = ["--row-mask", mask_path]
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T150 STAGE train {key} (lr {lr}, rows {subset or t129.N64}) ===", flush=True)
            t106.run([torch_py, "-m", "trainer.train_nnue", data / t129.QS64, "--out", out, "--qat", "--qat-tie",
                      "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600",
                      "--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
                      "--lr", lr, "--epochs", "1", "--init-weights", start, "--eval-npy", labels,
                      "--limit-train", t129.N64, "--compact-cache", data / "caches/compact_qs129", *extra],
                     logs / f"{name}.log", cwd=repo, env=env)
        failures += export(name, out, "nnue_final.pt", data, logs, torch_py, repo, env)
        names.append(name)
    base, bad = std_frame.measure_sets(ADOPTED, env)
    failures += bad
    results = {}
    for name in names:
        cand, bad = std_frame.measure_sets(name, env)
        total, per, per_set, bad2 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        failures += bad + bad2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T150_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    ok = [n for n in names if results[n]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda n: results[n]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: std_frame.spaced_set(20285000)}
        cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs)
        cc, b2 = std_frame.measure_sets(best, env, sets=cs)
        ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm ({best} - {ADOPTED})")
        failures += b1 + b2 + b3
        results[best]["confirm"] = {"pooled": ct, "per_anchor": cper}
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if ct["delta_elo"] > 0 else "NOT_CONFIRMED"
        print(f"T150_CONFIRM {best} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 150, "results": results, "best": best, "failures": failures,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"T150_VERDICT verdict={verdict} best={best} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
