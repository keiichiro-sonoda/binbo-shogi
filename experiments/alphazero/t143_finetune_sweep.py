#!/usr/bin/env python3
"""Task 143: tune the Task 142 fine-tune (lr, epochs) on the same public data (nodchip/shogi_suisho5_depth9, 7.24M).

Task 142's fine-tune (from task129, 2 epochs, lr 1e-5, final weights) became the adopted eval (task142_finetune).
Here four variants of the same fine-tune, each from the SAME starting point (task129) so they are comparable with
Task 142's arm: lr 5e-6 x2, lr 2e-5 x2, lr 1e-5 x1, lr 1e-5 x4 (final weights each).

Amended before any frame was played (2026-10-19 14:10): the frames use the new spaced seed sets
std_frame.SEED_SETS (s280 / s281 / s282, 400 distinct openings each; the old sets had 57) and the confirmation set
is spaced too (20283000 + 100 i). The adopted eval plays the new sets first.

Stages: `train` (GPU, ~minutes each; runs while a floodgate test occupies the CPU) and `frames` (CPU, 3 seed sets
each, run after the floodgate test). Decision (fixed before the run; base = the adopted task142_finetune, which has
all three sets): 3-set pooled paired delta per variant; the best variant by point estimate with CI low > 0 is
confirmed on a fresh set 20273801-08 (prefix conf143, both evals): point > 0 -> ADOPT_CANDIDATE, else
NOT_CONFIRMED; no variant with CI low > 0 -> NO_BETTER. Descriptive: every variant's delta.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402

TAG = "task143"
START = "task129_student_64m_qsrel_e12"
ADOPTED = "task142_finetune"
VARIANTS = {"lr5e-6_e2": ("5e-6", 2), "lr2e-5_e2": ("2e-5", 2), "lr1e-5_e1": ("1e-5", 1), "lr1e-5_e4": ("1e-5", 4)}
CONF_SEEDS, CONF_PREFIX = std_frame.spaced_set(20283000), "conf143"  # spaced (Task 143 seed fix), not run yet
N_PUB = 7_240_673


def main():
    stage = sys.argv[1]
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    pub = data / "caches/public_suisho5_val7m"
    if stage == "train":
        for key, (lr, ep) in VARIANTS.items():
            ev, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
            if not (out / "history.json").exists():
                out.mkdir(parents=True, exist_ok=True)
                print(f"=== T143 STAGE train {key} ===", flush=True)
                t106.run([torch_py, "-m", "trainer.train_nnue", pub, "--out", out, "--qat", "--qat-tie",
                          "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600",
                          "--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
                          "--lr", lr, "--epochs", ep, "--init-weights", data / "models" / START / "nnue_best.pt",
                          "--eval-npy", data / "work/task142/labels_public", "--limit-train", N_PUB,
                          "--compact-cache", Path(str(pub) + "_compact")], logs / f"{ev}.log", cwd=repo, env=env)
            eval_dir, bnnq = data / "eval_dirs" / ev, out / "nnue_final.bnnq"
            if not (eval_dir / "nn.bin").exists():
                eval_dir.mkdir(parents=True, exist_ok=True)
                t106.run([torch_py, "-m", "trainer.export_nnue", out, "--weights", "nnue_final.pt", "--out", bnnq],
                         logs / f"{ev}_export.log", cwd=repo, env=env)
                t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", bnnq, "--template",
                          data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                         logs / f"{ev}_yane.log", cwd=repo, env=env)
            gate = logs / f"{ev}_0cp.log"
            if not gate.exists():
                t106.run(["bash", "scripts/binbo_to_yane.sh", bnnq], gate, cwd=repo, env=env)
            text = gate.read_text(encoding="utf-8")
            print(f"T143_TRAINED {key} 0cp={'GREEN' if 'RED' not in text and text.count('GREEN') >= 2 else 'RED'}",
                  flush=True)
        return 0
    base, bad = std_frame.measure_sets(ADOPTED, env)
    if bad:
        raise RuntimeError(f"adopted frame: {bad}")
    results = {}
    for key in VARIANTS:
        ev = f"{TAG}_{key}"
        text = (logs / f"{ev}_0cp.log").read_text(encoding="utf-8")
        fails = [] if "RED" not in text and text.count("GREEN") >= 2 else ["0cp gate RED"]
        cand, b1 = std_frame.measure_sets(ev, env)
        total, per, per_set, b2 = std_frame.delta_sets(cand, base, f"({ev} - adopted)")
        results[ev] = {"pooled": total, "per_anchor": per, "per_set": per_set, "failures": fails + b1 + b2}
        print(f"T143_RESULT {ev} pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              f"failures={fails + b1 + b2}", flush=True)
    ok = [e for e, r in results.items() if not r["failures"] and r["pooled"]["ci95"][0] > 0]
    verdict, best = "NO_BETTER", None
    if ok:
        best = max(ok, key=lambda e: results[e]["pooled"]["delta_elo"])
        cs = {CONF_PREFIX: CONF_SEEDS}
        cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs)
        cc, b2 = std_frame.measure_sets(best, env, sets=cs)
        ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm ({best} - adopted)")
        results[best]["confirm"] = {"pooled": ct, "per_anchor": cper}
        verdict = "INVALID" if b1 + b2 + b3 else "ADOPT_CANDIDATE" if ct["delta_elo"] > 0 else "NOT_CONFIRMED"
        print(f"T143_CONFIRM {best} pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 143, "base": ADOPTED, "results": results, "best": best, "verdict": verdict},
                              indent=2), encoding="utf-8")
    print(f"T143_VERDICT verdict={verdict} best={best} results={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
