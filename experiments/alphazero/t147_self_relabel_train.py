#!/usr/bin/env python3
"""Task 147: the first "雑巾絞り" (wringing) step -- train on our own positions labelled by binbo's own depth-9 search.

Why (docs/teacher_direction_2026-10-20.md): our labels were "Gikou2 distilled into WS-T + a shallow MCTS"; Task 142
showed deep alpha-beta labels from a strong engine are much better; binbo itself (YaneuraOu + the adopted eval
task142_finetune) is now stronger than WS-T, so its own depth-9 values should be better labels, all from our own
pipeline. Labels: Task 146 (`t146_self_relabel.py run --depth 9`, part-wise reset, internal value, side to move,
mate +-32000) for all 63,950,000 train rows of caches/shared_64m_qs129 and the 50k val rows -> work/task146_full/labels.

Arms (fixed before the labels exist; all --fast, the adopted recipe):
  W  "wring": Task 129's recipe (lr 6e-5, Factorizer, mirror, 12 epochs, best epoch by val loss -- val labels are the
     new depth-9 labels too) from scratch on our positions with the depth-9 labels.
  WF "wring + Suisho5 finish": W's weights fine-tuned exactly like task142_finetune (nodchip/shogi_suisho5_depth9
     validation file 7.24M, lr 1e-5, 2 epochs, final weights).
Decision (new seed sets s280/s281/s282 pooled, paired):
  primary (the wringing effect): W - task129_student_64m_qsrel_e12 (same positions, labels WS-T vs binbo depth 9).
  adoption: WF - task142_finetune: CI low > 0 -> confirm on a fresh spaced set (std_frame.spaced_set(20284000),
     prefix conf147, both evals): point > 0 -> ADOPT_CANDIDATE else NOT_CONFIRMED; CI high < 0 -> WORSE; else NEUTRAL.
  descriptive: W - task142_finetune.
Gates: labels complete (assemble manifest), 0cp GREEN, no frame failures.
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

TAG = "task147"
OWN, ADOPTED = "task129_student_64m_qsrel_e12", "task142_finetune"
W_NAME, WF_NAME = f"{TAG}_wring", f"{TAG}_wring_finish"
CONF_PREFIX = "conf147"
N_PUB = 7_240_673


def export(name, model_dir, weights, data, logs, torch_py, repo, env):
    eval_dir, bnnq = data / "eval_dirs" / name, model_dir / weights.replace(".pt", ".bnnq")
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", weights, "--out", bnnq],
                 logs / f"{name}_export.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", bnnq, "--template",
                  data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 logs / f"{name}_yane.log", cwd=repo, env=env)
    gate = logs / f"{name}_0cp.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", bnnq], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    return [] if "RED" not in text and text.count("GREEN") >= 2 else [f"{name}: 0cp gate RED"]


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    labels = data / "work/task146_full/labels"
    if not (labels / "meta.json").exists():
        raise FileNotFoundError(f"{labels}/meta.json (run Task 146 assemble first)")
    common = ["--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
              "--checkpoint-minutes", "10", "--resume"]
    w_dir = data / "models" / W_NAME
    if not (w_dir / "history.json").exists():
        w_dir.mkdir(parents=True, exist_ok=True)
        print("=== T147 STAGE train W (own positions, binbo depth-9 labels) ===", flush=True)
        t106.run([torch_py, "-m", "trainer.train_nnue", data / t129.QS64, "--out", w_dir, *t129.LR_ARGS, *common,
                  "--epochs", t129.EPOCHS, "--eval-npy", labels, "--limit-train", t129.N64,
                  "--compact-cache", data / "caches/compact_qs129"], logs / f"{W_NAME}.log", cwd=repo, env=env)
    pub = data / "caches/public_suisho5_val7m"
    wf_dir = data / "models" / WF_NAME
    if not (wf_dir / "history.json").exists():
        wf_dir.mkdir(parents=True, exist_ok=True)
        print("=== T147 STAGE finish WF (Suisho5 7.24M, lr 1e-5, 2 epochs) ===", flush=True)
        t106.run([torch_py, "-m", "trainer.train_nnue", pub, "--out", wf_dir, "--qat", "--qat-tie", "--feature-dim",
                  "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", *common, "--lr", "1e-5",
                  "--epochs", "2", "--init-weights", w_dir / "nnue_best.pt", "--eval-npy",
                  data / "work/task142/labels_public", "--limit-train", N_PUB, "--compact-cache",
                  Path(str(pub) + "_compact")], logs / f"{WF_NAME}.log", cwd=repo, env=env)
    failures = export(W_NAME, w_dir, "nnue_best.pt", data, logs, torch_py, repo, env)
    failures += export(WF_NAME, wf_dir, "nnue_final.pt", data, logs, torch_py, repo, env)
    frames = {}
    for ev in (OWN, ADOPTED, W_NAME, WF_NAME):
        frames[ev], bad = std_frame.measure_sets(ev, env)
        failures += bad
    results = {}
    for a, b in ((W_NAME, OWN), (WF_NAME, ADOPTED), (W_NAME, ADOPTED)):
        total, per, per_set, bad = std_frame.delta_sets(frames[a], frames[b], f"({a} - {b})")
        failures += bad
        results[f"{a} - {b}"] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T147_DELTA ({a} - {b}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    lo, hi = results[f"{WF_NAME} - {ADOPTED}"]["pooled"]["ci95"]
    verdict = "INVALID" if failures else "WORSE" if hi < 0 else "NEUTRAL"
    if not failures and lo > 0:
        cs = {CONF_PREFIX: std_frame.spaced_set(20284000)}
        cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs)
        cc, b2 = std_frame.measure_sets(WF_NAME, env, sets=cs)
        ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm ({WF_NAME} - {ADOPTED})")
        failures += b1 + b2 + b3
        results["confirm"] = {"pooled": ct, "per_anchor": cper}
        verdict = "INVALID" if failures else "ADOPT_CANDIDATE" if ct["delta_elo"] > 0 else "NOT_CONFIRMED"
        print(f"T147_CONFIRM pooled={ct['delta_elo']:+.2f} ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 147, "results": results, "failures": failures, "verdict": verdict}, indent=2),
                   encoding="utf-8")
    print(f"T147_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
