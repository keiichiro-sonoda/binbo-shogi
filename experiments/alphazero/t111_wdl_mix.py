#!/usr/bin/env python3
"""Task 111: mix the game result into the label (wdl lambda), on rows whose result is known.

Plan of 10/5 (cheap ideas before volume), item 2. Task 051 (2M positions, KP net,
gikou teacher) found result mixing hurt; it named the conditions to revisit (more
expressive net, DL-style teacher, far more data) and all three hold now. Found while
designing: gen-teacher stores draw AND unknown (max-ply, gikou crash) as result 0,
the runs run1m..run4m (7.95M rows) carry no result at all, and the val shards (run1m)
none either. So the result is mixed only into rows with a win/loss
(`--wdl-known-only`, 76.7% of the masked training rows); every other row, and all of
val, keeps the pure eval target.

PRE-REGISTERED (committed before the run; no result of either arm is seen):
  * two arms, identical except lambda; both warm-start from the adopted eval
    (`task109_student_64m_e24/nnue_best.pt`) with the Task 110 continuation (fresh
    Adam, lr 1e-5 cosine over 4 epochs, lr-min-frac 0.01), same cache/labels/mask:
      A: --objective wdl --wdl-lambda 1.0                  (win-rate space, no result)
      B: --objective wdl --wdl-lambda 0.75 --wdl-known-only (25% result on known rows)
    candidate of each arm = its val-best epoch.
  * one variable for the verdict: lambda (B vs A).
  * PRIMARY = B vs A, both at 100,000 nodes, 1600 games = 16 x 100 (seeds 20271801-16).
  * FRAME = A and B each vs tanuki- @10,000 and AobaNNUE @5,000 nodes, 400 games each
    (seeds 20271511-14); pooled delta (B - A) over the 800 shared openings.
  * descriptive only: (A - Task 110 student) pooled on the same anchors = the effect of
    the win-rate objective itself (Task 110 = the same continuation in cp space);
    (B - adopted) pooled vs the Task 109 frame games.
  * gates: 0cp GREEN for both arms, no illegal move / engine failure, openings shared.
  * decision table (B vs A):
      0 a gate failed                                   -> INVALID (non-terminal)
      1 primary CI low > 0 and pooled frame CI low > 0  -> MIX_HELPS   (adoption still needs B > adopted)
      2 primary CI high < 0                             -> MIX_HURTS
      3 otherwise                                       -> MIX_NEUTRAL

Interruptible (scripts/job.sh pause|resume task111).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS, paired_h2h_delta  # noqa: E402
from t103_absolute import conditions, paired_series_delta  # noqa: E402

TAG = "task111"
ADOPTED = "task109_student_64m_e24"
CACHE, LABELS, MASK = "caches/shared_64m", "work/task107/rescore_search", "work/task107/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
NODES = 100_000
ANCHORS = {"tanuki": 10_000, "aoba": 5_000}
PRIMARY_SEEDS = tuple(20271801 + i for i in range(16))
FRAME_SEEDS, FRAME_GAMES = (20271511, 20271512, 20271513, 20271514), 100
ARMS = {"A": ["--objective", "wdl", "--wdl-lambda", "1.0"],
        "B": ["--objective", "wdl", "--wdl-lambda", "0.75", "--wdl-known-only"]}


def main():
    repo = Path(os.path.realpath(__file__)).parents[2]
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data / "logs"))
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    aoba = data / "engines/aobannue/AobaNNUE"
    anchor_bin = {"tanuki": (yane, data / "engines/yaneuraou/eval"),
                  "aoba": (aoba / "source/YaneuraOu-by-gcc", aoba / "eval")}
    start = data / "models" / ADOPTED / "nnue_best.pt"
    for path in (binbo, yane, start, data / CACHE / "meta.json", data / LABELS / "meta.json", data / MASK):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T111_PREFLIGHT_OK head={head}", flush=True)
    failures, flags = [], []
    n_train = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))["n_train"]

    evals, hists = {}, {}
    for arm, extra in ARMS.items():
        model_dir = data / "models" / f"{TAG}_student_{arm}"
        eval_dir = data / "eval_dirs" / f"{TAG}_student_{arm}"
        if not (model_dir / "history.json").exists():
            print(f"=== T111 STAGE train arm {arm} ({' '.join(extra)}) ===", flush=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
            t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS,
                      "--epochs", "4", "--lr-schedule", "cosine", "--lr-min-frac", "0.01", "--init-weights", start,
                      *extra, "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE, "--eval-npy", data / LABELS,
                      "--limit-train", n_train, "--row-mask", data / MASK, "--mmap",
                      "--checkpoint-minutes", "10", "--resume", *stop_args],
                     logs / f"{TAG}_student_{arm}.log", cwd=repo, env=env)
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                      "--out", model_dir / "nnue_best.bnnq"], logs / f"{TAG}_export_{arm}.log", cwd=repo, env=env)
            t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                      "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                     logs / f"{TAG}_yane_{arm}.log", cwd=repo, env=env)
        gate = logs / f"{TAG}_0cp_{arm}.log"
        if not gate.exists():
            t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
        text = gate.read_text(encoding="utf-8")
        if "RED" in text or text.count("GREEN") < 2:
            failures.append(f"0cp gate RED for arm {arm}")
        hists[arm] = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
        evals[arm] = eval_dir
        print(f"T111_STUDENT_READY arm={arm} best_epoch={hists[arm]['best_epoch']}", flush=True)

    def ex(eval_a, bin_b, eval_b, nodes_b, seed, games):
        return [binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", eval_a, "--a-nodes", NODES,
                "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes_b, "--seed", seed, "--games", games,
                "--opening-plies", "6", "--max-plies", "320"]

    print("=== T111 STAGE primary (B vs A, 16 x 100, 100k nodes each) ===", flush=True)
    tsvs = [logs / f"{TAG}_primary_c{i:02d}_seed{s}_n{t106.CHUNK_GAMES}.tsv" for i, s in enumerate(PRIMARY_SEEDS)]
    t106.run_chunks([ex(evals["B"], yane, evals["A"], NODES, s, t106.CHUNK_GAMES) for s in PRIMARY_SEEDS],
                    tsvs, env, repo)
    rows, bad = t106.combine(tsvs, t106.CHUNK_GAMES)
    failures += bad
    primary = paired_h2h_delta(rows, label="B_minus_A")
    print(f"T111_PRIMARY B-vs-A W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} "
          f"score={primary['score']:.4f} elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]}",
          flush=True)

    print("=== T111 STAGE frame (A and B vs both anchors) ===", flush=True)
    rows_by = {}   # (arm, anchor) -> rows
    for arm in ("A", "B"):
        for name, nodes in ANCHORS.items():
            bin_b, eval_b = anchor_bin[name]
            ftsvs = [logs / f"{TAG}_frame_{arm}_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv"
                     for i, s in enumerate(FRAME_SEEDS)]
            t106.run_chunks([ex(evals[arm], bin_b, eval_b, nodes, s, FRAME_GAMES) for s in FRAME_SEEDS],
                            ftsvs, env, repo)
            r, b = t106.combine(ftsvs, FRAME_GAMES)
            failures += b
            rows_by[(arm, name)] = r
            h = paired_h2h_delta(r, label=f"{arm}_minus_{name}@{nodes}")
            print(f"T111_FRAME arm={arm} {name}@{nodes} score={h['score']:.4f} elo={h['elo']:+.1f}", flush=True)
    for name, nodes in ANCHORS.items():   # earlier arms on the same seeds: Task 110 (cp), Task 109 (adopted)
        for ref, tag in (("T110", "task110"), ("ADOPTED", "task109")):
            rtsvs = [logs / f"{tag}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
            r, b = t106.combine(rtsvs, FRAME_GAMES)
            failures += b
            rows_by[(ref, name)] = r

    def pooled(x, y, label):
        a = rows_by[(x, "tanuki")] + rows_by[(x, "aoba")]
        b = rows_by[(y, "tanuki")] + rows_by[(y, "aoba")]
        if any(p["opening"] != q["opening"] or p["color"] != q["color"] for p, q in zip(a, b)):
            failures.append(f"{label}: openings differ")
        return paired_series_delta(a, b, label=label)

    frame = pooled("B", "A", "(B - A) pooled over both anchors")
    obj = pooled("A", "T110", "(A - Task110 cp) pooled: win-rate objective itself")
    vs_adopted = pooled("B", "ADOPTED", "(B - adopted) pooled")
    per_anchor = {name: paired_series_delta(rows_by[("B", name)], rows_by[("A", name)], label=f"(B - A) vs {name}")
                  for name in ANCHORS}
    for name, d in per_anchor.items():
        print(f"T111_FRAME_DELTA {name} B-A={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
    print(f"T111_FRAME_POOLED B-A={frame['delta_elo']:+.2f} ci={[round(v, 2) for v in frame['ci95']]}", flush=True)
    print(f"T111_DESCRIPTIVE A-T110={obj['delta_elo']:+.2f} ci={[round(v, 2) for v in obj['ci95']]} "
          f"B-adopted={vs_adopted['delta_elo']:+.2f} ci={[round(v, 2) for v in vs_adopted['ci95']]}", flush=True)

    low, high = primary["ci95"]
    verdict = ("INVALID" if failures else "MIX_HELPS" if low > 0 and frame["ci95"][0] > 0
               else "MIX_HURTS" if high < 0 else "MIX_NEUTRAL")
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({
        "task": 111, "git": {"head": head, "pushed": False},
        "training": {arm: {k: h.get(k) for k in ("best_epoch", "best_val_loss", "epochs")} for arm, h in hists.items()},
        "primary": {"h2h": primary, "conditions": conditions(tsvs), "tsvs": [str(t) for t in tsvs]},
        "frame": {"pooled_B_minus_A": frame, "per_anchor_B_minus_A": per_anchor,
                  "descriptive": {"A_minus_task110": obj, "B_minus_adopted": vs_adopted}},
        "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T111_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
