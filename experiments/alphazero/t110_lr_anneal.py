#!/usr/bin/env python3
"""Task 110: anneal the learning rate at the end (cosine over 4 epochs) from the adopted eval.

Every student so far trained at a CONSTANT lr 1e-5, and Task 109 plateaued (best epoch
22 of 24). Decaying the lr at the end of training is the standard cheap finish. One
variable: the lr schedule of a short continuation; data, labels, mask, recipe as Task 109.

PRE-REGISTERED (committed before the run; no annealed result is seen):
  * start = the adopted eval's weights (`task109_student_64m_e24/nnue_best.pt`, epoch 22),
    fresh Adam (--init-weights), --lr 1e-5 --lr-schedule cosine --lr-min-frac 0.01,
    4 epochs (lr per epoch 1.00e-5, 0.85e-5, 0.50e-5, 0.15e-5). The adopted model
    directory is only read.
  * candidate = nnue_best.pt of these 4 epochs (val-loss selected as every task).
  * games at fixed nodes (Task 108 standard): candidate and adopted at 100,000 nodes;
    anchors tanuki- @10,000 and AobaNNUE @5,000.
  * PRIMARY = candidate vs adopted, 1600 games = 16 chunks x 100 (seeds 20271701-16).
  * FRAME = candidate vs each anchor, 400 games (seeds 20271511-14; the adopted arm is
    Task 109's frame games, which used the same seeds and node budgets). Pooled delta
    (candidate - adopted) over the 800 shared openings decides; per-anchor deltas are
    descriptive.
  * gates: 0cp GREEN, no illegal move / engine failure, openings identical to the
    adopted arm.
  * decision table:
      0 a gate failed                                     -> INVALID (non-terminal)
      1 primary CI low > 0 and pooled frame CI low > 0    -> ANNEAL_HELPS (adoption candidate)
      2 primary CI high < 0                               -> ANNEAL_HURTS
      3 otherwise                                         -> ANNEAL_NEUTRAL
  * flags only: FIRST_EPOCH_BEST (best = epoch 1, i.e. annealing never improved val).

Interruptible (scripts/job.sh pause|resume task110).
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

ADOPTED = "task109_student_64m_e24"
ADOPTED_FRAME_TAG = "task109"   # adopted arm = Task 109 frame games
TAG = "task110"
EPOCHS = 4
NODES = 100_000
ANCHORS = {"tanuki": 10_000, "aoba": 5_000}
PRIMARY_SEEDS = tuple(20271701 + i for i in range(16))
FRAME_SEEDS, FRAME_GAMES = (20271511, 20271512, 20271513, 20271514), 100
CACHE, LABELS, MASK = "caches/shared_64m", "work/task107/rescore_search", "work/task107/mask/train_keep.npy"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"


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
    src_model = data / "models" / ADOPTED
    model_dir = data / "models" / f"{TAG}_student_anneal"
    eval_dir = data / "eval_dirs" / f"{TAG}_student_anneal"
    adopted_eval = data / "eval_dirs" / ADOPTED
    for path in (binbo, yane, src_model / "nnue_best.pt", data / CACHE / "meta.json", data / LABELS / "meta.json",
                 data / MASK, adopted_eval / "nn.bin", *[b for b, _ in anchor_bin.values()]):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T110_PREFLIGHT_OK head={head} adopted={ADOPTED} epochs={EPOCHS}", flush=True)
    failures, flags = [], []

    if not (model_dir / "history.json").exists():
        print(f"=== T110 STAGE train: warm start from {ADOPTED} (epoch 22), cosine lr over {EPOCHS} epochs ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        n_train = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))["n_train"]
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS,
                  "--epochs", EPOCHS, "--lr-schedule", "cosine", "--lr-min-frac", "0.01",
                  "--init-weights", src_model / "nnue_best.pt",
                  "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", n_train, "--row-mask", data / MASK, "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    best = hist["best_epoch"]
    print(f"T110_TRAINED best_epoch={best} epochs={hist['epochs']}", flush=True)
    if best == 1:
        flags.append("FIRST_EPOCH_BEST")
    out = logs / f"{TAG}_results.json"

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

    def ex(eval_a, bin_b, eval_b, nodes_b, seed, games):
        return [binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", eval_a, "--a-nodes", NODES,
                "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes_b, "--seed", seed, "--games", games,
                "--opening-plies", "6", "--max-plies", "320"]

    print("=== T110 STAGE primary (annealed vs adopted, 16 x 100, 100k nodes each) ===", flush=True)
    tsvs = [logs / f"{TAG}_primary_c{i:02d}_seed{s}_n{t106.CHUNK_GAMES}.tsv" for i, s in enumerate(PRIMARY_SEEDS)]
    t106.run_chunks([ex(eval_dir, yane, adopted_eval, NODES, s, t106.CHUNK_GAMES) for s in PRIMARY_SEEDS],
                    tsvs, env, repo)
    rows, bad = t106.combine(tsvs, t106.CHUNK_GAMES)
    failures += bad
    cond_p = conditions(tsvs)
    primary = paired_h2h_delta(rows, label="anneal_minus_adopted")
    print(f"T110_PRIMARY W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} score={primary['score']:.4f} "
          f"elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]}", flush=True)

    print("=== T110 STAGE frame (anchors; adopted arm = Task 109 frame games) ===", flush=True)
    cand_all, adopted_all, per_anchor = [], [], {}
    for name, nodes in ANCHORS.items():
        bin_b, eval_b = anchor_bin[name]
        ctsvs = [logs / f"{TAG}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
        t106.run_chunks([ex(eval_dir, bin_b, eval_b, nodes, s, FRAME_GAMES) for s in FRAME_SEEDS], ctsvs, env, repo)
        atsvs = [logs / f"{ADOPTED_FRAME_TAG}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
        rc, bc = t106.combine(ctsvs, FRAME_GAMES)
        ra, ba = t106.combine(atsvs, FRAME_GAMES)
        failures += bc + ba
        if any(x["opening"] != y["opening"] or x["color"] != y["color"] for x, y in zip(rc, ra)):
            failures.append(f"frame {name}: openings differ from the adopted arm")
        hc = paired_h2h_delta(rc, label=f"anneal_minus_{name}@{nodes}")
        d = paired_series_delta(rc, ra, label=f"(anneal - adopted) vs {name}@{nodes}")
        per_anchor[name] = {"nodes": nodes, "candidate": hc, "delta": d, "tsvs": [str(t) for t in ctsvs]}
        print(f"T110_FRAME {name}@{nodes} cand={hc['score']:.4f} ({hc['elo']:+.1f}) "
              f"delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
        cand_all += rc
        adopted_all += ra
    pooled = paired_series_delta(cand_all, adopted_all, label="(anneal - adopted), both anchors pooled")
    print(f"T110_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)

    low, high = primary["ci95"]
    verdict = ("INVALID" if failures else "ANNEAL_HELPS" if low > 0 and pooled["ci95"][0] > 0
               else "ANNEAL_HURTS" if high < 0 else "ANNEAL_NEUTRAL")
    out.write_text(json.dumps({
        "task": 110, "git": {"head": head, "pushed": False},
        "training": {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")},
        "history": hist.get("history"),
        "primary": {"h2h": primary, "conditions": cond_p, "tsvs": [str(t) for t in tsvs]},
        "frame": {"per_anchor": per_anchor, "pooled": pooled},
        "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T110_VERDICT verdict={verdict} flags={flags} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
