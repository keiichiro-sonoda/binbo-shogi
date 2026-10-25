#!/usr/bin/env python3
"""Task 112: mix the game result into the label but keep the cp objective.

Task 111: mixing the result (lambda 0.75, known rows) pointed positive on every
measure (+16 direct, +30 pooled) but was masked by moving to the win-rate objective,
which on its own was ~-23 vs cp. Here the result is mixed in win-rate space and the
target turned back into cp (`train_nnue.py --cp-result-lambda`), so the loss stays
the cp huber of every adopted eval. Rows whose result is draw/unknown keep their
target exactly.

PRE-REGISTERED (committed before the run; no result of the arm is seen):
  * one variable: the label on known-result rows. Control = the Task 110 student
    (`task110_student_anneal`): the same continuation (warm start from the adopted
    eval, fresh Adam, lr 1e-5 cosine over 4 epochs) in cp with pure eval labels.
    Arm C = identical + `--cp-result-lambda 0.75`.
  * candidate = C's val-best epoch.
  * PRIMARY = C vs the Task 110 student, both 100,000 nodes, 1600 games = 16 x 100
    (seeds 20271901-16).
  * FRAME = C vs tanuki- @10,000 and AobaNNUE @5,000, 400 games each (seeds
    20271511-14); pooled delta (C - Task 110) over the 800 shared openings (the
    Task 110 arm = its frame games).
  * descriptive only: (C - adopted) pooled vs the Task 109 frame games; per-anchor deltas.
  * gates: 0cp GREEN, no illegal move / engine failure, openings shared.
  * decision table (C vs Task 110):
      0 a gate failed                                   -> INVALID (non-terminal)
      1 primary CI low > 0 and pooled frame CI low > 0  -> CP_MIX_HELPS  (adoption still needs C > adopted)
      2 primary CI high < 0                             -> CP_MIX_HURTS
      3 otherwise                                       -> CP_MIX_NEUTRAL

Interruptible (scripts/job.sh pause|resume task112).
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
CONTROL = "task110_student_anneal"
CONTROL_FRAME_TAG = "task110"
ADOPTED_FRAME_TAG = "task109"
TAG = "task112"
EPOCHS = 4
NODES = 100_000
ANCHORS = {"tanuki": 10_000, "aoba": 5_000}
PRIMARY_SEEDS = tuple(20271901 + i for i in range(16))
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
    model_dir = data / "models" / f"{TAG}_student_cpmix"
    eval_dir = data / "eval_dirs" / f"{TAG}_student_cpmix"
    control_eval = data / "eval_dirs" / CONTROL
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
    print(f"T112_PREFLIGHT_OK head={head} control={CONTROL}", flush=True)
    failures, flags = [], []

    if not (model_dir / "history.json").exists():
        print("=== T112 STAGE train: Task 110 continuation + --cp-result-lambda 0.75 ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        n_train = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))["n_train"]
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS,
                  "--epochs", EPOCHS, "--lr-schedule", "cosine", "--lr-min-frac", "0.01",
                  "--init-weights", src_model / "nnue_best.pt", "--cp-result-lambda", "0.75",
                  "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", n_train, "--row-mask", data / MASK, "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    best = hist["best_epoch"]
    print(f"T112_TRAINED best_epoch={best} epochs={hist['epochs']}", flush=True)
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

    print("=== T112 STAGE primary (C vs Task 110 student, 16 x 100, 100k nodes each) ===", flush=True)
    tsvs = [logs / f"{TAG}_primary_c{i:02d}_seed{s}_n{t106.CHUNK_GAMES}.tsv" for i, s in enumerate(PRIMARY_SEEDS)]
    t106.run_chunks([ex(eval_dir, yane, control_eval, NODES, s, t106.CHUNK_GAMES) for s in PRIMARY_SEEDS],
                    tsvs, env, repo)
    rows, bad = t106.combine(tsvs, t106.CHUNK_GAMES)
    failures += bad
    cond_p = conditions(tsvs)
    primary = paired_h2h_delta(rows, label="cpmix_minus_task110")
    print(f"T112_PRIMARY W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} score={primary['score']:.4f} "
          f"elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]}", flush=True)

    print("=== T112 STAGE frame (anchors; control arm = Task 110 frame games) ===", flush=True)
    cand_all, adopted_all, per_anchor = [], [], {}   # "adopted_all" = the control (Task 110) arm here
    real_adopted_all = []
    for name, nodes in ANCHORS.items():
        bin_b, eval_b = anchor_bin[name]
        ctsvs = [logs / f"{TAG}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
        t106.run_chunks([ex(eval_dir, bin_b, eval_b, nodes, s, FRAME_GAMES) for s in FRAME_SEEDS], ctsvs, env, repo)
        atsvs = [logs / f"{CONTROL_FRAME_TAG}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
        real_adopted_all += t106.combine([logs / f"{ADOPTED_FRAME_TAG}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv"
                                          for i, s in enumerate(FRAME_SEEDS)], FRAME_GAMES)[0]
        rc, bc = t106.combine(ctsvs, FRAME_GAMES)
        ra, ba = t106.combine(atsvs, FRAME_GAMES)
        failures += bc + ba
        if any(x["opening"] != y["opening"] or x["color"] != y["color"] for x, y in zip(rc, ra)):
            failures.append(f"frame {name}: openings differ from the adopted arm")
        hc = paired_h2h_delta(rc, label=f"cpmix_minus_{name}@{nodes}")
        d = paired_series_delta(rc, ra, label=f"(cpmix - task110) vs {name}@{nodes}")
        per_anchor[name] = {"nodes": nodes, "candidate": hc, "delta": d, "tsvs": [str(t) for t in ctsvs]}
        print(f"T112_FRAME {name}@{nodes} cand={hc['score']:.4f} ({hc['elo']:+.1f}) "
              f"delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
        cand_all += rc
        adopted_all += ra
    pooled = paired_series_delta(cand_all, adopted_all, label="(cpmix - task110), both anchors pooled")
    vs_adopted = paired_series_delta(cand_all, real_adopted_all, label="(cpmix - adopted), both anchors pooled")
    print(f"T112_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
    print(f"T112_DESCRIPTIVE C-adopted={vs_adopted['delta_elo']:+.2f} ci={[round(v, 2) for v in vs_adopted['ci95']]}", flush=True)

    low, high = primary["ci95"]
    verdict = ("INVALID" if failures else "CP_MIX_HELPS" if low > 0 and pooled["ci95"][0] > 0
               else "CP_MIX_HURTS" if high < 0 else "CP_MIX_NEUTRAL")
    out.write_text(json.dumps({
        "task": 112, "git": {"head": head, "pushed": False},
        "training": {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")},
        "history": hist.get("history"),
        "primary": {"h2h": primary, "conditions": cond_p, "tsvs": [str(t) for t in tsvs]},
        "frame": {"per_anchor": per_anchor, "pooled": pooled, "descriptive_vs_adopted": vs_adopted},
        "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T112_VERDICT verdict={verdict} flags={flags} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
