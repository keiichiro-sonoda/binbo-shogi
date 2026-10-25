#!/usr/bin/env python3
"""Task 109: train the adopted 64M student for 24 epochs instead of 12.

Task 107's student (`task107_student_64m`, the adopted eval) still improved at its
last epoch (best epoch 12 of 12, flag LAST_EPOCH_BEST). Its train_state.pt holds
the state at the start of epoch 13 and the learning rate is constant, so resuming
it with --epochs 24 is the same run as a 24-epoch run from scratch (bit-exact on
CPU, Task 106; on GPU up to its usual nondeterminism). The adopted model directory
is never touched: it is copied to a new one first.

PRE-REGISTERED (committed before the run; no 24-epoch result is seen):
  * one variable: epochs (12 -> 24). Cache, labels, mask, recipe and seed as Task 107.
  * candidate = nnue_best.pt of the 24-epoch run (val-loss selected, as every task).
    If the best epoch is still 12 the candidate IS the adopted eval: verdict
    NO_NEW_BEST, no games.
  * every game at a fixed node budget (Task 108 standard): candidate and adopted at
    `go nodes 100000`; anchors tanuki- @10,000 and AobaNNUE @5,000 nodes.
  * PRIMARY = candidate vs adopted, yane-vs-yane-ex, 1600 games = 16 chunks x 100
    (seeds 20271601-16), paired bootstrap over opening pairs.
  * FRAME = candidate vs each anchor, 400 games (4 x 100, seeds 20271511-14 = the
    Task 108 confirmation seeds, so the adopted arm is reused: node-fixed games
    reproduce). Pooled frame delta (candidate - adopted) over the 800 shared
    openings (both anchors), plus each anchor's delta (descriptive).
  * gates: 0cp GREEN, no illegal move / engine failure, openings identical to the
    Task 108 adopted arm.
  * decision table:
      0 a gate failed                                     -> INVALID (non-terminal)
      - best epoch == 12                                  -> NO_NEW_BEST
      1 primary CI low > 0 and pooled frame CI low > 0    -> EPOCHS_HELP (adoption candidate)
      2 primary CI high < 0                               -> EPOCHS_HURT
      3 otherwise                                         -> EPOCHS_NEUTRAL
  * flags only: LAST_EPOCH_BEST (best epoch == 24).

Interruptible (scripts/job.sh pause|resume task109).
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

import t106_filter as t106  # noqa: E402
from t098_pipeline import NNUE_ARGS, paired_h2h_delta  # noqa: E402
from t103_absolute import conditions, paired_series_delta  # noqa: E402

ADOPTED = "task107_student_64m"
TAG = "task109"
EPOCHS = 24
NODES = 100_000
ANCHORS = {"tanuki": 10_000, "aoba": 5_000}
PRIMARY_SEEDS = tuple(20271601 + i for i in range(16))
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
    model_dir = data / "models" / f"{TAG}_student_64m_e{EPOCHS}"
    eval_dir = data / "eval_dirs" / f"{TAG}_student_64m_e{EPOCHS}"
    adopted_eval = data / "eval_dirs" / ADOPTED
    for path in (binbo, yane, src_model / "train_state.pt", data / CACHE / "meta.json", data / LABELS / "meta.json",
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
    print(f"T109_PREFLIGHT_OK head={head} adopted={ADOPTED} epochs={EPOCHS}", flush=True)
    failures, flags = [], []

    if not model_dir.exists():
        tmp = model_dir.with_name(model_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src_model, tmp)
        for name in ("history.json", "nnue_final.pt", "nnue_best.bnnq"):   # written again by this run
            (tmp / name).unlink(missing_ok=True)
        tmp.rename(model_dir)
        print(f"T109_COPIED {src_model} -> {model_dir} (adopted model untouched)", flush=True)
    if not (model_dir / "history.json").exists():
        print(f"=== T109 STAGE train epochs 13..{EPOCHS} (resume) ===", flush=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        n_train = json.loads((data / CACHE / "meta.json").read_text(encoding="utf-8"))["n_train"]
        t106.run([torch_py, "-m", "trainer.train_nnue", data / CACHE, "--out", model_dir, *NNUE_ARGS,
                  "--epochs", EPOCHS, "--augment-mirror", "--mirror-cache", data / MIRROR_CACHE,
                  "--eval-npy", data / LABELS, "--limit-train", n_train, "--row-mask", data / MASK, "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    best = hist["best_epoch"]
    print(f"T109_TRAINED best_epoch={best} epochs={hist['epochs']}", flush=True)
    if best == EPOCHS:
        flags.append("LAST_EPOCH_BEST")
    out = logs / f"{TAG}_results.json"
    if best <= 12:
        out.write_text(json.dumps({"task": 109, "verdict": "NO_NEW_BEST", "history": hist}, indent=2), encoding="utf-8")
        print(f"T109_VERDICT verdict=NO_NEW_BEST results={out}", flush=True)
        return 0

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

    print("=== T109 STAGE primary (24-epoch vs adopted, 16 x 100, 100k nodes each) ===", flush=True)
    tsvs = [logs / f"{TAG}_primary_c{i:02d}_seed{s}_n{t106.CHUNK_GAMES}.tsv" for i, s in enumerate(PRIMARY_SEEDS)]
    t106.run_chunks([ex(eval_dir, yane, adopted_eval, NODES, s, t106.CHUNK_GAMES) for s in PRIMARY_SEEDS],
                    tsvs, env, repo)
    rows, bad = t106.combine(tsvs, t106.CHUNK_GAMES)
    failures += bad
    cond_p = conditions(tsvs)
    primary = paired_h2h_delta(rows, label="e24_minus_adopted")
    print(f"T109_PRIMARY W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} score={primary['score']:.4f} "
          f"elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]}", flush=True)

    print("=== T109 STAGE frame (anchors; adopted arm = Task 108 confirmation) ===", flush=True)
    cand_all, adopted_all, per_anchor = [], [], {}
    for name, nodes in ANCHORS.items():
        bin_b, eval_b = anchor_bin[name]
        ctsvs = [logs / f"{TAG}_frame_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
        t106.run_chunks([ex(eval_dir, bin_b, eval_b, nodes, s, FRAME_GAMES) for s in FRAME_SEEDS], ctsvs, env, repo)
        atsvs = [logs / f"task108_confirm_{name}{nodes}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
        rc, bc = t106.combine(ctsvs, FRAME_GAMES)
        ra, ba = t106.combine(atsvs, FRAME_GAMES)
        failures += bc + ba
        if any(x["opening"] != y["opening"] or x["color"] != y["color"] for x, y in zip(rc, ra)):
            failures.append(f"frame {name}: openings differ from the Task 108 adopted arm")
        hc = paired_h2h_delta(rc, label=f"e24_minus_{name}@{nodes}")
        d = paired_series_delta(rc, ra, label=f"(e24 - adopted) vs {name}@{nodes}")
        per_anchor[name] = {"nodes": nodes, "candidate": hc, "delta": d, "tsvs": [str(t) for t in ctsvs]}
        print(f"T109_FRAME {name}@{nodes} cand={hc['score']:.4f} ({hc['elo']:+.1f}) "
              f"delta={d['delta_elo']:+.2f} ci={[round(v, 2) for v in d['ci95']]}", flush=True)
        cand_all += rc
        adopted_all += ra
    pooled = paired_series_delta(cand_all, adopted_all, label="(e24 - adopted), both anchors pooled")
    print(f"T109_FRAME_POOLED delta={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)

    low, high = primary["ci95"]
    verdict = ("INVALID" if failures else "EPOCHS_HELP" if low > 0 and pooled["ci95"][0] > 0
               else "EPOCHS_HURT" if high < 0 else "EPOCHS_NEUTRAL")
    out.write_text(json.dumps({
        "task": 109, "git": {"head": head, "pushed": False},
        "training": {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")},
        "history": hist.get("history"),
        "primary": {"h2h": primary, "conditions": cond_p, "tsvs": [str(t) for t in tsvs]},
        "frame": {"per_anchor": per_anchor, "pooled": pooled},
        "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T109_VERDICT verdict={verdict} flags={flags} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
