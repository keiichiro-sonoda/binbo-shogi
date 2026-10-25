#!/usr/bin/env python3
"""Task 138: how much does one training run vary (seed), and does averaging the last epochs help (SWA)?

Task 135 retrained the adopted recipe with the fast trainer (same seed, only the float summation order
differs) and the standard frame said +24.2 [+0.5, +48.2]. Run-to-run variation of the recipe has never
been measured, yet every "neutral / slightly better" verdict depends on it. Training now takes 1.8 h,
so: train the adopted recipe (Task 129: caches/shared_64m_qs129 + compact_qs129, labels
work/task129/labels, lr 6e-5, Factorizer, mirror, 12 epochs, --fast) with two new seeds, saving every
epoch, and play each in the standard frame. The same runs give SWA candidates for free: the plain mean
of the epoch 9-12 weights (float parameters, before fake quant; the factorizer is linear so it averages
with W), exported like any eval.

Pre-registered (fixed before the run), base = the adopted task129_student_64m_qsrel_e12 throughout:
  * descriptive: delta (run - adopted) for seed B, seed C (and Task 135 = seed A on the fast path);
    their spread is the run-to-run yardstick.
  * SWA: delta (swa - adopted) per seed, and the paired delta (swa - its own best epoch) per seed.
  * if any SWA or seed run has CI low > 0 vs adopted: confirm it on seeds 20273501-08 (prefix conf135,
    where the adopted eval's games are already played by t135_confirm.py); confirm CI low > 0 ->
    ADOPT_CANDIDATE (adoption still decided by the author), else NOT_CONFIRMED. At most the best two
    by point estimate are confirmed (time budget).
  * gates: 0cp GREEN, no failures in the frames -> otherwise INVALID for that eval.

Runs as two threads: GPU (train B, then C) and CPU (export / 0cp / frames as soon as a model exists).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import queue  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402

TAG = "task138"
ADOPTED = "task129_student_64m_qsrel_e12"
SEEDS = {"B": 20261019, "C": 20261020}
SWA_EPOCHS = (9, 10, 11, 12)
LABELS, COMPACT = "work/task129/labels", "caches/compact_qs129"
CONF_SEEDS, CONF_PREFIX = tuple(20273501 + i for i in range(8)), "conf135"


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ, PYTHONPATH=str(repo))
    print(f"T138_PREFLIGHT_OK head={head}", flush=True)
    ready, failures, results, lock = queue.Queue(), {}, {}, threading.Lock()

    def train(key, seed):
        out = data / "models" / f"{TAG}_seed{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T138 STAGE train seed {key}={seed} ===", flush=True)
            t106.run([torch_py, "-m", "trainer.train_nnue", data / t129.QS64, "--out", out, *t129.LR_ARGS,
                      "--factorize", "--epochs", t129.EPOCHS, "--augment-mirror", "--mirror-cache",
                      data / t129.MIRROR_CACHE, "--eval-npy", data / LABELS, "--limit-train", t129.N64,
                      "--fast", "--compact-cache", data / COMPACT, "--save-epochs", "--seed", seed,
                      "--checkpoint-minutes", "10", "--resume"], logs / f"{TAG}_seed{key}.log", cwd=repo, env=env)
        return out

    def swa(model_dir):
        out = model_dir / "nnue_swa.pt"
        if not out.exists():
            code = ("import sys,torch;ps=[torch.load(p,map_location='cpu') for p in sys.argv[2:]];"
                    "torch.save({k:(sum(p[k].double() for p in ps)/len(ps)).to(ps[0][k].dtype) "
                    "if ps[0][k].is_floating_point() else ps[0][k] for k in ps[0]},sys.argv[1])")
            subprocess.run([torch_py, "-c", code, out, *[model_dir / f"nnue_epoch{e:02d}.pt" for e in SWA_EPOCHS]],
                           check=True)
        return "nnue_swa.pt"

    def export(name, model_dir, weights):
        eval_dir = data / "eval_dirs" / name
        bnnq = model_dir / weights.replace(".pt", ".bnnq")
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
        return [] if "RED" not in text and text.count("GREEN") >= 2 else ["0cp gate RED"]

    gpu_error = []

    def gpu():
        try:
            for key, seed in SEEDS.items():
                ready.put((key, train(key, seed)))
        except Exception as e:  # noqa: BLE001 -- reported at the end; the CPU side finishes what it has
            gpu_error.append(repr(e))
            print(f"T138_GPU_FAILED {e!r}", flush=True)
        finally:
            ready.put(None)

    th = threading.Thread(target=gpu, daemon=True)
    th.start()
    base, bad = std_frame.measure(ADOPTED, env)
    if bad:
        raise RuntimeError(f"adopted frame: {bad}")
    frames = {}

    def measure(name, model_dir, weights):
        fails = export(name, model_dir, weights)
        rows, bad = std_frame.measure(name, env)
        fails += bad
        pooled, per, bad = std_frame.delta(rows, base, f"({name} - adopted)")
        fails += bad
        frames[name] = rows
        failures[name] = fails
        results[name] = {"pooled": pooled, "per_anchor": per}
        print(f"T138_DELTA {name} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()) + f" failures={fails}", flush=True)

    while (item := ready.get()) is not None:
        key, model_dir = item
        single, avg = f"{TAG}_seed{key}", f"{TAG}_seed{key}_swa"
        measure(single, model_dir, "nnue_best.pt")
        measure(avg, model_dir, swa(model_dir))
        pooled, per, bad = std_frame.delta(frames[avg], frames[single], f"({avg} - {single})")
        results[f"{avg}_vs_single"] = {"pooled": pooled, "per_anchor": per}
        print(f"T138_SWA_PAIRED seed {key} pooled={pooled['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in pooled['ci95']]}", flush=True)
        hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
        results[single]["best_epoch"] = hist["best_epoch"]
    th.join()

    ok = [n for n in frames if not failures[n] and results[n]["pooled"]["ci95"][0] > 0]
    ok = sorted(ok, key=lambda n: -results[n]["pooled"]["delta_elo"])[:2]
    if ok:
        conf_base, bad = std_frame.measure(ADOPTED, env, seeds=CONF_SEEDS, prefix=CONF_PREFIX)
        for name in ok:
            rows, bad2 = std_frame.measure(name, env, seeds=CONF_SEEDS, prefix=CONF_PREFIX)
            pooled, per, bad3 = std_frame.delta(rows, conf_base, f"confirm ({name} - adopted)")
            fails = bad + bad2 + bad3
            verdict = "INVALID" if fails else "ADOPT_CANDIDATE" if pooled["ci95"][0] > 0 else "NOT_CONFIRMED"
            results[name]["confirm"] = {"pooled": pooled, "per_anchor": per, "verdict": verdict, "failures": fails}
            print(f"T138_CONFIRM {name} pooled={pooled['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in pooled['ci95']]} verdict={verdict}", flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 138, "git": {"head": head}, "seeds": SEEDS, "swa_epochs": SWA_EPOCHS,
                               "results": results, "failures": failures, "gpu_error": gpu_error},
                              indent=2), encoding="utf-8")
    print(f"T138_DONE results={out}", flush=True)
    return 1 if gpu_error else 0


if __name__ == "__main__":
    sys.exit(main())
