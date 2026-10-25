#!/usr/bin/env python3
"""Task 142: does public teacher data help? nodchip/shogi_suisho5_depth9 (MIT) validation file, 7.24M positions.

Data (docs/public_data_survey_2026-10-19.md): Hugging Face `nodchip/shogi_suisho5_depth9`, `shuffled.bin`
(289,626,920 B, sha256 7590e128...), YaneuraOu PackedSfenValue: Suisho5 depth 9, positions already replaced by the
leaf of their qsearch PV, shuffled. 7,240,673 positions, no position in check (20k sample), score = side to move's
cp on YaneuraOu's scale (win rate = sigmoid(cp/600) = the same scale as our 1200*atanh(v)); 29% |score| >= 2000,
8.5% mate scores (clamped to +-2000 by the trainer like our labels).

Two arms, each vs the adopted eval (task129_student_64m_qsrel_e12):
  A "mix": the adopted recipe (Task 129: lr 6e-5, Factorizer, mirror, 12 epochs, --fast) on our 63.95M rows
     + the 7.24M public rows (labels: ours for ours, Suisho5's score for theirs) = 71,190,673 rows.
  B "finetune": start from the adopted eval's weights, 2 epochs on the 7.24M public rows only, lr 1e-5, final
     weights (the "finish on someone else's data" practice, e.g. WCSC34 tanuki- on tayayan's data).
Decision (fixed before the run, docs/evaluation_protocol.md 2026-10-19): 3 seed sets pooled (std, conf135,
conf138; 4,800 games per eval), paired delta vs adopted: CI low > 0 -> confirm on a fresh set 20273701-08
(prefix conf142, both evals, 800 x 2): point estimate > 0 -> ADOPT_CANDIDATE else NOT_CONFIRMED;
CI high < 0 -> WORSE; otherwise NEUTRAL. Gates: conversion round-trip on a sample (features -> board == psv
board), 0cp GREEN, no frame failures; failures -> INVALID.
Val stays our 50k (WS-T labels).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from trainer import kp  # noqa: E402

TAG = "task142"
ADOPTED = "task129_student_64m_qsrel_e12"
PSV_SHA = "7590e12897e82633a2be727045df60ad8e34b0115d410804d0bbf55e1351b26a"
CONF_SEEDS, CONF_PREFIX = tuple(20273701 + i for i in range(8)), "conf142"


def _rows(args):
    path, lo, hi = args
    a = np.fromfile(path, dtype=cshogi.PackedSfenValue, count=hi - lo, offset=lo * 40)
    b = cshogi.Board()
    out = np.empty((hi - lo, 77), np.int32)
    for i, e in enumerate(a):
        b.set_psfen(e["sfen"])
        out[i] = kp.canonical_shared_row(b)
    return lo, out


def build(data, work):
    """Public cache (7.24M) and mix cache (ours + public), labels dirs, compact caches. Idempotent."""
    psv = data / "public/nodchip_suisho5_depth9/shuffled.bin"
    pub, mix = data / "caches/public_suisho5_val7m", data / "caches/mix_qs129_suisho5"
    src = data / t129.QS64
    lab_pub, lab_mix, lab_old = work / "labels_public", work / "labels_mix", data / "work/task129/labels"
    val = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    a = np.fromfile(psv, dtype=cshogi.PackedSfenValue)
    n = len(a)
    if not (pub / "meta.json").exists():
        sha = subprocess.run(["sha256sum", psv], capture_output=True, text=True, check=True).stdout.split()[0]
        if sha != PSV_SHA:
            raise RuntimeError(f"psv sha256 {sha} != {PSV_SHA}")
        pub.mkdir(parents=True, exist_ok=True)
        idx = np.lib.format.open_memmap(pub / "train_indices.npy", mode="w+", dtype=np.int32, shape=(n, 77))
        step = 200_000
        with Pool(12) as pool:
            for lo, out in pool.imap_unordered(_rows, [(str(psv), lo, min(n, lo + step)) for lo in range(0, n, step)]):
                idx[lo:lo + len(out)] = out
        idx.flush()
        rng, b = np.random.default_rng(20271420), cshogi.Board()
        for i in rng.choice(n, 2000, replace=False):  # gate: features decode back to the psv board
            b.set_psfen(a[i]["sfen"])
            if kp.canonical_shared_row_to_sfen(np.asarray(idx[i])).split()[:3] != b.sfen().split()[:3]:
                raise AssertionError(f"row {i}: features do not decode to the psv board")
        np.save(pub / "train_evals.npy", a["score"].astype(np.int16))
        np.save(pub / "train_wdl.npy", a["game_result"].astype(np.int8))
        for f in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
            (pub / f).symlink_to((src / f).resolve())
        meta = dict(val, n_train=n, built_by="t142 (nodchip/shogi_suisho5_depth9 shuffled.bin)", source_psv=str(psv),
                    source_sha256=PSV_SHA)
        meta.pop("qsearch_replace", None)
        (pub / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"T142_PUBLIC rows={n} gate=2000/2000", flush=True)
    for d, train in ((lab_pub, a["score"].astype(np.int16)),
                     (lab_mix, None)):
        if (d / "meta.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        if train is None:
            train = np.concatenate([np.load(lab_old / "train_dlshogi_evals.npy"), a["score"].astype(np.int16)])
        np.save(d / "train_dlshogi_evals.npy", train)
        shutil.copy(lab_old / "val_dlshogi_evals.npy", d / "val_dlshogi_evals.npy")
        (d / "meta.json").write_text(json.dumps({"train": len(train), "public_rows": n}), encoding="utf-8")
    if not (mix / "meta.json").exists():
        mix.mkdir(parents=True, exist_ok=True)
        old = np.load(src / "train_indices.npy", mmap_mode="r")
        new = np.load(pub / "train_indices.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(mix / "train_indices.npy", mode="w+", dtype=np.int32,
                                        shape=(len(old) + n, 77))
        for lo in range(0, len(old), 1 << 20):
            hi = min(len(old), lo + (1 << 20))
            out[lo:hi] = old[lo:hi]
        out[len(old):] = new
        out.flush()
        np.save(mix / "train_evals.npy", np.concatenate([np.load(src / "train_evals.npy"), np.load(pub / "train_evals.npy")]))
        np.save(mix / "train_wdl.npy", np.concatenate([np.load(src / "train_wdl.npy"), np.load(pub / "train_wdl.npy")]))
        for f in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
            (mix / f).symlink_to((src / f).resolve())
        meta = dict(val, n_train=len(old) + n, built_by="t142 (shared_64m_qs129 + public_suisho5_val7m)")
        (mix / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for c in (pub, mix):
        comp = Path(str(c) + "_compact")
        if not (comp / "compact.json").exists():
            subprocess.run([sys.executable, "-m", "trainer.compact_cache", c, comp], check=True,
                           cwd=std_frame.paths()[0], env=dict(os.environ, PYTHONPATH=str(std_frame.paths()[0])))
    return pub, mix, lab_pub, lab_mix, n


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
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    print(f"T142_PREFLIGHT_OK head={head}", flush=True)
    pub, mix, lab_pub, lab_mix, n_pub = build(data, work)
    common = ["--qat", "--qat-tie", "--feature-dim", "256", "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600",
              "--factorize", "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE, "--fast",
              "--checkpoint-minutes", "10", "--resume"]
    arms = {
        "finetune": (pub, lab_pub, n_pub, ["--lr", "1e-5", "--epochs", "2", "--init-weights",
                                           data / "models" / ADOPTED / "nnue_best.pt"], "nnue_final.pt"),
        "mix": (mix, lab_mix, t129.N64 + n_pub, ["--lr", "6e-5", "--epochs", "12"], "nnue_best.pt"),
    }
    ready = []
    lock = threading.Condition()

    def train(name):
        cache, labels, n, extra, weights = arms[name]
        out = data / "models" / f"{TAG}_{name}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T142 STAGE train {name} ===", flush=True)
            t106.run([torch_py, "-m", "trainer.train_nnue", cache, "--out", out, *common, *extra, "--eval-npy", labels,
                      "--limit-train", n, "--compact-cache", Path(str(cache) + "_compact")],
                     logs / f"{TAG}_{name}.log", cwd=repo, env=env)
        return out, weights

    def gpu():
        try:
            for name in ("finetune", "mix"):
                r = train(name)
                with lock:
                    ready.append((name, *r))
                    lock.notify()
        finally:
            with lock:
                ready.append(None)
                lock.notify()

    threading.Thread(target=gpu, daemon=True).start()
    base, bad = std_frame.measure_sets(ADOPTED, env)
    if bad:
        raise RuntimeError(f"adopted frame: {bad}")
    results, failures = {}, {}
    while True:
        with lock:
            while not ready:
                lock.wait()
            item = ready.pop(0)
        if item is None:
            break
        name, model_dir, weights = item
        ev = f"{TAG}_{name}"
        eval_dir, bnnq = data / "eval_dirs" / ev, model_dir / weights.replace(".pt", ".bnnq")
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", weights, "--out", bnnq],
                     logs / f"{ev}_export.log", cwd=repo, env=env)
            t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", bnnq, "--template",
                      data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                     logs / f"{ev}_yane.log", cwd=repo, env=env)
        gate = logs / f"{ev}_0cp.log"
        if not gate.exists():
            t106.run(["bash", "scripts/binbo_to_yane.sh", bnnq], gate, cwd=repo, env=env)
        text = gate.read_text(encoding="utf-8")
        fails = [] if "RED" not in text and text.count("GREEN") >= 2 else ["0cp gate RED"]
        cand, bad = std_frame.measure_sets(ev, env)
        total, per, per_set, bad2 = std_frame.delta_sets(cand, base, f"({ev} - adopted)")
        fails += bad + bad2
        lo, hi = total["ci95"]
        verdict, confirm = None, None
        if fails:
            verdict = "INVALID"
        elif lo > 0:
            cs = {CONF_PREFIX: CONF_SEEDS}
            cb, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs)
            cc, b2 = std_frame.measure_sets(ev, env, sets=cs)
            ct, cper, _cps, b3 = std_frame.delta_sets(cc, cb, f"confirm ({ev} - adopted)")
            fails += b1 + b2 + b3
            confirm = {"pooled": ct, "per_anchor": cper}
            verdict = "INVALID" if fails else "ADOPT_CANDIDATE" if ct["delta_elo"] > 0 else "NOT_CONFIRMED"
        else:
            verdict = "WORSE" if hi < 0 else "NEUTRAL"
        results[ev] = {"pooled": total, "per_anchor": per, "per_set": per_set, "confirm": confirm, "verdict": verdict}
        failures[ev] = fails
        print(f"T142_RESULT {ev} pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              f"sets={ {k: round(v['delta_elo'], 1) for k, v in per_set.items()} } verdict={verdict} failures={fails}",
              flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 142, "git": {"head": head}, "results": results, "failures": failures},
                              indent=2), encoding="utf-8")
    print(f"T142_DONE results={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
