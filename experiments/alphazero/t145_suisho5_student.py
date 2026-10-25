#!/usr/bin/env python3
"""Task 145: a "Suisho5 student" -- the same network and recipe trained from scratch on Suisho5's labels only.

Why (docs/teacher_direction_2026-10-20.md): our labels so far are "Gikou2 distilled into WS-T + a shallow MCTS",
Task 142 showed Suisho5's depth-9 labels are much better, and the author wants to see how far a strong teacher from
the start would take the same student, to compare with growing our own teacher ("雑巾絞り", the main line).
This is a yardstick, not an adoption candidate.

Data: nodchip/shogi_suisho5_depth9 (MIT) training split `shuffled.7z.001` (the first volume of the 185 GB split
archive; the data are shuffled, so a prefix is a random sample). The first 63,950,000 PackedSfenValue records
(the size of our own train set) are streamed out of the archive (`7z x -so`), positions already replaced by their
qsearch PV leaf. Converted to our features like Task 142 (2,000-row board round-trip gate). Labels = PSV score
(side to move), clamped to +-2000 by the trainer. Val = our 50k (WS-T labels; val loss is not comparable).
Recipe = Task 129's (lr 6e-5, Factorizer, mirror, 12 epochs, best epoch by val loss, --fast), from scratch.

Comparison (fixed before the run; descriptive): the new seed sets s280/s281/s282 pooled, paired deltas
  (Suisho5 student - task129), (Suisho5 student - task142_finetune), and (task142_finetune - task129).
Gates: 0cp GREEN, conversion gate, no frame failures.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import json  # noqa: E402
import subprocess  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from t142_public_suisho5 import _rows  # noqa: E402
from trainer import kp  # noqa: E402

TAG = "task145"
N = 63_950_000
OWN, ADOPTED = "task129_student_64m_qsrel_e12", "task142_finetune"
CAND = f"{TAG}_suisho5_student"


def extract(data):
    """Stream the first N records of the archive into a raw PSV file (no full extraction)."""
    # 7z keeps its header at the END of the archive (in volume .008). `vol/` holds a symlink to the real .001,
    # sparse placeholders of the exact sizes for .002-.007, and .008 sparse except its last 64 MiB (fetched with an
    # HTTP range request), so 7z can read the header and stream the beginning of the data from .001 only.
    vol = data / "public/nodchip_suisho5_depth9/vol"
    real = data / "public/nodchip_suisho5_depth9/shuffled.7z.001"
    if not (vol / "shuffled.7z.001").exists():
        (vol / "shuffled.7z.001").symlink_to(real)
    src = vol / "shuffled.7z.001"
    out = data / "public/nodchip_suisho5_depth9/train_first64m.bin"
    want = N * 40
    if out.exists() and out.stat().st_size == want:
        return out
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        p = subprocess.Popen(["7z", "x", "-so", str(src)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        left = want
        while left:
            chunk = p.stdout.read(min(left, 64 << 20))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
        p.kill()
        p.wait()
    if tmp.stat().st_size != want:
        raise RuntimeError(f"archive gave {tmp.stat().st_size} bytes < {want}")
    os.replace(tmp, out)
    return out


def build(data):
    psv = extract(data)
    cache = data / "caches/public_suisho5_train64m"
    labels = data / "work" / TAG / "labels"
    src = data / t129.QS64
    if not (cache / "meta.json").exists():
        cache.mkdir(parents=True, exist_ok=True)
        idx = np.lib.format.open_memmap(cache / "train_indices.npy", mode="w+", dtype=np.int32, shape=(N, 77))
        step = 500_000
        with Pool(16) as pool:
            for lo, out in pool.imap_unordered(_rows, [(str(psv), lo, min(N, lo + step)) for lo in range(0, N, step)]):
                idx[lo:lo + len(out)] = out
        idx.flush()
        a = np.memmap(psv, dtype=cshogi.PackedSfenValue, mode="r")
        rng, b = np.random.default_rng(20271450), cshogi.Board()
        for i in rng.choice(N, 2000, replace=False):
            b.set_psfen(a[i]["sfen"])
            if kp.canonical_shared_row_to_sfen(np.asarray(idx[i])).split()[:3] != b.sfen().split()[:3]:
                raise AssertionError(f"row {i}: features do not decode to the psv board")
        np.save(cache / "train_evals.npy", np.asarray(a["score"]).astype(np.int16))
        np.save(cache / "train_wdl.npy", np.asarray(a["game_result"]).astype(np.int8))
        for f in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
            (cache / f).symlink_to((src / f).resolve())
        meta = dict(json.loads((src / "meta.json").read_text(encoding="utf-8")), n_train=N,
                    built_by="t145 (nodchip/shogi_suisho5_depth9 shuffled.7z.001, first 63,950,000 records)")
        meta.pop("qsearch_replace", None)
        (cache / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"T145_CACHE rows={N} gate=2000/2000", flush=True)
    if not (labels / "meta.json").exists():
        labels.mkdir(parents=True, exist_ok=True)
        np.save(labels / "train_dlshogi_evals.npy", np.load(cache / "train_evals.npy"))
        np.save(labels / "val_dlshogi_evals.npy", np.load(data / "work/task129/labels/val_dlshogi_evals.npy"))
        (labels / "meta.json").write_text(json.dumps({"train": "Suisho5 PSV score", "val": "work/task129/labels"}),
                                          encoding="utf-8")
    comp = Path(str(cache) + "_compact")
    if not (comp / "compact.json").exists():
        repo = std_frame.paths()[0]
        subprocess.run([sys.executable, "-m", "trainer.compact_cache", cache, comp], check=True, cwd=repo,
                       env=dict(os.environ, PYTHONPATH=str(repo)))
    return cache, labels, comp


def main():
    repo, data, logs, *_ = std_frame.paths()
    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    cache, labels, comp = build(data)
    out, eval_dir = data / "models" / CAND, data / "eval_dirs" / CAND
    if not (out / "history.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        print("=== T145 STAGE train (Task 129 recipe, Suisho5 labels, from scratch) ===", flush=True)
        t106.run([torch_py, "-m", "trainer.train_nnue", cache, "--out", out, *t129.LR_ARGS, "--factorize",
                  "--epochs", t129.EPOCHS, "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE,
                  "--eval-npy", labels, "--limit-train", N, "--fast", "--compact-cache", comp,
                  "--checkpoint-minutes", "10", "--resume"], logs / f"{CAND}.log", cwd=repo, env=env)
    bnnq = out / "nnue_best.bnnq"
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", out, "--weights", "nnue_best.pt", "--out", bnnq],
                 logs / f"{CAND}_export.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", bnnq, "--template",
                  data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 logs / f"{CAND}_yane.log", cwd=repo, env=env)
    gate = logs / f"{CAND}_0cp.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", bnnq], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    failures = [] if "RED" not in text and text.count("GREEN") >= 2 else ["0cp gate RED"]
    frames = {}
    for ev in (OWN, ADOPTED, CAND):
        frames[ev], bad = std_frame.measure_sets(ev, env)
        failures += bad
    results = {}
    for a, b in ((CAND, OWN), (CAND, ADOPTED), (ADOPTED, OWN)):
        total, per, per_set, bad = std_frame.delta_sets(frames[a], frames[b], f"({a} - {b})")
        failures += bad
        results[f"{a} - {b}"] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T145_DELTA ({a} - {b}) pooled={total['delta_elo']:+.2f} ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
    hist = json.loads((out / "history.json").read_text(encoding="utf-8"))
    res = logs / f"{TAG}_results.json"
    res.write_text(json.dumps({"task": 145, "results": results, "failures": failures,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss")}}, indent=2),
                   encoding="utf-8")
    print(f"T145_DONE failures={failures} results={res}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
