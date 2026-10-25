"""Task 136: does the student fail to FIT the teacher's king danger, or fail to GENERALISE it? (diagnosis, no training)

Task 133: on crowded-king positions the teacher (WS-T S=8) says -56 cp per enemy piece near the side to
move's king, the adopted student says +20. Task 134: relabelling those rows with S=32 moved the labels by
-104 cp on average but the student's coefficient by only ~30% of the teacher's change.

Here we compare the student with the labels it was TRAINED ON, row by row, as a function of
att2_stm / att2_opp (enemy pieces within Chebyshev 2 of each king; Task 134's definition):
  residual = student_cp - teacher_cp   (positive = the student is more optimistic than its teacher)
on (a) a fixed random sample of the training rows and (b) the held-out val rows.
  - residual grows with att2 on TRAIN too  -> the student cannot represent it (capacity / features).
  - train flat but val grows                -> it memorises but does not generalise.
Students: task129 (adopted, labels work/task129/labels) and task134 (labels work/task134/labels; also
scored against the task129 labels to see how far it moved). Forward = the QAT forward with fake quant ON
(what the trainer optimised; equal to the deployed int eval up to a few cp, Task 058). Original
orientation (no mirror). Targets clamped to +-2000 like the trainer.
Caveats: val is the original 50k val of the 64M cache (NOT quiescence-replaced like the train rows, Task 128),
so train-vs-val also differs in position type; the held-out side is better read from Task 133 (unseen games,
teacher -56 vs student +20). task134's val labels are the task129 ones (Task 134 did not touch val).

  PYTHONPATH=. venv/bin/python experiments/alphazero/t136_bridge_residual.py [--sample 2000000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from trainer import kp  # noqa: E402
from trainer.binbo_to_yane import build_collapse_map  # noqa: E402
from trainer.nnue import QATSharedNNUE  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from t134_sharp_relabel_s32 import att2_from_half  # noqa: E402

DATA = os.path.expanduser(os.environ.get("DATA_ROOT", "~/binbo-shogi-data"))
CACHE = f"{DATA}/caches/shared_64m_qs129"
STUDENTS = {
    "task129": (f"{DATA}/models/task129_student_64m_qsrel_e12/nnue_best.pt", f"{DATA}/work/task129/labels"),
    "task134": (f"{DATA}/models/task134_student_sharp32/nnue_best.pt", f"{DATA}/work/task134/labels"),
}
BUCKETS = [0, 1, 2, 3, 4, 5]  # last = 5+
CLAMP = 2000.0


def load_model(path, device):
    m = QATSharedNNUE(total_features=kp.CANONICAL_SHARED_TOTAL, feature_dim=256, hidden_dim=32, fake_quant=True,
                      scale_cp=600.0, tie_collapse_map=build_collapse_map(vm=False), factorize=True)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def predict(model, idx, device, batch=16384):
    out = np.empty(len(idx), np.float32)
    own_n, opp_n = 39, 38
    for s in range(0, len(idx), batch):
        b = torch.from_numpy(np.ascontiguousarray(idx[s:s + batch])).long()
        n = len(b)
        own = b[:, :own_n].reshape(-1).to(device)
        opp = b[:, own_n:].reshape(-1).to(device)
        oo = torch.arange(0, n * own_n, own_n, device=device)
        po = torch.arange(0, n * opp_n, opp_n, device=device)
        out[s:s + n] = (model(own, oo, opp, po) * 600.0).float().cpu().numpy()
    return out


def table(name, res, a_stm, a_opp, f):
    def line(label, mask):
        r = res[mask]
        if len(r) < 30:
            return
        se = r.std(ddof=1) / np.sqrt(len(r))
        f(f"  {label:10s} n={len(r):9,d}  bias {r.mean():+7.1f} (+-{2 * se:4.1f})  mae {np.abs(r).mean():6.1f}")
    f(f"[{name}] all: n={len(res):,} bias {res.mean():+.1f} mae {np.abs(res).mean():.1f}")
    for key, a in (("att2_stm", a_stm), ("att2_opp", a_opp)):
        f(f" by {key}:")
        for k in BUCKETS:
            line(f"{k}{'+' if k == BUCKETS[-1] else ''}", a >= k if k == BUCKETS[-1] else a == k)
    X = np.column_stack([a_stm, a_opp, np.ones(len(res))]).astype(np.float64)
    beta, *_ = np.linalg.lstsq(X, res.astype(np.float64), rcond=None)
    f(f" OLS residual = {beta[0]:+.2f}*att2_stm {beta[1]:+.2f}*att2_opp {beta[2]:+.1f}")
    return {"bias": float(res.mean()), "mae": float(np.abs(res).mean()),
            "coef_stm": float(beta[0]), "coef_opp": float(beta[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20271360)
    ap.add_argument("--out", default=f"{DATA}/work/task136")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = open(os.path.join(a.out, "residual.txt"), "w", encoding="utf-8")

    def f(s):
        print(s, flush=True)
        log.write(s + "\n")

    t0 = time.time()
    tr_all = np.load(f"{CACHE}/train_indices.npy", mmap_mode="r")
    rows = np.sort(np.random.default_rng(a.seed).choice(len(tr_all), a.sample, replace=False))
    tr = tr_all[rows]  # sorted rows: mostly sequential reads
    va = np.load(f"{CACHE}/val_indices.npy")
    a_stm_tr = np.load(f"{DATA}/work/task134/att2_stm.npy")[rows]
    a_opp_tr = np.load(f"{DATA}/work/task134/att2_opp.npy")[rows]
    a_stm_va, a_opp_va = att2_from_half(va[:, :39]), att2_from_half(va[:, 39:])
    f(f"device={device} train sample {a.sample:,} of {len(tr_all):,} (seed {a.seed}), val {len(va):,}; "
      f"loaded in {time.time() - t0:.0f}s")
    f(f"att2_stm>=3: train {np.mean(a_stm_tr >= 3):.3f} val {np.mean(a_stm_va >= 3):.3f}; "
      f"att2_opp>=3: train {np.mean(a_opp_tr >= 3):.3f} val {np.mean(a_opp_va >= 3):.3f}")
    labels = {}
    for lab in ("task129", "task134"):
        d = STUDENTS[lab][1]
        labels[lab] = (np.load(f"{d}/train_dlshogi_evals.npy", mmap_mode="r")[rows].astype(np.float32).clip(-CLAMP, CLAMP),
                       np.load(f"{d}/val_dlshogi_evals.npy").astype(np.float32).clip(-CLAMP, CLAMP))
    summary = {}
    for st, (path, _) in STUDENTS.items():
        model = load_model(path, device)
        p_tr, p_va = predict(model, tr, device), predict(model, va, device)
        for lab in (("task129", "task134") if st == "task134" else ("task129",)):
            y_tr, y_va = labels[lab]
            for split, p, y, s_, o_ in (("train", p_tr, y_tr, a_stm_tr, a_opp_tr), ("val", p_va, y_va, a_stm_va, a_opp_va)):
                name = f"student {st} vs labels {lab}, {split}"
                f("")
                summary[name] = table(name, p - y, s_, o_, f)
    json.dump(summary, open(os.path.join(a.out, "residual.json"), "w"), indent=1)
    f(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
