"""Task 135: where does one training step of the adopted recipe spend its time? (measurement only)

The Task 129 recipe (QATSharedNNUE fd256/h32, --qat --qat-tie --factorize, mirror, Huber cp,
batch 1024, Adam lr 6e-5, 64M rows via --mmap) runs at ~26k positions/s (2,454 s per 63.95M-row
epoch = 39 ms per step). This script rebuilds the same model / data path as trainer/train_nnue.py
`run_batches` (shared branch) and times each stage of a step with torch.cuda.synchronize() between
stages, so the stage times add up to a (slightly slower) step:

  S0 rows    slice of the epoch's random order
  S1 gather  indices[rows] from the memory-mapped cache (random rows of a 19.7 GB file)
  S2 long    .long() + mirror flip (torch.where through mirror_perm), on CPU
  S3 h2d     own/opp split, flatten, copy to GPU (+ target copy, offsets)
  S4 fwd     forward (ft_weight merge + fake quant of the full table + embedding_bag + MLP)
  S5 bwd     loss + backward
  S6 step    optimizer.step (dense Adam over every parameter)
  S7 item    loss.item() (the per-step host sync the trainer does)

Also: the same steps WITHOUT the intermediate syncs (true step time), and a torch.profiler table of
the top CUDA kernels. Nothing is trained or saved; the trainer is not changed.

  PYTHONPATH=. venv/bin/python experiments/alphazero/t135_train_profile.py --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from trainer import kp, mirror  # noqa: E402
from trainer.binbo_to_yane import build_collapse_map  # noqa: E402
from trainer.nnue import QATSharedNNUE  # noqa: E402

DATA = os.path.expanduser(os.environ.get("DATA_ROOT", "~/binbo-shogi-data"))
CACHE = f"{DATA}/caches/shared_64m_qs129"
MIRROR = f"{DATA}/caches/shared_8m_mirror_perm_shared.npy"
STAGES = ["S0_rows", "S1_gather", "S2_long_mirror", "S3_h2d", "S4_fwd", "S5_bwd", "S6_step", "S7_item"]


def build(device, sparse=False, fused=False):
    model = QATSharedNNUE(sparse_ft=sparse, total_features=kp.CANONICAL_SHARED_TOTAL, feature_dim=256, hidden_dim=32,
                          fake_quant=True, scale_cp=600.0, tie_collapse_map=build_collapse_map(vm=False),
                          factorize=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=6e-5, **({"fused": True} if fused else {}))
    return model, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20271350)
    ap.add_argument("--profile-steps", type=int, default=30)
    ap.add_argument("--sparse-ft", action="store_true", help="Task 135 fast option")
    ap.add_argument("--fused-adam", action="store_true", help="Task 135 fast option")
    a = ap.parse_args()
    device = "cuda"
    idx = torch.from_numpy(np.load(f"{CACHE}/train_indices.npy", mmap_mode="r"))
    tgt = torch.from_numpy(np.load(f"{CACHE}/train_evals.npy").astype(np.float32)).clamp(-2000, 2000)
    perm = torch.from_numpy(mirror.build_mirror_perm_shared(MIRROR)).long()
    n, fpp = idx.shape
    own_n = (fpp - 1) // 2 + 1
    opp_n = fpp - own_n
    print(f"rows={n} fpp={fpp} dtype={idx.dtype} file_GB={idx.numel() * idx.element_size() / 1e9:.1f}")
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    order = torch.randperm(n, generator=gen)
    print(f"randperm(64M) {time.time() - t0:.2f}s (once per epoch)")
    model, opt = build(device, a.sparse_ft, a.fused_adam)
    print(f"sparse_ft={a.sparse_ft} fused_adam={a.fused_adam}")
    crit = nn.HuberLoss(delta=600.0)
    b = a.batch

    def step(start, timed):
        t = {}
        sync = torch.cuda.synchronize if timed else (lambda: None)
        c = time.perf_counter()

        def mark(k):
            nonlocal c
            sync()
            now = time.perf_counter()
            t[k] = now - c
            c = now
        rows = order[start:start + b]
        mark("S0_rows")
        bi = idx[rows]
        mark("S1_gather")
        bi = bi.long()
        flip = torch.rand(b, generator=gen) < 0.5
        bi = torch.where(flip.unsqueeze(1), perm[bi], bi)
        mark("S2_long_mirror")
        target = tgt[rows].to(device)
        own = bi[:, :own_n].reshape(-1).to(device)
        opp = bi[:, own_n:].reshape(-1).to(device)
        oo = torch.arange(0, b * own_n, own_n, dtype=torch.long, device=device)
        po = torch.arange(0, b * opp_n, opp_n, dtype=torch.long, device=device)
        mark("S3_h2d")
        opt.zero_grad(set_to_none=True)
        logits = model(own, oo, opp, po)
        mark("S4_fwd")
        loss = crit(logits * 600.0, target)
        loss.backward()
        mark("S5_bwd")
        opt.step()
        mark("S6_step")
        loss.item()
        mark("S7_item")
        return t

    cur = 0
    for _ in range(a.warmup):
        step(cur, True)
        cur += b
    acc = {k: [] for k in STAGES}
    for _ in range(a.steps):
        for k, v in step(cur, True).items():
            acc[k].append(v)
        cur += b
    tot = sum(np.mean(v) for v in acc.values())
    print(f"\n== synced stages, {a.steps} steps of {b} (after {a.warmup} warm-up) ==")
    print(f"{'stage':16s} {'mean ms':>8s} {'p50':>7s} {'p90':>7s} {'share':>6s}")
    for k in STAGES:
        v = np.array(acc[k]) * 1e3
        print(f"{k:16s} {v.mean():8.2f} {np.median(v):7.2f} {np.percentile(v, 90):7.2f} {v.mean() / tot / 1e3:6.1%}")
    print(f"{'sum':16s} {tot * 1e3:8.2f}")

    # Unsynced: the real step time (GPU work overlaps the next step's CPU work).
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        step(cur, False)
        cur += b
    torch.cuda.synchronize()
    real = (time.perf_counter() - t0) / a.steps
    print(f"\nunsynced real step = {real * 1e3:.2f} ms -> {b / real:,.0f} pos/s "
          f"(epoch of {n:,} rows = {n / b * real / 60:.1f} min; trainer measured 2454 s = 40.9 min)")

    # Same rows again = page cache hot for those rows: isolates the disk part of S1.
    hot = []
    for i in range(a.steps):
        rows = order[a.warmup * b + i * b: a.warmup * b + (i + 1) * b]
        c = time.perf_counter()
        _ = idx[rows]
        hot.append(time.perf_counter() - c)
    print(f"S1 gather on rows just read (page cache hot): {np.mean(hot) * 1e3:.2f} ms")

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.profile_steps):
            step(cur, False)
            cur += b
        torch.cuda.synchronize()
    print(f"\n== torch.profiler, {a.profile_steps} steps: top CUDA kernels ==")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))


if __name__ == "__main__":
    main()
