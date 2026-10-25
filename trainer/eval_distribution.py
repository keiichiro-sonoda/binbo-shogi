#!/usr/bin/env python3
"""Summarize the evaluation (centipawn) distribution of a teacher-data run.

Reads the hcpe shards of a `gen-teacher` run (manifest.json), takes the `eval`
field (Gikou cp, side-to-move viewpoint) of every position, and reports how the
evaluations are distributed: histogram, bucket fractions (near-equal .. decisive),
|cp| quantiles, sign balance, and the same after clamping to +/-C (the cp
regression target of Task 033).

The point (Task 035) is to test the hypothesis that Gikou-vs-Gikou games dwell on
near-equal positions and contain few clearly decisive ones, which would explain
why the student cannot learn large evaluations (Task 031-034). Counting only; no
training.
"""
from __future__ import annotations

import argparse
import json
import os

import cshogi
import numpy as np


def load_evals(run_dir: str) -> np.ndarray:
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    shards = [s for s in manifest["shards"] if s["done"]]
    shards.sort(key=lambda s: s["index"])
    parts = []
    for shard in shards:
        path = os.path.join(run_dir, shard["file"])
        entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
        parts.append(entries["eval"].astype(np.int32))
    return np.concatenate(parts), len(shards)


def pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="gen-teacher output dir (with manifest.json)")
    parser.add_argument("--clamp", type=int, default=5000, help="clamp magnitude (Task 033 target)")
    args = parser.parse_args()

    evals, n_shards = load_evals(args.run_dir)
    n = evals.size
    a = np.abs(evals)

    print(f"run_dir={args.run_dir}")
    print(f"shards={n_shards} positions={n}")
    print(f"eval(cp, side-to-move): min={evals.min()} max={evals.max()} "
          f"mean={evals.mean():.1f} median={int(np.median(evals))} std={evals.std():.1f}")
    print(f"|cp|: mean={a.mean():.1f} median={int(np.median(a))}")

    print("\n=== |cp| quantiles ===")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q:<4} = {int(np.percentile(a, q))}")

    print("\n=== buckets by |cp| (magnitude of advantage) ===")
    buckets = [
        ("near-equal  |cp|<200      ", (a < 200)),
        ("moderate    200<=|cp|<600 ", (a >= 200) & (a < 600)),
        ("clear       600<=|cp|<1500", (a >= 600) & (a < 1500)),
        ("decisive    1500<=|cp|<5000", (a >= 1500) & (a < 5000)),
        ("mate-ish    |cp|>=5000    ", (a >= 5000)),
    ]
    for label, mask in buckets:
        c = int(mask.sum())
        print(f"  {label}: {c:>8} ({pct(c, n)})")

    print("\n=== fraction with |cp| >= threshold ===")
    for t in (200, 500, 1000, 1500, 2000, 2600, 5000, 10000):
        c = int((a >= t).sum())
        print(f"  |cp|>={t:<5}: {c:>8} ({pct(c, n)})")

    print("\n=== sign balance (side-to-move viewpoint) ===")
    pos = int((evals > 0).sum())
    neg = int((evals < 0).sum())
    zero = int((evals == 0).sum())
    print(f"  positive(手番有利)={pos} ({pct(pos, n)})  negative={neg} ({pct(neg, n)})  zero={zero} ({pct(zero, n)})")
    dec = a >= 1500
    dec_pos = int((evals[dec] > 0).sum())
    dec_neg = int((evals[dec] < 0).sum())
    nd = int(dec.sum())
    if nd:
        print(f"  within decisive(|cp|>=1500): positive={dec_pos} ({pct(dec_pos, nd)})  negative={dec_neg} ({pct(dec_neg, nd)})")

    print(f"\n=== after clamp to +/-{args.clamp} (Task 033 target) ===")
    c = np.clip(evals, -args.clamp, args.clamp)
    ca = np.abs(c)
    clamped = int((a > args.clamp).sum())
    print(f"  clamped positions(|cp|>{args.clamp}): {clamped} ({pct(clamped, n)})")
    print(f"  clamped |cp|: mean={ca.mean():.1f} median={int(np.median(ca))} p90={int(np.percentile(ca,90))} p99={int(np.percentile(ca,99))}")

    print("\n=== signed histogram (cp bins, % of positions) ===")
    edges = [-100000, -5000, -2600, -1500, -600, -200, 0, 200, 600, 1500, 2600, 5000, 100000]
    hist, _ = np.histogram(evals, bins=edges)
    labels = ["<-5000", "-5000..-2600", "-2600..-1500", "-1500..-600", "-600..-200", "-200..0",
              "0..200", "200..600", "600..1500", "1500..2600", "2600..5000", ">5000"]
    for lab, h in zip(labels, hist):
        bar = "#" * int(round(60.0 * h / n))
        print(f"  {lab:>14}: {h:>8} ({pct(h, n)}) {bar}")

    print("\neval_distribution=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
