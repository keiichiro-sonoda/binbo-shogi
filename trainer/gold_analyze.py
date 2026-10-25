#!/usr/bin/env python3
"""Task 070 — gold-subset depth diagnostic (analysis only).

Reads a label_cost.py TSV (pos_id, depth, cp, is_mate, mate_val, elapsed_ms, capped)
produced at depths d12/d14/d16 on a FIXED position set, and answers the one question
Task 070 cares about: does d14 carry genuinely new information toward the d16 "gold"?

Treating d16 as the gold reference, on positions with a valid (non-mate, non-capped) cp
at ALL of d12/d14/d16 it reports:
  * closeness to d16: |cp_d12 - cp_d16| vs |cp_d14 - cp_d16| (median/mean/p90), and the
    fraction of positions where d14 is strictly closer to d16 than d12 (>50% = d14 adds info);
  * Delta cp(d12 -> d14) distribution (median/p90);
  * sign-flip rate vs d16 for d12 and d14 (d14 should disagree with d16 less often);
  * near-zero sign-flip rate, restricted to |cp_d16| < --near-zero (balanced positions);
  * label-time p50/p90/p99 per depth (non-capped; the per-position cost the loop pays).

Pure analysis: no engine, no torch. Example:
  $VENV_PY trainer/gold_analyze.py --tsv $LOGS_DIR/task070_gold.tsv --depths 12,14,16
"""
from __future__ import annotations

import argparse
import csv
import statistics


def pct(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, help="label_cost.py output TSV")
    ap.add_argument("--depths", default="12,14,16", help="ascending; last = gold reference")
    ap.add_argument("--near-zero", type=float, default=50.0, help="|cp_gold| threshold for balanced positions")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    gold = depths[-1]

    cp = {}        # (pos_id, depth) -> int cp (valid: non-mate, non-capped)
    mate = set()   # pos_id with a mate at any depth (excluded from cp metrics)
    times = {d: [] for d in depths}
    mate_ct = {d: 0 for d in depths}
    capped_ct = {d: 0 for d in depths}
    n_at = {d: 0 for d in depths}
    with open(args.tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            d = int(r["depth"])
            if d not in n_at:
                continue
            n_at[d] += 1
            is_mate = r["is_mate"] == "1"
            is_capped = r["capped"] == "1"
            if is_mate:
                mate_ct[d] += 1; mate.add(int(r["pos_id"]))
            if is_capped:
                capped_ct[d] += 1
            if (not is_capped) and r["elapsed_ms"]:
                try:
                    times[d].append(float(r["elapsed_ms"]))
                except ValueError:
                    pass
            if (not is_mate) and (not is_capped) and r["cp"] != "":
                cp[(int(r["pos_id"]), d)] = int(r["cp"])

    pids = sorted({pid for (pid, _d) in cp})
    full = [pid for pid in pids if pid not in mate and all((pid, d) in cp for d in depths)]

    print(f"=== gold diagnostic: {args.tsv} | depths={depths} gold=d{gold} ===")
    print(f"positions: total_sampled≈{max(n_at.values()) if n_at else 0}, "
          f"valid-cp-at-all-depths(non-mate,non-capped)={len(full)}")
    print("\n-- per-depth label time (non-capped, ms) + mate%/capped% --")
    print(f"{'depth':>5} {'n':>5} {'mate%':>6} {'capped%':>8} {'p50':>8} {'p90':>8} {'p99':>8}")
    for d in depths:
        n = max(1, n_at[d])
        print(f"{d:>5} {n_at[d]:>5} {100.0*mate_ct[d]/n:>5.1f}% {100.0*capped_ct[d]/n:>7.1f}% "
              f"{pct(times[d],0.50):>8.1f} {pct(times[d],0.90):>8.1f} {pct(times[d],0.99):>8.1f}")

    if not full:
        print("\nno positions valid at all depths -> cannot compute closeness")
        return 0

    d_lo, d_mid = depths[0], depths[1]   # 12, 14
    abs_lo = [abs(cp[(p, d_lo)] - cp[(p, gold)]) for p in full]
    abs_mid = [abs(cp[(p, d_mid)] - cp[(p, gold)]) for p in full]
    closer = sum(1 for p in full if abs(cp[(p, d_mid)] - cp[(p, gold)]) < abs(cp[(p, d_lo)] - cp[(p, gold)]))
    ties = sum(1 for p in full if abs(cp[(p, d_mid)] - cp[(p, gold)]) == abs(cp[(p, d_lo)] - cp[(p, gold)]))
    delta = [abs(cp[(p, d_mid)] - cp[(p, d_lo)]) for p in full]

    print(f"\n-- closeness to gold d{gold} (n={len(full)}) --")
    print(f"{'metric':>16} {'median':>8} {'mean':>8} {'p90':>8}")
    print(f"{'|d'+str(d_lo)+'-d'+str(gold)+'|':>16} {statistics.median(abs_lo):>8.1f} "
          f"{statistics.fmean(abs_lo):>8.1f} {pct(abs_lo,0.90):>8.1f}")
    print(f"{'|d'+str(d_mid)+'-d'+str(gold)+'|':>16} {statistics.median(abs_mid):>8.1f} "
          f"{statistics.fmean(abs_mid):>8.1f} {pct(abs_mid,0.90):>8.1f}")
    print(f"  d{d_mid} strictly closer to d{gold} than d{d_lo}: "
          f"{100.0*closer/len(full):.1f}%  (ties {100.0*ties/len(full):.1f}%, "
          f"d{d_lo} closer {100.0*(len(full)-closer-ties)/len(full):.1f}%)")
    print(f"  |Δcp(d{d_lo}->d{d_mid})|: median {statistics.median(delta):.1f}  p90 {pct(delta,0.90):.1f}")

    def signflip(d, subset):
        n = fl = 0
        for p in subset:
            a, g = cp[(p, d)], cp[(p, gold)]
            if a == 0 or g == 0:
                continue
            n += 1
            if (a > 0) != (g > 0):
                fl += 1
        return fl, n

    print(f"\n-- sign disagreement vs gold d{gold} (lower = better aligned) --")
    for d in (d_lo, d_mid):
        fl, n = signflip(d, full)
        print(f"  d{d} vs d{gold}: {100.0*fl/max(1,n):.1f}%  ({fl}/{n})")
    near = [p for p in full if abs(cp[(p, gold)]) < args.near_zero]
    print(f"  near-zero (|cp_d{gold}|<{args.near_zero:.0f}, n={len(near)}):")
    for d in (d_lo, d_mid):
        fl, n = signflip(d, near)
        print(f"    d{d} vs d{gold}: {100.0*fl/max(1,n):.1f}%  ({fl}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
