#!/usr/bin/env python3
"""Compare game rows of two yane-vs-yane TSVs, ignoring wall-clock columns (elapsed_ms, *_avg_ms)."""
import sys
KEEP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 20]  # drop 6 elapsed_ms, 10 cand_avg_ms, 14 base_avg_ms, 15-19 running W/D/L/llr/verdict
def rows(p):
    out = []
    for line in open(p, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) == 21 and f[0].isdigit():
            out.append(tuple(f[i] for i in KEEP))
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
n = min(len(a), len(b))
diff = [i for i in range(n) if a[i] != b[i]]
print(f"compared={n} (a={len(a)} b={len(b)}) mismatches={len(diff)}" + (f" first={a[diff[0]]} vs {b[diff[0]]}" if diff else ""))
