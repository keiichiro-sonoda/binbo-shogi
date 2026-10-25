#!/usr/bin/env python3
"""Cross-check the Rust NNUE forward pass against the Python model.

For a diverse sample of positions (stride-sampled across hcpe shards), this:

  1. loads the trained MinimalNNUE from its .pt and computes the logit in Python
     (the same forward used at training time),
  2. emits each position's SFEN,
  3. runs the Rust `nnue-eval` binary (which loads the exported f32 weights and
     runs its own forward) on those SFENs,
  4. compares the two logits per position.

NNUE forward is floating point, so bit-exact equality is not expected; both sides
compute in f32 and differ only by summation/rounding order. The bar is a small
absolute error. Reports max/mean absolute error and the worst cases.

Exit code is non-zero if the max absolute error exceeds --tol.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import cshogi
import numpy as np
import torch

try:
    from . import kp
    from .nnue import CanonicalNNUE, MinimalNNUE
except ImportError:
    import kp
    from nnue import CanonicalNNUE, MinimalNNUE


def stride_indices(n: int, want: int) -> list[int]:
    if want <= 0 or n <= want:
        return list(range(n))
    step = n / want
    return sorted({min(n - 1, int(i * step)) for i in range(want)})


def collect(hcpe_files: list[str], limit: int, canonical: bool = False, shared: bool = False,
            ka: bool = False, kav2: bool = False):
    per_file = max(1, (limit + len(hcpe_files) - 1) // len(hcpe_files))
    board = cshogi.Board()
    sfens: list[str] = []
    index_rows: list[tuple[int, ...]] = []
    turns = {"black": 0, "white": 0}
    for path in hcpe_files:
        entries = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
        for i in stride_indices(len(entries), per_file):
            if len(sfens) >= limit:
                break
            board.set_hcp(entries[i]["hcp"])
            sfens.append(board.sfen())
            if kav2:
                index_rows.append(kp.canonical_shared_kav2_row(board))
            elif ka:
                index_rows.append(kp.canonical_shared_ka_row(board))
            elif shared:
                index_rows.append(kp.canonical_shared_row(board))
            else:
                index_rows.append(kp.board_to_kp_indices(board, canonical=canonical))
            turns["black" if board.turn == cshogi.BLACK else "white"] += 1
        if len(sfens) >= limit:
            break
    return sfens, index_rows, turns


def python_logits(model, index_rows: list[tuple[int, ...]], canonical: bool = False,
                  shared: bool = False) -> np.ndarray:
    if shared:
        own_flat: list[int] = []
        opp_flat: list[int] = []
        own_off: list[int] = []
        opp_off: list[int] = []
        oc = pc = 0
        for row in index_rows:
            # Generic own/opp split: own=(fpp-1)//2+1 (body+side), opp=rest.
            # shared HalfKP fpp=77 -> 39/38; HalfKA fpp=81 -> 41/40.
            own_n = (len(row) - 1) // 2 + 1
            own, opp = row[:own_n], row[own_n:]
            own_off.append(oc)
            opp_off.append(pc)
            own_flat.extend(own)
            opp_flat.extend(opp)
            oc += len(own)
            pc += len(opp)
        with torch.no_grad():
            logits = model(
                torch.tensor(own_flat, dtype=torch.long),
                torch.tensor(own_off, dtype=torch.long),
                torch.tensor(opp_flat, dtype=torch.long),
                torch.tensor(opp_off, dtype=torch.long),
            )
        return logits.numpy().astype(np.float64)
    if canonical:
        own_flat: list[int] = []
        opp_flat: list[int] = []
        own_off: list[int] = []
        opp_off: list[int] = []
        oc = pc = 0
        for row in index_rows:
            own, opp = kp.split_halves(row)
            own_off.append(oc)
            opp_off.append(pc)
            own_flat.extend(own)
            opp_flat.extend(opp)
            oc += len(own)
            pc += len(opp)
        with torch.no_grad():
            logits = model(
                torch.tensor(own_flat, dtype=torch.long),
                torch.tensor(own_off, dtype=torch.long),
                torch.tensor(opp_flat, dtype=torch.long),
                torch.tensor(opp_off, dtype=torch.long),
            )
        return logits.numpy().astype(np.float64)

    flat: list[int] = []
    offsets: list[int] = []
    cursor = 0
    for row in index_rows:
        offsets.append(cursor)
        flat.extend(row)
        cursor += len(row)
    flat_t = torch.tensor(flat, dtype=torch.long)
    off_t = torch.tensor(offsets, dtype=torch.long)
    with torch.no_grad():
        logits = model(flat_t, off_t)
    return logits.numpy().astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hcpe", nargs="+", help="hcpe shard file(s) to sample from")
    parser.add_argument(
        "--model-dir",
        default=os.path.expanduser("~/binbo-shogi-data/models/model_wd1e-3"),
    )
    parser.add_argument("--weights", default="nnue_best.pt")
    parser.add_argument("--bin", default="nnue_f32.bin")
    parser.add_argument("--binary", default="target/release/binbo-shogi")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-3, help="max allowed absolute logit error")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "model_config.json"), encoding="utf-8") as f:
        import json
        config = json.load(f)
    kav2 = bool(config.get("canonical_shared_kav2", False))
    ka = bool(config.get("canonical_shared_ka", False))
    shared = bool(config.get("canonical_shared", False)) or ka or kav2  # all use the shared path
    canonical = bool(config.get("canonical", False))
    model_cls = CanonicalNNUE if (canonical or shared) else MinimalNNUE
    model = model_cls(
        total_features=int(config["total_features"]),
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.weights), map_location="cpu"))
    model.eval()

    print(f"model_dir={args.model_dir} weights={args.weights} bin={args.bin} canonical={canonical} shared={shared} ka={ka} kav2={kav2}")
    print(f"hcpe_files={len(args.hcpe)} limit={args.limit} tol={args.tol:g}")
    sfens, index_rows, turns = collect(args.hcpe, args.limit, canonical, shared, ka, kav2)
    n = len(sfens)
    if n == 0:
        print("no positions", file=sys.stderr)
        return 1
    print(f"positions={n} turn_black={turns['black']} turn_white={turns['white']}")

    py = python_logits(model, index_rows, canonical, shared)

    with tempfile.TemporaryDirectory() as tmp:
        sfen_path = os.path.join(tmp, "sfens.txt")
        out_path = os.path.join(tmp, "rust_logits.txt")
        with open(sfen_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sfens) + "\n")
        proc = subprocess.run(
            [args.binary, "nnue-eval", os.path.join(args.model_dir, args.bin), sfen_path, out_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("rust nnue-eval failed:", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 1
        with open(out_path, encoding="utf-8") as f:
            rust = np.array([float(x) for x in f.read().split()], dtype=np.float64)

    if rust.shape[0] != n:
        print(f"count mismatch: rust={rust.shape[0]} python={n}", file=sys.stderr)
        return 1

    abs_err = np.abs(py - rust)
    rel_err = abs_err / (np.abs(py) + 1e-6)
    order = np.argsort(abs_err)[::-1]

    print("")
    print("=== logit numeric agreement (Rust vs Python) ===")
    print(f"checked={n}")
    print(f"python_logit_range=[{py.min():.4f}, {py.max():.4f}]")
    print(f"max_abs_err={abs_err.max():.3e}")
    print(f"mean_abs_err={abs_err.mean():.3e}")
    print(f"median_abs_err={np.median(abs_err):.3e}")
    print(f"max_rel_err={rel_err.max():.3e}")
    print(f"count_abs_err>1e-4={int((abs_err > 1e-4).sum())}")
    print(f"count_abs_err>{args.tol:g}={int((abs_err > args.tol).sum())}")
    print("--- worst cases ---")
    for k in order[: args.max_examples]:
        print(f"  abs_err={abs_err[k]:.3e} py={py[k]:.6f} rust={rust[k]:.6f} sfen='{sfens[k]}'")

    print("")
    ok = abs_err.max() <= args.tol
    print(f"tol={args.tol:g} max_abs_err={abs_err.max():.3e}")
    print("nnue_rust_matches_python=ok" if ok else "nnue_rust_matches_python=FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
