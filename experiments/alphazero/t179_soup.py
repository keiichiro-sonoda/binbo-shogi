#!/usr/bin/env python3
"""Task 179: weight averaging ("model soup") of fine-tunes that share one parent.

    soup = base + alpha * mean(member - base)

alpha = 1 is the plain average of the members; alpha > 1 extrapolates along their mean
direction ("task arithmetic"). Float tensors are averaged in float64; integer buffers
(the Factorizer's collapse_map) must be identical across all checkpoints and are copied.
The result is a model dir (nnue_final.pt + model_config.json + soup.json) that the usual
export (trainer.export_nnue -> trainer.binbo_to_yane -> 0cp gate) turns into an nn.bin.

    python t179_soup.py OUT_DIR --base task154_R_result30 --members a,b,c [--alpha 1.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

MODELS = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data")) / "models"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    path = MODELS / name / "nnue_final.pt"
    return path, torch.load(path, map_location="cpu")


def soup(base, members, alpha):
    out, stats = {}, {}
    for key, b in base.items():
        if not torch.is_floating_point(b):
            for m in members:
                if not torch.equal(m[key], b):
                    raise ValueError(f"integer buffer {key} differs between checkpoints")
            out[key] = b.clone()
            continue
        b64 = b.double()
        deltas = [m[key].double() - b64 for m in members]
        mean = sum(deltas) / len(deltas)
        out[key] = (b64 + alpha * mean).to(b.dtype)
        stats[key] = {"base_norm": float(b64.norm()), "mean_delta_norm": float(mean.norm()),
                      "member_delta_norms": [float(d.norm()) for d in deltas]}
    return out, stats


def cosines(base, members):
    """Cosine similarity between the members' whole-model deltas (float tensors only)."""
    flat = [torch.cat([(m[k].double() - base[k].double()).flatten() for k in base if torch.is_floating_point(base[k])])
            for m in members]
    return [[float(torch.nn.functional.cosine_similarity(a, b, dim=0)) for b in flat] for a in flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", required=True)
    ap.add_argument("--members", required=True, help="comma-separated model names under models/")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    names = args.members.split(",")
    base_path, base = load(args.base)
    loaded = [load(n) for n in names]
    for n, (_, m) in zip(names, loaded):
        if m.keys() != base.keys() or any(m[k].shape != base[k].shape for k in base):
            raise ValueError(f"{n}: checkpoint layout differs from {args.base}")
    config = json.loads((MODELS / args.base / "model_config.json").read_text(encoding="utf-8"))
    for n in names:
        if json.loads((MODELS / n / "model_config.json").read_text(encoding="utf-8")) != config:
            raise ValueError(f"{n}: model_config.json differs from {args.base}")
    state, stats = soup(base, [m for _, m in loaded], args.alpha)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out / "nnue_final.pt")
    shutil.copy(MODELS / args.base / "model_config.json", args.out / "model_config.json")
    record = {"base": args.base, "base_sha256": sha256(base_path), "alpha": args.alpha,
              "members": {n: sha256(p) for n, (p, _) in zip(names, loaded)},
              "delta_cosines": cosines(base, [m for _, m in loaded]), "tensors": stats,
              "out_sha256": sha256(args.out / "nnue_final.pt")}
    (args.out / "soup.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    ft = stats["feature_transform.weight"]
    print(f"T179_SOUP {args.out.name} alpha={args.alpha} members={len(names)} "
          f"ft_mean_delta={ft['mean_delta_norm']:.4f} ft_member_deltas={[round(x, 4) for x in ft['member_delta_norms']]}",
          flush=True)


if __name__ == "__main__":
    main()
