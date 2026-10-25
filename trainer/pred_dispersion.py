#!/usr/bin/env python3
"""Compare a trained model's predicted |cp| on decisive positions, train vs val.

Reads the prepare_data cache (train/val) and a trained model, runs forward only
(no training), and reports the mean predicted |cp| on decisive positions
(|target cp| >= threshold) separately for the train split (seen games) and the
val split (unseen games), plus prediction std.

If the model predicts large magnitude on train but compresses on val, the
"magnitude compression" diagnosed in Task 031-034 is a generalization gap
(overfit), not a lack of decisive data (Task 035 shows decisive is ~24%) nor a
capacity limit (Task 034 shows more capacity does not help).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

try:
    from . import kp
    from .nnue import CanonicalNNUE, MinimalNNUE
except ImportError:
    import kp
    from nnue import CanonicalNNUE, MinimalNNUE


def predict_cp(model, idx: np.ndarray, scale_cp: float, fpp: int,
               canonical: bool = False, shared: bool = False) -> np.ndarray:
    n = idx.shape[0]
    with torch.no_grad():
        if canonical or shared:
            if shared:
                # Canonical-shared layout: [0:39] own (body+side), [39:77] opp body.
                own = idx[:, :39]
                opp = idx[:, 39:77]
            else:
                # Two-band sorted-77: [0:38] own body, [38:76] opp body, [76] side.
                own = np.concatenate([idx[:, :38], idx[:, 76:77]], axis=1)
                opp = idx[:, 38:76]
            own_flat = torch.from_numpy(np.ascontiguousarray(own).reshape(-1).astype(np.int64))
            opp_flat = torch.from_numpy(np.ascontiguousarray(opp).reshape(-1).astype(np.int64))
            own_off = torch.arange(0, n * own.shape[1], own.shape[1], dtype=torch.long)
            opp_off = torch.arange(0, n * opp.shape[1], opp.shape[1], dtype=torch.long)
            return (model(own_flat, own_off, opp_flat, opp_off) * scale_cp).numpy()
        flat = torch.from_numpy(idx.reshape(-1).astype(np.int64))
        off = torch.arange(0, n * fpp, fpp, dtype=torch.long)
        return (model(flat, off) * scale_cp).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", help="prepare_data cache dir")
    parser.add_argument("model_dir", help="dir with nnue_best.pt and model_config.json")
    parser.add_argument("--weights", default="nnue_best.pt")
    parser.add_argument("--decisive-cp", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=60000)
    args = parser.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "model_config.json"), encoding="utf-8"))
    scale_cp = float(cfg["scale_cp"])
    shared = bool(cfg.get("canonical_shared", False))
    canonical = bool(cfg.get("canonical", False))
    total_features = int(cfg.get("total_features", kp.TOTAL_FEATURES))
    model_cls = CanonicalNNUE if (canonical or shared) else MinimalNNUE
    model = model_cls(total_features, int(cfg["feature_dim"]), int(cfg["hidden_dim"]))
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.weights), map_location="cpu"))
    model.eval()
    meta = json.load(open(os.path.join(args.cache_dir, "meta.json"), encoding="utf-8"))
    fpp = meta["features_per_position"]

    print(f"model_dir={args.model_dir} weights={args.weights} decisive_cp>={args.decisive_cp} "
          f"canonical={canonical} shared={shared}")
    for name in ("train", "val"):
        idx = np.load(os.path.join(args.cache_dir, f"{name}_indices.npy"))[: args.limit]
        ev = np.load(os.path.join(args.cache_dir, f"{name}_evals.npy")).astype(np.int32)[: args.limit]
        pred = predict_cp(model, idx, scale_cp, fpp, canonical, shared)
        dec = np.abs(ev) >= args.decisive_cp
        print(f"\n[{name}] n={len(ev)} decisive_frac={dec.mean():.3f}")
        print(f"  decisive target |cp| mean = {np.abs(ev[dec]).mean():.0f}")
        print(f"  decisive PRED   |cp| mean = {np.abs(pred[dec]).mean():.0f}")
        print(f"  pred std = {pred.std():.0f}   target std = {ev.std():.0f}")
    print("\npred_dispersion=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
