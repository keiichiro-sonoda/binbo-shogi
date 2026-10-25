#!/usr/bin/env python3
"""Evaluate a QATSharedNNUE checkpoint two ways and report the quantization residual (Task 058).

For each SFEN we compute, via binbo's own shared-HalfKP feature extraction
(``kp.canonical_shared_halves``, the trusted side):

  - ``float`` : the CONTINUOUS float eval the trainer optimized (fake_quant OFF), in cp.
  - ``int``   : the DEPLOYED integer eval -- the exact YaneuraOu-style int arithmetic on
                the quantized weights (``nnue.quantize_qat_state``), binbo-numbered. This
                mirrors ``src/eval/nnue.rs::quant_forward`` / the BNNQ deployment, so it is
                the same value binbo's ``nnue-eval`` produces on the exported BNNQ file.

The residual ``int - float`` is the quantization error: small (a few cp) means the trained
eval loads undistorted. ``--out-float`` / ``--out-int`` write per-line cp so an external
driver (scripts/qat_residual.sh) can also confirm the Rust BNNQ forward == this ``int``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cshogi
import numpy as np
import torch

try:
    from . import kp
    from .nnue import (QATSharedNNUE, QAT_FV_SCALE, QAT_WEIGHT_SCALE_BITS,
                       fold_ft_factor, quantize_qat_state, tied_embedding_weight)
    from .binbo_to_yane import build_collapse_map
except ImportError:
    import kp
    from nnue import (QATSharedNNUE, QAT_FV_SCALE, QAT_WEIGHT_SCALE_BITS,
                      fold_ft_factor, quantize_qat_state, tied_embedding_weight)
    from binbo_to_yane import build_collapse_map


def _trunc_div(x: int, d: int) -> int:
    q = abs(x) // d
    return q if x >= 0 else -q


def load_model(model_dir: str, weights: str) -> tuple[QATSharedNNUE, dict]:
    with open(os.path.join(model_dir, "model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if not config.get("qat", False):
        raise SystemExit(f"{model_dir} is not a QAT model (config.qat is false)")
    tie_map = build_collapse_map(vm=bool(config.get("vm", False))) if config.get("qat_tie", False) else None
    model = QATSharedNNUE(
        total_features=int(config["total_features"]),
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        hidden_dim2=config.get("hidden_dim2"),  # Task 149
        factorize=bool(config.get("factorize", False)),
        fake_quant=False,  # continuous reference forward
        scale_cp=float(config["scale_cp"]),
        tie_collapse_map=tie_map,
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, weights), map_location="cpu"))
    model.eval()
    return model, config


def float_eval(model: QATSharedNNUE, own: list[int], opp: list[int], scale_cp: float) -> float:
    own_t = torch.tensor(own, dtype=torch.long)
    opp_t = torch.tensor(opp, dtype=torch.long)
    z = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        logit = model(own_t, z, opp_t, z)
    return float(logit.item()) * scale_cp


def int_eval(qd: dict, own: list[int], opp: list[int], fd: int,
             shift: int = QAT_WEIGHT_SCALE_BITS, fv_scale: int = QAT_FV_SCALE) -> int:
    """Exact int forward on the quantized weights (== src/eval/nnue.rs::quant_forward)."""
    emb = qd["emb"]                         # int16 [total, fd]
    ftb = qd["ft_bias"].astype(np.int64)    # [fd]
    w0 = qd["w0"].astype(np.int64); b0 = qd["b0"].astype(np.int64)   # [hidden, 2fd], [hidden]
    w1 = qd["w1"].astype(np.int64); b1 = qd["b1"].astype(np.int64)   # [hidden, hidden]
    w2 = qd["w2"].astype(np.int64).reshape(-1); b2 = qd["b2"].astype(np.int64)  # [hidden], [1]
    acc_o = ftb.copy()
    for f in own:
        acc_o += emb[f]
    acc_p = ftb.copy()
    for f in opp:
        acc_p += emb[f]
    x = np.concatenate([np.clip(acc_o, 0, 127), np.clip(acc_p, 0, 127)])
    h0 = np.clip((b0 + w0 @ x) >> shift, 0, 127)
    h1 = np.clip((b1 + w1 @ h0) >> shift, 0, 127)
    o2 = int(b2[0] + int(w2 @ h1))
    return _trunc_div(o2, fv_scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("sfens")
    ap.add_argument("--weights", default="nnue_best.pt")
    ap.add_argument("--out-float", help="write continuous float cp per line")
    ap.add_argument("--out-int", help="write deployed int cp per line")
    args = ap.parse_args()

    model, config = load_model(args.model_dir, args.weights)
    fd = int(config["feature_dim"])
    scale_cp = float(config["scale_cp"])
    raw = torch.load(os.path.join(args.model_dir, args.weights), map_location="cpu")
    raw = fold_ft_factor(raw)  # Task 149: same fold-before-tie as export.
    if config.get("qat_tie", False):
        raw["feature_transform.weight"] = tied_embedding_weight(
            raw["feature_transform.weight"], build_collapse_map(vm=bool(config.get("vm", False))))
    qd = quantize_qat_state(raw, scale_cp)

    sfens = [l.strip() for l in open(args.sfens) if l.strip()]
    fcp, icp = [], []
    for s in sfens:
        own, opp = kp.canonical_shared_halves(cshogi.Board(s))
        fcp.append(float_eval(model, list(own), list(opp), scale_cp))
        icp.append(int_eval(qd, list(own), list(opp), fd))
    fcp = np.array(fcp); icp = np.array(icp, dtype=float)
    resid = icp - fcp
    if args.out_float:
        with open(args.out_float, "w") as f:
            f.write("\n".join(f"{v:.4f}" for v in fcp) + "\n")
    if args.out_int:
        with open(args.out_int, "w") as f:
            f.write("\n".join(str(int(v)) for v in icp) + "\n")
    print(f"N={len(sfens)} weights={args.weights} fake_quant_trained={config.get('qat_fake_quant')}")
    print(f"  float cp  range=[{fcp.min():.0f},{fcp.max():.0f}]  int cp range=[{icp.min():.0f},{icp.max():.0f}]")
    print(f"  residual (int-float): mean={resid.mean():+.2f}cp  std={resid.std():.2f}cp  "
          f"max|.|={np.abs(resid).max():.1f}cp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
