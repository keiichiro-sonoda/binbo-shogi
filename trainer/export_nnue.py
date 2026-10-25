#!/usr/bin/env python3
"""Export a trained MinimalNNUE (.pt) to a self-describing binary for Rust.

The Rust inference side (`src/eval/nnue.rs`) reads this file with no JSON
dependency: a small header gives the dimensions, then the seven tensors follow
as little-endian f32 in C (row-major) order, in a fixed sequence.

File layout (`nnue_f32.bin`):

  offset  bytes  content
  0       4      magic b"BNN1"
  4       4      total_features  (u32 LE)
  8       4      feature_dim     (u32 LE)
  12      4      hidden_dim      (u32 LE)
  16      4      scale_cp        (f32 LE)
  20      ...    tensors, each f32 LE, row-major, in this order:
                   1. feature_transform.weight  (total_features x feature_dim)
                   2. network.0.weight          (hidden_dim x feature_dim)   # Linear(feature_dim->hidden_dim)
                   3. network.0.bias            (hidden_dim)
                   4. network.2.weight          (hidden_dim x hidden_dim)
                   5. network.2.bias            (hidden_dim)
                   6. network.4.weight          (1 x hidden_dim)
                   7. network.4.bias            (1)

PyTorch Linear stores weight as (out_features, in_features) and computes
y = W x + b, so weight[i, j] is the coefficient from input j to output i. The
Rust loader keeps the same (out, in) row-major layout, so no transpose is
needed on either side.
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
import torch

try:
    from .quant_format import header as quant_header
except ImportError:
    from quant_format import header as quant_header

try:
    from . import kp
    from .nnue import QAT_FV_SCALE, QAT_WEIGHT_SCALE_BITS, fold_ft_factor, quantize_qat_state, tied_embedding_weight
except ImportError:
    import kp
    from nnue import QAT_FV_SCALE, QAT_WEIGHT_SCALE_BITS, fold_ft_factor, quantize_qat_state, tied_embedding_weight

MAGIC_NAIVE = b"BNN1"
MAGIC_TWOBAND = b"BNN2"  # Task 046: 2-band non-shared canonical HalfKP
MAGIC_SHARED = b"BNN3"   # Task 047: true canonical shared-W1 HalfKP
MAGIC_SHARED_KA = b"BNN4"  # Task 052: HalfKA shared band (kings as features)
MAGIC_SHARED_KAV2 = b"BNN5"  # Task 053: HalfKAv2 (KA minus anchor-king redundancy)
MAGIC_KPP_ZONE = b"BNN6"     # Task 053: HalfKP shared + king-zone KPP (2 tables, 4 acc)
MAGIC_QUANT = b"BNNQ"        # Task 058: quantization-aware shared HalfKP (int16 FT / int8 affine)

# (state_dict key, expected shape as a tuple of dim-names) in write order.
# `net_in_dim` = feature_dim (naive) or 2*feature_dim (canonical, concatenated halves).
TENSOR_ORDER = [
    ("feature_transform.weight", ("total_features", "feature_dim")),
    ("network.0.weight", ("hidden_dim", "net_in_dim")),
    ("network.0.bias", ("hidden_dim",)),
    ("network.2.weight", ("hidden_dim", "hidden_dim")),
    ("network.2.bias", ("hidden_dim",)),
    ("network.4.weight", ("one", "hidden_dim")),
    ("network.4.bias", ("one",)),
]


def export_kpp(args, config) -> int:
    """Export KppNNUE (Task 053, BNN6): two embedding tables + the 4*fd network.
    Header adds a 4th u32 (kpp_total) after scale_cp, so the layout is
    magic | kp_total | feature_dim | hidden_dim | scale_cp | kpp_total | tensors."""
    fd = int(config["feature_dim"])
    h = int(config["hidden_dim"])
    kp_total = int(config["total_features"])
    kpp_total = int(config["kpp_zone_total"])
    scale_cp = float(config["scale_cp"])
    net_in = 4 * fd
    order = [
        ("kp_emb.weight", (kp_total, fd)),
        ("kpp_emb.weight", (kpp_total, fd)),
        ("network.0.weight", (h, net_in)),
        ("network.0.bias", (h,)),
        ("network.2.weight", (h, h)),
        ("network.2.bias", (h,)),
        ("network.4.weight", (1, h)),
        ("network.4.bias", (1,)),
    ]
    state = torch.load(os.path.join(args.model_dir, args.weights), map_location="cpu")
    out_path = os.path.join(args.model_dir, args.out)
    written = 0
    with open(out_path, "wb") as f:
        f.write(MAGIC_KPP_ZONE)
        f.write(struct.pack("<III", kp_total, fd, h))
        f.write(struct.pack("<f", scale_cp))
        f.write(struct.pack("<I", kpp_total))  # BNN6 extra header field
        for key, shape in order:
            t = state[key]
            if tuple(t.shape) != shape:
                raise AssertionError(f"{key}: shape {tuple(t.shape)} != expected {shape}")
            arr = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
            f.write(arr.tobytes())
            written += arr.size
            print(f"  wrote {key:28s} shape={tuple(t.shape)} floats={arr.size}")
    print(f"kpp_zone magic=BNN6 kp_total={kp_total} kpp_total={kpp_total} fd={fd} hidden={h} scale_cp={scale_cp:g}")
    print(f"floats={written} bytes={os.path.getsize(out_path)} out={out_path}")
    print("export_nnue=ok")
    return 0


def export_qat(args, config) -> int:
    """Export QATSharedNNUE (Task 058, BNNQ): quantize the trained float weights to the
    same integer grids the model's fake-quant used (trainer/nnue.quantize_qat_state), so
    binbo's Mode::Quant forward (src/eval/nnue.rs::load_quant) reproduces the QAT float
    eval. Byte layout matches the Rust reader and trainer/yane_to_binbo.convert_quant:
    magic | total | fd | hidden | fv_scale(i32) | weight_scale_bits(u32) |
    emb(i16) ft_bias(i16) w0(i8) b0(i32) w1(i8) b1(i32) w2(i8) b2(i32).
    Task 149: asymmetric hidden widths use the explicit BNQ2 header in quant_format."""
    fd = int(config["feature_dim"])
    hidden = int(config["hidden_dim"])
    hidden2 = int(config.get("hidden_dim2", hidden))  # Task 149
    total = int(config["total_features"])
    scale_cp = float(config["scale_cp"])
    state = torch.load(os.path.join(args.model_dir, args.weights), map_location="cpu")
    state = fold_ft_factor(state)   # Task 125 factorizer: merge V[p] into every king's row first
    if bool(config.get("qat_tie", False)):
        # Tied model (Task 059): collapse the [total+1, fd] embedding to the deployed
        # [total, fd] (equalised collapsed rows, side rows -> 0) so binbo_to_yane can write
        # it bit-exact. build_collapse_map derives the mapping from the verified remap R.
        try:
            from .binbo_to_yane import build_collapse_map
        except ImportError:
            from binbo_to_yane import build_collapse_map
        state["feature_transform.weight"] = tied_embedding_weight(
            state["feature_transform.weight"], build_collapse_map(vm=bool(config.get("vm", False))))
    q = quantize_qat_state(state, scale_cp)
    out_path = os.path.join(args.model_dir, args.out)
    body = bytearray()
    body += quant_header(total, fd, hidden, hidden2, QAT_FV_SCALE, QAT_WEIGHT_SCALE_BITS)
    shapes = dict(emb=(total, fd), ft_bias=(fd,), w0=(hidden, 2 * fd), b0=(hidden,),
                  w1=(hidden2, hidden), b1=(hidden2,), w2=(1, hidden2), b2=(1,))
    for key, dt in (("emb", "<i2"), ("ft_bias", "<i2"), ("w0", "<i1"), ("b0", "<i4"),
                    ("w1", "<i1"), ("b1", "<i4"), ("w2", "<i1"), ("b2", "<i4")):
        if q[key].shape != shapes[key]:
            raise ValueError(f"{key}: {q[key].shape} != configured {shapes[key]}")
        body += np.ascontiguousarray(q[key], dtype=dt).tobytes()
    with open(out_path, "wb") as f:
        f.write(body)
    print(f"qat magic={body[:4].decode()} total={total} fd={fd} hidden={hidden}/{hidden2} fv_scale={QAT_FV_SCALE} "
          f"weight_scale_bits={QAT_WEIGHT_SCALE_BITS} scale_cp={scale_cp:g}")
    for key in ("emb", "ft_bias", "w0", "b0", "w1", "b1", "w2", "b2"):
        a = q[key]
        print(f"  {key:8s} shape={a.shape} dtype={a.dtype} range=[{int(a.min())},{int(a.max())}]")
    print(f"bytes={os.path.getsize(out_path)} out={out_path}")
    print("export_nnue=ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="dir with the .pt and model_config.json")
    parser.add_argument("--weights", default="nnue_best.pt")
    parser.add_argument("--out", default="nnue_f32.bin")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if bool(config.get("qat", False)):
        return export_qat(args, config)
    kpp_zone = bool(config.get("kpp_zone", False))
    if kpp_zone:
        return export_kpp(args, config)
    kav2 = bool(config.get("canonical_shared_kav2", False))
    ka = bool(config.get("canonical_shared_ka", False))
    shared = bool(config.get("canonical_shared", False))
    two_band = bool(config.get("canonical", False))
    # All canonical variants (two-band / shared / KA / KAv2) concatenate two
    # accumulators -> network.0 input is 2*feature_dim; naive sums into one.
    two_acc = kav2 or ka or shared or two_band
    feature_dim = int(config["feature_dim"])
    dims = {
        "total_features": int(config["total_features"]),
        "feature_dim": feature_dim,
        "net_in_dim": 2 * feature_dim if two_acc else feature_dim,
        "hidden_dim": int(config["hidden_dim"]),
        "one": 1,
    }
    scale_cp = float(config["scale_cp"])
    magic = (MAGIC_SHARED_KAV2 if kav2 else MAGIC_SHARED_KA if ka else MAGIC_SHARED if shared
             else MAGIC_TWOBAND if two_band else MAGIC_NAIVE)
    print(f"kav2={kav2} ka={ka} shared={shared} two_band={two_band} magic={magic.decode()} "
          f"total_features={dims['total_features']} net_in_dim={dims['net_in_dim']}")

    pt_path = os.path.join(args.model_dir, args.weights)
    state = torch.load(pt_path, map_location="cpu")

    out_path = os.path.join(args.model_dir, args.out)
    written_floats = 0
    with open(out_path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<III", dims["total_features"], dims["feature_dim"], dims["hidden_dim"]))
        f.write(struct.pack("<f", scale_cp))
        for key, shape_names in TENSOR_ORDER:
            tensor = state[key]
            expected = tuple(dims[name] for name in shape_names)
            if tuple(tensor.shape) != expected:
                raise AssertionError(f"{key}: shape {tuple(tensor.shape)} != expected {expected}")
            array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
            f.write(array.tobytes())
            written_floats += array.size
            print(f"  wrote {key:28s} shape={tuple(tensor.shape)} floats={array.size}")

    expected_floats = (
        dims["total_features"] * dims["feature_dim"]
        + dims["hidden_dim"] * dims["net_in_dim"] + dims["hidden_dim"]
        + dims["hidden_dim"] * dims["hidden_dim"] + dims["hidden_dim"]
        + dims["hidden_dim"] + 1
    )
    assert written_floats == expected_floats, (written_floats, expected_floats)
    header_bytes = len(magic) + 4 * 3 + 4
    total_bytes = header_bytes + written_floats * 4
    actual_bytes = os.path.getsize(out_path)
    assert actual_bytes == total_bytes, (actual_bytes, total_bytes)

    print(f"total_features={dims['total_features']} feature_dim={dims['feature_dim']} "
          f"hidden_dim={dims['hidden_dim']} scale_cp={scale_cp:g}")
    print(f"floats={written_floats} bytes={actual_bytes} out={out_path}")
    print("export_nnue=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
