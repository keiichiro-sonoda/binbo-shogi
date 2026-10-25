#!/usr/bin/env python3
"""Bridge: convert a YaneuraOu halfkp_256x2-32-32 nn.bin to binbo BNN3, and provide a
bit-exact reference forward (Task 057).

Two things live here:

1. ``yane_forward_int(parsed, board)`` — replicates YaneuraOu's *exact* integer NNUE
   forward (FT clamp[0,127] -> affine -> ClippedReLU(x>>6, clamp[0,127]) -> affine ...
   -> out/FV_SCALE, C-truncating division) using binbo's feature extraction routed
   through the validated remap (``yane_index_map.build_remap``). Verified bit-exact
   against the engine's ``e`` command, so it proves the whole feature/numbering/quant
   pipeline reproduces YaneuraOu.

2. ``convert(parsed)`` — de-quantizes + remaps the weights into a binbo BNN3 file so
   binbo's Rust ``nnue-eval`` can run it. NOTE: binbo's net uses plain ReLU (no upper
   clamp, no >>6), so binbo's *float* forward can only track YaneuraOu's *clamped int*
   forward up to a global scale and only where activations don't saturate -> expect
   high correlation, not equality. (Phase-2 implication: a binbo eval meant to run in
   YaneuraOu must be trained quantization-aware / clamped.)
"""
from __future__ import annotations

import argparse
import struct
import sys

import cshogi
import numpy as np

from trainer import kp
from trainer import yane_nnue as yn
from trainer import yane_index_map as ym
from trainer.quant_format import header as quant_header

FV_SCALE = 16
WEIGHT_SCALE_BITS = 6           # the >>6 between affine layers (kWeightScaleBits)
MAGIC_SHARED = b"BNN3"
MAGIC_QUANT = b"BNNQ"           # Task 058: binbo quantization-aware shared HalfKP


def _trunc_div(x: int, d: int) -> int:
    # C++ integer division truncates toward zero (Python // floors).
    q = abs(x) // d
    return q if x >= 0 else -q


def yane_forward_int(parsed: dict, board: cshogi.Board) -> int:
    """Exact YaneuraOu NNUE eval for ``board`` (== engine ``e``), via the remap."""
    ftw = parsed["_ftw2d"]          # int64 [125388, 256], feature-major
    ftb = parsed["_ftb"]            # int64 [256]
    w0, b0 = parsed["_w0"], parsed["_b0"]   # [32,512],[32]
    w1, b1 = parsed["_w1"], parsed["_b1"]   # [32,32],[32]
    w2, b2 = parsed["_w2"], parsed["_b2"]   # [1,32],[1]
    stm, non = ym.yane_active(board)
    acc_s = ftb.copy()
    acc_n = ftb.copy()
    for f in stm:
        acc_s += ftw[f]
    for f in non:
        acc_n += ftw[f]
    x = np.concatenate([np.clip(acc_s, 0, 127), np.clip(acc_n, 0, 127)])
    o0 = b0 + w0 @ x
    x1 = np.clip(o0 >> 6, 0, 127)
    # Task 149: affine files pad H1=8 to 32; padding weights are zero.
    o1 = b1 + w1 @ np.pad(x1, (0, w1.shape[1] - len(x1)))
    x2 = np.clip(o1 >> 6, 0, 127)
    o2 = int((b2 + w2 @ np.pad(x2, (0, w2.shape[1] - len(x2))))[0])
    return _trunc_div(o2, FV_SCALE)


def _attach_int_views(parsed: dict) -> dict:
    parsed["_ftw2d"] = parsed["ft_w"].astype(np.int64).reshape(yn.KIN, len(parsed["ft_bias"]))
    parsed["_ftb"] = parsed["ft_bias"].astype(np.int64)
    L = parsed["layers"]
    parsed["_w0"] = L[0]["weight"].astype(np.int64).reshape(L[0]["out"], L[0]["pin"])
    parsed["_b0"] = L[0]["bias"].astype(np.int64)
    parsed["_w1"] = L[1]["weight"].astype(np.int64).reshape(L[1]["out"], L[1]["pin"])
    parsed["_b1"] = L[1]["bias"].astype(np.int64)
    parsed["_w2"] = L[2]["weight"].astype(np.int64).reshape(L[2]["out"], L[2]["pin"])
    parsed["_b2"] = L[2]["bias"].astype(np.int64)
    return parsed


def convert(parsed: dict, R: np.ndarray, scale_cp: float = 1.0) -> bytes:
    """De-quant + remap a parsed YaneuraOu net into binbo BNN3 bytes (fd=256, hidden=32).

    ft_bias is folded into network.0.bias since binbo's embedding has no FT bias. Plain
    float weights (no >>6 / no clamp) -> binbo's plain-ReLU forward ~ YaneuraOu up to scale.
    """
    if len(parsed["ft_bias"]) != 256 or parsed["layers"][0]["out"] != 32:
        raise ValueError("Task 149: wide nets require --convert-quant (BNQ2)")
    fd = yn.FT_DIM                  # 256
    hidden = parsed["layers"][0]["out"]   # 32
    net_in = 2 * fd                 # 512
    ftw = parsed["ft_w"].astype(np.float32).reshape(yn.KIN, fd)
    ftb = parsed["ft_bias"].astype(np.float32)
    w0 = parsed["layers"][0]["weight"].astype(np.float32).reshape(hidden, net_in)
    b0 = parsed["layers"][0]["bias"].astype(np.float32)
    w1 = parsed["layers"][1]["weight"].astype(np.float32).reshape(hidden, hidden)
    b1 = parsed["layers"][1]["bias"].astype(np.float32)
    w2 = parsed["layers"][2]["weight"].astype(np.float32).reshape(1, hidden)
    b2 = parsed["layers"][2]["bias"].astype(np.float32)

    emb = np.zeros((kp.CANONICAL_SHARED_TOTAL, fd), dtype=np.float32)  # 176744 x 256
    valid = R >= 0
    emb[np.arange(kp.VIEW_DIM)[valid]] = ftw[R[valid]]
    # fold ft_bias (same for both perspectives) into network.0.bias
    b0_folded = b0 + w0 @ np.concatenate([ftb, ftb])

    out = bytearray()
    out += MAGIC_SHARED
    out += struct.pack("<III", kp.CANONICAL_SHARED_TOTAL, fd, hidden)
    out += struct.pack("<f", scale_cp)
    for arr in (emb, w0, b0_folded, w1, b1, w2, b2):
        out += np.ascontiguousarray(arr, dtype="<f4").tobytes()
    return bytes(out)


def _affine_w(layer: dict) -> np.ndarray:
    """[out, in_real] int8 affine weight (drop the SIMD padding past in_real)."""
    return layer["weight"].reshape(layer["out"], layer["pin"])[:, : layer["in_real"]]


def convert_quant(parsed: dict, R: np.ndarray) -> bytes:
    """Remap a parsed YaneuraOu net into binbo BNNQ (Task 058), keeping the integer
    quantization intact so binbo's Mode::Quant forward reproduces ``yane_forward_int``
    (== engine ``e``) bit-exact.

    Unlike ``convert`` (BNN3 float, plain ReLU), this keeps int16 FT / int8 affine and
    a SEPARATE ft_bias (added per perspective BEFORE the ClippedReLU[0,127], so it must
    not be folded into the first affine). Layout matches ``src/eval/nnue.rs::load_quant``:
    magic | total | fd | hidden | fv_scale(i32) | weight_scale_bits(u32) |
    emb(i16) ft_bias(i16) w0(i8) b0(i32) w1(i8) b1(i32) w2(i8) b2(i32).
    """
    fd = len(parsed["ft_bias"])  # Task 149: dimension comes from the parsed header.
    hidden = parsed["layers"][0]["out"]    # 32
    ftw = parsed["ft_w"].reshape(yn.KIN, fd)            # int16 [125388, 256]
    emb = np.zeros((kp.CANONICAL_SHARED_TOTAL, fd), dtype=np.int16)
    valid = R >= 0
    emb[np.arange(kp.VIEW_DIM)[valid]] = ftw[R[valid]]  # side rows (VIEW_DIM, +1) stay 0
    ft_bias = parsed["ft_bias"].astype(np.int16)
    L = parsed["layers"]

    out = bytearray()
    out += quant_header(kp.CANONICAL_SHARED_TOTAL, fd, hidden,
                        L[1]["out"], FV_SCALE, WEIGHT_SCALE_BITS)
    out += np.ascontiguousarray(emb, dtype="<i2").tobytes()
    out += np.ascontiguousarray(ft_bias, dtype="<i2").tobytes()
    for k in range(3):
        out += np.ascontiguousarray(_affine_w(L[k]), dtype="<i1").tobytes()
        out += np.ascontiguousarray(L[k]["bias"], dtype="<i4").tobytes()
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nn_bin")
    ap.add_argument("--convert", metavar="OUT", help="write binbo BNN3 (float) to OUT")
    ap.add_argument("--convert-quant", metavar="OUT",
                    help="write binbo BNNQ (int, quantization-aware; Task 058) to OUT")
    ap.add_argument("--scale-cp", type=float, default=1.0)
    ap.add_argument("--verify-forward", metavar="SFENS",
                    help="file of SFENs; print int forward (== engine e) per line")
    args = ap.parse_args()
    parsed = yn.parse(args.nn_bin)
    if args.convert:
        R = ym.build_remap()
        data = convert(parsed, R, args.scale_cp)
        with open(args.convert, "wb") as f:
            f.write(data)
        print(f"convert: wrote {len(data)} B -> {args.convert} "
              f"(total={kp.CANONICAL_SHARED_TOTAL} fd={yn.FT_DIM} hidden={parsed['layers'][0]['out']})")
    if args.convert_quant:
        R = ym.build_remap()
        data = convert_quant(parsed, R)
        with open(args.convert_quant, "wb") as f:
            f.write(data)
        print(f"convert_quant: wrote {len(data)} B -> {args.convert_quant} ({data[:4].decode()} "
              f"total={kp.CANONICAL_SHARED_TOTAL} fd={len(parsed['ft_bias'])} "
              f"hidden={parsed['layers'][0]['out']}/{parsed['layers'][1]['out']} "
              f"fv_scale={FV_SCALE} weight_scale_bits={WEIGHT_SCALE_BITS})")
    if args.verify_forward:
        _attach_int_views(parsed)
        with open(args.verify_forward) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                print(yane_forward_int(parsed, cshogi.Board(s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
