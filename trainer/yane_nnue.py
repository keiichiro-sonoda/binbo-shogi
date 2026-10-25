#!/usr/bin/env python3
"""Plain HalfKP nn.bin reader/writer: 256x2-32-32 and 512x2-8-96 (Task 149).

This is the serialization half of the binbo<->YaneuraOu eval bridge. It does NOT
interpret the HalfKP feature numbering (that is the remap in ``yane_index_map.py``);
it only reads/writes the byte layout so that a round-trip is exact and the engine
can load what we write.

The 256 layout below is retained; Task 149 derives dimensions, hashes, structure
and 32-column padding from the supported architectures for template-free export.
Byte layout (all little-endian, verified against the upstream source at
``~/binbo-shogi-data/engines/yaneuraou/source/eval/nnue/`` and confirmed by an exact
byte accounting against the 64,217,066 B tanuki- eval). The spec is
``docs/nnue_yaneura_export.md`` §2-4.

  header:
    u32  version  = 0x7AF32F16
    u32  khash    = overall architecture hash (kHashValue)
    u32  archsize
    byte arch[archsize]                 (178 bytes for this arch)
  FeatureTransformer component:
    u32  ft_hash
    i16  ft_bias[256]
    i16  ft_weight[256 * 125388]        (plain order; semantics fixed by converter)
  Network component (kLayerStacks == 1):
    u32  net_hash
    affine 512->32 : i32 bias[32]  i8 weight[32 * 512]   (plain, out-major, padded)
    affine  32->32 : i32 bias[32]  i8 weight[32 * 32]
    affine  32-> 1 : i32 bias[1]   i8 weight[1  * 32]

The engine stores affine weights internally scrambled (``GetWeightIndex``) on
SSSE3/AVX2, but the FILE is plain order (it descrambles on read); so we read/write
plain order and never scramble.
"""
from __future__ import annotations

import hashlib
import re
import struct
import sys

import numpy as np

VERSION = 0x7AF32F16
FT_DIM = 256                 # kTransformedFeatureDimensions
FE_END = 1548
SQ_NB = 81
KIN = SQ_NB * FE_END         # 125388 = HalfKP kInputDimensions
# (out, in_real) for the three affine layers, input-to-output order.
AFFINE = [(32, 512), (32, 32), (1, 32)]


def _pad(n: int) -> int:
    return ((n + 31) // 32) * 32  # CeilToMultiple(n, kMaxSimdWidth=32)


def halfkp_architecture(fd=256, h1=32, h2=32, sparse=False):
    """Task 149: plain HalfKP layout/hashes from upstream layer definitions.

    SparseInput and dense affine share the hash and file layout. The generated
    512x2_8_96 header uses SparseInput for its first affine.
    """
    if (fd, h1, h2) not in ((256, 32, 32), (512, 8, 96)):
        raise ValueError(f"unsupported HalfKP architecture: {(fd, h1, h2)}")
    mask = 0xffffffff
    ft_hash = 0x5D69D5B8 ^ (2 * fd)
    h = 0xEC42E90D ^ (2 * fd)  # InputSlice, offset=0
    net = f"InputSlice[{2 * fd}(0:{2 * fd})]"
    layers = []
    for i, (out, inp) in enumerate(((h1, 2 * fd), (h2, h1), (1, h2))):
        h = ((0xCC03DAE4 + out) ^ (h >> 1) ^ (h << 31)) & mask
        name = "AffineTransformSparseInput" if i == 0 and sparse else "AffineTransform"
        net = f"{name}[{out}<-{inp}]({net})"
        layers.append(dict(out=out, in_real=inp, pin=_pad(inp)))
        if i < 2:
            h = (h + 0x538D24C7) & mask
            net = f"ClippedReLU[{out}]({net})"
    arch = f"Features=HalfKP(Friend)[{KIN}->{fd}x2],Network={net}".encode("ascii")
    return dict(version=VERSION, khash=ft_hash ^ h, ft_hash=ft_hash,
                net_hash=h, arch=arch, fd=fd, layers=layers)


def architecture_from_string(arch):
    """Task 149: decode only supported plain HalfKP structures, without guessing."""
    match = re.match(rb"Features=HalfKP\(Friend\)\[125388->(\d+)x2\],Network=", arch)
    if match:
        fd = int(match[1])
        for h1, h2 in ((32, 32), (8, 96)):
            for sparse in (False, True):
                if (fd, h1, h2) not in ((256, 32, 32), (512, 8, 96)):
                    continue
                p = halfkp_architecture(fd, h1, h2, sparse)
                if p["arch"] == arch:
                    return p
    raise ValueError(f"unsupported nn.bin architecture: {arch!r}")


def parse(path: str) -> dict:
    with open(path, "rb") as f:
        buf = f.read()
    o = 0
    version, khash, archsize = struct.unpack_from("<III", buf, o)
    o += 12
    arch = buf[o:o + archsize]
    o += archsize
    spec = architecture_from_string(arch)
    fd = spec["fd"]
    (ft_hash,) = struct.unpack_from("<I", buf, o)
    o += 4
    ft_bias = np.frombuffer(buf, "<i2", fd, o).copy()
    o += fd * 2
    ft_w = np.frombuffer(buf, "<i2", fd * KIN, o).copy()  # flat
    o += fd * KIN * 2
    (net_hash,) = struct.unpack_from("<I", buf, o)
    o += 4
    if (version, khash, ft_hash, net_hash) != tuple(spec[k] for k in
                                                  ("version", "khash", "ft_hash", "net_hash")):
        raise ValueError("nn.bin architecture hash/version mismatch")
    layers = []
    for layer in spec["layers"]:
        out, in_real = layer["out"], layer["in_real"]
        pin = _pad(in_real)
        b = np.frombuffer(buf, "<i4", out, o).copy()
        o += out * 4
        w = np.frombuffer(buf, "<i1", out * pin, o).copy()  # flat, out-major padded
        o += out * pin
        layers.append({"bias": b, "weight": w, "out": out, "in_real": in_real, "pin": pin})
    assert o == len(buf), f"byte accounting off: consumed {o} != file {len(buf)}"
    return {
        "version": version, "khash": khash, "arch": arch, "fd": fd,
        "ft_hash": ft_hash, "ft_bias": ft_bias, "ft_w": ft_w,
        "net_hash": net_hash, "layers": layers,
    }


def write(path: str, p: dict) -> int:
    out = bytearray()
    out += struct.pack("<III", p["version"], p["khash"], len(p["arch"]))
    out += bytes(p["arch"])
    out += struct.pack("<I", p["ft_hash"])
    out += p["ft_bias"].astype("<i2").tobytes()
    out += p["ft_w"].astype("<i2").tobytes()
    out += struct.pack("<I", p["net_hash"])
    for L in p["layers"]:
        out += L["bias"].astype("<i4").tobytes()
        out += L["weight"].astype("<i1").tobytes()
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: yane_nnue.py <nn.bin> [out_roundtrip.bin]", file=sys.stderr)
        return 2
    src = sys.argv[1]
    p = parse(src)
    print(f"version=0x{p['version']:08X} khash=0x{p['khash']:08X} archsize={len(p['arch'])}")
    print(f"arch={bytes(p['arch']).decode('ascii', 'replace')}")
    print(f"ft_hash=0x{p['ft_hash']:08X} net_hash=0x{p['net_hash']:08X}")
    print(f"ft_bias{p['ft_bias'].shape} ft_w(flat)={p['ft_w'].size} "
          f"range=[{p['ft_w'].min()},{p['ft_w'].max()}]")
    for i, L in enumerate(p["layers"]):
        print(f"  affine{i} out={L['out']} in={L['in_real']} pin={L['pin']} "
              f"bias_range=[{L['bias'].min()},{L['bias'].max()}] "
              f"w_range=[{L['weight'].min()},{L['weight'].max()}]")
    if len(sys.argv) >= 3:
        dst = sys.argv[2]
        n = write(dst, p)
        same = _sha(src) == _sha(dst)
        print(f"round-trip: wrote {n} B to {dst}  byte-identical={same}")
        if not same:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
