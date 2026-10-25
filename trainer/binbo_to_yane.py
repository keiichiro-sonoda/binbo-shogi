#!/usr/bin/env python3
"""Writer: binbo BNNQ -> YaneuraOu nn.bin (Task 059, the inverse of Task 057's reader).

Task 057 read a YaneuraOu nn.bin into binbo; Task 058 made binbo quantization-aware
(BNNQ, ``src/eval/nnue.rs`` ``Mode::Quant``). This module goes the other way: it
serialises a binbo BNNQ into YaneuraOu's plain ``nn.bin`` so the self-made eval can be
loaded by the real engine. It reuses the validated halves of the bridge:

  - ``yane_index_map.build_remap`` (R: binbo shared idx -> YaneuraOu HalfKP idx),
  - ``yane_nnue.parse`` / ``yane_nnue.write`` (byte-exact layout, Task 057),

and only adds the *inverse* scatter + assembly. No forward/remap logic is re-derived.

Lossiness (measured, Task 059): R is many-to-one (YaneuraOu fe_end=1548 folds
gold/+P/+L/+N/+S into one slot; 13122 gold slots <- 5 binbo features each) and binbo
has 2 side-to-move features YaneuraOu lacks. So a binbo eval is bit-exact writable only
if those collapsed embeddings are EQUAL and the side rows are 0 -- which converted
tanuki satisfies by construction, and a binbo-*trained* eval satisfies when trained with
``--qat-tie`` (``build_collapse_map`` ties them). The writer ASSERTS the equality so a
non-tied eval fails loudly rather than silently writing a corrupted slot.
"""
from __future__ import annotations

import argparse

import numpy as np

from trainer import kp
from trainer import yane_nnue as yn
from trainer import yane_index_map as ym
from trainer.quant_format import read_header


def parse_bnnq(path: str) -> dict:
    """Task 149: parse legacy BNNQ or BNQ2 (Rust supports the legacy format)."""
    with open(path, "rb") as f:
        buf = f.read()
    meta, o = read_header(buf)  # Task 149: legacy BNNQ or explicit BNQ2.
    total, fd, hidden, hidden2 = (meta[k] for k in ("total", "fd", "hidden", "hidden2"))

    def take(count, dt, size):
        nonlocal o
        a = np.frombuffer(buf, dt, count, o).copy()
        o += count * size
        return a

    emb = take(total * fd, "<i2", 2)
    ft_bias = take(fd, "<i2", 2)
    w0 = take(hidden * 2 * fd, "<i1", 1)
    b0 = take(hidden, "<i4", 4)
    w1 = take(hidden2 * hidden, "<i1", 1)
    b1 = take(hidden2, "<i4", 4)
    w2 = take(hidden2, "<i1", 1)
    b2 = take(1, "<i4", 4)
    assert o == len(buf), f"BNNQ byte accounting off: {o} != {len(buf)}"
    return {**meta, "emb": emb, "ft_bias": ft_bias,
            "w0": w0, "b0": b0, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def build_collapse_map(vm: bool = False) -> np.ndarray:
    """``collapse_map[binbo_idx] -> canonical binbo row`` for tying (Task 059).

    Binbo features that share a YaneuraOu HalfKP slot (the 5-way gold collapse) are sent
    to one representative (the smallest binbo idx of the R-group), so their embeddings can
    be tied. The 2 side-to-move rows (no YaneuraOu slot) are sent to a PAD row at index
    ``CANONICAL_SHARED_TOTAL`` (the model freezes it to 0 via EmbeddingBag padding_idx).
    Returned length is ``CANONICAL_SHARED_TOTAL``; values are in ``[0, CANONICAL_SHARED_TOTAL]``."""
    R = ym.build_remap()                       # [VIEW_DIM]
    total = kp.CANONICAL_SHARED_TOTAL
    pad = total
    cmap = np.arange(total, dtype=np.int64)
    rep: dict[int, int] = {}
    for b in range(kp.VIEW_DIM):
        y = int(R[b])
        if y < 0:
            continue
        rep.setdefault(y, b)
        cmap[b] = rep[y]
    cmap[kp.VIEW_DIM:total] = pad               # side rows -> pad (frozen 0)
    if vm:
        # HalfKP_vm tie (Task 132): a right-side king row first moves to its mirror row,
        # then to that row's gold-collapse representative. The deployed table is still
        # plain HalfKP (mirrored rows hold copies), so the YaneuraOu build is unchanged.
        cmap[:kp.VIEW_DIM] = cmap[kp.vm_mirror_map()]
    return cmap


def bnnq_to_yane(bnnq: dict, template: dict) -> dict:
    """Assemble a YaneuraOu parsed-dict from BNNQ weights + a template's hashes/arch.

    Task 149: metadata can come from a same-architecture nn.bin or the layer definitions.
    Hashes/structure depend on architecture, not weight values."""
    fd, hidden, hidden2 = bnnq["fd"], bnnq["hidden"], bnnq["hidden2"]
    spec = yn.architecture_from_string(template["arch"])
    if (fd, hidden, hidden2) != (spec["fd"], spec["layers"][0]["out"], spec["layers"][1]["out"]):
        raise ValueError("Task 149: BNNQ dimensions differ from nn.bin architecture")
    if bnnq["total"] != kp.CANONICAL_SHARED_TOTAL:
        raise ValueError("unsupported BNNQ HalfKP feature count")

    # FeatureTransformer: inverse scatter binbo emb -> YaneuraOu ft_w via R.
    R = ym.build_remap()
    emb2d = bnnq["emb"].reshape(bnnq["total"], fd)
    if np.any(emb2d[kp.VIEW_DIM:]):
        raise ValueError("side features must be zero (train with --qat-tie)")
    yane_ftw2d = np.zeros((yn.KIN, fd), dtype=np.int16)
    valid = R >= 0
    bvals = np.arange(kp.VIEW_DIM)[valid]
    yvals = R[valid]
    yane_ftw2d[yvals] = emb2d[bvals]
    # bit-exact guard: every binbo row sharing a YaneuraOu slot must be identical
    # (tied/tanuki pass; an un-tied trained eval fails loudly instead of corrupting a slot).
    if not np.array_equal(yane_ftw2d[yvals], emb2d[bvals]):
        bad = int((yane_ftw2d[yvals] != emb2d[bvals]).any(axis=1).sum())
        raise AssertionError(
            f"{bad} binbo features collapse to a YaneuraOu slot with DIFFERENT embeddings "
            f"(eval not tied; train with --qat-tie or convert from tanuki)")

    p = {
        "version": template["version"], "khash": template["khash"], "arch": template["arch"], "fd": fd,
        "ft_hash": template["ft_hash"], "net_hash": template["net_hash"],
        "ft_bias": bnnq["ft_bias"].astype(np.int16),
        "ft_w": yane_ftw2d.reshape(-1).astype(np.int16),
        "layers": [],
    }

    # Affine layers: pad binbo [out, in_real] to YaneuraOu [out, pin] (zero columns past
    # in_real) and flatten; biases copy directly.
    bnnq_w = [bnnq["w0"].reshape(hidden, 2 * fd), bnnq["w1"].reshape(hidden2, hidden),
              bnnq["w2"].reshape(1, hidden2)]
    bnnq_b = [bnnq["b0"], bnnq["b1"], bnnq["b2"]]
    for k, tL in enumerate(template["layers"]):
        out, in_real, pin = tL["out"], tL["in_real"], tL["pin"]
        w = bnnq_w[k]
        if w.shape != (out, in_real):
            raise ValueError(f"affine{k}: BNNQ {w.shape} != YaneuraOu (out,in_real)=({out},{in_real})")
        wp = np.zeros((out, pin), dtype=np.int8)
        wp[:, :in_real] = w.astype(np.int8)
        p["layers"].append({
            "out": out, "in_real": in_real, "pin": pin,
            "bias": bnnq_b[k].astype(np.int32),
            "weight": wp.reshape(-1).astype(np.int8),
        })
    return p


def write_nn_bin(from_bnnq: str, template_nn: str | None, out_path: str) -> dict:
    bnnq = parse_bnnq(from_bnnq)
    # Task 149: no learned template is needed; layer headers determine the layout.
    template = (yn.parse(template_nn) if template_nn else yn.halfkp_architecture(
        bnnq["fd"], bnnq["hidden"], bnnq["hidden2"], sparse=True))
    p = bnnq_to_yane(bnnq, template)
    n = yn.write(out_path, p)
    return {"bytes": n, "fd": bnnq["fd"], "hidden": bnnq["hidden"], "hidden2": bnnq["hidden2"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-bnnq", required=True, help="binbo BNNQ file")
    ap.add_argument("--template", help="same-arch nn.bin; omit to generate Task 149 plain HalfKP header")
    ap.add_argument("--out", required=True, help="output nn.bin")
    args = ap.parse_args()
    info = write_nn_bin(args.from_bnnq, args.template, args.out)
    print(f"binbo_to_yane: wrote {info['bytes']} B -> {args.out} "
          f"(fd={info['fd']} hidden={info['hidden']}/{info['hidden2']}, template={args.template})")
    print("binbo_to_yane=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
