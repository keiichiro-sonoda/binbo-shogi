"""Task 135: lossless shared HalfKP cache, streamed conversion and resident decoding.

Each row has two uint8 king anchors and 77 uint16 codes (156 bytes). Codes
0..2181 are piece offsets, 2182/2183 are side features. Higher codes address
an exact int32 escape stream relative to a per-row offset (only stored when
needed). Column order, duplicates, padding and even unusual int32 values survive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch

try:
    from . import kp
except ImportError:  # run as a top-level module (trainer/ on sys.path)
    import kp

PIECES = kp.FEATURES_PER_KING
ESCAPE = PIECES + 2
FORMAT = "shared-halfkp-compact-v1"


def encode_chunk(original):
    """Task 135: encode every int32 value, including noncanonical rows via escapes."""
    n = len(original)
    kings = np.zeros((n, 2), dtype=np.uint8)
    codes = np.empty((n, 77), dtype=np.uint16)
    for half, (lo, hi) in enumerate(((0, 39), (39, 77))):
        values = original[:, lo:hi]
        body = (values >= 0) & (values < kp.VIEW_DIM)
        first = body.argmax(axis=1)
        king = np.where(body.any(axis=1), values[np.arange(n), first] // PIECES, 0)
        kings[:, half] = king
        normal = body & (values // PIECES == king[:, None])
        side = (values == kp.VIEW_DIM) | (values == kp.VIEW_DIM + 1)
        codes[:, lo:hi] = np.where(normal, values % PIECES,
                                  np.where(side, PIECES + values - kp.VIEW_DIM, 65535))
    escaped = codes == 65535
    counts = escaped.sum(axis=1)
    bases = np.cumsum(counts, dtype=np.int64) - counts
    local = escaped.cumsum(axis=1) - 1
    codes[escaped] = (ESCAPE + local[escaped]).astype(np.uint16)
    return kings, codes, bases, original[escaped].astype(np.int32)


def decode_chunk(kings, codes, bases=None, escapes=None):
    """Independent NumPy decoder used to verify the stored stream (Task 135)."""
    anchors = np.repeat(kings.astype(np.int32), [39, 38], axis=1)
    c = codes.astype(np.int32)
    decoded = np.where(c < PIECES, anchors * PIECES + c, kp.VIEW_DIM + c - PIECES)
    mask = c >= ESCAPE
    if mask.any():
        addresses = bases[:, None] + c - ESCAPE
        decoded[mask] = escapes[addresses[mask]]
    return decoded.astype('<i4', copy=False)


def convert_split(source, output, split, chunk_rows):
    """Task 135: bounded-memory conversion, then verify every persisted column."""
    original = np.load(source / f"{split}_indices.npy", mmap_mode="r")
    if original.dtype != np.int32 or original.ndim != 2 or original.shape[1] != 77:
        raise ValueError(f"expected int32 (n,77), got {original.dtype} {original.shape}")
    n = len(original)
    kings = np.lib.format.open_memmap(output / f"{split}_kings.npy", mode="w+",
                                     dtype=np.uint8, shape=(n, 2))
    codes = np.lib.format.open_memmap(output / f"{split}_codes.npy", mode="w+",
                                     dtype=np.uint16, shape=(n, 77))
    # Offsets/escapes are written incrementally, never held for the full dataset.
    bases = None
    raw_path = output / f"{split}_escapes.raw"
    escape_count = 0
    with raw_path.open("wb") as raw:
        for start in range(0, n, chunk_rows):
            end = min(n, start + chunk_rows)
            k, c, offsets, extra = encode_chunk(original[start:end])
            kings[start:end], codes[start:end] = k, c
            if len(extra) and bases is None:
                bases = np.lib.format.open_memmap(output / f"{split}_bases.npy", mode="w+",
                                                 dtype=np.int64, shape=(n,))
                bases[:start] = 0  # earlier chunks have no escape codes
            if bases is not None:
                bases[start:end] = offsets + escape_count
            extra.tofile(raw)
            escape_count += len(extra)
            if start // chunk_rows % 32 == 0 or end == n:
                print(f"{split}: encoded {end:,}/{n:,} escapes={escape_count:,}", flush=True)
    kings.flush()
    codes.flush()
    if bases is not None:
        bases.flush()
    escapes = None
    if escape_count:
        escapes = np.lib.format.open_memmap(output / f"{split}_escapes.npy", mode="w+",
                                            dtype=np.int32, shape=(escape_count,))
        raw = np.memmap(raw_path, mode="r", dtype=np.int32, shape=(escape_count,))
        for start in range(0, escape_count, chunk_rows * 77):
            escapes[start:start + chunk_rows * 77] = raw[start:start + chunk_rows * 77]
        escapes.flush()
        del raw
    raw_path.unlink()
    # Reopen persisted arrays, and compare to the original, rather than merely
    # verifying the encoder's in-memory outputs.
    kings = np.load(output / f"{split}_kings.npy", mmap_mode="r")
    codes = np.load(output / f"{split}_codes.npy", mmap_mode="r")
    bases = np.load(output / f"{split}_bases.npy", mmap_mode="r") if escape_count else None
    escapes = np.load(output / f"{split}_escapes.npy", mmap_mode="r") if escape_count else None
    source_hash, decoded_hash = hashlib.sha256(), hashlib.sha256()
    mismatches = 0
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        decoded = decode_chunk(kings[start:end], codes[start:end],
                               bases[start:end] if bases is not None else None, escapes)
        expected = original[start:end]
        mismatches += int(np.count_nonzero(decoded != expected))
        source_hash.update(expected.astype('<i4', copy=False).tobytes())
        decoded_hash.update(decoded.tobytes())
        if start // chunk_rows % 32 == 0 or end == n:
            print(f"{split}: verified {end:,}/{n:,} mismatches={mismatches}", flush=True)
    result = dict(rows=n, escapes=escape_count, mismatches=mismatches,
                  source_sha256=source_hash.hexdigest(), decoded_sha256=decoded_hash.hexdigest())
    print(f"{split}: rows={n:,} mismatches={mismatches} "
          f"decoded_sha256={result['decoded_sha256']}", flush=True)
    if mismatches or result['source_sha256'] != result['decoded_sha256']:
        raise RuntimeError(f"round-trip verification failed: {result}")
    return result


STAMP_BYTES = 64 << 20


def source_stamp(source, split):
    """Task 135 (review): cheap identity of the source indices file -- size, mtime and sha256 of the
    first/last 64 MiB -- checked at load so a source cache regenerated in place (same row count)
    cannot silently pair stale compact rows with new labels. The full-stream sha256 is in the manifest."""
    path = (Path(source) / f"{split}_indices.npy").resolve()
    st = path.stat()
    with open(path, "rb") as f:
        head = hashlib.sha256(f.read(STAMP_BYTES)).hexdigest()
        f.seek(max(0, st.st_size - STAMP_BYTES))
        tail = hashlib.sha256(f.read(STAMP_BYTES)).hexdigest()
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, head_sha256=head, tail_sha256=tail)


def stamp(output):
    """Task 135 (review): add source stamps to a manifest written before stamps existed, after
    re-hashing the whole source stream and checking it against the recorded source_sha256."""
    manifest = Path(output) / "compact.json"
    meta = json.loads(manifest.read_text())
    for split, info in meta["splits"].items():
        original = np.load(Path(meta["source"]) / f"{split}_indices.npy", mmap_mode="r")
        h = hashlib.sha256()
        for start in range(0, len(original), 65536):
            h.update(np.asarray(original[start:start + 65536]).astype("<i4", copy=False).tobytes())
        if h.hexdigest() != info["source_sha256"]:
            raise RuntimeError(f"{split}: source changed since conversion; convert again")
        info["source_stamp"] = source_stamp(meta["source"], split)
        print(f"{split}: source sha256 matches, stamped", flush=True)
    manifest.write_text(json.dumps(meta, indent=2) + "\n")


def convert(source, output, chunk_rows=65536):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("source and compact output must differ")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "compact.json"
    if manifest.exists():
        raise FileExistsError(f"{manifest} already exists; use a new directory")
    started = time.monotonic()
    splits = {s: convert_split(source, output, s, chunk_rows) for s in ("train", "val")}
    for s in splits:
        splits[s]["source_stamp"] = source_stamp(source, s)
    result = dict(format=FORMAT, source=str(source), splits=splits,
                  seconds=time.monotonic() - started)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"compact_cache=ok seconds={result['seconds']:.1f}", flush=True)
    return result


def _resident(path, device, limit=None, chunk_rows=65536, signed=False):
    array = np.load(path, mmap_mode="r")
    if limit is not None:
        array = array[:limit]
    if signed:
        array = array.view(np.int16)  # uint16 gather is not supported by all torch backends
    dtype = torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype
    result = torch.empty(array.shape, dtype=dtype, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for start in range(0, len(array), chunk_rows):
            result[start:start + chunk_rows].copy_(torch.from_numpy(array[start:start + chunk_rows]))
    return result


class CompactCache:
    """Task 135: resident encoded arrays; batch decoding has no data-dependent shape."""

    resident = True

    def __init__(self, directory, split, device="cuda", limit=None, source=None):
        directory = Path(directory)
        meta = json.loads((directory / "compact.json").read_text())
        if meta["format"] != FORMAT:
            raise ValueError("unsupported compact format")
        if source is not None and Path(source).resolve() != Path(meta["source"]):
            raise ValueError("compact cache belongs to a different source cache")
        info = meta["splits"][split]
        if info["mismatches"] or info["source_sha256"] != info["decoded_sha256"]:
            raise ValueError("compact cache was not verified")
        if "source_stamp" not in info:
            raise ValueError(f"{directory}: no source stamp; run python -m trainer.compact_cache --stamp {directory}")
        if source_stamp(meta["source"], split) != info["source_stamp"]:
            raise ValueError(f"{split}: the source cache changed after conversion; convert again")
        self.device = torch.device(device)
        self.kings = _resident(directory / f"{split}_kings.npy", device, limit)
        self.codes = _resident(directory / f"{split}_codes.npy", device, limit, signed=True)
        self.bases = self.escapes = None
        if info["escapes"]:
            self.bases = _resident(directory / f"{split}_bases.npy", device, limit)
            self.escapes = _resident(directory / f"{split}_escapes.npy", device)
        self.shape = (len(self.kings), 77)
        self.half = torch.tensor([0] * 39 + [1] * 38, dtype=torch.long, device=device)

    def __getitem__(self, rows):
        rows = rows.to(self.device, non_blocking=True)
        c = self.codes.index_select(0, rows).long().bitwise_and(65535)
        kings = self.kings.index_select(0, rows).long()[:, self.half]
        decoded = torch.where(c < PIECES, kings * PIECES + c, kp.VIEW_DIM + c - PIECES)
        if self.escapes is not None:
            addresses = self.bases.index_select(0, rows)[:, None] + c - ESCAPE
            extra = self.escapes[addresses.clamp(0, len(self.escapes) - 1)].long()
            decoded = torch.where(c >= ESCAPE, extra, decoded)
        return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--stamp", metavar="COMPACT_DIR", help="add source stamps to an existing manifest")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    if args.stamp:
        return stamp(args.stamp)
    if not (args.source and args.output):
        parser.error("source and output are required")
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    convert(args.source, args.output, args.chunk_rows)


if __name__ == "__main__":
    main()
