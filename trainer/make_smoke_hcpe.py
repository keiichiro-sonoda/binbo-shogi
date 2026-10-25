#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cshogi
import numpy as np


EVAL_PATTERN = np.array(
    [-1200, -800, -300, -50, 0, 50, 300, 800, 1200, 450, -650, 150],
    dtype=np.int16,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_hcpe")
    parser.add_argument("output_hcpe")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    source = np.fromfile(args.source_hcpe, dtype=cshogi.HuffmanCodedPosAndEval)
    if len(source) == 0:
        raise ValueError(f"source hcpe is empty: {args.source_hcpe}")

    selected = source[np.arange(args.count) % len(source)].copy()
    evals = EVAL_PATTERN[np.arange(args.count) % len(EVAL_PATTERN)]
    selected["eval"] = evals

    output = Path(args.output_hcpe)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.tofile(output)

    file_size = output.stat().st_size
    expected_size = args.count * cshogi.HuffmanCodedPosAndEval.itemsize
    if file_size != expected_size:
        raise AssertionError(f"file size mismatch: {file_size} != {expected_size}")

    print(f"source={args.source_hcpe}")
    print(f"output={output}")
    print(f"source_entries={len(source)}")
    print(f"written_entries={len(selected)}")
    print(f"hcpe_entry_size={cshogi.HuffmanCodedPosAndEval.itemsize}")
    print(f"file_size={file_size}")
    print(f"eval_min={int(evals.min())}")
    print(f"eval_max={int(evals.max())}")
    print(f"eval_unique={len(set(int(value) for value in evals))}")
    print("smoke_hcpe=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
