#!/usr/bin/env python3
"""Task 188 data: the first 2 x 63,950,000 records of the Suisho5 public training data as two training caches.

Chunk A = records [0, N) (the Task 145 student's data, `public/.../train_first64m.bin`, rebuilt into
caches/public_suisho5_train64m after the 10/23 cleanup); chunk B = records [N, 2N) (new). The archive is streamed
with the Task 145 trick (real .001 + placeholder .002-.008 in `vol/`); its first N records must equal chunk A's file
byte for byte. Each chunk becomes a cache (features via Task 145's converter, evals, wdl, the shared val as links),
a compact cache, and a one-part manifest (work/task188/chunk_{a,b}.json) for the chunked trainer (Task 188a).

    python t188_student2x_data.py extract|cache-a|cache-b|manifests
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import cshogi  # noqa: E402
import std_frame  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from t142_public_suisho5 import _rows  # noqa: E402
from t172_build import dump, part_record  # noqa: E402
from trainer import kp  # noqa: E402
from trainer.multi_cache import FORMAT_MULTI, Manifest  # noqa: E402

N = 63_950_000
REC = 40
DATA = std_frame.paths()[1]
PUB = DATA / "public/nodchip_suisho5_depth9"
RAW = {"a": PUB / "train_first64m.bin", "b": PUB / "train_second64m.bin"}
CACHE = {"a": DATA / "caches/public_suisho5_train64m", "b": DATA / "caches/public_suisho5_train64m_b"}
WORK = DATA / "work/task188"


def sha_head(path, n):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = n
        while left:
            block = f.read(min(left, 64 << 20))
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def extract():
    """Stream records [0, 2N): check [0, N) against chunk A's file, write [N, 2N) as chunk B."""
    if RAW["b"].exists() and RAW["b"].stat().st_size == N * REC:
        print("T188_EXTRACT reuse", flush=True)
        return
    vol = PUB / "vol"
    if not (vol / "shuffled.7z.001").exists():
        (vol / "shuffled.7z.001").symlink_to(PUB / "shuffled.7z.001")
    want_a = sha_head(RAW["a"], N * REC)
    p = subprocess.Popen(["7z", "x", "-so", str(vol / "shuffled.7z.001")], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    h, left = hashlib.sha256(), N * REC
    while left:
        block = p.stdout.read(min(left, 64 << 20))
        if not block:
            raise RuntimeError("archive ended inside chunk A")
        h.update(block)
        left -= len(block)
    if h.hexdigest() != want_a:
        p.kill()
        raise RuntimeError("the archive's first N records differ from train_first64m.bin")
    tmp = RAW["b"].with_suffix(".tmp")
    with open(tmp, "wb") as f:
        left = N * REC
        while left:
            block = p.stdout.read(min(left, 64 << 20))
            if not block:
                break
            f.write(block)
            left -= len(block)
    p.kill()
    p.wait()
    if tmp.stat().st_size != N * REC:
        raise RuntimeError(f"chunk B has {tmp.stat().st_size} bytes < {N * REC}")
    os.replace(tmp, RAW["b"])
    print(f"T188_EXTRACT chunk_a_sha={want_a[:16]} (matches) chunk_b={RAW['b']}", flush=True)


def build_cache(key):
    psv, cache = RAW[key], CACHE[key]
    if (cache / "meta.json").exists():
        print(f"T188_CACHE {key} reuse", flush=True)
        return
    src = DATA / t129.QS64   # the shared val and the meta layout, as Task 145
    cache.mkdir(parents=True, exist_ok=True)
    idx = np.lib.format.open_memmap(cache / "train_indices.npy", mode="w+", dtype=np.int32, shape=(N, 77))
    step = 500_000
    with Pool(16) as pool:
        for lo, out in pool.imap_unordered(_rows, [(str(psv), lo, min(N, lo + step)) for lo in range(0, N, step)]):
            idx[lo:lo + len(out)] = out
    idx.flush()
    a = np.memmap(psv, dtype=cshogi.PackedSfenValue, mode="r")
    rng, b = np.random.default_rng(20271880 + ord(key)), cshogi.Board()
    for i in rng.choice(N, 2000, replace=False):
        b.set_psfen(a[i]["sfen"])
        if kp.canonical_shared_row_to_sfen(np.asarray(idx[i])).split()[:3] != b.sfen().split()[:3]:
            raise AssertionError(f"chunk {key} row {i}: features do not decode to the psv board")
    np.save(cache / "train_evals.npy", np.asarray(a["score"]).astype(np.int16))
    np.save(cache / "train_wdl.npy", np.asarray(a["game_result"]).astype(np.int8))
    for f in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
        (cache / f).symlink_to((src / f).resolve())
    meta = dict(json.loads((src / "meta.json").read_text(encoding="utf-8")), n_train=N,
                built_by=f"t188 chunk {key} (nodchip/shogi_suisho5_depth9 shuffled.7z.001, records "
                         f"{N if key == 'b' else 0}..{2 * N if key == 'b' else N})")
    meta.pop("qsearch_replace", None)
    (cache / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    subprocess.run([sys.executable, "-B", "-m", "trainer.compact_cache", str(cache), str(cache) + "_compact"], check=True)
    print(f"T188_CACHE {key} rows={N} gate=2000/2000", flush=True)


def manifests():
    WORK.mkdir(parents=True, exist_ok=True)
    val = json.loads((DATA / "work/task181/M.json").read_text(encoding="utf-8"))["val"]
    for key in ("a", "b"):
        cache = CACHE[key]
        path = WORK / f"chunk_{key}.json"
        rec = part_record(cache, Path(str(cache) + "_compact"), cache / "train_evals.npy", N, "train", None, 0)
        dump(path, dict(format=FORMAT_MULTI, rows=N, parts=[rec], val=val, arm=f"chunk_{key}",
                        provenance=f"Suisho5 public training data, records chunk {key} (Task 188)",
                        result_lambda_cli=1, result_mix=0))
        Manifest(path)
        print(f"T188_MANIFEST {path}", flush=True)


if __name__ == "__main__":
    {"extract": extract, "cache-a": lambda: build_cache("a"), "cache-b": lambda: build_cache("b"),
     "manifests": manifests}[sys.argv[1]]()
