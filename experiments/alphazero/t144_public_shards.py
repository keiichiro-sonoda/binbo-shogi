#!/usr/bin/env python3
"""Task 144: stream public PSV shards through qsearch, exact dedup and fine-tune caches.

Source labels are ORIGINAL side-to-move cp/results. Default --changed drop keeps
only identical positions; --changed inherit retains the original label, negated
on odd PV parity (Task 119), WITHOUT claiming it is a relabelled leaf (Task 128).
Hash = BLAKE2b-128 of reversible canonical-shared int32 LE features, i.e. board,
hands and turn, excluding ply/labels. Hash hits always require full row equality.
Source PSV files are never deleted: delete each download only after process
reports source_complete. No downloads, training or GPU use occur here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import sys
import time
from urllib.parse import quote

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cshogi  # noqa: E402
import numpy as np  # noqa: E402
import qs_replace as qs  # noqa: E402
from trainer import kp, mirror  # noqa: E402

FORMAT = "task144-public-shards-v1"
ARRAYS = {"features": ("<i4", (77,)), "scores": ("<i2", ()),
          "results": ("i1", ()), "hashes": ("u1", (16,)),
          "source_rows": ("<i8", ()), "changed": ("?", ()), "flip": ("?", ())}
DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data")).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as inp:
        while block := inp.read(4 << 20):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_files(directory):
    for p in Path(directory).iterdir():
        if p.is_file() and not p.is_symlink():
            with p.open("rb") as inp:
                os.fsync(inp.fileno())
    sync_dir(directory)


def staging(path, owner):
    """Task 144: only discard staging directories carrying our matching receipt."""
    receipt = path / ".task144-stage.json"
    if path.exists():
        require(receipt.is_file() and read_json(receipt) == owner, f"unowned staging directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    atomic_json(receipt, owner)


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def fingerprint(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))


def row_hash(row):
    return hashlib.blake2b(np.asarray(row, dtype="<i4").tobytes(), digest_size=16,
                           person=b"task144-board-v1").digest()


def mirror_rows(rows, perm):
    out = perm[rows].astype("<i4")
    out[:, :39].sort(axis=1)
    out[:, 39:].sort(axis=1)
    return out


def load_part(path, verify=True):
    path = Path(path).resolve()
    meta = read_json(path / "part.json")
    require(meta["format"] == FORMAT, f"unknown part format: {path}")
    for name, (dtype, shape) in ARRAYS.items():
        p = path / (name + ".npy")
        a = np.load(p, mmap_mode="r")
        require(a.dtype == np.dtype(dtype) and a.shape == (meta["rows"], *shape), f"bad array: {p}")
        if verify:
            require(sha256(p) == meta["files"][p.name], f"part checksum mismatch: {p}")
    return meta


def process(args):
    """Task 144: publish independently atomic bounded parts, with resume receipts."""
    out = args.out.resolve()
    yane, ev = args.yane.resolve(), args.eval_dir.resolve()
    require(not any(c.isspace() for c in str(out)), "qsearch output path cannot contain whitespace")
    require(not any(c.isspace() for c in str(ev)), "EvalDir cannot contain whitespace")
    recipe = dict(format=FORMAT, changed=args.changed, threads=args.threads, part_rows=args.part_rows,
                  keep_psv=args.keep_psv, verify_rows=args.verify_rows,
                  yane=fingerprint(yane), eval=fingerprint(ev / "nn.bin"),
                  tool_sha256=sha256(__file__), kp_sha256=sha256(kp.__file__),
                  qs_sha256=sha256(qs.__file__))
    with locked(out):
        manifest = out / "manifest.json"
        state = read_json(manifest) if manifest.exists() else dict(recipe=recipe, sources=[], parts=[])
        require(state["recipe"] == recipe, "process recipe changed; use a new --out directory")
        parts_dir = out / "parts"
        parts_dir.mkdir(exist_ok=True)
        atomic_json(manifest, state)
        engine = None
        try:
            for source_path in args.psv:
                source_path = source_path.resolve()
                source = fingerprint(source_path)
                require(source["bytes"] > 0 and source["bytes"] % qs.PSV.itemsize == 0,
                        f"empty/truncated PSV: {source_path}")
                source["rows"] = source["bytes"] // qs.PSV.itemsize
                old = next((s for s in state["sources"] if s["sha256"] == source["sha256"]), None)
                if old is None:
                    state["sources"].append(source)
                elif old.get("complete"):
                    for lo in range(0, source["rows"], args.part_rows):
                        hi = min(source["rows"], lo + args.part_rows)
                        info = load_part(parts_dir / f"{source['sha256']}_{lo:012d}_{hi:012d}")
                        require(info["recipe"] == recipe, "incompatible completed part")
                    print(f"source_complete reused={source_path} rows={old['rows']}", flush=True)
                    continue
                else:
                    source = old
                atomic_json(manifest, state)
                raw = np.memmap(source_path, dtype=qs.PSV, mode="r")
                for lo in range(0, len(raw), args.part_rows):
                    hi = min(len(raw), lo + args.part_rows)
                    name = f"{source['sha256']}_{lo:012d}_{hi:012d}"
                    part, tmp = parts_dir / name, parts_dir / (name + ".partial")
                    if part.exists():
                        info = load_part(part)
                        require(info["recipe"] == recipe and info["source"]["sha256"] == source["sha256"]
                                and (info["lo"], info["hi"]) == (lo, hi), "incompatible resume part")
                    else:
                        staging(tmp, dict(task=144, operation="process", target=str(part)))
                        start = time.monotonic()
                        marker = np.array(raw[lo:hi])
                        marker["score"], marker["game_result"] = 1, 0
                        inp, leaf = tmp / "qsearch_input.psv", tmp / "qsearch_leaf.psv"
                        marker.tofile(inp)
                        if engine is None:
                            engine = qs.QSearchEngine(yane, ev, threads=args.threads, hash_mb=16)
                        t = time.monotonic()
                        stats = engine.run(inp, leaf, timeout=args.timeout)
                        qs_seconds = time.monotonic() - t
                        require(stats["records"] == hi - lo and stats["decode_errors"] == 0
                                and stats["illegal_pv"] == 0, f"qsearch failed: {stats}")
                        require(leaf.stat().st_size == (hi - lo) * qs.PSV.itemsize, "qsearch lost records")
                        leaves = np.fromfile(leaf, dtype=qs.PSV)
                        changed, flip = qs.leaf_rows(marker, leaves)
                        t = time.monotonic()
                        # Raw bounded streams avoid allocating a full shard or a second feature matrix.
                        streams = {key: (tmp / (key + ".raw")).open("wb") for key in ARRAYS}
                        board = cshogi.Board()
                        kept = in_check = checked = 0
                        sample = set(np.random.default_rng(144).choice(hi - lo,
                                     min(args.verify_rows, hi - lo), replace=False).tolist())
                        try:
                            for i, entry in enumerate(leaves):
                                board.set_psfen(entry["sfen"])
                                if board.is_check():
                                    in_check += 1
                                    continue
                                if args.changed == "drop" and changed[i]:
                                    continue
                                features = np.asarray(kp.canonical_shared_row(board), dtype="<i4")
                                require(features.shape == (77,), f"row {lo+i}: expected 77 features")
                                if i in sample:
                                    require(kp.canonical_shared_row_to_sfen(features).split()[:3]
                                            == board.sfen().split()[:3], f"row {lo+i}: feature round trip")
                                    checked += 1
                                sign = -1 if flip[i] else 1
                                score = int(raw[lo+i]["score"]) * sign
                                result = int(raw[lo+i]["game_result"]) * sign
                                require(-32768 <= score <= 32767, "sign-adjusted cp outside int16 (no wrap allowed)")
                                require(result in (-1, 0, 1), f"invalid signed PSV game_result: {result}")
                                values = dict(features=features, scores=score, results=result,
                                              hashes=np.frombuffer(row_hash(features), dtype="u1"),
                                              source_rows=lo+i, changed=changed[i], flip=flip[i])
                                for key, value in values.items():
                                    streams[key].write(np.asarray(value, dtype=ARRAYS[key][0]).tobytes())
                                kept += 1
                        finally:
                            for stream in streams.values():
                                stream.close()
                        feature_seconds = time.monotonic() - t
                        files = {}
                        for key, (dtype, shape) in ARRAYS.items():
                            target = tmp / (key + ".npy")
                            a = np.lib.format.open_memmap(target, mode="w+", dtype=dtype, shape=(kept, *shape))
                            if kept:
                                src = np.memmap(tmp / (key + ".raw"), mode="r", dtype=dtype, shape=a.shape)
                                a[:] = src
                                del src
                            a.flush()
                            del a
                            (tmp / (key + ".raw")).unlink()
                            files[target.name] = sha256(target)
                        if not args.keep_psv:
                            inp.unlink()
                            leaf.unlink()
                        info = dict(format=FORMAT, recipe=recipe, source=source.copy(), lo=lo, hi=hi,
                                    rows=kept, changed=int(changed.sum()), flipped=int(flip.sum()),
                                    kept_changed=int(changed[np.load(tmp / "source_rows.npy") - lo].sum()),
                                    dropped_changed=int(changed.sum()) if args.changed == "drop" else 0,
                                    in_check=in_check, errors=0, qsearch=stats, roundtrip_checked=checked,
                                    qsearch_seconds=qs_seconds, feature_seconds=feature_seconds,
                                    seconds=time.monotonic() - start, files=files)
                        atomic_json(tmp / "part.json", info)
                        sync_files(tmp)
                        os.replace(tmp, part)
                        sync_dir(parts_dir)
                    receipt = dict(path=str(part), source_sha256=source["sha256"], lo=lo, hi=hi,
                                   rows=info["rows"], changed=info["changed"], flipped=info["flipped"],
                                   in_check=info["in_check"], errors=info["errors"],
                                   qsearch=info["qsearch"], qsearch_seconds=info["qsearch_seconds"],
                                   feature_seconds=info["feature_seconds"], seconds=info["seconds"])
                    if not any(p["path"] == str(part) for p in state["parts"]):
                        state["parts"].append(receipt)
                    atomic_json(manifest, state)
                    print("part " + json.dumps(receipt), flush=True)
                source["complete"] = True
                atomic_json(manifest, state)
                print(f"source_complete file={source_path} sha256={source['sha256']} rows={source['rows']}", flush=True)
        finally:
            if engine is not None:
                engine.close()


def expand_parts(paths):
    parts = []
    for path in paths:  # explicit argument order, then source SHA / offset order
        p = path.resolve()
        if (p / "part.json").exists():
            parts.append(p)
        else:
            m = read_json(p / "manifest.json")
            require(all(s.get("complete") for s in m["sources"]), f"incomplete process source: {p}")
            parts.extend(Path(x["path"]) for x in sorted(m["parts"], key=lambda x: (x["source_sha256"], x["lo"])))
    require(parts, "no processed parts")
    return parts


class RowStore:
    """Task 144: bounded mmap access to full rows for collision confirmation."""

    def __init__(self, sources, perm):
        self.sources, self.perm, self.arrays = sources, perm, {}

    def chunk(self, sid, lo, hi):
        source = self.sources[sid]
        if sid not in self.arrays:
            # Cap open mappings; huge jobs may have thousands of small parts.
            if len(self.arrays) >= 16:
                self.arrays.pop(next(iter(self.arrays)))
            self.arrays[sid] = np.load(source["file"]["path"], mmap_mode="r")
        a = self.arrays[sid][lo:hi]
        return mirror_rows(a, self.perm) if source.get("mirror") else a


def dedup(args):
    """Task 144: disk index, references first, stable first surviving row wins."""
    parts = expand_parts(args.parts)
    sources = []
    for kind, path, mirrored in (("existing", args.existing / "train_indices.npy", False),
                                  ("val", args.val_cache / "val_indices.npy", False),
                                  ("val_mirror", args.val_cache / "val_indices.npy", True)):
        sources.append(dict(kind=kind, file=fingerprint(path), mirror=mirrored))
    part_meta = [load_part(p) for p in parts]
    for p, meta in zip(parts, part_meta):
        sources.append(dict(kind="part", file=fingerprint(p / "features.npy"), part=str(p),
                            part_json_sha256=sha256(p / "part.json"), rows=meta["rows"]))
    config = dict(format=FORMAT, sources=sources, order="arguments, then (source sha256, record offset)",
                  hash="blake2b-128 canonical shared <i4, person=task144-board-v1",
                  mirror_sha256=sha256(mirror.__file__), tool_sha256=sha256(__file__))
    out = args.out.resolve()
    with locked(out):
        cp = out / "config.json"
        if cp.exists():
            require(read_json(cp) == config, "dedup inputs changed; use a new --out directory")
        else:
            require(not (out / "manifest.json").exists(), "dedup output contains another manifest")
            atomic_json(cp, config)
        perm = mirror.build_mirror_perm_shared()
        store = RowStore(sources, perm)
        db = sqlite3.connect(out / "index.sqlite3")
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA threads=1")
        db.execute("CREATE TABLE IF NOT EXISTS positions (h BLOB, sid INTEGER, r INTEGER, "
                   "PRIMARY KEY(h,sid,r)) WITHOUT ROWID")
        db.execute("CREATE TABLE IF NOT EXISTS progress (k TEXT PRIMARY KEY, result TEXT)")
        db.commit()
        started = time.monotonic()
        try:
            for sid in range(3):
                store.chunk(sid, 0, 0)
                a = store.arrays[sid]
                require(a.dtype == np.int32 and a.ndim == 2 and a.shape[1] == 77, "bad reference features")
                n = len(a)
                key = f"ref{sid}"
                record = db.execute("SELECT result FROM progress WHERE k=?", (key,)).fetchone()
                lo = json.loads(record[0])["end"] if record else 0
                while lo < n:
                    hi = min(n, lo + args.chunk_rows)
                    rows = store.chunk(sid, lo, hi)
                    with db:
                        db.executemany("INSERT OR IGNORE INTO positions VALUES (?,?,?)",
                                       ((row_hash(row), sid, lo+i) for i, row in enumerate(rows)))
                        db.execute("INSERT OR REPLACE INTO progress VALUES (?,?)", (key, json.dumps(dict(end=hi))))
                    lo = hi
                    print(f"reference={sources[sid]['kind']} rows={hi}/{n}", flush=True)
            receipts = []
            for k, (part, meta) in enumerate(zip(parts, part_meta)):
                key = f"part{k}"
                previous = db.execute("SELECT result FROM progress WHERE k=?", (key,)).fetchone()
                if previous:
                    receipt = json.loads(previous[0])
                    require(sha256(out / receipt["mask"]) == receipt["mask_sha256"], "dedup resume mask damaged")
                    receipts.append(receipt)
                    continue
                n, sid = meta["rows"], k + 3
                mask_path = out / f"keep_{k:06d}.npy"
                temp_mask = out / f"keep_{k:06d}.tmp.npy"
                keep = np.lib.format.open_memmap(temp_mask, mode="w+", dtype=bool, shape=(n,))
                counts = dict(input=n, kept=0, existing=0, val=0, val_mirror=0, parts=0,
                              hash_collisions=0, full_compares=0)
                hashes = np.load(part / "hashes.npy", mmap_mode="r")
                try:
                    db.execute("BEGIN")
                    for lo in range(0, n, args.chunk_rows):
                        rows = store.chunk(sid, lo, min(n, lo + args.chunk_rows))
                        for i, row in enumerate(rows, lo):
                            h = row_hash(row)
                            require(h == hashes[i].tobytes(), f"bad stored hash: {part} row {i}")
                            match = None
                            for ref_sid, ref_row in db.execute("SELECT sid,r FROM positions WHERE h=? ORDER BY sid,r", (h,)):
                                counts["full_compares"] += 1
                                if np.array_equal(row, store.chunk(ref_sid, ref_row, ref_row+1)[0]):
                                    match = ref_sid
                                    break
                                counts["hash_collisions"] += 1
                            keep[i] = match is None
                            if match is None:
                                db.execute("INSERT INTO positions VALUES (?,?,?)", (h, sid, i))
                                counts["kept"] += 1
                            else:
                                kind = sources[match]["kind"]
                                counts["parts" if kind == "part" else kind] += 1
                        print(f"dedup part={k} rows={min(n,lo+args.chunk_rows)}/{n}", flush=True)
                    keep.flush()
                    del keep
                    with temp_mask.open("rb") as inp:
                        os.fsync(inp.fileno())
                    os.replace(temp_mask, mask_path)
                    sync_dir(out)
                    receipt = dict(part=str(part), part_json_sha256=sha256(part / "part.json"),
                                   mask=mask_path.name, mask_sha256=sha256(mask_path), counts=counts)
                    db.execute("INSERT INTO progress VALUES (?,?)", (key, json.dumps(receipt)))
                    db.commit()  # mask durable before DB commit; crash before commit simply redoes this part
                except BaseException:
                    db.rollback()
                    raise
                receipts.append(receipt)
                print("dedup_counts " + json.dumps(counts), flush=True)
            totals = {key: sum(p["counts"][key] for p in receipts) for key in receipts[0]["counts"]}
            result = dict(config=config, parts=receipts, counts=totals, seconds=time.monotonic()-started)
            atomic_json(out / "manifest.json", result)
            print("dedup_complete " + json.dumps(totals), flush=True)
        finally:
            db.close()


def assemble(args):
    """Task 144: stream accepted rows and atomically publish cache and labels."""
    dedup_dir, out = args.dedup.resolve(), args.out.resolve()
    labels = args.labels.resolve() if args.labels else Path(str(out) + "_labels")
    require(out != labels and not out.is_relative_to(labels) and not labels.is_relative_to(out),
            "cache and labels directories must be separate")
    manifest = read_json(dedup_dir / "manifest.json")
    require(read_json(dedup_dir / "config.json") == manifest["config"], "dedup config mismatch")
    tmp, lab_tmp = Path(str(out) + ".partial"), Path(str(labels) + ".partial")
    assembly_id = hashlib.sha256((sha256(dedup_dir / "manifest.json") + str(out) + str(labels)
                                 + str(args.val_cache.resolve()) + str(args.val_labels.resolve())).encode()).hexdigest()
    # A crash between the two directory renames may leave only labels published.
    if not out.exists() and labels.exists() and (tmp / "meta.json").exists():
        require(read_json(tmp / "meta.json").get("assembly_id") == assembly_id
                and read_json(labels / "meta.json").get("assembly_id") == assembly_id
                and not lab_tmp.exists(), "incompatible partial publication")
        os.replace(labels, lab_tmp)
    require(not out.exists() and not labels.exists(), "assemble targets must not exist")
    require(not any(p.is_relative_to(tmp) or p.is_relative_to(lab_tmp)
                    for p in [dedup_dir, *[Path(r["part"]) for r in manifest["parts"]]]), "output contains inputs")
    for p in (tmp, lab_tmp):
        staging(p, dict(task=144, operation="assemble", assembly_id=assembly_id))
    n = manifest["counts"]["kept"]
    require(n > 0, "dedup kept no rows; cannot assemble training data")
    targets = {name: np.lib.format.open_memmap(tmp / f"train_{name}.npy", mode="w+", dtype=dtype,
                                             shape=(n, 77) if name == "indices" else (n,))
               for name, dtype in (("indices", "<i4"), ("evals", "<i2"), ("wdl", "i1"))}
    label = np.lib.format.open_memmap(lab_tmp / "train_dlshogi_evals.npy", mode="w+", dtype="<i2", shape=(n,))
    pos, provenance = 0, []
    for receipt in manifest["parts"]:
        part = Path(receipt["part"])
        require(sha256(part / "part.json") == receipt["part_json_sha256"], "part receipt changed")
        meta = load_part(part)
        mask_path = dedup_dir / receipt["mask"]
        require(sha256(mask_path) == receipt["mask_sha256"], "dedup mask changed")
        mask = np.load(mask_path, mmap_mode="r")
        require(mask.dtype == bool and mask.shape == (meta["rows"],), "bad dedup mask")
        arrays = {dst: np.load(part / f"{src}.npy", mmap_mode="r")
                  for dst, src in (("indices", "features"), ("evals", "scores"), ("wdl", "results"))}
        count = 0
        for lo in range(0, len(mask), args.chunk_rows):
            selected = mask[lo:lo+args.chunk_rows]
            size = int(selected.sum())
            for key in arrays:
                targets[key][pos:pos+size] = arrays[key][lo:lo+args.chunk_rows][selected]
            label[pos:pos+size] = arrays["evals"][lo:lo+args.chunk_rows][selected]
            pos += size
            count += size
        require(count == receipt["counts"]["kept"], "mask count mismatch")
        provenance.append(dict(source=meta["source"], lo=meta["lo"], hi=meta["hi"], kept=count,
                               changed_policy=meta["recipe"]["changed"], part=str(part),
                               recipe=meta["recipe"],
                               part_json_sha256=receipt["part_json_sha256"]))
    require(pos == n, "assembled count mismatch")
    for a in [*targets.values(), label]:
        a.flush()
    val_cache, val_labels = args.val_cache.resolve(), args.val_labels.resolve()
    meta = read_json(val_cache / "meta.json")
    require(meta.get("canonical_shared") and meta["features_per_position"] == 77, "incompatible validation cache")
    val = np.load(val_cache / "val_indices.npy", mmap_mode="r")
    require(val.dtype == np.int32 and val.shape == (meta["n_val"], 77), "bad val features")
    for key, dtype in (("evals", np.int16), ("wdl", np.int8)):
        a = np.load(val_cache / f"val_{key}.npy", mmap_mode="r")
        require(a.shape == (len(val),) and a.dtype == dtype, "bad validation labels")
    require(np.load(val_labels / "val_dlshogi_evals.npy", mmap_mode="r").shape == (len(val),), "val label count mismatch")
    # Ensure the validation positions excluded at dedup are the ones installed here.
    val_ref = manifest["config"]["sources"][1]["file"]
    require(fingerprint(val_cache / "val_indices.npy") == val_ref, "validation differs from dedup reference")
    for filename in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
        (tmp / filename).symlink_to((val_cache / filename).resolve())
    shutil.copyfile(val_labels / "val_dlshogi_evals.npy", lab_tmp / "val_dlshogi_evals.npy")
    meta.update(n_train=n, built_by="Task 144 public shards", assembly_id=assembly_id, source_parts=provenance,
                dedup=manifest, source_meta={"format": "PackedSfenValue", "score_perspective": "leaf side to move"})
    meta.pop("qsearch_replace", None)
    atomic_json(tmp / "meta.json", meta)
    atomic_json(lab_tmp / "meta.json", dict(task=144, train=n, val=len(val), cache=str(out), assembly_id=assembly_id,
                val_source=fingerprint(val_labels / "val_dlshogi_evals.npy"),
                train_labels="original public cp, negated on odd PV; see cache source_parts"))
    sync_files(lab_tmp)
    sync_files(tmp)
    os.replace(lab_tmp, labels)
    sync_dir(labels.parent)
    try:
        os.replace(tmp, out)
        sync_dir(out.parent)
    except BaseException:
        # Recoverable staging if the second publication fails in this process.
        os.replace(labels, lab_tmp)
        raise
    print(f"assembled rows={n} cache={out} labels={labels}", flush=True)
    command = [sys.executable, "-m", "trainer.compact_cache", str(out), str(out) + "_compact"]
    print("cd " + shlex.quote(str(REPO)) + " && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
          "CUDA_VISIBLE_DEVICES='' " + shlex.join(command), flush=True)


def plan(args):
    """Task 144: offline shard plan; actual HF listing preferred over estimates."""
    dataset, stamp = {"hao": ("shogi_hao_depth9", 1695340981),
                      "tanuki": ("tanuki-.nnue-pytorch-2024-07-30.1", 1722714224)}[args.dataset]
    if args.listing:
        listing = read_json(args.listing)
        if isinstance(listing, dict):
            listing = listing["siblings"] if "siblings" in listing else listing["files"]
        files = []
        for item in listing:
            name = item.get("path", item.get("rfilename", ""))
            if Path(name).name.startswith("kifu.tag=train.depth=9.") and name.endswith(".bin"):
                size = item.get("size", (item.get("lfs") or {}).get("size"))
                require(size is not None and int(size) > 0 and int(size) % 40 == 0, f"invalid PSV size for {name}")
                files.append((name, int(size)))
        files.sort()
        require(len({name for name, _ in files}) == len(files), "listing contains duplicate shard paths")
        estimated = False
    else:
        files = [(f"kifu.tag=train.depth=9.num_positions=1000000000.start_time={stamp}.thread_index={i:03d}.bin",
                  args.estimated_bytes) for i in range(998)]
        estimated = True
    survival = (1-args.changed_drop_rate) * (1-args.dedup_drop_rate)
    chosen, expected = [], 0
    for name, size in files:
        rows = math.floor((size // 40) * survival)
        chosen.append(dict(file=name, bytes=size, raw_rows=size//40, expected_rows=rows,
                           url=f"https://huggingface.co/datasets/nodchip/{dataset}/resolve/main/{quote(name, safe='=./')}",
                           sizes_and_names_estimated=estimated))
        expected += rows
        if expected >= args.target_rows:
            break
    before_dedup = sum(math.floor(x["raw_rows"] * (1-args.changed_drop_rate)) for x in chosen)
    disk = dict(processed_parts=before_dedup * 337, cache_and_labels=expected * 313,
                compact=expected * 156, index=(7_340_673 + expected) * 40,
                download=max((x["bytes"] for x in chosen), default=0),
                working_part=200_000 * 750)
    result = dict(dataset="nodchip/"+dataset, target_rows=args.target_rows, expected_rows=expected,
                  target_reached=expected >= args.target_rows, changed_drop_rate=args.changed_drop_rate,
                  dedup_drop_rate=args.dedup_drop_rate, download_bytes=sum(x["bytes"] for x in chosen),
                  peak_download_bytes=max((x["bytes"] for x in chosen), default=0),
                  processed_bytes_estimate=disk["processed_parts"], disk_bytes_estimate=disk,
                  peak_total_bytes_estimate=sum(disk.values()),
                  note="Sequential download/process/delete. Processed parts, SQLite index and final/compact caches "
                       "also need disk. No random shuffle is performed; trainer shuffles batches. "
                       "Default 50% changed/5% dedup drops are assumptions, NOT measured on hao/tanuki.",
                  shards=chosen)
    print(json.dumps(result, indent=2))
    require(result["target_reached"], "listing exhausted before target; adjust rates/target or supply all pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("process", help="qsearch into atomic resumable parts; never removes inputs")
    p.add_argument("psv", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--yane", type=Path, default=Path(os.environ.get("YANE", DATA / "engines/yaneuraou/source/YaneuraOu-by-gcc")))
    p.add_argument("--eval-dir", type=Path, default=DATA / "eval_dirs/task129_student_64m_qsrel_e12")
    p.add_argument("--threads", type=int, choices=(1, 2), default=2)
    p.add_argument("--changed", choices=("drop", "inherit"), default="drop")
    p.add_argument("--part-rows", type=int, default=200_000)
    p.add_argument("--verify-rows", type=int, default=500)
    p.add_argument("--timeout", type=int, default=600, help="seconds per bounded qsearch part")
    p.add_argument("--keep-psv", action="store_true", help="retain marker/leaf PSV per part for audits")
    p.set_defaults(func=process)
    p = sub.add_parser("dedup", help="exclude existing train, val/mirror, then repeated new rows")
    p.add_argument("parts", type=Path, nargs="+", help="process directories or individual part directories; order matters")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--existing", type=Path, default=DATA / "caches/public_suisho5_val7m")
    p.add_argument("--val-cache", type=Path, default=DATA / "caches/shared_64m")
    p.add_argument("--chunk-rows", type=int, default=100_000)
    p.set_defaults(func=dedup)
    p = sub.add_parser("assemble", help="stream a completed dedup snapshot into NEW cache/labels directories")
    p.add_argument("dedup", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", type=Path, help="default OUT_labels")
    p.add_argument("--val-cache", type=Path, default=DATA / "caches/shared_64m")
    p.add_argument("--val-labels", type=Path, default=DATA / "work/task129/labels")
    p.add_argument("--chunk-rows", type=int, default=100_000)
    p.set_defaults(func=assemble)
    p = sub.add_parser("plan", help="offline fetch plan; --listing accepts HF tree API JSON (all pages)")
    p.add_argument("--dataset", choices=("hao", "tanuki"), required=True)
    p.add_argument("--target-rows", type=int, required=True)
    p.add_argument("--listing", type=Path)
    p.add_argument("--estimated-bytes", type=int, default=290_000_000,
                   help="conservative fallback bytes/shard (estimated names/sizes); prefer --listing")
    p.add_argument("--changed-drop-rate", type=float, default=0.5)
    p.add_argument("--dedup-drop-rate", type=float, default=0.05)
    p.set_defaults(func=plan)
    args = parser.parse_args()
    for key in ("part_rows", "chunk_rows", "timeout", "target_rows", "estimated_bytes"):
        if hasattr(args, key):
            require(getattr(args, key) > 0, f"--{key.replace('_','-')} must be positive")
    if hasattr(args, "verify_rows"):
        require(args.verify_rows >= 0, "--verify-rows must be nonnegative")
    if args.command == "plan":
        require(0 <= args.changed_drop_rate < 1 and 0 <= args.dedup_drop_rate < 1, "drop rates must be in [0,1)")
        require(args.estimated_bytes % 40 == 0, "estimated bytes must be divisible by 40")
    args.func(args)


if __name__ == "__main__":
    main()
