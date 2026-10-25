#!/usr/bin/env python3
"""Task 146: deterministic CPU self-search labels; source scripts/env.sh first.

Scores are side-to-move INTERNAL YaneuraOu values, not displayed centipawns.
For this engine cp = trunc(100*v/90); invert exactly with sign*ceil(abs(cp)*90/100).
Mate (including mate -0) maps to +/-32000; the trainer later clamps to +/-2000.
Every part starts with usinewgame + isready (this build recreates workers and clears
TT/history; per position it cost ~133 ms vs ~8 ms of search, so it was moved to the
part boundary) and searches its rows in a fixed order, Threads=1, fixed hash, no
book/ponder/time limit. Node searches use
the last unbounded PV score, not an unfinished iteration's aspiration bound.
Parts contain cache row numbers and int16 scores, never altered cache positions.
"""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

sys.dont_write_bytecode = True
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import cshogi  # noqa: E402
import numpy as np  # noqa: E402
from trainer import kp  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
SCORE_RE = re.compile(r"\bscore (cp|mate) ([+-]?\d+|[+-])(?=\s|$)")
OPTIONS = {"Threads": 1, "NumaPolicy": "none", "MultiPV": 1,
           "USI_OwnBook": "false", "BookFile": "no_book", "USI_Ponder": "false",
           "Stochastic_Ponder": "false", "DepthLimit": 0, "NodesLimit": 0,
           "PvInterval": 0, "OutputFailLHPV": "false", "ConsiderationMode": "true",
           "FV_SCALE": 16, "DrawValueBlack": -2, "DrawValueWhite": -2,
           "MaxMovesToDraw": 0, "EnteringKingRule": "CSARule27", "ResignValue": 99999}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, writer):
    """Task 146: publish only a flushed file, then persist the directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_write(path, obj):
    atomic(path, lambda f: f.write((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()))


def parent_death(expected_parent):
    """Task 146: Linux child cleanup also works when the driver is SIGKILLed."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != expected_parent:
        os._exit(125)


def internal_score(kind, value):
    sign = -1 if str(value).startswith("-") else 1
    if kind == "mate":
        return sign * 32000
    return sign * ((abs(int(value)) * 90 + 99) // 100)


class Engine:
    """Task 146: bounded binary-pipe USI client with no reader-thread leaks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = b""
        self.lines = deque(maxlen=12)
        self.selector = selectors.DefaultSelector()
        parent = os.getpid()
        self.proc = subprocess.Popen([cfg["engine"]], cwd=REPO, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     bufsize=0, start_new_session=True,
                                     preexec_fn=lambda: parent_death(parent))
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            self.send("usi")
            header = self.until("usiok")
            available = {s.split(" type ")[0][12:] for s in header if s.startswith("option name ")}
            for key, value in cfg["options"].items():
                require(key in available, f"engine option missing: {key}")
                self.send(f"setoption name {key} value {value}")
            self.reset()
        except BaseException:
            self.close()
            raise

    def send(self, line):
        self.proc.stdin.write((line + "\n").encode())

    def readline(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError(f"USI timeout pid={self.proc.pid}; tail={list(self.lines)}")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"engine EOF pid={self.proc.pid}; tail={list(self.lines)}")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.decode(errors="replace").strip()
        self.lines.append(line)
        return line

    def until(self, prefix):
        deadline = time.monotonic() + self.cfg["timeout"]
        lines = []
        while True:
            line = self.readline(deadline)
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def reset(self):
        self.send("usinewgame")
        self.send("isready")
        self.until("readyok")

    def search(self, sfen):
        # Task 146 (Claude, measured): isready per position costs ~133 ms vs ~8 ms for a depth-9 search, so the
        # engine is reset once per part (label_job) and the part's rows are searched in a fixed order; the
        # transposition table carries over inside a part only, so results stay a function of the part alone.
        self.send("position sfen " + sfen)
        limit = dict(self.cfg["limit"])
        cap = self.cfg.get("node_cap")
        if cap and "nodes" not in limit:
            limit["nodes"] = cap  # whichever comes first; see run(): only pathological positions reach it
        self.send("go " + " ".join(f"{k} {v}" for k, v in limit.items()))
        deadline = time.monotonic() + self.cfg["timeout"]
        score, depth, nodes, mate, raw = None, 0, 0, False, 0
        while True:
            line = self.readline(deadline)
            if line.startswith("bestmove "):
                bestmove = line.split()[1]
                require(score is not None, f"bestmove without exact score: {sfen}: {list(self.lines)}")
                if "depth" in limit and "nodes" not in limit and not mate:
                    require(depth >= limit["depth"], f"incomplete depth: {line}, depth={depth}")
                # node-capped (depth < limit): the last exact (unbounded) score is kept, as for node searches
                return score, depth, nodes, mate, raw, bestmove
            if not line.startswith("info ") or " string " in line:
                continue
            tokens = line.split()
            if "multipv" in tokens and tokens[tokens.index("multipv") + 1] != "1":
                continue
            match = SCORE_RE.search(line)
            if match and "lowerbound" not in tokens and "upperbound" not in tokens:
                kind, value = match.groups()
                score = internal_score(kind, value)
                raw = int(value) if value not in ("+", "-") else (1 if value == "+" else -1)
                depth = int(tokens[tokens.index("depth") + 1]) if "depth" in tokens else 0
                nodes = int(tokens[tokens.index("nodes") + 1]) if "nodes" in tokens else 0
                mate = kind == "mate"

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("quit")
                try:
                    self.proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        except (BrokenPipeError, ProcessLookupError):
            self.proc.wait()
        finally:
            self.selector.close()
            self.proc.stdin.close()
            self.proc.stdout.close()


def array_info(path):
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    stat = path.stat()
    require(arr.dtype == np.int32 and arr.ndim == 2 and arr.shape[1] == 77,
            f"expected int32 (N,77): {path}")
    return {"path": str(path.resolve()), "shape": list(arr.shape), "dtype": str(arr.dtype),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def config(args):
    cache, engine, evaluation = args.cache.resolve(), args.engine.resolve(), args.eval_dir.resolve()
    require(not (evaluation / "eval_options.txt").exists(), "eval_options.txt must be absent (implicit overrides)")
    sources = {s: array_info(cache / f"{s}_indices.npy") for s in ("train", "val")}
    return {"engine": str(engine), "engine_sha256": digest(engine), "eval_dir": str(evaluation),
            "eval_sha256": digest(evaluation / "nn.bin"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "script_sha256": digest(__file__), "sources": sources,
            "cache_meta_sha256": digest(cache / "meta.json"),
            "limit": {k: getattr(args, k) for k in ("depth", "nodes") if getattr(args, k)},
            "options": dict(OPTIONS, USI_Hash=args.hash_mb, EvalDir=str(evaluation)),
            "score_unit": "YaneuraOu internal Value; sign(cp)*ceil(abs(cp)*90/100); mate +/-32000",
            "perspective": "side_to_move", "reset": "usinewgame + isready once per part; rows in fixed order within the part",
            "versions": {"numpy": np.__version__, "cshogi": getattr(cshogi, "__version__", "unknown")}}


_CFG = _ARRAYS = _ENGINE = None


def worker_init(cfg, parent):
    global _CFG, _ARRAYS, _ENGINE
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    parent_death(parent)
    _CFG = cfg
    _ARRAYS = {s: np.load(v["path"], mmap_mode="r") for s, v in cfg["sources"].items()}
    _ENGINE = None


def material(board):
    values = (0, 100, 300, 300, 400, 700, 800, 500, 0, 500, 500, 500, 500, 900, 1000)
    total = sum(values[p & 15] * (1 if p < 16 else -1) for p in board.pieces)
    hand_values = (100, 300, 300, 400, 500, 700, 800)
    for color, hand in enumerate(board.pieces_in_hand):
        total += (1 if color == 0 else -1) * sum(v * n for v, n in zip(hand_values, hand))
    return total * (1 if board.turn == 0 else -1)


def label_job(job):
    """Task 146: retry the WHOLE part with a new engine after EOF/timeout."""
    global _ENGINE
    split, rows, destination = job
    rows = np.asarray(rows, dtype=np.int64)
    for attempt in range(_CFG["retries"] + 1):
        started = time.monotonic()
        try:
            if _ENGINE is None:
                _ENGINE = Engine(_CFG)
            _ENGINE.reset()  # once per part: fresh worker + TT/history (see Engine.search)
            if _CFG.get("verbose"):
                print(f"part_start {split} {rows[0]} engine_pid={_ENGINE.proc.pid}", flush=True)
            scores = np.empty(len(rows), np.int16)
            depths, counts = np.empty(len(rows), np.int16), np.empty(len(rows), np.int64)
            mates, raws = np.empty(len(rows), np.bool_), np.empty(len(rows), np.int32)
            turns, mats = np.empty(len(rows), np.int8), np.empty(len(rows), np.int32)
            for i, row in enumerate(rows):
                sfen = kp.canonical_shared_row_to_sfen(_ARRAYS[split][row])
                board = cshogi.Board(sfen)
                score, depth, nodes, mate, raw, move = _ENGINE.search(sfen)
                require(-32000 <= score <= 32000, f"score outside int16 policy: {score}")
                if move not in ("resign", "win"):
                    require(move in {cshogi.move_to_usi(m) for m in board.legal_moves},
                            f"illegal bestmove {move}: {sfen}")
                scores[i], depths[i], counts[i], mates[i], raws[i] = score, depth, nodes, mate, raw
                turns[i], mats[i] = board.turn, material(board)
            result = dict(rows=rows, scores=scores, depths=depths, nodes=counts, mates=mates,
                          raw_scores=raws, turns=turns, material=mats,
                          seconds=np.array(time.monotonic() - started),
                          recipe_id=np.array(_CFG.get("recipe_id", "bench")), worker=np.array(os.getpid()))
            if destination:
                atomic(destination, lambda f: np.savez(f, **result))
                return split, len(rows), float(result["seconds"]), int(result["worker"])
            return result
        except (RuntimeError, TimeoutError, OSError) as exc:
            print(f"T146 retry split={split} row={rows[0]} attempt={attempt + 1}: {exc}", file=sys.stderr, flush=True)
            if _ENGINE:
                _ENGINE.close()
                _ENGINE = None
            if attempt == _CFG["retries"]:
                raise


def pool_for(cfg, procs):
    return mp.get_context("spawn").Pool(procs, initializer=worker_init, initargs=(cfg, os.getpid()))


def corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def statistics(scores):
    return {"n": len(scores), "mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "quantiles_0_1_5_25_50_75_95_99_100": np.percentile(scores, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist(),
            "abs_ge_2000": float(np.mean(np.abs(scores.astype(np.int32)) >= 2000)),
            "positive": float(np.mean(scores > 0)), "zero": float(np.mean(scores == 0))}


def benchmark(args, cfg):
    require(args.rows <= cfg["sources"]["train"]["shape"][0], "sample exceeds train")
    rows = np.random.default_rng(args.seed).choice(cfg["sources"]["train"]["shape"][0], args.rows, replace=False)
    # Fixed 25-row jobs (not procs-dependent): the engine resets once per job, so the job boundaries must not
    # depend on --procs for the scores to be reproducible across process counts.
    jobs = [("train", rows[i:i + 25], None) for i in range(0, len(rows), 25)]
    started = time.monotonic()
    with pool_for(cfg, args.procs) as pool:
        results = list(pool.imap_unordered(label_job, jobs))
    elapsed = time.monotonic() - started
    arrays = {k: np.concatenate([r[k] for r in results]) for k in
              ("rows", "scores", "depths", "nodes", "mates", "raw_scores", "turns", "material")}
    order = np.argsort(arrays["rows"])
    arrays = {k: a[order] for k, a in arrays.items()}
    old = np.load(args.old_labels / "train_dlshogi_evals.npy", mmap_mode="r")
    require(old.dtype == np.int16 and len(old) == cfg["sources"]["train"]["shape"][0], "WS-T alignment mismatch")
    reference = np.asarray(old[arrays["rows"]], dtype=np.float64)
    workers = {}
    for r in results:
        w = workers.setdefault(str(int(r["worker"])), {"rows": 0, "seconds": 0.0})
        w["rows"] += len(r["rows"])
        w["seconds"] += float(r["seconds"])
    for w in workers.values():
        w["positions_s_active"] = w["rows"] / w["seconds"]
    rate = args.rows / elapsed
    score = arrays["scores"].astype(float)
    report = dict(task=146, config=cfg, seed=args.seed, elapsed_s=elapsed, procs=args.procs,
                  positions_s_total=rate, positions_s_per_process=rate / args.procs, workers=workers,
                  scores=statistics(arrays["scores"]), wst_correlation=corr(score, reference),
                  wst_clamped_correlation=corr(np.clip(score, -2000, 2000), np.clip(reference, -2000, 2000)),
                  median_abs_delta_wst=float(np.median(np.abs(score - reference))),
                  depth_quantiles=np.percentile(arrays["depths"], [0, 25, 50, 75, 100]).tolist(),
                  mean_nodes=float(arrays["nodes"].mean()), mate_fraction=float(arrays["mates"].mean()),
                  material_correlation_by_turn={str(t): corr(arrays["material"][arrays["turns"] == t],
                                                              score[arrays["turns"] == t]) for t in (0, 1)},
                  wall_hours_63950000_linear={str(p): 63950000 / (rate / args.procs * p) / 3600 for p in (4, 8, 20, 28)},
                  wall_hours_63950000_28cpu_derated=[63950000 / (rate / args.procs * 28 * f) / 3600 for f in (.8, .6)],
                  extrapolation="28-CPU estimates, not measurements: linear optimistic; derated 60-80% for P/E+HT, bandwidth, load")
    out = args.out or args.work / f"bench_d{args.depth or 0}_n{args.nodes or 0}_s{args.seed}.json"
    json_write(out, report)
    atomic(out.with_suffix(".npz"), lambda f: np.savez(f, **arrays, wst=reference))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def part_jobs(manifest, work):
    for split in ("train", "val"):
        for start in range(0, manifest["counts"][split], manifest["part_rows"]):
            end = min(start + manifest["part_rows"], manifest["counts"][split])
            yield split, range(start, end), work / "parts" / f"{split}_{start:09d}_{end:09d}.npz"


def read_part(path, rows, recipe_id):
    rows = np.asarray(rows, dtype=np.int64)
    with np.load(path, allow_pickle=False) as a:
        require(str(a["recipe_id"]) == recipe_id and a["rows"].dtype == np.int64 and
                np.array_equal(a["rows"], rows), f"part identity mismatch: {path}")
        scores = a["scores"]
        require(scores.dtype == np.int16 and scores.shape == rows.shape and
                np.all(np.abs(scores.astype(np.int32)) <= 32000), f"invalid scores: {path}")
        return scores.copy()


def locked(work):
    work.mkdir(parents=True, exist_ok=True)
    handle = (work / ".lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another run/assemble holds {work}/.lock") from None
    return handle


def run(args, cfg):
    counts = {s: getattr(args, s + "_rows") if getattr(args, s + "_rows") is not None
              else cfg["sources"][s]["shape"][0] for s in ("train", "val")}
    for s, n in counts.items():
        require(0 <= n <= cfg["sources"][s]["shape"][0], f"invalid {s} row count")
    identity = dict(config={k: v for k, v in cfg.items() if k not in ("timeout", "retries", "verbose", "node_cap")},
                    counts=counts, part_rows=args.part_rows, format_version=1)
    recipe_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
    manifest = dict(identity, recipe_id=recipe_id)
    # Task 146 (Claude, 2026-10-20): a code-only change (script_sha256 / git_head) may resume an existing work dir
    # when everything that determines the scores is identical. Added for --node-cap: the cap must exceed every
    # node count already stored in the reused parts (checked below), so those parts are exactly what the new code
    # would produce; only positions that would otherwise time out (> 120 s) are capped.
    code_keys = ("script_sha256", "git_head")

    def scoring(m):
        return dict(m, config={k: v for k, v in m["config"].items() if k not in code_keys}, recipe_id=None)
    with locked(args.work):
        path = args.work / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            require(scoring(old) == scoring(manifest), "manifest mismatch; use another --work (never mix recipes)")
            if old != manifest:
                recipe_id, manifest = old["recipe_id"], dict(manifest, recipe_id=old["recipe_id"])
                history = old.get("code_history", []) + [{k: cfg[k] for k in code_keys} | {"node_cap": cfg.get("node_cap")}]
                json_write(path, dict(old, code_history=history))
                print(f"T146 resume with changed code (scores unaffected): {history[-1]}", flush=True)
        else:
            json_write(path, manifest)
        if cfg.get("node_cap"):
            biggest = 0
            for _s, _r, part in part_jobs(manifest, args.work):
                if part.exists():
                    with np.load(part) as z:
                        biggest = max(biggest, int(z["nodes"].max()))
            require(cfg["node_cap"] > biggest, f"--node-cap {cfg['node_cap']} must exceed reused max nodes {biggest}")
            print(f"T146 node_cap={cfg['node_cap']} > reused max nodes {biggest}", flush=True)
        cfg["recipe_id"] = recipe_id
        pending, reused = [], 0
        for split, rows, path in part_jobs(manifest, args.work):
            try:
                read_part(path, rows, recipe_id)
                reused += len(rows)
            except (OSError, ValueError, EOFError, KeyError, RuntimeError, zipfile.BadZipFile):
                pending.append((split, rows, path))
        # Queued ranges stay compact even for all 64M rows.
        total, done = sum(counts.values()), reused
        print(f"T146 start total={total} reused={reused} pending_parts={len(pending)}", flush=True)
        started, last = time.monotonic(), time.monotonic()
        with pool_for(cfg, args.procs) as pool:
            iterator = pool.imap_unordered(label_job, pending, chunksize=1)
            left = len(pending)
            while left:
                try:
                    split, n, seconds, worker = iterator.next(timeout=1)
                    done += n
                    left -= 1
                except mp.TimeoutError:
                    pass
                now = time.monotonic()
                if now - last >= args.log_seconds or left == 0:
                    rate = (done - reused) / max(now - started, .001)
                    eta = (total - done) / rate / 3600 if rate else None
                    state = dict(done=done, total=total, reused=reused, new=done-reused,
                                 positions_s=rate, eta_hours=eta, elapsed_s=now-started)
                    json_write(args.work / "progress.json", state)
                    print("T146 " + canonical(state), flush=True)
                    last = now
        if not pending:
            print(f"T146 complete reused={reused}; no searches needed", flush=True)


def assemble(args):
    """Task 146: stage both aligned arrays; meta.json becomes visible last."""
    with locked(args.work):
        manifest = json.loads((args.work / "manifest.json").read_text())
        out = args.out.resolve()
        require(not out.exists(), "--out must be a new directory; interrupted staging is safe to remove")
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="." + out.name + ".", dir=out.parent))
        try:
            for split in ("train", "val"):
                path = staging / f"{split}_dlshogi_evals.npy"
                arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16,
                                                shape=(manifest["counts"][split],))
                for s, rows, part in part_jobs(manifest, args.work):
                    if s == split:
                        arr[rows] = read_part(part, rows, manifest["recipe_id"])
                arr.flush()
                del arr
                with path.open("rb") as f:
                    os.fsync(f.fileno())
            full = all(manifest["counts"][s] == manifest["config"]["sources"][s]["shape"][0] for s in ("train", "val"))
            meta = dict(manifest, task=146, train=manifest["counts"]["train"], val=manifest["counts"]["val"],
                        complete_cache=full, slice="full" if full else "prefix fixture; not a full-cache training label directory",
                        label_sha256={s: digest(staging / f"{s}_dlshogi_evals.npy") for s in ("train", "val")})
            json_write(staging / "meta.json", meta)
            sync_dir(staging)
            os.rename(staging, out)
            sync_dir(out.parent)
        except BaseException:
            import shutil
            shutil.rmtree(staging)
            raise
        print(f"T146 assembled {out} train={meta['train']} val={meta['val']} full_cache={full}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("bench", "run", "assemble"):
        q = sub.add_parser(name)
        q.add_argument("--work", type=Path, default=DATA / "work/task146_dryrun")
        if name == "assemble":
            q.add_argument("--out", required=True, type=Path)
            continue
        q.add_argument("--cache", type=Path, default=DATA / "caches/shared_64m_qs129")
        q.add_argument("--engine", type=Path, default=Path(os.environ.get("YANE", DATA / "engines/yaneuraou/source/YaneuraOu-by-gcc")))
        q.add_argument("--eval-dir", type=Path, default=DATA / "eval_dirs/task142_finetune")
        limits = q.add_mutually_exclusive_group(required=True) if name == "run" else q
        limits.add_argument("--depth", type=int)
        limits.add_argument("--nodes", type=int)
        q.add_argument("--procs", type=int, default=4)
        q.add_argument("--hash-mb", type=int, default=16)
        q.add_argument("--timeout", type=float, default=120)
        q.add_argument("--retries", type=int, default=2)
        q.add_argument("--node-cap", type=int, default=None,
                       help="with --depth: also stop at this many nodes (pathological positions); not part of the recipe")
        q.add_argument("--verbose", action="store_true")
        if name == "bench":
            q.add_argument("--rows", type=int, default=2000)
            q.add_argument("--seed", type=int, default=20271460)
            q.add_argument("--old-labels", type=Path, default=DATA / "work/task129/labels")
            q.add_argument("--out", type=Path)
        else:
            q.add_argument("--part-rows", type=int, default=1000)
            q.add_argument("--train-rows", type=int, help="prefix fixture only; default ALL train rows")
            q.add_argument("--val-rows", type=int, help="prefix fixture only; default ALL 50k val rows")
            q.add_argument("--log-seconds", type=float, default=10)
    return p


def main():
    args = parser().parse_args()
    if args.command == "assemble":
        assemble(args)
        return
    require(1 <= args.procs <= 28 and args.hash_mb > 0 and args.timeout > 0 and args.retries >= 0, "invalid resource limits")
    require((args.depth or args.nodes) and all(v is None or v > 0 for v in (args.depth, args.nodes)), "positive depth/nodes required")
    require(getattr(args, "rows", 1) > 0 and getattr(args, "part_rows", 1) > 0, "positive row counts required")
    cfg = config(args)
    cfg.update(timeout=args.timeout, retries=args.retries, verbose=args.verbose,
               node_cap=getattr(args, "node_cap", None))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.command == "bench":
        benchmark(args, cfg)
    else:
        run(args, cfg)


if __name__ == "__main__":
    main()
