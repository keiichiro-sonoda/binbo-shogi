#!/usr/bin/env python3
"""Task 072 — re-label the FIRST N positions of a teacher cache with gikou at a
deeper depth (e.g. d12), CACHE-ALIGNED and CRASH-SAFE (unbiased).

Why: gikou (技巧2) EOF-crashes from warm-state accumulation when search reaches
depth>=~8. gen-teacher's self-play batch at d12 "abandons" the crashed (sharp)
games, so its surviving positions are overwhelmingly QUIET (Task 072-pre: decisive
>=5000cp 5.37% -> 0.03%) = a weaker, flatter label set than d8. But a FRESH/cold
gikou evaluates any single position at d12/d16 with no crash. So we re-label a FIXED
position set (the cache's own positions, decisive-rich) with cold gikou@d12:
  * same position distribution  -> pure one-variable (label depth d8 -> d12)
  * never abandon a position (cold-RETRY on EOF) -> no decisive-position bias.

Mechanism (dlshogi_rescore.py cache-aligned pattern): stream the first N positions
of the cache in prepare_data order from the source hcpe shards, decode hcp->sfen
(cshogi), ASSERT the stored d8 eval matches the cache's *_evals.npy[:N] (proves
order), then re-evaluate each sfen with cold gikou@depth. Output a replay-component
cache: indices/wdl reused from the source cache[:N] (depth-independent), evals = d12.

Crash-safety per worker: a fresh gikou process per chunk; proactively restart every
--restart-every positions (cap warm-state below the crash onset); on EOF/timeout,
restart and RE-SEARCH the same position cold (never skip). Resumable: each chunk's
evals are checkpointed to chunk_*.npy and skipped on re-run.
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading, queue, subprocess
import multiprocessing as mp
import numpy as np
import cshogi

try:
    from .prepare_data import shard_files
except ImportError:
    from prepare_data import shard_files

MATE_CP = 30000  # gikou `score mate N` -> sign(N)*MATE_CP (decisive, within i16; d8 set is cp-only)

# ---- worker globals (fork inheritance: avoid pickling 1M sfens) ----
_SFENS: list[str] = []
_FALLBACK: np.ndarray | None = None  # d8 evals, used only if a cold retry still fails
_ARGS = None


# --------------------------- gikou USI driver ---------------------------
class Gikou:
    def __init__(self, path):
        self.path = path
        self.p = None
        self.q = None

    def start(self):
        # gikou loads its eval (params.bin/progress.bin/...) from the CURRENT WORKING
        # DIRECTORY; without this it runs a ZERO eval (reports score cp 0 for everything,
        # still searches). Match Rust GikouEngine (gikou.rs:419 .current_dir(parent)).
        cwd = os.path.dirname(os.path.abspath(self.path))
        self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=cwd)
        self.q = queue.Queue()
        def rdr(p, q):
            for line in p.stdout:
                q.put(line.rstrip("\n"))
            q.put(None)
        threading.Thread(target=rdr, args=(self.p, self.q), daemon=True).start()
        self._send("usi")
        if self._wait("usiok", 10)[1] != "ok":
            raise RuntimeError("gikou usi handshake failed")
        self._send("setoption name Threads value 1")
        self._send("setoption name OwnBook value false")
        self._send("isready")
        if self._wait("readyok", 30)[1] != "ok":
            raise RuntimeError("gikou isready failed")
        self._send("usinewgame")

    def _send(self, s):
        try:
            self.p.stdin.write(s + "\n"); self.p.stdin.flush(); return True
        except (BrokenPipeError, OSError, AttributeError):
            return False

    def _wait(self, token, cap_s, capture_score=False):
        """Read until a line starts with token. Returns (line, status). If
        capture_score, also stash the latest `score cp/mate` payload on self._score."""
        dl = time.time() + cap_s
        while True:
            rem = dl - time.time()
            if rem <= 0:
                return None, "timeout"
            try:
                line = self.q.get(timeout=rem)
            except queue.Empty:
                return None, "timeout"
            if line is None:
                return None, "eof"
            if capture_score and line.startswith("info ") and " score " in line:
                self._score = line
            if line.startswith(token):
                return line, "ok"

    def search(self, sfen, depth, cap_s):
        """Return (cp:int|None, status). cp from side-to-move `score`. mate->+-MATE_CP.
        Watchdog: most positions finish `go depth` in <0.1s, but a sharp tail can run
        seconds. Rather than time-out->retry (catastrophic), send `stop` at cap_s and
        take gikou's current best (time-limited eval). Only EOF (real crash) -> retry."""
        self._score = None
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if not self._send(f"position sfen {sfen}"): return None, "eof"
        if not self._send(f"go depth {depth}"): return None, "eof"
        start = time.time()
        stop_dl = start + cap_s
        hard_dl = stop_dl + 3.0   # grace for gikou to emit bestmove after `stop`
        stopped = False
        while True:
            now = time.time()
            if not stopped and now >= stop_dl:
                self._send("stop"); stopped = True
            rem = (hard_dl if stopped else stop_dl) - now
            if rem <= 0:
                if not stopped:
                    self._send("stop"); stopped = True; continue
                return None, "timeout"   # gikou ignored stop (rare) -> caller cold-retries
            try:
                line = self.q.get(timeout=rem)
            except queue.Empty:
                continue
            if line is None:
                return None, "eof"
            if line.startswith("info ") and " score " in line:
                self._score = line
            if line.startswith("bestmove"):
                break
        if not self._score:
            return None, "noscore"
        t = self._score.split()
        try:
            i = t.index("score")
        except ValueError:
            return None, "noscore"
        kind, val = t[i + 1], int(t[i + 2])
        if kind == "mate":
            cp = MATE_CP if val >= 0 else -MATE_CP
        else:
            cp = val
        return int(np.clip(cp, -32767, 32767)), "ok"

    def close(self):
        try:
            if self.p:
                self._send("quit")
                self.p.kill()
        except Exception:
            pass
        self.p = None


def _relabel_chunk(task):
    """Re-label _SFENS[lo:hi] with cold gikou@depth. Save evals to chunk_path.
    Returns (chunk_id, n, n_crash, n_hardfail, elapsed)."""
    chunk_id, lo, hi, chunk_path = task
    a = _ARGS
    out = np.empty(hi - lo, dtype=np.int16)
    g = Gikou(a.gikou)
    g.start()
    since = 0
    n_crash = n_hardfail = 0
    t0 = time.time()
    for j in range(lo, hi):
        if since >= a.restart_every:
            g.close(); g.start(); since = 0
        cp, st = g.search(_SFENS[j], a.depth, a.cap_s)
        if st != "ok":
            n_crash += 1
            g.close(); g.start(); since = 0          # cold restart
            cp, st = g.search(_SFENS[j], a.depth, a.cap_s)  # cold RETRY (never abandon)
            if st != "ok":
                n_hardfail += 1
                cp = int(_FALLBACK[j])               # last resort: keep d8 label (logged)
        out[j - lo] = cp
        since += 1
    g.close()
    np.save(chunk_path, out)
    return chunk_id, hi - lo, n_crash, n_hardfail, time.time() - t0


def stream_first_n(cache_dir, n):
    """Yield the first n positions of the cache in prepare_data order: returns
    (sfens list, d8_evals np.int16) read from the source hcpe shards."""
    with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    run_dir, extras = meta["run_dir"], meta["extra_runs"]
    val_count = len(meta["val_shards"])
    primary, _ = shard_files(run_dir)
    order = [(run_dir, primary[:-val_count])] + [(rd, shard_files(rd)[0]) for rd in extras]
    sfens, d8 = [], []
    board = cshogi.Board()
    done = False
    for rd, shards in order:
        if done: break
        for sh in shards:
            ent = np.fromfile(os.path.join(rd, sh["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
            for e in ent:
                board.set_hcp(np.asarray(e["hcp"]))
                sfens.append(board.sfen())
                d8.append(int(e["eval"]))
                if len(sfens) >= n:
                    done = True; break
            if done: break
    return sfens, np.asarray(d8, dtype=np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, help="source cache (e.g. shared_32m) to re-label the first N of")
    ap.add_argument("--gikou", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of leading positions to re-label")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=20000, help="positions per checkpointed chunk")
    ap.add_argument("--restart-every", type=int, default=50, help="proactive gikou restart cadence (warm-state cap)")
    ap.add_argument("--cap-s", type=float, default=30.0, help="per-search wall cap")
    ap.add_argument("--out", required=True, help="output replay-component cache dir")
    args = ap.parse_args()
    global _ARGS, _SFENS, _FALLBACK
    _ARGS = args

    print(f"[relabel] cache={args.cache_dir} n={args.n} depth={args.depth} workers={args.workers} "
          f"chunk={args.chunk_size} restart_every={args.restart_every}", flush=True)
    t0 = time.time()
    sfens, d8 = stream_first_n(args.cache_dir, args.n)
    if len(sfens) < args.n:
        raise SystemExit(f"only {len(sfens)} positions available < n={args.n}")
    # ALIGNMENT ASSERT: stored d8 (from hcpe) == cache train_evals[:n]
    cache_ev = np.load(os.path.join(args.cache_dir, "train_evals.npy"), mmap_mode="r")
    if not np.array_equal(d8, np.asarray(cache_ev[:args.n])):
        nmis = int(np.sum(d8 != np.asarray(cache_ev[:args.n])))
        raise SystemExit(f"ALIGNMENT MISMATCH: {nmis}/{args.n} d8 evals differ from cache (order broken)")
    print(f"[relabel] streamed+aligned {args.n} positions in {time.time()-t0:.0f}s (d8==cache OK)", flush=True)

    _SFENS = sfens
    _FALLBACK = d8
    os.makedirs(args.out, exist_ok=True)
    ckdir = os.path.join(args.out, "_chunks")
    os.makedirs(ckdir, exist_ok=True)
    tasks = []
    for ci, lo in enumerate(range(0, args.n, args.chunk_size)):
        hi = min(lo + args.chunk_size, args.n)
        cpath = os.path.join(ckdir, f"chunk_{ci:05d}.npy")
        if os.path.exists(cpath) and np.load(cpath, mmap_mode="r").shape[0] == hi - lo:
            continue  # resume: skip done chunk
        tasks.append((ci, lo, hi, cpath))
    n_chunks = (args.n + args.chunk_size - 1) // args.chunk_size
    print(f"[relabel] {len(tasks)}/{n_chunks} chunks to do ({n_chunks-len(tasks)} already done)", flush=True)

    tot_crash = tot_hard = 0
    if tasks:
        with mp.Pool(args.workers) as pool:
            for ci, m, nc, nh, dt in pool.imap_unordered(_relabel_chunk, tasks):
                tot_crash += nc; tot_hard += nh
                print(f"  chunk {ci} done m={m} crash={nc} hardfail={nh} {dt:.0f}s "
                      f"({m/max(1e-9,dt):.0f}/s)", flush=True)

    # concat chunks in order
    d12 = np.empty(args.n, dtype=np.int16)
    for ci, lo in enumerate(range(0, args.n, args.chunk_size)):
        hi = min(lo + args.chunk_size, args.n)
        d12[lo:hi] = np.load(os.path.join(ckdir, f"chunk_{ci:05d}.npy"))

    # build replay-component cache: indices/wdl reused from source[:n] (depth-independent), evals=d12
    idx = np.asarray(np.load(os.path.join(args.cache_dir, "train_indices.npy"), mmap_mode="r")[:args.n])
    wdl = np.asarray(np.load(os.path.join(args.cache_dir, "train_wdl.npy"), mmap_mode="r")[:args.n])
    np.save(os.path.join(args.out, "train_indices.npy"), idx)
    np.save(os.path.join(args.out, "train_evals.npy"), d12)
    np.save(os.path.join(args.out, "train_wdl.npy"), wdl)
    meta = {"source": "gikou_relabel (Task 072)", "relabeled_from": args.cache_dir,
            "depth": args.depth, "n": int(args.n), "gikou": args.gikou,
            "restart_every": args.restart_every, "crashes": int(tot_crash), "hard_fails": int(tot_hard),
            "mate_cp": MATE_CP}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    a8, a12 = np.abs(d8.astype(np.int32)), np.abs(d12.astype(np.int32))
    print(f"[relabel] DONE n={args.n} crashes={tot_crash} hard_fails={tot_hard} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[relabel] mean|cp| d8={a8.mean():.1f} d12={a12.mean():.1f} | "
          f"decisive>=5000 d8={100*np.mean(a8>=5000):.2f}% d12={100*np.mean(a12>=5000):.2f}% | "
          f">=2000 d8={100*np.mean(a8>=2000):.2f}% d12={100*np.mean(a12>=2000):.2f}% | "
          f">=1000 d8={100*np.mean(a8>=1000):.2f}% d12={100*np.mean(a12>=1000):.2f}%", flush=True)
    print(f"[relabel] |d12-d8| median={np.median(np.abs(a12-a8)):.0f} "
          f"sign-flip={100*np.mean((d8>0)!=(d12>0)):.2f}%  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
