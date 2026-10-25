#!/usr/bin/env python3
"""Task 065 — self-reinforcement label-generation cost probe (measurement only).

Drives the current champion eval inside YaneuraOu at *fixed* search depths and
measures, on a fixed set of diverse positions:

  * per-position labeling WALL TIME at each depth (median / mean / p90), and
  * cp STABILITY across depth (Δcp(d -> d+2) distribution),

so Task 066 can pick a label depth (time vs. cp-saturation trade-off) and a
realistic per-cycle size. We label nothing for training here; we only time the
search and read its score. Champion is read-only (loaded via EvalDir).

Measurement hygiene (so the numbers are honest):
  * single YaneuraOu instance, Threads=1, serial  -> clean per-position wall time;
  * `usinewgame` (clears TT) before every `go depth d` -> each timing is a COLD
    search, which is exactly what one labeling pass pays per position;
  * the same fixed SFEN set is used at every depth -> Δcp(d->d+2) is a same-
    position depth difference.

Reuses the patterns in trainer/verify_mirror.py (hcpe->SFEN sampler + USI go-depth
driver) and the YaneuraOu setoption set from src/main.rs::start_yane_match_engine
(EvalDir / USI_OwnBook=false / Threads=1 / MinimumThinkingTime=0 / NetworkDelay*=0).

Example:
  env PYTHONPATH=$REPO $VENV_PY trainer/label_cost.py \
      --yane $YANE --eval-dir $DATA_ROOT/eval_dirs/task064_32m_lr1e-5 \
      --hcpe-run $TEACHER_DIR/run1m --positions 500 --depths 8,10,12,14 \
      --seed 20260629 --out $LOGS_DIR/task065_labelcost.tsv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

import cshogi
import numpy as np


# ---- positions: diverse, seed-reproducible sample from teacher hcpe ----------
def sample_sfens(run_dir: str, n: int, seed: int) -> list[str]:
    """Spread n positions across all done shards (random offsets, seeded)."""
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        shards = [s for s in json.load(f)["shards"] if s["done"]]
    shards.sort(key=lambda s: s["index"])
    if not shards:
        raise SystemExit(f"no done shards in {run_dir}")
    rng = np.random.default_rng(seed)
    per_shard = max(1, n // len(shards) + 1)
    board = cshogi.Board()
    sfens: list[str] = []
    for shard in shards:
        entries = np.fromfile(os.path.join(run_dir, shard["file"]),
                              dtype=cshogi.HuffmanCodedPosAndEval)
        if len(entries) == 0:
            continue
        k = min(per_shard, len(entries))
        idx = rng.choice(len(entries), size=k, replace=False)
        for i in sorted(int(x) for x in idx):
            board.set_hcp(entries[i]["hcp"])
            sfens.append(board.sfen())
            if len(sfens) >= n:
                return sfens
    return sfens


# ---- USI driver: YaneuraOu(champion EvalDir), go depth d, timed --------------
class Yane:
    def __init__(self, path: str, eval_dir: str, usi_hash: int = 256):
        self.path = path
        self.eval_dir = eval_dir
        self.usi_hash = usi_hash
        self._start()

    def _start(self):
        self.p = subprocess.Popen(
            [self.path], cwd=os.path.dirname(self.path) or ".",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("usi"); self._wait("usiok")
        for name, value in [
            ("EvalDir", self.eval_dir),
            ("USI_OwnBook", "false"),
            ("Threads", "1"),
            ("USI_Hash", str(self.usi_hash)),
            ("MinimumThinkingTime", "0"),
            ("NetworkDelay", "0"),
            ("NetworkDelay2", "0"),
        ]:
            self._send(f"setoption name {name} value {value}")
        self._send("isready"); self._wait("readyok")

    def _send(self, line: str):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()

    def _wait(self, marker: str, timeout: float = 120.0):
        end = time.time() + timeout
        while time.time() < end:
            line = self.p.stdout.readline()
            if line == "":
                raise EOFError(f"engine EOF waiting for {marker}")
            if line.strip() == marker:
                return
        raise TimeoutError(marker)

    def go_depth(self, sfen: str, depth: int, cap_s: float = 30.0):
        """Cold (TT cleared) fixed-depth search, hard-bounded to cap_s wall seconds.
        Returns (cp, is_mate, mate_val, elapsed_ms, capped). cp is None when mated.

        A watchdog thread enforces the cap regardless of engine output cadence (a
        blocking readline alone can't time out a silent engine — the depth-18 hang
        in the first run): after cap_s it sends `stop` (engine returns best-so-far),
        and if the engine ignores `stop` it kills the process (readline -> EOF ->
        caller restarts). capped=True means the cp is from a shallower-than-`depth`
        search, so the caller excludes it from the cp-stability ladder."""
        self._send("usinewgame")          # clear TT -> cold search (per-label cost)
        self._send(f"position sfen {sfen}")
        score_cp = None
        mate_val = None
        done = threading.Event()
        capped = {"v": False}

        def watchdog():
            if not done.wait(cap_s):
                capped["v"] = True
                try:
                    self._send("stop")            # force bestmove even if silent
                except Exception:
                    pass
                if not done.wait(10.0):           # engine ignored stop -> force EOF
                    try:
                        self.p.kill()
                    except Exception:
                        pass

        t0 = time.perf_counter()
        self._send(f"go depth {depth}")
        wd = threading.Thread(target=watchdog, daemon=True)
        wd.start()
        try:
            while True:
                line = self.p.stdout.readline()
                if line == "":
                    raise EOFError("engine EOF during go")
                t = line.split()
                if not t:
                    continue
                if t[0] == "info" and "score" in t:
                    j = t.index("score")
                    if t[j + 1] == "cp":
                        score_cp = int(t[j + 2]); mate_val = None
                    elif t[j + 1] == "mate":
                        mate_val = int(t[j + 2]); score_cp = None
                elif t[0] == "bestmove":
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return score_cp, (mate_val is not None), mate_val, elapsed_ms, capped["v"]
        finally:
            done.set()

    def restart(self):
        self.close(); self._start()

    def close(self):
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.p.kill()
        except Exception:
            pass


# ---- analysis ----------------------------------------------------------------
def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def analyze(rows: list[dict], depths: list[int], n_pos: int,
            workers: int, cycle_sizes: list[int], sat_cp: float):
    print("\n==== per-depth wall time (cold search, Threads=1, single instance) ====")
    print("(t_* over non-capped completions = true fixed-depth time; capped% = exceeded cap)")
    print(f"{'depth':>5} {'n':>5} {'mate%':>6} {'capped%':>8} {'t_med_ms':>10} {'t_mean_ms':>10} {'t_p90_ms':>10}")
    time_by_depth: dict[int, float] = {}
    for d in depths:
        at_d = [r for r in rows if r["depth"] == d]
        ts = [r["elapsed_ms"] for r in at_d
              if not r["capped"] and not math.isnan(r["elapsed_ms"])]
        mates = sum(1 for r in at_d if r["is_mate"])
        capped = sum(1 for r in at_d if r["capped"])
        if not at_d:
            continue
        med = statistics.median(ts) if ts else float("nan")
        time_by_depth[d] = med
        print(f"{d:>5} {len(at_d):>5} {100.0*mates/len(at_d):>5.1f}% {100.0*capped/len(at_d):>7.1f}% "
              f"{med:>10.1f} {(statistics.fmean(ts) if ts else float('nan')):>10.1f} {pct(ts,0.90):>10.1f}")

    # cp stability: same-position Δcp(d -> d+2), excluding mates and capped (their
    # cp is from a shallower-than-depth search) per pair.
    cp = {}  # (pos_id, depth) -> cp
    mate_pos = set()
    for r in rows:
        if r["is_mate"]:
            mate_pos.add(r["pos_id"])
        elif r["cp"] is not None and not r["capped"]:
            cp[(r["pos_id"], r["depth"])] = r["cp"]
    print("\n==== cp stability: Δcp(d -> d+2) over same positions (mates excluded) ====")
    print(f"{'pair':>9} {'n':>5} {'med|Δ|':>8} {'p90|Δ|':>8} {'mean|Δ|':>8} {'signflip%':>9}")
    sat_depth = None
    ladder = [(depths[i], depths[i + 1]) for i in range(len(depths) - 1)]
    for d, d2 in ladder:
        deltas, flips = [], 0
        for pid in range(n_pos):
            if pid in mate_pos:
                continue
            a = cp.get((pid, d)); b = cp.get((pid, d2))
            if a is None or b is None:
                continue
            deltas.append(abs(b - a))
            if (a > 0) != (b > 0) and a != 0 and b != 0:
                flips += 1
        if not deltas:
            continue
        med = statistics.median(deltas)
        print(f"{d:>4}->{d2:<4} {len(deltas):>5} {med:>8.1f} {pct(deltas,0.90):>8.1f} "
              f"{statistics.fmean(deltas):>8.1f} {100.0*flips/len(deltas):>8.1f}%")
        if sat_depth is None and med < sat_cp:
            sat_depth = d  # cp already ~stable entering this step
    print(f"\nsaturation: first depth with median|Δcp(d->d+2)| < {sat_cp}cp = "
          f"{sat_depth if sat_depth is not None else 'not reached in grid'}")

    # per-cycle extrapolation (serial median time / workers).
    print(f"\n==== per-cycle label-gen extrapolation (workers={workers}, "
          f"median time / worker; optimistic floor) ====")
    header = "depth " + " ".join(f"{f'{c//1000}k' if c<1_000_000 else f'{c//1_000_000}M':>9}" for c in cycle_sizes)
    print(header)
    for d in depths:
        if d not in time_by_depth:
            continue
        cells = []
        for c in cycle_sizes:
            hrs = c * (time_by_depth[d] / 1000.0) / workers / 3600.0
            cells.append(f"{hrs:>8.2f}h")
        print(f"{d:>5} " + " ".join(cells))
    print("(extrapolation = N_positions * median_per_pos_serial_time / workers; "
          "assumes ~linear scaling & no memory bottleneck = optimistic lower bound.)")


# ---- main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yane", default=os.environ.get("YANE"), help="YaneuraOu binary (default $YANE)")
    ap.add_argument("--eval-dir", required=True, help="champion EvalDir (contains nn.bin)")
    ap.add_argument("--hcpe-run", required=True, help="teacher gen-teacher run dir (manifest.json) to sample positions from")
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--depths", default="8,10,12,14")
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--usi-hash", type=int, default=256, help="USI_Hash MB per instance")
    ap.add_argument("--timeout", type=float, default=300.0, help="per go-depth timeout (s)")
    ap.add_argument("--workers", type=int, default=16, help="assumed parallel workers for cycle extrapolation")
    ap.add_argument("--cycle-sizes", default="1000000,5000000")
    ap.add_argument("--sat-cp", type=float, default=12.0, help="median|Δcp| threshold for saturation")
    ap.add_argument("--out", required=True, help="TSV output path")
    args = ap.parse_args()

    if not args.yane or not os.path.exists(args.yane):
        raise SystemExit(f"YaneuraOu binary not found: {args.yane}")
    if not os.path.exists(os.path.join(args.eval_dir, "nn.bin")):
        raise SystemExit(f"no nn.bin in eval-dir: {args.eval_dir}")
    depths = [int(x) for x in args.depths.split(",")]
    cycle_sizes = [int(x) for x in args.cycle_sizes.split(",")]

    print(f"sampling {args.positions} positions from {args.hcpe_run} (seed={args.seed}) ...", flush=True)
    sfens = sample_sfens(args.hcpe_run, args.positions, args.seed)
    n_pos = len(sfens)
    print(f"sampled {n_pos} positions; depths={depths}; eval_dir={args.eval_dir}", flush=True)
    # persist the exact position set for reproducibility (sidecar next to the TSV).
    with open(args.out + ".sfens", "w", encoding="utf-8") as f:
        for s in sfens:
            f.write(s + "\n")

    eng = Yane(args.yane, args.eval_dir, usi_hash=args.usi_hash)
    rows: list[dict] = []
    t_start = time.time()
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("pos_id\tdepth\tcp\tis_mate\tmate_val\telapsed_ms\tcapped\n")
        for pid, sfen in enumerate(sfens):
            for d in depths:
                try:
                    cp, is_mate, mate_val, ms, capped = eng.go_depth(sfen, d, cap_s=args.timeout)
                except EOFError as e:
                    print(f"  pos {pid} depth {d}: {type(e).__name__} ({e}); restarting engine", flush=True)
                    eng.restart()
                    cp, is_mate, mate_val, ms, capped = (None, False, None, float("nan"), True)
                rows.append({"pos_id": pid, "depth": d, "cp": cp, "is_mate": is_mate,
                             "elapsed_ms": ms, "capped": capped})
                out.write(f"{pid}\t{d}\t{'' if cp is None else cp}\t{int(is_mate)}\t"
                          f"{'' if mate_val is None else mate_val}\t{ms:.3f}\t{int(capped)}\n")
            if (pid + 1) % 25 == 0:
                el = time.time() - t_start
                print(f"  {pid+1}/{n_pos} positions done ({el:.0f}s elapsed)", flush=True)
    eng.close()
    print(f"\nwrote {args.out} ({len(rows)} rows); wall {time.time()-t_start:.0f}s", flush=True)

    analyze(rows, depths, n_pos, args.workers, cycle_sizes, args.sat_cp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
