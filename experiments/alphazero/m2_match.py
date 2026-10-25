# Task 077 M2 step 3: USI match driver — AZ engine (m2_usi.py) vs YaneuraOu champion
# (or gikou@6 reference). Python reimplementation of the conditions of the repo's Rust
# harnesses (yane-vs-yane / yane-vs-gikou), because no generic usi-vs-usi harness exists
# and Task 077 commits documents only.
#
# Conditions (mirroring src/main.rs):
#   yane options : EvalDir=<champion>, USI_OwnBook=false, Threads=1,
#                  MinimumThinkingTime=0, NetworkDelay=0, NetworkDelay2=0
#   gikou options: OwnBook=false, Threads=1, DepthLimit=6, MinThinkingTime=10 (cwd=gikou dir!)
#   go           : "go btime 0 wtime 0 byoyomi <ms>"   (default 100)
#   openings     : seeded uniform-random legal 6 plies; each opening played twice with
#                  colors swapped (paired); max_plies=320 => draw.
# Arbiter = cshogi.Board: legality of every move, mate, repetition (is_draw), plies.
# Result: per-game TSV + W/D/L, score, elo diff with 95% CI (normal approx on the
# per-game score {0, 0.5, 1}; elo = -400*log10(1/s - 1) applied to the CI endpoints).

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import cshogi

import m3_sprt

DATA_ROOT = os.path.expanduser("~/binbo-shogi-data")
YANE = f"{DATA_ROOT}/engines/yaneuraou/source/YaneuraOu-by-gcc"
GIKOU = f"{DATA_ROOT}/engines/gikou/gikou"
CHAMPION_EVAL = f"{DATA_ROOT}/eval_dirs/task064_32m_lr1e-5"


class UsiProc:
    def __init__(self, name, cmd, cwd=None, setoptions=(), ready_timeout=600):
        self.name = name
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )
        self.q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("usi")
        self._wait("usiok", 30)
        for k, v in setoptions:
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait("readyok", ready_timeout)

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line.strip())
        self.q.put(None)  # EOF

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait(self, token, timeout):
        t0 = time.time()
        while True:
            remaining = timeout - (time.time() - t0)
            if remaining <= 0:
                raise TimeoutError(f"{self.name}: timeout waiting {token}")
            line = self.q.get(timeout=remaining)
            if line is None:
                raise EOFError(f"{self.name}: engine EOF waiting {token}")
            if line.startswith(token):
                return line

    def newgame(self):
        self._send("usinewgame")

    def bestmove(self, moves, go_cmd, timeout=90):
        pos = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos)
        self._send(go_cmd)
        t0 = time.time()
        line = self._wait("bestmove", timeout)
        return line.split()[1], time.time() - t0

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def random_opening(rng, plies):
    board = cshogi.Board()
    moves = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            return None
        m = legal[rng.integers(len(legal))]
        moves.append(cshogi.move_to_usi(m))
        board.push(m)
    return moves if list(board.legal_moves) else None


def play_game(black, white, opening, go_cmd, max_plies, log):
    """Return (result_for_black in {1,0.5,0}, reason, plies)."""
    board = cshogi.Board()
    moves = list(opening)
    for m in moves:
        board.push_usi(m)
    times = {black.name: [], white.name: []}
    black.newgame(); white.newgame()
    while len(moves) < max_plies:
        legal = {cshogi.move_to_usi(m) for m in board.legal_moves}
        if not legal:
            return (0.0, "mate", len(moves)) if board.turn == cshogi.BLACK else (1.0, "mate", len(moves))
        rep = board.is_draw()
        if rep == cshogi.REPETITION_DRAW:
            return (0.5, "sennichite", len(moves))
        if rep == cshogi.REPETITION_WIN:  # side to move wins (opponent perpetual check)
            return (1.0 if board.turn == cshogi.BLACK else 0.0, "rep_win", len(moves))
        if rep == cshogi.REPETITION_LOSE:
            return (0.0 if board.turn == cshogi.BLACK else 1.0, "rep_lose", len(moves))
        mover = black if board.turn == cshogi.BLACK else white
        mover_is_black = board.turn == cshogi.BLACK
        try:
            mv, dt = mover.bestmove(moves, go_cmd)
        except (TimeoutError, EOFError) as e:
            log(f"  ENGINE_FAIL {mover.name}: {e}")
            return (0.0 if mover_is_black else 1.0, f"engine_fail:{mover.name}", len(moves))
        times[mover.name].append(dt)
        if mv == "resign":
            return (0.0 if mover_is_black else 1.0, "resign", len(moves))
        if mv == "win":
            return (1.0 if mover_is_black else 0.0, "declare_win", len(moves))
        if mv not in legal:
            log(f"  ILLEGAL by {mover.name}: {mv}")
            return (0.0 if mover_is_black else 1.0, f"illegal:{mv}", len(moves))
        board.push_usi(mv)
        moves.append(mv)
    return (0.5, "max_plies", len(moves))


def wdl(scores):
    a = np.asarray(scores, dtype=np.float64)
    return int((a == 1).sum()), int((a == 0.5).sum()), int((a == 0).sum())


def summarize(scores):
    w, d, l = wdl(scores)
    s, lo, hi = m3_sprt.score_ci(w, d, l)
    e = m3_sprt.elo_from_score
    return (f"az W-D-L {w}-{d}-{l}/{len(scores)} score {s:.4f} (95%CI {lo:.4f}..{hi:.4f}) "
            f"elo {e(s):+.1f} (CI {e(lo):+.1f}..{e(hi):+.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--az-ckpt", required=True)
    ap.add_argument("--az-sims", type=int, default=16)
    ap.add_argument("--opponent", choices=["yane", "gikou", "az2"], default="yane")
    ap.add_argument("--opp-ckpt", default=None, help="checkpoint for --opponent az2")
    ap.add_argument("--opp-sims", type=int, default=128)
    ap.add_argument("--sprt", action="store_true",
                    help="sequential trinomial SPRT (H1: az >= elo1 stronger); stops early on a verdict")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=10.0)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--byoyomi", type=int, default=100)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = os.path.expanduser("~/task074_frameworks/venv_pgx/bin/python")
    az_cmd = [py, os.path.join(here, "m2_usi.py"), "--ckpt", args.az_ckpt, "--sims", str(args.az_sims)]

    tsv_path = args.tsv or os.path.join(here, f"m2_match_{args.opponent}_sims{args.az_sims}.tsv")
    tsv = open(tsv_path, "a")

    def log(msg):
        print(msg, flush=True)

    log(f"az: ckpt={args.az_ckpt} sims={args.az_sims}")
    log(f"opponent={args.opponent} byoyomi={args.byoyomi} games={args.games} "
        f"opening_plies={args.opening_plies} max_plies={args.max_plies} seed={args.seed}")

    az = UsiProc("az", az_cmd)
    if args.opponent == "az2":
        assert args.opp_ckpt, "--opp-ckpt required for az2"
        opp = UsiProc("az2", [py, os.path.join(here, "m2_usi.py"),
                              "--ckpt", args.opp_ckpt, "--sims", str(args.opp_sims)])
        log(f"az2: ckpt={args.opp_ckpt} sims={args.opp_sims}")
    elif args.opponent == "yane":
        opp = UsiProc("yane", [YANE], setoptions=[
            ("EvalDir", CHAMPION_EVAL), ("USI_OwnBook", "false"), ("Threads", "1"),
            ("MinimumThinkingTime", "0"), ("NetworkDelay", "0"), ("NetworkDelay2", "0"),
        ])
        log(f"yane_options=EvalDir:{CHAMPION_EVAL},USI_OwnBook:false,Threads:1,MinimumThinkingTime:0,NetworkDelay:0,NetworkDelay2:0")
    else:
        opp = UsiProc("gikou", [GIKOU], cwd=os.path.dirname(GIKOU), setoptions=[
            ("OwnBook", "false"), ("Threads", "1"), ("DepthLimit", "6"), ("MinThinkingTime", "10"),
        ])
        log("gikou_options=OwnBook:false,Threads:1,DepthLimit:6,MinThinkingTime:10")

    go_cmd = f"go btime 0 wtime 0 byoyomi {args.byoyomi}"
    rng = np.random.default_rng(args.seed)
    scores = []  # az perspective
    reasons = {}
    t0 = time.time()
    g = 0
    sprt_verdict = None
    while g < args.games and sprt_verdict is None:
        opening = None
        while opening is None:
            opening = random_opening(rng, args.opening_plies)
        for az_black in (True, False):
            if g >= args.games:
                break
            black, white = (az, opp) if az_black else (opp, az)
            r_black, reason, plies = play_game(black, white, opening, go_cmd, args.max_plies, log)
            s_az = r_black if az_black else 1.0 - r_black
            scores.append(s_az)
            reasons[reason] = reasons.get(reason, 0) + 1
            tsv.write(f"{g}\t{'b' if az_black else 'w'}\t{s_az}\t{reason}\t{plies}\t{' '.join(opening)}\n")
            tsv.flush()
            g += 1
            if args.sprt:
                w_, d_, l_ = wdl(scores)
                llr, verdict = m3_sprt.sprt(w_, d_, l_, args.elo0, args.elo1)
                if verdict != m3_sprt.CONTINUE:
                    lo_b, hi_b = m3_sprt.bounds()
                    log(f"SPRT {verdict} llr={llr:.3f} bounds=({lo_b:.3f},{hi_b:.3f}) after {g} games")
                    sprt_verdict = verdict
                    break
            if g % 10 == 0:
                extra = ""
                if args.sprt:
                    w_, d_, l_ = wdl(scores)
                    llr, _ = m3_sprt.sprt(w_, d_, l_, args.elo0, args.elo1)
                    extra = f" llr={llr:.3f}"
                log(f"[{g}/{args.games}] {summarize(scores)}{extra}  ({(time.time()-t0)/60:.1f} min)")
    az.quit(); opp.quit()
    if args.sprt:
        w_, d_, l_ = wdl(scores)
        llr, verdict = m3_sprt.sprt(w_, d_, l_, args.elo0, args.elo1)
        log(f"SPRT_FINAL verdict={verdict} llr={llr:.3f} elo0={args.elo0} elo1={args.elo1} games={g}")
    log(f"FINAL {summarize(scores)}")
    log(f"reasons: {sorted(reasons.items(), key=lambda kv: -kv[1])}")
    log(f"tsv: {tsv_path}")
    log("MATCH_DONE")


if __name__ == "__main__":
    main()
