#!/usr/bin/env python3
"""Micro-benchmark for the qsearch control (docs/BACKLOG.md): speed and agreement only.

Two ways to find the quiet position Q below a training position P, both a
capture-only alpha-beta (all evasions when in check, stand-pat otherwise):
  * fast  : stand-pat / move ordering by MATERIAL; WS-T evaluates only the final Q.
  * wst   : stand-pat by the WS-T raw value at every node (batch-1 calls, cached).
For each P the label is the WS-T raw value of Q, from P's side to move.

What this can and cannot say: it measures cost (positions/s -> hours for 32M)
and how far the two variants disagree (same Q? |label difference|), plus how
far each moves from the raw value and how close each lands to the S=8 search
label. It does NOT measure accuracy: Task 099 showed that closeness to a
gikou-anchored gold judged search labels worse, and games later showed them
better. Only training + games decide accuracy.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from m2_usi import load_model  # noqa: E402
from t099_search_label import (build_batched_search, raw_value_fn, sample_run2m, score_sfens,  # noqa: E402
                               sfens_to_state, value_to_cp)

WST = Path.home() / "task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt"
# material in cp, indexed by cshogi piece type 1..14 (P L N S G B R K +P +L +N +S +B +R)
VALUE = [0, 90, 315, 405, 495, 540, 855, 990, 0, 540, 540, 540, 540, 945, 1395]
HAND_VALUE = [90, 315, 405, 495, 540, 855, 990]          # cshogi hand order P L N S G B R
MATE = 30000.0
MAX_PLY = 16                                             # capture-sequence cap


def material(board):
    """Material balance from the side to move, in cp."""
    s = 0
    for code in board.pieces:
        if code:
            v = VALUE[code & 15]
            s += v if (code < 16) == (board.turn == cshogi.BLACK) else -v
    own, opp = board.pieces_in_hand[board.turn], board.pieces_in_hand[1 - board.turn]
    s += sum(c * v for c, v in zip(own, HAND_VALUE)) - sum(c * v for c, v in zip(opp, HAND_VALUE))
    return float(s)


def candidate_moves(board):
    in_check = board.is_check()
    moves = [m for m in board.legal_moves if in_check or cshogi.move_cap(m)]
    # MVV ordering: biggest victim first
    moves.sort(key=lambda m: -VALUE[cshogi.move_cap(m) & 15] if cshogi.move_cap(m) else 0)
    return moves, in_check


class QSearch:
    """Capture-only negamax with alpha-beta; returns (value, pv_moves)."""

    def __init__(self, evaluate):
        self.evaluate = evaluate      # board -> cp from side to move
        self.nodes = 0
        self.truncated = 0

    def search(self, board, alpha, beta, ply):
        self.nodes += 1
        moves, in_check = candidate_moves(board)
        if in_check and not moves:
            return -MATE + ply, []
        best_pv = []
        if not in_check:
            stand = self.evaluate(board)
            if stand >= beta or ply >= MAX_PLY:
                if ply >= MAX_PLY and moves:
                    self.truncated += 1
                return stand, []
            alpha = max(alpha, stand)
            best = stand
        else:
            best = -MATE
            if ply >= MAX_PLY:
                self.truncated += 1
                return self.evaluate(board), []
        for m in moves:
            board.push(m)
            value, pv = self.search(board, -beta, -alpha, ply + 1)
            board.pop()
            value = -value
            if value > best:
                best, best_pv = value, [m] + pv
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best, best_pv


def quiet_of(board, pv):
    for m in pv:
        board.push(m)
    sfen = board.sfen()
    for _ in pv:
        board.pop()
    return sfen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--wst-positions", type=int, default=500, help="subset for the slow WS-T-guided variant")
    parser.add_argument("--run-dir", default=str(Path.home() / "binbo-shogi-data/teacher/run32m_g"))
    parser.add_argument("--seed", type=int, default=20271004)
    parser.add_argument("--out", default=str(Path.home() / "binbo-shogi-data/logs/bench_qsearch.json"))
    args = parser.parse_args()

    sfens, gikou = sample_run2m(args.run_dir, args.positions, args.seed)
    forward, model, _info = load_model(str(WST))
    value_of = raw_value_fn(forward, model)
    search = build_batched_search(forward, model, 8)

    def wst_cp(sfen_list, batch=128):
        out = []
        for i in range(0, len(sfen_list), batch):
            out.append(np.asarray(value_of(sfens_to_state(sfen_list[i:i + batch]).observation)))
        return value_to_cp(np.concatenate(out))

    # warm the JIT for both shapes used below
    wst_cp(sfens[:128]); wst_cp(sfens[:1], batch=1); score_sfens(search, sfens[:128], 128)

    # existing references: raw WS-T at P and the S=8 search label (the adopted eval's teacher)
    t0 = time.time()
    s_val, _vq, r_val, _mv = score_sfens(search, sfens, 128)
    t_search = time.time() - t0
    raw_p, search_p = value_to_cp(r_val), value_to_cp(s_val)

    board = cshogi.Board()
    # ---- fast: material-guided qsearch, then one batched WS-T call on all Q --------
    t0 = time.time()
    qs = QSearch(material)
    q_fast, plies_fast = [], []
    for sfen in sfens:
        board.set_sfen(sfen)
        _v, pv = qs.search(board, -MATE, MATE, 0)
        q_fast.append(quiet_of(board, pv))
        plies_fast.append(len(pv))
    t_fast_qs = time.time() - t0
    t0 = time.time()
    raw_q = wst_cp(q_fast)
    sign = np.where(np.array(plies_fast) % 2 == 1, -1.0, 1.0)
    fast = sign * raw_q
    t_fast_eval = time.time() - t0
    fast_nodes, fast_trunc = qs.nodes, qs.truncated

    # ---- wst: WS-T stand-pat at every node (batch 1, cached by sfen) --------------
    cache = {}

    def wst_eval(b):
        key = b.sfen()
        if key not in cache:
            cache[key] = float(wst_cp([key], batch=1)[0])
        return cache[key]

    n_w = min(args.wst_positions, len(sfens))
    t0 = time.time()
    qw = QSearch(wst_eval)
    wst_lab, plies_w, same_q = [], [], []
    for i in range(n_w):
        board.set_sfen(sfens[i])
        value, pv = qw.search(board, -MATE, MATE, 0)
        wst_lab.append(np.clip(value, -5000, 5000))
        plies_w.append(len(pv))
        same_q.append(quiet_of(board, pv) == q_fast[i])
    t_wst = time.time() - t0
    wst_lab = np.array(wst_lab)

    def stats(x, ref, n=None):
        n = n or len(x)
        d = np.abs(x[:n] - ref[:n])
        return {"median_abs": float(np.median(d)), "mean_abs": float(d.mean()),
                "frac_over_50cp": float((d > 50).mean()), "corr": float(np.corrcoef(x[:n], ref[:n])[0, 1])}

    pf = np.array(plies_fast)
    res = {
        "n": len(sfens), "n_wst": n_w, "run_dir": args.run_dir, "seed": args.seed,
        "speed_pos_per_s": {
            "s8_search(GPU,batch128)": len(sfens) / t_search,
            "fast_qsearch_cpu_only(1 core)": len(sfens) / t_fast_qs,
            "fast_total(qsearch+WS-T at Q)": len(sfens) / (t_fast_qs + t_fast_eval),
            "wst_guided(batch1)": n_w / t_wst,
        },
        "nodes_per_pos": {"fast": fast_nodes / len(sfens), "wst": qw.nodes / n_w,
                          "wst_unique_evals": len(cache) / n_w},
        "truncated_frac": {"fast": fast_trunc / len(sfens), "wst": qw.truncated / n_w},
        "pv_plies": {"fast_zero_frac": float((pf == 0).mean()), "fast_mean": float(pf.mean()),
                     "wst_zero_frac": float((np.array(plies_w) == 0).mean())},
        "agreement_fast_vs_wst": {"same_Q_frac": float(np.mean(same_q)), **stats(fast[:n_w], wst_lab)},
        "moved_from_raw": {"fast": stats(fast, raw_p), "wst": stats(wst_lab, raw_p, n_w),
                           "s8_search": stats(search_p, raw_p)},
        "vs_s8_search_label": {"fast": stats(fast, search_p), "wst": stats(wst_lab, search_p, n_w),
                               "raw": stats(raw_p, search_p)},
        "vs_gikou_d8": {"fast": stats(fast, np.clip(gikou, -5000, 5000)),
                        "wst": stats(wst_lab, np.clip(gikou, -5000, 5000), n_w),
                        "s8_search": stats(search_p, np.clip(gikou, -5000, 5000)),
                        "raw": stats(raw_p, np.clip(gikou, -5000, 5000))},
    }
    res["hours_for_32M"] = {k: 31_950_000 / v / 3600 for k, v in res["speed_pos_per_s"].items()}
    # a batched WS-T-guided qsearch would still need every unique node evaluated once:
    evals_per_s = len(sfens) / t_fast_eval                      # batched (128) WS-T throughput
    res["hours_for_32M"]["wst_guided_if_perfectly_batched(estimate)"] = (
        31_950_000 * res["nodes_per_pos"]["wst_unique_evals"] / evals_per_s / 3600)
    res["wst_batched_evals_per_s"] = evals_per_s
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
