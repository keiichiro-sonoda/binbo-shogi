#!/usr/bin/env python3
"""Task 133: does the adopted eval underrate king danger? (diagnostic only, no training)

Motivation (2026-10-16/17): in a floodgate loss (game 4 vs Gokaku) binbo rated its attack at
+500..+970 while the opponent saw the position as roughly even, and was then mated; and binbo
keeps its king near the centre far more often than strong engines (60 plies: >=3 files from the
5th file 13% vs 43%, docs/denryusen7_prep.md). Hypothesis: the eval underrates the danger to the
king of the side to move (or overrates attacks on the opponent's king).

Method: take middlegame positions (plies 30..120 every 6) from (A) 176 floodgate games between
engines rated >= 3800 and (B) binbo's own 40 self-play games (book1); evaluate each position with
binbo (YaneuraOu + adopted eval), tanuki- Hao and AobaNNUE at the same fixed node count, single
thread (reproducible). Per position compute king-safety features for both kings. Regress
  binbo_cp = a1 * tanuki_cp + a2 * aoba_cp + sum_k b_k * feature_k + c
(the two yardsticks are separate covariates: AobaNNUE's cp scale is about twice tanuki-'s)
on positions with |tanuki| <= 1500 and no mate scores. A positive coefficient on "danger to the
side to move's king" means binbo is too optimistic exactly when its own king is in danger.
Bootstrap CIs over positions (games are clustered: resample games).

  t133_king_safety_probe.py collect --out positions.tsv
  t133_king_safety_probe.py evaluate positions.tsv --nodes 200000 --jobs 4 --out evals.tsv
  t133_king_safety_probe.py analyze evals.tsv [--target binbo]
  ~/task074_frameworks/venv_pgx/bin/python t133_king_safety_probe.py label evals.tsv --out labels.tsv
      adds the teacher's labels for the same positions: WS-T raw value, WS-T + Gumbel MCTS S=8 (the
      labeller of the adopted eval's data, Tasks 104/128/129) and S=32; then `analyze labels.tsv
      --target wst_s8` asks whether the teacher itself shows the same optimism (cause = teacher) or not
      (cause = learning).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import cshogi
import numpy as np

DATA = os.path.expanduser(os.environ.get("DATA_ROOT", "~/binbo-shogi-data"))
YANE = f"{DATA}/engines/yaneuraou/source/YaneuraOu-by-gcc"
ENGINES = {
    "binbo": (YANE, f"{DATA}/eval_dirs/task129_student_64m_qsrel_e12"),
    "binbo134": (YANE, f"{DATA}/eval_dirs/task134_student_sharp32"),  # Task 134 follow-up (--engines)
    "tanuki": (YANE, f"{DATA}/engines/yaneuraou/eval"),
    "aoba": (f"{DATA}/engines/aobannue/AobaNNUE/source/YaneuraOu-by-gcc", f"{DATA}/engines/aobannue/AobaNNUE/eval"),
}
SETS = {
    "floodgate_top": f"{DATA}/books/probe_games/floodgate/*.csa",
    "binbo_selfplay": f"{DATA}/denryusen/xub/book1/srv/**/*.csa",
}
BASE = ["binbo", "tanuki", "aoba"]  # engines evaluated by default (Task 133)
FEATURES = ["stm_def1", "stm_def2", "stm_att2", "stm_opp_hand", "stm_kfile", "stm_krank",
            "opp_def1", "opp_def2", "opp_att2", "opp_my_hand", "opp_kfile", "opp_krank"]


def king_features(b: cshogi.Board, color: int) -> dict:
    """color = 0 (black) / 1 (white). Squares are file*9+rank with file 0 = 1筋, rank 0 = 一段."""
    k = b.king_square(color)
    kf, kr = divmod(k, 9)
    def1 = def2 = att2 = 0
    for f in range(max(0, kf - 2), min(9, kf + 3)):
        for r in range(max(0, kr - 2), min(9, kr + 3)):
            if (f, r) == (kf, kr):
                continue
            pc = b.piece(f * 9 + r)
            if pc == cshogi.NONE:
                continue
            own = (pc < 16) == (color == 0)  # cshogi: black pieces 1..14, white 17..30
            near = max(abs(f - kf), abs(r - kr)) == 1
            if own:
                def2 += 1
                def1 += near
            else:
                att2 += 1
    hand = b.pieces_in_hand[1 - color]  # opponent's hand (7 counts)
    rank_from_home = kr if color == 1 else 8 - kr  # 0 = own back rank
    return {"def1": def1, "def2": def2, "att2": att2, "opp_hand": int(sum(hand)),
            "kfile": abs(kf - 4), "krank": rank_from_home}


def cmd_label(a):
    """GPU stage (venv_pgx, JAX): label the positions with the teacher."""
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from m2_usi import load_model
    from t099_search_label import build_batched_search, score_sfens, sign_oracle, value_to_cp
    wst = os.path.expanduser("~/task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt")
    lines = open(a.evals, encoding="utf-8").read().splitlines()
    head, rows = lines[0].split("\t"), [l.split("\t") for l in lines[1:]]
    sfens = [r[head.index("sfen")] for r in rows]
    forward, model, _ = load_model(wst)
    cols = {}
    for sims in a.sims:
        sign_oracle(forward, model, sims)
        search = build_batched_search(forward, model, sims)
        values, _q, raw, _mv = score_sfens(search, sfens, a.batch)
        cols[f"wst_s{sims}"] = value_to_cp(values)
        cols["wst_raw"] = value_to_cp(raw)
        print(f"labelled {len(sfens)} positions with S={sims}", flush=True)
    names = list(cols)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\t".join(head + [x for n in names for x in (f"{n}_kind", f"{n}_v")]) + "\n")
        for i, r in enumerate(rows):
            f.write("\t".join(r + [x for n in names for x in ("cp", str(int(round(cols[n][i]))))]) + "\n")
    print(f"-> {a.out}")


def cmd_collect(a):
    import cshogi.CSA
    rows = []
    for set_name, pattern in SETS.items():
        for p in sorted(glob.glob(pattern, recursive=True)):
            k = cshogi.CSA.Parser.parse_file(p)[0]
            b = cshogi.Board(k.sfen)
            for ply, mv in enumerate(k.moves):
                if 30 <= ply <= 120 and ply % 6 == 0 and not b.is_check():
                    stm = b.turn
                    fs = king_features(b, stm)
                    fo = king_features(b, 1 - stm)
                    rows.append([set_name, os.path.basename(p), ply, b.sfen()] +
                                [fs["def1"], fs["def2"], fs["att2"], fs["opp_hand"], fs["kfile"], fs["krank"],
                                 fo["def1"], fo["def2"], fo["att2"], fo["opp_hand"], fo["kfile"], fo["krank"]])
                b.push(mv)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\t".join(["set", "game", "ply", "sfen"] + FEATURES) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} positions -> {a.out}")


def eval_chunk(args):
    name, sfens, nodes = args
    binary, evaldir = ENGINES[name]
    p = subprocess.Popen([binary], cwd=os.path.dirname(binary), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def wait(tok):
        for line in p.stdout:
            if line.startswith(tok):
                return line

    for o, v in (("EvalDir", evaldir), ("Threads", "1"), ("USI_Hash", "64"), ("USI_OwnBook", "false"),
                 ("BookFile", "no_book"), ("EnteringKingRule", "CSARule27")):
        send(f"setoption name {o} value {v}")
    send("isready")
    wait("readyok")
    out = []
    for s in sfens:
        send("usinewgame")
        send(f"position sfen {s}")
        send(f"go nodes {nodes}")
        score = None
        for line in p.stdout:
            if " score " in line:
                t = line.split()
                i = t.index("score")
                score = (t[i + 1], int(t[i + 2]))
            if line.startswith("bestmove"):
                break
        out.append(score)
    send("quit")
    p.wait(timeout=30)
    return name, out


def cmd_evaluate(a):
    lines = open(a.positions, encoding="utf-8").read().splitlines()
    head, rows = lines[0].split("\t"), [l.split("\t") for l in lines[1:]]
    sfens = [r[head.index("sfen")] for r in rows]
    names = a.engines or BASE
    jobs = []
    step = (len(sfens) + a.chunks - 1) // a.chunks
    for name in names:
        for i in range(0, len(sfens), step):
            jobs.append((name, sfens[i:i + step], a.nodes))
    res = {n: [] for n in names}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, out in ex.map(eval_chunk, jobs):  # map keeps job order = chunk order per engine
            res[name] += out
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\t".join(head + [f"{n}_{k}" for n in names for k in ("kind", "v")]) + "\n")
        for i, r in enumerate(rows):
            extra = []
            for n in names:
                s = res[n][i]
                extra += [s[0], str(s[1])] if s else ["none", "0"]
            f.write("\t".join(r + extra) + "\n")
    print(f"{len(rows)} positions x {len(names)} engines @ {a.nodes} nodes -> {a.out}")


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def cmd_analyze(a):
    lines = open(a.evals, encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    tgt, minus = a.target, a.minus
    tv = [tgt] + ([minus] if minus else [])
    ok = [r for r in rows if all(r[f"{n}_kind"] == "cp" for n in dict.fromkeys(BASE + tv))]
    ok = [r for r in ok if abs(int(r["tanuki_v"])) <= a.ref_max and all(abs(int(r[f"{n}_v"])) <= 3000 for n in tv)]
    val = (lambda r: int(r[f"{tgt}_v"]) - int(r[f"{minus}_v"])) if minus else (lambda r: int(r[f"{tgt}_v"]))
    print(f"target = {tgt}" + (f" - {minus} (coef = change of the {tgt} coefficient vs {minus}, same positions)" if minus else ""))
    print(f"positions: {len(rows)} total, {len(ok)} used (all cp, |tanuki| <= {a.ref_max})")
    for set_name in ["all"] + list(SETS):
        sub = ok if set_name == "all" else [r for r in ok if r["set"] == set_name]
        if len(sub) < 50:
            continue
        y = np.array([val(r) for r in sub], float)
        X = np.column_stack([[float(r["tanuki_v"]) for r in sub], [float(r["aoba_v"]) for r in sub]]
                            + [[float(r[f]) for r in sub] for f in FEATURES] + [np.ones(len(sub))])
        beta = ols(X, y)
        games = sorted({r["game"] for r in sub})
        by_game = {g: [i for i, r in enumerate(sub) if r["game"] == g] for g in games}
        rng = random.Random(a.seed)
        boots = []
        for _ in range(a.boot):
            idx = [i for g in (rng.choice(games) for _ in games) for i in by_game[g]]
            boots.append(ols(X[idx], y[idx]))
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        names = ["tanuki", "aoba"] + FEATURES + ["const"]
        corr_t = np.corrcoef(y, [int(r["tanuki_v"]) for r in sub])[0, 1]
        corr_a = np.corrcoef(y, [int(r["aoba_v"]) for r in sub])[0, 1]
        corr_ta = np.corrcoef([int(r["tanuki_v"]) for r in sub], [int(r["aoba_v"]) for r in sub])[0, 1]
        print(f"\n== {set_name}: {len(sub)} positions, {len(games)} games. corr {tgt}~tanuki {corr_t:.3f}, "
              f"{tgt}~aoba {corr_a:.3f}, tanuki~aoba {corr_ta:.3f}")
        print("coef (cp per unit)  [95% CI, game bootstrap]   * = CI excludes 0")
        for n, b0, l0, h0 in zip(names, beta, lo, hi):
            star = "*" if l0 > 0 or h0 < 0 else " "
            print(f"  {n:13s} {b0:+9.2f}  [{l0:+8.2f}, {h0:+8.2f}] {star}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("positions")
    e.add_argument("--nodes", type=int, default=200000)
    e.add_argument("--jobs", type=int, default=4)
    e.add_argument("--chunks", type=int, default=8)
    e.add_argument("--out", required=True)
    e.add_argument("--engines", nargs="+", choices=list(ENGINES), default=None, help=f"default: {' '.join(BASE)}")
    n = sub.add_parser("analyze")
    n.add_argument("evals")
    n.add_argument("--ref-max", type=int, default=1500)
    n.add_argument("--boot", type=int, default=1000)
    n.add_argument("--seed", type=int, default=20271333)
    n.add_argument("--target", default="binbo", help="column to explain: binbo / binbo134 / wst_s8 / wst_s32 / wst_raw")
    n.add_argument("--minus", default=None, help="explain target - this column instead (e.g. --target binbo134 --minus binbo)")
    lb = sub.add_parser("label")
    lb.add_argument("evals")
    lb.add_argument("--sims", type=int, nargs="+", default=[8, 32])
    lb.add_argument("--batch", type=int, default=128)
    lb.add_argument("--out", required=True)
    a = ap.parse_args()
    {"collect": cmd_collect, "evaluate": cmd_evaluate, "analyze": cmd_analyze, "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    main()
