#!/usr/bin/env python3
"""Task 066 — self-reinforcement label generation (1-cycle rehearsal).

Generates champion-labeled self-play training data: the current champion eval, loaded into
YaneuraOu, plays itself at a FIXED depth; every visited position is labeled with that search's
cp (side-to-move) and, at game end, the game's WDL outcome (side-to-move +1/0/-1). The output is
a prepare_data-compatible cache (train_indices/evals/wdl.npy in the canonical-shared layout), so
`train_nnue --objective wdl --wdl-lambda` consumes it unchanged.

`--mix-replay <shared_32m> --replay-n N` concatenates N rows sampled from the 32M champion cache
(anti-forgetting warm-start mix), and copies that cache's pinned val split so generation-vs-
generation val stays comparable. Reuses `trainer.kp.canonical_shared_row` (the exact KP encoding
prepare_data uses) and the label_cost-style watchdog'd USI driver (a silent deep search can't hang
us). Champion is read-only (loaded via EvalDir).

Example:
  env PYTHONPATH=$REPO $VENV_PY trainer/selfplay_label.py \
    --yane $YANE --eval-dir $DATA_ROOT/eval_dirs/task064_32m_lr1e-5 \
    --positions 120000 --depth 12 --random-plies 8 --seed 20260629 \
    --mix-replay $DATA_ROOT/caches/shared_32m --replay-n -1 \
    --out $DATA_ROOT/caches/gen1_mix
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time

import cshogi
import numpy as np

try:
    from . import kp
except ImportError:
    import kp

INT16_MAX = 32767


class Yane:
    """YaneuraOu(champion EvalDir, Threads=1) USI driver: fixed-depth search -> (bestmove, cp)."""

    def __init__(self, path: str, eval_dir: str, usi_hash: int = 256, multipv: int = 1):
        self.path = path
        self.eval_dir = eval_dir
        self.usi_hash = usi_hash
        self.multipv = max(1, int(multipv))
        self._start()

    def _start(self):
        import subprocess
        self.p = subprocess.Popen(
            [self.path], cwd=os.path.dirname(self.path) or ".",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("usi"); self._wait("usiok")
        for name, value in [
            ("EvalDir", self.eval_dir), ("USI_OwnBook", "false"), ("Threads", "1"),
            ("USI_Hash", str(self.usi_hash)), ("MinimumThinkingTime", "0"),
            ("NetworkDelay", "0"), ("NetworkDelay2", "0"), ("MultiPV", str(self.multipv)),
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

    def go_depth(self, sfen: str, depth: int, cap_s: float = 15.0):
        """Cold-TT fixed-depth search, watchdog-bounded. Returns (bestmove_usi, cp_or_None, is_mate)."""
        self._send("usinewgame")
        self._send(f"position sfen {sfen}")
        cp = None
        is_mate = False
        done = threading.Event()

        def watchdog():
            if not done.wait(cap_s):
                try:
                    self._send("stop")
                except Exception:
                    pass
                if not done.wait(10.0):
                    try:
                        self.p.kill()
                    except Exception:
                        pass

        self._send(f"go depth {depth}")
        threading.Thread(target=watchdog, daemon=True).start()
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
                        cp = int(t[j + 2]); is_mate = False
                    elif t[j + 1] == "mate":
                        m = int(t[j + 2])
                        cp = INT16_MAX if m >= 0 else -INT16_MAX; is_mate = True
                elif t[0] == "bestmove":
                    return (t[1] if len(t) > 1 else "resign"), cp, is_mate
        finally:
            done.set()

    def go_depth_multi(self, sfen: str, depth: int, cap_s: float = 15.0):
        """Cold-TT fixed-depth MultiPV search (Task 069). Returns (cands, best, is_mate) where
        cands = [(move_usi, cp, is_mate), ...] sorted by multipv index (cands[0] = best move).
        cp is side-to-move POV; mate -> +/-INT16_MAX. Used for temperature move-sampling; the
        position label is cands[0]'s cp (= champion's deterministic eval, unchanged)."""
        self._send("usinewgame")
        self._send(f"position sfen {sfen}")
        pvs: dict = {}            # multipv index -> (cp, is_mate, first_move_usi)
        best_simple = None        # fallback from the 'bestmove' line
        done = threading.Event()

        def watchdog():
            if not done.wait(cap_s):
                try:
                    self._send("stop")
                except Exception:
                    pass
                if not done.wait(10.0):
                    try:
                        self.p.kill()
                    except Exception:
                        pass

        self._send(f"go depth {depth}")
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            while True:
                line = self.p.stdout.readline()
                if line == "":
                    raise EOFError("engine EOF during go")
                t = line.split()
                if not t:
                    continue
                if t[0] == "info" and "score" in t and "pv" in t:
                    if "lowerbound" in t or "upperbound" in t:
                        continue                      # incomplete (aspiration) score
                    k = int(t[t.index("multipv") + 1]) if "multipv" in t else 1
                    j = t.index("score")
                    if t[j + 1] == "cp":
                        cp_k, mate_k = int(t[j + 2]), False
                    elif t[j + 1] == "mate":
                        m = int(t[j + 2]); cp_k, mate_k = (INT16_MAX if m >= 0 else -INT16_MAX), True
                    else:
                        continue
                    pv_first = t[t.index("pv") + 1]
                    pvs[k] = (cp_k, mate_k, pv_first)
                elif t[0] == "bestmove":
                    best_simple = t[1] if len(t) > 1 else "resign"
                    break
        finally:
            done.set()

        if not pvs:                                   # resign / win / (none) / no info
            return [], best_simple, False
        cands = [(mv, cp_k, mate_k) for k, (cp_k, mate_k, mv) in sorted(pvs.items())]
        best_mv, best_cp, best_mate = cands[0]
        return cands, best_mv, best_mate

    def close(self):
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.p.kill()
        except Exception:
            pass


def clamp16(cp: int) -> int:
    return max(-INT16_MAX, min(INT16_MAX, int(cp)))


def sample_move_index(cps, temperature: float, rng: np.random.Generator) -> int:
    """Softmax over winrate(cp/600) with temperature T (Task 069). cp is side-to-move POV,
    so higher cp = better move -> sampled with higher prob. Returns an index into cps.
    Done in winrate space (not raw cp) because shogi cp blows up in the endgame/material."""
    w = 1.0 / (1.0 + np.exp(-np.asarray(cps, dtype=np.float64) / 600.0))
    logits = w / max(1e-6, temperature)
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    return int(rng.choice(len(cps), p=p))


def play_one_game(eng: Yane, depth: int, random_plies: int, max_plies: int,
                  rng: np.random.Generator, cap_s: float,
                  temperature: float = 0.0, sample_until_ply: int = 0):
    """One champion self-play game. Returns (rows, winner) where rows = list of
    (indices_tuple, cp_stm_int16, stm_color) and winner in {0,1,None(draw)}.
    With temperature>0 (Task 069), the move PLAYED in the opening-midgame (ply <= sample_until_ply)
    is temperature-sampled among the top-MultiPV candidates; the position's cp LABEL is always the
    best (multipv-1) cp = champion's deterministic eval, unchanged."""
    b = cshogi.Board()
    for _ in range(random_plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(int(moves[rng.integers(len(moves))]))

    rows: list[tuple] = []
    winner = None  # 0=black wins, 1=white wins, None=draw
    plies = b.move_number
    while True:
        rep = b.is_draw()
        if rep == cshogi.REPETITION_DRAW:
            winner = None; break
        if rep == cshogi.REPETITION_WIN:
            winner = b.turn; break               # side to move wins the repetition
        if rep == cshogi.REPETITION_LOSE:
            winner = 1 - b.turn; break
        legal = list(b.legal_moves)
        if not legal:
            winner = 1 - b.turn; break            # checkmated: side to move loses
        if plies >= max_plies:
            winner = None; break                  # adjudicate long games as draw
        sfen = b.sfen()
        try:
            cands, best, _is_mate = eng.go_depth_multi(sfen, depth, cap_s=cap_s)
        except EOFError:
            eng.close(); eng._start()
            break                                  # abandon this game on engine death
        if not cands or best in ("resign", "(none)", "win"):
            if best == "resign":
                winner = 1 - b.turn
            elif best == "win":
                winner = b.turn
            break
        legal_usi = {cshogi.move_to_usi(m): m for m in legal}
        best_cp = cands[0][1]                       # LABEL = champion's eval (multipv-1), always
        # Move to PLAY: temperature-sample among legal candidates in the opening-midgame; else best.
        play_usi = best
        if temperature > 0.0 and random_plies < plies <= sample_until_ply:
            legal_cands = [(mv_usi, cp_k) for (mv_usi, cp_k, _m) in cands if mv_usi in legal_usi]
            if len(legal_cands) > 1:
                i = sample_move_index([c for _, c in legal_cands], temperature, rng)
                play_usi = legal_cands[i][0]
        mv = legal_usi.get(play_usi)
        if mv is None:
            break                                  # illegal/unparsable -> abandon
        rows.append((kp.canonical_shared_row(b), clamp16(best_cp), b.turn))
        b.push(int(mv))
        plies += 1
    return rows, winner


def generate(eng: Yane, target_positions: int, depth: int, random_plies: int,
             max_plies: int, seed: int, cap_s: float,
             temperature: float = 0.0, sample_until_ply: int = 0):
    rng = np.random.default_rng(seed)
    idx_list: list = []
    eval_list: list = []
    wdl_list: list = []
    games = 0
    t0 = time.time()
    while len(idx_list) < target_positions:
        rows, winner = play_one_game(eng, depth, random_plies, max_plies, rng, cap_s,
                                     temperature=temperature, sample_until_ply=sample_until_ply)
        games += 1
        for indices, cp, stm in rows:
            idx_list.append(indices)
            eval_list.append(cp)
            wdl_list.append(0 if winner is None else (1 if stm == winner else -1))
        if games % 25 == 0:
            el = time.time() - t0
            print(f"  games={games} positions={len(idx_list)}/{target_positions} "
                  f"({el:.0f}s, {len(idx_list)/max(1,el):.0f} pos/s)", flush=True)
    indices = np.asarray(idx_list, dtype=np.int32)
    evals = np.asarray(eval_list, dtype=np.int16)
    wdl = np.asarray(wdl_list, dtype=np.int8)
    print(f"self-play done: games={games} positions={len(indices)} "
          f"wdl(win/draw/loss)={np.mean(wdl>0):.2f}/{np.mean(wdl==0):.2f}/{np.mean(wdl<0):.2f} "
          f"eval[min/med/max]={evals.min()}/{int(np.median(evals))}/{evals.max()}", flush=True)
    return indices, evals, wdl


def _worker_generate(task):
    """Multiprocessing worker (Task 068): one self-play shard with its own engine + seed.
    Defined at module level so it is picklable for multiprocessing.Pool. The Yane subprocess
    (and its watchdog threads) are created AFTER fork, inside this child = no shared handles."""
    (yane_path, eval_dir, usi_hash, multipv, n_target, depth, random_plies, max_plies,
     seed, cap_s, temperature, sample_until_ply) = task
    eng = Yane(yane_path, eval_dir, usi_hash=usi_hash, multipv=multipv)
    idx, ev, wdl = generate(eng, n_target, depth, random_plies, max_plies, seed, cap_s,
                            temperature=temperature, sample_until_ply=sample_until_ply)
    eng.close()
    return idx, ev, wdl


def generate_parallel(args):
    """Fan generation across N worker processes (each its own Yane + distinct seed), then
    concatenate and trim to the target. Same per-position labelling as the serial path."""
    import math
    import multiprocessing as mp
    n_workers = args.workers
    per = math.ceil(args.positions / n_workers)
    tasks = [(args.yane, args.eval_dir, args.usi_hash, args.top_n, per, args.depth, args.random_plies,
              args.max_plies, args.seed + 100003 * k, args.cap_s, args.temperature, args.sample_until_ply)
             for k in range(n_workers)]
    print(f"self-play(parallel): workers={n_workers} per_worker={per} target={args.positions} "
          f"depth={args.depth} seed={args.seed} temperature={args.temperature} "
          f"top_n={args.top_n} sample_until_ply={args.sample_until_ply}", flush=True)
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_generate, tasks)
    new_idx = np.concatenate([r[0] for r in results], axis=0)[:args.positions]
    new_eval = np.concatenate([r[1] for r in results], axis=0)[:args.positions]
    new_wdl = np.concatenate([r[2] for r in results], axis=0)[:args.positions]
    print(f"self-play(parallel) done: positions={new_idx.shape[0]} "
          f"wdl(win/draw/loss)={np.mean(new_wdl>0):.2f}/{np.mean(new_wdl==0):.2f}/{np.mean(new_wdl<0):.2f} "
          f"eval[min/med/max]={new_eval.min()}/{int(np.median(new_eval))}/{new_eval.max()}", flush=True)
    return new_idx, new_eval, new_wdl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yane", default=os.environ.get("YANE"))
    ap.add_argument("--eval-dir", default=None, help="champion EvalDir (read-only; not needed with --reuse-new)")
    ap.add_argument("--positions", type=int, default=120000, help="target NEW self-play positions")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--random-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--cap-s", type=float, default=15.0, help="per-search wall cap (watchdog)")
    ap.add_argument("--usi-hash", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="policy-diversity temperature (Task 069). >0 = temperature-sample the PLAYED "
                         "move among top-MultiPV candidates in winrate(cp/600) space during the "
                         "opening-midgame; the position cp LABEL stays the deterministic best. 0 = "
                         "deterministic best move (default, unchanged).")
    ap.add_argument("--top-n", type=int, default=1,
                    help="MultiPV candidates considered for temperature sampling (Task 069). "
                         "1 = single-PV (default, == old behavior). Use ~4 with --temperature>0.")
    ap.add_argument("--sample-until-ply", type=int, default=40,
                    help="only temperature-sample while game ply <= this (endgame stays deterministic).")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel self-play engines (Task 068). N>1 fans generation across N "
                         "processes (each its own Yane + distinct seed) for ~Nx throughput; same "
                         "per-position labelling. 1 = serial (default, unchanged).")
    ap.add_argument("--mix-replay", default=None, help="shared_32m cache dir to sample replay rows from")
    ap.add_argument("--replay-n", type=int, default=-1, help="replay rows (-1 = match new count = 1:1)")
    ap.add_argument("--reuse-new", default=None,
                    help="reuse NEW self-play rows from an existing cache (its first n_new rows) instead "
                         "of generating them; skips the engine entirely. For replay-ratio sweeps (Task 067) "
                         "where new self-play is held fixed and only --replay-n varies.")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="when NO --mix-replay (pure self-play, Task 071), hold out this fraction of the "
                         "generated NEW rows (tail) as the val split, since no pinned 技巧2 val exists. "
                         "val = self-consistency monitor only, NOT an adoption criterion (adoption = "
                         "gen-vs-gen + gikou). 0 = no val (default; with --mix-replay, val is copied "
                         "from the replay cache instead and this flag is ignored).")
    ap.add_argument("--out", required=True, help="output cache dir")
    args = ap.parse_args()

    src_meta: dict = {}
    if args.reuse_new:
        # Replay-ratio sweep (Task 067): reuse a prior cache's NEW rows verbatim, no engine, no GPU.
        # The first n_new rows of a selfplay_label cache are the NEW self-play data (concat order [new, replay]).
        with open(os.path.join(args.reuse_new, "meta.json"), encoding="utf-8") as f:
            src_meta = json.load(f)
        n_new = int(src_meta["n_new"])
        ni = np.load(os.path.join(args.reuse_new, "train_indices.npy"), mmap_mode="r")
        ne = np.load(os.path.join(args.reuse_new, "train_evals.npy"), mmap_mode="r")
        nw = np.load(os.path.join(args.reuse_new, "train_wdl.npy"), mmap_mode="r")
        new_idx = np.asarray(ni[:n_new], dtype=np.int32)
        new_eval = np.asarray(ne[:n_new], dtype=np.int16)
        new_wdl = np.asarray(nw[:n_new], dtype=np.int8)
        print(f"reuse-new: loaded {new_idx.shape[0]} NEW rows from {args.reuse_new} "
              f"(champion={src_meta.get('champion_eval_dir')} depth={src_meta.get('depth')} "
              f"seed={src_meta.get('seed')})", flush=True)
    else:
        if not args.yane or not os.path.exists(args.yane):
            raise SystemExit(f"YaneuraOu not found: {args.yane}")
        if not args.eval_dir or not os.path.exists(os.path.join(args.eval_dir, "nn.bin")):
            raise SystemExit(f"no nn.bin in eval-dir: {args.eval_dir}")
        if args.workers and args.workers > 1:
            new_idx, new_eval, new_wdl = generate_parallel(args)
        else:
            print(f"self-play: champion={args.eval_dir} depth={args.depth} random_plies={args.random_plies} "
                  f"target={args.positions} seed={args.seed} temperature={args.temperature} "
                  f"top_n={args.top_n} sample_until_ply={args.sample_until_ply}", flush=True)
            eng = Yane(args.yane, args.eval_dir, usi_hash=args.usi_hash, multipv=args.top_n)
            new_idx, new_eval, new_wdl = generate(
                eng, args.positions, args.depth, args.random_plies, args.max_plies, args.seed, args.cap_s,
                temperature=args.temperature, sample_until_ply=args.sample_until_ply)
            eng.close()

    train_idx, train_eval, train_wdl = new_idx, new_eval, new_wdl
    val_idx = val_eval = val_wdl = None
    if args.mix_replay:
        rep_n = new_idx.shape[0] if args.replay_n < 0 else args.replay_n
        ri = np.load(os.path.join(args.mix_replay, "train_indices.npy"), mmap_mode="r")
        re_ = np.load(os.path.join(args.mix_replay, "train_evals.npy"), mmap_mode="r")
        rw = np.load(os.path.join(args.mix_replay, "train_wdl.npy"), mmap_mode="r")
        rep_n = min(rep_n, ri.shape[0])
        sel = np.sort(np.random.default_rng(args.seed + 1).choice(ri.shape[0], size=rep_n, replace=False))
        rep_idx = np.asarray(ri[sel], dtype=np.int32)
        rep_eval = np.asarray(re_[sel], dtype=np.int16)
        rep_wdl = np.asarray(rw[sel], dtype=np.int8)
        train_idx = np.concatenate([new_idx, rep_idx], axis=0)
        train_eval = np.concatenate([new_eval, rep_eval], axis=0)
        train_wdl = np.concatenate([new_wdl, rep_wdl], axis=0)
        print(f"mix: new={new_idx.shape[0]} + replay={rep_n} -> train={train_idx.shape[0]} "
              f"(replay from {args.mix_replay})", flush=True)
    elif args.val_frac and args.val_frac > 0.0:
        # Task 071 (pure self-play, no replay): hold out a val split from the generated NEW rows
        # themselves, since no pinned 技巧2 val exists. val = self-consistency monitor only.
        n_total = new_idx.shape[0]
        n_v = min(max(1, int(round(n_total * args.val_frac))), n_total - 1)
        cut = n_total - n_v
        train_idx, train_eval, train_wdl = new_idx[:cut], new_eval[:cut], new_wdl[:cut]
        val_idx, val_eval, val_wdl = new_idx[cut:], new_eval[cut:], new_wdl[cut:]
        print(f"val-frac: holdout {n_v} val (tail) from {n_total} new self-play rows -> "
              f"train={train_idx.shape[0]} val={n_v}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "train_indices.npy"), train_idx)
    np.save(os.path.join(args.out, "train_evals.npy"), train_eval)
    np.save(os.path.join(args.out, "train_wdl.npy"), train_wdl)

    n_val = 0
    if args.mix_replay:  # copy pinned val (comparable across generations)
        for name in ("val_indices.npy", "val_evals.npy", "val_wdl.npy"):
            src = os.path.join(args.mix_replay, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, name))
        vp = os.path.join(args.out, "val_indices.npy")
        n_val = int(np.load(vp, mmap_mode="r").shape[0]) if os.path.exists(vp) else 0
    elif val_idx is not None:  # Task 071: self-play holdout val
        np.save(os.path.join(args.out, "val_indices.npy"), val_idx)
        np.save(os.path.join(args.out, "val_evals.npy"), val_eval)
        np.save(os.path.join(args.out, "val_wdl.npy"), val_wdl)
        n_val = int(val_idx.shape[0])

    # Provenance of the NEW rows: inherit from the source cache when reusing, else this run's args.
    eff = (lambda k, v: src_meta.get(k, v)) if args.reuse_new else (lambda k, v: v)
    meta = {
        "source": "selfplay_label (Task 066/067/068)",
        "champion_eval_dir": eff("champion_eval_dir", args.eval_dir),
        "depth": eff("depth", args.depth),
        "random_plies": eff("random_plies", args.random_plies),
        "seed": eff("seed", args.seed),
        "workers": (0 if args.reuse_new else args.workers),
        "temperature": (0.0 if args.reuse_new else args.temperature),
        "top_n": (0 if args.reuse_new else args.top_n),
        "sample_until_ply": (0 if args.reuse_new else args.sample_until_ply),
        "reused_from": args.reuse_new,
        "n_new": int(new_idx.shape[0]),
        "n_train": int(train_idx.shape[0]), "n_val": n_val,
        "features_per_position": 77, "canonical_shared": True,
        "total_features": int(kp.CANONICAL_SHARED_TOTAL),
        "mix_replay": args.mix_replay,
        "replay_n": (int(train_idx.shape[0] - new_idx.shape[0]) if args.mix_replay else 0),
        "val_frac": args.val_frac,
        "val_shards": ("copied_from_mix_replay" if args.mix_replay
                       else ("selfplay_holdout" if val_idx is not None else "none")),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote cache_dir={args.out} n_train={meta['n_train']} n_val={meta['n_val']}", flush=True)
    print("selfplay_label=ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
