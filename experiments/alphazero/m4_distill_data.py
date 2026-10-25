# Task 080 M4 step 0: distillation dataset with POLICY targets.
# Same sampling/format as m2_distill_data (reused directly), plus per position the
# teacher bestmove as a Pgx action id:
#   bestMove16 --(board.move_from_move16)--> move --(move_to_usi)--> USI
#            --(m2_moves.usi_to_action, side-to-move color)--> action in [0, 2187)
# Positions whose bestmove fails to expand/convert or is illegal (per cshogi) are
# SKIPPED and counted (expected ~0).
# Output: {train,val}_obs.npy (packbits memmap), _v.npy (float32), _a.npy (int16).

import argparse
import json
import os
import time

import numpy as np
import cshogi
import jax
import jax.numpy as jnp

from pgx._src.games.shogi import _observe
import m2_moves
from m2_distill_data import (
    RUNS, OBS_BYTES, parse_sfen_np, batch_state, shard_list, sample_split,
)


def decode_with_best(sel, skip_counter):
    board = cshogi.Board()
    for path, idx in sel:
        ent = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
        for i in idx:
            e = ent[i]
            board.set_hcp(np.asarray(e["hcp"]))
            try:
                m = board.move_from_move16(int(e["bestMove16"]))
                if m not in set(board.legal_moves):
                    skip_counter["illegal"] += 1
                    continue
                usi = cshogi.move_to_usi(m)
            except Exception:
                skip_counter["expand_fail"] += 1
                continue
            yield board.sfen(), int(e["eval"]), usi


def build(sel, out_prefix, obs_fn, skip_counter, chunk=4096, log_every=200_000):
    n_max = sum(len(idx) for _, idx in sel)
    obs_mm = np.lib.format.open_memmap(
        out_prefix + "_obs.npy", mode="w+", dtype=np.uint8, shape=(n_max, OBS_BYTES)
    )
    v = np.zeros(n_max, dtype=np.float32)
    a = np.zeros(n_max, dtype=np.int16)
    colors, boards, hands, cps, acts = [], [], [], [], []
    done, t0 = 0, time.time()

    def flush():
        nonlocal done
        if not colors:
            return
        obs = np.asarray(obs_fn(batch_state(colors, boards, hands)))
        obs_mm[done : done + len(colors)] = np.packbits(obs.reshape(len(colors), -1), axis=1)
        v[done : done + len(colors)] = np.tanh(np.asarray(cps, dtype=np.float64) / 1200.0)
        a[done : done + len(colors)] = np.asarray(acts, dtype=np.int16)
        done += len(colors)
        for b in (colors, boards, hands, cps, acts):
            b.clear()
        if done % log_every < chunk:
            rate = done / (time.time() - t0)
            print(f"  {out_prefix}: {done} ({rate:.0f} pos/s)", flush=True)

    for sfen, cp, usi in decode_with_best(sel, skip_counter):
        c, pb, hd = parse_sfen_np(sfen)
        try:
            act = m2_moves.usi_to_action(usi, c)
        except ValueError:
            skip_counter["map_fail"] += 1
            continue
        colors.append(c)
        boards.append(pb)
        hands.append(hd)
        cps.append(cp)
        acts.append(act)
        if len(colors) >= chunk:
            flush()
    flush()
    obs_mm.flush()
    # truncate arrays to the actually-written count (skips shrink n)
    np.save(out_prefix + "_v.npy", v[:done])
    np.save(out_prefix + "_a.npy", a[:done])
    if done < n_max:
        final = np.lib.format.open_memmap(
            out_prefix + "_obs_final.npy", mode="w+", dtype=np.uint8, shape=(done, OBS_BYTES)
        )
        final[:] = obs_mm[:done]
        final.flush()
        del final, obs_mm
        os.replace(out_prefix + "_obs_final.npy", out_prefix + "_obs.npy")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", default=os.path.expanduser("~/binbo-shogi-data/teacher"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4_data"))
    ap.add_argument("--n-train", type=int, default=2_000_000)
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20260702)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shards = shard_list(args.teacher_dir)
    train_sel, val_sel = sample_split(shards, args.n_train, args.n_val, args.seed)
    val_shards = sorted({os.path.relpath(p, args.teacher_dir) for p, _ in val_sel})
    print(f"shards: {len(shards)}; val shards: {val_shards}", flush=True)

    obs_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(True))))
    skips = {"expand_fail": 0, "illegal": 0, "map_fail": 0}
    n_val = build(val_sel, os.path.join(args.out_dir, "val"), obs_fn, skips)
    n_train = build(train_sel, os.path.join(args.out_dir, "train"), obs_fn, skips)

    meta = dict(seed=args.seed, runs=RUNS, n_train=n_train, n_val=n_val,
                val_shards=val_shards, skips=skips,
                target_v="tanh(cp/1200)", target_a="bestMove16->usi->pgx action int16",
                obs="packbits of _observe(flip=True)")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
