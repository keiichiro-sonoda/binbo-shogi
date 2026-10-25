# Task 077 M2 step 0: build the value-distillation dataset from teacher hcpe shards.
#
#   hcp --(cshogi)--> sfen --(numpy parse, from_sfen-equivalent)--> GameState fields
#       --(_observe(flip=True), vmap batch)--> obs --packbits--> memmap
#   target v = tanh(cp / 1200)  ==  2*sigmoid(cp/600) - 1   (scale_cp=600, existing convention)
#
# flip=True is REQUIRED: state.observation during play equals _observe(_x, flip=True)
# (verified empirically over plies; flip=False is the opponent's view).
#
# The sfen parse is a numpy lift of pgx.experimental.shogi.from_sfen WITHOUT the jitted
# _from_board legal-mask computation (not needed for observations; per-position device
# round-trips made the from_sfen path ~130 pos/s). --verify N gates the lifted parser
# against the original from_sfen path (board/hand/color/obs exact equality).
# One deliberate difference: multi-digit hand counts ("18P") are parsed correctly here,
# while upstream from_sfen reads them digit-by-digit (latent upstream bug).
#
# Sampling: train positions are drawn per-shard with a seeded RNG across all 32M runs;
# val comes from held-out tail shards (disjoint shards => no game overlap with train).
# Storage: obs bool (9,9,119) -> np.packbits => 1205 B/pos (2M ~ 2.4 GB memmap).

import argparse
import json
import os
import time

import numpy as np
import cshogi
import jax
import jax.numpy as jnp

from pgx._src.games.shogi import GameState, _observe

RUNS = ["run1m", "run1m_b", "run2m_c", "run4m_d", "run8m_e", "run16m_f"]
OBS_BITS = 9 * 9 * 119  # 9639
OBS_BYTES = (OBS_BITS + 7) // 8  # 1205

# fmt: off
BOARD_CHAR_DIR = ["P", "L", "N", "S", "B", "R", "G", "K", "", "", "", "", "", "", "p", "l", "n", "s", "b", "r", "g", "k"]
HAND_CHAR_DIR = ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"]
# fmt: on


def parse_sfen_np(sfen):
    """sfen -> (color, board81 int32 side-to-move POV, hand (2,7) int32 side-to-move POV).
    Numpy equivalent of from_sfen + (_from_board's white flip), minus legal-mask compute."""
    board_s, color_s, hand_s, _step = sfen.split()
    piece_board = np.zeros(81, dtype=np.int32)
    for i, rank_s in enumerate(board_s.split("/")):
        rank = []
        piece = 0
        for ch in rank_s:
            if ch.isdigit():
                rank.extend([-1] * int(ch))
            elif ch == "+":
                piece += 8
            else:
                piece += BOARD_CHAR_DIR.index(ch)
                rank.append(piece)
                piece = 0
        piece_board[9 * i : 9 * i + 9] = rank
    s_hand = np.zeros(14, dtype=np.int32)
    if hand_s != "-":
        num = ""
        for ch in hand_s:
            if ch.isdigit():
                num += ch  # accumulate (upstream from_sfen mis-parses "18P" as 8)
            else:
                s_hand[HAND_CHAR_DIR.index(ch)] = int(num) if num else 1
                num = ""
    piece_board = np.rot90(piece_board.reshape(9, 9), k=1).flatten()
    hand = s_hand.reshape(2, 7)
    color = 0 if color_s == "b" else 1
    if color == 1:  # _flip: to side-to-move POV
        mask = piece_board >= 0
        piece_board = np.where(mask, (piece_board + 14) % 28, piece_board)[::-1].copy()
        hand = hand[::-1].copy()
    return color, piece_board, hand


_PROTO = GameState()


def batch_state(colors, boards, hands):
    n = len(colors)
    return GameState(
        step_count=jnp.zeros(n, jnp.int32),
        color=jnp.asarray(np.asarray(colors, np.int32)),
        board=jnp.asarray(np.stack(boards)),
        hand=jnp.asarray(np.stack(hands)),
        cache_m2b=jnp.tile(_PROTO.cache_m2b, (n, 1)),  # _observe re-derives via _set_cache
        cache_king=jnp.tile(_PROTO.cache_king, n),
        legal_action_mask=jnp.tile(_PROTO.legal_action_mask, (n, 1)),  # unused by _observe
    )


def shard_list(teacher_dir):
    out = []
    for r in RUNS:
        rd = os.path.join(teacher_dir, r)
        for f in sorted(os.listdir(rd)):
            if f.endswith(".hcpe"):
                out.append(os.path.join(rd, f))
    return out


def sample_split(shards, n_train, n_val, seed):
    """Return (train_sel, val_sel): lists of (shard_path, np.array indices).
    Val = entries from the tail shards (disjoint shards from train)."""
    rng = np.random.default_rng(seed)
    ent_size = np.dtype(cshogi.HuffmanCodedPosAndEval).itemsize
    counts = [os.path.getsize(p) // ent_size for p in shards]
    val_sel, acc = [], 0
    k = len(shards)
    while acc < n_val and k > 0:
        k -= 1
        take = min(counts[k], n_val - acc)
        idx = np.arange(counts[k]) if take == counts[k] else rng.choice(counts[k], take, replace=False)
        val_sel.append((shards[k], np.sort(idx)))
        acc += take
    train_shards = shards[:k]
    total = sum(counts[:k])
    base = [int(n_train * counts[i] / total) for i in range(k)]
    rem = n_train - sum(base)
    for i in range(rem):
        base[i % k] += 1
    train_sel = []
    for i, p in enumerate(train_shards):
        take = min(base[i], counts[i])
        idx = rng.choice(counts[i], take, replace=False)
        train_sel.append((p, np.sort(idx)))
    return train_sel, val_sel


def decode(sel):
    """Yield (sfen, cp) for selected entries."""
    board = cshogi.Board()
    for path, idx in sel:
        ent = np.fromfile(path, dtype=cshogi.HuffmanCodedPosAndEval)
        for i in idx:
            e = ent[i]
            board.set_hcp(np.asarray(e["hcp"]))
            yield board.sfen(), int(e["eval"])


def build(sel, out_prefix, obs_fn, chunk=4096, log_every=100_000):
    n = sum(len(idx) for _, idx in sel)
    obs_mm = np.lib.format.open_memmap(
        out_prefix + "_obs.npy", mode="w+", dtype=np.uint8, shape=(n, OBS_BYTES)
    )
    v = np.zeros(n, dtype=np.float32)
    colors, boards, hands, cps = [], [], [], []
    done, t0 = 0, time.time()

    def flush():
        nonlocal done
        if not colors:
            return
        obs = np.asarray(obs_fn(batch_state(colors, boards, hands)))
        obs_mm[done : done + len(colors)] = np.packbits(obs.reshape(len(colors), -1), axis=1)
        v[done : done + len(colors)] = np.tanh(np.asarray(cps, dtype=np.float64) / 1200.0)
        done += len(colors)
        for b in (colors, boards, hands, cps):
            b.clear()
        if done % log_every < chunk:
            rate = done / (time.time() - t0)
            print(f"  {out_prefix}: {done}/{n} ({rate:.0f} pos/s, ETA {(n-done)/max(rate,1e-9)/60:.1f} min)", flush=True)

    for sfen, cp in decode(sel):
        c, pb, hd = parse_sfen_np(sfen)
        colors.append(c)
        boards.append(pb)
        hands.append(hd)
        cps.append(cp)
        if len(colors) >= chunk:
            flush()
    flush()
    obs_mm.flush()
    np.save(out_prefix + "_v.npy", v)
    return n


def verify_against_from_sfen(sel, n_check, obs_fn):
    """Gate: lifted numpy parser must reproduce from_sfen exactly (057 lesson)."""
    from pgx.experimental.shogi import from_sfen

    checked = 0
    for sfen, _cp in decode(sel):
        c, pb, hd = parse_sfen_np(sfen)
        ref = from_sfen(sfen)._x
        assert int(ref.color) == c, f"color mismatch: {sfen}"
        assert np.array_equal(np.asarray(ref.board), pb), f"board mismatch: {sfen}"
        assert np.array_equal(np.asarray(ref.hand), hd), f"hand mismatch: {sfen}"
        o_np = np.asarray(obs_fn(batch_state([c], [pb], [hd])))[0]
        o_ref = np.asarray(_observe(ref, flip=jnp.bool_(True)))
        assert np.array_equal(o_np, o_ref), f"obs mismatch: {sfen}"
        checked += 1
        if checked >= n_check:
            break
    print(f"verify: numpy parser == from_sfen on {checked} positions (board/hand/color/obs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", default=os.path.expanduser("~/binbo-shogi-data/teacher"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2_data"))
    ap.add_argument("--n-train", type=int, default=2_000_000)
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--verify", type=int, default=2000, help="gate the numpy parser against from_sfen on N positions")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shards = shard_list(args.teacher_dir)
    print(f"shards: {len(shards)} across {RUNS}")
    train_sel, val_sel = sample_split(shards, args.n_train, args.n_val, args.seed)
    val_shards = sorted({os.path.relpath(p, args.teacher_dir) for p, _ in val_sel})
    print(f"val shards (held out): {val_shards}")

    obs_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(True))))
    obs1_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(True))))

    if args.verify > 0:
        verify_against_from_sfen(val_sel, args.verify, obs1_fn)

    n_val = build(val_sel, os.path.join(args.out_dir, "val"), obs_fn)
    n_train = build(train_sel, os.path.join(args.out_dir, "train"), obs_fn)

    meta = dict(
        seed=args.seed,
        runs=RUNS,
        n_train=n_train,
        n_val=n_val,
        val_shards=val_shards,
        target="tanh(cp/1200)",
        obs="packbits of _observe(flip=True), (9,9,119) bool, 1205 B/pos",
        verify_positions=args.verify,
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
