# Task 076 M1 part 2b: SFEN <-> Pgx bridge feasibility for champion-distillation warm-start.
#
# Validates the data path  SFEN -> pgx.experimental.shogi.from_sfen -> state -> observation
# that a champion-labeled distillation set would use, and measures its throughput.
#
# Checks:
#   1. startpos: from_sfen gives exactly 30 legal actions (known shogi fact) and
#      observation shape (9, 9, 119) bool.
#   2. roundtrip on diverse real positions: play random-policy games in Pgx, sample
#      positions (incl. hands/promotions), assert to_sfen -> from_sfen reproduces
#      board, hand, color and legal_action_mask exactly.
#   3. throughput: from_sfen conversions/s (Python-loop bound) and batched _observe
#      states/s, extrapolated to a 1M-position distillation set.

import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from pgx.experimental.shogi import from_sfen, to_sfen
from pgx._src.games.shogi import _observe

STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

env = pgx.make("shogi")


def check_startpos():
    s = from_sfen(STARTPOS)
    n_legal = int(s.legal_action_mask.sum())
    obs = _observe(s._x, flip=jnp.bool_(False))
    print(f"startpos: legal_actions={n_legal} (expect 30), obs shape={obs.shape} dtype={obs.dtype}")
    assert n_legal == 30
    assert obs.shape == (9, 9, 119)
    rt = to_sfen(s)
    print(f"startpos roundtrip: {'OK' if rt == STARTPOS else f'MISMATCH: {rt}'}")
    assert rt == STARTPOS


def collect_positions(n_games=40, seed=0):
    """Random-policy selfplay to harvest diverse real positions (hands, promotions)."""
    rng = jax.random.PRNGKey(seed)
    init = jax.jit(env.init)
    step = jax.jit(env.step)
    states = []
    for g in range(n_games):
        rng, k = jax.random.split(rng)
        s = init(k)
        ply = 0
        while not bool(s.terminated) and ply < 150:
            rng, k = jax.random.split(rng)
            logits = jnp.where(s.legal_action_mask, 0.0, -jnp.inf)
            a = jax.random.categorical(k, logits)
            s = step(s, a)
            ply += 1
            if ply % 10 == 0 and not bool(s.terminated):
                states.append(s)
    return states


def check_roundtrip(states):
    n_hand = 0
    n_promo = 0
    for i, s in enumerate(states):
        sfen = to_sfen(s)
        s2 = from_sfen(sfen)
        assert np.array_equal(np.asarray(s._x.board), np.asarray(s2._x.board)), f"board mismatch @{i}: {sfen}"
        assert np.array_equal(np.asarray(s._x.hand), np.asarray(s2._x.hand)), f"hand mismatch @{i}: {sfen}"
        assert int(s._x.color) == int(s2._x.color), f"color mismatch @{i}: {sfen}"
        assert np.array_equal(
            np.asarray(s.legal_action_mask), np.asarray(s2.legal_action_mask)
        ), f"legal mask mismatch @{i}: {sfen}"
        o1 = np.asarray(_observe(s._x, flip=jnp.bool_(False)))
        o2 = np.asarray(_observe(s2._x, flip=jnp.bool_(False)))
        assert np.array_equal(o1, o2), f"observation mismatch @{i}: {sfen}"
        n_hand += int(np.asarray(s._x.hand).sum() > 0)
        b = np.asarray(s._x.board)
        n_promo += int((((8 <= b) & (b < 14)) | (b >= 22)).any())  # 8..13 mine / 22..27 opp promoted
    print(f"roundtrip: {len(states)} positions OK (with hand pieces: {n_hand}, with promotions: {n_promo})")


def measure_throughput(states):
    sfens = [to_sfen(s) for s in states]
    # warm-up (JIT of _from_board inside from_sfen)
    _ = from_sfen(sfens[0])
    st = time.time()
    converted = [from_sfen(x) for x in sfens]
    jax.block_until_ready(converted[-1].legal_action_mask)
    dt = time.time() - st
    per_s = len(sfens) / dt
    print(f"from_sfen: {len(sfens)} positions in {dt:.2f}s = {per_s:.1f} pos/s "
          f"-> 1M positions ~ {1e6 / per_s / 3600:.1f} h (single process)")

    # batched observation compute (the GPU-side part of dataset building)
    xs = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[c._x for c in converted])
    obs_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(False))))
    _ = jax.block_until_ready(obs_fn(xs))  # compile
    st = time.time()
    reps = 20
    for _ in range(reps):
        out = obs_fn(xs)
    jax.block_until_ready(out)
    dt = time.time() - st
    per_s = len(converted) * reps / dt
    print(f"batched _observe: {per_s:,.0f} obs/s -> 1M positions ~ {1e6 / per_s:.1f} s")


if __name__ == "__main__":
    print(f"devices: {jax.devices()}")
    check_startpos()
    states = collect_positions()
    print(f"collected {len(states)} positions from random-policy games")
    check_roundtrip(states)
    measure_throughput(states)
    print("M1 2B BRIDGE CHECK DONE")
