# Task 077 M2 step 2: minimal USI engine = AZNet + mctx Gumbel MCTS (batch=1).
#
# Protocol: usi / isready (loads ckpt, JIT-warms search) / usinewgame /
#           position startpos [moves ...] | position sfen <sfen> [moves ...] /
#           go ... -> "bestmove <usi>" / quit.
# Search: mctx.gumbel_muzero_policy, batch=1, gumbel_scale=0.0 (deterministic play,
# no exploration noise), root/recurrent identical in structure to train_m1 selfplay.
# No legal moves (mate) -> "bestmove resign".
#
# --bench N: instead of serving USI, measure per-move latency for several sims values.

import argparse
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import mctx
import pgx

import m2_moves
from m2_distill_train import make_forward
from t091_model import KIND as T091_KIND, make_dual_forward

env = pgx.make("shogi")


def load_model(ckpt_path, seed=0):
    if ckpt_path == "random":
        forward = make_forward(64, 5)
        dummy = jnp.zeros((2, 9, 9, 119), dtype=bool)
        model = forward.init(jax.random.PRNGKey(seed), dummy)
        return forward, model, dict(source="random-init 64ch,5L")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("kind") == T091_KIND:
        cfg = ckpt["network_config"]
        forward = make_dual_forward(cfg)
        metrics = ckpt.get("metrics", {})
        return forward, ckpt["model"], dict(
            source=ckpt_path,
            arch="transformer_frozen_value",
            iteration=ckpt.get("iteration"),
            value_corr=metrics.get("value_corr_teacher"),
            top1=metrics.get("top1_teacher"),
        )
    cfg = ckpt["config"]
    # arch dispatch (Task 087 T1): CNN ckpts have no 'arch' key -> default 'cnn' (backward-compat)
    forward = make_forward(cfg.get("num_channels", 128), cfg.get("num_layers", 10),
                           cfg.get("resnet_v2", True), arch=cfg.get("arch", "cnn"),
                           d_model=cfg.get("d_model", 256), n_heads=cfg.get("n_heads", 8),
                           ff_mult=cfg.get("ff_mult", 4))
    return forward, ckpt["model"], dict(source=ckpt_path, val_corr=ckpt.get("val_corr"),
                                        arch=cfg.get("arch", "cnn"))


def build_search(forward, model, num_simulations):
    def recurrent_fn(model_, rng_key, action, state):
        del rng_key
        params, st = model_
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        (logits, value), _ = forward.apply(params, st, state.observation, is_eval=True)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return (
            mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value),
            state,
        )

    @jax.jit
    def search(state_b, key):
        params, st = model
        (logits, value), _ = forward.apply(params, st, state_b.observation, is_eval=True)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state_b)
        out = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state_b.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=0.0,
        )
        return out.action[0], value[0]

    return search


class Engine:
    def __init__(self, ckpt, sims, seed=20260702):
        self.forward, self.model, self.info = load_model(ckpt)
        self.search = build_search(self.forward, self.model, sims)
        self.jstep = jax.jit(env.step)
        self.jinit = jax.jit(env.init)
        self.key = jax.random.PRNGKey(seed)
        self.state = None
        self.moves = []

    def new_state(self, sfen=None):
        if sfen is None:
            return self.jinit(jax.random.PRNGKey(0))
        from pgx.experimental.shogi import from_sfen
        from pgx._src.games.shogi import _observe
        s = from_sfen(sfen)
        return s.replace(observation=_observe(s._x, flip=jnp.bool_(True)))

    def set_position(self, tokens):
        # tokens after "position": ["startpos", ("moves", m1, ...)] or ["sfen", <4 fields>, ...]
        if tokens[0] == "startpos":
            base_sfen, rest = None, tokens[1:]
        else:
            base_sfen, rest = " ".join(tokens[1:5]), tokens[5:]
        moves = rest[1:] if rest[:1] == ["moves"] else []
        # prefix cache: extend the previous state if possible
        if (self.state is not None and self.moves is not None
                and len(moves) >= len(self.moves) and moves[: len(self.moves)] == self.moves
                and getattr(self, "_base", "__none__") == (base_sfen or "__startpos__")):
            new = moves[len(self.moves):]
        else:
            self.state = self.new_state(base_sfen)
            self._base = base_sfen or "__startpos__"
            self.moves = []
            new = moves
        for usi in new:
            a = m2_moves.usi_to_action(usi, int(self.state._x.color))
            self.state = self.jstep(self.state, jnp.int32(a))
            self.moves.append(usi)

    def bestmove(self):
        mask = np.asarray(self.state.legal_action_mask)
        if bool(self.state.terminated) or not mask.any():
            return "resign", 0.0
        self.key, sub = jax.random.split(self.key)
        batched = jax.tree_util.tree_map(lambda a: a[None], self.state)
        action, v = self.search(batched, sub)
        usi = m2_moves.action_to_usi(int(action), np.asarray(self.state._x.board), int(self.state._x.color))
        return usi, float(v)


def serve(args):
    eng = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        tok = line.split()
        if tok[0] == "usi":
            print("id name binbo-az-m2")
            print("id author binbo-shogi")
            print("usiok", flush=True)
        elif tok[0] == "isready":
            if eng is None:
                eng = Engine(args.ckpt, args.sims)
                eng.set_position(["startpos"])  # JIT warmup: step + search compile
                eng.bestmove()
                eng.set_position(["startpos"])
            print("readyok", flush=True)
        elif tok[0] == "usinewgame":
            pass
        elif tok[0] == "position":
            eng.set_position(tok[1:])
        elif tok[0] == "go":
            t0 = time.time()
            usi, v = eng.bestmove()
            cp = int(np.clip(1200 * np.arctanh(np.clip(v, -0.999999, 0.999999)), -30000, 30000))
            print(f"info depth 1 score cp {cp} time {int((time.time()-t0)*1000)}")
            print(f"bestmove {usi}", flush=True)
        elif tok[0] == "quit":
            break


def bench(args):
    eng = Engine(args.ckpt, args.sims)
    rng = np.random.default_rng(0)
    for sims in [int(s) for s in args.bench_sims.split(",")]:
        eng.search = build_search(eng.forward, eng.model, sims)
        # positions: random playouts from startpos
        times = []
        for g in range(args.bench):
            eng.set_position(["startpos"])
            eng.state = eng.jinit(jax.random.PRNGKey(g))
            for _ in range(int(rng.integers(10, 60))):
                mask = np.asarray(eng.state.legal_action_mask)
                if bool(eng.state.terminated) or not mask.any():
                    break
                a = rng.choice(np.nonzero(mask)[0])
                eng.state = eng.jstep(eng.state, jnp.int32(a))
            eng.bestmove()  # warm (first call per sims compiles)
            t0 = time.time()
            eng.bestmove()
            times.append(time.time() - t0)
        print(f"sims={sims}: mean {np.mean(times)*1000:.0f} ms/move  median {np.median(times)*1000:.0f} ms  (n={len(times)})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="random", help="pickle from m2_distill_train, or 'random'")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--bench", type=int, default=0, help="measure per-move latency on N positions instead of serving USI")
    ap.add_argument("--bench-sims", default="4,8,16,32,64")
    args = ap.parse_args()
    if args.bench > 0:
        bench(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
