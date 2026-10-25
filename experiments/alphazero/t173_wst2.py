#!/usr/bin/env python3
"""Task 173: compact self-play dataset, resumable WS-T fine tuning, paired evaluation.

Run with the pgx Python environment; --repo points to the read-only source tree.
Only requested output directories are written. Pickles must be trusted inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
WORK = DATA_ROOT / "work/task173_wst2"
FRAMEWORKS = Path(os.environ.get("PGX_HOME", Path.home() / "task074_frameworks"))
WS_SHA = "20bf5e826e307cf191078f1cb86bf2c6f7bf79ef6e89bb2d40829844950fe2b3"
SEED = 20271730
FORMAT = "task173-wst2-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_write(path, obj):
    atomic(path, (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def log(obj):
    print(canonical(obj), flush=True)


def setup(args):
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("JAX_PLATFORMS", "")
    source = args.repo.resolve() / "experiments/alphazero"
    require((source / "network.py").is_file(), "--repo must contain experiments/alphazero/network.py")
    sys.path.insert(0, str(source))
    import numpy as np
    import jax
    devices = jax.devices()
    device = next((d for d in devices if d.platform == "gpu"), devices[0])
    require(args.device != "gpu" or device.platform == "gpu", "GPU requested but unavailable")
    # CPU parsing/encoding is vectorized; GPU remains available for the network.
    return np, jax, device


def record_dtype():
    import numpy as np
    return np.dtype([("sfen", "S192"), ("action", "<u2"), ("value", "<f4"),
                     ("result", "i1"), ("game", "<u4"), ("ply", "<u2"),
                     ("score", "<i2"), ("mate", "u1")])


def load_game(path):
    game = read(path)
    require(game.get("format") == "task153-selfplay-v1", f"not a raw self-play game: {path}")
    require(game.get("content_sha256") == identity({k: v for k, v in game.items() if k != "content_sha256"}),
            f"game content checksum mismatch: {path}")
    require(game["winner"] in (-1, 0, 1), f"invalid winner: {path}")
    return game


def game_key(game):
    # Game ids restart in each round. The recipe includes seed and engine/eval hashes.
    return f"{game['recipe_id']}:{int(game['game_id'])}"


def exclusions(probes):
    paths, keys, provenance = set(), set(), []
    for folder in probes:
        path = folder / "roots.json"
        roots = read(path)["roots"]
        for root in roots:
            source = Path(root["source"]).resolve()
            if str(source) not in paths:
                game = load_game(source)
                require(str(game["game_id"]) == str(root["game_id"]), "probe source/game_id mismatch")
                require(digest(source) == root["source_sha256"], "probe source checksum mismatch")
                paths.add(str(source))
                keys.add(game_key(game))
            else:
                require(str(read(source)["game_id"]) == str(root["game_id"]), "probe id mismatch")
        provenance.append(dict(path=str(path.resolve()), sha256=digest(path), roots=len(roots)))
    return paths, keys, provenance


def split_games(games, fraction=0.02, seed=SEED):
    require(len(games) >= 2 and 0 < fraction < 1, "need >=2 games and 0<val fraction<1")
    order = sorted(games, key=lambda g: (identity([seed, g["key"]]), g["key"]))
    nval = min(len(games) - 1, max(1, int(len(games) * fraction)))
    val = {g["key"] for g in order[:nval]}
    return {g["key"]: ("val" if g["key"] in val else "train") for g in games}


def value_target(score, result, mate=False):
    require(-32000 <= score <= 32000 and result in (-1, 0, 1), "invalid target")
    if mate or abs(score) == 32000:
        require(score != 0, "zero mate score")
        p = float(score > 0)
    else:
        # score is internal Value (NOT the source's raw USI centipawns).
        p = 1 / (1 + math.exp(-max(-3000, min(3000, score)) / 600))
    mixed = p if result == 0 else 0.7 * p + 0.3 * ((result + 1) / 2)
    return 2 * mixed - 1


def build(args):
    np, _, _ = setup(args)
    import cshogi
    import m2_moves
    out = args.out.resolve()
    require(not out.exists(), f"output already exists; use a new --out: {out}")
    excluded_paths, excluded_keys, probes = exclusions(args.probes)
    games, seen, counts = [], {}, Counter()
    started = time.perf_counter()
    for folder in sorted({p.resolve() for p in args.games}):
        files = sorted(folder.glob("game_*.json"))
        require(files, f"no raw games in {folder}")
        for path in files:
            if str(path) in excluded_paths:
                counts["excluded_files"] += 1
                continue
            game = load_game(path)
            key = game_key(game)
            if key in excluded_keys:
                counts["excluded_aliases"] += 1
                continue
            if key in seen:
                require(seen[key] == game["content_sha256"], f"conflicting duplicate game: {key}")
                counts["duplicate_games"] += 1
                continue
            seen[key] = game["content_sha256"]
            games.append(dict(key=key, source=str(path), game_id=str(game["game_id"]),
                              content_sha256=game["content_sha256"]))
            if len(games) % 10000 == 0:
                log(dict(scan_games=len(games), counts=dict(counts)))
            if args.max_games and len(games) >= args.max_games:
                break
        if args.max_games and len(games) >= args.max_games:
            break
        log(dict(scan_games=len(games), counts=dict(counts)))
    games.sort(key=lambda g: g["key"])
    split = split_games(games, args.val_fraction, args.seed)
    require(not (set(split) & excluded_keys), "excluded game entered split")
    out.mkdir(parents=True)
    provenance = []
    with ExitStack() as stack:
        streams = {s: stack.enter_context((out / (s + ".bin")).open("xb")) for s in ("train", "val")}
        for gi, item in enumerate(games):
            game = load_game(item["source"])
            require(game["content_sha256"] == item["content_sha256"], "game changed during build")
            random_plies = set(game["random_plies"])
            packed, plies = [], set()
            for row in game["rows"]:
                ply = int(row["ply"])
                if ply < max(16, game["opening_plies"]) or ply in random_plies:
                    counts["skipped_opening"] += 1
                    continue
                if not (row["depth"] >= 9 or row["mate"]):
                    counts["skipped_shallow_node_cap"] += 1  # Task 173 (Claude): node-capped searches (~400 rows, Task 156)
                    continue
                require(ply not in plies and 0 <= ply <= 65535, "duplicate/invalid ply")
                plies.add(ply)
                board = cshogi.Board(row["sfen"])
                turn = board.turn
                expected = 0 if game["winner"] == -1 else (1 if turn == game["winner"] else -1)
                require(row["turn"] == turn and row["result"] == expected
                        and str(row["game_id"]) == item["game_id"], "row result/id/turn mismatch")
                require(int(row["sfen"].split()[3]) == ply + 1, "SFEN ply mismatch")
                legal = {cshogi.move_to_usi(m) for m in board.legal_moves}
                require(row["move"] in legal, f"illegal d9 target in {item['source']} ply={ply}")
                if ply < len(game["moves"]):
                    require(game["moves"][ply] == row["move"], "chosen move differs from game move")
                sfen = row["sfen"].encode("ascii")
                require(len(sfen) <= 192, "SFEN exceeds record capacity")
                action = m2_moves.usi_to_action(row["move"], turn)
                packed.append((sfen, action, value_target(row["score"], row["result"], row["mate"]),
                               row["result"], gi, ply, row["score"], row["mate"]))
                if args.max_positions_per_game and len(packed) >= args.max_positions_per_game:
                    break
            side = split[item["key"]]
            if not packed:  # Task 173 (Claude): e.g. the 15-ply game of Task 156; keep the index, write no rows
                counts["skipped_empty_games"] += 1
                provenance.append(dict(**item, split=side, positions=0))
                continue
            streams[side].write(np.asarray(packed, dtype=record_dtype()).tobytes())
            counts[side] += len(packed)
            counts[side + "_games"] += 1
            provenance.append(dict(**item, split=side, positions=len(packed)))
            if (gi + 1) % 10000 == 0:
                log(dict(built_games=gi + 1, counts=dict(counts)))
        for f in streams.values():
            f.flush()
            os.fsync(f.fileno())
    atomic(out / "games.jsonl", ("\n".join(canonical(p) for p in provenance) + "\n").encode())
    meta = dict(format=FORMAT, seed=args.seed, val_fraction=args.val_fraction, counts=dict(counts),
                dtype=record_dtype().descr, record_bytes=record_dtype().itemsize,
                excluded_keys=sorted(excluded_keys), excluded_paths=sorted(excluded_paths), probes=probes,
                subset=bool(args.max_games or args.max_positions_per_game),
                max_games=args.max_games, max_positions_per_game=args.max_positions_per_game,
                target="2*(0.7*sigmoid(clip(internal_Value,-3000,3000)/600)+0.3*z)-1; draws pure eval; mate p=0/1",
                files={f: digest(out / f) for f in ("train.bin", "val.bin", "games.jsonl")})
    meta["dataset_id"] = identity(meta)
    json_write(out / "meta.json", meta)  # completion marker written last
    log(dict(build_done=meta["dataset_id"], seconds=time.perf_counter() - started, counts=dict(counts)))


class Dataset:
    def __init__(self, folder, verify=True):
        import numpy as np
        self.folder = Path(folder)
        self.meta = read(self.folder / "meta.json")
        require(self.meta["format"] == FORMAT, "wrong dataset format")
        require(self.meta["dataset_id"] == identity({k: v for k, v in self.meta.items() if k != "dataset_id"}),
                "dataset metadata mismatch")
        for name, sha in self.meta["files"].items():
            if verify:
                require(digest(self.folder / name) == sha, f"dataset checksum mismatch: {name}")
        self.games = [json.loads(line) for line in (self.folder / "games.jsonl").read_text().splitlines()]
        keys = [g["key"] for g in self.games]
        require(len(keys) == len(set(keys)) and not (set(keys) & set(self.meta["excluded_keys"])),
                "duplicate/excluded games in dataset")
        self.rows = {}
        for side in ("train", "val"):
            path = self.folder / (side + ".bin")
            require(path.stat().st_size == self.meta["counts"][side] * record_dtype().itemsize, "truncated records")
            rows = np.memmap(path, mode="r", dtype=record_dtype())
            allowed = np.array([g["split"] == side for g in self.games])
            for i in range(0, len(rows), 1_000_000):
                ids = rows["game"][i:i + 1_000_000]
                require((ids < len(allowed)).all() and allowed[ids].all(), "game split leakage in records")
            self.rows[side] = rows


class Encoder:
    """Host SFEN parser + batched verified pgx constructors (history unavailable).

    Same empty-history/flip=True convention as t167. CPU encoding is transferred
    as a batch to the model device. No expanded observation cache is stored.
    """
    def __init__(self, jax):
        import numpy as np
        import jax.numpy as jnp
        from pgx.experimental.shogi import _from_board
        from pgx._src.games.shogi import _observe
        self.jax, self.np = jax, np
        self.cpu = jax.devices("cpu")[0]

        def encode(color, board, hand):
            state = _from_board(color, board, hand)
            return _observe(state._x, flip=jnp.bool_(True)), state.legal_action_mask

        self.compiled = jax.jit(jax.vmap(encode))

    @staticmethod
    def parse(sfen):
        import numpy as np
        board, color, hand, step = sfen.split()
        require(color in ("b", "w") and int(step) > 0, "bad SFEN side/step")
        ranks = []
        pieces = {p: i for i, p in enumerate("PLNSBRGK")}
        for rank in board.split("/"):
            squares, promoted = [], False
            for char in rank:
                if char.isdigit():
                    require(not promoted and char != "0", "bad SFEN empty run")
                    squares.extend([-1] * int(char))
                elif char == "+":
                    require(not promoted, "double promotion")
                    promoted = True
                else:
                    squares.append(pieces[char.upper()] + 14 * char.islower() + 8 * promoted)
                    promoted = False
            require(len(squares) == 9 and not promoted, "bad SFEN rank")
            ranks.append(squares)
        require(len(ranks) == 9, "bad SFEN board")
        hands = np.zeros((2, 7), np.int32)
        if hand != "-":
            tokens = re.findall(r"(\d*)([PLNSBRGplnsbrg])", hand)
            require("".join(n + p for n, p in tokens) == hand, "bad SFEN hand")
            for count, piece in tokens:
                hands[int(piece.islower()), "PLNSBRG".index(piece.upper())] = int(count or 1)
        return int(color == "w"), np.rot90(np.array(ranks, np.int32), k=1).reshape(81), hands

    def batch(self, sfens, size=None, check_legal=False):
        np = self.np
        require(len(sfens) > 0, "empty encoding batch")
        size = size or len(sfens)
        require(size >= len(sfens), "padding size too small")
        padded = list(sfens) + [sfens[-1]] * (size - len(sfens))
        parsed = [self.parse(s) for s in padded]
        arrays = [np.stack(col) for col in zip(*parsed)]
        with self.jax.default_device(self.cpu):
            obs, mask = self.compiled(*[self.jax.device_put(a, self.cpu) for a in arrays])
        obs, mask = np.asarray(obs), np.asarray(mask)
        require(mask[:len(sfens)].any(axis=1).all(), "terminal root in data")
        if check_legal:
            import cshogi
            import m2_moves
            for sfen, legal in zip(sfens, mask):
                board = cshogi.Board(sfen)
                moves = {cshogi.move_to_usi(m) for m in board.legal_moves}
                expected = {m2_moves.usi_to_action(m, board.turn) for m in moves}
                require(expected == set(np.flatnonzero(legal)), f"pgx/cshogi legal set mismatch: {sfen}")
        return obs, mask


def load_checkpoint(path, jax, device, baseline=False):
    require(not baseline or digest(path) == WS_SHA, "baseline is not the identified WS-T checkpoint")
    with Path(path).open("rb") as f:
        checkpoint = pickle.load(f)
    from m2_distill_train import make_forward
    cfg = checkpoint["config"]
    require(cfg["arch"] == "transformer" and cfg["num_actions"] == 2187, "expected WS-T transformer")
    forward = make_forward(**cfg)
    return checkpoint, forward, jax.device_put(checkpoint["model"], device)


def learning_rate(step, settings):
    warmup = settings["warmup"]
    if warmup and step < warmup:
        return settings["lr"] * (step + 1) / warmup
    progress = min(1., max(0., (step - warmup) / max(1, settings["total_steps"] - warmup - 1)))
    return settings["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train_step_fn(forward, jax):
    import jax.numpy as jnp
    import optax
    optimizer = optax.adam(1.0)  # External scalar schedule; moment state survives resume.

    def loss_fn(params, state, obs, legal, targets, values):
        (logits, value), state = forward.apply(params, state, obs, is_eval=False)
        logits = jnp.where(legal, logits, jnp.finfo(logits.dtype).min)
        p_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        v_loss = ((value - values) ** 2).mean()
        return p_loss + v_loss, (state, p_loss, v_loss)

    @jax.jit
    def step(model, opt_state, obs, legal, targets, values, rate):
        (loss, (state, pol, val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            model[0], model[1], obs, legal, targets, values)
        updates, opt_state = optimizer.update(grads, opt_state, model[0])
        updates = jax.tree_util.tree_map(lambda u: u * rate, updates)
        model = (optax.apply_updates(model[0], updates), state)
        return model, opt_state, jnp.stack([loss, pol, val])

    return optimizer, step


def prepare(rows, encoder, jax, device, check=False):
    import numpy as np
    sfens = [s.decode("ascii") for s in rows["sfen"]]
    obs, mask = encoder.batch(sfens, check_legal=check)
    actions = rows["action"].astype(np.int32)
    require(mask[np.arange(len(rows)), actions].all(), "illegal policy target in pgx mask")
    return tuple(jax.device_put(a, device) for a in (obs, mask, actions, rows["value"]))


class Sampler:
    """Seeded per-epoch permutation; absolute step reconstructs resume cursor.

    Each step has a fixed size, wrapping at epoch boundaries (no dropped rows).
    """
    def __init__(self, n, batch, seed):
        self.n, self.batch, self.seed = n, batch, seed
        self.epoch, self.permutation = None, None

    def indices(self, step):
        import numpy as np
        offset, remaining, parts = step * self.batch, self.batch, []
        while remaining:
            epoch, start = divmod(offset, self.n)
            if epoch != self.epoch:
                self.permutation = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.n)
                self.epoch = epoch
            take = min(remaining, self.n - start)
            parts.append(self.permutation[start:start + take])
            offset += take
            remaining -= take
        return np.sort(np.concatenate(parts))


def benchmark(rows, encoder, jax, device, model, optimizer, step, args):
    candidates = [int(args.batch)] if args.batch != "auto" else ([32, 64, 128, 256, 512, 1024] if device.platform == "gpu" else [8])
    results = []
    for batch in candidates:
        sample = rows[__import__("numpy").arange(batch) % len(rows)]
        try:
            t0 = time.perf_counter()
            inputs = prepare(sample, encoder, jax, device)
            test_model, test_opt, losses = step(model, optimizer.init(model[0]), *inputs, args.lr)
            jax.block_until_ready((test_model, test_opt, losses))
            # The first updated state can have different weak-type metadata
            # from optimizer.init; warm that signature before timing as well.
            test_model, test_opt, losses = step(test_model, test_opt, *inputs, args.lr)
            jax.block_until_ready((test_model, test_opt, losses))
            compile_seconds = time.perf_counter() - t0
            start = time.perf_counter()
            sampler = Sampler(len(rows), batch, args.seed)
            for bench_step in range(args.bench_steps):
                # Re-parse, encode, transfer, train, synchronize every iteration.
                sample = rows[sampler.indices(bench_step)]
                inputs = prepare(sample, encoder, jax, device)
                test_model, test_opt, losses = step(test_model, test_opt, *inputs, args.lr)
                jax.block_until_ready((test_model, test_opt, losses))
            elapsed = time.perf_counter() - start
            result = dict(batch=batch, steps=args.bench_steps, seconds=elapsed,
                          seconds_per_step=elapsed / args.bench_steps,
                          positions_per_second=batch * args.bench_steps / elapsed,
                          compile_seconds=compile_seconds)
            results.append(result)
            log(dict(benchmark=result))
            del test_model, test_opt, inputs
        except Exception as exc:
            if args.batch != "auto" or not any(s in str(exc).lower() for s in ("out of memory", "resource_exhausted")):
                raise
            log(dict(benchmark_batch_failed=batch, error=str(exc)[:400]))
            break
    require(results, "no batch size fits")
    return max(results, key=lambda r: r["batch"]), results


def train(args):
    np, jax, device = setup(args)
    dataset = Dataset(args.data)
    encoder = Encoder(jax)
    rows = dataset.rows["train"]
    out = args.out.resolve()
    source = args.resume if args.resume else args.init
    checkpoint, forward, model = load_checkpoint(source, jax, device, baseline=not args.resume)
    optimizer, step_fn = train_step_fn(forward, jax)
    if args.resume:
        require(checkpoint.get("format") == FORMAT, "not a resumable Task 173 checkpoint")
        require(checkpoint["dataset_id"] == dataset.meta["dataset_id"], "resume dataset changed")
        require(args.steps is None and args.epochs is None, "resume uses saved schedule; omit --steps/--epochs")
        settings = checkpoint["train_settings"]
        step, opt_state = checkpoint["step"], jax.device_put(checkpoint["opt_state"], device)
        require(opt_state is not None, "missing Adam state")
        bench = checkpoint["benchmark"]
        require(out.exists(), "resume --out must exist")
    else:
        require(not out.exists(), "training output already exists; use --resume or a new --out")
        started = time.perf_counter()
        best, measurements = benchmark(rows, encoder, jax, device, model, optimizer, step_fn, args)
        batch = best["batch"]
        if args.steps is not None:
            total = args.steps
        elif args.epochs is not None:
            total = math.ceil(len(rows) * args.epochs / batch)
        else:
            remaining = args.hours * 3600 - (time.perf_counter() - started)
            total = max(1, int(0.9 * max(0., remaining) / best["seconds_per_step"]))
        settings = dict(batch=batch, total_steps=total, seed=args.seed, lr=args.lr,
                        warmup=min(args.warmup, max(0, total // 10)),
                        schedule="linear warmup then cosine to 0.1*lr", sampling="seeded epoch permutation; wrap tail")
        bench = dict(device=str(device), backend=device.platform, measurements=measurements,
                     selected=best, target_hours=args.hours, default_budget_safety_fraction=0.9,
                     scope="host SFEN parse + CPU pgx legal/obs + transfer + forward/backward/Adam + synchronization; no JIT",
                     estimated_training_seconds=total * best["seconds_per_step"])
        step, opt_state = 0, optimizer.init(model[0])
        out.mkdir(parents=True)
        json_write(out / "throughput.json", bench)
    require(settings["total_steps"] > 0, "no training steps")
    sampler = Sampler(len(rows), settings["batch"], settings["seed"])
    stop = min(settings["total_steps"], step + args.stop_after) if args.stop_after else settings["total_steps"]
    run_meta = dict(script_sha256=digest(__file__),
                    dependencies={n: digest(args.repo / "experiments/alphazero" / n)
                                  for n in ("network.py", "m2_moves.py", "m2_distill_train.py")},
                    packages={n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "pgx", "dm-haiku", "optax", "numpy")},
                    device=str(device), cpu_affinity=sorted(os.sched_getaffinity(0)))
    if args.resume:
        require(checkpoint["run_meta"]["script_sha256"] == run_meta["script_sha256"], "resume trainer source changed")
        require(checkpoint["run_meta"]["dependencies"] == run_meta["dependencies"], "resume network/encoding dependencies changed")
        require(checkpoint["run_meta"]["packages"] == run_meta["packages"], "resume package versions changed")
    log(dict(training=settings, positions=len(rows), dataset_id=dataset.meta["dataset_id"], start_step=step))
    started = time.perf_counter()

    def save(current):
        blob = dict(format=FORMAT, kind="WS-T2 own self-play fine tune", config=checkpoint["config"],
                    model=jax.device_get(model), opt_state=jax.device_get(opt_state), step=current,
                    iteration=0, rng_key=np.asarray(jax.random.PRNGKey(settings["seed"])),
                    train_settings=settings, dataset_id=dataset.meta["dataset_id"], benchmark=bench,
                    baseline_sha256=WS_SHA, run_meta=run_meta)
        path = out / f"step_{current:09d}.ckpt"
        atomic(path, pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))
        json_write(out / "latest.json", dict(checkpoint=path.name, step=current, sha256=digest(path)))
        return path

    with (out / "history.jsonl").open("a") as history:
        for current in range(step, stop):
            batch = rows[sampler.indices(current)]
            inputs = prepare(batch, encoder, jax, device, check=current == step)
            rate = learning_rate(current, settings)
            model, opt_state, losses = step_fn(model, opt_state, *inputs, rate)
            loss = np.asarray(losses)
            require(np.isfinite(loss).all(), "nonfinite training loss; no checkpoint published")
            if (current + 1) % args.log_every == 0 or current == step or current + 1 == stop:
                entry = dict(step=current + 1, loss=float(loss[0]), policy_ce=float(loss[1]),
                             value_mse=float(loss[2]), lr=rate, seconds=time.perf_counter() - started)
                history.write(canonical(entry) + "\n")
                history.flush()
                log(entry)
            if (current + 1) % args.checkpoint_every == 0 or current + 1 == stop:
                last_path = save(current + 1)
    if stop <= step:
        last_path = Path(source)
    log(dict(train_done=str(last_path), step=stop, schedule_complete=stop == settings["total_steps"]))


def policy_predictions(sfens, predict, model, encoder, jax, device, batch, check=True):
    import numpy as np
    for start in range(0, len(sfens), batch):
        chunk = sfens[start:start + batch]
        obs, mask = encoder.batch(chunk, size=batch, check_legal=check)
        logits, value = predict(model, jax.device_put(obs, device))
        logits, value = np.asarray(logits)[:len(chunk)], np.asarray(value)[:len(chunk)]
        mask = mask[:len(chunk)]
        require(np.isfinite(logits).all() and np.isfinite(value).all(), "nonfinite inference")
        logits = np.where(mask, logits, -np.inf)
        order = np.argsort(-logits, axis=1, kind="stable")[:, :2]
        require(mask[np.arange(len(chunk)), order[:, 0]].all(), "unmasked top-1")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        for i in range(len(chunk)):
            top = [int(a) for a in order[i] if mask[i, a]]
            yield dict(top_actions=top, probability=float(probs[i, top[0]]), value=float(value[i]))


def gate_rule(rows, key="agree", big="l4"):
    # Exact t167_posthoc_agree_alloc.py rule; that script cannot be imported
    # because it executes file reads/bootstrap at module scope.
    a, d = [x for x in rows if x[key]], [x for x in rows if not x[key]]
    n = len(rows)
    require(n > 0, "empty gate cohort")
    if not d:
        return (0., 0., 0., 0.)  # Preserve the original script's convention.
    q = max(0., min(1., (1.6 * n - len(a) - 1.6 * len(d)) / (2.4 * len(d))))
    loss = (sum(x["ln"] for x in a) + sum(q * x[big] + (1 - q) * x["lu"] for x in d)) / n
    base = sum(x["lu"] for x in rows) / n
    return (base - loss, sum(x["lu"] - x["ln"] for x in a) / n,
            sum(q * (x["lu"] - x[big]) for x in d) / n, q)


def gate_summary(rows, repeats):
    import numpy as np
    result = {}
    for trim in (False, True):
        rs = sorted(rows, key=lambda x: -abs(x["lu"] - x["ln"]) if x["agree"] else 0)[2:] if trim else rows
        if not rs:
            continue
        for key in (("agree",) if trim else ("agree", "agree3")):
            for big in ("l4", "ll"):
                require(all(x[big] is not None for x in rs), "missing live continuation loss")
                point = gate_rule(rs, key, big)
                rng = random.Random(20271698 if trim else 20271699)
                bs = sorted(gate_rule([rng.choice(rs) for _ in rs], key, big)[0] for _ in range(repeats))
                ci = [bs[int(.025 * repeats)], bs[min(repeats - 1, math.ceil(.975 * repeats) - 1)]] if repeats else None
                result[f"{'trim2_' if trim else ''}{key}_{big}"] = dict(
                    n=len(rs), n_agree=sum(x[key] for x in rs), reduction_cp=point[0],
                    ci95_cp=ci, agree_part_cp=point[1], disagree_part_cp=point[2], q4n=point[3],
                    nominal_mean_nodes_ratio=(sum(x[key] for x in rs) + (len(rs) - sum(x[key] for x in rs)) * (1.6 + 2.4 * point[3])) / len(rs),
                    all_agree_original_convention=all(x[key] for x in rs))
    return result


def probe_records(folders, expected=1184, limit=None):
    records, hashes = [], []
    for folder in folders:
        roots = read(folder / "roots.json")["roots"]
        report_doc = read(folder / "report.json")
        report = report_doc["roots"]
        manifest = read(folder / "search_manifest.json")
        require(report_doc["search_recipe"] == manifest, "report/search manifest mismatch")
        require(manifest["roots_sha256"] == identity(read(folder / "roots.json")), "search/roots mismatch")
        indexed = {r["root_id"]: r for r in roots}
        require(len(indexed) == len(roots) and len(report) == len(roots), "duplicate/missing probe roots")
        require(len({r["root_id"] for r in report}) == len(report), "duplicate report roots")
        n = str(manifest["budgets"][0])
        for row in report:
            require(row["root_id"] in indexed, "report root missing from roots.json")
            root = indexed[row["root_id"]]
            require(str(row["game_id"]) == str(root["game_id"]), "probe report game mismatch")
            if not row["stable"] or row["mate"]:
                continue
            search_path = folder / "search" / (row["root_id"] + ".json")
            search = read(search_path)
            require(search["root"] == root and search["recipe_id"] == manifest["recipe_id"], "search provenance mismatch")
            records.append(dict(folder=str(folder.resolve()), root=root, report=row,
                                short_move=search["budgets"][n]["bestmove"], search_sha256=digest(search_path)))
        hashes.append(dict(folder=str(folder.resolve()), files={name: digest(folder / name)
                           for name in ("roots.json", "report.json", "search_manifest.json")}))
    require(len(records) == expected, f"stable nonmate probe count: {len(records)}, expected {expected}")
    return (records[:limit] if limit else records), hashes


def eval_model(path, baseline, records, dataset, encoder, jax, device, args):
    import numpy as np
    import m2_moves
    ckpt, forward, model = load_checkpoint(path, jax, device, baseline=baseline)

    @jax.jit
    def predict(model, obs):
        return forward.apply(model[0], model[1], obs, is_eval=True)[0]

    if not baseline:
        require(ckpt.get("dataset_id") == dataset.meta["dataset_id"], "candidate trained on a different dataset")
        require(ckpt.get("baseline_sha256") == WS_SHA, "candidate has a different warm start")
    sfens = [r["root"]["sfen"] for r in records]
    predictions = list(policy_predictions(sfens, predict, model, encoder, jax, device, args.batch))
    top1 = top2 = 0
    gate_rows, detail = [], []
    for rec, pred in zip(records, predictions):
        root, report = rec["root"], rec["report"]
        color = int(root["sfen"].split()[1] == "w")
        target = m2_moves.usi_to_action(report["reference_best_move"], color)
        short = m2_moves.usi_to_action(rec["short_move"], color)
        top1 += target == pred["top_actions"][0]
        top2 += target in pred["top_actions"]
        agree = short == pred["top_actions"][0]
        gate_rows.append(dict(ln=report["loss_n"], lu=report["loss_uniform"], l4=report["loss_4n"],
                              ll=report.get("loss_live_4n"), agree=agree,
                              agree3=agree and pred["probability"] >= 0.3))
        detail.append(dict(folder=rec["folder"], root_id=root["root_id"], game_id=root["game_id"], **pred))
    # Stream validation memmap; do not materialize all observations or SFEN strings.
    rows = dataset.rows["val"]
    if args.val_limit:
        rows = rows[:args.val_limit]
    counts, totals = Counter(), Counter()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        sfens = [s.decode("ascii") for s in chunk["sfen"]]
        preds = list(policy_predictions(sfens, predict, model, encoder, jax, device, args.batch,
                                       check=args.check_val_legal or start == 0))
        for row, pred in zip(chunk, preds):
            counts["positions"] += 1
            counts["top1"] += int(pred["top_actions"][0] == row["action"])
            counts["top2"] += int(int(row["action"]) in pred["top_actions"])
            p, z = (pred["value"] + 1) / 2, (int(row["result"]) + 1) / 2
            # Draw outcome=0.5 for Brier; training draws use pure d9 eval.
            totals["brier"] += (p - z) ** 2
            totals["target_mse"] += (pred["value"] - float(row["value"])) ** 2
            label = "draw" if row["result"] == 0 else "decisive"
            counts[label] += 1
            totals[label + "_brier"] += (p - z) ** 2
        if (start // args.batch) % 100 == 0:
            log(dict(evaluating="ws2" if baseline else "wst2", val_positions=counts["positions"]))
    n = counts["positions"]
    heldout = dict(positions=n, top1=counts["top1"] / n, top2=counts["top2"] / n,
                   value_brier=totals["brier"] / n, target_mse=totals["target_mse"] / n,
                   by_result={k: dict(n=counts[k], value_brier=totals[k + "_brier"] / counts[k] if counts[k] else None)
                              for k in ("draw", "decisive")})
    groups = {"combined": gate_rows}
    for folder in args.probes:
        groups[str(folder.resolve())] = [r for rec, r in zip(records, gate_rows) if rec["folder"] == str(folder.resolve())]
    return dict(checkpoint=str(path.resolve()), sha256=digest(path),
                config=ckpt["config"], step=ckpt.get("step"),
                probe=dict(n=len(records), top1=top1 / len(records), top2=top2 / len(records)),
                heldout=heldout, gate={k: gate_summary(v, args.bootstrap) for k, v in groups.items() if v},
                policy=detail)


def evaluate(args):
    _, jax, device = setup(args)
    dataset = Dataset(args.data)
    require(args.allow_subset or not (dataset.meta["subset"] or args.probe_limit or args.val_limit),
            "subset eval needs --allow-subset; cannot report full validation")
    # Reconstruct current exclusions and require every probe game excluded from
    # BOTH training and validation, even if evaluating a separately built dataset.
    paths, keys, probes = exclusions(args.probes)
    require(not (keys & {g["key"] for g in dataset.games}), "probe game in train/validation")
    require(not (paths & {g["source"] for g in dataset.games}), "probe source in train/validation")
    require(keys <= set(dataset.meta["excluded_keys"]), "dataset lacks current probe exclusion audit")
    records, hashes = probe_records(args.probes, limit=args.probe_limit)
    encoder = Encoder(jax)
    results = {name: eval_model(path, name == "ws2", records, dataset, encoder, jax, device, args)
               for name, path in (("ws2", args.baseline), ("wst2", args.checkpoint))}
    for name, result in results.items():
        json_write(args.out / (name + "_policy.json"), result.pop("policy"))
    a, b = results["ws2"], results["wst2"]
    summary = dict(format=FORMAT, subset=bool(dataset.meta["subset"] or args.probe_limit or args.val_limit),
                   dataset_id=dataset.meta["dataset_id"], probe_provenance=hashes,
                   search_provenance=[dict(folder=r["folder"], root_id=r["root"]["root_id"], sha256=r["search_sha256"]) for r in records],
                   bootstrap=args.bootstrap, backend=device.platform, results=results,
                   delta=dict(probe_top1=b["probe"]["top1"] - a["probe"]["top1"],
                              probe_top2=b["probe"]["top2"] - a["probe"]["top2"],
                              heldout_top1=b["heldout"]["top1"] - a["heldout"]["top1"],
                              heldout_brier=b["heldout"]["value_brier"] - a["heldout"]["value_brier"]))
    json_write(args.out / "eval.json", summary)
    log(dict(eval_done=str(args.out / "eval.json"), delta=summary["delta"], subset=summary["subset"]))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd(), help="read-only binbo-shogi source tree")
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    ap.add_argument("--cpu-threads", type=int, default=4)
    sub = ap.add_subparsers(dest="command", required=True)
    probes = [DATA_ROOT / "work/strategy/gate0", DATA_ROOT / "work/strategy/probe13/main"]
    b = sub.add_parser("build", help="compact raw SFEN+targets, game-level exclusion and split")
    b.add_argument("--games", nargs="+", type=Path, default=[DATA_ROOT / f"work/{t}/games" for t in ("task154", "task159")])
    b.add_argument("--probes", nargs="+", type=Path, default=probes)
    b.add_argument("--out", type=Path, default=WORK / "dataset")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--val-fraction", type=float, default=0.02)
    b.add_argument("--max-games", type=int, default=0, help="smoke test only; 0=all")
    b.add_argument("--max-positions-per-game", type=int, default=0, help="smoke test only; 0=all")
    t = sub.add_parser("train", help="warm start identified WS-T; resumable Adam fine tune")
    t.add_argument("--data", type=Path, default=WORK / "dataset")
    t.add_argument("--init", type=Path, default=FRAMEWORKS / "pgx/examples/alphazero/t087_wst/ws2.ckpt")
    t.add_argument("--out", type=Path, default=WORK / "train")
    t.add_argument("--resume", type=Path, help="checkpoint, uses saved optimizer/schedule/batch/seed")
    mode = t.add_mutually_exclusive_group()
    mode.add_argument("--steps", type=int, default=None, help="total optimizer steps")
    mode.add_argument("--epochs", type=float, default=None)
    t.add_argument("--hours", type=float, default=3., help="auto step budget when steps/epochs absent")
    t.add_argument("--batch", default="auto", help="largest tested fitting batch: 32..1024 on GPU, 8 on CPU; or an integer")
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--seed", type=int, default=SEED)
    t.add_argument("--bench-steps", type=int, default=5)
    t.add_argument("--checkpoint-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--stop-after", type=int, default=0, help="stop early at a resumable boundary without changing schedule")
    e = sub.add_parser("eval", help="paired baseline/candidate probe, heldout, gate-0 evaluation")
    e.add_argument("--data", type=Path, default=WORK / "dataset")
    e.add_argument("--baseline", type=Path, default=FRAMEWORKS / "pgx/examples/alphazero/t087_wst/ws2.ckpt")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--probes", nargs="+", type=Path, default=probes)
    e.add_argument("--out", type=Path, default=WORK / "eval")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--bootstrap", type=int, default=2000)
    e.add_argument("--probe-limit", type=int, default=None, help="smoke test only")
    e.add_argument("--val-limit", type=int, default=None, help="smoke test only")
    e.add_argument("--allow-subset", action="store_true")
    e.add_argument("--check-val-legal", action="store_true", help="full pgx/cshogi legal-set equality on all validation rows")
    return ap


def main():
    args = parser().parse_args()
    require(args.cpu_threads > 0, "positive CPU threads required")
    for name in ("max_games", "max_positions_per_game", "bootstrap", "stop_after", "warmup"):
        if hasattr(args, name):
            require(getattr(args, name) >= 0, f"{name} must be >=0")
    for name in ("checkpoint_every", "log_every", "bench_steps", "lr", "hours", "steps", "epochs", "probe_limit", "val_limit"):
        value = getattr(args, name, None)
        require(value is None or value > 0, f"{name} must be positive")
    if hasattr(args, "batch"):
        require(args.batch == "auto" or int(args.batch) > 0, "batch must be auto or positive")
    {"build": build, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
