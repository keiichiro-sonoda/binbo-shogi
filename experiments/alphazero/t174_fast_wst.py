#!/usr/bin/env python3
"""Task 174: measured WS-T2 input acceleration. Requires the pgx/cshogi venv.

Original data/source/checkpoints are read-only. Cache generation is optional:
the default C++ online encoder avoids a preprocessing pass. FP32 is the default.
All output directories must be new, except explicit training resume.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
WORK = ROOT / "work/task174_fast_wst"
ORIGINAL = ROOT / "work/task173_wst2"
FORMAT = "task174-fast-wst-v1"
CACHE_FORMAT = "task174-packed-v1"
old = None


def setup(args):
    global old
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError("--cpu-threads must be 1..4")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(args.repo.resolve() / "experiments/alphazero"))
    old = importlib.import_module("t173_wst2")
    return old.setup(args)


def packed_dtype():
    import numpy as np
    # All spatial arrays are in pgx square order before observation rot90.
    # Pieces replace 28 one-hot planes; effects are the remaining 34 planes.
    return np.dtype([("pieces", "u1", (81,)), ("effects", "u1", (345,)),
                     ("scalars", "u1", (8,)), ("legal", "u1", (274,)),
                     ("action", "<u2"), ("value", "<f4")])


class FastEncoder:
    """cshogi C++ dlshogi features / legal moves, with pgx's exact layout.

    Float32 buffers are mandatory for cshogi's raw-pointer interface. Zero them
    before every position (the C++ routine sets active features only).
    """
    def __init__(self):
        import numpy as np
        import cshogi
        import cshogi.dlshogi as dl
        old.require((dl.FEATURES1_NUM, dl.FEATURES2_NUM) == (62, 57),
                    "unexpected cshogi features (disable nyugyoku features)")
        self.np, self.cshogi, self.dl = np, cshogi, dl
        self.board = cshogi.Board()
        self.f1 = np.zeros((62, 9, 9), np.float32)
        self.f2 = np.zeros((57, 9, 9), np.float32)

    def planes(self, sfens):
        np = self.np
        spatial = np.empty((len(sfens), 62, 81), bool)
        scalars = np.empty((len(sfens), 57), bool)
        mask = np.zeros((len(sfens), 2187), bool)
        for i, sfen in enumerate(sfens):
            self.board.set_sfen(sfen)
            self.f1.fill(0)
            self.f2.fill(0)
            self.dl.make_input_features(self.board, self.f1, self.f2)
            spatial[i] = self.f1.reshape(62, 81)
            scalars[i] = self.f2[:, 0, 0]
            labels = [self.dl.make_move_label(m, self.board.turn) for m in self.board.legal_moves]
            old.require(bool(labels), "terminal root in data")
            mask[i, labels] = True
        return spatial, scalars, mask

    def batch(self, sfens, size=None, check_legal=False):
        np = self.np
        size = size or len(sfens)
        old.require(0 < len(sfens) <= size, "empty/oversize encoding batch")
        spatial, scalars, mask = self.planes(list(sfens) + [sfens[-1]] * (size - len(sfens)))
        planes = np.concatenate((spatial, np.broadcast_to(scalars[:, :, None], (size, 57, 81))), axis=1)
        obs = np.rot90(planes.reshape(size, 119, 9, 9).transpose(0, 2, 3, 1), k=3, axes=(1, 2))
        # An independent action conversion is useful when called by old.eval.
        if check_legal:
            import m2_moves
            for sfen, legal in zip(sfens, mask):
                board = self.cshogi.Board(sfen)
                expected = {m2_moves.usi_to_action(self.cshogi.move_to_usi(m), board.turn) for m in board.legal_moves}
                old.require(expected == set(np.flatnonzero(legal)), "C++/pgx action numbering mismatch")
        return obs, mask

    def pack(self, rows):
        np = self.np
        sfens = [s.decode("ascii") for s in rows["sfen"]]
        spatial, scalars, legal = self.planes(sfens)
        old.require(legal[np.arange(len(rows)), rows["action"]].all(), "illegal policy target")
        packed = np.empty(len(rows), packed_dtype())
        mine, theirs = spatial[:, :14], spatial[:, 31:45]
        packed["pieces"] = ((mine * np.arange(1, 15, dtype=np.uint8)[None, :, None]).sum(axis=1)
                            + (theirs * np.arange(15, 29, dtype=np.uint8)[None, :, None]).sum(axis=1))
        effects = np.concatenate((spatial[:, 14:31], spatial[:, 45:62]), axis=1)
        packed["effects"] = np.packbits(effects.reshape(len(rows), -1), axis=1)
        packed["scalars"] = np.packbits(scalars, axis=1)
        packed["legal"] = np.packbits(legal, axis=1)
        packed["action"], packed["value"] = rows["action"], rows["value"]
        return packed


def host_inputs(rows):
    import numpy as np
    # Contiguous buffers allow genuinely asynchronous device_put on CUDA.
    return tuple(np.ascontiguousarray(rows[n], dtype=np.int32 if n == "action" else None)
                 for n in ("pieces", "effects", "scalars", "legal", "action", "value"))


def unpack_inputs(pieces, effects, scalars, legal, action, value):
    import jax.numpy as jnp
    b = pieces.shape[0]
    def bits(buf, n):
        return (((buf[:, :, None] >> jnp.arange(7, -1, -1, dtype=jnp.uint8)) & 1)
                .reshape(b, -1)[:, :n].astype(jnp.bool_))
    effects = bits(effects, 34 * 81).reshape(b, 34, 81)
    mine = pieces[:, None, :] == jnp.arange(1, 15)[None, :, None]
    theirs = pieces[:, None, :] == jnp.arange(15, 29)[None, :, None]
    scalars = jnp.broadcast_to(bits(scalars, 57)[:, :, None], (b, 57, 81))
    planes = jnp.concatenate((mine, effects[:, :17], theirs, effects[:, 17:], scalars), axis=1)
    obs = jnp.rot90(planes.reshape(b, 119, 9, 9).transpose(0, 2, 3, 1), k=3, axes=(1, 2))
    return obs, bits(legal, 2187), action, value


def precision_policy(compute):
    """Opt-in: FP32 parameters/Adam, bf16 trunk, FP32 loss and normalization.

    Policies act on imported module classes; no network source is changed.
    """
    import haiku as hk
    import jmp
    import jax.numpy as jnp
    from network import TransformerNet, EncoderLayer
    classes = (TransformerNet, EncoderLayer, hk.Linear, hk.MultiHeadAttention, hk.LayerNorm)
    for cls in classes:
        hk.mixed_precision.clear_policy(cls)
    if compute == "bf16":
        for cls in classes:
            hk.mixed_precision.set_policy(cls, jmp.Policy(
                param_dtype=jnp.float32,
                compute_dtype=jnp.float32 if cls is hk.LayerNorm else jnp.bfloat16,
                output_dtype=jnp.float32 if cls is TransformerNet else jnp.bfloat16))


def make_step(forward, jax, donate=True, accumulation=1):
    import jax.numpy as jnp
    import optax
    optimizer = optax.adam(1.)
    def loss_fn(params, state, *packed):
        obs, legal, action, value = unpack_inputs(*packed)
        (logits, pred), state = forward.apply(params, state, obs, is_eval=False)
        logits, pred = logits.astype(jnp.float32), pred.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pol = optax.softmax_cross_entropy_with_integer_labels(logits, action).mean()
        val = ((pred - value) ** 2).mean()
        return pol + val, (state, jnp.stack([pol + val, pol, val]))

    @partial(jax.jit, donate_argnums=(0, 1) if donate else ())
    def step(model, opt_state, *data_rate):
        *packed, rate = data_rate
        if accumulation == 1:
            (_, (state, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model[0], model[1], *packed)
        else:
            # Transformer has no mutable state / dropout / BatchNorm. Averaging
            # equal-sized microbatches is the same objective, with different sums.
            chunked = tuple(x.reshape(accumulation, x.shape[0] // accumulation, *x.shape[1:]) for x in packed)
            zeros = jax.tree_util.tree_map(jnp.zeros_like, model[0])
            def body(carry, data):
                gsum, lsum = carry
                (_, (_, ls)), g = jax.value_and_grad(loss_fn, has_aux=True)(model[0], model[1], *data)
                return (jax.tree_util.tree_map(jnp.add, gsum, g), lsum + ls), None
            (grads, losses), _ = jax.lax.scan(body, (zeros, jnp.zeros(3, jnp.float32)), chunked)
            grads = jax.tree_util.tree_map(lambda x: x / accumulation, grads)
            losses, state = losses / accumulation, model[1]
        updates, opt_state = optimizer.update(grads, opt_state, model[0])
        updates = jax.tree_util.tree_map(lambda u: u * rate, updates)
        return (optax.apply_updates(model[0], updates), state), opt_state, losses
    return optimizer, step


def source_meta(args, jax):
    return dict(script_sha256=old.digest(__file__),
                dependencies={n: old.digest(args.repo / "experiments/alphazero" / n)
                              for n in ("t173_wst2.py", "network.py", "m2_moves.py", "m2_distill_train.py")},
                packages={n: importlib.metadata.version(n) for n in
                          ("jax", "jaxlib", "pgx", "dm-haiku", "optax", "numpy", "cshogi", "jmp")},
                backend=jax.default_backend(), affinity=sorted(os.sched_getaffinity(0)))


def sample_check(rows, jax, device, count=512, seed=174):
    import numpy as np
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), min(count, len(rows)), replace=False))
    enc, reference = FastEncoder(), old.Encoder(jax)
    unpack = jax.jit(unpack_inputs)
    checked = 0
    for start in range(0, len(indices), 64):
        chunk = rows[indices[start:start + 64]]
        sfens = [s.decode() for s in chunk["sfen"]]
        obs, mask = reference.batch(sfens, check_legal=True)
        fast_obs, fast_mask = enc.batch(sfens, check_legal=True)
        packed = enc.pack(chunk)
        decoded = unpack(*tuple(jax.device_put(x, device) for x in host_inputs(packed)))
        old.require(np.array_equal(obs, fast_obs), "C++ observation != t173")
        old.require(np.array_equal(mask, fast_mask), "C++ legal mask != t173")
        for expected, actual in zip((obs, mask, chunk["action"].astype(np.int32), chunk["value"]), decoded):
            old.require(np.array_equal(expected, np.asarray(actual)), "packed observation/mask/target != t173")
        checked += len(chunk)
    # Both colors and multi-digit hands, plus a checked king. These are encoding
    # roots only; they need not occur in the self-play corpus.
    extra = ["4k4/9/9/9/9/9/9/9/4K4 b 18P4L4N4S4G2B2R 1",
             "4k4/9/9/9/9/9/9/9/4K4 w 18p4l4n4s4g2b2r 1",
             "4k4/9/9/9/9/9/4r4/9/4K4 b 18P4L4N4S4G2BR 1"]
    a, m = reference.batch(extra, check_legal=True)
    b, n = enc.batch(extra, check_legal=True)
    old.require(np.array_equal(a, b) and np.array_equal(m, n), "edge-case encoding differs")
    return dict(sample_positions=checked, edge_positions=len(extra), bit_exact=True,
                observations=True, legal_masks=True, targets=True)


class Cache:
    def __init__(self, path, dataset, verify=True):
        import numpy as np
        self.path = Path(path)
        self.meta = old.read(self.path / "meta.json")
        old.require(self.meta["format"] == CACHE_FORMAT, "wrong cache format")
        old.require(self.meta["dataset_id"] == dataset.meta["dataset_id"], "cache dataset mismatch")
        old.require(self.meta["encoder_sources"] == encoder_sources(), "cache encoder/version changed")
        old.require(self.meta["record_bytes"] == packed_dtype().itemsize, "cache dtype changed")
        old.require(not self.meta["subset"], "partial benchmark cache cannot be used for training")
        self.rows = {}
        for split in ("train", "val"):
            p = self.path / (split + ".bin")
            old.require(p.stat().st_size == len(dataset.rows[split]) * packed_dtype().itemsize, "cache size mismatch")
            if verify:
                old.require(old.digest(p) == self.meta["files"][p.name], "cache checksum mismatch")
            self.rows[split] = np.memmap(p, mode="r", dtype=packed_dtype())


def encoder_sources():
    import inspect
    import cshogi.dlshogi as dl
    import cshogi._cshogi as native
    from pgx._src.games import shogi
    # This script also includes training; a source change deliberately invalidates
    # old caches instead of silently trusting a different encoding.
    return {"script": old.digest(__file__), "t173": old.digest(old.__file__),
            "pgx": old.digest(inspect.getfile(shogi)), "cshogi_native": old.digest(native.__file__),
            "cshogi_dlshogi": old.digest(inspect.getfile(dl))}


class Feeder:
    def __init__(self, dataset, jax, device, cache=None):
        self.dataset, self.jax, self.device = dataset, jax, device
        self.cache = cache
        self.encoder = FastEncoder()

    def host(self, indices):
        if self.cache:
            packed = self.cache.rows["train"][indices]
        else:
            packed = self.encoder.pack(self.dataset.rows["train"][indices])
        return host_inputs(packed)

    def get(self, indices):
        return tuple(self.jax.device_put(x, self.device) for x in self.host(indices))


def batches(feeder, sampler, start, stop, depth):
    """One bounded producer. Cursor and ordering are identical to t173.Sampler."""
    if not depth:
        for i in range(start, stop):
            yield i, feeder.get(sampler.indices(i))
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="wst-input") as pool:
        queue = deque()
        cursor = start
        def produce(i):
            return feeder.get(sampler.indices(i))
        while cursor < min(stop, start + depth):
            queue.append((cursor, pool.submit(produce, cursor)))
            cursor += 1
        while queue:
            i, future = queue.popleft()
            data = future.result()
            if cursor < stop:
                queue.append((cursor, pool.submit(produce, cursor)))
                cursor += 1
            yield i, data


def time_call(fn, jax, repeats=5):
    import numpy as np
    t = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    warm = time.perf_counter() - t
    values = []
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        values.append(time.perf_counter() - t)
    return result, dict(seconds=float(np.median(values)), minimum_seconds=min(values),
                        warmup_seconds=warm, repeats=repeats)


def stage_report(fn, jax, batch, repeats):
    result, timing = time_call(fn, jax, repeats)
    timing["positions_per_second"] = batch / timing["seconds"]
    return result, timing


def bench(args):
    np, jax, device = setup(args)
    old.require(not args.out.exists(), "benchmark output exists; choose a new --out")
    args.out.mkdir(parents=True)
    old.atomic(args.out / "measured_trainer.py", Path(__file__).read_bytes())
    # Do not scan/hash 6.8 GB just to time one batch. No such shortcut in train.
    meta = old.read(args.data / "meta.json")
    rows = np.memmap(args.data / "train.bin", mode="r", dtype=old.record_dtype())
    old.require(len(rows) == meta["counts"]["train"], "truncated dataset")
    b = int(args.batch)
    report = dict(format=FORMAT, backend=device.platform, batch=b, compute=args.compute,
                  accumulation=args.accumulation, dataset_id=meta["dataset_id"],
                  benchmark_only_dataset_sha_not_scanned=True, run_meta=source_meta(args, jax))
    report["equality"] = sample_check(rows, jax, device, args.check_sample)
    sample_idx = np.sort(np.random.default_rng(174).choice(len(rows), b, replace=False))
    stages = report["stages"] = {}
    def measure(name, fn, positions=None):
        value, stages[name] = stage_report(fn, jax, positions or b, args.bench_steps)
        old.log(dict(stage=name, **stages[name]))
        return value
    sample = measure("read_batch_warm", lambda: rows[sample_idx])
    # New random rows per repetition: includes demand page faults. OS caches are
    # not cleared on a shared machine. Use this as an observed access-pattern test.
    rng = np.random.default_rng(175)
    random_indices = [np.sort(rng.choice(len(rows), b, replace=False)) for _ in range(args.bench_steps + 1)]
    cursor = iter(random_indices)
    measure("read_batch_changing_rows", lambda: rows[next(cursor)])
    reference = old.Encoder(jax)
    sfens = measure("decode_sfen", lambda: [s.decode("ascii") for s in sample["sfen"]])
    parsed = measure("parse_sfen", lambda: [np.stack(c) for c in zip(*[reference.parse(s) for s in sfens])])
    from pgx._src.games.shogi import GameState, _flip, _observe, _legal_action_mask
    def normalize(c, board, hand):
        state = GameState(color=c, board=board, hand=hand)
        return jax.lax.cond(c % 2 == 1, _flip, lambda s: s, state)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        norm_fn = jax.jit(jax.vmap(normalize))
        normalized = measure("normalize_cpu", lambda: norm_fn(*parsed))
        legal_fn = jax.jit(jax.vmap(_legal_action_mask))
        obs_fn = jax.jit(jax.vmap(lambda s: _observe(s, flip=True)))
        measure("legal_mask_pgx_cpu", lambda: legal_fn(normalized))
        measure("observation_pgx_cpu", lambda: obs_fn(normalized))
    original_obs, original_mask = measure("t173_encoding_fused_cpu", lambda: reference.batch(sfens))
    encoder = FastEncoder()
    measure("cshogi_observation_and_legal", lambda: encoder.batch(sfens))
    packed = measure("cshogi_compact_encode", lambda: encoder.pack(sample))
    inputs_host = measure("compact_host_buffers", lambda: host_inputs(packed))
    inputs = measure("transfer_compact", lambda: tuple(jax.device_put(x, device) for x in inputs_host))
    measure("transfer_original", lambda: tuple(jax.device_put(x, device) for x in
            (original_obs, original_mask, sample["action"].astype(np.int32), sample["value"])))
    measure("unpack_on_device", lambda: jax.jit(unpack_inputs)(*inputs))
    measure("compact_input_end_to_end", lambda: tuple(jax.device_put(x, device) for x in host_inputs(encoder.pack(sample))))
    # Compact memmap, not only an in-RAM buffer. This is explicitly a small sample.
    packed.tofile(args.out / "sample_packed.bin")
    cached = np.memmap(args.out / "sample_packed.bin", mode="r", dtype=packed_dtype())
    measure("cache_read_and_transfer_warm", lambda: tuple(jax.device_put(x, device) for x in host_inputs(cached[np.arange(b)])))
    ckpt, forward, model = old.load_checkpoint(args.init, jax, device, baseline=True)
    precision_policy(args.compute)
    import jax.numpy as jnp
    import optax
    optimizer, step = make_step(forward, jax, donate=False, accumulation=args.accumulation)
    # Separate F/B and Adam measurements are diagnostic; their sum is not the
    # fused step time. Standalone Adam receives already resident gradients.
    def loss(params, state, *data):
        obs, mask, action, value = unpack_inputs(*data)
        (logits, pred), _ = forward.apply(params, state, obs, is_eval=False)
        logits, pred = logits.astype(jnp.float32), pred.astype(jnp.float32)
        logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
        return optax.softmax_cross_entropy_with_integer_labels(logits, action).mean() + ((pred - value) ** 2).mean()
    grads_fn = jax.jit(jax.value_and_grad(loss))
    micro = b // args.accumulation
    grad_inputs = tuple(x[:micro] for x in inputs)
    grad_stage = "forward_backward" if args.accumulation == 1 else "forward_backward_microbatch"
    _, grads = measure(grad_stage, lambda: grads_fn(model[0], model[1], *grad_inputs), positions=micro)
    report["forward_backward_profile_batch"] = micro
    opt = jax.device_put(optimizer.init(model[0]), device)
    rate = jax.device_put(np.float32(args.lr), device)
    @jax.jit
    def update(params, opt_state, grad):
        u, state = optimizer.update(grad, opt_state, params)
        u = jax.tree_util.tree_map(lambda x: x * rate, u)
        return optax.apply_updates(params, u), state
    measure("optimizer", lambda: update(model[0], opt, grads))
    mm, oo = model, opt
    def fused():
        nonlocal mm, oo
        mm, oo, losses = step(mm, oo, *inputs, rate)
        return mm, oo, losses
    measure("fused_train_resident", fused)
    def end_to_end(kind):
        nonlocal mm, oo
        if kind == "t173":
            data = old.prepare(sample, reference, jax, device)
            mm, oo, ls = old_step(mm, oo, *data, rate)
        else:
            data = tuple(jax.device_put(x, device) for x in host_inputs(encoder.pack(sample)))
            mm, oo, ls = step(mm, oo, *data, rate)
        return mm, oo, ls
    if args.compute == "fp32" and args.accumulation == 1:
        _, old_step = old.train_step_fn(forward, jax)
        measure("t173_end_to_end_warm_rows", lambda: end_to_end("t173"))
    measure("t174_end_to_end_sync_warm_rows", lambda: end_to_end("fast"))
    # Real ordered prefetch with distinct random data each step; includes initial
    # permutation construction separately to avoid hiding a 32M-row permutation.
    dataset_stub = argparse.Namespace(rows={"train": rows})
    feeder = Feeder(dataset_stub, jax, device)
    sampler = old.Sampler(len(rows), b, old.SEED)
    t = time.perf_counter()
    sampler.indices(0)
    report["sampler_initial_seconds"] = time.perf_counter() - t
    # Warm donation's signatures on independent copies; benchmark does not mutate
    # the identified baseline. Keep FP32 rate strongly typed from the first call.
    _, donated = make_step(forward, jax, donate=True, accumulation=args.accumulation)
    dm = jax.tree_util.tree_map(lambda x: jax.device_put(np.array(x), device), model)
    do = jax.device_put(optimizer.init(dm[0]), device)
    for _ in range(2):
        dm, do, ls = donated(dm, do, *inputs, rate)
        jax.block_until_ready((dm, do, ls))
    t = time.perf_counter()
    for i, data in batches(feeder, sampler, 0, args.bench_steps, args.prefetch):
        dm, do, ls = donated(dm, do, *data, rate)
    jax.block_until_ready((dm, do, ls))
    elapsed = time.perf_counter() - t
    stages["t174_pipeline_changing_rows"] = dict(seconds=elapsed / args.bench_steps,
        total_seconds=elapsed, positions_per_second=b * args.bench_steps / elapsed, steps=args.bench_steps)
    report["packed_bytes_per_position"] = packed_dtype().itemsize
    report["original_transfer_bytes_per_position"] = 9 * 9 * 119 + 2187 + 8
    report["gpu_inference"] = gpu_estimate(args.throughput, args.history, b, stages,
                                             report["sampler_initial_seconds"])
    old.json_write(args.out / "bench.json", report)
    old.log(dict(bench_done=str(args.out / "bench.json"), pipeline=stages["t174_pipeline_changing_rows"],
                 gpu_inference=report["gpu_inference"]))


def gpu_estimate(path, history_path, batch, stages, sampler_seconds):
    if not path.is_file():
        return None
    doc = old.read(path)
    matching = next((r for r in doc["measurements"] if r["batch"] == batch), None)
    if not matching:
        return None
    import numpy as np
    total = matching["seconds_per_step"]
    host = sum(stages[k]["seconds"] for k in ("t173_encoding_fused_cpu", "decode_sfen"))
    report = dict(measured_gpu_calibration_seconds=total,
                  calibration_includes_initial_epoch_permutation=True,
                  current_cpu_initial_permutation_seconds=sampler_seconds,
                  current_cpu_permutation_amortized_over_old_bench=sampler_seconds / matching["steps"],
                  calibration_minus_encoding_seconds=total - host,
                  caveat="Separate CPU/GPU runs, backend/load/page-cache differences. Residual includes H2D/sync and is not measured GPU kernel time. Initial permutation is inside old 5-step timer. bf16 speed is unmeasured.")
    if not history_path.is_file() or batch != doc["selected"]["batch"]:
        return report
    history = [json.loads(s) for s in history_path.read_text().splitlines() if s.strip()]
    intervals = [(b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                 for a, b in zip(history, history[1:]) if b["step"] > a["step"] and b["seconds"] > a["seconds"]]
    if not intervals:
        return report
    steady = float(np.median(intervals))
    warm_read = stages["read_batch_warm"]["seconds"]
    cold_read = stages["read_batch_changing_rows"]["seconds"]
    residual = steady - host - warm_read
    fast = stages["compact_input_end_to_end"]["seconds"]
    report.update(history_median_seconds_per_step=steady,
        history_median_positions_per_second=batch / steady,
        history_final_step=history[-1]["step"], history_final_elapsed_seconds=history[-1]["seconds"],
        measured_cpu_old_encoding_seconds=host, measured_cpu_changing_read_seconds=cold_read,
        required_seconds_for_10x_calibration=total / 10,
        required_seconds_for_10x_history=steady / 10,
        warm_read_assumption_gpu_plus_unaccounted_seconds=residual,
        changing_read_assumption_residual_seconds=steady - host - cold_read)
    if residual > 0:
        report.update(conditional_fp32_sequential_positions_per_second=batch / (residual + fast + warm_read),
            conditional_fp32_prefetch_positions_per_second=batch / max(residual, fast + warm_read),
            conditional_fp32_prefetch_with_changing_reads_positions_per_second=batch / max(residual, fast + cold_read),
            conditional_warm_input_only_max_speedup_vs_history=steady / residual,
            conditional_warm_input_only_max_speedup_vs_calibration=total / residual)
    return report


def encode(args):
    np, jax, device = setup(args)
    old.require(not args.out.exists(), "cache output exists; choose a new --out")
    dataset = old.Dataset(args.data)
    equality = sample_check(dataset.rows["train"], jax, device, args.check_sample)
    counts = {s: min(len(r), args.limit) if args.limit else len(r) for s, r in dataset.rows.items()}
    import shutil
    needed = sum(counts.values()) * packed_dtype().itemsize
    old.require(shutil.disk_usage(args.out.parent).free > needed + (1 << 30), "insufficient disk space (keep 1 GiB free)")
    args.out.mkdir(parents=True)
    encoder, files, timings = FastEncoder(), {}, {}
    for split in ("train", "val"):
        path = args.out / (split + ".bin")
        t = time.perf_counter()
        with path.open("xb") as f:
            for start in range(0, counts[split], args.batch):
                encoder.pack(dataset.rows[split][start:min(start + args.batch, counts[split])]).tofile(f)
                if start % (args.batch * 100) == 0:
                    old.log(dict(encoding=split, positions=start, total=counts[split]))
            f.flush()
            os.fsync(f.fileno())
        timings[split] = dict(seconds=time.perf_counter() - t, positions=counts[split])
        timings[split]["positions_per_second"] = counts[split] / timings[split]["seconds"]
        files[path.name] = old.digest(path)
    # Meta is a completion marker; a failed partial cache cannot be consumed.
    old.json_write(args.out / "meta.json", dict(format=CACHE_FORMAT, dataset_id=dataset.meta["dataset_id"],
        record_bytes=packed_dtype().itemsize, encoder_sources=encoder_sources(), counts=counts,
        subset=any(counts[s] != len(dataset.rows[s]) for s in counts),
        files=files, timing=timings, equality=equality))
    old.log(dict(cache_done=str(args.out), timings=timings))


def calibrate(args, feeder, rows, model, forward, jax, device):
    import numpy as np
    candidates = ([32, 64, 128, 256] if device.platform == "gpu" else [8]) if args.batch == "auto" else [int(args.batch)]
    measurements = []
    optimizer, step = make_step(forward, jax, donate=not args.no_donate, accumulation=args.accumulation)
    for batch in candidates:
        old.require(batch % args.accumulation == 0, "batch must be divisible by accumulation")
        try:
            # Donation must never consume the baseline used for the real run.
            mm = jax.tree_util.tree_map(lambda x: jax.device_put(np.array(x), device), model)
            oo = jax.device_put(optimizer.init(mm[0]), device)
            sampler = old.Sampler(len(rows), batch, args.seed)
            t = time.perf_counter()
            sampler.indices(0)
            sampler_seconds = time.perf_counter() - t
            data = feeder.get(np.arange(batch) % len(rows))
            rate = jax.device_put(np.float32(args.lr), device)
            t = time.perf_counter()
            for _ in range(2):
                mm, oo, ls = step(mm, oo, *data, rate)
                jax.block_until_ready((mm, oo, ls))
            compile_seconds = time.perf_counter() - t
            t = time.perf_counter()
            for _, data in batches(feeder, sampler, 0, args.bench_steps, args.prefetch):
                mm, oo, ls = step(mm, oo, *data, rate)
            jax.block_until_ready((mm, oo, ls))
            elapsed = time.perf_counter() - t
            result = dict(batch=batch, seconds_per_step=elapsed / args.bench_steps,
                          positions_per_second=batch * args.bench_steps / elapsed,
                          compile_seconds=compile_seconds, initial_sampler_seconds=sampler_seconds)
            measurements.append(result)
            old.log(dict(benchmark=result))
            del mm, oo, data
        except Exception as exc:
            if args.batch != "auto" or not any(s in str(exc).lower() for s in ("out of memory", "resource_exhausted")):
                raise
            old.log(dict(batch_failed=batch, error=str(exc)[:300]))
            break
    old.require(bool(measurements), "no fitting batch")
    # Fastest measured throughput, rather than largest fitting batch.
    return max(measurements, key=lambda r: r["positions_per_second"]), measurements, optimizer, step


def train(args):
    np, jax, device = setup(args)
    old.require(args.prefetch >= 0, "prefetch must be nonnegative")
    dataset = old.Dataset(args.data)
    out = args.out.resolve()
    checkpoint, forward, model = old.load_checkpoint(args.resume or args.init, jax, device, baseline=not args.resume)
    run_meta = source_meta(args, jax)
    if args.resume:
        old.require(checkpoint.get("format") == FORMAT, "resume requires a Task 174 checkpoint; --init is the identified WS-T")
        old.require(checkpoint["dataset_id"] == dataset.meta["dataset_id"], "resume dataset changed")
        old.require(args.steps is None and args.epochs is None, "resume uses saved schedule")
        settings, bench = checkpoint["train_settings"], checkpoint["benchmark"]
        for key in ("script_sha256", "dependencies", "packages", "backend"):
            old.require(checkpoint["run_meta"][key] == run_meta[key], f"resume {key} changed")
        args.compute, args.accumulation = settings["compute"], settings["accumulation"]
        step_num = checkpoint["step"]
        old.require(out.exists(), "resume output does not exist")
    else:
        old.require(not out.exists(), "training output exists; use --resume or new --out")
        step_num = 0
    precision_policy(args.compute)
    if args.accumulation > 1:
        old.require(not model[1], "gradient accumulation requires stateless Transformer")
    equality = sample_check(dataset.rows["train"], jax, device, args.check_sample)
    cache = Cache(args.cache, dataset) if args.cache else None
    feeder = Feeder(dataset, jax, device, cache)
    if not args.resume:
        started = time.perf_counter()
        best, measurements, optimizer, step_fn = calibrate(args, feeder, dataset.rows["train"], model, forward, jax, device)
        batch = best["batch"]
        total = args.steps or (math.ceil(len(dataset.rows["train"]) * args.epochs / batch) if args.epochs else
            max(1, int(0.9 * max(0., args.hours * 3600 - (time.perf_counter() - started)) / best["seconds_per_step"])))
        settings = dict(batch=batch, total_steps=total, seed=args.seed, lr=args.lr,
                        warmup=min(args.warmup, max(0, total // 10)), compute=args.compute,
                        accumulation=args.accumulation, schedule="linear warmup then cosine to 0.1*lr",
                        sampling="t173 seeded epoch permutation; sorted batch; wrap tail")
        bench = dict(device=str(device), backend=device.platform, selected=best, measurements=measurements,
                     scope="C++ encoding or compact cache + H2D + in-jit unpack + forward/backward/Adam; bounded prefetch; no JIT",
                     estimated_training_seconds=total * best["seconds_per_step"], equality=equality,
                     cache=str(args.cache) if args.cache else None)
        out.mkdir(parents=True)
        old.json_write(out / "throughput.json", bench)
    if args.resume:
        optimizer, step_fn = make_step(forward, jax, donate=not args.no_donate, accumulation=settings["accumulation"])
    # Adam's count starts as an uncommitted scalar by default. The first update
    # commits it, changing the JIT signature and causing a second compilation.
    # Commit the entire initial state and reuse calibration's compiled function.
    opt_state = jax.device_put(checkpoint["opt_state"] if args.resume else optimizer.init(model[0]), device)
    sampler = old.Sampler(len(dataset.rows["train"]), settings["batch"], settings["seed"])
    stop = min(settings["total_steps"], step_num + args.stop_after) if args.stop_after else settings["total_steps"]
    old.log(dict(training=settings, positions=len(dataset.rows["train"]), start_step=step_num, equality=equality))
    started = time.perf_counter()
    last_path = args.resume
    def save(current):
        blob = dict(format=FORMAT, kind="WS-T2 accelerated fine tune", config=checkpoint["config"],
                    model=jax.device_get(model), opt_state=jax.device_get(opt_state), step=current,
                    iteration=0, rng_key=np.asarray(jax.random.PRNGKey(settings["seed"])),
                    train_settings=settings, dataset_id=dataset.meta["dataset_id"], benchmark=bench,
                    baseline_sha256=old.WS_SHA, run_meta=run_meta,
                    numerical_mode=dict(compute=args.compute, accumulation=args.accumulation))
        path = out / f"step_{current:09d}.ckpt"
        old.atomic(path, pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))
        old.json_write(out / "latest.json", dict(checkpoint=path.name, step=current, sha256=old.digest(path)))
        return path
    with (out / "history.jsonl").open("a") as history:
        import jax.numpy as jnp
        finite_fn = jax.jit(lambda ls: jnp.isfinite(ls).all())
        combine_finite = jax.jit(jnp.logical_and)
        finite = jax.device_put(np.bool_(True), device)
        for current, data in batches(feeder, sampler, step_num, stop, args.prefetch):
            rate = old.learning_rate(current, settings)
            rate_device = jax.device_put(np.float32(rate), device)
            model, opt_state, losses = step_fn(model, opt_state, *data, rate_device)
            # Only a scalar chain is kept; no per-step device->host synchronization.
            finite = combine_finite(finite, finite_fn(losses))
            logging = (current + 1) % args.log_every == 0 or current == step_num or current + 1 == stop
            saving = (current + 1) % args.checkpoint_every == 0 or current + 1 == stop
            if logging or saving:
                old.require(bool(np.asarray(finite)), "nonfinite loss; no checkpoint published")
            if logging:
                loss = np.asarray(losses)
                entry = dict(step=current + 1, loss=float(loss[0]), policy_ce=float(loss[1]),
                             value_mse=float(loss[2]), lr=rate, seconds=time.perf_counter() - started)
                history.write(old.canonical(entry) + "\n")
                history.flush()
                old.log(entry)
            if saving:
                last_path = save(current + 1)
    old.log(dict(train_done=str(last_path), step=stop, schedule_complete=stop == settings["total_steps"],
                 compiled_signatures=step_fn._cache_size()))


def evaluate(args):
    # Keep t173's full provenance, exclusion audit, paired metrics and gate rule.
    # Inference defaults to FP32, including for weights learned in bf16 mode.
    setup(args)
    precision_policy("fp32")
    original_encoder = old.Encoder
    try:
        old.Encoder = lambda jax: FastEncoder()
        old.evaluate(args)
    finally:
        old.Encoder = original_encoder


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--device", choices=("cpu", "gpu", "auto"), default="auto")
    ap.add_argument("--cpu-threads", type=int, default=4)
    sub = ap.add_subparsers(dest="command", required=True)
    # Follow t173's configurable PGX_HOME convention rather than machine paths.
    baseline = Path(os.environ.get("PGX_HOME", Path.home() / "task074_frameworks")) / "pgx/examples/alphazero/t087_wst/ws2.ckpt"
    def common(p):
        p.add_argument("--data", type=Path, default=ORIGINAL / "dataset")
        p.add_argument("--out", type=Path, default=WORK / p.prog.split()[-1])
    def training_options(p):
        p.add_argument("--init", type=Path, default=baseline)
        p.add_argument("--batch", default="auto")
        p.add_argument("--compute", choices=("fp32", "bf16"), default="fp32")
        p.add_argument("--accumulation", type=int, default=1, help="equal microbatches per optimizer update; --batch is effective batch")
        p.add_argument("--prefetch", type=int, default=2)
        p.add_argument("--bench-steps", type=int, default=5)
        p.add_argument("--lr", type=float, default=1e-5)
        p.add_argument("--check-sample", type=int, default=512)
    b = sub.add_parser("bench", help="bit equality and synchronized per-stage / pipeline profiling")
    common(b)
    training_options(b)
    b.set_defaults(batch="256")
    b.add_argument("--throughput", type=Path, default=ORIGINAL / "train/throughput.json")
    b.add_argument("--history", type=Path, default=ORIGINAL / "train/history.jsonl")
    e = sub.add_parser("encode", help="optional 714-byte/position cache (completion marker and checksums)")
    common(e)
    e.add_argument("--batch", type=int, default=1024)
    e.add_argument("--limit", type=int, default=0, help="per-split prefix for timing only; rejected by train")
    e.add_argument("--check-sample", type=int, default=512)
    t = sub.add_parser("train", help="same WS-T2 schedule/targets/sampler; bounded input prefetch")
    common(t)
    training_options(t)
    t.add_argument("--cache", type=Path)
    t.add_argument("--resume", type=Path)
    mode = t.add_mutually_exclusive_group()
    mode.add_argument("--steps", type=int)
    mode.add_argument("--epochs", type=float)
    t.add_argument("--hours", type=float, default=3.)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--seed", type=int, default=20271730)
    t.add_argument("--checkpoint-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--stop-after", type=int, default=0)
    t.add_argument("--no-donate", action="store_true")
    ev = sub.add_parser("eval", help="t173 paired FP32 evaluation with fast C++ features")
    common(ev)
    ev.add_argument("--baseline", type=Path, default=baseline)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--probes", type=Path, nargs="+", default=[ROOT / "work/strategy/gate0", ROOT / "work/strategy/probe13/main"])
    ev.add_argument("--batch", type=int, default=64)
    ev.add_argument("--bootstrap", type=int, default=2000)
    ev.add_argument("--probe-limit", type=int)
    ev.add_argument("--val-limit", type=int)
    ev.add_argument("--allow-subset", action="store_true")
    ev.add_argument("--check-val-legal", action="store_true")
    return ap


def main():
    args = parser().parse_args()
    for name in ("bench_steps", "check_sample", "accumulation", "lr", "hours", "steps", "epochs", "checkpoint_every", "log_every", "probe_limit", "val_limit"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("stop_after", "warmup", "limit", "bootstrap", "prefetch"):
        if getattr(args, name, 0) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if hasattr(args, "batch") and args.batch != "auto":
        if int(args.batch) <= 0 or int(args.batch) % getattr(args, "accumulation", 1):
            raise ValueError("batch must be positive and divisible by accumulation")
    if args.command == "bench" and args.batch == "auto":
        raise ValueError("bench requires an integer --batch")
    {"bench": bench, "encode": encode, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
