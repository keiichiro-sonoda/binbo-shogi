#!/usr/bin/env python3
"""Task 167 probes 1/3. Source scripts/env.sh; see reports/task167_probe_tools.md.

All centipawns below are displayed USI cp (not Task 153 internal Value).
Recommended order: sample, policy, search, report. Search may precede policy;
rerun search afterwards to fill the missing policy-child evaluations.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import re
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
PGX_HOME = Path(os.environ.get("PGX_HOME", Path.home() / "task074_frameworks"))
FORMAT = "t167-probe13-v1"
CKPT_SHA = "20bf5e826e307cf191078f1cb86bf2c6f7bf79ef6e89bb2d40829844950fe2b3"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def identity(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value, text=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(value if text else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        Path(tmp).unlink(missing_ok=True)


def compatible(path, obj):
    if path.exists():
        require(read(path) == obj, f"configuration changed: use another --work ({path})")
    else:
        atomic(path, obj)


def roots_at(work):
    obj = read(work / "roots.json")
    require(obj["format"] == FORMAT and obj["roots"], "empty/incompatible roots")
    require(len({r["game_id"] for r in obj["roots"]}) == len(obj["roots"]), "game holdout violated")
    return obj


def part_path(work, root):
    return work / "search" / (root["root_id"] + ".json")


def game_rows(path):
    # Actual Task 159 raw games use Task 153 JSON, not its derived QS npz.
    if path.suffix == ".json":
        game = read(path)
        require(game["format"] == "task153-selfplay-v1", f"not a raw game: {path}")
        if "content_sha256" in game:
            require(game["content_sha256"] == identity({k: v for k, v in game.items()
                                                      if k != "content_sha256"}), f"bad checksum: {path}")
        return str(game["game_id"]), game["rows"]
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        # Only explicit raw SFEN columns are accepted; QS features aren't roots.
        keys = {k: next((x for x in choices if x in z), None) for k, choices in
                dict(sfen=("sfen", "sfens"), score=("score", "scores"), ply=("ply", "plies")).items()}
        require(all(keys.values()) and "game_id" in z, f"npz lacks raw SFEN/score/ply/game_id: {path}")
        gid = z["game_id"].reshape(-1)
        require(len(set(map(str, gid))) == 1, f"multiple games in {path}")
        cols = {k: z[v] for k, v in keys.items()}
        require(len({len(v) for v in cols.values()}) == 1, f"column lengths: {path}")
        return str(gid[0]), [dict(sfen=(s.decode() if isinstance(s, bytes) else str(s)),
                                score=int(v), ply=int(p)) for s, v, p in
                               zip(cols["sfen"], cols["score"], cols["ply"])]


def sample(args):
    import cshogi
    require(not (args.work / "roots.json").exists(), "roots.json already exists; reuse it or use another --work")
    files = sorted(args.games_dir.glob("*.json")) + sorted(args.games_dir.glob("*.npz"))
    require(files, f"no games: {args.games_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    selected, seen, scanned = [], set(), 0
    for path in files:
        scanned += 1
        gid, rows = game_rows(path)
        if gid in seen:
            continue
        seen.add(gid)
        candidates = []
        for row in rows:
            if args.min_ply <= row["ply"] <= args.max_ply and abs(row["score"]) < args.max_score:
                board = cshogi.Board(row["sfen"])
                if not board.is_check() and list(board.legal_moves) and not board.is_nyugyoku():
                    candidates.append(row)
        if not candidates:
            continue
        row = rng.choice(candidates)
        selected.append(dict(root_id=f"r{len(selected):04d}", game_id=gid, sfen=row["sfen"],
                             ply=row["ply"], source_internal_score=row["score"],
                             source=str(path.resolve()), source_sha256=digest(path)))
        if len(selected) == args.count:
            break
    require(len(selected) == args.count, f"only {len(selected)} eligible games")
    manifest = args.games_dir.parent / "manifest.json"
    atomic(args.work / "roots.json", dict(format=FORMAT, seed=args.seed,
           sampling="shuffle games, first eligible games, one uniform eligible row per game; no judgement outcomes used",
           filters=dict(min_ply=args.min_ply, max_ply=args.max_ply, abs_internal_score_lt=args.max_score,
                        not_in_check=True, no_declaration=True),
           source_manifest_sha256=digest(manifest) if manifest.exists() else None,
           files_scanned=scanned, roots=selected))
    print(f"sample: {len(selected)} roots / distinct games; scanned={scanned}", flush=True)


class Engine:
    def __init__(self, cfg):
        # Reuse the locally verified binary-pipe USI transport, timeout and cleanup.
        import t146_self_relabel as old
        self.inner = old.Engine(dict(engine=cfg["engine"], options=cfg["options"], timeout=cfg["timeout"]))
        self.cfg = cfg

    def close(self):
        self.inner.close()

    def search(self, sfen, nodes, multipv=1, reset=True):
        import cshogi
        eng = self.inner
        eng.send(f"setoption name MultiPV value {multipv}")
        # Every root-budget AND child-budget is cold: no order-dependent TT advantage.
        if reset:
            eng.reset()
        board = cshogi.Board(sfen)
        legal = {cshogi.move_to_usi(m) for m in board.legal_moves}
        if not legal or board.is_nyugyoku():
            cp = -3000 if not legal else 3000
            return dict(bestmove="resign" if not legal else "win", score_cp=cp, mate=True,
                        raw_score="terminal", depth=0, nodes=0, requested_nodes=nodes, pv=[],
                        iterations=[], multipv={}, gap_cp=None, terminal=True, seconds=0)
        started = time.monotonic()
        eng.send("position sfen " + sfen)
        eng.send(f"go nodes {nodes}")
        deadline, latest, iterations = time.monotonic() + self.cfg["timeout"], {}, {}
        actual_nodes, final_info = 0, None
        while True:
            line = eng.readline(deadline)
            if line.startswith("bestmove "):
                best = line.split()[1]
                require(best in legal, f"illegal/terminal bestmove at nonterminal root: {best}: {sfen}")
                # A node stop can return a new bestmove having only a bound.
                # Retain the last exact completed-iteration value, just like
                # Task 146, and preserve the final bound/PV separately for audit.
                require(iterations, f"no exact score: {list(eng.lines)}")
                result = dict(max(iterations.values(), key=lambda x: x["depth"]))
                pair = [(d, x[1], x[2]) for d, x in latest.items() if 1 in x and 2 in x]
                gap = None
                if pair:
                    _, a, b = max(pair, key=lambda x: x[0])
                    gap = a["score_cp"] - b["score_cp"]
                result.update(bestmove=best, requested_nodes=nodes, score_nodes=result["nodes"], nodes=actual_nodes,
                              score_matches_bestmove=result["pv"][0] == best, final_info=final_info,
                              iterations=sorted(iterations.values(), key=lambda x: x["depth"]),
                              multipv={str(d): {str(k): v for k, v in x.items()} for d, x in latest.items()},
                              gap_cp=gap, seconds=time.monotonic() - started)
                return result
            tok = line.split()
            if line.startswith("info ") and "nodes" in tok and "string" not in tok:
                actual_nodes = max(actual_nodes, int(tok[tok.index("nodes") + 1]))
            if not line.startswith("info ") or "string" in tok or "pv" not in tok:
                continue
            match = re.search(r"\bscore (cp|mate) ([+-]?\d+|[+-])(?=\s|$)", line)
            if match and ("multipv" not in tok or tok[tok.index("multipv") + 1] == "1"):
                final_info = line
            if not match or {"upperbound", "lowerbound"}.intersection(tok):
                continue
            kind, raw = match.groups()
            cp = (-3000 if raw.startswith("-") else 3000) if kind == "mate" else int(raw)
            def field(key, default=0):
                return int(tok[tok.index(key) + 1]) if key in tok else default
            pv = tok[tok.index("pv") + 1:]
            if not pv:
                continue
            require(pv[0] in legal, f"illegal PV: {line}")
            item = dict(score_cp=cp, mate=kind == "mate", raw_score=f"{kind} {raw}",
                        depth=field("depth"), nodes=field("nodes"), pv=pv)
            rank = field("multipv", 1)
            latest.setdefault(item["depth"], {})[rank] = item
            if rank == 1:
                iterations[item["depth"]] = item


_ENGINE = _CFG = None


def init_worker(cfg, parent):
    global _ENGINE, _CFG
    import t146_self_relabel as old
    old.parent_death(parent)
    _CFG, _ENGINE = cfg, None
    atexit.register(close_worker)


def close_worker():
    if _ENGINE is not None:
        _ENGINE.close()


def search_job(job):
    global _ENGINE
    import cshogi
    root, work, policy = job
    path = part_path(work, root)
    part = read(path) if path.exists() else dict(format=FORMAT, recipe_id=_CFG["recipe_id"],
            root=root, budgets={}, multipv_n=None, children={})
    require(part["recipe_id"] == _CFG["recipe_id"] and part["root"] == root, f"incompatible {path}")
    def run(sfen, nodes, multipv=1, prefix_nodes=None):
        global _ENGINE
        for attempt in range(2):
            try:
                if _ENGINE is None:
                    _ENGINE = Engine(_CFG)
                if prefix_nodes is None:
                    return _ENGINE.search(sfen, nodes, multipv)
                prefix = _ENGINE.search(sfen, prefix_nodes)
                result = _ENGINE.search(sfen, nodes, reset=False)
                result.update(prefix=prefix, allocation_requested_nodes=prefix_nodes + nodes)
                return result
            except (RuntimeError, OSError, TimeoutError):
                if _ENGINE is not None:
                    _ENGINE.close()
                _ENGINE = None
                if attempt:
                    raise
    for n in _CFG["budgets"]:
        if str(n) not in part["budgets"]:
            part["budgets"][str(n)] = run(root["sfen"], n)
            atomic(path, part)
    if part["multipv_n"] is None:
        part["multipv_n"] = run(root["sfen"], _CFG["budgets"][0], 2)
        atomic(path, part)
    if "continued_4n" not in part:
        # A deployable allocation pays for its first N nodes on EVERY root.
        # Re-searching promoted roots cold at 4N would actually cost 1.8N/root.
        # Measure N + warm-TT 3N as a separate, honestly charged 4N outcome.
        n = _CFG["budgets"][0]
        part["continued_4n"] = run(root["sfen"], 3 * n, prefix_nodes=n)
        require(part["continued_4n"]["prefix"]["bestmove"] == part["budgets"][str(n)]["bestmove"],
                "cold N prefix is not reproducible")
        atomic(path, part)
    moves = {r["bestmove"] for r in part["budgets"].values()}
    moves.add(part["continued_4n"]["bestmove"])
    if policy is not None:
        moves.update(x["move"] for x in policy["top_moves"][:2])
    board = cshogi.Board(root["sfen"])
    legal = {cshogi.move_to_usi(m): m for m in board.legal_moves}
    for move in sorted(moves):
        require(move in legal, f"illegal candidate {move}")
        child = cshogi.Board(root["sfen"])
        child.push(legal[move])
        for n in _CFG["budgets"][-2:]:
            record = part["children"].setdefault(move, {})
            if str(n) not in record:
                record[str(n)] = run(child.sfen(), n)
                atomic(path, part)
    return root["root_id"], len(part["children"])


def search(args):
    import t146_self_relabel as old
    require(args.procs > 0, "--procs must be positive")
    roots = roots_at(args.work)
    require(not (args.eval_dir / "eval_options.txt").exists(), "eval_options.txt overrides fixed experiment options")
    budgets = [args.nodes, round(1.6 * args.nodes), 4 * args.nodes, 20 * args.nodes, 40 * args.nodes]
    require(args.nodes >= 5, "N must be >=5")
    options = dict(old.OPTIONS, Threads=1, USI_Hash=64, EvalDir=str(args.eval_dir.resolve()))
    cfg = dict(engine=str(args.engine.resolve()), engine_sha256=digest(args.engine),
               eval_sha256=digest(args.eval_dir / "nn.bin"), options=options, budgets=budgets,
               roots_sha256=identity(roots), script_sha256=digest(__file__),
               transport_sha256=digest(old.__file__),
               score_unit="USI displayed cp; mate +/-3000",
               reset="cold usinewgame+isready per search except continued_4n's explicit N+warm3N pair")
    cfg["recipe_id"] = identity(cfg)
    compatible(args.work / "search_manifest.json", cfg)
    policy_path = args.work / "policy.json"
    policy = read(policy_path) if policy_path.exists() else None
    if policy:
        require(policy["roots_sha256"] == identity(roots), "policy roots mismatch")
    else:
        print("search: policy missing; rerun search after policy for its child evaluations", flush=True)
    runtime = dict(cfg, timeout=args.timeout)
    jobs = [(r, args.work, policy["roots"][r["root_id"]] if policy else None) for r in roots["roots"]]
    # Spawn avoids inheriting JAX/native-library threads; one private engine per worker.
    with mp.get_context("spawn").Pool(args.procs, initializer=init_worker,
                                       initargs=(runtime, os.getpid())) as pool:
        for i, (rid, nchild) in enumerate(pool.imap_unordered(search_job, jobs), 1):
            print(f"search: {i}/{len(jobs)} {rid} child_moves={nchild}", flush=True)


def policy(args):
    # Limit XLA's CPU pools as well as BLAS; do this before importing JAX.
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
        os.sched_setaffinity(0, cpus)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        # Empty JAX_PLATFORMS uses backend priorities and tolerates absent CUDA.
        # Explicit 'cuda,cpu' instead makes a CUDA initialization failure fatal.
        os.environ.setdefault("JAX_PLATFORMS", "")
    import cshogi
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pgx
    from pgx.experimental.shogi import from_sfen, to_sfen, _from_board
    from pgx._src.games.shogi import _observe
    import m2_moves
    from m2_distill_train import make_forward
    roots = roots_at(args.work)
    require(digest(args.checkpoint) == CKPT_SHA, "checkpoint differs from the identified old WS-T")
    with args.checkpoint.open("rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    require(cfg.get("arch") == "transformer", "expected WS-T transformer")
    forward = make_forward(cfg["num_channels"], cfg["num_layers"], cfg["resnet_v2"],
                           arch=cfg["arch"], d_model=cfg["d_model"], n_heads=cfg["n_heads"], ff_mult=cfg["ff_mult"])
    devices = jax.devices()
    device = next((d for d in devices if d.platform == "gpu"), devices[0])
    model = jax.device_put(ckpt["model"], device)
    observe = jax.jit(lambda x: _observe(x, flip=jnp.bool_(True)))

    def encode(sfen):
        state = from_sfen(sfen)
        # Installed pgx from_sfen parses multi-digit hand counts one digit at a
        # time. Repair only that host-parser defect with its verified constructor.
        if re.search(r"\d{2}", sfen.split()[2]):
            absolute = from_sfen(sfen.split()[0] + " b - 1")
            hand = np.zeros((2, 7), dtype=np.int32)
            for count, letter in re.findall(r"(\d*)([A-Za-z])", sfen.split()[2]):
                hand[int(letter.islower()), "PLNSBRG".index(letter.upper())] = int(count or 1)
            state = _from_board(int(sfen.split()[1] == "w"), absolute._x.board, hand)
        return state, observe(state._x)

    @jax.jit
    def predict(obs):
        (logits, _), _ = forward.apply(model[0], model[1], obs, is_eval=True)
        return logits

    def infer(sfens):
        encoded = [encode(s) for s in sfens]
        obs = jnp.stack([x[1] for x in encoded])
        logits = np.asarray(predict(obs))  # synchronizes host transfer, including GPU
        return encoded, logits

    output = {}
    # Validate the entire legal set against independent cshogi, not just top-k.
    for root in roots["roots"]:
        (state_obs,), logits = infer([root["sfen"]])
        state, _ = state_obs
        require(to_sfen(state).split()[:3] == root["sfen"].split()[:3], "pgx SFEN roundtrip mismatch")
        board = cshogi.Board(root["sfen"])
        legal = {cshogi.move_to_usi(m) for m in board.legal_moves}
        actions = np.flatnonzero(np.asarray(state.legal_action_mask))
        moves = [m2_moves.action_to_usi(int(a), np.asarray(state._x.board), int(state._x.color)) for a in actions]
        require(set(moves) == legal and len(moves) == len(legal), f"pgx/cshogi legal set mismatch: {root['root_id']}")
        values = logits[0, actions].astype(np.float64)
        probs = np.exp(values - values.max())
        probs /= probs.sum()
        order = np.argsort(-probs, kind="stable")[:args.top_k]
        output[root["root_id"]] = dict(game_id=root["game_id"], legal_count=len(legal),
              top_moves=[dict(move=moves[i], action=int(actions[i]), probability=float(probs[i])) for i in order])
        print(f"policy: {root['root_id']} top2={[moves[i] for i in order[:2]]}", flush=True)
    latency = {}
    sfens = [r["sfen"] for r in roots["roots"]]
    for batch in (1, 8):
        inputs = [sfens[i % len(sfens)] for i in range(batch)]
        infer(inputs)  # excludes JIT compilation; timed calls redo SFEN encoding
        times = []
        for i in range(args.latency_repeats):
            inputs = [sfens[(i * batch + j) % len(sfens)] for j in range(batch)]
            start = time.perf_counter()
            infer(inputs)
            times.append((time.perf_counter() - start) * 1000)
        latency[str(batch)] = dict(p50_ms=float(np.percentile(times, 50)), p99_ms=float(np.percentile(times, 99)),
                                  repeats=len(times), raw_ms=times,
                                  scope="SFEN parse + pgx legal mask/observation + stack/H2D + forward + synchronized logits D2H; no top-k")
    result = dict(format=FORMAT, roots_sha256=identity(roots), checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=CKPT_SHA, network_config=cfg, epoch=ckpt.get("epoch"),
                  device=str(device), backend=device.platform,
                  packages={x: importlib.metadata.version(x) for x in ("jax", "jaxlib", "pgx", "dm-haiku", "cshogi")},
                  pgx_path=str(Path(pgx.__file__).resolve()), script_sha256=digest(__file__),
                  dependencies={x.name: digest(x) for x in (HERE / "network.py", HERE / "m2_distill_train.py",
                                                           HERE / "m2_moves.py")},
                  cpu_affinity=sorted(os.sched_getaffinity(0)), latency=latency, roots=output)
    path = args.work / "policy.json"
    if path.exists():
        previous = read(path)
        require(previous["roots_sha256"] == result["roots_sha256"] and previous["checkpoint_sha256"] == CKPT_SHA,
                "incompatible policy output; use another --work")
    atomic(path, result)
    print("policy latency: " + canonical(latency), flush=True)


def ranks(values):
    """Empirical midranks, so tied depth/zero-PV-change roots stay tied."""
    from collections import Counter
    counts = Counter(values)
    table, before = {}, 0
    for value in sorted(counts):
        table[value] = before + (counts[value] - 1) / 2
        before += counts[value]
    return [table[x] for x in values]


def allocation(rows, key):
    """20% promoted, randomize a fractional boundary to keep EXACT expected budget."""
    n = len(rows)
    if not n:
        return None, []
    if key in ("cheap", "cheap_free", "cheap_live"):
        gap = ranks([-r["gap_cp"] if r["gap_cp"] is not None else 0 for r in rows]) if key == "cheap" else [0] * n
        change = ranks([r["pv_changes"] for r in rows])
        depth = ranks([-r["depth"] for r in rows])
        priorities = [a + b + c for a, b, c in zip(gap, change, depth)]
    else:
        priorities = [r["loss_n"] if key == "oracle_loss" else r["loss_n"] - r["loss_4n"] for r in rows]
    order = sorted(range(n), key=lambda i: (-priorities[i], rows[i]["root_id"], i))
    weights = [0.0] * n
    quota = n / 5
    full = math.floor(quota)
    for i in order[:full]:
        weights[i] = 1.0
    if full < n:
        weights[order[full]] = quota - full
    deep_key = "loss_live_4n" if key == "cheap_live" else "loss_4n"
    loss = sum((1 - w) * r["loss_n"] + w * r[deep_key] for r, w in zip(rows, weights)) / n
    return loss, weights


def statistics(rows):
    if not rows:
        return dict(roots=0, eligible=0, rescued=0, rescue_rate=None, mean_recovery_cp=None,
                    uniform_loss_cp=None, oracle_loss_mean_cp=None, oracle_loss_reduction_cp=None,
                    oracle_gain_mean_cp=None, oracle_gain_reduction_cp=None, cheap_mean_cp=None, cheap_reduction_cp=None,
                    cheap_free_mean_cp=None, cheap_free_reduction_cp=None, cheap_live_mean_cp=None, cheap_live_reduction_cp=None)
    eligible = [r for r in rows if r["loss_n"] >= 100]
    uniform = sum(r["loss_uniform"] for r in rows) / len(rows)
    result = dict(roots=len(rows), eligible=len(eligible), rescued=sum(r["recovery_cp"] >= 50 for r in eligible),
                  rescue_rate=sum(r["recovery_cp"] >= 50 for r in eligible) / len(eligible) if eligible else None,
                  mean_recovery_cp=sum(r["recovery_cp"] for r in eligible) / len(eligible) if eligible else None,
                  uniform_loss_cp=uniform)
    for key in ("oracle_loss", "oracle_gain", "cheap", "cheap_free", "cheap_live"):
        loss, _ = allocation(rows, key)
        result[key + "_mean_cp"] = loss
        result[key + "_reduction_cp"] = uniform - loss
    return result


def bootstrap(rows, repeats, seed):
    import numpy as np
    estimate = statistics(rows)
    if not rows:
        return dict(estimate=estimate, ci95={}, replicates=repeats, valid_replicates={})
    # One root per game; resampling whole records is precisely game-level bootstrap.
    require(len({r["game_id"] for r in rows}) == len(rows), "bootstrap expects one root per game")
    rng = np.random.default_rng(seed)
    samples = {k: [] for k, v in estimate.items() if k not in ("roots", "eligible", "rescued")}
    for _ in range(repeats):
        value = statistics([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        for k in samples:
            if value[k] is not None:
                samples[k].append(value[k])
    return dict(estimate=estimate, ci95={k: list(map(float, np.percentile(v, [2.5, 97.5]))) if v else None
                                       for k, v in samples.items()}, replicates=repeats,
                valid_replicates={k: len(v) for k, v in samples.items()})


def report(args):
    roots = roots_at(args.work)
    manifest, policy = read(args.work / "search_manifest.json"), read(args.work / "policy.json")
    require(policy["roots_sha256"] == manifest["roots_sha256"] == identity(roots), "roots identity mismatch")
    n, uniform_n, four_n, ref_n, check_n = map(str, manifest["budgets"])
    rows = []
    for root in roots["roots"]:
        part = read(part_path(args.work, root))
        require(part["root"] == root and part["recipe_id"] == manifest["recipe_id"], "search identity mismatch")
        results, children = part["budgets"], part["children"]
        require(all(k in results for k in (n, uniform_n, four_n, ref_n, check_n)) and part["multipv_n"], "incomplete search")
        top = [x["move"] for x in policy["roots"][root["root_id"]]["top_moves"][:2]]
        require("continued_4n" in part, "missing N+3N continuation; rerun search")
        live = part["continued_4n"]
        candidates = set(top + [x["bestmove"] for x in results.values()] + [live["bestmove"]])
        require(all(m in children and ref_n in children[m] and check_n in children[m] for m in candidates),
                f"missing child evaluation at {root['root_id']}; rerun search after policy")
        values = {m: -children[m][ref_n]["score_cp"] for m in candidates}
        ref_best = results[ref_n]["bestmove"]
        baseline = values[ref_best]
        ref, check = results[ref_n], results[check_n]
        move_changed = ref["bestmove"] != check["bestmove"]
        delta = abs(ref["score_cp"] - check["score_cp"])
        mate = any(x["mate"] for x in results.values()) or live["mate"] or part["multipv_n"]["mate"] or any(
            children[m][k]["mate"] for m in candidates for k in (ref_n, check_n))
        mate |= any(x["mate"] for iteration in part["multipv_n"]["multipv"].values() for x in iteration.values())
        seq = results[n]["iterations"]
        changes = sum(a["pv"][0] != b["pv"][0] for a, b in zip(seq, seq[1:]))
        short_move = results[n]["bestmove"]
        row = dict(root_id=root["root_id"], game_id=root["game_id"], stable=not move_changed and delta <= 100,
                   reference_move_changed=move_changed, reference_value_delta_cp=delta, mate=mate,
                   reference_best_move=ref_best, reference_best_child_cp=baseline, move_values_cp=values,
                   loss_n=baseline - values[short_move], loss_uniform=baseline - values[results[uniform_n]["bestmove"]],
                   loss_4n=baseline - values[results[four_n]["bestmove"]],
                   loss_live_4n=baseline - values[live["bestmove"]], live_nodes=live["nodes"] + live["prefix"]["nodes"],
                   policy_top2=top, recovery_cp=max(values[m] for m in top) - values[short_move],
                   gap_cp=part["multipv_n"]["gap_cp"], pv_changes=changes, depth=results[n]["depth"],
                   reference_scores_match_bestmove=all(results[k].get("score_matches_bestmove", True) for k in (ref_n, check_n)),
                   child_max_value_delta_cp=max(abs(children[m][ref_n]["score_cp"] - children[m][check_n]["score_cp"])
                                                for m in candidates),
                   actual_nodes={k: v["nodes"] for k, v in results.items()})
        rows.append(row)
    groups = dict(main=[r for r in rows if r["stable"] and not r["mate"]],
                  unstable_nonmate=[r for r in rows if not r["stable"] and not r["mate"]],
                  mate=[r for r in rows if r["mate"]], all=rows,
                  child_stable_sensitivity=[r for r in rows if r["stable"] and not r["mate"]
                                            and r["child_max_value_delta_cp"] <= 100])
    summary = {k: bootstrap(v, args.bootstrap, args.seed) for k, v in groups.items()}
    # Expose the actual promotion ranking/weights for audit and reproducibility.
    for key in ("oracle_loss", "oracle_gain", "cheap", "cheap_free", "cheap_live"):
        _, weights = allocation(groups["main"], key)
        summary["main"][key + "_promotion_weights"] = {r["root_id"]: w for r, w in zip(groups["main"], weights)}
    result = dict(format=FORMAT, work=str(args.work.resolve()), search_recipe=manifest,
                  policy={k: v for k, v in policy.items() if k != "roots"}, bootstrap_seed=args.seed,
                  definitions=dict(reference="ref(m)=-child_20N(m); ref(best)=ref(root_20N.bestmove); signed loss, no clipping",
                    stability="root_20N.bestmove == root_40N.bestmove AND abs(root value change)<=100 USI cp",
                    rescue="among stable nonmate loss_N>=100, max(ref(policy top2))-ref(N.bestmove)>=50",
                    cheap="sum of empirical midranks: small MultiPV2 N gap, more single-PV N iteration bestmove changes, shallow N depth; no labels/fitting",
                    cheap_free="same fixed rule without gap, using only the original single-PV N run; no extra MultiPV measurement cost",
                    cheap_live="gap-free rule, N screening then warm-TT go nodes 3N on promoted roots; total 4N includes screening, no extra MultiPV; practical equal-budget estimate",
                    score="last exact completed iteration; stopped bestmove may differ from its PV; final_info retains bounds; score_matches_bestmove flags this",
                    oracle_loss="promote largest reference loss at N (requested oracle)",
                    oracle_gain="promote largest loss_N-loss_4N (true hindsight upper bound for this binary allocation)",
                    allocation="20% at 4N, remainder N; fractional boundary randomized in expectation, exactly 1.6N mean REQUESTED nodes",
                    bootstrap="paired game bootstrap; rerank/reallocate within every replicate; percentile 95% CI; undefined rescue replicates omitted",
                    cost="comparison counts main go-node budgets only; MultiPV diagnostic/policy/reference/child searches are measurement overhead; actual nodes logged",
                    heldout="one Task159 game per root, unused for these judgement/selection rules; not a claim of being excluded from past eval training",
                    mate="any root-budget or candidate-child mate/declaration/terminal -> separate mate stratum, mate +/-3000"),
                  counts=dict(total=len(rows), stable=sum(r["stable"] for r in rows), mate=sum(r["mate"] for r in rows),
                              unstable=sum(not r["stable"] for r in rows),
                              reference_surpassed=sum(any(v > r["reference_best_child_cp"] for v in r["move_values_cp"].values()) for r in rows),
                              missing_gap=sum(r["gap_cp"] is None for r in rows)), summary=summary, roots=rows)
    atomic(args.work / "report.json", result)
    def fmt(group, key):
        s = summary[group]
        value = s["estimate"][key]
        if value is None:
            return "算出不能（対象0）"
        ci = s["ci95"][key]
        return f"{value:.4f} [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci else str(value)
    lines = ["# Task 167 Probe 1 / 3", "", f"全{len(rows)}局面、主集計{len(groups['main'])}、不安定{result['counts']['unstable']}、詰み関連{result['counts']['mate']}。",
             f"N={n}、一様={uniform_n}、重点={four_n}、参照={ref_n}/{check_n} nodes。", "",
             f"Probe 1: 短探索の損失100cp以上 {summary['main']['estimate']['eligible']}件中、WS-T上位2手のいずれかで50cp以上回復 {summary['main']['estimate']['rescued']}件。",
             f"救済率 {fmt('main', 'rescue_rate')}。", "",
             f"Probe 3: 一様の平均参照損失 {fmt('main', 'uniform_loss_cp')} cp。",
             f"参照損失順oracleの平均損失削減 {fmt('main', 'oracle_loss_reduction_cp')} cp。",
             f"実現改善量順oracle（上限）の削減 {fmt('main', 'oracle_gain_reduction_cp')} cp。",
             f"短探索だけの固定ルールの削減 {fmt('main', 'cheap_reduction_cp')} cp。", "",
             f"追加MultiPV不要の固定ルール（PV変化＋深さ）の削減 {fmt('main', 'cheap_free_reduction_cp')} cp。", "",
             f"実行費用込み（Nで選別＋重点局のみTT保持で追加3N）の削減 {fmt('main', 'cheap_live_reduction_cp')} cp。", "",
             "削減は一様−配分後（正が改善）。CIは局単位の対応つきbootstrap。各再標本でも順位と配分を再計算。",
             "重点枠は20%。端数1局は確率的な配分の期待損失を使い、要求ノード総量を一様1.6Nと一致させる。",
             "MultiPV2・policy・参照・子局面の測定費用は比較予算に含まない。実測nodesはJSONに保存。",
             "冷たい4Nの反実仮想と、実行費用込みのN＋追加3Nを分けて表示。選別後に冷たい4Nを丸ごとやり直す実装なら平均1.8Nを要する。",
             "安定性は20N/40Nの根の最善手一致かつ値差100cp以内。子局面値差も100cp以内の感度分析をJSONに併記。",
             "参照bestの子探索より他候補が高い場合も符号つき損失を保持。小標本・詰み群・不安定群は主集計と分ける。", "",
             f"WS-T: `{policy['checkpoint']}`", f"sha256: `{policy['checkpoint_sha256']}`", "",
             "推論時間（ms、SFEN符号化を含み、JITコンパイルを除外、logitsの同期転送まで）:"]
    for batch, lat in policy["latency"].items():
        lines.append(f"- batch {batch}: p50={lat['p50_ms']:.3f}, p99={lat['p99_ms']:.3f} ({lat['repeats']}反復、{policy['backend']})")
    lines += ["", "不安定群・詰み群・全件・子局面安定群の統計、各局面の値と配分、正確な設定は `report.json` を参照。", ""]
    atomic(args.work / "summary.md", "\n".join(lines), text=True)
    print("report: " + canonical(result["counts"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, func in (("sample", sample), ("search", search), ("policy", policy), ("report", report)):
        p = sub.add_parser(command)
        p.set_defaults(func=func)
        p.add_argument("--work", type=Path, default=DATA / "work/strategy/probe13")
        if command == "sample":
            p.add_argument("--games-dir", type=Path, default=DATA / "work/task159/games")
            p.add_argument("--count", type=int, default=256)
            p.add_argument("--seed", type=int, default=20271670)
            p.add_argument("--min-ply", type=int, default=20)
            p.add_argument("--max-ply", type=int, default=150)
            p.add_argument("--max-score", type=int, default=1500)
        elif command == "search":
            p.add_argument("--engine", type=Path, default=Path(os.environ.get("YANE", DATA / "engines/yaneuraou/source/YaneuraOu-by-gcc")))
            p.add_argument("--eval-dir", type=Path, default=DATA / "eval_dirs/task154_R_result30")
            p.add_argument("--nodes", type=int, default=100000, help="N; budgets N,1.6N,4N,20N,40N")
            p.add_argument("--procs", type=int, default=20)
            p.add_argument("--timeout", type=float, default=300)
        elif command == "policy":
            p.add_argument("--checkpoint", type=Path, default=PGX_HOME / "pgx/examples/alphazero/t087_wst/ws2.ckpt")
            p.add_argument("--device", choices=("auto", "cpu"), default="auto")
            p.add_argument("--cpu-threads", type=int, default=4)
            p.add_argument("--top-k", type=int, default=8)
            p.add_argument("--latency-repeats", type=int, default=100)
        else:
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--seed", type=int, default=20271673)
    args = parser.parse_args()
    args.work = args.work.expanduser().resolve()
    for key in ("count", "cpu_threads", "latency_repeats", "bootstrap"):
        if hasattr(args, key):
            require(getattr(args, key) > 0, f"--{key.replace('_', '-')} must be positive")
    if hasattr(args, "top_k"):
        require(args.top_k >= 2, "--top-k must be >=2")
    args.func(args)


if __name__ == "__main__":
    main()
