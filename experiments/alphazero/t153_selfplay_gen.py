#!/usr/bin/env python3
"""Task 153: own self-play -> qsearch leaves -> result-bearing HalfKP cache.

Source scripts/env.sh first. One atomic JSON part per game; deterministic game
seeds and game-boundary resets make process scheduling irrelevant. Scores use
Task 146's internal Value (mate +/-32000), always from the position's turn.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
import zipfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cshogi
import numpy as np
import qs_replace as qs
import t146_self_relabel as old
from trainer import kp, mirror

DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data")).resolve()
FORMAT = "task153-selfplay-v1"
MAX_PROCS = max(28, os.cpu_count() or 1)   # 28 on the 14700KF; the Xeon (Task 185) has 56
require, digest, atomic, json_write = old.require, old.digest, old.atomic, old.json_write


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(old.canonical(value).encode()).hexdigest()


def game_seed(seed, game_id):
    """Task 153: independent streams for every game ID, including resumed ranges."""
    return int.from_bytes(hashlib.sha256(f"t153:{seed}:{game_id}".encode()).digest()[:8], "big")


def engine_config(args):
    engine, evaluation = args.engine.resolve(), args.eval_dir.resolve()
    require(not (evaluation / "eval_options.txt").exists(), "eval_options.txt may override fixed options")
    require(not any(c.isspace() for c in str(evaluation)), "EvalDir cannot contain whitespace")
    return dict(engine=str(engine), engine_sha256=digest(engine), eval_dir=str(evaluation),
                eval_sha256=digest(evaluation / "nn.bin"),
                options=dict(old.OPTIONS, USI_Hash=args.hash_mb, EvalDir=str(evaluation)),
                limit=dict(depth=getattr(args, "depth", 1)), node_cap=getattr(args, "nodes_cap", None),
                timeout=args.timeout)


class GameEngine(old.Engine):
    """Task 153: retain complete game history and accept terminal USI responses."""

    def __init__(self, cfg):
        # The base constructor normally resets. Defer that reset to game start,
        # including the first game, so there is exactly one reset per game.
        self.initializing = True
        super().__init__(cfg)
        self.initializing = False

    def reset(self):
        if not self.initializing:
            super().reset()

    def search_game(self, moves):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        limit = dict(self.cfg["limit"])
        if self.cfg.get("node_cap"):
            limit["nodes"] = self.cfg["node_cap"]
        self.send("go " + " ".join(f"{k} {v}" for k, v in limit.items()))
        deadline, result = time.monotonic() + self.cfg["timeout"], None
        while True:
            line = self.readline(deadline)
            if line.startswith("bestmove "):
                move = line.split()[1]
                if move in ("win", "resign"):
                    return move, result
                require(result is not None, f"bestmove without exact score: {list(self.lines)}")
                require(result["mate"] or "nodes" in limit or result["depth"] >= limit["depth"],
                        f"incomplete depth: {result}")
                return move, result
            tokens = line.split()
            if not line.startswith("info ") or "string" in tokens:
                continue
            if "multipv" in tokens and tokens[tokens.index("multipv") + 1] != "1":
                continue
            match = old.SCORE_RE.search(line)
            if match and not {"lowerbound", "upperbound"}.intersection(tokens):
                kind, value = match.groups()
                score = old.internal_score(kind, value)
                require(-32000 <= score <= 32000, f"score out of range: {line}")
                result = dict(score=score, raw=value, mate=kind == "mate",
                              depth=int(tokens[tokens.index("depth") + 1]) if "depth" in tokens else 0,
                              nodes=int(tokens[tokens.index("nodes") + 1]) if "nodes" in tokens else 0)


def terminal(board, ply, max_ply, occurrences=1):
    """Task 153: four identical board/hands/turn states; perpetual check is draw."""
    if board.is_game_over():
        return 1 - board.turn, "no_legal_moves"
    if occurrences >= 4:
        # Treat even perpetual check as a draw, a deliberately documented
        # training adjudication policy rather than a claim about official rules.
        return -1, "repetition_draw"
    if board.is_nyugyoku():
        return board.turn, "declaration_27_win"
    if ply >= max_ply:
        return -1, "max_ply_draw"
    return None


def play_game(engine, cfg, game_id):
    """Task 153: one bounded deterministic game, with labels filled at its end."""
    seed = game_seed(cfg["seed"], game_id)
    rng, board = random.Random(seed), cshogi.Board()
    opening = rng.randint(cfg["opening_min"], cfg["opening_max"])
    moves, rows, random_plies = [], [], []
    seen = Counter({tuple(board.sfen().split()[:3]): 1})
    engine.reset()
    while True:
        ply = len(moves)
        end = terminal(board, ply, cfg["max_ply"], seen[tuple(board.sfen().split()[:3])])
        if end is not None:
            winner, reason = end
            break
        legal = {cshogi.move_to_usi(m): m for m in board.legal_moves}
        if ply < opening:
            move = rng.choice(sorted(legal))
            random_plies.append(ply)
        else:
            move, score = engine.search_game(moves)
            if move == "win":
                require(board.is_nyugyoku(), "engine declaration disagrees with cshogi LAW_27")
                winner, reason = board.turn, "declaration_27_win"
                break
            if move == "resign":
                winner, reason = 1 - board.turn, "engine_resign"
                break
            require(move in legal, f"illegal engine move {move}: {board.sfen()}")
            noise = ply < cfg["random_until_ply"] and rng.random() < cfg["random_move_prob"]
            adjudicate = abs(score["score"]) >= cfg["eval_limit"]
            if noise and not adjudicate:
                move = rng.choice(sorted(legal))
                random_plies.append(ply)
            elif ply >= cfg["write_minply"]:
                rows.append(dict(sfen=board.sfen(), move=move, ply=ply, turn=board.turn,
                                 game_id=game_id, **score))
            if adjudicate:
                winner = board.turn if score["score"] > 0 else 1 - board.turn
                reason = "eval_limit"
                break
        moves.append(move)
        board.push(legal[move])
        seen[tuple(board.sfen().split()[:3])] += 1
    for row in rows:
        row["result"] = 0 if winner == -1 else (1 if row["turn"] == winner else -1)
    return dict(format=FORMAT, recipe_id=cfg["recipe_id"], game_id=game_id, seed=seed,
                opening_plies=opening, random_plies=random_plies, moves=moves, rows=rows,
                winner=winner, reason=reason, final_sfen=board.sfen())


_CFG = _ENGINE = None


def worker_init(cfg, parent):
    global _CFG, _ENGINE
    old.parent_death(parent)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _CFG, _ENGINE = cfg, None


def game_job(job):
    """Task 153: restart the engine and replay the whole game after a crash."""
    global _ENGINE
    game_id, path = job
    for attempt in range(_CFG["retries"] + 1):
        try:
            if _ENGINE is None:
                _ENGINE = GameEngine(_CFG["engine"])
            game = play_game(_ENGINE, _CFG, game_id)
            game["content_sha256"] = identity(game)
            json_write(path, game)  # No PID, timing or destination in deterministic parts.
            return game_id, len(game["rows"]), game["reason"]
        except (RuntimeError, TimeoutError, OSError) as exc:
            print(f"T153 retry game={game_id} attempt={attempt + 1}: {exc}", file=sys.stderr, flush=True)
            if _ENGINE is not None:
                _ENGINE.close()
                _ENGINE = None
            if attempt == _CFG["retries"]:
                raise


def game_path(work, game_id):
    return work / "games" / f"game_{game_id:012d}.json"


def game_ids(manifest):
    for lo, hi in manifest["ranges"]:
        yield from range(lo, hi)


def load_game(path, manifest, game_id):
    game = read_json(path)
    require(game.get("content_sha256") == identity({k: v for k, v in game.items() if k != "content_sha256"}),
            f"game checksum mismatch: {path}")
    require(game["format"] == FORMAT and game["recipe_id"] == manifest["recipe_id"]
            and game["game_id"] == game_id and game["seed"] == game_seed(manifest["config"]["seed"], game_id),
            f"incompatible game: {path}")
    require(game["winner"] in (-1, 0, 1) and isinstance(game["moves"], list), f"bad game: {path}")
    for r in game["rows"]:
        require(r["game_id"] == game_id and -32000 <= r["score"] <= 32000
                and r["turn"] in (0, 1) and r["result"] ==
                (0 if game["winner"] == -1 else (1 if r["turn"] == game["winner"] else -1)),
                f"bad row: {path}")
    return game


def generate(args):
    """Task 153: extend game ranges and reuse only complete compatible games."""
    work = args.out.resolve()
    engine = engine_config(args)
    engine.pop("timeout")  # Runtime timeout/retries/process count do not affect labels.
    cfg = {k: getattr(args, k) for k in ("seed", "opening_min", "opening_max", "write_minply",
                                       "max_ply", "eval_limit", "random_move_prob", "random_until_ply")}
    cfg.update(engine=engine, policy=dict(score_unit="internal Value, STM; mate +/-32000",
               eval_limit="abs(score)>=limit at root: score-positive side wins, before playing move",
               repetition="four occurrences of identical board/hands/turn -> draw, including perpetual check; not cshogi twofold search detection",
               declaration="LAW_27 winner", max_ply="draw; played plies include random prefix",
               record="searched legal non-random roots at ply>=write_minply; terminal win/resign omitted",
               reset="usinewgame + isready exactly once per game; position startpos moves FULL_HISTORY"),
               dependencies={Path(p).name: digest(p) for p in (__file__, old.__file__)},
               numpy=np.__version__, cshogi=importlib.metadata.version("cshogi"), python=sys.version.split()[0])
    recipe = identity(cfg)
    started = time.monotonic()
    with old.locked(work):
        path = work / "manifest.json"
        manifest = read_json(path) if path.exists() else dict(format=FORMAT, config=cfg, recipe_id=recipe, ranges=[])
        require(manifest["recipe_id"] == recipe and manifest["config"] == cfg, "recipe changed; use another --out")
        intervals = sorted(manifest["ranges"] + [[args.start_game, args.start_game + args.games]])
        merged = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        manifest["ranges"] = merged
        json_write(path, manifest)
        pending, reused = [], 0
        for gid in range(args.start_game, args.start_game + args.games):
            p = game_path(work, gid)
            try:
                load_game(p, manifest, gid)
                reused += 1
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pending.append((gid, p))
        runtime = dict(cfg, recipe_id=recipe, retries=args.retries,
                       engine=dict(engine, timeout=args.timeout))
        counts, rows = Counter(), 0
        print(f"T153 gen requested={args.games} reused={reused} pending={len(pending)}", flush=True)
        if pending:
            with mp.get_context("spawn").Pool(args.procs, initializer=worker_init,
                                               initargs=(runtime, os.getpid())) as pool:
                iterator = pool.imap_unordered(game_job, pending)
                done, last = 0, time.monotonic()
                while done < len(pending):
                    try:
                        gid, n, reason = iterator.next(timeout=1)
                        rows += n
                        counts[reason] += 1
                        done += 1
                    except mp.TimeoutError:
                        pass
                    if time.monotonic() - last >= 10 or done == len(pending):
                        print(f"T153 games={done}/{len(pending)} rows={rows} seconds={time.monotonic()-started:.2f}", flush=True)
                        last = time.monotonic()
        report = dict(new_games=len(pending), reused_games=reused, new_rows=rows,
                      seconds=time.monotonic() - started, outcomes=dict(counts))
        json_write(work / "last_gen.json", report)
        return report


def qs_recipe(args, manifest):
    cfg = engine_config(args)
    cfg.pop("limit")
    cfg.pop("node_cap")
    cfg.pop("timeout")
    return dict(engine=cfg, generation_recipe=manifest["recipe_id"], changed=args.changed,
                helpers={Path(p).name: digest(p) for p in (__file__, qs.__file__, kp.__file__)},
                reset="once per game part, one QS worker for deterministic leaves",
                policy="drop in-check leaves; negate score AND result on odd PV; retain root ply/move separately")


def load_qpart(path, recipe, source_sha):
    with np.load(path, allow_pickle=False) as z:
        require(str(z["recipe_id"]) == recipe and str(z["source_sha256"]) == source_sha, "QS part identity mismatch")
        data = {k: z[k].copy() for k in z.files}
    n = len(data["scores"])
    for key, dtype, shape in (("features", np.int32, (n, 77)), ("scores", np.int16, (n,)),
                              ("results", np.int8, (n,)), ("source_rows", np.int32, (n,)),
                              ("flip", np.bool_, (n,)), ("changed", np.bool_, (n,))):
        require(data[key].dtype == dtype and data[key].shape == shape, f"bad QS {key}: {path}")
    require(np.isin(data["results"], [-1, 0, 1]).all(), "bad QS results")
    return data


def qsearch(args):
    """Task 153: marker-score qsearch via the same extension as qs_replace.py."""
    work = args.work.resolve()
    require(not any(c.isspace() for c in str(work)), "qsearch paths cannot contain whitespace")
    started, engine = time.monotonic(), None
    batch = getattr(args, "_batch", None)
    with (old.locked(work) if batch is None else nullcontext()):
        manifest = read_json(work / "manifest.json")
        cfg = qs_recipe(args, manifest)
        recipe = identity(cfg)
        folder = work / "qs"
        folder.mkdir(exist_ok=True)
        mpath = folder / "manifest.json"
        if mpath.exists():
            require(read_json(mpath)["config"] == cfg, "QS recipe changed; use another work")
        if batch is None:
            json_write(mpath, dict(config=cfg, recipe_id=recipe))
        if args.procs > 1 and batch is None:
            ids = list(game_ids(manifest))
            jobs = []
            for i in range(min(args.procs, len(ids))):
                child = argparse.Namespace(**vars(args))
                child._batch = ids[i::args.procs]
                jobs.append(child)
            counts = Counter()
            with mp.get_context("spawn").Pool(args.procs, initializer=worker_init,
                                               initargs=({}, os.getpid())) as pool:
                iterator = pool.imap_unordered(qsearch, jobs)
                done, last = 0, time.monotonic()
                while done < len(jobs):
                    try:
                        report = iterator.next(timeout=1)
                        counts.update({k: v for k, v in report.items() if k != "seconds"})
                        done += 1
                    except mp.TimeoutError:
                        pass
                    if time.monotonic() - last >= 10:
                        print(f"T153 QS workers={done}/{len(jobs)} seconds={time.monotonic()-started:.2f}", flush=True)
                        last = time.monotonic()
            report = dict(counts, seconds=time.monotonic() - started)
            json_write(folder / "last_qs.json", report)
            print("T153 QS " + old.canonical(report), flush=True)
            return report
        counts = Counter()
        try:
            for gid in (game_ids(manifest) if batch is None else batch):
                source = game_path(work, gid)
                game = load_game(source, manifest, gid)
                source_sha = digest(source)
                dest = folder / (source.stem + ".npz")
                try:
                    part = load_qpart(dest, recipe, source_sha)
                    counts["reused_games"] += 1
                except (OSError, ValueError, EOFError, KeyError, RuntimeError, zipfile.BadZipFile):
                    rows = game["rows"]
                    marker = np.zeros(len(rows), dtype=qs.PSV)
                    board = cshogi.Board()
                    for i, row in enumerate(rows):
                        board.set_sfen(row["sfen"])
                        board.to_psfen(marker[i]["sfen"])
                        marker[i]["gamePly"] = row["ply"] + 1
                    marker["score"] = 1
                    # Keep scratch files within WORK; a killed command leaves no published part.
                    with tempfile.TemporaryDirectory(prefix=".qs_", dir=work) as scratch:
                        inp, out = Path(scratch) / "in.psv", Path(scratch) / "out.psv"
                        marker.tofile(inp)
                        for attempt in range(args.retries + 1):
                            try:
                                if engine is None:
                                    engine = old.Engine(dict(cfg["engine"], timeout=args.timeout))
                                else:
                                    engine.reset()
                                engine.send(f"qsearch_psv {inp} {out} 1")
                                lines = engine.until("info string qsearch_psv ")
                                line = lines[-1]
                                require("done:" in line, f"qsearch failed: {line}")
                                stats = {k: int(v) for k, v in (s.split("=") for s in line.split("done:")[1].split())}
                                require(stats["records"] == len(rows) and stats["decode_errors"] == 0
                                        and stats["illegal_pv"] == 0, f"qsearch errors: {stats}")
                                require(out.stat().st_size == len(rows) * qs.PSV.itemsize, "truncated qsearch output")
                                leaves = np.fromfile(out, dtype=qs.PSV)
                                break
                            except (OSError, RuntimeError, TimeoutError) as exc:
                                print(f"T153 QS retry game={gid} attempt={attempt+1}: {exc}", file=sys.stderr, flush=True)
                                if engine is not None:
                                    engine.close()
                                    engine = None
                                if attempt == args.retries:
                                    raise
                    changed, flip = qs.leaf_rows(marker, leaves)
                    require(np.array_equal(marker["sfen"][:, 0] & 1 ^ (leaves["sfen"][:, 0] & 1), flip),
                            "packed turn disagrees with QS parity")
                    features, scores, results, selected = [], [], [], []
                    in_check = 0
                    for i, entry in enumerate(leaves):
                        board.set_psfen(entry["sfen"])
                        if board.is_check():
                            in_check += 1
                            continue
                        if args.changed == "drop" and changed[i]:
                            continue
                        feature = kp.canonical_shared_row(board)
                        require(len(feature) == 77, "invalid HalfKP width")
                        require(kp.canonical_shared_row_to_sfen(feature).split()[:3] == board.sfen().split()[:3],
                                "HalfKP round trip failed")
                        sign = -1 if flip[i] else 1
                        require(board.turn == (rows[i]["turn"] ^ int(flip[i])), "leaf turn mismatch")
                        features.append(feature)
                        scores.append(rows[i]["score"] * sign)
                        results.append(rows[i]["result"] * sign)
                        selected.append(i)
                    selected = np.asarray(selected, np.int32)
                    part = dict(features=np.asarray(features, np.int32).reshape(-1, 77),
                                scores=np.asarray(scores, np.int16), results=np.asarray(results, np.int8),
                                source_rows=selected, flip=flip[selected], changed=changed[selected],
                                input_rows=np.array(len(rows)), changed_rows=np.array(int(changed.sum())),
                                flipped_rows=np.array(int(flip.sum())), in_check=np.array(in_check),
                                source_sha256=np.array(source_sha), recipe_id=np.array(recipe), game_id=np.array(gid))
                    atomic(dest, lambda f: np.savez_compressed(f, **part))
                    counts["new_games"] += 1
                counts.update(input_rows=int(part["input_rows"]), kept_rows=len(part["scores"]),
                              changed_rows=int(part["changed_rows"]), flipped_rows=int(part["flipped_rows"]),
                              in_check=int(part["in_check"]))
        finally:
            if engine is not None:
                engine.close()
        report = dict(counts, seconds=time.monotonic() - started)
        if batch is None:
            json_write(folder / "last_qs.json", report)
            print("T153 QS " + old.canonical(report), flush=True)
        return report


def heldout(gid, fraction, seed):
    return game_seed(seed, gid) / 2**64 < fraction


def assemble(args):
    """Task 153: bounded-memory exact SQLite dedup, then atomic cache publication."""
    work, out = args.work.resolve(), args.out.resolve()
    require(not out.exists(), "--out must be new")
    require(not work.is_relative_to(out), "output cannot contain input work")
    with old.locked(work):
        manifest = read_json(work / "manifest.json")
        qm = read_json(work / "qs/manifest.json")
        require(qm["config"]["generation_recipe"] == manifest["recipe_id"], "QS generation mismatch")
        ids = list(game_ids(manifest))
        split = {gid: ("val" if heldout(gid, args.holdout_fraction, args.split_seed) else "train") for gid in ids}
        if args.holdout_fraction:
            require(set(split.values()) == {"train", "val"}, "holdout needs at least one whole game in each split")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="." + out.name + ".", dir=out.parent))
        db, streams = None, {}
        try:
            db = sqlite3.connect(tmp / "dedup.sqlite")
            db.execute("PRAGMA cache_size=-32768")
            db.execute("CREATE TABLE seen (row BLOB PRIMARY KEY, split TEXT) WITHOUT ROWID")
            counts, provenance = Counter(), []
            validation = {}
            if not args.holdout_fraction:
                for key, dtype, shape in (("indices", np.int32, (77,)), ("evals", np.int16, ()), ("wdl", np.int8, ())):
                    p = (args.val_cache / f"val_{key}.npy").resolve()
                    a = np.load(p, mmap_mode="r")
                    require(a.dtype == dtype and a.shape[1:] == shape, f"bad shared val {key}")
                    validation[key] = dict(path=str(p), sha256=digest(p), rows=len(a))
                require(len({v["rows"] for v in validation.values()}) == 1, "shared val alignment mismatch")
                val = np.load(validation["indices"]["path"], mmap_mode="r")
                # Exclude both actual validation positions and training-time LR augmentation.
                for row in val:
                    board = cshogi.Board(kp.canonical_shared_row_to_sfen(row))
                    reflected = np.asarray(kp.canonical_shared_row(cshogi.Board(mirror.flip_sfen(board.sfen()))), dtype="<i4")
                    for key in (row.astype("<i4", copy=False).tobytes(), reflected.tobytes()):
                        db.execute("INSERT OR IGNORE INTO seen VALUES (?, 'shared_val')", (key,))
                db.commit()
            for s in ("train", "val") if args.holdout_fraction else ("train",):
                for k in ("indices", "evals", "wdl", "game_ids", "plies"):
                    streams[s, k] = (tmp / f"{s}_{k}.raw").open("wb")
            # Held-out games first, so a duplicate can never move holdout into train.
            for gid in sorted(ids, key=lambda g: (split[g] != "val", g)):
                source = game_path(work, gid)
                game = load_game(source, manifest, gid)
                p = work / "qs" / (source.stem + ".npz")
                z = load_qpart(p, qm["recipe_id"], digest(source))
                s = split[gid]
                counts["input_rows"] += len(z["scores"])
                kept = 0
                for i, row in enumerate(z["features"]):
                    key = row.astype("<i4", copy=False).tobytes()
                    cursor = db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (key, s))
                    if not cursor.rowcount:
                        counts["dedup_or_val_overlap"] += 1
                        continue
                    if s == "val":
                        # Prevent a mirrored holdout position from leaking via augmentation.
                        board = cshogi.Board(kp.canonical_shared_row_to_sfen(row))
                        reflected = np.asarray(kp.canonical_shared_row(cshogi.Board(mirror.flip_sfen(board.sfen()))), dtype="<i4")
                        db.execute("INSERT OR IGNORE INTO seen VALUES (?, 'val_mirror')", (reflected.tobytes(),))
                    values = dict(indices=(row, "<i4"), evals=(z["scores"][i], "<i2"),
                                  wdl=(z["results"][i], "i1"), game_ids=(gid, "<i8"),
                                  plies=(game["rows"][int(z["source_rows"][i])]["ply"], "<i4"))
                    for k, (value, dtype) in values.items():
                        streams[s, k].write(np.asarray(value, dtype=dtype).tobytes())
                    counts[s] += 1
                    kept += 1
                db.commit()
                provenance.append(dict(game_id=gid, split=s, kept=kept, game_sha256=digest(source),
                                       qs_sha256=digest(p), reason=game["reason"], winner=game["winner"]))
            require(counts["train"] > 0 and (not args.holdout_fraction or counts["val"] > 0), "empty split after dedup")
            for f in streams.values():
                f.close()
            streams.clear()
            db.close()
            db = None
            (tmp / "dedup.sqlite").unlink()
            (tmp / "labels").mkdir()
            for s in ("train", "val") if args.holdout_fraction else ("train",):
                n = counts[s]
                for k, dtype in (("indices", "<i4"), ("evals", "<i2"), ("wdl", "i1"),
                                 ("game_ids", "<i8"), ("plies", "<i4")):
                    shape = (n, 77) if k == "indices" else (n,)
                    raw = tmp / f"{s}_{k}.raw"
                    src = np.memmap(raw, mode="r", dtype=dtype, shape=shape)
                    dst = np.lib.format.open_memmap(tmp / f"{s}_{k}.npy", mode="w+", dtype=dtype, shape=shape)
                    for lo in range(0, n, 65536):
                        dst[lo:lo+65536] = src[lo:lo+65536]
                    dst.flush()
                    del src, dst
                    raw.unlink()
                # Relative link survives the staging directory rename.
                (tmp / "labels" / f"{s}_dlshogi_evals.npy").symlink_to(f"../{s}_evals.npy")
            if not args.holdout_fraction:
                counts["val"] = validation["indices"]["rows"]
                for k, info in validation.items():
                    (tmp / f"val_{k}.npy").symlink_to(info["path"])
                label = (args.val_labels / "val_dlshogi_evals.npy").resolve()
                a = np.load(label, mmap_mode="r")
                require(a.dtype == np.int16 and a.shape == (counts["val"],), "bad shared val labels")
                validation["labels"] = dict(path=str(label), sha256=digest(label))
                (tmp / "labels/val_dlshogi_evals.npy").symlink_to(label)
            meta = dict(task=153, format=FORMAT, canonical=True, canonical_shared=True,
                        features_per_position=77, total_features=kp.CANONICAL_SHARED_TOTAL,
                        n_train=counts["train"], n_val=counts["val"], counts=dict(counts),
                        # train_nnue reads val_shards (Task 154 fix): the shared val keeps its shard list; holdout has none.
                        val_shards=([] if args.holdout_fraction else
                                    json.loads((args.val_cache / "meta.json").read_text(encoding="utf-8"))["val_shards"]),
                        generation=manifest, qsearch=qm, sources=provenance, shared_validation=validation,
                        split=dict(holdout_fraction=args.holdout_fraction, seed=args.split_seed,
                                   policy="whole game hash split; val first; exact global dedup; exclude val mirrors"),
                        labels="labels/; train score and WDL are leaf side-to-move; root plies preserved")
            json_write(tmp / "meta.json", meta)
            json_write(tmp / "labels/meta.json", dict(task=153, train=counts["train"], val=counts["val"],
                       cache=str(out), provenance="../meta.json", score_unit="internal Value, STM"))
            # Sync data before publishing the one-directory transaction.
            for p in tmp.rglob("*"):
                if p.is_file() and not p.is_symlink():
                    with p.open("rb") as f:
                        os.fsync(f.fileno())
            old.sync_dir(tmp / "labels")
            old.sync_dir(tmp)
            os.rename(tmp, out)
            old.sync_dir(out.parent)
        except BaseException:
            for f in streams.values():
                f.close()
            if db is not None:
                db.close()
            shutil.rmtree(tmp)
            raise
    print(f"T153 assembled train={counts['train']} val={counts['val']} labels={out / 'labels'}", flush=True)
    print(shlex.join([sys.executable, "-B", "-m", "trainer.compact_cache", str(out), str(out) + "_compact"]), flush=True)
    return meta


def benchmark(args):
    """Task 153: measure complete games and actual QS-kept positions, no training."""
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for depth in args.depths:
        work = args.out / f"d{depth}"
        # Never count reused games as newly generated benchmark work.
        require(not work.exists(), f"benchmark target already exists: {work}")
        g = argparse.Namespace(**vars(args))
        g.out, g.depth = work, depth
        generated = generate(g)
        q = argparse.Namespace(**vars(args))
        q.work, q.depth, q.changed = work, depth, "drop"
        processed = qsearch(q)
        seconds = generated["seconds"]
        kept = processed["kept_rows"]
        rate = kept / seconds
        pipeline_rate = kept / (seconds + processed["seconds"])
        row = dict(depth=depth, procs=args.procs, games=args.games, generation_seconds=seconds,
                   qs_seconds=processed["seconds"], games_hour=args.games / seconds * 3600,
                   recorded=generated["new_rows"], kept=kept, recorded_hour=generated["new_rows"] / seconds * 3600,
                   kept_hour=rate * 3600, kept_fraction=kept / max(1, generated["new_rows"]),
                   kept_hour_including_qs=kept / (seconds + processed["seconds"]) * 3600,
                   outcomes=generated["outcomes"],
                   hours_at_measured_P={str(n): n / rate / 3600 if rate else None for n in (10000000, 32000000, 64000000)},
                   hours_at_28_linear={str(n): n / (rate / args.procs * 28) / 3600 if rate else None for n in (10000000, 32000000, 64000000)},
                   hours_at_28_efficiency_60_80={str(n): [n / (rate / args.procs * 28 * f) / 3600 if rate else None
                                                        for f in (.8, .6)] for n in (10000000, 32000000, 64000000)},
                   pipeline_hours_at_measured_P={str(n): n / pipeline_rate / 3600 if pipeline_rate else None
                                                 for n in (10000000, 32000000, 64000000)},
                   pipeline_hours_at_28_efficiency_60_80={str(n): [n / (pipeline_rate / args.procs * 28 * f) / 3600 if pipeline_rate else None
                                                                  for f in (.8, .6)] for n in (10000000, 32000000, 64000000)})
        reports.append(row)
        print("T153 bench " + old.canonical(row), flush=True)
        json_write(args.out / "bench.json", dict(results=reports, estimates="28 CPU rates are extrapolations, not measurements; excludes dedup/assembly and thermal effects",
                   recommendation="depth 9, Threads 1, Hash 16; pilot <=8 processes, calibrate 20/26 before production; no quality claim"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("gen", "qs", "assemble", "bench"):
        q = sub.add_parser(name)
        if name in ("qs", "assemble"):
            q.add_argument("--work", type=Path, required=True)
        if name in ("gen", "bench", "assemble"):
            q.add_argument("--out", type=Path, required=True)
        if name == "assemble":
            q.add_argument("--val-cache", type=Path, default=DATA / "caches/shared_64m")
            q.add_argument("--val-labels", type=Path, default=DATA / "work/task142/labels_public")
            q.add_argument("--holdout-fraction", type=float, default=0)
            q.add_argument("--split-seed", type=int, default=20271531)
            continue
        q.add_argument("--engine", type=Path, default=Path(os.environ.get("YANE", DATA / "engines/yaneuraou/source/YaneuraOu-by-gcc")))
        q.add_argument("--eval-dir", type=Path, default=DATA / "eval_dirs/task142_finetune")
        q.add_argument("--hash-mb", type=int, default=16)
        q.add_argument("--timeout", type=float, default=120)
        q.add_argument("--retries", type=int, default=2)
        if name == "qs":
            q.add_argument("--procs", type=int, default=1, help="independent single-thread QS engines; game resets keep parts reproducible")
            q.add_argument("--changed", choices=("drop", "inherit"), default="drop",
                           help="inherit keeps changed leaves with parity-adjusted ROOT labels; experimental")
            continue
        q.add_argument("--games", type=int, default=24 if name == "bench" else None, required=name == "gen")
        q.add_argument("--depth", type=int, required=name == "gen", default=None)
        if name == "bench":
            q.add_argument("--depths", type=int, nargs="+", default=[7, 8, 9])
        q.add_argument("--nodes-cap", type=int)
        q.add_argument("--procs", type=int, default=4)
        q.add_argument("--start-game", type=int, default=0)
        q.add_argument("--seed", type=int, default=20271530)
        q.add_argument("--opening-min", type=int, default=6)
        q.add_argument("--opening-max", type=int, default=16)
        q.add_argument("--write-minply", type=int, default=16, help="zero-based number of already played plies")
        q.add_argument("--max-ply", type=int, default=256)
        q.add_argument("--eval-limit", type=int, default=3000, help="internal Value, not displayed USI cp")
        q.add_argument("--random-move-prob", type=float, default=0)
        q.add_argument("--random-until-ply", type=int, default=32)
    return p


def main():
    args = parser().parse_args()
    if args.command == "assemble":
        require(0 <= args.holdout_fraction < 1, "holdout fraction must be in [0,1)")
        assemble(args)
        return
    require(args.hash_mb > 0 and args.timeout > 0 and args.retries >= 0, "invalid engine resources")
    if args.command == "qs":
        require(1 <= args.procs <= MAX_PROCS, "invalid QS process count")
        qsearch(args)
        return
    require(args.games > 0 and args.start_game >= 0 and 1 <= args.procs <= MAX_PROCS, "invalid games/process count")
    require(0 <= args.opening_min <= args.opening_max < args.max_ply <= 65534, "invalid opening/max-ply")
    require(0 <= args.write_minply < args.max_ply and 0 < args.eval_limit <= 32000, "invalid recording policy")
    require(0 <= args.random_move_prob <= 1 and args.random_until_ply >= 0, "invalid diversity policy")
    require(args.nodes_cap is None or args.nodes_cap > 0, "positive nodes cap required")
    require(all(d > 0 for d in (args.depths if args.command == "bench" else [args.depth])), "positive depth required")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    (benchmark if args.command == "bench" else generate)(args)


if __name__ == "__main__":
    main()
