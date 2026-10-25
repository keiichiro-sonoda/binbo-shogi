#!/usr/bin/env python3
"""Task 153: light real-engine integration checks, including SIGKILL recovery."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import t153_selfplay_gen as t
import cshogi
import numpy as np


def run(out, name, command, good=True):
    with (out / (name + ".log")).open("w") as f:
        p = subprocess.run(command, cwd=t.REPO, stdout=f, stderr=subprocess.STDOUT, timeout=180)
    t.require((p.returncode == 0) == good, f"unexpected exit {p.returncode}: {name}.log")


def replay(work):
    """Task 153: independent cshogi replay, legal roots/moves, and outcome signs."""
    counts = Counter()
    manifest = t.read_json(work / "manifest.json")
    for gid in t.game_ids(manifest):
        game = t.load_game(t.game_path(work, gid), manifest, gid)
        board, by_ply = cshogi.Board(), {r["ply"]: r for r in game["rows"]}
        seen = Counter({tuple(board.sfen().split()[:3]): 1})
        t.require(len(by_ply) == len(game["rows"]), "duplicate root ply")
        for ply in range(len(game["moves"]) + 1):
            legal = {cshogi.move_to_usi(m): m for m in board.legal_moves}
            if ply in by_ply:
                row = by_ply[ply]
                t.require(row["sfen"] == board.sfen() and row["move"] in legal, "illegal/mismatched recorded root")
                t.require(ply not in game["random_plies"] and ply >= manifest["config"]["write_minply"], "recorded random/skipped root")
                if ply < len(game["moves"]):
                    t.require(row["move"] == game["moves"][ply], "recorded move was not played")
                counts["legal_rows"] += 1
                if row["result"] == 1:
                    counts["winner_turns"] += 1
                    counts["positive_winner_scores"] += row["score"] > 0
                if row["result"]:
                    counts["decisive_rows"] += 1
                    counts["score_sign_agrees"] += row["score"] * row["result"] > 0
            if ply < len(game["moves"]):
                t.require(game["moves"][ply] in legal, "illegal played move")
                board.push(legal[game["moves"][ply]])
                seen[tuple(board.sfen().split()[:3])] += 1
        t.require(board.sfen() == game["final_sfen"], "final position mismatch")
        if game["reason"] == "repetition_draw":
            t.require(seen[tuple(board.sfen().split()[:3])] == 4, "draw before fourth occurrence")
        if game["reason"] == "eval_limit":
            # The adjudication search is before playing its chosen move.
            if len(game["moves"]) in by_ply:
                row = by_ply[len(game["moves"])]
                t.require(row["result"] == (1 if row["score"] > 0 else -1), "eval-limit winner sign")
        counts[game["reason"]] += 1
    return dict(counts)


def children(pid):
    try:
        return [int(s) for s in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def kill_run(out, name, command, engine=False):
    """Task 153: interrupt actual in-flight jobs rather than fabricated parts."""
    with (out / (name + ".log")).open("w") as f:
        p = subprocess.Popen(command, cwd=t.REPO, stdout=f, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        killed = False
        try:
            while time.monotonic() < deadline and p.poll() is None:
                if engine:
                    for worker in children(p.pid):
                        for child in children(worker):
                            try:
                                cmd = Path(f"/proc/{child}/cmdline").read_bytes()
                            except FileNotFoundError:
                                continue
                            if b"YaneuraOu" in cmd:
                                os.kill(child, signal.SIGKILL)
                                killed = True
                                break
                        if killed:
                            break
                elif list((Path(command[command.index("--out") + 1]) / "games").glob("*.json")):
                    p.kill()
                    killed = True
                if killed:
                    break
                time.sleep(.01)
            t.require(killed, "failed to interrupt active run")
            code = p.wait(timeout=180)
            t.require(code == (0 if engine else -signal.SIGKILL), f"unexpected killed run exit: {code}")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
    if engine:
        t.require("T153 retry" in (out / (name + ".log")).read_text(), "engine kill did not exercise retry")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    t.require(not out.exists(), "test --out must be new")
    out.mkdir(parents=True)
    command = [sys.executable, "-B", str(Path(t.__file__))]
    common = ["--games", "36", "--depth", "3", "--procs", "4", "--max-ply", "160",
              "--eval-limit", "2000", "--random-move-prob", "0.1"]

    def gen(work, extra=()):
        return command + ["gen", "--out", str(out / work)] + common + list(extra)

    run(out, "clean", gen("clean"))
    run(out, "det_first", gen("det", ["--games", "12", "--procs", "1"]))
    run(out, "det_range", gen("det", ["--games", "24", "--start-game", "12"]))
    kill_run(out, "killed", gen("killed"))
    run(out, "resume", gen("killed"))
    kill_run(out, "engine_crash", gen("crashed"), engine=True)
    expected = sorted((out / "clean/games").glob("*.json"))
    for name in ("det", "killed", "crashed"):
        for part in expected:
            t.require(part.read_bytes() == (out / name / "games" / part.name).read_bytes(), f"nondeterminism: {name}/{part.name}")
    run(out, "reuse", gen("clean"))
    t.require(t.read_json(out / "clean/last_gen.json")["reused_games"] == 36, "resume did not reuse all games")
    # Detect a valid-JSON corruption as well as truncation, and regenerate exactly.
    damaged = out / "crashed/games" / expected[0].name
    game = t.read_json(damaged)
    game["seed"] += 1
    damaged.write_text(json.dumps(game))
    run(out, "repair", gen("crashed"))
    t.require(damaged.read_bytes() == expected[0].read_bytes(), "corrupt game repair changed data")
    run(out, "recipe_reject", gen("clean", ["--depth", "4"]), good=False)

    signs = replay(out / "clean")
    t.require(signs["winner_turns"] > 0 and signs["positive_winner_scores"] / signs["winner_turns"] > .5,
              "winner scores do not mostly have positive sign")
    print("T153 integration " + t.old.canonical(signs), flush=True)
    run(out, "qs", command + ["qs", "--work", str(out / "clean")])
    run(out, "qs_resume", command + ["qs", "--work", str(out / "clean")])
    shutil.copytree(out / "clean/games", out / "parallel/games")
    shutil.copyfile(out / "clean/manifest.json", out / "parallel/manifest.json")
    run(out, "qs_parallel", command + ["qs", "--work", str(out / "parallel"), "--procs", "4"])
    for part in (out / "clean/qs").glob("*.npz"):
        t.require(part.read_bytes() == (out / "parallel/qs" / part.name).read_bytes(), "QS process count changes parts")
    shutil.copytree(out / "clean/games", out / "inherit/games")
    shutil.copyfile(out / "clean/manifest.json", out / "inherit/manifest.json")
    run(out, "qs_inherit", command + ["qs", "--work", str(out / "inherit"), "--changed", "inherit"])
    flipped = 0
    for part in (out / "inherit/qs").glob("*.npz"):
        game = t.read_json(out / "inherit/games" / (part.stem + ".json"))
        with np.load(part) as z:
            for j, i in enumerate(z["source_rows"]):
                sign = -1 if z["flip"][j] else 1
                t.require(z["scores"][j] == game["rows"][i]["score"] * sign and
                          z["results"][j] == game["rows"][i]["result"] * sign, "odd leaf result/score sign")
                flipped += int(z["flip"][j])
    t.require(flipped > 0, "did not exercise odd QS leaf")
    run(out, "assemble", command + ["assemble", "--work", str(out / "clean"), "--out", str(out / "cache")])
    run(out, "holdout", command + ["assemble", "--work", str(out / "clean"), "--out", str(out / "holdout"),
                                   "--holdout-fraction", ".25"])
    for cache in (out / "cache", out / "holdout"):
        for s in ("train", "val"):
            x = np.load(cache / f"{s}_indices.npy")
            y, w = np.load(cache / f"{s}_evals.npy"), np.load(cache / f"{s}_wdl.npy")
            t.require(x.dtype == np.int32 and x.shape == (len(y), 77) and y.dtype == np.int16
                      and w.dtype == np.int8 and len(w) == len(y), "cache schema mismatch")
            t.require(np.array_equal(y, np.load(cache / "labels" / f"{s}_dlshogi_evals.npy"))
                      or (cache.name == "cache" and s == "val"), "training label alignment")
        train = np.load(cache / "train_indices.npy")
        t.require(len({r.tobytes() for r in train}) == len(train), "within-train duplicates")
    t.require((out / "cache/val_indices.npy").is_symlink(), "default shared val is not linked")
    missing = out / "clean/qs" / (expected[0].stem + ".npz")
    missing.rename(missing.with_suffix(".hidden"))
    run(out, "incomplete_reject", command + ["assemble", "--work", str(out / "clean"), "--out", str(out / "incomplete")], good=False)
    t.require(not (out / "incomplete").exists(), "partial assembly was published")
    missing.with_suffix(".hidden").rename(missing)
    run(out, "nodes_cap", gen("nodes_cap", ["--games", "2", "--depth", "9", "--nodes-cap", "1000", "--procs", "2"]))
    replay(out / "nodes_cap")
    train_ids, val_ids = (set(np.load(out / "holdout" / f"{s}_game_ids.npy")) for s in ("train", "val"))
    t.require(train_ids and val_ids and not train_ids.intersection(val_ids), "game-level split leaked")
    val_keys = {r.tobytes() for r in np.load(out / "holdout/val_indices.npy")}
    t.require(not val_keys.intersection(r.tobytes() for r in np.load(out / "holdout/train_indices.npy")),
              "duplicate position crosses game holdout")
    run(out, "compact", [sys.executable, "-B", "-m", "trainer.compact_cache", str(out / "cache"), str(out / "compact")])
    run(out, "compact_holdout", [sys.executable, "-B", "-m", "trainer.compact_cache", str(out / "holdout"), str(out / "holdout_compact")])
    # Independent terminal fixtures: both colors' forced mate and a king shuffle repetition.
    for sfen, move, winner in (("4k4/9/4P4/9/9/9/9/9/4K4 b G 1", "G*5b", 0),
                               ("4k4/9/9/9/9/9/4p4/9/4K4 w g 1", "G*5h", 1)):
        board = cshogi.Board(sfen)
        board.push_usi(move)
        t.require(t.terminal(board, 1, 256) == (winner, "no_legal_moves"), "mate winner sign")
    board = cshogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
    seen = Counter({tuple(board.sfen().split()[:3]): 1})
    for ply, move in enumerate(["5i6i", "5a6a", "6i5i", "6a5a"] * 3, 1):
        board.push_usi(move)
        key = tuple(board.sfen().split()[:3])
        seen[key] += 1
        end = t.terminal(board, ply, 256, seen[key])
        t.require(end == ((-1, "repetition_draw") if ply == 12 else None), "fourfold repetition policy")
    t.require(t.terminal(cshogi.Board(), 256, 256) == (-1, "max_ply_draw"), "max-ply policy")
    board = cshogi.Board("4K4/PPPPPPPPP/4+P4/9/9/9/9/9/4k4 b 2R2B4G4S4N4L8P 1")
    t.require(board.is_nyugyoku() and t.terminal(board, 0, 256) == (0, "declaration_27_win"),
              "declaration policy")
    result = dict(ok=True, games=36, signs=signs, odd_qs_leaves_checked=flipped,
                  byte_identical=["1 vs 4 processes / ID ranges", "driver SIGKILL resume", "engine SIGKILL retry", "corruption repair"],
                  qs_process_determinism="1 vs 4 processes: byte-identical NPZ parts",
                  compact_cache="both default shared validation and game holdout accepted; all elements round-trip",
                  incomplete_assembly="refused without publishing output", nodes_cap="real depth 9 + 1000 nodes exercised")
    t.json_write(out / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
