#!/usr/bin/env python3
"""Task 146: light real-engine tests, restricted to a fresh dryrun directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t146_self_relabel as t  # noqa: E402

np, cshogi = t.np, t.cshogi
SCRIPT = Path(t.__file__)


def invoke(work, name, argv, success=True):
    with (work / (name + ".log")).open("w") as log:
        p = subprocess.run([sys.executable, "-B", str(SCRIPT)] + list(map(str, argv)),
                           stdout=log, stderr=subprocess.STDOUT, timeout=180)
    t.require((p.returncode == 0) == success, f"unexpected exit={p.returncode}: {name}.log")


def interrupt(work, name, argv, crash_engine):
    """Task 146: kill during a real part, not between completed checkpoints."""
    path = work / (name + ".log")
    with path.open("w") as log:
        p = subprocess.Popen([sys.executable, "-B", str(SCRIPT)] + list(map(str, argv)) + ["--verbose"],
                             stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                content = path.read_text()
                match = re.search(r"part_start train 0 engine_pid=(\d+)", content)
                if match:
                    break
                t.require(p.poll() is None and time.monotonic() < deadline, "no live part_start")
                time.sleep(.01)
            time.sleep(.02)
            target = int(match[1]) if crash_engine else p.pid
            os.kill(target, signal.SIGKILL)
            result = p.wait(timeout=90)
            t.require(result == 0 if crash_engine else result == -signal.SIGKILL, "wrong killed/retry status")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
    if crash_engine:
        t.require("T146 retry" in path.read_text(), "engine crash did not exercise retry")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    work = args.out.resolve()
    t.require(work.is_relative_to((t.DATA / "work/task146_dryrun").resolve()), "test output must be in dryrun")
    work.mkdir(parents=True, exist_ok=False)
    results = {}
    # Exhaustive round trip against the actual C++ integer output formula.
    for value in range(-32000, 32001):
        cp = (-1 if value < 0 else 1) * (100 * abs(value) // 90)
        assert t.internal_score("cp", str(cp)) == value
    assert t.internal_score("mate", "-0") == -32000
    results["internal_roundtrip"] = 64001
    options = t.parser().parse_args(["bench", "--depth", "6"])
    cfg = t.config(options)
    cfg.update(timeout=30, retries=2)
    engine = t.Engine(cfg)
    try:
        mates = []
        for sfen in ("4k4/9/4P4/9/9/9/9/9/4K4 b G 1",
                     "4k4/9/9/9/9/9/4p4/9/4K4 w g 1"):
            board = cshogi.Board(sfen)
            oracle = board.mate_move_in_1ply()
            assert oracle
            score, depth, nodes, mate, raw, move = engine.search(sfen)
            assert mate and score == 32000
            assert move in {cshogi.move_to_usi(m) for m in board.legal_moves}
            board.push_usi(move)
            assert board.is_game_over() and board.is_check()
            loss = engine.search(board.sfen())
            assert loss[0] == -32000 and loss[3]
            mates.append(dict(sfen=sfen, move=move, score=score, mate_plies=raw,
                              checkmated_score=loss[0]))
        results["mate_both_colours"] = mates
    finally:
        engine.close()
    for tag, procs in (("det_a", 1), ("det_b", 4)):
        invoke(work, tag, ["bench", "--rows", 200, "--depth", 8, "--procs", procs,
                           "--out", work / (tag + ".json")])
    with np.load(work / "det_a.npz") as a, np.load(work / "det_b.npz") as b:
        for key in ("rows", "scores", "depths", "nodes", "mates", "raw_scores"):
            assert np.array_equal(a[key], b[key]), key
    results["determinism_d8_200rows_1vs4procs"] = "exact scores/raw/depth/nodes/mates"
    run_args = ["run", "--depth", 6, "--procs", 2, "--part-rows", 64,
                "--train-rows", 128, "--val-rows", 64, "--log-seconds", 1]
    for name in ("clean", "killed", "crashed"):
        argv = run_args + ["--work", work / name]
        if name != "clean":
            interrupt(work, name, argv, crash_engine=name == "crashed")
        if name != "crashed":
            invoke(work, name + "_run", argv)
        # Must reuse every completed part on the next run.
        invoke(work, name + "_reuse", argv)
        assert "reused=192 pending_parts=0" in (work / (name + "_reuse.log")).read_text()
        invoke(work, name + "_assemble", ["assemble", "--work", work / name, "--out", work / (name + "_labels")])
    for split in ("train", "val"):
        baseline = np.load(work / "clean_labels" / f"{split}_dlshogi_evals.npy")
        assert baseline.dtype == np.int16
        for name in ("killed", "crashed"):
            assert np.array_equal(baseline, np.load(work / (name + "_labels") / f"{split}_dlshogi_evals.npy"))
    # Changed recipe and incomplete assembly must fail, never leave plausible zeros.
    invoke(work, "recipe_mismatch", ["run", "--nodes", 20000, "--work", work / "clean",
                                    "--train-rows", 128, "--val-rows", 64, "--part-rows", 64], success=False)
    part = sorted((work / "clean/parts").glob("train_*.npz"))[0]
    saved = part.read_bytes()
    part.write_bytes(b"interrupted/corrupt zip")
    invoke(work, "incomplete", ["assemble", "--work", work / "clean", "--out", work / "incomplete_labels"], success=False)
    assert not (work / "incomplete_labels").exists()
    invoke(work, "repair", run_args + ["--work", work / "clean"])
    with np.load(part) as repaired:
        expected = np.load(work / "clean_labels/train_dlshogi_evals.npy")[repaired["rows"]]
        assert np.array_equal(repaired["scores"], expected)
    assert saved  # original completed zip existed before deliberate corruption
    results.update(run_assemble={"train": 128, "val": 64},
                   parent_sigkill_resume="identical", engine_sigkill_retry="identical",
                   completed_reuse="192/192", corrupt_part_repair="identical",
                   incomplete_assemble="rejected", changed_recipe="rejected")
    t.json_write(work / "results.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
