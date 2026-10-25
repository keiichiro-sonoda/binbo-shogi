"""Task 149: fail-closed 0cp gate: source BNNQ, written nn.bin, real engine."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

import cshogi
import numpy as np

from trainer import kp, yane_nnue, yane_to_binbo
from trainer.binbo_to_yane import parse_bnnq, write_nn_bin
from trainer.qat_eval import int_eval


def source_views(q):
    """Task 149: source-numbered tensors; never use the writer's scatter here."""
    fd, h1, h2 = q["fd"], q["hidden"], q["hidden2"]
    return {**q, "emb": q["emb"].reshape(q["total"], fd),
            "w0": q["w0"].reshape(h1, 2 * fd), "w1": q["w1"].reshape(h2, h1)}


def engine_evals(engine, eval_dir, sfens, log):
    """Task 149: sequential protocol commands, bounded runtime, Threads=1."""
    commands = ["usi", "setoption name Threads value 1", "setoption name USI_Hash value 64",
                "setoption name BookFile value no_book",
                "setoption name USI_OwnBook value false",
                f"setoption name EvalDir value {Path(eval_dir).resolve()}",
                "setoption name FV_SCALE value 16", "isready"]
    for sfen in sfens:
        commands.extend((f"position sfen {sfen}", "e"))
    commands.append("quit")
    result = subprocess.run([str(Path(engine).resolve())], input="\n".join(commands) + "\n",
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    Path(log).write_text(result.stdout)
    result.check_returncode()
    if "readyok" not in result.stdout or re.search(r"error|mismatch|failed|No such option", result.stdout, re.I):
        raise RuntimeError(f"engine load/protocol failed; see {log}")
    return np.array([int(v) for v in re.findall(r"^eval = (-?\d+)\s*$", result.stdout, re.M)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-bnnq", required=True)
    ap.add_argument("--engine", required=True, help="architecture-specific engine binary")
    ap.add_argument("--template")
    ap.add_argument("--positions", default="tools/parity_positions.sfen")
    ap.add_argument("--work-dir", required=True, help="persistent output directory for nn.bin and logs")
    args = ap.parse_args()
    work = Path(args.work_dir)
    (work / "eval").mkdir(parents=True, exist_ok=True)
    sfens = [s.strip() for s in Path(args.positions).read_text().splitlines() if s.strip()]
    if not sfens:
        raise ValueError("empty position set")
    written = work / "eval/nn.bin"
    write_nn_bin(args.from_bnnq, args.template, str(written))
    q = source_views(parse_bnnq(args.from_bnnq))
    src = np.array([int_eval(q, *kp.canonical_shared_halves(cshogi.Board(s)), q["fd"],
                             q["weight_scale_bits"], q["fv_scale"]) for s in sfens])
    p = yane_nnue.parse(str(written))
    yane_to_binbo._attach_int_views(p)
    dst = np.array([yane_to_binbo.yane_forward_int(p, cshogi.Board(s)) for s in sfens])
    eng = engine_evals(args.engine, written.parent, sfens, work / "engine.log")
    np.savetxt(work / "source_cp.txt", src, fmt="%d")
    np.savetxt(work / "written_cp.txt", dst, fmt="%d")
    np.savetxt(work / "engine_cp.txt", eng, fmt="%d")
    if eng.shape != src.shape:
        raise RuntimeError(f"RED: engine count {len(eng)} != positions {len(src)}")
    green = True
    for name, values in (("source BNNQ", src), ("written nn.bin", dst)):
        exact = int(np.count_nonzero(values == eng))
        good = exact == len(sfens)
        green &= good
        print(f"{name} vs engine: exact={exact}/{len(sfens)} "
              f"max|d|={int(np.abs(values-eng).max())} -> {'GREEN(0cp)' if good else 'RED'}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
