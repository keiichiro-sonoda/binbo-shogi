#!/usr/bin/env python3
"""Task 103c: where do the 4M students stand in absolute terms?

Tasks 100-102 only ever played the search-label student against the raw-label
student. This measures both 4M students (Task 102a) against the two external
frames the project uses: the champion (task064_32m_lr1e-5, the 32M gikou@d8
distillation) via yane-vs-yane, and the gikou@6 anchor via yane-vs-gikou.
Descriptive, pre-registered as such: no verdict beyond the numbers, except the
champion-frame difference (search - raw), which is a second, independent read
of Task 102a's +124 taken through a different opponent.

Protocol: the Task 100 match protocol (100 ms byoyomi, Threads 1, 6 random
opening plies, max 320 plies, 4 workers x 200 games, seeds 20270911-14), so
the four series share openings and can be paired across students.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t094_pipeline import elo_from_score  # noqa: E402
from t098_pipeline import BOOTSTRAP_SAMPLES, load_yane_tsv, paired_h2h_delta, validate_h2h  # noqa: E402

H2H_WORKERS, GAMES_PER_WORKER = 4, 200
SEEDS = (20270911, 20270912, 20270913, 20270914)
PAIRED_SEED = 20270930


def run_workers(commands, tsvs, env, cwd):
    procs = []
    for command, tsv in zip(commands, tsvs):
        if tsv.exists() and "elapsed_ms=" in tsv.read_text(encoding="utf-8"):
            continue
        print(f"RUN {' '.join(map(str, command))} log={tsv}", flush=True)
        handle = tsv.open("w", encoding="utf-8")
        procs.append((subprocess.Popen([str(c) for c in command], cwd=cwd, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT), handle, tsv))
    for process, handle, tsv in procs:
        rc = process.wait()
        handle.close()
        if rc != 0:
            raise RuntimeError(f"worker failed rc={rc}: {tsv}")


def combine(tsvs):
    rows, offset = [], 0
    bad = []
    for tsv in tsvs:
        part = load_yane_tsv(tsv)
        shape = validate_h2h(part, tsv, expected_games=GAMES_PER_WORKER)
        bad += shape["bad_reasons"]
        for row in part:
            row = dict(row)
            row["game"] += offset
            rows.append(row)
        offset += len(part)
    validate_h2h(rows, "combined", expected_games=GAMES_PER_WORKER * len(tsvs))
    return rows, bad


def conditions(tsvs):
    depth = {"a": [], "b": []}
    nodes = {"a": [], "b": []}
    for tsv in tsvs:
        for line in Path(tsv).read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 21 or not fields[0].isdigit():
                continue
            depth["a"].append(float(fields[8])); nodes["a"].append(float(fields[9]))
            depth["b"].append(float(fields[12])); nodes["b"].append(float(fields[13]))
    return {side: {"avg_depth": float(np.mean(depth[side])), "avg_nodes": float(np.mean(nodes[side]))}
            for side in ("a", "b")}


def paired_series_delta(rows_a, rows_b, label):
    """(a - b) Elo over shared opening pairs: both series use the same seeds, so
    pair k of series a and pair k of series b are the same opening."""
    pa = np.array([r["score"] for r in rows_a]).reshape(-1, 2).mean(axis=1)
    pb = np.array([r["score"] for r in rows_b]).reshape(-1, 2).mean(axis=1)
    if pa.shape != pb.shape:
        raise RuntimeError("series lengths differ")
    n = len(pa)
    point = elo_from_score(pa.mean()) - elo_from_score(pb.mean())
    rng = np.random.default_rng(PAIRED_SEED)
    deltas = np.empty(BOOTSTRAP_SAMPLES)
    chunk = 10_000
    for start in range(0, BOOTSTRAP_SAMPLES, chunk):
        end = min(start + chunk, BOOTSTRAP_SAMPLES)
        idx = rng.integers(0, n, size=(end - start, n))
        sa = np.clip(pa[idx].mean(axis=1), 1e-9, 1 - 1e-9)
        sb = np.clip(pb[idx].mean(axis=1), 1e-9, 1 - 1e-9)
        deltas[start:end] = 400.0 * (np.log10(sa / (1 - sa)) - np.log10(sb / (1 - sb)))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"label": label, "delta_elo": float(point), "ci95": [float(low), float(high)],
            "n_pairs": int(n), "bootstrap_seed": PAIRED_SEED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task103c")
    parser.add_argument("--students", nargs="+", default=["task102a_student_search", "task102a_student_raw"])
    parser.add_argument("--champion", default="task064_32m_lr1e-5")
    parser.add_argument("--opponents", nargs="+", default=["champion", "gikou6"],
                        choices=["champion", "gikou6"])
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    eval_dirs = data_root / "eval_dirs"
    binbo = repo / "target/release/binbo-shogi"
    yane = data_root / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    gikou = data_root / "engines/gikou/gikou"
    champion = eval_dirs / args.champion
    for path in (binbo, yane, gikou, champion / "nn.bin", *[eval_dirs / s / "nn.bin" for s in args.students]):
        if not path.exists():
            raise FileNotFoundError(path)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T103C_PREFLIGHT_OK head={head} students={args.students} champion={args.champion}", flush=True)

    results = {"task": "103c", "git": {"head": head, "pushed": False}, "series": {}, "protocol":
               "100ms/Threads1, 6 opening plies, max 320, 4 workers x 200 games, seeds 20270911-14; "
               "gikou anchor = DepthLimit 6 (yane-vs-gikou default), 100 ms think"}
    rows_by = {}
    for student in args.students:
        for opponent in args.opponents:
            tag = f"{args.run_tag}_{student}_vs_{opponent}"
            tsvs, commands = [], []
            for index, seed in enumerate(SEEDS[:H2H_WORKERS]):
                tsv = logs / f"{tag}_w{index}_seed{seed}_n{GAMES_PER_WORKER}.tsv"
                if opponent == "champion":
                    command = [binbo, "yane-vs-yane", yane, eval_dirs / student, champion, "100", seed,
                               GAMES_PER_WORKER, "6", "320", "0", "0", "0.05", "0.05"]
                else:
                    command = [binbo, "yane-vs-gikou", yane, eval_dirs / student, gikou, "100", "6", seed,
                               GAMES_PER_WORKER, "6", "320", "100", "0", "0", "0.05", "0.05"]
                tsvs.append(tsv); commands.append(command)
            run_workers(commands, tsvs, env, repo)
            rows, bad = combine(tsvs)
            h2h = paired_h2h_delta(rows, label=f"{student}_minus_{opponent}")
            cond = conditions(tsvs)
            rows_by[(student, opponent)] = rows
            results["series"][f"{student}_vs_{opponent}"] = {
                "h2h": h2h, "conditions": cond, "bad_reasons": bad, "tsvs": [str(t) for t in tsvs]}
            print(f"T103C_SERIES {student} vs {opponent}: W-D-L={h2h['wins']}-{h2h['draws']}-{h2h['losses']} "
                  f"score={h2h['score']:.4f} elo={h2h['elo']:+.2f} ci={[round(v, 2) for v in h2h['ci95']]} "
                  f"depth={cond['a']['avg_depth']:.2f}/{cond['b']['avg_depth']:.2f} "
                  f"nodes={cond['a']['avg_nodes']:.0f}/{cond['b']['avg_nodes']:.0f} bad={len(bad)}", flush=True)

    if len(args.students) == 2:
        s0, s1 = args.students
        for opponent in args.opponents:
            delta = paired_series_delta(rows_by[(s0, opponent)], rows_by[(s1, opponent)],
                                        label=f"({s0} - {s1}) in the {opponent} frame")
            results[f"frame_delta_{opponent}"] = delta
            print(f"T103C_FRAME_DELTA {opponent}: {delta['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in delta['ci95']]}", flush=True)
    failures = [k for k, v in results["series"].items() if v["bad_reasons"]]
    results["prerequisites"] = {"failures": failures}
    results["verdict"] = "DESCRIPTIVE_OK" if not failures else "INVALID"
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"T103C_PIPELINE_DONE verdict={results['verdict']} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
