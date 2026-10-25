#!/usr/bin/env python3
"""Task 127: recalibrate the two anchors' node budgets for the new adopted eval.

docs/evaluation_protocol.md §7: when the adopted eval wins about 90% against the
anchors, raise their budgets and record a new yardstick. The adopted eval
`task126_student_64m_factor_e24` now scores 0.84-0.88 against tanuki- @10,000 and
AobaNNUE @5,000 (Task 126). Same engines, same procedure as Task 108; only the budgets
move (a changed budget is a new yardstick; the old games stay valid for old comparisons).

PROCEDURE (fixed before running; nothing here measures a new student):
  * candidate = the adopted eval `task126_student_64m_factor_e24`, `go nodes 100000`.
  * ladder (factor 2): tanuki- Haó 20k, 40k, 80k, 160k; AobaNNUE 10k, 20k, 40k, 80k.
    104 games per rung (4 x 26, seeds 20272401-04), yane-vs-yane-ex, 6 random opening
    plies, max 320 plies, colour-swapped pairs.
  * pick, per anchor, the rung whose score is closest to 0.65 among rungs with
    0.55 <= score <= 0.75 and 0 errors (Task 108's rule); if none qualifies, the closest
    to 0.65 with 0 errors (flag OUT_OF_BAND).
  * the new standard frame = the picked budgets on the standard seeds 20272001-08,
    800 games per anchor; the adopted eval's games there are its reference point and
    are reused by every later comparison (std_frame naming, nodes in the file name).
  * gates: 0 errors / illegal moves.

Interruptible (scripts/job.sh pause|resume task127).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t098_pipeline import paired_h2h_delta  # noqa: E402
from t103_absolute import conditions  # noqa: E402

TAG = "task127"
CAND = "task126_student_64m_factor_e24"
LADDER = {"tanuki": (20_000, 40_000, 80_000, 160_000), "aoba": (10_000, 20_000, 40_000, 80_000)}
LADDER_SEEDS, LADDER_GAMES = (20272401, 20272402, 20272403, 20272404), 26
TARGET, BAND = 0.65, (0.55, 0.75)


def main():
    repo, data, logs, binbo, yane, anchors = std_frame.paths()
    cand = data / "eval_dirs" / CAND
    for path in (binbo, yane, cand / "nn.bin", *[b for b, _ in anchors.values()]):
        if not path.exists():
            raise FileNotFoundError(path)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T127_PREFLIGHT_OK head={head} cand={CAND}@{std_frame.NODES} ladder={LADDER}", flush=True)
    failures, flags = [], []

    print("=== T127 STAGE ladder ===", flush=True)
    ladder, picks = {}, {}
    for name, rungs in LADDER.items():
        bin_b, eval_b = anchors[name]
        for nodes in rungs:
            tsvs = [logs / f"{TAG}_ladder_{name}{nodes}_w{i}_seed{s}_n{LADDER_GAMES}.tsv"
                    for i, s in enumerate(LADDER_SEEDS)]
            commands = [[binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", cand, "--a-nodes", std_frame.NODES,
                         "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes, "--seed", s,
                         "--games", LADDER_GAMES, "--opening-plies", "6", "--max-plies", "320"] for s in LADDER_SEEDS]
            t106.run_chunks(commands, tsvs, env, repo)
            rows, bad = t106.combine(tsvs, LADDER_GAMES)
            cond = conditions(tsvs)
            score = sum(r["score"] for r in rows) / len(rows)
            ladder[f"{name}@{nodes}"] = {"score": score, "errors": len(bad), "cond": cond}
            print(f"T127_LADDER {name}@{nodes} score={score:.3f} errors={len(bad)} "
                  f"nodes={cond['a']['avg_nodes']:.0f}/{cond['b']['avg_nodes']:.0f}", flush=True)
        ok = [(n, ladder[f"{name}@{n}"]["score"]) for n in rungs if ladder[f"{name}@{n}"]["errors"] == 0]
        inband = [x for x in ok if BAND[0] <= x[1] <= BAND[1]]
        pool = inband or ok
        if not pool:
            failures.append(f"{name}: every rung had errors")
            continue
        if not inband:
            flags.append(f"OUT_OF_BAND_{name}")
        picks[name] = min(pool, key=lambda x: abs(x[1] - TARGET))[0]
        print(f"T127_PICK {name} nodes={picks[name]}", flush=True)

    frame = {}
    if not failures:
        print("=== T127 STAGE new standard frame (adopted eval, picked budgets, seeds 20272001-08) ===", flush=True)
        std_frame.ANCHORS = dict(picks)   # the new yardstick; file names carry the node counts
        rows, bad = std_frame.measure(CAND, env)
        failures += bad
        for name, nodes in picks.items():
            h = paired_h2h_delta(rows[name], label=f"{CAND}_minus_{name}@{nodes}")
            frame[name] = {"nodes": nodes, "h2h": h}
            print(f"T127_FRAME {name}@{nodes} W-D-L={h['wins']}-{h['draws']}-{h['losses']} score={h['score']:.4f} "
                  f"elo={h['elo']:+.2f} ci={[round(v, 2) for v in h['ci95']]}", flush=True)
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 127, "git": {"head": head}, "candidate": {"eval": CAND, "nodes": std_frame.NODES},
                               "ladder": ladder, "picks": picks, "frame": frame, "failures": failures,
                               "flags": flags}, indent=2), encoding="utf-8")
    print(f"T127_DONE picks={picks} failures={failures} flags={flags} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
