#!/usr/bin/env python3
"""Task 186: recalibrate the yardsticks for the new adopted eval task184_S_MR (docs/evaluation_protocol.md §7).

The student-based evals now score 0.77-0.85 against tanuki- @20k / AobaNNUE @10k and ~0.82 against micro @11k
(Task 184), so the Elo resolution is poor. Same engines, same rule as Task 108 / 127; only the budgets move.
Changes from Task 127, fixed before running: 208 games per rung (8 chunks x 26) on chunk seeds 100 apart
(Task 143: seeds 1 apart share openings), and micro (the confirmation anchor, Task 151) is recalibrated too.

  * candidate = task184_S_MR @100,000 nodes.
  * ladder (factor 2, the current rung included for reference): tanuki- 20k/40k/80k/160k, AobaNNUE 10k/20k/40k/80k,
    micro 11k/22k/44k/88k; chunk seeds std_frame.spaced_set(20299000), 6 random opening plies, max 320 plies.
  * pick per anchor the rung closest to 0.65 among rungs with 0.55 <= score <= 0.75 and 0 errors; else the closest
    with 0 errors (flag OUT_OF_BAND).
  * new reference frame: S_MR on the three sets s280/s281/s282 at the picked tanuki-/AobaNNUE budgets (4,800 games,
    played on the Xeon via xeon_frame). micro needs no stored frame (confirmations play both sides).
  * gates: 0 errors.
std_frame's ANCHORS / CONFIRM_ANCHORS are switched by a separate commit after this run (old games keep their names).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import xeon_frame  # noqa: E402
from t103_absolute import conditions  # noqa: E402

TAG = "task186"
CAND = "task184_S_MR"
LADDER = {"tanuki": (20_000, 40_000, 80_000, 160_000), "aoba": (10_000, 20_000, 40_000, 80_000),
          "micro": (11_000, 22_000, 44_000, 88_000)}
SEEDS, GAMES = std_frame.spaced_set(20299000), 26
TARGET, BAND = 0.65, (0.55, 0.75)


def main():
    repo, data, logs, binbo, yane, anchors = std_frame.paths()
    cand = data / "eval_dirs" / CAND
    env = dict(os.environ)
    ladder, picks, failures, flags = {}, {}, [], []
    commands, tsvs, index = [], [], []
    for name, rungs in LADDER.items():
        bin_b, eval_b = anchors[name]
        for nodes in rungs:
            names = [logs / f"{TAG}_ladder_{name}{nodes}_c{i:02d}_seed{s}_n{GAMES}.tsv" for i, s in enumerate(SEEDS)]
            for s, tsv in zip(SEEDS, names):
                commands.append([binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", cand, "--a-nodes", std_frame.NODES,
                                 "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes, "--seed", s, "--games", GAMES,
                                 "--opening-plies", "6", "--max-plies", "320"])
                tsvs.append(tsv)
            index.append((name, nodes, names))
    print(f"=== T186 STAGE ladder ({len(commands)} chunks, parallel {t106.PARALLEL}) ===", flush=True)
    t106.run_chunks(commands, tsvs, env, repo)
    for name, nodes, names in index:
        rows, bad = t106.combine(names, GAMES)
        cond = conditions(names)
        score = sum(r["score"] for r in rows) / len(rows)
        ladder[f"{name}@{nodes}"] = {"score": score, "games": len(rows), "errors": len(bad), "cond": cond}
        print(f"T186_LADDER {name}@{nodes} score={score:.3f} games={len(rows)} errors={len(bad)} "
              f"nodes={cond['a']['avg_nodes']:.0f}/{cond['b']['avg_nodes']:.0f}", flush=True)
    for name, rungs in LADDER.items():
        ok = [(n, ladder[f"{name}@{n}"]["score"]) for n in rungs if ladder[f"{name}@{n}"]["errors"] == 0]
        inband = [x for x in ok if BAND[0] <= x[1] <= BAND[1]]
        pool = inband or ok
        if not pool:
            failures.append(f"{name}: every rung had errors")
            continue
        if not inband:
            flags.append(f"OUT_OF_BAND_{name}")
        picks[name] = min(pool, key=lambda x: abs(x[1] - TARGET))[0]
        print(f"T186_PICK {name} nodes={picks[name]}", flush=True)
    frame = {}
    if not failures:
        which = {k: picks[k] for k in ("tanuki", "aoba")}
        out, bad = xeon_frame.remote_measure_sets(CAND, which=which)
        failures += bad
        for set_name, per in out.items():
            for anchor, rows in per.items():
                frame[f"{set_name}/{anchor}@{which[anchor]}"] = sum(r["score"] for r in rows) / len(rows)
        print(f"T186_FRAME {CAND} {json.dumps({k: round(v, 4) for k, v in frame.items()})}", flush=True)
    (logs / f"{TAG}_results.json").write_text(json.dumps({"ladder": ladder, "picks": picks, "frame": frame,
                                                          "flags": flags, "failures": failures}, indent=1, default=str),
                                             encoding="utf-8")
    print(f"T186_DONE picks={picks} flags={flags} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
