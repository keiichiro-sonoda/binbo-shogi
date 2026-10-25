#!/usr/bin/env python3
"""Task 186b: midpoint rungs (x sqrt 2) for the anchors with no in-band rung in Task 186 (fixed before running):
AobaNNUE 14,000 and micro 15,500, 208 games each on Task 186's chunk seeds; re-pick with the same rule over all rungs."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
from t186_recalibrate import BAND, CAND, GAMES, LADDER, SEEDS, TAG, TARGET  # noqa: E402

MID = {"aoba": 14_000, "micro": 15_500}


def main():
    repo, data, logs, binbo, yane, anchors = std_frame.paths()
    cand, env = data / "eval_dirs" / CAND, dict(os.environ)
    prev = {}   # Task 186's rungs, recomputed from its TSVs (its results file is written only after its frame)
    for name in MID:
        for nodes in LADDER[name]:
            names = [logs / f"{TAG}_ladder_{name}{nodes}_c{i:02d}_seed{s}_n{GAMES}.tsv" for i, s in enumerate(SEEDS)]
            rows, bad = t106.combine(names, GAMES)
            prev[f"{name}@{nodes}"] = {"score": sum(r["score"] for r in rows) / len(rows), "errors": len(bad)}
    commands, tsvs, index = [], [], []
    for name, nodes in MID.items():
        bin_b, eval_b = anchors[name]
        names = [logs / f"{TAG}_ladder_{name}{nodes}_c{i:02d}_seed{s}_n{GAMES}.tsv" for i, s in enumerate(SEEDS)]
        for s, tsv in zip(SEEDS, names):
            commands.append([binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", cand, "--a-nodes", std_frame.NODES,
                             "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes, "--seed", s, "--games", GAMES,
                             "--opening-plies", "6", "--max-plies", "320"])
            tsvs.append(tsv)
        index.append((name, nodes, names))
    t106.run_chunks(commands, tsvs, env, repo)
    picks = {}
    for name, nodes, names in index:
        rows, bad = t106.combine(names, GAMES)
        score = sum(r["score"] for r in rows) / len(rows)
        print(f"T186B_LADDER {name}@{nodes} score={score:.3f} games={len(rows)} errors={len(bad)}", flush=True)
        rungs = {int(k.split("@")[1]): v["score"] for k, v in prev.items() if k.startswith(name + "@") and v["errors"] == 0}
        if not bad:
            rungs[nodes] = score
        inband = {n: sc for n, sc in rungs.items() if BAND[0] <= sc <= BAND[1]}
        pool = inband or rungs
        picks[name] = min(pool, key=lambda n: abs(pool[n] - TARGET))
        print(f"T186B_PICK {name} nodes={picks[name]} inband={bool(inband)} rungs={rungs}", flush=True)
    (logs / f"{TAG}b_results.json").write_text(json.dumps({"picks": picks}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
