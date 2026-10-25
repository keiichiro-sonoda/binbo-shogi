#!/usr/bin/env python3
"""Task 108: calibrate the two external anchors (tanuki- Haó, AobaNNUE) by node budget.

Decision of 10/5 (docs/yardstick_survey_2026-10-05.md section 5): replace the
saturating gikou@7 (the adopted eval scores 91%) with two node-limited anchors of
lineages unlike our teachers. This fixes each anchor's node budget once; after
that the anchors never change (a changed budget is a new yardstick).

PROCEDURE (fixed before running; nothing here measures a new student):
  * candidate = the adopted eval `task107_student_64m` on YaneuraOu (halfkp_256),
    `go nodes 100000` (~65 ms/move; every future comparison uses the same budget).
  * anchors, ladder of node budgets (factor 2):
      tanuki- Haó (same YaneuraOu build, FV_SCALE default): 10k, 20k, 40k, 80k
      AobaNNUE (its own halfkp_768x2-16-64 build, FV_SCALE 40 default): 5k, 10k, 20k, 40k, 80k
  * ladder: 104 games per rung (4 x 26: pairs must not straddle workers; seeds 20271501-04), yane-vs-yane-ex,
    6 random opening plies, max 320 plies, colour-swapped pairs.
  * pick, per anchor, the rung whose score is closest to 0.65 among rungs with
    0.55 <= score <= 0.75 and 0 errors; if none qualifies, the closest to 0.65
    with 0 errors (flag OUT_OF_BAND).
  * confirm the picked rung with 400 games (4 x 100, seeds 20271511-14) and record
    its score/Elo with the paired-bootstrap CI; this is the anchor's reference
    point for the adopted eval.
  * gates: 0 errors / illegal moves; node-fixed games are reproducible (checked
    on 4 games before the ladder).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import t106_filter as t106  # noqa: E402
from t098_pipeline import paired_h2h_delta  # noqa: E402
from t103_absolute import conditions  # noqa: E402

CAND, CAND_NODES = "task107_student_64m", 100_000
LADDER = {"tanuki": (10_000, 20_000, 40_000, 80_000), "aoba": (5_000, 10_000, 20_000, 40_000, 80_000)}
LADDER_SEEDS, LADDER_GAMES = (20271501, 20271502, 20271503, 20271504), 26
CONFIRM_SEEDS, CONFIRM_GAMES = (20271511, 20271512, 20271513, 20271514), 100
TARGET, BAND = 0.65, (0.55, 0.75)


def main():
    repo = Path(os.path.realpath(__file__)).parents[2]
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data / "logs"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    aoba = data / "engines/aobannue/AobaNNUE"
    anchors = {"tanuki": (yane, data / "engines/yaneuraou/eval"),
               "aoba": (aoba / "source/YaneuraOu-by-gcc", aoba / "eval")}
    cand = data / "eval_dirs" / CAND
    for path in (binbo, yane, cand / "nn.bin", *[b for b, _ in anchors.values()], *[e / "nn.bin" for _, e in anchors.values()]):
        if not path.exists():
            raise FileNotFoundError(path)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T108_PREFLIGHT_OK head={head} cand={CAND}@{CAND_NODES} nodes", flush=True)

    def series(name, nodes, seeds, games, stage):
        bin_b, eval_b = anchors[name]
        tsvs = [logs / f"task108_{stage}_{name}{nodes}_w{i}_seed{s}_n{games}.tsv" for i, s in enumerate(seeds)]
        commands = [[binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", cand, "--a-nodes", CAND_NODES,
                     "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes, "--seed", s, "--games", games,
                     "--opening-plies", "6", "--max-plies", "320"] for s in seeds]
        t106.run_chunks(commands, tsvs, env, repo)
        rows, bad = t106.combine(tsvs, games)
        return rows, bad, conditions(tsvs), tsvs

    print("=== T108 STAGE ladder ===", flush=True)
    ladder, picks, failures, flags = {}, {}, [], []
    for name, rungs in LADDER.items():
        for nodes in rungs:
            rows, bad, cond, _ = series(name, nodes, LADDER_SEEDS, LADDER_GAMES, "ladder")
            score = sum(r["score"] for r in rows) / len(rows)
            ladder[f"{name}@{nodes}"] = {"score": score, "errors": len(bad), "cond": cond}
            print(f"T108_LADDER {name}@{nodes} score={score:.3f} errors={len(bad)} "
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
        print(f"T108_PICK {name} nodes={picks[name]}", flush=True)

    print("=== T108 STAGE confirm ===", flush=True)
    confirm = {}
    for name, nodes in picks.items():
        rows, bad, cond, tsvs = series(name, nodes, CONFIRM_SEEDS, CONFIRM_GAMES, "confirm")
        failures += bad
        h = paired_h2h_delta(rows, label=f"{CAND}@{CAND_NODES}_minus_{name}@{nodes}")
        confirm[name] = {"nodes": nodes, "h2h": h, "conditions": cond, "tsvs": [str(t) for t in tsvs]}
        print(f"T108_CONFIRM {name}@{nodes} W-D-L={h['wins']}-{h['draws']}-{h['losses']} score={h['score']:.4f} "
              f"elo={h['elo']:+.2f} ci={[round(v, 2) for v in h['ci95']]}", flush=True)
    out = logs / "task108_results.json"
    out.write_text(json.dumps({"task": 108, "git": {"head": head}, "candidate": {"eval": CAND, "nodes": CAND_NODES},
                               "ladder": ladder, "picks": picks, "confirm": confirm, "failures": failures,
                               "flags": flags}, indent=2), encoding="utf-8")
    print(f"T108_DONE picks={picks} failures={failures} flags={flags} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
