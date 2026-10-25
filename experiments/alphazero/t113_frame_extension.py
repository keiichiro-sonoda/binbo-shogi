#!/usr/bin/env python3
"""Task 113: more anchor games so +20..+30 effects can be told apart.

Tasks 110-112 all ended "direct match significant, pooled anchors straddle 0": with
400 games per anchor (800 pooled) the half-width is ~+-40 Elo, too wide for the
+20..+30 effects at stake. Result mixing pointed positive in both Task 111 and 112,
and the Task 112 student (C) sat at +37.4 [-0.6, +75.8] over the adopted eval.

PRE-REGISTERED (committed before the run; no new game is seen):
  * arms: adopted `task109_student_64m_e24`, control `task110_student_anneal`,
    C `task112_student_cpmix`; all at 100,000 nodes (Task 108 standard).
  * NEW anchor games only decide (the earlier 400/anchor/arm were already seen;
    pooling them would be optional stopping): each arm vs tanuki- @10,000 and
    AobaNNUE @5,000, 800 games each = 8 chunks x 100 (seeds 20272001-08, the same
    for every arm so openings pair). Pooled over both anchors: 1600 games per arm
    (half-width ~+-28).
  * DIRECT = C vs adopted, 1600 games = 16 x 100 (seeds 20272101-16), new.
  * question 1 (adoption): C vs adopted
      direct CI low > 0 and new pooled frame (C - adopted) CI low > 0 -> ADOPT_C
      direct CI high < 0                                               -> C_WORSE
      otherwise                                                        -> C_NOT_CONFIRMED
  * question 2 (mechanism): new pooled frame (C - control) = result mixing
      CI low > 0 -> MIX_CONFIRMED / CI high < 0 -> MIX_REFUTED / else MIX_UNRESOLVED
  * descriptive only: (control - adopted) new pooled = the lr anneal; every delta
    again on old + new games combined; per-anchor deltas.
  * gates: no illegal move / engine failure, openings identical across arms.
  * node-fixed games do not depend on load; 4 chunks at a time (6 overheated: guard stop at 100°C).

Interruptible (scripts/job.sh pause|resume task113).
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
from t103_absolute import paired_series_delta  # noqa: E402

TAG = "task113"
ARMS = {"adopted": "task109_student_64m_e24", "control": "task110_student_anneal", "C": "task112_student_cpmix"}
OLD_FRAME_TAG = {"adopted": "task109", "control": "task110", "C": "task112"}   # earlier 400/anchor, seeds 20271511-14
NODES = 100_000
ANCHORS = {"tanuki": 10_000, "aoba": 5_000}
FRAME_SEEDS = tuple(20272001 + i for i in range(8))
DIRECT_SEEDS = tuple(20272101 + i for i in range(16))
OLD_SEEDS = (20271511, 20271512, 20271513, 20271514)
CHUNK = 100
t106.PARALLEL = 4   # 6 hit 100°C and the thermal guard stopped the run (10/6 12:24); node-fixed games are unaffected


def main():
    repo = Path(os.path.realpath(__file__)).parents[2]
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data / "logs"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    aoba = data / "engines/aobannue/AobaNNUE"
    anchor_bin = {"tanuki": (yane, data / "engines/yaneuraou/eval"),
                  "aoba": (aoba / "source/YaneuraOu-by-gcc", aoba / "eval")}
    evals = {arm: data / "eval_dirs" / name for arm, name in ARMS.items()}
    for path in (binbo, yane, *[e / "nn.bin" for e in evals.values()]):
        if not path.exists():
            raise FileNotFoundError(path)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ)
    print(f"T113_PREFLIGHT_OK head={head} arms={ARMS}", flush=True)
    failures = []

    def ex(eval_a, bin_b, eval_b, nodes_b, seed):
        return [binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", eval_a, "--a-nodes", NODES,
                "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes_b, "--seed", seed, "--games", CHUNK,
                "--opening-plies", "6", "--max-plies", "320"]

    new, old = {}, {}   # (arm, anchor) -> rows
    for arm in ARMS:
        for name, nodes in ANCHORS.items():
            print(f"=== T113 STAGE frame {arm} vs {name}@{nodes} (8 x 100) ===", flush=True)
            bin_b, eval_b = anchor_bin[name]
            tsvs = [logs / f"{TAG}_frame_{arm}_{name}{nodes}_c{i:02d}_seed{s}_n{CHUNK}.tsv" for i, s in enumerate(FRAME_SEEDS)]
            t106.run_chunks([ex(evals[arm], bin_b, eval_b, nodes, s) for s in FRAME_SEEDS], tsvs, env, repo)
            rows, bad = t106.combine(tsvs, CHUNK)
            failures += bad
            new[(arm, name)] = rows
            h = paired_h2h_delta(rows, label=f"{arm}_vs_{name}")
            print(f"T113_FRAME {arm} {name}@{nodes} score={h['score']:.4f} elo={h['elo']:+.1f}", flush=True)
            otsvs = [logs / f"{OLD_FRAME_TAG[arm]}_frame_{name}{nodes}_w{i}_seed{s}_n100.tsv" for i, s in enumerate(OLD_SEEDS)]
            old[(arm, name)] = t106.combine(otsvs, 100)[0]

    print("=== T113 STAGE direct (C vs adopted, 16 x 100) ===", flush=True)
    dtsvs = [logs / f"{TAG}_direct_c{i:02d}_seed{s}_n{CHUNK}.tsv" for i, s in enumerate(DIRECT_SEEDS)]
    t106.run_chunks([ex(evals["C"], yane, evals["adopted"], NODES, s) for s in DIRECT_SEEDS], dtsvs, env, repo)
    drows, bad = t106.combine(dtsvs, CHUNK)
    failures += bad
    direct = paired_h2h_delta(drows, label="C_minus_adopted_direct")
    print(f"T113_DIRECT C-vs-adopted W-D-L={direct['wins']}-{direct['draws']}-{direct['losses']} "
          f"score={direct['score']:.4f} elo={direct['elo']:+.2f} ci={[round(v, 2) for v in direct['ci95']]}", flush=True)

    def pooled(src, x, y, label):
        a = src[(x, "tanuki")] + src[(x, "aoba")]
        b = src[(y, "tanuki")] + src[(y, "aoba")]
        if len(a) != len(b) or any(p["opening"] != q["opening"] or p["color"] != q["color"] for p, q in zip(a, b)):
            failures.append(f"{label}: openings differ")
        return paired_series_delta(a, b, label=label)

    both = {k: old[k] + new[k] for k in new}
    res = {}
    for x, y in (("C", "adopted"), ("C", "control"), ("control", "adopted")):
        res[f"{x}-{y}"] = {
            "new": pooled(new, x, y, f"({x} - {y}) new pooled"),
            "old_plus_new": pooled(both, x, y, f"({x} - {y}) old+new pooled"),
            "new_per_anchor": {n: paired_series_delta(new[(x, n)], new[(y, n)], label=f"({x} - {y}) new {n}")
                               for n in ANCHORS},
        }
        r = res[f"{x}-{y}"]
        print(f"T113_POOLED {x}-{y} new={r['new']['delta_elo']:+.2f} ci={[round(v, 2) for v in r['new']['ci95']]} "
              f"| old+new={r['old_plus_new']['delta_elo']:+.2f} ci={[round(v, 2) for v in r['old_plus_new']['ci95']]} "
              f"| tanuki={r['new_per_anchor']['tanuki']['delta_elo']:+.1f} aoba={r['new_per_anchor']['aoba']['delta_elo']:+.1f}",
              flush=True)

    lo, hi = direct["ci95"]
    q1 = ("INVALID" if failures else "ADOPT_C" if lo > 0 and res["C-adopted"]["new"]["ci95"][0] > 0
          else "C_WORSE" if hi < 0 else "C_NOT_CONFIRMED")
    m = res["C-control"]["new"]["ci95"]
    q2 = "INVALID" if failures else "MIX_CONFIRMED" if m[0] > 0 else "MIX_REFUTED" if m[1] < 0 else "MIX_UNRESOLVED"
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 113, "git": {"head": head, "pushed": False}, "arms": ARMS,
                               "direct": {"h2h": direct, "tsvs": [str(t) for t in dtsvs]},
                               "pooled": res, "failures": failures, "q1": q1, "q2": q2}, indent=2), encoding="utf-8")
    print(f"T113_VERDICT q1={q1} q2={q2} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
