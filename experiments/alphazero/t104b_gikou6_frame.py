#!/usr/bin/env python3
"""Task 104b: confirm SEARCH_BEATS_CHAMPION in the gikou@6 frame before adoption.

Task 104's decision table row 1 says "confirm in absolute frames, then adopt".
Task 104 S2 already played the 32M search student against gikou@6 (800 games,
seeds 20270911-14, +334.1 [+300.5, +372.3]); the champion has no series under
the same protocol (only Task 064's 83%, other seeds and game count). This runs
exactly that missing series -- champion vs gikou@6, same protocol and seeds --
and differences the two over shared opening pairs. The search student is NOT
replayed: its S2 TSVs are the other arm.

PRE-REGISTERED (committed before the run; the champion series is unseen):
  * one variable: the yane eval (search student vs champion), the opponent and
    the openings held fixed. Protocol = Task 100/103c/104 S2: 100 ms byoyomi,
    Threads 1, 6 random opening plies, max 320 plies, 4 workers x 200 games,
    seeds 20270911-14; gikou@6 = DepthLimit 6, 100 ms think.
  * statistic: frame delta = Elo(search student vs gikou@6) - Elo(champion vs
    gikou@6), paired bootstrap over the 400 shared opening pairs
    (t103_absolute.paired_series_delta, 200,000 samples, seed 20270930).
    Planned half-width ~45 Elo (each series alone ~36; pairing may shrink it).
  * prerequisites (else INVALID, non-terminal): 800 games in each arm, no
    illegal move / engine failure, every pair k has the same opening in both
    arms, gikou depth ~6 in both arms.
  * decision table:
      0 prerequisites failed  -> INVALID
      1 CI low  > 0           -> FRAME_CONFIRMS   (adoption condition met; the
                                  eval swap itself waits for human approval)
      2 CI high < 0           -> FRAME_REVERSES   (do not adopt; read Task 104's
                                  +134 as matchup, next = ensemble-label arm)
      3 CI straddles 0        -> FRAME_INCONCLUSIVE (hold adoption; next = both
                                  arms +800 games on seeds 20271011-14, same rule
                                  on the 1600-game arms)
  * descriptive only: the champion's own gikou@6 Elo (vs Task 064's 83%), the
    ratio frame delta / direct +134.2.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t098_pipeline import paired_h2h_delta  # noqa: E402
from t103_absolute import combine, conditions, paired_series_delta, run_workers  # noqa: E402

SEEDS = (20270911, 20270912, 20270913, 20270914)
GAMES_PER_WORKER = 200
CHAMPION = "task064_32m_lr1e-5"
SEARCH_ARM_TAG = "task104_s2_vs_gikou6"          # Task 104 S2, reused as is
DIRECT_ELO = 134.20                              # Task 104 primary, for the ratio only
GIKOU_DEPTH_TOL = 0.1


def verdict_of(ci):
    low, high = ci
    if low > 0:
        return "FRAME_CONFIRMS"
    if high < 0:
        return "FRAME_REVERSES"
    return "FRAME_INCONCLUSIVE"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default="task104b")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data_root / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    gikou = data_root / "engines/gikou/gikou"
    champion = data_root / "eval_dirs" / CHAMPION
    search_tsvs = [logs / f"{SEARCH_ARM_TAG}_w{i}_seed{s}_n{GAMES_PER_WORKER}.tsv" for i, s in enumerate(SEEDS)]
    champ_tsvs = [logs / f"{args.run_tag}_champion_vs_gikou6_w{i}_seed{s}_n{GAMES_PER_WORKER}.tsv"
                  for i, s in enumerate(SEEDS)]
    for path in (binbo, yane, gikou, champion / "nn.bin", *search_tsvs):
        if not path.exists():
            raise FileNotFoundError(path)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not args.dry_run:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    commands = [[binbo, "yane-vs-gikou", yane, champion, gikou, "100", "6", seed, GAMES_PER_WORKER,
                 "6", "320", "100", "0", "0", "0.05", "0.05"] for seed in SEEDS]
    if args.dry_run:
        for command, tsv in zip(commands, champ_tsvs):
            print("DRYRUN", " ".join(map(str, command)), f"log={tsv}")
        print(f"DRYRUN search arm (reused) = {[t.name for t in search_tsvs]}")
        return 0
    print(f"T104B_PREFLIGHT_OK head={head} champion={CHAMPION} search_arm={SEARCH_ARM_TAG}", flush=True)
    print("=== T104B STAGE champion vs gikou@6 (800 games) ===", flush=True)
    run_workers(commands, champ_tsvs, dict(os.environ), repo)

    print("=== T104B STAGE frame delta ===", flush=True)
    rows_s, bad_s = combine(search_tsvs)
    rows_c, bad_c = combine(champ_tsvs)
    cond_s, cond_c = conditions(search_tsvs), conditions(champ_tsvs)
    failures = [f"search arm: {b}" for b in bad_s] + [f"champion arm: {b}" for b in bad_c]
    mismatched = [k for k, (a, b) in enumerate(zip(rows_s, rows_c))
                  if a["opening"] != b["opening"] or a["color"] != b["color"]]
    if len(rows_s) != len(rows_c) or mismatched:
        failures.append(f"opening/colour mismatch at {len(mismatched)} games (first {mismatched[:5]})")
    for name, cond in (("search", cond_s), ("champion", cond_c)):
        if abs(cond["b"]["avg_depth"] - 6.0) > GIKOU_DEPTH_TOL:
            failures.append(f"{name} arm: gikou depth {cond['b']['avg_depth']:.2f} != 6")

    h_s = paired_h2h_delta(rows_s, label="search_student_32M_minus_gikou6")
    h_c = paired_h2h_delta(rows_c, label="champion_minus_gikou6")
    delta = paired_series_delta(rows_s, rows_c, label="(search student - champion) in the gikou@6 frame")
    verdict = "INVALID" if failures else verdict_of(delta["ci95"])
    print(f"T104B_CHAMPION_VS_GIKOU6 W-D-L={h_c['wins']}-{h_c['draws']}-{h_c['losses']} "
          f"score={h_c['score']:.4f} elo={h_c['elo']:+.2f} ci={[round(v, 2) for v in h_c['ci95']]} "
          f"depth={cond_c['a']['avg_depth']:.2f}/{cond_c['b']['avg_depth']:.2f}", flush=True)
    print(f"T104B_SEARCH_VS_GIKOU6 elo={h_s['elo']:+.2f} ci={[round(v, 2) for v in h_s['ci95']]} (Task 104 S2)",
          flush=True)
    print(f"T104B_FRAME_DELTA delta={delta['delta_elo']:+.2f} ci={[round(v, 2) for v in delta['ci95']]} "
          f"ratio_to_direct={delta['delta_elo'] / DIRECT_ELO:.2f}", flush=True)
    if failures:
        print(f"T104B_PREREQ_FAILED {failures}", flush=True)
    print(f"T104B_VERDICT verdict={verdict}", flush=True)

    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps({
        "task": "104b", "git": {"head": head, "pushed": False},
        "protocol": "100ms/Threads1, 6 opening plies, max 320, 4 workers x 200 games, seeds 20270911-14; "
                    "gikou@6 = DepthLimit 6, 100 ms think",
        "search_arm": {"h2h": h_s, "conditions": cond_s, "tsvs": [str(t) for t in search_tsvs]},
        "champion_arm": {"h2h": h_c, "conditions": cond_c, "tsvs": [str(t) for t in champ_tsvs]},
        "frame_delta": delta, "ratio_to_direct": delta["delta_elo"] / DIRECT_ELO,
        "failures": failures, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T104B_DONE verdict={verdict} results={output}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
