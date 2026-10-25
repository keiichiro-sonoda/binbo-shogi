#!/usr/bin/env python3
"""Standard anchor frame (docs/evaluation_protocol.md): reusable per-eval anchor games.

Every eval is played at 100,000 nodes against tanuki- Hao @20,000 and AobaNNUE @10,000
nodes (Task 127; Task 108's yardstick was @10,000 / @5,000) on the standard seeds 20272001-08 (8 chunks x 100 games per anchor). Node-fixed
games reproduce, so an eval's games are played once and reused by every later
comparison: TSVs are named std_<eval>_<anchor><nodes>_c<i>_seed<s>_n100.tsv. For the
adopted eval of Task 113 (`task109_student_64m_e24`) the existing Task 113 games are
used as-is (same seeds, same eval, same budgets). The node count is in every file name,
so the old yardstick's games stay on disk and are never mixed into the new one.

Task 118: the eval side may run with a non-default FV_SCALE (`fv`); those games are
named std_<eval>_fv<fv>_... . Confirmation games on fresh seeds use another prefix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import t106_filter as t106  # noqa: E402
from t103_absolute import paired_series_delta  # noqa: E402

NODES = 100_000
ANCHORS = {"tanuki": 40_000, "aoba": 14_000}   # Task 186 (2026-10-24); reference eval task184_S_MR (scores 0.620 / 0.596)
ANCHORS_TASK127 = {"tanuki": 20_000, "aoba": 10_000}   # 2026-10-10 .. 10-24; games stay under their own names
OLD_ANCHORS = {"tanuki": 10_000, "aoba": 5_000}   # Task 108 yardstick (Tasks 108-126)
SEEDS = tuple(20272001 + i for i in range(8))
CHUNK = 100
FV_DEFAULT = 16   # YaneuraOu's default; our export folds the cp scale so that out/16 = trained cp
LEGACY = {"task109_student_64m_e24": "task113_frame_adopted"}   # games already on the standard seeds
t106.PARALLEL = int(os.environ.get("FRAME_PARALLEL", "8"))  # Task 141: 8 verified on Xubuntu (identical games, 83°C)
# Task 141 (protocol A, 2026-10-19): the decision pools three seed sets (2,400 paired games per anchor) so the
# adopted eval's frozen games are not one lucky/unlucky 800-game sample (Task 138: its standard-set score was
# ~16 Elo below the other two sets). The adopted eval has all three already; candidates play all three.
# Task 143 (2026-10-19, Codex review of Task 142): yane-vs-yane-ex draws the opening from seed + game/2, so
# chunk seeds 1 apart share almost every opening -- an 8-chunk set had only 57 distinct openings. The sets
# below space chunk seeds 100 apart (each chunk uses 50 openings), giving 400 distinct openings per set.
OLD_SEED_SETS = {"std": SEEDS,
                 "conf135": tuple(20273501 + i for i in range(8)),
                 "conf138": tuple(20273601 + i for i in range(8))}
SEED_SETS = {f"s{b // 1000 % 1000}": tuple(b + 100 * i for i in range(8)) for b in (20280000, 20281000, 20282000)}


# Task 151 (2026-10-21, author's decision): つようら王11micro (MIT, SFNN ka2 + progress8, another lineage than
# tanuki-) is used ONLY for adoption confirmation, not in the routine 3-set frame (it would add ~50% game time).
# Calibrated: the adopted task142 @100k scores 0.673 against micro @11,000 nodes (reports/task151_tsuyou_ladder.md).
CONFIRM_ANCHORS = {"micro": 15_500}   # Task 186 (2026-10-24, task184_S_MR scores 0.618); Task 151's 11_000 before


def spaced_set(base):
    """Task 143: a fresh confirmation set whose 8 chunks do not share openings (seeds 100 apart)."""
    return tuple(base + 100 * i for i in range(8))


def paths():
    repo = Path(os.path.realpath(__file__)).parents[2]
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data / "logs"))
    yane = data / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    aoba = data / "engines/aobannue/AobaNNUE"
    anchors = {"tanuki": (yane, data / "engines/yaneuraou/eval"),
               "aoba": (aoba / "source/YaneuraOu-by-gcc", aoba / "eval"),
               # Task 151: built from YaneuraOu GitHub 32165a8 (progress buckets), confirmation only
               "micro": (data / "engines/yaneuraou980/source/YaneuraOu-SFNN_ka2_2048_8_64_progress8",
                         next((data / "public/tsuyouraou11").glob("TuyouraOu11micro-*/"), None))}
    return repo, data, logs, repo / "target/release/binbo-shogi", yane, anchors


def tsv_names(logs, eval_name, anchor, fv=FV_DEFAULT, seeds=SEEDS, prefix="std", nodes=None):
    # Task 186: the file name must carry the budget actually played (a recalibration plays other budgets)
    nodes = nodes or {**ANCHORS, **CONFIRM_ANCHORS}[anchor]
    if prefix == "std" and fv == FV_DEFAULT and tuple(seeds) == SEEDS and nodes == OLD_ANCHORS.get(anchor):
        stem = LEGACY.get(eval_name, f"std_{eval_name}")
    else:
        stem = f"{prefix}_{eval_name}" + ("" if fv == FV_DEFAULT else f"_fv{fv}")
    return [logs / f"{stem}_{anchor}{nodes}_c{i:02d}_seed{s}_n{CHUNK}.tsv" for i, s in enumerate(seeds)]


def measure(eval_name, env=None, fv=FV_DEFAULT, seeds=SEEDS, prefix="std"):
    """Play (or reuse) the standard frame of `eval_name` (eval side at FV_SCALE `fv`);
    returns {anchor: rows}, failures."""
    repo, data, logs, binbo, yane, anchors = paths()
    eval_dir = data / "eval_dirs" / eval_name
    env = env or dict(os.environ)
    out, failures = {}, []
    for anchor, nodes in ANCHORS.items():
        bin_b, eval_b = anchors[anchor]
        tsvs = tsv_names(logs, eval_name, anchor, fv, seeds, prefix, nodes)
        fv_opt = [] if fv == FV_DEFAULT else ["--a-opt", f"FV_SCALE={fv}"]
        commands = [[binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", eval_dir, *fv_opt, "--a-nodes", NODES,
                     "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes, "--seed", s, "--games", CHUNK,
                     "--opening-plies", "6", "--max-plies", "320"] for s in seeds]
        print(f"=== STD_FRAME {eval_name} fv={fv} {prefix} vs {anchor}@{nodes} ===", flush=True)
        t106.run_chunks(commands, tsvs, env, repo)
        rows, bad = t106.combine(tsvs, CHUNK)
        out[anchor] = rows
        failures += bad
    return out, failures


def measure_sets(eval_name, env=None, fv=FV_DEFAULT, sets=None, which=None):
    """Task 141: play (or reuse) `eval_name` on every seed set and anchor, all chunks submitted together so
    `FRAME_PARALLEL` games run at once; returns {set: {anchor: rows}}, failures."""
    repo, data, logs, binbo, yane, anchors = paths()
    sets = sets or SEED_SETS
    use = which or ANCHORS  # {anchor: nodes}; CONFIRM_ANCHORS for the micro confirmation (Task 151)
    eval_dir = data / "eval_dirs" / eval_name
    env = env or dict(os.environ)
    fv_opt = [] if fv == FV_DEFAULT else ["--a-opt", f"FV_SCALE={fv}"]
    commands, tsvs, index = [], [], []
    for set_name, seeds in sets.items():
        for anchor, nodes in use.items():
            bin_b, eval_b = anchors[anchor]
            names = tsv_names(logs, eval_name, anchor, fv, seeds, set_name, nodes)
            for s, tsv in zip(seeds, names):
                commands.append([binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", eval_dir, *fv_opt,
                                 "--a-nodes", NODES, "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes,
                                 "--seed", s, "--games", CHUNK, "--opening-plies", "6", "--max-plies", "320"])
                tsvs.append(tsv)
            index.append((set_name, anchor, names))
    print(f"=== STD_FRAME_SETS {eval_name} fv={fv} sets={list(sets)} parallel={t106.PARALLEL} ===", flush=True)
    t106.run_chunks(commands, tsvs, env, repo)
    out, failures = {}, []
    for set_name, anchor, names in index:
        rows, bad = t106.combine(names, CHUNK)
        out.setdefault(set_name, {})[anchor] = rows
        failures += bad
    return out, failures


def delta_sets(cand, base, label):
    """Task 141: pooled (cand - base) over all seed sets and both anchors, paired on shared openings;
    also per anchor and per set (descriptive)."""
    sets = list(base)
    names = list(base[sets[0]])
    pooled = {n: [r for s in sets for r in cand[s][n]] for n in names}
    pooled_b = {n: [r for s in sets for r in base[s][n]] for n in names}
    total, per, bad = delta(pooled, pooled_b, label)
    per_set = {s: delta(cand[s], base[s], f"{label} {s}")[0] for s in sets}
    return total, per, per_set, bad


def delta(cand, base, label):
    """Pooled (cand - base) over both anchors, paired on the shared openings."""
    names = list(base)  # the anchors that were played (ANCHORS, or CONFIRM_ANCHORS)
    a = [r for n in names for r in cand[n]]
    b = [r for n in names for r in base[n]]
    bad = [] if len(a) == len(b) and all(p["opening"] == q["opening"] and p["color"] == q["color"]
                                         for p, q in zip(a, b)) else [f"{label}: openings differ"]
    per = {n: paired_series_delta(cand[n], base[n], label=f"{label} {n}") for n in names}
    return paired_series_delta(a, b, label=label), per, bad
