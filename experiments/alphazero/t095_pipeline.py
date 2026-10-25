#!/usr/bin/env python3
"""Task 095 sequential runner: the r=1.0 supervised-only control.

Task 094 found paired dElo vs WS-T of +21.3 (r=0.25) and +79.1 (r=0.50), both
FLAT.  Missing is the r=1.0 arm -- every one of the 512 policy updates per
iteration taken from the same 2M supervised set, zero self-play.  Without it,
"+79 is the self-play contribution" and "+79 is just more supervised training"
are not separable.  With it we get a four-point dose-response in r and, more
sharply, the DIRECT paired comparison r=0.50 vs r=1.00 in which the WS-T
baseline cancels: that difference is the marginal value of self-play.

Stage order (regression before equivalence: the backward-compatibility gate is
what licenses editing t094_loop_anchored.py in place, so it must fail fastest):

  0 preflight + t095_gate self-test        ~1 min   no GPU
  1 backward-compat regression             ~24 min  r=0.25/0.50, 1 iter, vs 094
  2 self-play-skip equivalence             ~31 min  force/skip/skip2 digests
  3 r=1.0 loop, 25 iterations              ~25 min  self-play skipped
  4 2M cache + WS-T rescore + r100 rescore ~21 min
  5 five 200-game matches vs gikou@6      ~196 min
  6 gate + verdict                         ~2 min

No auto-extension: a POSITIVE result stops and reports for human design.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t094_pipeline import load_tsv, run_logged  # noqa: E402
from t095_gate import (  # noqa: E402
    BLOCK_SEEDS,
    paired_delta_general,
    pairs_for_80pct_power,
    self_test,
    sign_class,
)

# Published Task 094 iter=1 metrics. The in-place edit to t094_loop_anchored.py
# must leave the r=0.25/0.50 code path numerically unchanged; stage 1 checks it.
T094_ITER1_REFERENCE = {
    "r025": {
        "visit_ce": 1.2837, "visit_entropy": 0.9802, "visit_argmax_agree": 0.7262,
        "anchor_ce": 1.5002, "anchor_top1": 0.5239, "episodes": 414,
    },
    "r050": {
        "visit_ce": 1.3001, "visit_entropy": 0.9828, "visit_argmax_agree": 0.7173,
        "anchor_ce": 1.4772, "anchor_top1": 0.5304, "episodes": 414,
    },
}
# A self-play iteration is a 512-step scan of 32-sim MCTS, so one ULP anywhere can
# flip an argmax and cascade. Measured 2026-07-29: most runs reproduce the Task 094
# iter=1 line bit-exactly, but runs intermittently diverge -- one r=0.25 run came
# back visit_ce +0.0079, visit_entropy +0.0057, episodes -2. That is ABOVE this
# tolerance, so a single run compared against a fixed reference false-alarms.
#
# The fix is retries, not a looser tolerance. A bit-exact match on seven coupled
# metrics is essentially impossible under a real code change, so "any attempt
# matches -> PASS" is sound; only a mismatch that REPRODUCES indicts the code.
REGRESSION_TOL = 0.005
REGRESSION_EPISODE_FRAC = 0.05
REGRESSION_ATTEMPTS = 2

VALUE_DIGEST = "2316885110a0e9532f83237160c9cdd8d8d2aa5d02b91995225f8e8d6520bbe1"
FLAT_POINT_MIN = -20.0  # reused from t094_pipeline for the dose-response label
# Held-out (m4_data val, 50k) top-1 of the arms r=1.0 is being compared against.
VAL_TOP1_REFERENCE = {"wst": 0.47158, "r025": 0.4781, "r050": 0.4796}

SERIES_ORDER = ("r000", "r025", "r050", "r100")


def parse_iter_metrics(log_path, iteration):
    """Pull the key=value fields off one `t094 iter=N ...` line."""
    text = Path(log_path).read_text(encoding="utf-8")
    match = re.search(rf"^t094 iter={iteration} (.*)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"no `t094 iter={iteration}` line in {log_path}")
    fields = {}
    for key, value in re.findall(r"([a-z_0-9]+)=([-+0-9.naeN]+)", match.group(1)):
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    return fields


def complete_tsv(path, expected_games):
    """True if the TSV is a finished, well-formed match; raise if it is partial.

    Also rejects an appended file: m2_match.py opens the TSV in append mode but
    always restarts the game counter at 0, so a rerun silently duplicates rows.
    """
    path = Path(path)
    if not path.exists():
        return False
    rows = load_tsv(path)
    if len(rows) != expected_games:
        raise RuntimeError(
            f"partial match TSV requires manual handling: {path} "
            f"({len(rows)} rows, expected {expected_games})"
        )
    if [row["game"] for row in rows] != list(range(expected_games)):
        raise RuntimeError(f"match TSV game column is not 0..{expected_games-1}: {path}")
    return True


def analyse(tsv_path, blocks, monitor, trajectory, regression, equivalence,
            *, loop_iters=25, games=200):
    """Stage 6, isolated so it can be exercised without waiting for a 4h run.

    `tsv_path` maps (block, series) -> TSV path. Returns the results dict plus
    the pieces the caller wants to print.
    """
    def blocks_for(series, block_ids):
        return [tsv_path[(block, series)] for block in block_ids]

    pooled_ids = list(range(1, blocks + 1))
    fresh_ids = [block for block in pooled_ids if block != 1]

    # Absolute placement of every arm on the shared block-1 openings.
    wst_block1 = blocks_for("wst", [1])
    vs_wst = {
        series: paired_delta_general(
            blocks_for(series, [1]), wst_block1, label=f"{series}_vs_wst_block1"
        )
        for series in SERIES_ORDER
    }

    delta_selfplay = {
        "pooled": dict(
            paired_delta_general(
                blocks_for("r050", pooled_ids), blocks_for("r100", pooled_ids),
                label="r050_minus_r100_pooled",
            ),
            contains_selected_block=True,
        ),
        "block1_only": dict(
            paired_delta_general(
                blocks_for("r050", [1]), blocks_for("r100", [1]),
                label="r050_minus_r100_block1",
            ),
            contains_selected_block=True,
        ),
    }
    if fresh_ids:
        # PRIMARY. Block 1 is excluded on purpose: +79.1 was reported as the
        # better of two arms, so it carries a selection bias of order +0.5 sigma
        # -- in the direction of "self-play helps", the hypothesis under test.
        delta_selfplay["primary_fresh_blocks"] = dict(
            paired_delta_general(
                blocks_for("r050", fresh_ids), blocks_for("r100", fresh_ids),
                label="r050_minus_r100_fresh",
            ),
            contains_selected_block=False,
        )
    primary_key = "primary_fresh_blocks" if fresh_ids else "block1_only"
    primary = delta_selfplay[primary_key]

    axis1 = sign_class(primary)
    above_wst = {series: vs_wst[series]["paired_ci95"][0] > 0.0 for series in vs_wst}
    if axis1 == "POS":
        verdict = ("SELFPLAY_POSITIVE_BOTH_ABOVE_WST" if above_wst["r100"]
                   else "SELFPLAY_POSITIVE_MARGINAL")
    elif axis1 == "NEG":
        verdict = ("SELFPLAY_HARMFUL_PURE_SUPERVISED_POSITIVE" if above_wst["r100"]
                   else "SELFPLAY_HARMFUL_NO_ABSOLUTE_GAIN")
    else:
        count = int(above_wst["r050"]) + int(above_wst["r100"])
        verdict = {
            2: "BOTH_ABOVE_WST_SELFPLAY_UNRESOLVED",
            1: "ONE_ARM_ABOVE_WST_SELFPLAY_UNRESOLVED",
            0: "NO_ARM_ABOVE_WST_SELFPLAY_UNRESOLVED",
        }[count]

    # Descriptive only -- never used to pick a branch. At a half-width of ~66 a
    # point-estimate reading of this curve is not decision-grade.
    curve = [vs_wst[series]["paired_delta_elo"] for series in SERIES_ORDER]
    rising = curve[0] < curve[1] < curve[2]
    if rising and curve[3] > curve[2]:
        dose_label = "MONOTONE_INCREASING"
    elif rising and abs(curve[3] - curve[2]) <= abs(FLAT_POINT_MIN):
        dose_label = "MONOTONE_TO_050_THEN_FLAT"
    elif rising and curve[3] < curve[2] - abs(FLAT_POINT_MIN):
        dose_label = "INTERIOR_OPTIMUM_AT_050"
    else:
        dose_label = "NON_MONOTONE"

    flags = []
    if not fresh_ids:
        flags.append("NO_SELECTION_FREE_BLOCK")
    if trajectory:
        first, last = min(trajectory), max(trajectory)
        val_first = trajectory[first].get("val_top1")
        val_last = trajectory[last].get("val_top1")
        anchor_first = trajectory[first].get("anchor_top1")
        anchor_last = trajectory[last].get("anchor_top1")
        if None not in (val_first, val_last, anchor_first, anchor_last):
            # Held-out top-1 falling while the training-set top-1 climbs is the
            # overfit signature; r=1.0 sees ~0.82 epochs of the 2M set (r=0.50: 0.41).
            if val_last < val_first - 0.002 and anchor_last > anchor_first + 0.01:
                flags.append("OVERFIT_SUSPECTED")
        # 085/089/090 all hit "best internally, not best externally". Pre-register
        # the flag so it is measured rather than rediscovered as a surprise.
        best_external = max(vs_wst, key=lambda series: vs_wst[series]["paired_delta_elo"])
        if val_last is not None and best_external != "r100":
            if val_last > max(VAL_TOP1_REFERENCE.values()):
                flags.append("INTERNAL_EXTERNAL_DECOUPLED")
    if monitor.get("value_model_digest") != VALUE_DIGEST:
        flags.append("VALUE_DIGEST_MISMATCH")

    reasons = {}
    illegal = 0
    for (block, series), path in sorted(tsv_path.items()):
        histogram = {}
        for row in load_tsv(path):
            histogram[row["reason"]] = histogram.get(row["reason"], 0) + 1
            if row["reason"].startswith(("illegal:", "engine_fail:")):
                illegal += 1
        reasons[f"block{block}_{series}"] = histogram
    if illegal:
        flags.append("ILLEGAL_OR_ENGINE_FAILURE")

    # Session/heterogeneity check: the same estimand computed block by block.
    # Block 1 is the selection-contaminated one; blocks 2 and 3 are the clean
    # replications, and disagreement between those two is a session effect.
    per_block = {
        str(block): paired_delta_general(
            blocks_for("r050", [block]), blocks_for("r100", [block]),
            label=f"r050_minus_r100_block{block}",
        )["paired_delta_elo"]
        for block in pooled_ids
    }

    results = {
        "task": 95,
        "config": {
            "seed": 20260714, "anchor_seed": 20260714, "sims": 32,
            "states_per_iter": 65_536, "updates_per_iter": 512,
            "loop_iters": loop_iters,
            "external_games_per_block": games, "external_sims": 128,
            "external_byoyomi_ms": 100,
            "blocks": {str(block): BLOCK_SEEDS[block] for block in pooled_ids},
            "bootstrap_samples": primary["bootstrap_samples"],
            "bootstrap_seed": primary["bootstrap_seed"],
        },
        "regression": regression,
        "equivalence": equivalence,
        "series_paths": {
            f"block{block}_{series}": str(path) for (block, series), path in tsv_path.items()
        },
        "vs_wst_block1": vs_wst,
        "above_wst": above_wst,
        "delta_selfplay": delta_selfplay,
        "delta_selfplay_primary_key": primary_key,
        "delta_selfplay_per_block": per_block,
        "dose_response_block1": dose_label,
        "dose_response_curve": dict(zip(SERIES_ORDER, curve)),
        "r100_monitor": monitor,
        "r100_trajectory": {str(key): value for key, value in trajectory.items()},
        "termination_reasons": reasons,
        "flags": flags,
        "pairs_for_80pct_power": pairs_for_80pct_power(primary),
        "verdict": verdict,
    }
    return results


def report(results):
    """Print the gate lines a human reads off the master log."""
    primary = results["delta_selfplay"][results["delta_selfplay_primary_key"]]
    print(
        f"GATE primary {primary['label']} n_pairs={primary['n_pairs']} "
        f"delta={primary['paired_delta_elo']:.2f} ci={primary['paired_ci95']} "
        f"class={sign_class(primary)} selection_free={not primary['contains_selected_block']}",
        flush=True,
    )
    for series in SERIES_ORDER:
        entry = results["vs_wst_block1"][series]
        print(
            f"GATE {series}_vs_wst delta={entry['paired_delta_elo']:.2f} "
            f"ci={entry['paired_ci95']} above_wst={results['above_wst'][series]}",
            flush=True,
        )
    curve = {key: round(value, 1) for key, value in results["dose_response_curve"].items()}
    print(f"DOSE {results['dose_response_block1']} curve={curve}", flush=True)
    print(f"PER_BLOCK delta_selfplay={results['delta_selfplay_per_block']}", flush=True)
    print(f"FLAGS {results['flags']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task095")
    parser.add_argument("--blocks", type=int, default=3, help="opening blocks to pool (1..3)")
    parser.add_argument("--games", type=int, default=200, help="games per block per series")
    parser.add_argument("--iters", type=int, default=25, help="r=1.0 loop iterations")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-equivalence", action="store_true")
    parser.add_argument("--on-equiv-fail", choices=("force", "abort"), default="force")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 3:
        parser.error("--blocks must be 1, 2 or 3")

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    teacher = Path(os.environ.get("TEACHER_DIR", data_root / "teacher"))
    torch_py = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    pgx_py = Path.home() / "task074_frameworks/venv_pgx/bin/python"
    anchor_data = clone / "m4_data"
    wst = clone / "t087_wst/ws2.ckpt"

    # ---- stage 0: preflight -------------------------------------------------
    for name in ("t094_loop_anchored.py", "t094_pipeline.py", "t095_gate.py",
                 "t095_equiv.py", "t095_pipeline.py"):
        source = repo / "experiments/alphazero" / name
        linked = clone / name
        if not linked.exists() or not os.path.samefile(source, linked):
            raise RuntimeError(
                f"clone {name} is not linked to experiments SSOT (run link_into_clone.sh)"
            )

    r100_ckpt_dir = models / f"{args.run_tag}_r100_ckpt"
    block1_tsv = {
        "wst": logs / "task093_wst_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r000": logs / "task093_iter25_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r025": logs / "task094_r025_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r050": logs / "task094_r050_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv",
    }
    series_ckpt = {
        "r050": models / "task094_r050_ckpt/000025.ckpt",
        "r100": r100_ckpt_dir / f"{args.iters:06d}.ckpt",
    }
    for path in [pgx_py, anchor_data / "train_obs.npy", wst, series_ckpt["r050"],
                 *block1_tsv.values()]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    logs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})

    self_test(logs)
    print("T095_PREFLIGHT_OK", flush=True)

    def loop_command(ckpt_dir, target, ratio, expected_iter_s, ckpt_interval=5,
                     monitor_interval=5):
        return [
            pgx_py, "t094_loop_anchored.py",
            f"max_num_iters={target}",
            "num_simulations=32",
            "selfplay_batch_size=128",
            "states_per_iter=65536",
            "training_batch_size=128",
            "learning_rate=0.0001",
            f"init_ckpt={wst}",
            f"ckpt_dir={ckpt_dir}",
            "resume=auto",
            f"ckpt_interval={ckpt_interval}",
            f"monitor_interval={monitor_interval}",
            f"expected_iter_s={expected_iter_s}",
            f"anchor_data_dir={anchor_data}",
            f"anchor_ratio={ratio}",
            "anchor_seed=20260714",
        ]

    # ---- stage 1: backward-compat regression --------------------------------
    # This is what licenses the in-place edit of t094_loop_anchored.py, so it
    # runs before anything expensive.
    regression = {"status": "SKIPPED"}
    if not args.skip_regression:

        def compare_to_reference(observed, reference, log):
            entry, worst, exceeded = {}, 0.0, []
            for key, want in reference.items():
                got = observed.get(key)
                if got is None:
                    raise RuntimeError(f"regression metric {key} missing from {log}")
                delta = got - want
                entry[key] = {"observed": got, "reference": want, "delta": delta}
                tol = want * REGRESSION_EPISODE_FRAC if key == "episodes" else REGRESSION_TOL
                if abs(delta) > tol:
                    entry[key]["exceeded"] = True
                    exceeded.append(key)
                if key != "episodes":
                    worst = max(worst, abs(delta))
            return entry, worst, exceeded

        arms_report = {}
        failed_arms = []
        for label, reference in T094_ITER1_REFERENCE.items():
            ratio = 0.25 if label == "r025" else 0.5
            attempts = []
            for attempt in range(1, REGRESSION_ATTEMPTS + 1):
                ckpt_dir = models / f"{args.run_tag}_regress_{label}_a{attempt}_ckpt"
                log = logs / f"{args.run_tag}_regress_{label}_a{attempt}.log"
                if not (ckpt_dir / "000001.ckpt").exists():
                    run_logged(loop_command(ckpt_dir, 1, ratio, 720, ckpt_interval=1),
                               log, cwd=clone, env=env)
                entry, worst, exceeded = compare_to_reference(
                    parse_iter_metrics(log, 1), reference, log
                )
                attempts.append({"attempt": attempt, "metrics": entry,
                                 "worst_abs_delta": worst, "exceeded": exceeded})
                print(f"T095_REGRESSION_ATTEMPT arm={label} attempt={attempt} "
                      f"worst_abs_delta={worst:.5f} exceeded={exceeded}", flush=True)
                if not exceeded:
                    break  # a bit-exact-ish match settles it; no need to retry
            matched = any(not item["exceeded"] for item in attempts)
            arms_report[label] = {"matched": matched, "attempts": attempts}
            if not matched:
                failed_arms.append(label)

        regression = {
            "status": "FAIL" if failed_arms else "PASS",
            "attempts_allowed": REGRESSION_ATTEMPTS,
            "failed_arms": failed_arms,
            "arms": arms_report,
        }
        print(f"T095_REGRESSION status={regression['status']} "
              f"failed_arms={failed_arms}", flush=True)
        if failed_arms:
            print("T095_PIPELINE_DONE verdict=REGRESSION_FAIL_ABORTED", flush=True)
            return 4

    # ---- stage 2: self-play-skip equivalence --------------------------------
    equivalence = {"status": "SKIPPED"}
    force_selfplay_for_main = False
    if not args.skip_equivalence:
        equiv_dirs = {}
        for label, force in (("force", True), ("skip", False), ("skip2", False)):
            ckpt_dir = models / f"{args.run_tag}_equiv_{label}_ckpt"
            equiv_dirs[label] = ckpt_dir
            if (ckpt_dir / "000002.ckpt").exists():
                print(f"SKIP existing equivalence checkpoint {ckpt_dir}", flush=True)
                continue
            run_env = dict(env)
            if force:
                run_env["T094_FORCE_SELFPLAY"] = "1"
            else:
                run_env.pop("T094_FORCE_SELFPLAY", None)
            run_logged(
                loop_command(ckpt_dir, 2, 1.0, 720 if force else 60,
                             ckpt_interval=1, monitor_interval=0),
                logs / f"{args.run_tag}_equiv_{label}.log", cwd=clone, env=run_env,
            )
        equiv_json = logs / f"{args.run_tag}_equivalence.json"
        try:
            run_logged([
                pgx_py, "t095_equiv.py",
                "--force-dir", equiv_dirs["force"],
                "--skip-dir", equiv_dirs["skip"],
                "--skip2-dir", equiv_dirs["skip2"],
                "--iters", "1,2",
                "--out", equiv_json,
            ], logs / f"{args.run_tag}_equiv_compare.log", cwd=clone, env=env)
        except RuntimeError:
            # t095_equiv.py exits 1 on EQUIV_FAIL but still writes the JSON. If
            # the file is absent the child really crashed -- do not swallow that.
            if not equiv_json.exists():
                raise
        equivalence = json.loads(equiv_json.read_text(encoding="utf-8"))
        print(f"T095_EQUIV status={equivalence['status']}", flush=True)
        if equivalence["status"] == "EQUIV_FAIL":
            if args.on_equiv_fail == "abort":
                print("T095_PIPELINE_DONE verdict=EQUIV_FAIL_ABORTED", flush=True)
                return 5
            # Fall back to the canonical path: generate self-play and discard it,
            # so the r=1.0 arm executes exactly the code r=0.25/0.50 executed.
            force_selfplay_for_main = True
            print("T095_EQUIV_FALLBACK running r=1.0 with generation ON", flush=True)

    # ---- stage 3: the r=1.0 arm ---------------------------------------------
    r100_log = logs / f"{args.run_tag}_r100_loop_to_{args.iters:03d}.log"
    if not series_ckpt["r100"].exists():
        run_env = dict(env)
        if force_selfplay_for_main:
            run_env["T094_FORCE_SELFPLAY"] = "1"
        run_logged(
            loop_command(r100_ckpt_dir, args.iters, 1.0,
                         720 if force_selfplay_for_main else 60),
            r100_log, cwd=clone, env=run_env,
        )
    else:
        print(f"SKIP existing loop checkpoint {series_ckpt['r100']}", flush=True)

    # ---- stage 4: 2M cache + rescore monitors -------------------------------
    work_root = Path("/tmp/binbo-task095")
    cache = work_root / "cache_2m"
    baseline_rescore = work_root / "rescore_wst"
    r100_rescore = work_root / "rescore_r100"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "meta.json").exists():
        run_logged([
            torch_py, "-m", "trainer.prepare_data",
            teacher / "run2m_c", cache, "--canonical-shared", "--val-shards", "4",
        ], logs / f"{args.run_tag}_prepare.log", cwd=repo, env=env)
    if not (baseline_rescore / "rescore_metrics.json").exists():
        run_logged([
            pgx_py, "t091_rescore.py",
            "--cache-dir", cache, "--ckpt", wst, "--out-dir", baseline_rescore,
        ], logs / f"{args.run_tag}_rescore_wst.log", cwd=clone, env=env)
    if not (r100_rescore / "rescore_metrics.json").exists():
        run_logged([
            pgx_py, "t091_rescore.py",
            "--cache-dir", cache, "--ckpt", series_ckpt["r100"], "--out-dir", r100_rescore,
            "--reference-value-dir", baseline_rescore,
            "--reference-policy-dir", baseline_rescore,
        ], logs / f"{args.run_tag}_rescore_r100.log", cwd=clone, env=env)
    monitor = json.loads((r100_rescore / "rescore_metrics.json").read_text(encoding="utf-8"))

    # ---- stage 5: matches ---------------------------------------------------
    # Block 1 gives r=1.0 the same 100 openings the other three arms already
    # played (so the dose-response curve is homogeneous). Blocks 2 and 3 add
    # selection-free pairs for r=0.50 vs r=1.00; WS-T is not replayed there
    # because it cancels in that difference. Within a block the series run
    # back-to-back so any session effect is common to both.
    schedule = [(1, "r100")]
    for block in (2, 3):
        if block <= args.blocks:
            schedule.extend([(block, "r050"), (block, "r100")])

    tsv_path = {(1, series): path for series, path in block1_tsv.items()}
    for block, series in schedule:
        tsv_path[(block, series)] = logs / (
            f"{args.run_tag}_{series}_iter{args.iters:03d}_vs_gikou6_"
            f"s128_b100_seed{BLOCK_SEEDS[block]}_n{args.games}.tsv"
        )

    for block, series in schedule:
        path = tsv_path[(block, series)]
        if complete_tsv(path, args.games):
            print(f"SKIP existing match TSV {path}", flush=True)
            continue
        run_logged([
            pgx_py, "m2_match.py",
            "--az-ckpt", series_ckpt[series],
            "--az-sims", "128",
            "--opponent", "gikou",
            "--games", str(args.games),
            "--byoyomi", "100",
            "--opening-plies", "6",
            "--max-plies", "320",
            "--seed", str(BLOCK_SEEDS[block]),
            "--tsv", path,
        ], Path(str(path).replace(".tsv", ".console.log")), cwd=clone, env=env)

    # ---- stage 6: gate + verdict -------------------------------------------
    trajectory = {}
    for iteration in range(5, args.iters + 1, 5):
        try:
            trajectory[iteration] = parse_iter_metrics(r100_log, iteration)
        except (ValueError, FileNotFoundError):
            continue

    results = analyse(
        tsv_path, args.blocks, monitor, trajectory, regression, equivalence,
        loop_iters=args.iters, games=args.games,
    )
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    print(f"T095_PIPELINE_DONE verdict={results['verdict']} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
