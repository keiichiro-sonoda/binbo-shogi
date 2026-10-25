#!/usr/bin/env python3
"""Task 094 sequential runner: two replay arms, external gates, optional extension."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


BASELINE_SLOPE = -47.7
FLAT_POINT_MIN = -20.0
BOOTSTRAP_SAMPLES = 200_000
BOOTSTRAP_SEED = 20260724


def run_logged(command, log_path, *, cwd, env):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {' '.join(map(str, command))} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed rc={return_code}: {command}")


def parse_iter_seconds(path):
    values = [
        float(value)
        for value in re.findall(r"^t094 iter=[0-9]+ iter_s=([0-9.]+)", Path(path).read_text(
            encoding="utf-8"
        ), re.MULTILINE)
    ]
    if not values:
        raise ValueError(f"no t094 iteration timings in {path}")
    return values


def load_tsv(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 6:
            raise ValueError(f"bad TSV row in {path}: {line!r}")
        rows.append({
            "game": int(fields[0]),
            "color": fields[1],
            "score": float(fields[2]),
            "reason": fields[3],
            "plies": int(fields[4]),
            "opening": fields[5],
        })
    return rows


def elo_from_score(score):
    score = min(max(float(score), 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10(score / (1.0 - score))


def paired_delta(candidate_path, baseline_path):
    candidate = load_tsv(candidate_path)
    baseline = load_tsv(baseline_path)
    if len(candidate) != 200 or len(baseline) != 200:
        raise ValueError("paired gate requires exactly 200 games per series")
    for cand, base in zip(candidate, baseline):
        for key in ("game", "color", "opening"):
            if cand[key] != base[key]:
                raise ValueError(f"paired structure mismatch for {key}: {cand} != {base}")
    candidate_pairs = np.asarray([
        (candidate[index]["score"] + candidate[index + 1]["score"]) / 2.0
        for index in range(0, 200, 2)
    ])
    baseline_pairs = np.asarray([
        (baseline[index]["score"] + baseline[index + 1]["score"]) / 2.0
        for index in range(0, 200, 2)
    ])
    candidate_score = float(candidate_pairs.mean())
    baseline_score = float(baseline_pairs.mean())
    point = elo_from_score(candidate_score) - elo_from_score(baseline_score)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    chunk = 10_000
    for start in range(0, BOOTSTRAP_SAMPLES, chunk):
        end = min(start + chunk, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, len(candidate_pairs), size=(end - start, len(candidate_pairs)))
        cand_scores = candidate_pairs[indices].mean(axis=1)
        base_scores = baseline_pairs[indices].mean(axis=1)
        cand_scores = np.clip(cand_scores, 1e-9, 1.0 - 1e-9)
        base_scores = np.clip(base_scores, 1e-9, 1.0 - 1e-9)
        deltas[start:end] = (
            400.0 * np.log10(cand_scores / (1.0 - cand_scores))
            - 400.0 * np.log10(base_scores / (1.0 - base_scores))
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    scores = np.asarray([row["score"] for row in candidate])
    return {
        "wins": int((scores == 1.0).sum()),
        "draws": int((scores == 0.5).sum()),
        "losses": int((scores == 0.0).sum()),
        "score": candidate_score,
        "baseline_score": baseline_score,
        "absolute_elo": elo_from_score(candidate_score),
        "paired_delta_elo": point,
        "paired_ci95": [float(low), float(high)],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def classify_gate(result):
    point = result["paired_delta_elo"]
    low, high = result["paired_ci95"]
    if low > 0.0:
        return "POSITIVE"
    if low <= 0.0 <= high and point > FLAT_POINT_MIN:
        return "FLAT"
    if point <= BASELINE_SLOPE:
        return "BASELINE_OR_WORSE"
    return "INCONCLUSIVE_IMPROVED"


def ensure_complete_tsv(path):
    path = Path(path)
    if not path.exists():
        return False
    rows = load_tsv(path)
    if len(rows) == 200:
        return True
    raise RuntimeError(f"partial match TSV requires manual handling: {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--run-tag", default="task094")
    args = parser.parse_args()

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
    baseline_tsv = logs / "task093_wst_vs_gikou6_s128_b100_seed20260702_n200.tsv"
    source_script = repo / "experiments/alphazero/t094_loop_anchored.py"
    clone_script = clone / "t094_loop_anchored.py"
    if not clone_script.exists() or not os.path.samefile(source_script, clone_script):
        raise RuntimeError("clone t094_loop_anchored.py is not linked to experiments SSOT")
    for required in (pgx_py, anchor_data / "train_obs.npy", wst, baseline_tsv):
        if not required.exists():
            raise FileNotFoundError(required)

    logs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
    })
    arms = {
        "r025": {"ratio": 0.25, "ckpt_dir": models / f"{args.run_tag}_r025_ckpt"},
        "r050": {"ratio": 0.50, "ckpt_dir": models / f"{args.run_tag}_r050_ckpt"},
    }

    def run_loop(label, target):
        arm = arms[label]
        checkpoint = arm["ckpt_dir"] / f"{target:06d}.ckpt"
        if checkpoint.exists():
            print(f"SKIP existing loop checkpoint {checkpoint}", flush=True)
            return None
        log = logs / f"{args.run_tag}_{label}_loop_to_{target:03d}.log"
        run_logged([
            pgx_py,
            "t094_loop_anchored.py",
            f"max_num_iters={target}",
            "num_simulations=32",
            "selfplay_batch_size=128",
            "states_per_iter=65536",
            "training_batch_size=128",
            "learning_rate=0.0001",
            f"init_ckpt={wst}",
            f"ckpt_dir={arm['ckpt_dir']}",
            "resume=auto",
            "ckpt_interval=5",
            "monitor_interval=5",
            "expected_iter_s=720",
            f"anchor_data_dir={anchor_data}",
            f"anchor_ratio={arm['ratio']}",
            "anchor_seed=20260714",
        ], log, cwd=clone, env=env)
        return log

    timing_logs = {}
    for label in arms:
        timing_logs[label] = run_loop(label, 5)
    medians = {}
    for label, log in timing_logs.items():
        if log is None:
            previous = logs / f"{args.run_tag}_{label}_loop_to_005.log"
            log = previous
        medians[label] = float(np.median(parse_iter_seconds(log)))
    projected_loop_hours = sum(medians.values()) * 25.0 / 3600.0
    print(
        f"BUDGET_CHECK medians={medians} projected_25x2={projected_loop_hours:.2f}h "
        f"budget={args.budget_hours:.2f}h",
        flush=True,
    )
    if projected_loop_hours > args.budget_hours:
        print("T094_BUDGET_PAUSE", flush=True)
        return 3

    for label in arms:
        run_loop(label, 25)

    work_root = Path("/tmp/binbo-task094")
    cache = work_root / "cache_2m"
    baseline_rescore = work_root / "rescore_wst"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "meta.json").exists():
        run_logged([
            torch_py,
            "-m",
            "trainer.prepare_data",
            teacher / "run2m_c",
            cache,
            "--canonical-shared",
            "--val-shards",
            "4",
        ], logs / f"{args.run_tag}_prepare.log", cwd=repo, env=env)
    if not (baseline_rescore / "rescore_metrics.json").exists():
        run_logged([
            pgx_py,
            "t091_rescore.py",
            "--cache-dir",
            cache,
            "--ckpt",
            wst,
            "--out-dir",
            baseline_rescore,
        ], logs / f"{args.run_tag}_rescore_wst.log", cwd=clone, env=env)

    results = {
        "task": 94,
        "config": {
            "seed": 20260714,
            "sims": 32,
            "states_per_iter": 65_536,
            "updates_per_iter": 512,
            "external_games": 200,
            "external_sims": 128,
            "external_byoyomi_ms": 100,
            "external_seed": 20260702,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "baseline_slope_elo_per_25": BASELINE_SLOPE,
        },
        "timing_median_s": medians,
        "projected_loop_hours": projected_loop_hours,
        "arms": {},
    }

    def gate(label, iteration):
        arm = arms[label]
        checkpoint = arm["ckpt_dir"] / f"{iteration:06d}.ckpt"
        rescore_dir = work_root / f"rescore_{label}_{iteration:03d}"
        if not (rescore_dir / "rescore_metrics.json").exists():
            run_logged([
                pgx_py,
                "t091_rescore.py",
                "--cache-dir",
                cache,
                "--ckpt",
                checkpoint,
                "--out-dir",
                rescore_dir,
                "--reference-value-dir",
                baseline_rescore,
                "--reference-policy-dir",
                baseline_rescore,
            ], logs / f"{args.run_tag}_{label}_rescore_{iteration:03d}.log", cwd=clone, env=env)
        match_tsv = logs / (
            f"{args.run_tag}_{label}_iter{iteration:03d}_vs_gikou6_"
            "s128_b100_seed20260702_n200.tsv"
        )
        match_log = match_tsv.with_suffix(".console.log")
        if not ensure_complete_tsv(match_tsv):
            run_logged([
                pgx_py,
                "m2_match.py",
                "--az-ckpt",
                checkpoint,
                "--az-sims",
                "128",
                "--opponent",
                "gikou",
                "--games",
                "200",
                "--byoyomi",
                "100",
                "--opening-plies",
                "6",
                "--max-plies",
                "320",
                "--seed",
                "20260702",
                "--tsv",
                match_tsv,
            ], match_log, cwd=clone, env=env)
        paired = paired_delta(match_tsv, baseline_tsv)
        paired["classification"] = classify_gate(paired)
        with (rescore_dir / "rescore_metrics.json").open(encoding="utf-8") as file:
            paired["monitor"] = json.load(file)
        paired["checkpoint"] = str(checkpoint)
        paired["match_tsv"] = str(match_tsv)
        print(f"GATE {label} iter={iteration} {paired}", flush=True)
        return paired

    positive = []
    for label in arms:
        result = gate(label, 25)
        results["arms"][label] = {"ratio": arms[label]["ratio"], "iter25": result}
        if result["classification"] == "POSITIVE":
            positive.append(label)

    if positive:
        selected = max(
            positive,
            key=lambda label: results["arms"][label]["iter25"]["paired_delta_elo"],
        )
        results["extension_arm"] = selected
        run_loop(selected, 50)
        result50 = gate(selected, 50)
        results["arms"][selected]["iter50"] = result50
        results["verdict"] = (
            "PASS_POSITIVE_EXTERNAL_TRANSFER"
            if result50["classification"] == "POSITIVE"
            else "POSITIVE_AT_25_NOT_CONFIRMED_AT_50"
        )
    else:
        classes = [
            results["arms"][label]["iter25"]["classification"] for label in arms
        ]
        if all(value == "BASELINE_OR_WORSE" for value in classes):
            results["verdict"] = "FAIL_ANCHOR_DID_NOT_STOP_FORGETTING"
        elif "FLAT" in classes:
            results["verdict"] = "FORGETTING_STOPPED_NO_POSITIVE_TRANSFER"
        else:
            results["verdict"] = "INCONCLUSIVE_IMPROVED_STOPPED_PER_RULE"

    output = logs / f"{args.run_tag}_results.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"T094_PIPELINE_DONE verdict={results['verdict']} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
