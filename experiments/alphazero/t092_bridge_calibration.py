#!/usr/bin/env python3
"""Task 092-0 Steps 1-2: calibrate the training-only NNUE policy bridge."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from t091_pipeline import parse_teacher_match, run_logged


TASK_SEED = 20260717
HEADS = ("none", "linear", "mlp")


def best_metrics(model_dir: Path) -> dict:
    history = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    best_epoch = int(history["best_epoch"])
    record = next(item for item in history["history"] if int(item["epoch"]) == best_epoch)
    return {"best_epoch": best_epoch, **record["val"]}


def train_model(py_torch: Path, repo: Path, cache: Path, rescore: Path,
                models: Path, logs: Path, head: str, label: str = "wst") -> Path:
    tag = f"task0920_{head.replace('-', '_')}_{label}"
    model_dir = models / tag
    history = model_dir / "history.json"
    if history.exists():
        saved = json.loads(history.read_text(encoding="utf-8"))["args"]
        if saved.get("policy_head", "linear") != head:
            raise AssertionError(f"stale model head mismatch: {model_dir}")
        return model_dir
    command = [
        py_torch, "-m", "trainer.train_nnue_policy_aux", cache,
        "--value-dir", rescore, "--out", model_dir,
        "--policy-head", head,
    ]
    if head != "none":
        command.extend(["--policy-dir", rescore])
        if label == "random":
            command.extend(["--policy-stem", "policy_random_actions"])
    run_logged(command, logs / f"{tag}.log", cwd=repo, env=os.environ)
    return model_dir


def export_eval(py_torch: Path, repo: Path, model_dir: Path, eval_dir: Path,
                yane_eval: Path, logs: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    bnnq = model_dir / "nnue_best.bnnq"
    run_logged([
        py_torch, "-m", "trainer.export_nnue", model_dir,
        "--weights", "nnue_best.pt", "--out", bnnq.name,
    ], logs / f"task0920_export_{model_dir.name}.log", cwd=repo, env=os.environ)
    run_logged([
        py_torch, "-m", "trainer.binbo_to_yane", "--from-bnnq", bnnq,
        "--template", yane_eval / "nn.bin", "--out", eval_dir / "nn.bin",
    ], logs / f"task0920_write_{model_dir.name}.log", cwd=repo, env=os.environ)
    gate_log = logs / f"task0920_gate_{model_dir.name}.log"
    run_logged([
        repo / "scripts/binbo_to_yane.sh", bnnq,
    ], gate_log, cwd=repo, env=os.environ)
    gate_text = gate_log.read_text(encoding="utf-8")
    if "RED" in gate_text or gate_text.count("GREEN") < 2:
        raise AssertionError(f"0cp gate failed: {model_dir}")


def write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    teacher = Path(os.environ.get("TEACHER_DIR", data_root / "teacher"))
    py_torch = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    yane_eval = Path(os.environ.get("YANE_EVAL", data_root / "engines/yaneuraou/eval"))
    work = Path("/tmp/binbo-task0920")
    cache = work / "cache_2m"
    rescore = work / "rescore_wst"
    results_path = logs / "task0920_results.json"
    logs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})
    os.environ.update(env)

    if not (cache / "meta.json").exists():
        run_logged([
            py_torch, "-m", "trainer.prepare_data", teacher / "run2m_c", cache,
            "--canonical-shared", "--val-shards", "4",
        ], logs / "task0920_prepare.log", cwd=repo, env=env)
    required_rescore = (
        rescore / "rescore_metrics.json",
        rescore / "train_policy_random_actions.npy",
        rescore / "val_policy_random_actions.npy",
    )
    if not all(path.exists() for path in required_rescore):
        run_logged([
            sys.executable, "t091_rescore.py", "--cache-dir", cache,
            "--ckpt", clone / "t087_wst/ws2.ckpt", "--out-dir", rescore,
            "--random-policy-seed", str(TASK_SEED),
        ], logs / "task0920_rescore_wst.log", cwd=clone, env=env)

    results = {
        "task": "092-0",
        "seed": TASK_SEED,
        "cache": str(cache),
        "rescore": str(rescore),
        "step1": {},
        "step2": None,
        "verdict": "running",
    }
    model_dirs = {
        head: train_model(py_torch, repo, cache, rescore, models, logs, head)
        for head in HEADS
    }
    metrics = {head: best_metrics(path) for head, path in model_dirs.items()}
    mlp_gate = (
        metrics["mlp"]["policy_top1"] >= 0.04
        and metrics["mlp"]["policy_ce"] < metrics["linear"]["policy_ce"]
    )
    if not mlp_gate:
        model_dirs["ft-mlp"] = train_model(
            py_torch, repo, cache, rescore, models, logs, "ft-mlp"
        )
        metrics["ft-mlp"] = best_metrics(model_dirs["ft-mlp"])
    pairwise = {}
    for left, right in itertools.combinations(metrics, 2):
        pairwise[f"{left}_minus_{right}"] = {
            key: metrics[left][key] - metrics[right][key]
            for key in ("value_mae", "value_corr", "value_std_ratio")
        }
    regularization_checks = {
        "value_mse_improved": (
            metrics["linear"]["value_mse"] < metrics["none"]["value_mse"]
        ),
        "value_mae_improved": (
            metrics["linear"]["value_mae"] < metrics["none"]["value_mae"]
        ),
        "value_corr_improved": (
            metrics["linear"]["value_corr"] > metrics["none"]["value_corr"]
        ),
        "value_std_ratio_improved": (
            abs(metrics["linear"]["value_std_ratio"] - 1.0)
            <= abs(metrics["none"]["value_std_ratio"] - 1.0)
        ),
    }
    regularization_works = all(regularization_checks.values())
    regularization_outcome = (
        "pass"
        if regularization_works
        else "mixed"
        if any(regularization_checks.values())
        else "fail"
    )
    policy_heads = [head for head in metrics if head != "none"]
    best_head = min(
        policy_heads,
        key=lambda head: (-metrics[head]["policy_top1"], metrics[head]["policy_ce"]),
    )
    results["step1"] = {
        "models": {head: str(path) for head, path in model_dirs.items()},
        "metrics": metrics,
        "pairwise": pairwise,
        "decision_a_regularization_works": regularization_works,
        "decision_a_outcome": regularization_outcome,
        "decision_a_checks": regularization_checks,
        "decision_b_mlp_large_improvement": mlp_gate,
        "best_head": best_head,
        "best_head_rule": "max val top1; lower CE tie-break",
    }
    write_results(results_path, results)
    print(f"T092_STEP1 {json.dumps(results['step1'], ensure_ascii=False)}", flush=True)

    random_model = train_model(
        py_torch, repo, cache, rescore, models, logs, best_head, label="random"
    )
    random_eval = data_root / "eval_dirs" / random_model.name
    export_eval(py_torch, repo, random_model, random_eval, yane_eval, logs)
    wst_model = model_dirs[best_head]
    run_logged([
        repo / "scripts/eval_vs_eval.sh", wst_model, random_eval,
        "nnue_best.pt", "100", "400", str(TASK_SEED), "6",
    ], logs / "task0920_positive_control.console.log", cwd=repo, env=env)
    tsv = logs / f"task061_{wst_model.name}_vs_{random_eval.name}_mt100.tsv"
    match = parse_teacher_match(tsv)
    passed = match["verdict"] == "accept_h1" or match["ci_lo"] > 0.0
    results["step2"] = {
        "head": best_head,
        "wst_model": str(wst_model),
        "random_model": str(random_model),
        "random_metrics": best_metrics(random_model),
        "match": match,
        "pass_rule": "accept_h1 or 95% CI lower bound > 0 Elo",
        "passed": passed,
    }
    results["verdict"] = "STEP2_PASS" if passed else "STEP2_FAIL_BRIDGE_NULL"
    write_results(results_path, results)
    print(f"T092_STEP2 {json.dumps(results['step2'], ensure_ascii=False)}", flush=True)
    print(f"T092_BRIDGE_CALIBRATION_DONE verdict={results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
