#!/usr/bin/env python3
"""Sequential Task 091 runner: loop chunks, monitors, mini teacher gates, final SPRT."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run_logged(command, log_path, *, cwd, env):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {' '.join(map(str, command))} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command], cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed rc={return_code}: {command}")


def parse_teacher_match(path):
    text = Path(path).read_text(encoding="utf-8")
    elo_match = re.search(r"candidate_elo=([-+0-9.]+) elo_ci95=\[([-+0-9.]+),([-+0-9.]+)\]", text)
    verdict_match = re.search(r"sprt_llr=([-+0-9.]+).*verdict=([a-z0-9_]+)", text)
    record_match = re.search(r"candidate_record\(W-L-D\)=([0-9]+)-([0-9]+)-([0-9]+)", text)
    if not (elo_match and verdict_match and record_match):
        raise ValueError(f"cannot parse teacher match: {path}")
    return {
        "elo": float(elo_match.group(1)),
        "ci_lo": float(elo_match.group(2)),
        "ci_hi": float(elo_match.group(3)),
        "llr": float(verdict_match.group(1)),
        "verdict": verdict_match.group(2),
        "wins": int(record_match.group(1)),
        "losses": int(record_match.group(2)),
        "draws": int(record_match.group(3)),
    }


def parse_internal_match(path):
    text = Path(path).read_text(encoding="utf-8")
    matches = re.findall(
        r"FINAL az W-D-L ([0-9]+)-([0-9]+)-([0-9]+)/([0-9]+).*?elo ([+-][0-9.]+) "
        r"\(CI ([+-][0-9.]+)\.\.([+-][0-9.]+)\)", text,
    )
    if not matches:
        raise ValueError(f"cannot parse internal match: {path}")
    wins, draws, losses, games, elo, lo, hi = matches[-1]
    return {
        "wins": int(wins), "draws": int(draws), "losses": int(losses),
        "games": int(games), "elo": float(elo), "ci_lo": float(lo), "ci_hi": float(hi),
    }


def update_streak(value, best, streak):
    if value >= best + 10.0:
        return value, 0, True
    return best, streak + 1, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, choices=[32, 48], required=True)
    parser.add_argument("--selfplay-batch", type=int, choices=[64, 128, 256], required=True)
    parser.add_argument("--base-iters", type=int, choices=[50, 75, 100], default=75)
    parser.add_argument("--hard-cap", type=int, default=150)
    parser.add_argument("--states-per-iter", type=int, default=65_536)
    parser.add_argument("--training-batch", type=int, choices=[128, 256], default=128)
    parser.add_argument("--iter-seconds-estimate", type=float, required=True)
    parser.add_argument("--budget-hours", type=float, default=24.0)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    teacher = Path(os.environ.get("TEACHER_DIR", data_root / "teacher"))
    py_torch = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    yane_eval = Path(os.environ.get("YANE_EVAL", data_root / "engines/yaneuraou/eval"))
    run_tag = args.run_tag or f"t091_s{args.sims}_b{args.selfplay_batch}"
    work_root = Path("/tmp/binbo-task091")
    work = work_root / run_tag
    cache = work_root / "cache_2m"
    baseline_rescore = work_root / "rescore_wst"
    baseline_model = models / "task091_teacher_wst"
    baseline_eval = data_root / "eval_dirs/task091_teacher_wst"
    ckpt_dir = clone / f"{run_tag}_ckpt"
    metrics_path = logs / f"{run_tag}_gates.json"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = dict(os.environ)
    env.update({
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
    })

    if not (cache / "meta.json").exists():
        run_logged([
            py_torch, "-m", "trainer.prepare_data", teacher / "run2m_c", cache,
            "--canonical-shared", "--val-shards", "4",
        ], logs / f"{run_tag}_prepare.log", cwd=repo, env=env)

    wst = clone / "t087_wst/ws2.ckpt"
    if not (baseline_rescore / "rescore_metrics.json").exists():
        run_logged([
            sys.executable, "t091_rescore.py", "--cache-dir", cache,
            "--ckpt", wst, "--out-dir", baseline_rescore,
        ], logs / f"{run_tag}_rescore_wst.log", cwd=clone, env=env)
    if not (baseline_model / "nnue_best.pt").exists():
        run_logged([
            py_torch, "-m", "trainer.train_nnue_policy_aux", cache,
            "--value-dir", baseline_rescore, "--policy-dir", baseline_rescore,
            "--out", baseline_model,
        ], logs / f"{run_tag}_train_wst.log", cwd=repo, env=env)
    baseline_eval.mkdir(parents=True, exist_ok=True)
    baseline_bnnq = baseline_model / "nnue_best.bnnq"
    if not (baseline_eval / "nn.bin").exists():
        run_logged([
            py_torch, "-m", "trainer.export_nnue", baseline_model,
            "--weights", "nnue_best.pt", "--out", baseline_bnnq.name,
        ], logs / f"{run_tag}_export_wst.log", cwd=repo, env=env)
        run_logged([
            py_torch, "-m", "trainer.binbo_to_yane", "--from-bnnq", baseline_bnnq,
            "--template", yane_eval / "nn.bin", "--out", baseline_eval / "nn.bin",
        ], logs / f"{run_tag}_write_wst.log", cwd=repo, env=env)
        run_logged([
            repo / "scripts/binbo_to_yane.sh", baseline_bnnq,
        ], logs / f"{run_tag}_gate_wst.log", cwd=repo, env=env)

    gate_data = {
        "run_tag": run_tag,
        "config": vars(args),
        "baseline": {"checkpoint": str(wst), "model": str(baseline_model)},
        "gates": [],
    }
    teacher_best = internal_best = 0.0
    teacher_streak = internal_streak = 0
    target = 25
    stop_reason = None
    best_gate = None

    while target <= args.hard_cap:
        run_logged([
            sys.executable, "t091_loop_frozen.py",
            f"max_num_iters={target}", f"num_simulations={args.sims}",
            f"selfplay_batch_size={args.selfplay_batch}",
            f"states_per_iter={args.states_per_iter}",
            f"training_batch_size={args.training_batch}",
            "learning_rate=0.0001", f"init_ckpt={wst}", f"ckpt_dir={ckpt_dir}",
            "resume=auto", "ckpt_interval=5", "monitor_interval=5",
            f"expected_iter_s={args.iter_seconds_estimate}", "teacher_val_dir=m4_data",
        ], logs / f"{run_tag}_loop_to_{target:03d}.log", cwd=clone, env=env)
        checkpoint = ckpt_dir / f"{target:06d}.ckpt"
        rescore = work / f"rescore_iter{target:03d}"
        model_dir = models / f"task091_{run_tag}_teacher_iter{target:03d}"
        if not (rescore / "rescore_metrics.json").exists():
            run_logged([
                sys.executable, "t091_rescore.py", "--cache-dir", cache,
                "--ckpt", checkpoint, "--out-dir", rescore,
                "--reference-value-dir", baseline_rescore,
                "--reference-policy-dir", baseline_rescore,
            ], logs / f"{run_tag}_rescore_{target:03d}.log", cwd=clone, env=env)
        if not (model_dir / "nnue_best.pt").exists():
            run_logged([
                py_torch, "-m", "trainer.train_nnue_policy_aux", cache,
                "--value-dir", baseline_rescore, "--policy-dir", rescore,
                "--out", model_dir,
            ], logs / f"{run_tag}_train_{target:03d}.log", cwd=repo, env=env)

        internal_log = logs / f"{run_tag}_internal_{target:03d}.log"
        run_logged([
            sys.executable, "m2_match.py", "--az-ckpt", checkpoint, "--az-sims", "128",
            "--opponent", "az2", "--opp-ckpt", wst, "--opp-sims", "128",
            "--games", "40", "--byoyomi", "100", "--seed", "20260714",
            "--tsv", logs / f"{run_tag}_internal_{target:03d}.tsv",
        ], internal_log, cwd=clone, env=env)
        internal = parse_internal_match(internal_log)

        run_logged([
            repo / "scripts/eval_vs_eval.sh", model_dir, baseline_eval,
            "nnue_best.pt", "100", "80", "20260714", "6",
        ], logs / f"{run_tag}_teacher_{target:03d}.console.log", cwd=repo, env=env)
        teacher_tsv = logs / (
            f"task061_{model_dir.name}_vs_{baseline_eval.name}_mt100.tsv"
        )
        teacher_match = parse_teacher_match(teacher_tsv)
        rescore_metrics = json.loads((rescore / "rescore_metrics.json").read_text())
        teacher_best, teacher_streak, teacher_new = update_streak(
            teacher_match["elo"], teacher_best, teacher_streak
        )
        internal_best, internal_streak, internal_new = update_streak(
            internal["elo"], internal_best, internal_streak
        )
        gate = {
            "iteration": target,
            "checkpoint": str(checkpoint),
            "model": str(model_dir),
            "internal": internal,
            "teacher": teacher_match,
            "rescore": rescore_metrics["metrics"],
            "teacher_new_high": teacher_new,
            "internal_new_high": internal_new,
            "teacher_no_improve_streak": teacher_streak,
            "internal_no_improve_streak": internal_streak,
            "elapsed_hours": (time.time() - started) / 3600.0,
        }
        gate_data["gates"].append(gate)
        metrics_path.write_text(json.dumps(gate_data, indent=2), encoding="utf-8")
        print(f"T091_GATE {json.dumps(gate)}", flush=True)
        if best_gate is None or teacher_match["elo"] > best_gate["teacher"]["elo"]:
            best_gate = gate

        if target >= 75 and teacher_streak >= 3 and internal_streak >= 3:
            stop_reason = "teacher_and_internal_no_new_high_for_3_gates"
            break
        if target < args.base_iters:
            target += 25
            continue
        if target >= args.hard_cap:
            stop_reason = "hard_cap"
            break
        projected = time.time() - started + 25 * args.iter_seconds_estimate
        if not teacher_new:
            stop_reason = "extension_requires_teacher_new_high"
            break
        if projected > args.budget_hours * 3600:
            stop_reason = "wallclock_budget"
            break
        target += 25

    if best_gate is None:
        raise AssertionError("no teacher gate completed")
    final_model = Path(best_gate["model"])
    run_logged([
        repo / "scripts/eval_vs_eval.sh", final_model, baseline_eval,
        "nnue_best.pt", "100", "400", "20260715", "6",
    ], logs / f"{run_tag}_teacher_final.console.log", cwd=repo, env=env)
    final_tsv = logs / f"task061_{final_model.name}_vs_{baseline_eval.name}_mt100.tsv"
    final = parse_teacher_match(final_tsv)
    passed = final["verdict"] == "accept_h1" or final["ci_lo"] > 0.0
    gate_data.update({
        "stop_reason": stop_reason,
        "best_iteration": best_gate["iteration"],
        "final_teacher": final,
        "verdict": "PASS" if passed else "FAIL",
        "elapsed_hours": (time.time() - started) / 3600.0,
    })
    metrics_path.write_text(json.dumps(gate_data, indent=2), encoding="utf-8")
    print(f"T091_FINAL {json.dumps(gate_data, ensure_ascii=False)}", flush=True)
    print("T091_PIPELINE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
