#!/usr/bin/env python3
"""Task 102b: is the search-label gain just the label SCALE?

Tasks 100/100b found the search-backed label (WS-T + Gumbel MCTS S=32 root
value) teaches a HalfKP NNUE +197 Elo (1M) / +155 Elo (2M) over the raw WS-T
value label. The search label also has a 3.8% smaller spread (cp std 1444 vs
1500 on the 1.96M train rows) and the search student searched shallower at equal
nodes (depth 14.46 vs 14.97), so the cheapest rival explanation is "a smaller
eval scale changes YaneuraOu's pruning / tree shape and that alone is worth Elo".

CONTROL (one variable = a linear rescale of the raw labels, made SCALE-EQUIVARIANT):
    scaled : round(cp_raw * 1443.790981472956 / 1500.437794209261)  (Task 100b's
             measured train std ratio, applied to train AND val), trained with
             --clamp-cp 2000*SCALE and --huber-delta 600*SCALE so that the target
             clamp and the loss are rescaled too. Without that (pre-launch review
             finding) the +/-2000 clamp would leave 14.5% of the rows with an
             unchanged target and shrink the effective target by only ~1.5%;
             with it, SCALE x (the raw student's function) is exactly the
             optimum of the control's objective, so the two students differ by
             the eval scale the engine sees and nothing else.
    raw    : Task 098's student `task098_student_wst` (same cache, same raw
             labels unscaled, same NNUE_ARGS, same seed)
Both on the full 1.96M-row cache_2m; the scaled student is trained here with
the 098 recipe, exported through the same 0cp gate, and plays the raw student
800 games (4 workers x 200, the Task 100 protocol and match seeds).

PRE-REGISTERED (committed before the run):
  * primary = paired bootstrap of scaled-vs-raw over 400 opening pairs.
  * decision table: 0 gates failed -> INVALID | 1 CI low > 77.5 (= half of
    100b's +155.0) -> SCALE_EXPLAINS_MOST | 2 CI low > 0 and <= 77.5 ->
    SCALE_PARTIAL | 3 CI straddles 0 -> SCALE_NOT_THE_MECHANISM | 4 CI high < 0
    -> SCALE_HURTS.
  * hard gates: the scaled npy reproduces the intended std ratio within 0.5%,
    0cp gate GREEN, zero illegal / engine failures.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t094_pipeline import run_logged  # noqa: E402
from t098_pipeline import NNUE_ARGS, paired_h2h_delta, yane_selftest  # noqa: E402
from t100_search_student import (  # noqa: E402
    H2H_GAMES_PER_WORKER, H2H_HALF_WIDTH_800, H2H_SEEDS, H2H_WORKERS, combine_h2h, now_jst,
    run_parallel_matches, search_conditions,
)

SCALE = 1443.790981472956 / 1500.437794209261      # Task 100b train: search std / raw std
CLAMP_CP_RAW, HUBER_DELTA_RAW = 2000.0, 600.0       # train_nnue defaults used by 098's raw student
CONTROL_ARGS = ["--clamp-cp", f"{CLAMP_CP_RAW * SCALE:.3f}", "--huber-delta", f"{HUBER_DELTA_RAW * SCALE:.3f}"]
REFERENCE_ELO_100B = 155.02
HALF_REFERENCE = REFERENCE_ELO_100B / 2.0
DECISION_TABLE = (
    "0 gates failed -> INVALID (non-terminal) | "
    f"1 CI low > {HALF_REFERENCE:.1f} -> SCALE_EXPLAINS_MOST | "
    f"2 0 < CI low <= {HALF_REFERENCE:.1f} -> SCALE_PARTIAL | "
    "3 CI straddles 0 -> SCALE_NOT_THE_MECHANISM | "
    "4 CI high < 0 -> SCALE_HURTS"
)


def make_scaled_labels(raw_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        raw = np.load(raw_dir / f"{split}_dlshogi_evals.npy")
        scaled = np.rint(raw.astype(np.float64) * SCALE).astype(np.int16)
        np.save(out_dir / f"{split}_dlshogi_evals.npy", scaled)
        ratio = float(scaled.astype(np.float64).std() / raw.astype(np.float64).std())
        clamped_ratio = float(np.clip(scaled, -CLAMP_CP_RAW * SCALE, CLAMP_CP_RAW * SCALE).astype(np.float64).std()
                              / np.clip(raw, -CLAMP_CP_RAW, CLAMP_CP_RAW).astype(np.float64).std())
        stats[split] = {"rows": int(raw.shape[0]), "raw_std": float(raw.std()),
                        "scaled_std": float(scaled.std()), "std_ratio": ratio,
                        "clamped_target_std_ratio": clamped_ratio,
                        "frac_abs_over_2000": float(np.mean(np.abs(raw.astype(np.int32)) > CLAMP_CP_RAW)),
                        "max_abs_change_cp": int(np.max(np.abs(scaled.astype(np.int32) - raw.astype(np.int32))))}
        if abs(ratio - SCALE) > 0.005:
            raise AssertionError(f"{split}: std ratio {ratio:.4f} != {SCALE:.4f}")
        print(f"T102B_SCALED {split} {json.dumps(stats[split])}", flush=True)
    (out_dir / "meta.json").write_text(json.dumps({"scale": SCALE, "source": str(raw_dir),
                                                   "stats": stats}, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task102b")
    parser.add_argument("--stage", choices=("all", "train", "match"), default="all",
                        help="train = labels+student+export+gate; match = games+verdict")
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    eval_dirs = data_root / "eval_dirs"
    torch_py = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data_root / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    yane_eval = data_root / "engines/yaneuraou/eval"
    cache = data_root / "work/task097/cache_2m"
    raw_dir = data_root / "work/task098/rescore_wst"
    raw_model = models / "task098_student_wst"
    raw_eval = eval_dirs / "task098_student_wst"
    work = data_root / "work" / args.run_tag
    scaled_dir = work / "rescore_scaled"
    model_dir = models / f"{args.run_tag}_student_scaled"
    eval_dir = eval_dirs / f"{args.run_tag}_student_scaled"
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})

    for path in (torch_py, binbo, yane, yane_eval / "nn.bin", cache / "meta.json",
                 raw_dir / "train_dlshogi_evals.npy", raw_model / "history.json",
                 raw_eval / "nn.bin"):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    history = json.loads((raw_model / "history.json").read_text(encoding="utf-8"))
    if history["n_train"] != 1_960_000:
        raise RuntimeError("task098_student_wst is not the 1.96M student")
    adapter = yane_selftest(logs)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    work.mkdir(parents=True, exist_ok=True)
    print(f"T102B_PREFLIGHT_OK head={head} stage={args.stage} start={now_jst():%H:%M} JST", flush=True)

    if args.stage in ("all", "train"):
        if not (scaled_dir / "meta.json").exists():
            make_scaled_labels(raw_dir, scaled_dir)
        if not (model_dir / "nnue_best.pt").exists():
            run_logged([str(torch_py), "-m", "trainer.train_nnue", str(cache), "--out",
                        str(model_dir), *NNUE_ARGS, *CONTROL_ARGS, "--eval-npy", str(scaled_dir)],
                       logs / f"{args.run_tag}_student_scaled.log", cwd=repo, env=env)
        print("T102B_STUDENT_DONE", flush=True)
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            run_logged([str(torch_py), "-m", "trainer.export_nnue", str(model_dir),
                        "--weights", "nnue_best.pt", "--out", str(model_dir / "nnue_best.bnnq")],
                       logs / f"{args.run_tag}_export_scaled.log", cwd=repo, env=env)
            run_logged([str(torch_py), "-m", "trainer.binbo_to_yane",
                        "--from-bnnq", str(model_dir / "nnue_best.bnnq"),
                        "--template", str(yane_eval / "nn.bin"), "--out", str(eval_dir / "nn.bin")],
                       logs / f"{args.run_tag}_yane_scaled.log", cwd=repo, env=env)
        gate_log = logs / f"{args.run_tag}_0cp_scaled.log"
        if not gate_log.exists():
            run_logged(["bash", "scripts/binbo_to_yane.sh", str(model_dir / "nnue_best.bnnq")],
                       gate_log, cwd=repo, env=env)
        gate_text = gate_log.read_text(encoding="utf-8")
        if "RED" in gate_text or gate_text.count("GREEN") < 2:
            raise RuntimeError(f"0cp gate failed: see {gate_log}")
        print("T102B_TRAIN_STAGE_DONE", flush=True)
        if args.stage == "train":
            return 0

    # ---- match + verdict ----------------------------------------------------
    for path in (eval_dir / "nn.bin", scaled_dir / "meta.json", model_dir / "history.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    scaled_stats = json.loads((scaled_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    start = now_jst()
    print(f"T102B_H2H_START {start:%Y-%m-%d %H:%M}", flush=True)
    tsvs = run_parallel_matches(binbo, yane, eval_dir, raw_eval, logs, args.run_tag,
                                H2H_GAMES_PER_WORKER, H2H_SEEDS[:H2H_WORKERS], env, repo)
    rows, shape = combine_h2h(tsvs, H2H_GAMES_PER_WORKER)
    h2h = paired_h2h_delta(rows, label="student_scaled_minus_raw")
    per_worker = [search_conditions(tsv) for tsv in tsvs]
    conditions = {side: {key: float(np.mean([c[side][key] for c in per_worker]))
                         for key in ("avg_depth", "avg_nodes")} for side in ("cand", "base")}
    print("T102B_H2H_DONE", flush=True)

    failures = []
    if shape["bad_reasons"]:
        failures.append(f"illegal_or_engine_fail_{len(shape['bad_reasons'])}")
    low, high = h2h["ci95"]
    if failures:
        verdict, terminal = "INVALID", False
    elif low > HALF_REFERENCE:
        verdict, terminal = "SCALE_EXPLAINS_MOST", True
    elif low > 0.0:
        verdict, terminal = "SCALE_PARTIAL", True
    elif high < 0.0:
        verdict, terminal = "SCALE_HURTS", True
    else:
        verdict, terminal = "SCALE_NOT_THE_MECHANISM", True
    flags = []
    if h2h["half_width"] > 1.5 * H2H_HALF_WIDTH_800:
        flags.append("UNDERPOWERED_VS_PLAN")
    student_history = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    results = {
        "task": "102b",
        "preregistration": {
            "one_variable": (f"scale-equivariant rescale of the raw WS-T labels by {SCALE:.5f} "
                             f"(100b std ratio): targets, --clamp-cp and --huber-delta all x SCALE"),
            "control_args": CONTROL_ARGS,
            "baseline": "task098_student_wst (same cache, raw labels, NNUE_ARGS, seed)",
            "primary": "yane-vs-yane 100ms/Threads1, 4 workers x 200 games, paired bootstrap",
            "decision_table": DECISION_TABLE, "reference_elo_100b": REFERENCE_ELO_100B,
            "hard_gates": ["std_ratio_within_0.5pct", "0cp_gate", "no_illegal"],
        },
        "scaled_label_stats": scaled_stats,
        "student": {"model_dir": str(model_dir), "eval_dir": str(eval_dir),
                    "best_epoch": student_history.get("best_epoch"),
                    "last_epoch": student_history.get("history", [None])[-1]},
        "primary_h2h": h2h, "search_conditions": conditions,
        "prerequisites": {"failures": failures, "adapter_selftest": adapter, "h2h_shape": shape,
                          "h2h_tsvs": [str(t) for t in tsvs]},
        "clock": {"match_start_jst": start.strftime("%Y-%m-%d %H:%M"),
                  "match_end_jst": now_jst().strftime("%Y-%m-%d %H:%M")},
        "flags": flags, "terminal": terminal, "verdict": verdict,
        "git": {"head": head, "exp_dirty": 0, "pushed": False},
    }
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"GATE primary h2h(scaled-student vs raw-student) n_pairs={h2h['n_pairs']} "
          f"W-D-L={h2h['wins']}-{h2h['draws']}-{h2h['losses']} score={h2h['score']:.4f} "
          f"elo={h2h['elo']:+.2f} ci={[round(v, 2) for v in h2h['ci95']]} hw={h2h['half_width']:.1f}",
          flush=True)
    print(f"SEARCH cand depth={conditions['cand']['avg_depth']:.2f} nodes={conditions['cand']['avg_nodes']:.0f} | "
          f"base depth={conditions['base']['avg_depth']:.2f} nodes={conditions['base']['avg_nodes']:.0f}", flush=True)
    print(f"FLAGS {flags}", flush=True)
    print(f"T102B_PIPELINE_DONE verdict={verdict} terminal={terminal} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
