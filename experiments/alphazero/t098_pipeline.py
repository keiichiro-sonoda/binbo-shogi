#!/usr/bin/env python3
"""Task 098: does the shared-trunk loop's value carry the gain into the NNUE?

Task 097 replicated the r=0.50 loop's external positive transfer (+26 Elo), but
the winning net's value tower is byte-identical to WS-T, so as a VALUE teacher it
is still WS-T. `t098_loop_shared.py` shares the trunk so the policy learning moves
the value, with `lambda * MSE(v, v_WS-T)` constraining the drift. This pipeline
selects lambda, runs the loop, and asks the Task 084 question: is the resulting
value a better NNUE teacher than WS-T's?

PRE-REGISTERED (committed locally before the lambda preflight; see the plan):
  * one variable = the value tower's treatment (two-tower frozen -> shared trunk
    + lambda regulariser). Everything else matches the Task 096 r=0.50 arm.
  * lambda from a 3-iteration log ladder {3, 30, 300}, chosen on VALUE HEALTH
    ONLY -- no external Elo is consulted, so the choice cannot contaminate the
    primary. Rule: the LARGEST lambda with value_mae_wst(iter3) >= 30 cp AND
    value_std_ratio(iter3) >= 0.95 AND value_corr_teacher(iter3) >= 0.92. If
    none qualifies, take the smallest and report "ladder too strong".

    Both the ladder's range and the direction of the rule were set from a
    1-iteration wiring smoke (disclosed here because it preceded this
    pre-registration): at lambda=3 the value moved 35.3 cp in a single iteration
    with std_ratio 1.0007 and corr_teacher 0.9363, while the penalty term was
    3 * 3.86e-4 = 0.0012 against a policy CE of ~1.4 -- i.e. 0.08% of the loss,
    constraining essentially nothing. So movement is easy, the binding risk is
    COLLAPSE, and the ladder has to reach strengths that actually bind. Taking
    the LARGEST qualifying lambda picks the strongest constraint that still lets
    the value move. The smoke observed mechanism only, never an outcome, so it
    cannot bias the primary.
  * stop rule: value_std_ratio < 0.80 mid-run halts everything. Task 082's
    gen100 collapse hit 0.475 and Task 084 measured that teacher at -145 Elo.
  * PRIMARY = Task 084's value-relative judging: rescore 2M with the candidate's
    own value -> train an NNUE student on those cp labels -> play it against a
    student trained identically on WS-T value labels, 400 pairs / 800 games,
    paired CI. Lower bound > 0 -> CARRIAGE_ESTABLISHED (which earns only the
    right to a 32M confirmation: Task 085 saw +237 at 2M reverse to -36 at 32M).
  * NULL branch splits three ways on the value metrics -- see DECISION_TABLE.
  * detection floor, disclosed: Task 092-0 measured a +-34 Elo half-width at 400
    games / 100 ms, where the arms differed by 0.26 cp of value MAE. 800 games
    should give ~+-24. A value drift below that floor cannot be resolved.
  * the gikou@6 leg is a TRIPWIRE, not a confirmation: 512 games gives ~+-66, far
    too wide to confirm the +26. It only catches a gross collapse.

Two implementation notes that differ from the plan, both cost-driven and
scientifically inert:
  * the WS-T reference value is computed online in the loop for both update
    kinds. Precomputing the 2M anchor pool offline would cost MORE (2M forwards
    once vs the 25*256*128 = 819k actually drawn).
  * `yane-vs-yane` is a DIRECT head-to-head, not two series against a common
    opponent, so the primary uses a one-sample paired bootstrap over its opening
    pairs rather than `t095_gate.paired_delta_general` (which computes a
    difference of two series). The gikou tripwire, which IS two-series, uses it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t094_pipeline import elo_from_score, run_logged  # noqa: E402
from t095_gate import BLOCK_SEEDS, paired_delta_general, self_test as gate_self_test  # noqa: E402
from t095_pipeline import complete_tsv  # noqa: E402


def parse_iter_metrics(log_path, iteration):
    """Pull the key=value fields off one `t098 iter=N ...` line.

    `t095_pipeline.parse_iter_metrics` hardcodes the `t094 iter=` prefix that
    Tasks 094-097 emit; this loop prints `t098 iter=`. t095_pipeline.py is the
    artifact of record for those tasks, so the prefix lives here instead.
    """
    import re

    text = Path(log_path).read_text(encoding="utf-8")
    match = re.search(rf"^t098 iter={iteration} (.*)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"no `t098 iter={iteration}` line in {log_path}")
    fields = {}
    for key, value in re.findall(r"([a-z_0-9]+)=([-+0-9.naeN]+)", match.group(1)):
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    return fields

# ---- pre-registered constants -------------------------------------------------
LAMBDA_LADDER = (3.0, 30.0, 300.0)
LAMBDA_PREFLIGHT_ITERS = 3
LAMBDA_STD_RATIO_MIN = 0.95
LAMBDA_CORR_MIN = 0.92
VALUE_STD_RATIO_STOP = 0.80
LOOP_ITERS = 25
LOOP_SEED = 20260714                      # = the Task 096 r=0.50 arm
TRIPWIRE_BLOCKS = (12, 13)                # WS-T already measured there in 097
TRIPWIRE_GAMES = 256
H2H_GAMES = 800                           # 400 opening pairs
BOOTSTRAP_SAMPLES = 200_000
H2H_BOOTSTRAP_SEED = 20269800
H2H_SEED = 20269801                       # match seed for yane-vs-yane
# Task 092-0: a ~10 cp value-MAE difference was not resolvable in Elo. Three
# times that is the pre-registered bar for "the value actually moved".
VALUE_MOVED_MAE_CP = 30.0
DETECTION_FLOOR_400 = 34.0                # measured half-width, 400 games @100ms
WST_VALUE_CORR_TEACHER = 0.9357           # the frozen null

DECISION_TABLE = (
    "0 prerequisites failed -> INSUFFICIENT (non-terminal) | "
    "1 h2h CI low > 0 -> CARRIAGE_ESTABLISHED (earns the 32M confirmation only) | "
    "2 h2h CI high < 0 -> CARRIAGE_NEGATIVE | "
    "3 CI straddles 0 and value MAE <= 30cp -> NULL_LAMBDA_TOO_LARGE | "
    "4 CI straddles 0 and value MAE > 30cp -> NULL_CARRIAGE_WALL"
)

NNUE_ARGS = ["--qat", "--qat-tie", "--feature-dim", "256", "--hidden-dim", "32",
             "--lr", "1e-5", "--epochs", "12", "--batch", "1024", "--scale-cp", "600"]


# ---- yane-vs-yane TSV -------------------------------------------------------
def load_yane_tsv(path):
    """Parse the 21-column yane-vs-yane TSV into scored, 0-indexed rows.

    The file carries three preamble lines, a header row, the games, then a
    summary block. `game` is 1-based in the file and `cand_color` is
    black/white; both are normalised here so the rows look like the
    `m2_match.py` format the rest of the arc uses.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 21 or not fields[0].isdigit():
            continue
        score = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(fields[3])
        if score is None:
            raise ValueError(f"unexpected result {fields[3]!r} in {path}")
        colour = {"black": "b", "white": "w"}.get(fields[1])
        if colour is None:
            raise ValueError(f"unexpected cand_color {fields[1]!r} in {path}")
        rows.append({
            "game": int(fields[0]) - 1, "color": colour, "score": score,
            "reason": fields[4], "plies": int(fields[5]),
            "opening_seed": fields[2], "opening": fields[20],
        })
    return rows


def validate_h2h(rows, path, expected_games=None):
    n = len(rows)
    if expected_games is not None and n != expected_games:
        raise RuntimeError(f"{path}: {n} games, expected {expected_games}")
    if n < 2 or n % 2:
        raise RuntimeError(f"{path}: need an even game count >= 2, got {n}")
    if [row["game"] for row in rows] != list(range(n)):
        raise RuntimeError(f"{path}: game column is not 0..{n-1} after normalisation")
    for index in range(0, n, 2):
        first, second = rows[index], rows[index + 1]
        if first["opening_seed"] != second["opening_seed"]:
            raise RuntimeError(f"{path}: rows {index}/{index+1} are not an opening pair")
        if (first["color"], second["color"]) != ("b", "w"):
            raise RuntimeError(f"{path}: rows {index}/{index+1} colours are not (b,w)")
    bad = [row["reason"] for row in rows
           if row["reason"].startswith(("illegal", "engine_fail", "error"))]
    return {"n_games": n, "n_pairs": n // 2, "bad_reasons": bad}


def paired_h2h_delta(rows, *, label="", samples=BOOTSTRAP_SAMPLES):
    """One-sample paired bootstrap of a direct head-to-head.

    Games 2k / 2k+1 share an opening with the colours swapped, so the opening
    pair is the resampling unit -- the same idiom as the rest of the arc, but for
    a head-to-head there is no second series to difference against.
    """
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    pairs = (scores[0::2] + scores[1::2]) / 2.0
    n = len(pairs)
    point = elo_from_score(float(pairs.mean()))
    rng = np.random.default_rng(H2H_BOOTSTRAP_SEED)
    deltas = np.empty(samples)
    chunk = 10_000
    for start in range(0, samples, chunk):
        end = min(start + chunk, samples)
        idx = rng.integers(0, n, size=(end - start, n))
        drawn = np.clip(pairs[idx].mean(axis=1), 1e-9, 1 - 1e-9)
        deltas[start:end] = 400.0 * np.log10(drawn / (1 - drawn))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "label": label, "elo": point, "ci95": [float(low), float(high)],
        "half_width": float(high - low) / 2.0,
        "score": float(pairs.mean()), "n_pairs": int(n), "n_games": int(len(scores)),
        "wins": int((scores == 1.0).sum()), "draws": int((scores == 0.5).sum()),
        "losses": int((scores == 0.0).sum()),
        "bootstrap_seed": H2H_BOOTSTRAP_SEED, "bootstrap_samples": samples,
    }


def yane_selftest(logs):
    """Round-trip the adapter against a stored yane-vs-yane run.

    The file's own summary block records candidate_score_rate and candidate_elo,
    so re-deriving them from the rows proves the parser, the colour/result
    mapping and the Elo transform all agree with the Rust side.
    """
    path = logs / "task061_task0920_ft_mlp_wst_vs_task0920_ft_mlp_random_mt100.tsv"
    if not path.exists():
        raise FileNotFoundError(f"self-test needs the Task 092-0 TSV: {path}")
    text = path.read_text(encoding="utf-8")
    want_score = float(
        [ln for ln in text.splitlines() if ln.startswith("candidate_score_rate=")][0]
        .split("=")[1].split("%")[0]) / 100.0
    want_elo = float(
        [ln for ln in text.splitlines() if ln.startswith("candidate_elo=")][0]
        .split("=")[1].split()[0])
    rows = load_yane_tsv(path)
    shape = validate_h2h(rows, path)
    scores = np.array([row["score"] for row in rows])
    got_score = float(scores.mean())
    got_elo = elo_from_score(got_score)
    if abs(got_score - want_score) > 5e-4:
        raise AssertionError(f"score round-trip: {got_score:.5f} vs {want_score:.5f}")
    if abs(got_elo - want_elo) > 0.1:
        raise AssertionError(f"elo round-trip: {got_elo:.2f} vs {want_elo:.2f}")
    print(f"T098_YANE_ADAPTER_OK games={shape['n_games']} pairs={shape['n_pairs']} "
          f"score={got_score:.4f}(file {want_score:.4f}) "
          f"elo={got_elo:.2f}(file {want_elo:.2f})", flush=True)
    return {"n_games": shape["n_games"], "score": got_score, "elo": got_elo}


# ---- lambda selection --------------------------------------------------------
def select_lambda(trials):
    """Strongest lambda that still lets the value move, and keeps it healthy.

    A 1-iteration wiring smoke at lambda=3 moved the value 35.3 cp (past the
    30 cp bar) with std_ratio 1.0007 and corr_teacher 0.9363 -- i.e. movement is
    easy and the binding risk is COLLAPSE, not stasis. At that lambda the
    penalty was 3 * 3.86e-4 = 0.0012 against a policy CE of ~1.4, so it was
    contributing 0.08% of the loss and constraining essentially nothing; hence
    the ladder reaches 300. Picking the LARGEST qualifying lambda takes the
    strongest available constraint that still permits the value to move.
    """
    qualifying = [
        value for value in sorted(trials)
        if trials[value]["value_mae_wst"] >= VALUE_MOVED_MAE_CP
        and trials[value]["value_std_ratio"] >= LAMBDA_STD_RATIO_MIN
        and trials[value]["value_corr_teacher"] >= LAMBDA_CORR_MIN
    ]
    if qualifying:
        return qualifying[-1], "qualified"
    return min(trials), "ladder_too_strong"


def analyse(h2h, tripwire, loop_metrics, lambda_record, rescore, prerequisites, students):
    failures = prerequisites["failures"]
    low, high = h2h["ci95"]
    mae = loop_metrics["value_mae_wst"]
    if failures:
        verdict, terminal = "INSUFFICIENT", False
    elif low > 0.0:
        verdict, terminal = "CARRIAGE_ESTABLISHED", True
    elif high < 0.0:
        verdict, terminal = "CARRIAGE_NEGATIVE", True
    elif mae <= VALUE_MOVED_MAE_CP:
        verdict, terminal = "NULL_LAMBDA_TOO_LARGE", True
    else:
        verdict, terminal = "NULL_CARRIAGE_WALL", True

    flags = []
    if h2h["half_width"] > 1.5 * DETECTION_FLOOR_400:
        flags.append("UNDERPOWERED_VS_PLAN")
    if loop_metrics["value_std_ratio"] < VALUE_STD_RATIO_STOP:
        flags.append("VALUE_STD_COLLAPSED")
    if lambda_record["status"] == "ladder_too_strong":
        flags.append("LAMBDA_LADDER_TOO_STRONG")
    if mae <= VALUE_MOVED_MAE_CP:
        flags.append("VALUE_BARELY_MOVED")
    return {
        "task": 98,
        "preregistration": {
            "one_variable": "value tower treatment: two-tower frozen -> shared trunk + lambda",
            "lambda_ladder": list(LAMBDA_LADDER),
            "lambda_rule": (f"largest lambda with value_mae_wst>={VALUE_MOVED_MAE_CP}cp, "
                            f"std_ratio>={LAMBDA_STD_RATIO_MIN} and "
                            f"corr_teacher>={LAMBDA_CORR_MIN} at iter "
                            f"{LAMBDA_PREFLIGHT_ITERS}; value health only, no Elo"),
            "stop_rule": f"value_std_ratio < {VALUE_STD_RATIO_STOP}",
            "primary": "Task 084 value-relative judging, 400 pairs / 800 games, paired CI",
            "decision_table": DECISION_TABLE,
            "value_moved_bar_cp": VALUE_MOVED_MAE_CP,
            "detection_floor_400_games": DETECTION_FLOOR_400,
            "tripwire_caveat": ("512 games gives ~+-66 Elo; it cannot confirm the +26, "
                                "only catch a gross collapse"),
            "promotion_caveat": ("Task 085 saw 2M +237 reverse to 32M -36; a 2M win "
                                 "earns only the right to a 32M confirmation"),
        },
        "lambda": lambda_record,
        "primary_h2h": h2h,
        "gikou_tripwire": tripwire,
        "loop_final_metrics": loop_metrics,
        "value_label_stats": rescore,
        "students": students,
        "prerequisites": prerequisites,
        "flags": flags,
        "terminal": terminal,
        "verdict": verdict,
    }


def report(results):
    h2h = results["primary_h2h"]
    print(f"GATE primary h2h(cand vs wst-student) n_pairs={h2h['n_pairs']} "
          f"score={h2h['score']:.4f} elo={h2h['elo']:+.2f} "
          f"ci={[round(v, 2) for v in h2h['ci95']]} hw={h2h['half_width']:.1f}", flush=True)
    metrics = results["loop_final_metrics"]
    print(f"VALUE std_ratio={metrics['value_std_ratio']:.4f} "
          f"mae_wst={metrics['value_mae_wst']:.1f}cp "
          f"corr_teacher={metrics['value_corr_teacher']:.4f}(WS-T {WST_VALUE_CORR_TEACHER}) "
          f"cp_std={metrics['value_cp_std']:.1f}", flush=True)
    for split, stats in results["value_label_stats"].items():
        print(f"LABELS {split} {stats}", flush=True)
    trip = results["gikou_tripwire"]
    if trip:
        print(f"TRIPWIRE gikou@6 delta={trip['paired_delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in trip['paired_ci95']]} n_pairs={trip['n_pairs']}",
              flush=True)
    print(f"LAMBDA {results['lambda']}", flush=True)
    print(f"FLAGS {results['flags']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task098")
    parser.add_argument("--iters", type=int, default=LOOP_ITERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    eval_dirs = data_root / "eval_dirs"
    torch_py = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    pgx_py = Path.home() / "task074_frameworks/venv_pgx/bin/python"
    binbo = repo / "target/release/binbo-shogi"
    yane = data_root / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    yane_eval = data_root / "engines/yaneuraou/eval"
    anchor_data = clone / "m4_data"
    wst = clone / "t087_wst/ws2.ckpt"
    cache = data_root / "work/task097/cache_2m"      # same recipe, already built
    work = data_root / "work" / args.run_tag
    ckpt_dir = models / f"{args.run_tag}_ckpt"
    main_ckpt = ckpt_dir / f"{args.iters:06d}.ckpt"

    if args.dry_run:
        print(f"DRYRUN work={work} ckpt_dir={ckpt_dir}")
        print(f"DRYRUN lambda ladder {LAMBDA_LADDER} x {LAMBDA_PREFLIGHT_ITERS} iter, "
              f"rule: largest with mae>={VALUE_MOVED_MAE_CP}cp, "
              f"std_ratio>={LAMBDA_STD_RATIO_MIN}, corr>={LAMBDA_CORR_MIN}")
        print(f"DRYRUN loop {args.iters} iter, stop if std_ratio < {VALUE_STD_RATIO_STOP}")
        print(f"DRYRUN tripwire gikou@6 blocks {TRIPWIRE_BLOCKS} x {TRIPWIRE_GAMES} "
              f"(paired vs the 097 WS-T games)")
        print(f"DRYRUN students: az_rescore x2 -> train_nnue x2 -> export x2")
        print(f"DRYRUN primary: yane-vs-yane {H2H_GAMES} games (SPRT disabled via elo0=elo1)")
        print(f"DRYRUN decision table: {DECISION_TABLE}")
        loop_h = (LAMBDA_PREFLIGHT_ITERS * len(LAMBDA_LADDER) + args.iters) * 305 / 3600
        print(f"DRYRUN est hours: loop {loop_h:.2f} + tripwire "
              f"{2*TRIPWIRE_GAMES*7.95/3600:.2f} + students 0.7 + h2h "
              f"{H2H_GAMES*11.35/3600:.2f}")
        return 0

    # ---- stage 0: preflight (analysis-side first, so it works pre-commit) ----
    for name in ("t094_loop_anchored.py", "t094_pipeline.py", "t095_gate.py",
                 "t095_pipeline.py", "t098_loop_shared.py", "t098_pipeline.py"):
        source = repo / "experiments/alphazero" / name
        linked = clone / name
        if not linked.exists() or not os.path.samefile(source, linked):
            raise RuntimeError(f"clone {name} is not linked to experiments SSOT")
    gate_self_test(logs)
    adapter = yane_selftest(logs)
    if args.self_test:
        print("T098_SELFTEST_OK", flush=True)
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    for path in (pgx_py, torch_py, binbo, yane, yane_eval / "nn.bin", wst,
                 anchor_data / "train_obs.npy", anchor_data / "val_v.npy",
                 cache / "meta.json"):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})
    print(f"T098_PREFLIGHT_OK head={head} exp_dirty=0", flush=True)

    # ---- stage 1: cargo test ------------------------------------------------
    if not args.skip_cargo:
        run_logged(["cargo", "test", "--release"], logs / f"{args.run_tag}_cargo_test.log",
                   cwd=repo, env=env)
        print("T098_CARGO_TEST_OK", flush=True)

    def loop_command(target, lam, out_dir, monitor=5, ckpt_every=5):
        return [
            pgx_py, "t098_loop_shared.py",
            f"max_num_iters={target}", "num_simulations=32",
            "selfplay_batch_size=128", "states_per_iter=65536",
            "training_batch_size=128", "learning_rate=0.0001",
            f"init_ckpt={wst}", f"ckpt_dir={out_dir}", "resume=auto",
            f"ckpt_interval={ckpt_every}", f"monitor_interval={monitor}",
            "expected_iter_s=420", f"anchor_data_dir={anchor_data}",
            "anchor_ratio=0.5", f"seed={LOOP_SEED}", f"anchor_seed={LOOP_SEED}",
            f"value_lambda={lam}", f"value_std_ratio_stop={VALUE_STD_RATIO_STOP}",
        ]

    # ---- stage 2: lambda preflight (value health only, never Elo) -----------
    trials = {}
    for lam in LAMBDA_LADDER:
        tag = f"lam{str(lam).replace('.', 'p')}"
        out_dir = models / f"{args.run_tag}_pre_{tag}_ckpt"
        log = logs / f"{args.run_tag}_pre_{tag}.log"
        if not (out_dir / f"{LAMBDA_PREFLIGHT_ITERS:06d}.ckpt").exists():
            run_logged(loop_command(LAMBDA_PREFLIGHT_ITERS, lam, out_dir,
                                    monitor=LAMBDA_PREFLIGHT_ITERS,
                                    ckpt_every=LAMBDA_PREFLIGHT_ITERS),
                       log, cwd=clone, env=env)
        trials[lam] = parse_iter_metrics(log, LAMBDA_PREFLIGHT_ITERS)
        print(f"T098_LAMBDA_TRIAL lambda={lam} "
              f"std_ratio={trials[lam].get('value_std_ratio')} "
              f"corr={trials[lam].get('value_corr_teacher')} "
              f"mae={trials[lam].get('value_mae_wst')} "
              f"anchor_top1={trials[lam].get('anchor_top1')}", flush=True)
    chosen, status = select_lambda(trials)
    lambda_record = {"chosen": chosen, "status": status,
                     "ladder": {str(k): v for k, v in trials.items()},
                     "rule": f"largest with mae>={VALUE_MOVED_MAE_CP}cp, "
                             f"std_ratio>={LAMBDA_STD_RATIO_MIN}, corr>={LAMBDA_CORR_MIN}"}
    print(f"T098_LAMBDA_SELECTED lambda={chosen} status={status}", flush=True)

    # ---- stage 3: the main loop ---------------------------------------------
    loop_log = logs / f"{args.run_tag}_loop_to_{args.iters:03d}.log"
    if not main_ckpt.exists():
        try:
            run_logged(loop_command(args.iters, chosen, ckpt_dir), loop_log, cwd=clone, env=env)
        except RuntimeError:
            if "T098_VALUE_STOP" not in loop_log.read_text(encoding="utf-8"):
                raise
            print("T098_LOOP_HALTED_BY_STOP_RULE", flush=True)
    print("T098_LOOP_DONE", flush=True)
    loop_metrics = parse_iter_metrics(loop_log, args.iters)

    # ---- stage 4: gikou@6 tripwire (paired against the 097 WS-T games) ------
    trip_paths, trip_base = [], []
    for block in TRIPWIRE_BLOCKS:
        seed = BLOCK_SEEDS[block]
        path = logs / (f"{args.run_tag}_shared_iter{args.iters:03d}_vs_gikou6_"
                       f"s128_b100_seed{seed}_n{TRIPWIRE_GAMES}.tsv")
        base = logs / f"task097_wst_vs_gikou6_s128_b100_seed{seed}_n{TRIPWIRE_GAMES}.tsv"
        if not base.exists():
            raise FileNotFoundError(base)
        if not complete_tsv(path, TRIPWIRE_GAMES):
            run_logged([pgx_py, "m2_match.py", "--az-ckpt", main_ckpt, "--az-sims", "128",
                        "--opponent", "gikou", "--games", str(TRIPWIRE_GAMES),
                        "--byoyomi", "100", "--opening-plies", "6", "--max-plies", "320",
                        "--seed", str(seed), "--tsv", path],
                       Path(str(path).replace(".tsv", ".console.log")), cwd=clone, env=env)
        trip_paths.append(path)
        trip_base.append(base)
    tripwire = paired_delta_general(trip_paths, trip_base, label="shared_minus_wst_gikou")
    print("T098_EXTERNAL_TRIPWIRE done", flush=True)

    # ---- stage 5: rescore -> students -> export -----------------------------
    # az_rescore.py is the Task 084 instrument. t091_rescore --reference-value-dir
    # would OVERWRITE the candidate's values with WS-T's (values[:] = reference)
    # and only then compute value_cp_std, so it would measure nothing here.
    rescore = {}
    students = {}
    for name, ckpt in (("cand", main_ckpt), ("wst", wst)):
        out = work / f"rescore_{name}"
        log = logs / f"{args.run_tag}_rescore_{name}.log"
        if not (out / "train_dlshogi_evals.npy").exists():
            run_logged([pgx_py, "az_rescore.py", "--cache-dir", cache,
                        "--ckpt", ckpt, "--out-dir", out], log, cwd=clone, env=env)
        rescore[name] = [ln for ln in log.read_text(encoding="utf-8").splitlines()
                         if "std" in ln or "corr" in ln]
        model_dir = models / f"{args.run_tag}_student_{name}"
        if not (model_dir / "nnue_best.pt").exists():
            run_logged([torch_py, "-m", "trainer.train_nnue", cache, "--out", model_dir,
                        *NNUE_ARGS, "--eval-npy", out],
                       logs / f"{args.run_tag}_student_{name}.log", cwd=repo, env=env)
        eval_dir = eval_dirs / f"{args.run_tag}_student_{name}"
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            run_logged([torch_py, "-m", "trainer.export_nnue", model_dir,
                        "--weights", "nnue_best.pt", "--out", model_dir / "nnue_best.bnnq"],
                       logs / f"{args.run_tag}_export_{name}.log", cwd=repo, env=env)
            run_logged([torch_py, "-m", "trainer.binbo_to_yane",
                        "--from-bnnq", model_dir / "nnue_best.bnnq",
                        "--template", yane_eval / "nn.bin", "--out", eval_dir / "nn.bin"],
                       logs / f"{args.run_tag}_yane_{name}.log", cwd=repo, env=env)
        gate_log = logs / f"{args.run_tag}_0cp_{name}.log"
        if not gate_log.exists():
            run_logged(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"],
                       gate_log, cwd=repo, env=env)
        gate_text = gate_log.read_text(encoding="utf-8")
        if "RED" in gate_text or gate_text.count("GREEN") < 2:
            raise RuntimeError(f"0cp gate failed for {name}: see {gate_log}")
        students[name] = {"model_dir": str(model_dir), "eval_dir": str(eval_dir),
                          "gate": "GREEN"}
    print("T098_STUDENTS_READY", flush=True)

    # ---- stage 6: student head-to-head, fixed N ------------------------------
    # eval_vs_eval.sh stops sequentially (Task 084 ended at 177 games). Calling
    # the binary directly with elo0 == elo1 makes every LLR term 1, so the SPRT
    # verdict stays Continue and the run goes to the full game count.
    h2h_tsv = logs / f"{args.run_tag}_student_cand_vs_wst_n{H2H_GAMES}.tsv"
    if not h2h_tsv.exists():
        run_logged([binbo, "yane-vs-yane", yane,
                    students["cand"]["eval_dir"], students["wst"]["eval_dir"],
                    "100", str(H2H_SEED), str(H2H_GAMES), "6", "320", "0", "0",
                    "0.05", "0.05"],
                   h2h_tsv, cwd=repo, env=env)
    rows = load_yane_tsv(h2h_tsv)
    shape = validate_h2h(rows, h2h_tsv, expected_games=H2H_GAMES)
    h2h = paired_h2h_delta(rows, label="student_cand_minus_wst")

    # ---- stage 7: gate ------------------------------------------------------
    failures = []
    if shape["bad_reasons"]:
        failures.append(f"illegal_or_engine_fail_{len(shape['bad_reasons'])}")
    if not main_ckpt.exists():
        failures.append("loop_checkpoint_missing")
    if loop_metrics.get("value_std_ratio", 0.0) < VALUE_STD_RATIO_STOP:
        failures.append("value_std_collapsed")
    prerequisites = {"failures": failures, "adapter_selftest": adapter,
                     "h2h_shape": shape}

    results = analyse(h2h, tripwire, loop_metrics, lambda_record, rescore,
                      prerequisites, students)
    results["git"] = {"head": head, "exp_dirty": 0, "pushed": False}
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    print(f"T098_PIPELINE_DONE verdict={results['verdict']} "
          f"terminal={results['terminal']} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
