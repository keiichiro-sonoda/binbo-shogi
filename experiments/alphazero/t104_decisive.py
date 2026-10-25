#!/usr/bin/env python3
"""Task 104: the 32M decisive test -- does a search-backed label beat gikou@d8 at champion scale?

Tasks 100-103 measured search-label students against raw-label students only
(1M +197 -> 2M +155 -> 4M +124 -> 8M +107, decay slowing; S=8 == S=32 at 1M).
None of that says whether the search label beats the label the champion was
distilled from. This is that test, on the champion's own ground.

ONE VARIABLE = the teacher label family, at the champion's corpus and recipe:
    champion : task064_32m_lr1e-5 = gikou@d8 search cp on caches/shared_32m
               (31,950,000 train rows + 50,000 val), recipe below. REUSED, not retrained.
    search   : WS-T + Gumbel MCTS S=8 root value, cp = 1200*atanh(v), on the SAME
               rows in cache order, trained with the SAME recipe (only --eval-npy differs).
Recipe (Task 064 of record): NNUE_ARGS (098 = --qat --qat-tie 256x2-32-32, lr 1e-5,
12 epochs, batch 1024, scale-cp 600) + --augment-mirror --mirror-cache
caches/shared_8m_mirror_perm.npy. As in 100-103 the search student's val rows
carry search labels, so nnue_best.pt is selected against its own teacher (the
champion's against gikou): each arm selects on its own family.

PRE-REGISTERED (committed before the run; results never seen):
  * primary = paired bootstrap over colour-swapped opening pairs of search student
    vs champion, yane-vs-yane 100 ms byoyomi, Threads 1, 6 random opening plies,
    max 320 plies (the 098-103 protocol), 1600 games = 4 workers x 400 games,
    seeds 20271001-04. Expected half-width ~16 Elo (098: 23.1 at 800 games).
    A GSPRT LLR (elo0=0, elo1=10, normal approximation) is RECORDED ONLY.
  * hard gates (all before any Elo is seen): sign oracle at S=8; per-shard
    gikou==cache alignment on every shard; raw-head wiring gate (>= 50% exact,
    every row within 10 cp) against work/task102a/rescore_raw on the prefix it
    covers (shared_32m's first 7,950,000 train rows ARE shared_8m's train rows;
    val is the same 5 shards) -- rows past the prefix are protected by the
    alignment assert only; search labels moved (median |search - raw| >= 30 cp,
    the S=8 bar of 103b, which measured 177); 0cp gate GREEN for the search
    student AND the champion; zero illegal moves / engine failures.
  * decision table (primary):
        0 prerequisites failed -> INVALID (non-terminal)
        1 CI low > 0           -> SEARCH_BEATS_CHAMPION: new champion candidate;
                                  next = confirm in the absolute frames, then adopt
        2 CI high < 0          -> CHAMPION_HOLDS: gikou@d8 labels win at 32M;
                                  next = the ensemble-label arm (mean of gikou@d8 and
                                  search cp on the same rows; labels reused, training only)
        3 CI straddles 0       -> PARITY: next = the same ensemble-label arm
  * secondary (descriptive, pre-registered classes; never change the primary):
      S1 the 32M scale-curve point: a raw-label student (WS-T raw head, written by
         the same rescore pass, same recipe) vs the search student, 800 games,
         seeds 20270911-14 x 200 (the 100-103 series). Against 103's two
         extrapolations (log2-linear 4-point fit +41, geometric-decay model +85):
         CI high < 0 -> REVERSED; CI high < 41 -> FASTER_DECAY; CI contains 41
         and not 85 -> LINEAR_DECAY; contains 85 and not 41 -> PLATEAU;
         contains both -> UNRESOLVED; CI low > 85 -> ABOVE_MODELS.
      S2 the search student vs gikou@6 (yane-vs-gikou, 800 games, seeds
         20270911-14 = 103c/103d), next to 8M's +74.5 [+51.2, +98.5].
  * order (GPU never shared): rescore -> search student -> export + 0cp ->
    primary match (verdict written here) -> raw student -> S1 -> S2.
  * resumable: shard parts, trained models, exported evals and finished TSVs are
    reused on re-run with the same command.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from m2_usi import load_model  # noqa: E402
from t094_pipeline import run_logged  # noqa: E402
from t098_pipeline import NNUE_ARGS, load_yane_tsv, paired_h2h_delta, validate_h2h, yane_selftest  # noqa: E402
from t099_search_label import build_batched_search, score_sfens, sign_oracle, value_to_cp  # noqa: E402
from t100_search_student import rescore_cache_with_search, search_conditions  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/GitHub/shogi/binbo-shogi"))
from trainer.prepare_data import shard_files  # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9))

# ---- pre-registered constants -------------------------------------------------
SIMS = 8
BATCH = 128                                    # t099 SEARCH_BATCH (103b ran at this)
LIMIT_TRAIN = 31_950_000                       # all of shared_32m's train rows
LABEL_MOVED_MIN_CP = 30.0                      # 103b's S=8 bar (measured 177)
CHAMPION = "task064_32m_lr1e-5"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"   # -> *_shared.npy, Task 064 of record
RECIPE_EXTRA = ["--augment-mirror"]
PRIMARY_WORKERS, PRIMARY_GAMES = 4, 400        # 1600 games = 800 opening pairs
PRIMARY_SEEDS = (20271001, 20271002, 20271003, 20271004)
PRIMARY_HALF_WIDTH_PLAN = 16.3                 # 23.1 / sqrt(2)
SECONDARY_WORKERS, SECONDARY_GAMES = 4, 200    # 800 games, the 100-103 series
SECONDARY_SEEDS = (20270911, 20270912, 20270913, 20270914)
SPRT_ELO0, SPRT_ELO1 = 0.0, 10.0               # recorded only
EXTRAP_LINEAR, EXTRAP_PLATEAU = 41.0, 85.0     # Task 103 report
SEARCH_ASYMMETRY_DEPTH, SEARCH_ASYMMETRY_NODES = 0.30, 0.05

DECISION_TABLE = (
    "0 prerequisites failed -> INVALID (non-terminal) | "
    "1 CI low > 0 -> SEARCH_BEATS_CHAMPION (confirm in absolute frames, then adopt) | "
    "2 CI high < 0 -> CHAMPION_HOLDS (next: ensemble-label arm) | "
    "3 CI straddles 0 -> PARITY (next: ensemble-label arm)"
)


def now_jst():
    return dt.datetime.now(JST)


def elo_to_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def gsprt_llr(rows, elo0=SPRT_ELO0, elo1=SPRT_ELO1):
    """Normal-approximation GSPRT LLR over games (recorded only, never decides)."""
    x = np.array([r["score"] for r in rows], dtype=np.float64)
    n, s = len(x), float(x.mean())
    var = float(np.mean(x * x) - s * s)
    if n == 0 or var <= 0:
        return None
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return {"elo0": elo0, "elo1": elo1, "llr": n * (s1 - s0) * (2 * s - s0 - s1) / (2 * var),
            "bounds": [math.log(0.05 / 0.95), math.log(0.95 / 0.05)]}


def run_series(binbo, yane, cand, base, logs, tag, games, seeds, env, cwd, gikou=None):
    """Complete colour-swapped series in parallel (one worker per seed)."""
    procs, tsvs = [], []
    for index, seed in enumerate(seeds):
        tsv = logs / f"{tag}_w{index}_seed{seed}_n{games}.tsv"
        tsvs.append(tsv)
        if tsv.exists() and "elapsed_ms=" in tsv.read_text(encoding="utf-8"):
            continue
        if gikou is None:
            command = [binbo, "yane-vs-yane", yane, cand, base, "100", seed, games,
                       "6", "320", "0", "0", "0.05", "0.05"]
        else:
            command = [binbo, "yane-vs-gikou", yane, cand, gikou, "100", "6", seed, games,
                       "6", "320", "100", "0", "0", "0.05", "0.05"]
        command = [str(c) for c in command]
        print(f"RUN {' '.join(command)} log={tsv}", flush=True)
        handle = tsv.open("w", encoding="utf-8")
        procs.append((subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT), handle, tsv))
    for process, handle, tsv in procs:
        rc = process.wait()
        handle.close()
        if rc != 0:
            raise RuntimeError(f"match worker failed rc={rc}: {tsv}")
    rows, offset, bad = [], 0, []
    for tsv in tsvs:
        part = load_yane_tsv(tsv)
        bad += validate_h2h(part, tsv, expected_games=games)["bad_reasons"]
        for row in part:
            row = dict(row)
            row["game"] += offset
            rows.append(row)
        offset += len(part)
    validate_h2h(rows, "combined", expected_games=games * len(tsvs))
    per = [search_conditions(t) for t in tsvs]
    cond = {side: {k: float(np.mean([c[side][k] for c in per])) for k in ("avg_depth", "avg_nodes")}
            for side in ("cand", "base")}
    return rows, bad, cond, tsvs


def classify_scale_point(ci):
    low, high = ci
    has_lin = low <= EXTRAP_LINEAR <= high
    has_pla = low <= EXTRAP_PLATEAU <= high
    if high < 0:
        return "REVERSED"
    if low > EXTRAP_PLATEAU:
        return "ABOVE_MODELS"
    if has_lin and has_pla:
        return "UNRESOLVED"
    if has_lin:
        return "LINEAR_DECAY"
    if has_pla:
        return "PLATEAU"
    if high < EXTRAP_LINEAR:
        return "FASTER_DECAY"
    return "BETWEEN_MODELS"


def self_test(cache, wst, raw_ref_dir, scratch):
    """Score one late shard (run16m_f, past the raw prefix) at its true cache
    offset: alignment, finiteness and label movement on the part of the corpus
    the wiring gate cannot see. The prefix path is exercised by 103a/103b."""
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    val_count = len(meta["val_shards"])
    primary, _ = shard_files(meta["run_dir"])
    groups = ([(meta["run_dir"], s) for s in primary[:-val_count]]
              + [(rd, s) for rd in meta["extra_runs"] for s in shard_files(rd)[0]])
    cache_eval = np.load(cache / "train_evals.npy", mmap_mode="r")
    offset = 0
    target = None
    for rd, shard in groups:
        m = os.path.getsize(os.path.join(rd, shard["file"])) // cshogi.HuffmanCodedPosAndEval.itemsize
        if Path(rd).name == "run16m_f":
            target = (rd, shard, offset, m)
            break
        offset += m
    if target is None:
        raise RuntimeError("self-test: run16m_f not found in the cache traversal")
    rd, shard, offset, m = target
    entries = np.fromfile(os.path.join(rd, shard["file"]), dtype=cshogi.HuffmanCodedPosAndEval)[:2000]
    if not np.array_equal(entries["eval"].astype(np.int16), np.asarray(cache_eval[offset:offset + len(entries)])):
        raise AssertionError(f"self-test: run16m_f shard {shard['index']} misaligned at offset {offset}")
    board = cshogi.Board()
    sfens = []
    for e in entries:
        board.set_hcp(np.asarray(e["hcp"]))
        sfens.append(board.sfen())
    forward, model, _ = load_model(str(wst))
    search = build_batched_search(forward, model, SIMS)
    t0 = time.time()
    values, _q, raw, _mv = score_sfens(search, sfens, BATCH)
    rate = len(sfens) / (time.time() - t0)
    s_cp, r_cp = value_to_cp(values), value_to_cp(raw)
    moved = float(np.median(np.abs(s_cp.astype(np.float64) - r_cp.astype(np.float64))))
    if not (np.isfinite(values).all() and np.isfinite(raw).all()):
        raise AssertionError("self-test: non-finite labels")
    if moved < LABEL_MOVED_MIN_CP:
        raise AssertionError(f"self-test: search labels did not move (median {moved:.0f} cp)")
    ref = np.load(raw_ref_dir / "train_dlshogi_evals.npy", mmap_mode="r")
    print(f"T104_SELFTEST_OK late_shard=run16m_f/{shard['index']} offset={offset} rows={len(sfens)} "
          f"moved_median={moved:.0f}cp rate={rate:.0f}/s (incl. JIT) raw_ref_rows={ref.shape[0]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", default="task104")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    parser.add_argument("--no-secondary", action="store_true", help="stop after the primary verdict")
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    eval_dirs = data_root / "eval_dirs"
    torch_py = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    binbo = repo / "target/release/binbo-shogi"
    yane = data_root / "engines/yaneuraou/source/YaneuraOu-by-gcc"
    yane_eval = data_root / "engines/yaneuraou/eval"
    gikou = data_root / "engines/gikou/gikou"
    wst = clone / "t087_wst/ws2.ckpt"
    cache = data_root / "caches/shared_32m"
    raw_ref = data_root / "work/task102a/rescore_raw"
    mirror_cache = data_root / MIRROR_CACHE
    champion_model = models / CHAMPION
    champion_eval = eval_dirs / CHAMPION
    work = data_root / "work" / args.run_tag
    search_dir, raw_dir = work / "rescore_search", work / "rescore_raw"
    recipe = [*NNUE_ARGS, *RECIPE_EXTRA, "--mirror-cache", str(mirror_cache)]

    if args.dry_run:
        rate = 960.0
        print(f"DRYRUN corpus {cache} train {LIMIT_TRAIN} + val 50000; labeller WS-T+search S={SIMS} batch={BATCH}")
        print(f"DRYRUN recipe {recipe}")
        print(f"DRYRUN champion {champion_model} (reused) / eval {champion_eval}")
        print(f"DRYRUN primary {PRIMARY_WORKERS}x{PRIMARY_GAMES} seeds={PRIMARY_SEEDS}; "
              f"secondary S1/S2 {SECONDARY_WORKERS}x{SECONDARY_GAMES} seeds={SECONDARY_SEEDS}")
        print(f"DRYRUN decision table: {DECISION_TABLE}")
        print(f"DRYRUN est: rescore {(LIMIT_TRAIN + 50000) / rate / 3600:.1f}h @ {rate:.0f}/s, "
              f"search student ~3.0h, export+gates ~0.2h, primary ~1.3h, raw student ~3.0h, S1+S2 ~1.4h")
        return 0

    # ---- stage 0: preflight + gates -------------------------------------------
    for name in ("m2_usi.py", "t094_pipeline.py", "t098_pipeline.py", "t099_search_label.py",
                 "t100_search_student.py", "t104_decisive.py"):
        linked = clone / name
        if not linked.exists() or not os.path.samefile(repo / "experiments/alphazero" / name, linked):
            raise RuntimeError(f"clone {name} is not linked to experiments SSOT")
    for path in (torch_py, binbo, yane, yane_eval / "nn.bin", gikou, wst, cache / "meta.json",
                 raw_ref / "train_dlshogi_evals.npy", raw_ref / "val_dlshogi_evals.npy",
                 champion_model / "nnue_best.bnnq", champion_eval / "nn.bin",
                 mirror_cache.with_name(mirror_cache.stem + "_shared.npy")):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    history = json.loads((champion_model / "history.json").read_text(encoding="utf-8"))
    if history.get("n_train") != LIMIT_TRAIN:
        raise RuntimeError(f"champion n_train {history.get('n_train')} != {LIMIT_TRAIN}")
    adapter = yane_selftest(logs)
    forward, model, _ = load_model(str(wst))
    oracle = sign_oracle(forward, model, SIMS)
    del forward, model
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})

    if args.self_test:
        self_test(cache, wst, raw_ref, work / "selftest")
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    work.mkdir(parents=True, exist_ok=True)
    print(f"T104_PREFLIGHT_OK head={head} champion={CHAMPION} start={now_jst():%m-%d %H:%M} JST", flush=True)
    if not args.skip_cargo:
        run_logged(["cargo", "test", "--release"], logs / f"{args.run_tag}_cargo_test.log", cwd=repo, env=env)
        print("T104_CARGO_TEST_OK", flush=True)

    # champion's own 0cp gate (bnnq -> yane nn.bin, in a temp dir; the eval of record is untouched)
    champ_gate = logs / f"{args.run_tag}_0cp_champion.log"
    if not champ_gate.exists():
        run_logged(["bash", "scripts/binbo_to_yane.sh", str(champion_model / "nnue_best.bnnq")],
                   champ_gate, cwd=repo, env=env)
    text = champ_gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        raise RuntimeError(f"0cp gate failed for the champion: {champ_gate}")
    print("T104_CHAMPION_0CP_GREEN", flush=True)

    # ---- stage 1: rescore (GPU) -------------------------------------------------
    print("=== T104 STAGE rescore ===", flush=True)
    if not (search_dir / "meta.json").exists():
        labels = rescore_cache_with_search(cache, wst, search_dir, limit_train=LIMIT_TRAIN, sims=SIMS,
                                           batch=BATCH, raw_dir=raw_ref, raw_ref_prefix=True,
                                           raw_out_dir=raw_dir)
    else:
        labels = json.loads((search_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    print(f"T104_RESCORE_DONE elapsed={labels['elapsed_s']/3600:.2f}h", flush=True)

    def student(name, label_dir):
        model_dir = models / f"{args.run_tag}_student_{name}"
        if not (model_dir / "nnue_best.pt").exists():
            print(f"=== T104 STAGE train {name} ===", flush=True)
            run_logged([str(torch_py), "-m", "trainer.train_nnue", str(cache), "--out", str(model_dir),
                        *recipe, "--eval-npy", str(label_dir), "--limit-train", str(LIMIT_TRAIN)],
                       logs / f"{args.run_tag}_student_{name}.log", cwd=repo, env=env)
        eval_dir = eval_dirs / f"{args.run_tag}_student_{name}"
        if not (eval_dir / "nn.bin").exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            run_logged([str(torch_py), "-m", "trainer.export_nnue", str(model_dir), "--weights", "nnue_best.pt",
                        "--out", str(model_dir / "nnue_best.bnnq")],
                       logs / f"{args.run_tag}_export_{name}.log", cwd=repo, env=env)
            run_logged([str(torch_py), "-m", "trainer.binbo_to_yane", "--from-bnnq",
                        str(model_dir / "nnue_best.bnnq"), "--template", str(yane_eval / "nn.bin"),
                        "--out", str(eval_dir / "nn.bin")],
                       logs / f"{args.run_tag}_yane_{name}.log", cwd=repo, env=env)
        gate = logs / f"{args.run_tag}_0cp_{name}.log"
        if not gate.exists():
            run_logged(["bash", "scripts/binbo_to_yane.sh", str(model_dir / "nnue_best.bnnq")],
                       gate, cwd=repo, env=env)
        gtext = gate.read_text(encoding="utf-8")
        if "RED" in gtext or gtext.count("GREEN") < 2:
            raise RuntimeError(f"0cp gate failed for {name}: {gate}")
        hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
        summary = {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")}
        print(f"T104_STUDENT_READY {name} best_epoch={summary['best_epoch']} gate=GREEN", flush=True)
        return {"model_dir": str(model_dir), "eval_dir": str(eval_dir), "gate": "GREEN", "training": summary}

    # ---- stage 2-3: search student, export, gate ----------------------------------
    students = {"search": student("search", search_dir)}

    # ---- stage 4: primary, search student vs champion -----------------------------
    failures = []
    if labels["train"]["raw_exact_vs_098"] < 0.50:
        failures.append("raw_head_mismatch_prefix")
    if labels["train"]["abs_median_search_vs_raw"] < LABEL_MOVED_MIN_CP:
        failures.append("search_label_not_applied")
    print("=== T104 STAGE primary match (search student vs champion, 1600 games) ===", flush=True)
    start = now_jst()
    rows, bad, cond, tsvs = run_series(binbo, yane, students["search"]["eval_dir"], champion_eval, logs,
                                       f"{args.run_tag}_primary", PRIMARY_GAMES, PRIMARY_SEEDS, env, repo)
    if bad:
        failures.append(f"illegal_or_engine_fail_{len(bad)}")
    h2h = paired_h2h_delta(rows, label="search_student_minus_champion")
    low, high = h2h["ci95"]
    if failures:
        verdict, terminal = "INVALID", False
    elif low > 0:
        verdict, terminal = "SEARCH_BEATS_CHAMPION", True
    elif high < 0:
        verdict, terminal = "CHAMPION_HOLDS", True
    else:
        verdict, terminal = "PARITY", True
    flags = []
    if h2h["half_width"] > 1.5 * PRIMARY_HALF_WIDTH_PLAN:
        flags.append("UNDERPOWERED_VS_PLAN")
    depth_gap = abs(cond["cand"]["avg_depth"] - cond["base"]["avg_depth"])
    nodes_ratio = cond["cand"]["avg_nodes"] / max(1.0, cond["base"]["avg_nodes"])
    if depth_gap > SEARCH_ASYMMETRY_DEPTH or abs(nodes_ratio - 1) > SEARCH_ASYMMETRY_NODES:
        flags.append("SEARCH_ASYMMETRY")
    results = {
        "task": 104,
        "preregistration": {
            "one_variable": f"teacher label family at champion scale: gikou@d8 (champion {CHAMPION}) vs "
                            f"WS-T + Gumbel MCTS S={SIMS} root value",
            "corpus": f"{cache.name}: {LIMIT_TRAIN} train rows + 50000 val (all), cache order",
            "recipe": recipe,
            "primary": f"yane-vs-yane 100ms/Threads1, {PRIMARY_WORKERS}x{PRIMARY_GAMES} games, seeds "
                       f"{PRIMARY_SEEDS}, paired bootstrap over opening pairs",
            "decision_table": DECISION_TABLE,
            "hard_gates": ["sign_oracle_S8", "per_shard_alignment", "raw_head_wiring_on_7.95M_prefix",
                           "search_label_moved_30cp", "0cp_gate_search_and_champion", "no_illegal"],
            "sprt_recorded_only": [SPRT_ELO0, SPRT_ELO1],
            "secondary": {"S1": f"raw vs search student @32M, 800 games; classes vs +{EXTRAP_LINEAR:.0f} "
                                f"(linear) / +{EXTRAP_PLATEAU:.0f} (plateau)",
                          "S2": "search student vs gikou@6, 800 games (103c/103d seeds)"},
        },
        "primary_h2h": h2h,
        "sprt_recorded_only": gsprt_llr(rows),
        "search_conditions": cond,
        "label_stats": labels,
        "students": students,
        "clock": {"match_start_jst": start.strftime("%Y-%m-%d %H:%M"), "match_end_jst": now_jst().strftime("%Y-%m-%d %H:%M")},
        "prerequisites": {"failures": failures, "adapter_selftest": adapter, "sign_oracle": oracle,
                          "bad_reasons": bad, "tsvs": [str(t) for t in tsvs]},
        "flags": flags,
        "verdict": verdict,
        "terminal": terminal,
        "git": {"head": head, "exp_dirty": 0, "pushed": False},
    }
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"T104_PRIMARY W-D-L={h2h['wins']}-{h2h['draws']}-{h2h['losses']} score={h2h['score']:.4f} "
          f"elo={h2h['elo']:+.2f} ci={[round(v, 2) for v in h2h['ci95']]} hw={h2h['half_width']:.1f} "
          f"depth={cond['cand']['avg_depth']:.2f}/{cond['base']['avg_depth']:.2f} "
          f"nodes={cond['cand']['avg_nodes']:.0f}/{cond['base']['avg_nodes']:.0f}", flush=True)
    print(f"FLAGS {flags}", flush=True)
    print(f"T104_VERDICT verdict={verdict} terminal={terminal} results={output}", flush=True)
    if failures or args.no_secondary:
        print(f"T104_PIPELINE_DONE verdict={verdict} secondary=skipped", flush=True)
        return 0 if not failures else 1

    # ---- secondary S1: the 32M scale-curve point -----------------------------------
    students["raw"] = student("raw", raw_dir)
    print("=== T104 STAGE S1 search vs raw student (800 games) ===", flush=True)
    rows1, bad1, cond1, tsvs1 = run_series(binbo, yane, students["search"]["eval_dir"],
                                           students["raw"]["eval_dir"], logs, f"{args.run_tag}_s1",
                                           SECONDARY_GAMES, SECONDARY_SEEDS, env, repo)
    s1 = paired_h2h_delta(rows1, label="search_student_minus_raw_student_32M")
    s1_class = classify_scale_point(s1["ci95"])
    print(f"T104_S1 W-D-L={s1['wins']}-{s1['draws']}-{s1['losses']} elo={s1['elo']:+.2f} "
          f"ci={[round(v, 2) for v in s1['ci95']]} class={s1_class} bad={len(bad1)}", flush=True)

    # ---- secondary S2: the gikou@6 frame --------------------------------------------
    print("=== T104 STAGE S2 search student vs gikou@6 (800 games) ===", flush=True)
    rows2, bad2, cond2, tsvs2 = run_series(binbo, yane, students["search"]["eval_dir"], None, logs,
                                           f"{args.run_tag}_s2_vs_gikou6", SECONDARY_GAMES,
                                           SECONDARY_SEEDS, env, repo, gikou=gikou)
    s2 = paired_h2h_delta(rows2, label="search_student_32M_minus_gikou6")
    print(f"T104_S2 W-D-L={s2['wins']}-{s2['draws']}-{s2['losses']} elo={s2['elo']:+.2f} "
          f"ci={[round(v, 2) for v in s2['ci95']]} depth={cond2['cand']['avg_depth']:.2f}/"
          f"{cond2['base']['avg_depth']:.2f} bad={len(bad2)}", flush=True)

    results["secondary"] = {
        "S1": {"h2h": s1, "class": s1_class, "conditions": cond1, "bad_reasons": bad1,
               "tsvs": [str(t) for t in tsvs1]},
        "S2": {"h2h": s2, "conditions": cond2, "bad_reasons": bad2, "tsvs": [str(t) for t in tsvs2]},
    }
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"T104_PIPELINE_DONE verdict={verdict} s1={s1_class} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
