#!/usr/bin/env python3
"""Task 105: the 1:1 average of the gikou@d8 label and the search label, plus a stronger anchor.

Task 104 adopted the search-label student (`task104_student_search`). Task 104's
decision table had the averaged-label arm as the next step, and the industry
practice (ensemble teachers) points the same way. Both labels already exist for
all 32M positions, so this costs one student training and games -- no rescore.
In parallel (CPU) the gikou@6 anchor, which the adopted eval now beats 87%, is
replaced by a stronger one so that the next students can still be told apart.

PRE-REGISTERED (committed before the run; no averaged-label result is seen):
  * one variable: the teacher label. Corpus `caches/shared_32m` (31,950,000 train
    + 50,000 val), the Task 104 recipe (098 NNUE_ARGS + --augment-mirror with the
    shared_8m mirror perm). Label = (clip(gikou, +-2000) + clip(search, +-2000)) / 2,
    float32, where gikou = the cache's own evals (the champion's teacher) and
    search = `work/task104/rescore_search` (the adopted eval's teacher). +-2000 is
    train_nnue's --clamp-cp, i.e. the range each parent was trained on.
  * anchor: gikou DepthLimit ANCHOR_LEVEL, 100 ms think (yane-vs-gikou). Chosen
    from a calibration pilot on separate seeds 20271101-04 (search student vs
    gikou@7, 100 games); depth >= 8 is known to crash gikou mid-game. Pilot result
    (10/4 02:00): 78-0-22 (score 0.78, vs 0.87 at gikou@6), 0 errors, anchor depth 6.98
    -> ANCHOR_LEVEL = 7.
  * stage yardstick (CPU, runs while the student trains): champion and search
    student vs the anchor, 800 games each, seeds 20270911-14 (the S2 protocol).
    Descriptive: both absolute values and the frame delta (search - champion),
    a re-read of Task 104b's +55.6 on the stronger anchor.
  * stage match (after both): PRIMARY = averaged student vs search student,
    yane-vs-yane 100 ms / Threads 1, 1600 games (4 x 400, seeds 20271001-04),
    paired bootstrap over opening pairs. FRAME = averaged student vs the anchor,
    800 games on the yardstick seeds, frame delta (avg - search) paired over the
    400 shared openings (t103_absolute.paired_series_delta).
  * hard gates (before any Elo): label build alignment (both parents 31,950,000 /
    50,000 rows, finite), averaged label differs from both parents (median |d| >=
    30 cp to each), 0cp GREEN for the averaged student, no illegal move / engine
    failure in any series, anchor depth within 0.1 of ANCHOR_LEVEL.
  * decision table:
      0 a gate failed                                   -> INVALID (non-terminal)
      1 primary CI low > 0 and frame CI low > 0         -> AVG_ADOPT (adoption candidate,
                                                           the swap waits for human approval)
      2 primary CI low > 0, frame CI low <= 0           -> AVG_DIRECT_ONLY (hold; matchup)
      3 primary CI high < 0                             -> SEARCH_HOLDS
      4 primary CI straddles 0                          -> PARITY
    rows 3/4 next = the qsearch control (docs/BACKLOG.md) or 64M.
  * flags only: TIME_ASYMMETRY when the yane nodes of two arms of one comparison
    differ by more than 5% (the yardstick runs beside GPU training).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t094_pipeline import run_logged  # noqa: E402
from t098_pipeline import NNUE_ARGS, paired_h2h_delta  # noqa: E402
from t103_absolute import combine, conditions, paired_series_delta, run_workers  # noqa: E402

CLAMP_CP = 2000.0
LABEL_MOVED_MIN_CP = 30.0
ANCHOR_LEVEL = 7
ANCHOR_THINK_MS = 100
CHAMPION = "task064_32m_lr1e-5"
SEARCH = "task104_student_search"
SEARCH_LABELS = "work/task104/rescore_search"
MIRROR_CACHE = "caches/shared_8m_mirror_perm.npy"
LIMIT_TRAIN = 31_950_000
PRIMARY_SEEDS, PRIMARY_GAMES = (20271001, 20271002, 20271003, 20271004), 400
FRAME_SEEDS, FRAME_GAMES = (20270911, 20270912, 20270913, 20270914), 200
NODES_TOL = 0.05


class Paths:
    def __init__(self, run_tag):
        self.repo = Path(os.path.realpath(__file__)).parents[2]
        self.data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
        self.logs = Path(os.environ.get("LOGS_DIR", self.data / "logs"))
        self.models = Path(os.environ.get("MODELS_DIR", self.data / "models"))
        self.eval_dirs = self.data / "eval_dirs"
        self.torch_py = Path(os.environ.get("VENV_PY", self.data / "venv/bin/python"))
        self.binbo = self.repo / "target/release/binbo-shogi"
        self.yane = self.data / "engines/yaneuraou/source/YaneuraOu-by-gcc"
        self.yane_eval = self.data / "engines/yaneuraou/eval"
        self.gikou = self.data / "engines/gikou/gikou"
        self.cache = self.data / "caches/shared_32m"
        self.work = self.data / "work" / run_tag
        self.labels = self.work / "avg_label"
        self.student = self.models / f"{run_tag}_student_avg"
        self.student_eval = self.eval_dirs / f"{run_tag}_student_avg"
        self.yardstick_json = self.logs / f"{run_tag}_yardstick.json"
        self.results_json = self.logs / f"{run_tag}_results.json"


def anchor_series(p, eval_dir, tag):
    """Colour-swapped series of one yane eval vs gikou@ANCHOR_LEVEL on the frame seeds."""
    tsvs = [p.logs / f"{tag}_w{i}_seed{s}_n{FRAME_GAMES}.tsv" for i, s in enumerate(FRAME_SEEDS)]
    commands = [[p.binbo, "yane-vs-gikou", p.yane, eval_dir, p.gikou, "100", ANCHOR_LEVEL, s, FRAME_GAMES,
                 "6", "320", ANCHOR_THINK_MS, "0", "0", "0.05", "0.05"] for s in FRAME_SEEDS]
    run_workers(commands, tsvs, dict(os.environ), p.repo)
    rows, bad = combine(tsvs)
    cond = conditions(tsvs)
    failures = list(bad)
    if abs(cond["b"]["avg_depth"] - ANCHOR_LEVEL) > 0.1:
        failures.append(f"{tag}: anchor depth {cond['b']['avg_depth']:.2f} != {ANCHOR_LEVEL}")
    return rows, cond, failures, tsvs


def same_openings(rows_a, rows_b):
    return len(rows_a) == len(rows_b) and all(
        a["opening"] == b["opening"] and a["color"] == b["color"] for a, b in zip(rows_a, rows_b))


def nodes_flag(cond_a, cond_b, name):
    ratio = cond_a["a"]["avg_nodes"] / max(1.0, cond_b["a"]["avg_nodes"])
    return [f"TIME_ASYMMETRY_{name}"] if abs(ratio - 1) > NODES_TOL else []


def stage_yardstick(p, run_tag):
    print("=== T105 STAGE yardstick (champion and search student vs gikou@%d) ===" % ANCHOR_LEVEL, flush=True)
    rc, cc, fc, tc = anchor_series(p, p.eval_dirs / CHAMPION, f"{run_tag}_champion_vs_gikou{ANCHOR_LEVEL}")
    rs, cs, fs, ts = anchor_series(p, p.eval_dirs / SEARCH, f"{run_tag}_search_vs_gikou{ANCHOR_LEVEL}")
    failures = fc + fs
    if not same_openings(rs, rc):
        failures.append("yardstick: openings differ between arms")
    hc = paired_h2h_delta(rc, label=f"champion_minus_gikou{ANCHOR_LEVEL}")
    hs = paired_h2h_delta(rs, label=f"search_student_minus_gikou{ANCHOR_LEVEL}")
    delta = paired_series_delta(rs, rc, label=f"(search student - champion) in the gikou@{ANCHOR_LEVEL} frame")
    for name, h, c in (("CHAMPION", hc, cc), ("SEARCH", hs, cs)):
        print(f"T105_YARDSTICK_{name} W-D-L={h['wins']}-{h['draws']}-{h['losses']} score={h['score']:.4f} "
              f"elo={h['elo']:+.2f} ci={[round(v, 2) for v in h['ci95']]} "
              f"depth={c['a']['avg_depth']:.2f}/{c['b']['avg_depth']:.2f} nodes={c['a']['avg_nodes']:.0f}", flush=True)
    print(f"T105_YARDSTICK_FRAME search-champion={delta['delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in delta['ci95']]} (104b @6: +55.6)", flush=True)
    out = {"anchor_level": ANCHOR_LEVEL, "champion": {"h2h": hc, "conditions": cc, "tsvs": [str(t) for t in tc]},
           "search": {"h2h": hs, "conditions": cs, "tsvs": [str(t) for t in ts]},
           "frame_delta_search_minus_champion": delta, "failures": failures,
           "flags": nodes_flag(cs, cc, "YARDSTICK")}
    p.yardstick_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"T105_YARDSTICK_DONE failures={failures} flags={out['flags']} out={p.yardstick_json}", flush=True)
    return 0 if not failures else 1


def build_labels(p):
    if (p.labels / "meta.json").exists():
        return json.loads((p.labels / "meta.json").read_text(encoding="utf-8"))
    p.labels.mkdir(parents=True, exist_ok=True)
    meta = {"rule": f"(clip(gikou,+-{CLAMP_CP:g}) + clip(search,+-{CLAMP_CP:g})) / 2, float32",
            "gikou": str(p.cache), "search": str(p.data / SEARCH_LABELS), "stats": {}}
    for split, rows in (("train", LIMIT_TRAIN), ("val", 50_000)):
        g = np.load(p.cache / f"{split}_evals.npy").astype(np.float32)
        s = np.load(p.data / SEARCH_LABELS / f"{split}_dlshogi_evals.npy").astype(np.float32)
        if g.shape != (rows,) or s.shape != (rows,):
            raise RuntimeError(f"{split}: shapes {g.shape} / {s.shape} != ({rows},)")
        gc, sc = np.clip(g, -CLAMP_CP, CLAMP_CP), np.clip(s, -CLAMP_CP, CLAMP_CP)
        avg = ((gc + sc) / 2.0).astype(np.float32)
        if not np.isfinite(avg).all():
            raise RuntimeError(f"{split}: non-finite averaged label")
        np.save(p.labels / f"{split}_dlshogi_evals.npy", avg)
        meta["stats"][split] = {
            "rows": rows, "std_avg": float(avg.std()), "std_gikou_clip": float(gc.std()),
            "std_search_clip": float(sc.std()), "corr_gikou_search": float(np.corrcoef(gc, sc)[0, 1]),
            "corr_avg_gikou": float(np.corrcoef(avg, gc)[0, 1]), "corr_avg_search": float(np.corrcoef(avg, sc)[0, 1]),
            "abs_median_avg_vs_gikou": float(np.median(np.abs(avg - gc))),
            "abs_median_avg_vs_search": float(np.median(np.abs(avg - sc))),
        }
        print(f"T105_LABELS {split} " + " ".join(f"{k}={v:.4g}" for k, v in meta["stats"][split].items()), flush=True)
    (p.labels / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train_student(p, run_tag, env):
    recipe = [*NNUE_ARGS, "--augment-mirror", "--mirror-cache", str(p.data / MIRROR_CACHE)]
    if not (p.student / "nnue_best.pt").exists():
        print("=== T105 STAGE train averaged-label student ===", flush=True)
        run_logged([p.torch_py, "-m", "trainer.train_nnue", p.cache, "--out", p.student, *recipe,
                    "--eval-npy", p.labels, "--limit-train", LIMIT_TRAIN],
                   p.logs / f"{run_tag}_student_avg.log", cwd=p.repo, env=env)
    if not (p.student_eval / "nn.bin").exists():
        p.student_eval.mkdir(parents=True, exist_ok=True)
        run_logged([p.torch_py, "-m", "trainer.export_nnue", p.student, "--weights", "nnue_best.pt",
                    "--out", p.student / "nnue_best.bnnq"], p.logs / f"{run_tag}_export_avg.log", cwd=p.repo, env=env)
        run_logged([p.torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", p.student / "nnue_best.bnnq",
                    "--template", p.yane_eval / "nn.bin", "--out", p.student_eval / "nn.bin"],
                   p.logs / f"{run_tag}_yane_avg.log", cwd=p.repo, env=env)
    gate = p.logs / f"{run_tag}_0cp_avg.log"
    if not gate.exists():
        run_logged(["bash", "scripts/binbo_to_yane.sh", p.student / "nnue_best.bnnq"], gate, cwd=p.repo, env=env)
    text = gate.read_text(encoding="utf-8")
    ok = "RED" not in text and text.count("GREEN") >= 2
    hist = json.loads((p.student / "history.json").read_text(encoding="utf-8"))
    summary = {k: hist.get(k) for k in ("n_train", "n_val", "best_epoch", "best_val_loss", "epochs", "lr")}
    print(f"T105_STUDENT_READY avg best_epoch={summary['best_epoch']} gate={'GREEN' if ok else 'RED'}", flush=True)
    return ok, summary


def stage_main(p, run_tag):
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})
    failures = []
    print("=== T105 STAGE labels ===", flush=True)
    meta = build_labels(p)
    for split in ("train", "val"):
        st = meta["stats"][split]
        if min(st["abs_median_avg_vs_gikou"], st["abs_median_avg_vs_search"]) < LABEL_MOVED_MIN_CP:
            failures.append(f"{split}: averaged label too close to a parent")
    gate_ok, training = train_student(p, run_tag, env)
    if not gate_ok:
        failures.append("0cp gate RED for the averaged student")

    print("=== T105 STAGE wait for the yardstick ===", flush=True)
    while not p.yardstick_json.exists():
        time.sleep(30)
    yard = json.loads(p.yardstick_json.read_text(encoding="utf-8"))
    failures += [f"yardstick: {f}" for f in yard["failures"]]

    print("=== T105 STAGE primary match (averaged vs search student, 1600 games) ===", flush=True)
    tsvs = [p.logs / f"{run_tag}_primary_w{i}_seed{s}_n{PRIMARY_GAMES}.tsv" for i, s in enumerate(PRIMARY_SEEDS)]
    commands = [[p.binbo, "yane-vs-yane", p.yane, p.student_eval, p.eval_dirs / SEARCH, "100", s, PRIMARY_GAMES,
                 "6", "320", "0", "0", "0.05", "0.05"] for s in PRIMARY_SEEDS]
    run_workers(commands, tsvs, env, p.repo)
    rows, bad = combine_primary(tsvs)
    failures += bad
    cond_p = conditions(tsvs)
    primary = paired_h2h_delta(rows, label="averaged_student_minus_search_student")
    print(f"T105_PRIMARY W-D-L={primary['wins']}-{primary['draws']}-{primary['losses']} "
          f"score={primary['score']:.4f} elo={primary['elo']:+.2f} ci={[round(v, 2) for v in primary['ci95']]} "
          f"depth={cond_p['a']['avg_depth']:.2f}/{cond_p['b']['avg_depth']:.2f} "
          f"nodes={cond_p['a']['avg_nodes']:.0f}/{cond_p['b']['avg_nodes']:.0f}", flush=True)

    print("=== T105 STAGE frame (averaged student vs gikou@%d) ===" % ANCHOR_LEVEL, flush=True)
    ra, ca, fa, ta = anchor_series(p, p.student_eval, f"{run_tag}_avg_vs_gikou{ANCHOR_LEVEL}")
    failures += fa
    rs, _ = combine([Path(t) for t in yard["search"]["tsvs"]])
    if not same_openings(ra, rs):
        failures.append("frame: openings differ between the averaged and search arms")
    ha = paired_h2h_delta(ra, label=f"averaged_student_minus_gikou{ANCHOR_LEVEL}")
    frame = paired_series_delta(ra, rs, label=f"(averaged - search student) in the gikou@{ANCHOR_LEVEL} frame")
    print(f"T105_FRAME avg_vs_anchor elo={ha['elo']:+.2f} ci={[round(v, 2) for v in ha['ci95']]} "
          f"avg-search={frame['delta_elo']:+.2f} ci={[round(v, 2) for v in frame['ci95']]}", flush=True)

    flags = list(yard.get("flags", []))
    if abs(cond_p["a"]["avg_nodes"] / max(1.0, cond_p["b"]["avg_nodes"]) - 1) > NODES_TOL:
        flags.append("TIME_ASYMMETRY_PRIMARY")
    flags += nodes_flag(ca, yard["search"]["conditions"], "FRAME")
    low, high = primary["ci95"]
    if failures:
        verdict = "INVALID"
    elif low > 0 and frame["ci95"][0] > 0:
        verdict = "AVG_ADOPT"
    elif low > 0:
        verdict = "AVG_DIRECT_ONLY"
    elif high < 0:
        verdict = "SEARCH_HOLDS"
    else:
        verdict = "PARITY"
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=p.repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    p.results_json.write_text(json.dumps({
        "task": 105, "git": {"head": head, "pushed": False}, "labels": meta, "training": training,
        "primary": {"h2h": primary, "conditions": cond_p, "tsvs": [str(t) for t in tsvs]},
        "frame": {"avg_vs_anchor": ha, "conditions": ca, "delta_avg_minus_search": frame,
                  "tsvs": [str(t) for t in ta]},
        "yardstick": yard, "failures": failures, "flags": flags, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"T105_VERDICT verdict={verdict} flags={flags} failures={failures} results={p.results_json}", flush=True)
    return 0 if not failures else 1


def combine_primary(tsvs):
    """combine() for the 400-game primary workers (t103_absolute.combine assumes 200)."""
    from t098_pipeline import load_yane_tsv, validate_h2h
    rows, offset, bad = [], 0, []
    for tsv in tsvs:
        part = load_yane_tsv(tsv)
        bad += validate_h2h(part, tsv, expected_games=PRIMARY_GAMES)["bad_reasons"]
        for row in part:
            row = dict(row)
            row["game"] += offset
            rows.append(row)
        offset += len(part)
    validate_h2h(rows, "combined", expected_games=PRIMARY_GAMES * len(tsvs))
    return rows, bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["yardstick", "main"])
    parser.add_argument("--run-tag", default="task105")
    args = parser.parse_args()
    p = Paths(args.run_tag)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"], cwd=p.repo,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    for path in (p.binbo, p.yane, p.gikou, p.cache / "train_evals.npy", p.data / SEARCH_LABELS / "meta.json",
                 p.eval_dirs / CHAMPION / "nn.bin", p.eval_dirs / SEARCH / "nn.bin",
                 p.data / MIRROR_CACHE.replace(".npy", "_shared.npy")):  # train_nnue reads the _shared variant
        if not path.exists():
            raise FileNotFoundError(path)
    print(f"T105_PREFLIGHT_OK stage={args.stage} anchor=gikou@{ANCHOR_LEVEL}", flush=True)
    return stage_yardstick(p, args.run_tag) if args.stage == "yardstick" else stage_main(p, args.run_tag)


if __name__ == "__main__":
    sys.exit(main())
