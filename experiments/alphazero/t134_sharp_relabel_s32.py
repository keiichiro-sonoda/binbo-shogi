#!/usr/bin/env python3
"""Task 134: relabel only the "sharp" positions (many enemy pieces near a king) with a deeper teacher search.

Motivation (Task 133, reports/task133_king_safety_probe.md): against the two yardsticks, the adopted
eval is too optimistic for the side to move when pieces crowd around the kings; for the attack on
the opponent's king (+27 cp per piece) the bias is in the teacher's labels themselves (WS-T + Gumbel
MCTS S=8: +57 cp per piece) and a deeper search halves it (S=32: +31). Task 103b found S=8 and S=32
indistinguishable on average over all positions; here only the positions where the two differ are
targeted.

One variable: the labels of the selected train rows (WS-T S=8 -> WS-T S=32, same network
`t087_wst/ws2.ckpt`, same value->cp mapping). Everything else is the adopted eval's recipe
(Task 129: caches/shared_64m_qs129, 63.95M rows, no mask, lr 6e-5, Factorizer, mirror, 12 epochs;
the constants are imported from t129_qs_relabel_64m so they cannot drift).

Selection (fixed before the run): train rows where either king has >= 3 enemy pieces within
Chebyshev distance 2 (computed straight from the cache's canonical-shared KP indices; checked
against the decoded board on 2,000 rows = 0 mismatches) = 6,687,765 rows (10.46% of train).
Val is not touched (val labels = the adopted eval's).

Decision (fixed before the run, docs/evaluation_protocol.md): standard frame (tanuki- @20k,
AobaNNUE @10k, 100k nodes, seeds 20272001-08, 800 games per anchor), paired delta
(candidate - adopted, pooled over both anchors):
  CI low > 0  -> confirm on fresh seeds 20272601-08 (prefix conf134); confirm CI low > 0 -> ADOPT_CANDIDATE,
                 else NOT_CONFIRMED
  CI high < 0 -> SHARP32_WORSE
  otherwise   -> SHARP32_NEUTRAL
Build gates: corr(new S=32 label, old S=8 label) on the selected rows > 0.5 (same perspective, same
network); 0cp export gate GREEN; failures -> INVALID.

  t134_sharp_relabel_s32.py              (driver; runs the GPU stage in venv_pgx itself)
  t134_sharp_relabel_s32.py --relabel-only   (internal)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
from trainer import kp  # noqa: E402

TAG = "task134"
ADOPTED = "task129_student_64m_qsrel_e12"
CAND = f"{TAG}_student_sharp32"
THRESH = 3
SIMS, BATCH = 32, 128
PART_ROWS = 200_000
CORR_MIN = 0.5
CONFIRM_SEEDS = tuple(20272601 + i for i in range(8))
CONFIRM_PREFIX = "conf134"
OLD_LABELS = "work/task129/labels"


def att2_from_half(h):
    """(n, k) canonical-shared indices of one half -> enemy pieces within Chebyshev 2 of that half's king."""
    nt = len(kp.BOARD_PIECE_TYPES)
    body = h < kp.VIEW_DIM
    feat = h % kp.FEATURES_PER_KING
    isboard = body & (feat < kp.BOARD_FEATURES_PER_KING)
    rel = (feat // 81) // nt
    psq = feat % 81
    k0 = (h[:, 0:1] // kp.FEATURES_PER_KING)  # column 0 is always a body feature (sorted, side feature last)
    kf, kr = np.divmod(k0, 9)
    pf, pr = np.divmod(psq, 9)
    near = np.maximum(np.abs(pf - kf), np.abs(pr - kr)) <= 2
    return (isboard & (rel == 1) & near).sum(1)


def select(data, work):
    out = work / "rows.npy"
    if out.exists():
        return np.load(out)
    idx = np.load(data / t129.QS64 / "train_indices.npy", mmap_mode="r")
    a = np.empty(len(idx), np.int8)
    b = np.empty(len(idx), np.int8)
    for s in range(0, len(idx), 2_000_000):
        chunk = np.asarray(idx[s:s + 2_000_000])
        a[s:s + len(chunk)] = att2_from_half(chunk[:, :39])
        b[s:s + len(chunk)] = att2_from_half(chunk[:, 39:])
    rows = np.flatnonzero(np.maximum(a, b) >= THRESH).astype(np.int64)
    np.save(work / "att2_stm.npy", a)
    np.save(work / "att2_opp.npy", b)
    np.save(out, rows)
    return rows


def relabel(data, work):
    """GPU stage (venv_pgx): S=32 labels for the selected rows, in resumable parts."""
    from m2_usi import load_model
    from t099_search_label import build_batched_search, score_sfens, sign_oracle, value_to_cp

    rows = np.load(work / "rows.npy")
    idx = np.load(data / t129.QS64 / "train_indices.npy", mmap_mode="r")
    forward, model, _ = load_model(str(t129.WST))
    sign_oracle(forward, model, SIMS)
    search = build_batched_search(forward, model, SIMS)
    part_dir = work / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    for k, lo in enumerate(range(0, len(rows), PART_ROWS)):
        part = part_dir / f"part_{k:03d}.npz"
        if part.exists():
            continue
        r = rows[lo:lo + PART_ROWS]
        sfens = [kp.canonical_shared_row_to_sfen(tuple(int(x) for x in idx[i])) for i in r]
        values, _q, _raw, _mv = score_sfens(search, sfens, BATCH)
        tmp = part.with_suffix(".tmp.npz")
        np.savez(tmp, rows=r, cp=value_to_cp(values).astype(np.int16))
        os.replace(tmp, part)
        done += len(r)
        rate = done / (time.time() - started)
        print(f"T134_RELABEL part={k} rows={lo + len(r)}/{len(rows)} rate={rate:.0f}/s "
              f"eta={(len(rows) - lo - len(r)) / rate / 3600:.2f}h", flush=True)
    return 0


def build(data, work):
    lab = work / "labels"
    if (lab / "meta.json").exists():
        return json.loads((lab / "meta.json").read_text(encoding="utf-8"))["stats"]
    rows = np.load(work / "rows.npy")
    old = np.load(data / OLD_LABELS / "train_dlshogi_evals.npy")
    new = old.copy()
    got = np.zeros(len(old), bool)
    for part in sorted((work / "parts").glob("part_*.npz")):
        z = np.load(part)
        new[z["rows"]] = z["cp"]
        got[z["rows"]] = True
    missing = int(len(rows) - got[rows].sum())
    sel_old, sel_new = old[rows].astype(np.float64), new[rows].astype(np.float64)
    stats = {"rows": int(len(old)), "selected": int(len(rows)), "selected_frac": len(rows) / len(old),
             "missing": missing, "corr_new_old": float(np.corrcoef(sel_old, sel_new)[0, 1]),
             "median_abs_shift": float(np.median(np.abs(sel_new - sel_old))),
             "mean_shift": float(np.mean(sel_new - sel_old)), "thresh": THRESH, "sims": SIMS}
    tmp = work / "labels.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "train_dlshogi_evals.npy", new)
    val = (data / OLD_LABELS / "val_dlshogi_evals.npy").resolve()
    if (tmp / "val_dlshogi_evals.npy").exists() or (tmp / "val_dlshogi_evals.npy").is_symlink():
        (tmp / "val_dlshogi_evals.npy").unlink()
    os.symlink(val, tmp / "val_dlshogi_evals.npy")
    (tmp / "meta.json").write_text(json.dumps({"source": OLD_LABELS, "labeller": f"WS-T S={SIMS} on rows with "
                                               f"max(att2) >= {THRESH}, else {OLD_LABELS}", "stats": stats},
                                              indent=2), encoding="utf-8")
    os.replace(tmp, lab)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel-only", action="store_true")
    args = parser.parse_args()
    repo, data, logs, _binbo, _yane, _anchors = std_frame.paths()
    work = data / "work" / TAG
    work.mkdir(parents=True, exist_ok=True)
    if args.relabel_only:
        return relabel(data, work)

    torch_py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    for path in (data / t129.QS64 / "meta.json", data / OLD_LABELS / "meta.json", data / "eval_dirs" / ADOPTED / "nn.bin",
                 t129.WST, t129.PGX_PY, data / t129.MIRROR_CACHE.replace(".npy", "_shared.npy")):
        if not path.exists():
            raise FileNotFoundError(path)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = dict(os.environ, PYTHONPATH=str(repo))
    print(f"T134_PREFLIGHT_OK head={head} adopted={ADOPTED} thresh={THRESH} sims={SIMS}", flush=True)
    failures = []

    rows = select(data, work)
    print(f"T134_SELECT rows={len(rows)} frac={len(rows) / t129.N64:.4f}", flush=True)
    if not (work / "labels/meta.json").exists():
        n_parts = (len(rows) + PART_ROWS - 1) // PART_ROWS
        if len(list((work / "parts").glob("part_*.npz"))) < n_parts:
            print(f"=== T134 STAGE relabel (S={SIMS}) ===", flush=True)
            t106.run([t129.PGX_PY, os.path.realpath(__file__), "--relabel-only"], logs / f"{TAG}_relabel.log",
                     cwd=Path(__file__).parent, env=env)
    stats = build(data, work)
    print("T134_BUILD " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    if stats["missing"]:
        failures.append(f"missing relabelled rows {stats['missing']}")
    if not stats["corr_new_old"] > CORR_MIN:
        failures.append(f"corr(new, old) {stats['corr_new_old']:.3f} <= {CORR_MIN}")
    if failures:
        raise RuntimeError(f"build gates failed: {failures}")

    model_dir, eval_dir = data / "models" / CAND, data / "eval_dirs" / CAND
    if not (model_dir / "history.json").exists():
        print(f"=== T134 STAGE train (Task 129 recipe, labels {TAG}) ===", flush=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        stop_args = ["--stop-file", t106.STOP] if t106.STOP else []
        t106.run([torch_py, "-m", "trainer.train_nnue", data / t129.QS64, "--out", model_dir, *t129.LR_ARGS,
                  "--factorize", "--epochs", t129.EPOCHS, "--augment-mirror", "--mirror-cache", data / t129.MIRROR_CACHE,
                  "--eval-npy", work / "labels", "--limit-train", t129.N64, "--mmap",
                  "--checkpoint-minutes", "10", "--resume", *stop_args],
                 logs / f"{TAG}_student.log", cwd=repo, env=env)
    if not (eval_dir / "nn.bin").exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        t106.run([torch_py, "-m", "trainer.export_nnue", model_dir, "--weights", "nnue_best.pt",
                  "--out", model_dir / "nnue_best.bnnq"], logs / f"{TAG}_export.log", cwd=repo, env=env)
        t106.run([torch_py, "-m", "trainer.binbo_to_yane", "--from-bnnq", model_dir / "nnue_best.bnnq",
                  "--template", data / "engines/yaneuraou/eval/nn.bin", "--out", eval_dir / "nn.bin"],
                 logs / f"{TAG}_yane.log", cwd=repo, env=env)
    gate = logs / f"{TAG}_0cp.log"
    if not gate.exists():
        t106.run(["bash", "scripts/binbo_to_yane.sh", model_dir / "nnue_best.bnnq"], gate, cwd=repo, env=env)
    text = gate.read_text(encoding="utf-8")
    if "RED" in text or text.count("GREEN") < 2:
        failures.append("0cp gate RED")
    hist = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    ad_hist = json.loads((data / "models" / ADOPTED / "history.json").read_text(encoding="utf-8"))
    print(f"T134_TRAINED best_epoch={hist['best_epoch']} best_val_loss={hist['best_val_loss']:.1f} "
          f"adopted={ad_hist['best_val_loss']:.1f}", flush=True)

    def compare(cand, base, label):
        pooled, per, bad = std_frame.delta(cand, base, label)
        failures.extend(bad)
        print(f"T134_DELTA {label} pooled={pooled['delta_elo']:+.2f} ci={[round(v, 2) for v in pooled['ci95']]} "
              + " ".join(f"{n}={d['delta_elo']:+.1f}" for n, d in per.items()), flush=True)
        return {"pooled": pooled, "per_anchor": per}

    def frame(name, seeds=std_frame.SEEDS, prefix="std"):
        res, bad = std_frame.measure(name, env, seeds=seeds, prefix=prefix)
        failures.extend(bad)
        return res

    print("=== T134 STAGE standard frame ===", flush=True)
    primary = compare(frame(CAND), frame(ADOPTED), "(sharp S=32 relabel - adopted)")
    lo, hi = primary["pooled"]["ci95"]
    confirm = None
    if failures:
        verdict = "INVALID"
    elif lo > 0:
        print(f"=== T134 STAGE confirm on fresh seeds {CONFIRM_SEEDS[0]}-{CONFIRM_SEEDS[-1]} ===", flush=True)
        confirm = compare(frame(CAND, CONFIRM_SEEDS, CONFIRM_PREFIX), frame(ADOPTED, CONFIRM_SEEDS, CONFIRM_PREFIX),
                          "confirm (sharp S=32 relabel - adopted)")
        verdict = ("INVALID" if failures else "ADOPT_CANDIDATE" if confirm["pooled"]["ci95"][0] > 0
                   else "NOT_CONFIRMED")
    elif hi < 0:
        verdict = "SHARP32_WORSE"
    else:
        verdict = "SHARP32_NEUTRAL"

    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 134, "git": {"head": head}, "build": stats, "anchors": std_frame.ANCHORS,
                               "training": {k: hist.get(k) for k in ("best_epoch", "best_val_loss", "epochs")},
                               "adopted_best_val_loss": ad_hist.get("best_val_loss"),
                               "primary": primary, "confirm": confirm, "failures": failures, "verdict": verdict},
                              indent=2), encoding="utf-8")
    print(f"T134_VERDICT verdict={verdict} failures={failures} results={out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
