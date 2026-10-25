#!/usr/bin/env python3
"""Read-only Task150 diagnosis. Source scripts/env.sh; CPU forward only, <=4 threads.

Usage: "$VENV_PY" experiments/alphazero/t150_why_probe.py --out reports/task150_why_metrics.json
No training, searches, or writes beneath DATA_ROOT. Public caches' *train* arrays
contain Suisho positions; their val arrays are symlinks to our old 50k validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4" if key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") else "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cshogi
from cshogi import CSA
import numpy as np
import torch
from trainer import kp
from trainer.nnue import QATSharedNNUE, fold_ft_factor

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
MODELS = {"129": "task129_student_64m_qsrel_e12/nnue_best.pt",
          "142": "task142_finetune/nnue_final.pt",
          "A": "task150_A_lr5e-6/nnue_final.pt", "B": "task150_B_lr1e-5/nnue_final.pt",
          "C": "task150_C_8m_lr1e-5/nnue_final.pt"}


def summary(a):
    a = np.asarray(a, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                q10_50_90=np.percentile(a, [10, 50, 90]).tolist())


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def board_stats(b):
    pieces = np.asarray(b.pieces)
    # Descriptive material proxy only; never used in an eval or training target.
    values = np.array([0, 100, 300, 300, 400, 700, 800, 500, 0, 500, 500, 500, 500, 900, 1000])
    sign = 1 if b.turn == 0 else -1
    hands = np.asarray(b.pieces_in_hand)
    mat = (values[pieces & 15] * np.where(pieces < 16, 1, -1)).sum()
    mat += ((hands[0] - hands[1]) * [100, 300, 300, 400, 500, 700, 800]).sum()
    result = dict(material=sign * mat, abs_material=abs(mat), hand_total=hands.sum(),
                  hand_own=hands[b.turn].sum(), hand_opp=hands[1-b.turn].sum(),
                  promoted=np.isin(pieces & 15, [9, 10, 11, 12, 13, 14]).sum(),
                  in_check=int(b.is_check()), turn=b.turn)
    for color, label in ((b.turn, "own"), (1-b.turn, "opp")):
        king = int(np.flatnonzero(pieces == (8 | (color << 4)))[0])
        rank = king % 9 if color == 0 else 8-king % 9
        files = king // 9 if color == 0 else 8-king // 9
        result[label + "_king_rank"] = rank + 1
        result[label + "_king_file"] = files + 1
        d = np.maximum(np.abs(np.arange(81)//9-king//9), np.abs(np.arange(81)%9-king%9))
        enemy = (pieces != 0) & ((pieces >> 4) != color) & ((pieces & 15) != 8)
        friend = (pieces != 0) & ((pieces >> 4) == color) & ((pieces & 15) != 8)
        result[label + "_enemy_r2"] = (enemy & (d <= 2)).sum()
        result[label + "_friend_r2"] = (friend & (d <= 2)).sum()
    return result


def stats_rows(rows):
    records = [board_stats(cshogi.Board(kp.canonical_shared_row_to_sfen(r))) for r in rows]
    return {k: summary([r[k] for r in records]) for k in records[0]}


def forward(state, rows):
    model = QATSharedNNUE(fake_quant=True, sparse_ft=True, factorize=True,
                          tie_collapse_map=state["collapse_map"])
    model.load_state_dict(state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for lo in range(0, len(rows), 512):
            x = torch.tensor(rows[lo:lo+512], dtype=torch.long)
            n = len(x)
            p = model(x[:, :39].reshape(-1), torch.arange(n)*39,
                      x[:, 39:].reshape(-1), torch.arange(n)*38)*600
            predictions.append(p.numpy())
    return np.concatenate(predictions).astype(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = Path(os.environ["DATA_ROOT"]).resolve()
    assert not args.out.resolve().is_relative_to(data), "DATA_ROOT is read-only"
    rng = np.random.default_rng(20271501)
    out = dict(seed=20271501, numpy=np.__version__, torch=torch.__version__, threads=4,
               models={}, sets={}, mask={}, displacement={})
    sets = {}
    for name, cache, split, n, label in (
        ("own_train", "shared_64m_qs129", "train", 8192, "train"),
        ("own_val", "shared_64m_qs129", "val", 50000, "val"),
        ("suisho_finish", "public_suisho5_val7m", "train", 32768, None),
        ("suisho_train", "public_suisho5_train64m", "train", 8192, None)):
        root = data / "caches" / cache
        x = np.load(root / f"{split}_indices.npy", mmap_mode="r")
        idx = np.sort(rng.choice(len(x), n, replace=False))
        ypath = (data / "work/task146_full/labels" / f"{label}_dlshogi_evals.npy"
                 if label else root / "train_evals.npy")
        y = np.load(ypath, mmap_mode="r")[idx].astype(float)
        sets[name] = dict(rows=np.array(x[idx]), target=y, indices=idx)
        out["sets"][name] = dict(source=str(root), total=len(x), n=n,
                                  raw_target=summary(y), abs_target=summary(abs(y)),
                                  abs_ge_2000=float(np.mean(abs(y)>=2000)))
        if label is None:
            file = "shuffled.bin" if name == "suisho_finish" else "train_first64m.bin"
            psv = np.memmap(data / "public/nodchip_suisho5_depth9" / file,
                            dtype=cshogi.PackedSfenValue, mode="r")
            assert np.array_equal(psv["score"][idx], y)
            out["sets"][name]["gamePly"] = summary(psv["gamePly"][idx])
        print("loaded", name, flush=True)
    mask = np.load(data / "work/task150/mask_8000000.npy", mmap_mode="r")
    assert mask.dtype == bool and len(mask) == 63950000
    chosen = np.flatnonzero(mask)
    assert len(chosen) == 8000000
    out["mask"] = dict(count=len(chosen), block4m_counts=np.histogram(chosen, bins=np.r_[np.arange(0, 63950000, 4000000), 63950000])[0].tolist())
    sub = np.sort(rng.choice(chosen, 8192, replace=False))
    del chosen
    x = np.load(data / "caches/shared_64m_qs129/train_indices.npy", mmap_mode="r")
    y = np.load(data / "work/task146_full/labels/train_dlshogi_evals.npy", mmap_mode="r")[sub].astype(float)
    sets["C_train"] = dict(rows=np.array(x[sub]), target=y, indices=sub)
    out["sets"]["C_train"] = dict(n=len(sub), raw_target=summary(y), abs_ge_2000=float(np.mean(abs(y)>=2000)))
    # Verify the actual mask against its advertised RNG recipe, then compare ALL
    # selected labels rather than confusing two noisy 8192-row samples with bias.
    expected = np.zeros(len(mask), dtype=bool)
    expected[np.random.default_rng(20271500).choice(len(mask), 8000000, replace=False)] = True
    out["mask"]["exact_seed_recipe_match"] = bool(np.array_equal(mask, expected))
    assert out["mask"]["exact_seed_recipe_match"]
    del expected
    labels = np.load(data / "work/task146_full/labels/train_dlshogi_evals.npy", mmap_mode="r")
    accum = {k: np.zeros(5) for k in ("all", "C")}
    for lo in range(0, len(mask), 1000000):
        raw = labels[lo:lo+1000000].astype(float)
        for k, v in (("all", raw), ("C", raw[mask[lo:lo+1000000]])):
            vclip = np.clip(v, -2000, 2000)
            accum[k] += [len(v), vclip.sum(), (vclip*vclip).sum(), np.sum(abs(v)>=2000), np.sum(abs(v)>=30000)]
    for k, (n, total, sq, extreme, mateish) in accum.items():
        out["mask"][k+"_full_label_stats"] = dict(n=int(n), clamped_mean=total/n,
            clamped_std=float(np.sqrt(sq/n-(total/n)**2)), abs_ge2000=extreme/n, abs_ge30000=mateish/n)
    # Only task142 games for the main comparison; retain actual ply before each move.
    game_rows, plies, game_ids = [], [], []
    files = sorted((data / "denryusen/floodgate").glob("*/me/records/*.csa"))
    used = []
    for file in files:
        for g in CSA.Parser.parse_file(str(file)):
            if "binbo_t142" not in g.names:
                continue
            used.append(str(file.relative_to(data)))
            b = cshogi.Board(g.sfen)
            for move in g.moves:
                if g.names[b.turn] == "binbo_t142":
                    game_rows.append(kp.canonical_shared_row(b))
                    plies.append(b.move_number)
                    game_ids.append(len(used)-1)
                assert b.is_legal(move), file
                b.push(move)
    assert game_rows, "No task142 CSA games"
    sets["floodgate"] = dict(rows=np.asarray(game_rows, dtype=np.int32), target=None,
                              game_ids=np.asarray(game_ids))
    out["sets"]["floodgate"] = dict(n=len(game_rows), files=used, gamePly=summary(plies),
                                      per_game_rows=np.bincount(game_ids).tolist())
    for name, s in sets.items():
        # All 8192 for train sets, 8192 fixed random rows for larger sets.
        pick = np.sort(rng.choice(len(s["rows"]), min(8192, len(s["rows"])), replace=False))
        out["sets"][name]["board_stats"] = stats_rows(s["rows"][pick])
        print("stats", name, flush=True)
    reference = torch.load(data / "models" / MODELS["142"], map_location="cpu", weights_only=True)
    folded_ref = fold_ft_factor(reference)
    for key, rel in MODELS.items():
        path = data / "models" / rel
        state = torch.load(path, map_location="cpu", weights_only=True)
        out["models"][key] = dict(path=rel, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        disp = {}
        for layer, w in state.items():
            if layer == "collapse_map":
                continue
            delta = w - reference[layer]
            disp[layer] = dict(l2=float(torch.linalg.vector_norm(delta)), rms=float(delta.square().mean().sqrt()))
        folded = fold_ft_factor(state)
        cm = state["collapse_map"]
        delta = folded["feature_transform.weight"][cm] - folded_ref["feature_transform.weight"][cm]
        disp["effective_tied_ft"] = dict(l2=float(delta.norm()), rms=float(delta.square().mean().sqrt()))
        del delta, folded
        out["displacement"][key] = disp
        for name, s in sets.items():
            s[key] = forward(state, s["rows"])
            print("forward", key, name, flush=True)
        del state
    for name, s in sets.items():
        r = out["sets"][name]
        r["metrics"] = {}
        for key in MODELS:
            p = s[key]
            m = dict(pred=summary(p), abs_pred=summary(abs(p)), delta142=summary(p-s["142"]),
                     mae_from142=float(abs(p-s["142"]).mean()), mae_from129=float(abs(p-s["129"]).mean()))
            if s["target"] is not None:
                y = np.clip(s["target"], -2000, 2000)
                d = abs(p-y)
                m.update(mae=float(d.mean()), corr=corr(p, y), raw_corr=corr(p, s["target"]),
                         huber=float(np.where(d<600, .5*d*d, 600*(d-300)).mean()))
                near = abs(s["target"]) < 1000
                m["target_abs_lt1000_mae"] = float(d[near].mean())
                # Paired difference SE is descriptive; rows are not independent games.
                e = d-abs(s["142"]-y)
                m["mae_change142"] = float(e.mean())
                m["paired_row_se"] = float(e.std()/np.sqrt(len(e)))
                m["mean_residual"] = float((p-y).mean())
                # Diagnostic on this same sample, NOT a proposed/deployable bias fit.
                centered = p-(p-y).mean()
                m["mean_residual_removed_mae"] = float(abs(centered-y).mean())
                m["mae_by_abs_target"] = {}
                for lower, upper in ((0, 100), (100, 500), (500, 1000), (1000, 2001)):
                    sel = (abs(y) >= lower) & (abs(y) < upper)
                    m["mae_by_abs_target"][f"{lower}_{upper}"] = dict(n=int(sel.sum()),
                        mae=float(d[sel].mean()), delta142_mean=float((p-s["142"])[sel].mean()))
            if name == "floodgate":
                m["per_game_mae_delta142"] = [float(abs(p-s["142"])[s["game_ids"] == i].mean()) for i in range(len(used))]
            r["metrics"][key] = m
        if s["target"] is not None:
            teacher_delta = np.clip(s["target"], -2000, 2000)-s["142"]
            r["movement_corr_teacher_residual"] = {k:corr(s[k]-s["142"], teacher_delta) for k in ("A", "B", "C")}
        finish = s["142"]-s["129"]
        r["finish_projection"] = {k:float(np.dot(s[k]-s["129"], finish)/np.dot(finish, finish)) for k in ("A", "B", "C")}
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    start = time.monotonic()
    main()
    print("seconds", round(time.monotonic()-start, 1))
