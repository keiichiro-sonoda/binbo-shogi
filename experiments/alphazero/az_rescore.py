#!/usr/bin/env python3
"""Task 084: re-score a prepare_data cache with an AlphaZero net's static value.

The AZ analog of trainer/dlshogi_rescore.py: reads the EXACT cache positions in the
same order, decodes hcp -> sfen -> Pgx obs (side-to-move POV, flip=True) -> AZ net
value v in [-1,1], converts to student-scale cp and writes {train,val}_dlshogi_evals.npy
(the filename train_nnue --eval-npy expects), with a gikou-eval ALIGNMENT ASSERT
against the cache (the Task 055 silently-breakable bit).

  cp = 1200*atanh(v)   (v=tanh(cp/1200) is the M2 distill target; winrate=(v+1)/2,
                        so cp = 600*logit(winrate) = 1200*atanh(v)), clamped +/-5000.

The AZ value is itself distilled from Gikou@d8 (M2/M4), so this tests whether the
AZ-denoised value is a better/worse NNUE teacher than the raw Gikou@d8 search cp.

Runs in venv_pgx (jax + cshogi). Reuses m4_distill_data (sfen parse + obs) and
m2_usi.load_model (ckpt -> forward at the ckpt's channel width).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cshogi
import jax
import jax.numpy as jnp

from pgx._src.games.shogi import _observe
from m4_distill_data import parse_sfen_np, batch_state
from m2_usi import load_model

# import prepare_data.shard_files from the binbo trainer (pure numpy/os)
sys.path.insert(0, os.path.expanduser("~/GitHub/shogi/binbo-shogi"))
from trainer.prepare_data import shard_files  # noqa: E402

CLAMP_CP = 5000.0


def value_to_cp(v: np.ndarray, clamp: float = CLAMP_CP) -> np.ndarray:
    """AZ value v in [-1,1] (side-to-move) -> student-scale cp, clamped."""
    v = np.clip(v.astype(np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    cp = 1200.0 * np.arctanh(v)
    return np.clip(cp, -clamp, clamp)


def rescore_cache_aligned(cache_dir, ckpt, out_dir, batch, verify):
    with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    run_dir, extras = meta["run_dir"], meta["extra_runs"]
    val_count = len(meta["val_shards"])
    n_train, n_val = meta["n_train"], meta["n_val"]

    forward, model, info = load_model(ckpt)
    print(f"az ckpt={ckpt} ({info})")
    obs_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(True))))

    @jax.jit
    def value_of(obs):
        (_logits, v), _ = forward.apply(model[0], model[1], obs, is_eval=True)
        return v

    def az_value(sfens):
        colors, boards, hands = [], [], []
        for s in sfens:
            c, pb, hd = parse_sfen_np(s)
            colors.append(c); boards.append(pb); hands.append(hd)
        obs = obs_fn(batch_state(colors, boards, hands))
        return np.asarray(value_of(obs))

    os.makedirs(out_dir, exist_ok=True)
    primary, _ = shard_files(run_dir)
    groups = {
        "train": [(run_dir, primary[:-val_count])] + [(rd, shard_files(rd)[0]) for rd in extras],
        "val": [(run_dir, primary[-val_count:])],
    }
    board = cshogi.Board()
    started = time.time()
    for split, n in (("train", n_train), ("val", n_val)):
        az_cp = np.zeros(n, dtype=np.int16)
        gikou = np.zeros(n, dtype=np.int16)
        cur = 0
        for rd, shards in groups[split]:
            for sh in shards:
                ent = np.fromfile(os.path.join(rd, sh["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
                m = ent.shape[0]
                gikou[cur:cur + m] = ent["eval"].astype(np.int16)
                # decode hcp -> sfen, then AZ-score in sub-batches
                sfens = []
                for e in ent:
                    board.set_hcp(np.asarray(e["hcp"]))
                    sfens.append(board.sfen())
                for b in range(0, m, batch):
                    v = az_value(sfens[b:b + batch])
                    az_cp[cur + b:cur + b + len(v)] = value_to_cp(v).astype(np.int16)
                cur += m
                if cur % 500000 < m:
                    rate = cur / max(1e-9, time.time() - started)
                    print(f"  {split}: {cur}/{n} ({rate:.0f}/s)", flush=True)
        if cur != n:
            raise AssertionError(f"{split}: scored {cur} != cache n {n}")
        cache_eval = np.load(os.path.join(cache_dir, f"{split}_evals.npy"))
        if not np.array_equal(gikou, cache_eval):
            nmis = int(np.sum(gikou != cache_eval))
            raise AssertionError(f"{split} ALIGNMENT MISMATCH: {nmis}/{n} gikou evals differ from cache")
        np.save(os.path.join(out_dir, f"{split}_dlshogi_evals.npy"), az_cp)
        gclip = np.clip(cache_eval, -CLAMP_CP, CLAMP_CP).astype(np.float64)
        corr = float(np.corrcoef(az_cp.astype(np.float64), gclip)[0, 1])
        absdiff = np.abs(az_cp.astype(np.float64) - gclip)
        print(f"  {split}: ALIGNED ok (gikou==cache), az cp std={az_cp.std():.0f} "
              f"(gikou clip std={gclip.std():.0f}) corr_with_gikou={corr:.3f} "
              f"|Δ|median={np.median(absdiff):.0f} -> {out_dir}/{split}_dlshogi_evals.npy", flush=True)
    print(f"az_rescore=ok elapsed={time.time() - started:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="AZ ckpt (m6_ckpt/000100.ckpt or m6_ws2/ws2.ckpt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--verify", type=int, default=0)
    a = ap.parse_args()
    rescore_cache_aligned(a.cache_dir, a.ckpt, a.out_dir, a.batch, a.verify)


if __name__ == "__main__":
    main()
