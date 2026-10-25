# Task 080 M4 step 1: joint value+policy distillation (WS2).
#
# Same trunk/optimizer/schedule as m2_distill_train (adam lr1e-3, batch 4096,
# 4 epochs) — the value side doubles as a same-conditions replication of M2
# (expect val corr ~= 0.90). Added: policy head trained with softmax CE against
# the teacher bestmove one-hot (Pgx action id from m4_distill_data).
# Metrics: val value corr/MSE + policy top-1 (teacher-move agreement).
# Checkpoint saved in the m3_loop/m2_usi-compatible format as m4_ckpt/000000.ckpt
# (doubles as loop iteration 0 = WS2).

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from m2_distill_train import make_forward, unpack


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4_data"))
    # WS2 lives OUTSIDE the loop's ckpt_dir (m4_ckpt): m3_loop resume=auto globs
    # *.ckpt there and would mis-parse/mis-resume a distill ckpt.
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4_ws2"))
    ap.add_argument("--num-channels", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=5)  # cnn: resnet blocks; transformer: encoder layers
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    # Task 087 T1: arch-only swap (one-variable). transformer = TransformerNet trunk.
    ap.add_argument("--arch", default="cnn", choices=["cnn", "transformer"])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=300, help="transformer LR warmup steps (adam)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr_obs = np.load(os.path.join(args.data_dir, "train_obs.npy"), mmap_mode="r")
    tr_v = np.load(os.path.join(args.data_dir, "train_v.npy"))
    tr_a = np.load(os.path.join(args.data_dir, "train_a.npy")).astype(np.int32)
    va_obs = np.load(os.path.join(args.data_dir, "val_obs.npy"), mmap_mode="r")
    va_v = np.load(os.path.join(args.data_dir, "val_v.npy"))
    va_a = np.load(os.path.join(args.data_dir, "val_a.npy")).astype(np.int32)
    print(f"train {len(tr_v)}  val {len(va_v)}", flush=True)

    forward = make_forward(args.num_channels, args.num_layers, arch=args.arch,
                           d_model=args.d_model, n_heads=args.n_heads, ff_mult=args.ff_mult)
    if args.arch == "transformer":
        # Transformers diverge at a flat high LR from scratch -> linear warmup then constant
        # (the one arch-specific accommodation; peak LR kept == CNN for a fair comparison).
        sched = optax.join_schedules(
            [optax.linear_schedule(0.0, args.lr, args.warmup), optax.constant_schedule(args.lr)],
            [args.warmup])
        optimizer = optax.adam(sched)
        print(f"arch=transformer d_model={args.d_model} layers={args.num_layers} "
              f"heads={args.n_heads} ff_mult={args.ff_mult} warmup={args.warmup}", flush=True)
    else:
        optimizer = optax.adam(args.lr)
    model = forward.init(jax.random.PRNGKey(args.seed), jnp.zeros((2, 9, 9, 119), dtype=bool))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(model[0]))
    print(f"params={n_params} ({n_params/1e6:.3f}M)", flush=True)
    opt_state = optimizer.init(model[0])

    def loss_fn(params, state, obs, v_tgt, a_tgt):
        (logits, v), state = forward.apply(params, state, obs, is_eval=False)
        v_loss = jnp.mean((v - v_tgt) ** 2)
        p_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, a_tgt))
        return v_loss + p_loss, (state, v_loss, p_loss)

    @jax.jit
    def train_step(model, opt_state, obs, v_tgt, a_tgt):
        params, state = model
        (loss, (state, v_loss, p_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, obs, v_tgt, a_tgt
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return (optax.apply_updates(params, updates), state), opt_state, v_loss, p_loss

    @jax.jit
    def predict(model, obs):
        (logits, v), _ = forward.apply(model[0], model[1], obs, is_eval=True)
        return logits, v

    def evaluate(model, obs_arr, v_arr, a_arr):
        vs, top1 = [], 0
        n = len(v_arr)
        for i in range(0, n, args.batch):
            end = min(i + args.batch, n)  # obs_arr may be longer than the subset (train-subset case)
            logits, v = predict(model, jnp.asarray(unpack(obs_arr[i:end])))
            vs.append(np.asarray(v))
            top1 += int((np.asarray(jnp.argmax(logits, axis=-1)) == a_arr[i:end]).sum())
        p = np.concatenate(vs)
        return (float(np.mean((p - v_arr) ** 2)), float(np.corrcoef(p, v_arr)[0, 1]), top1 / n)

    # Task 089: fixed train subset (contiguous first 50k of the memmap) to measure the
    # train-val gap -> distinguishes data-starvation (big gap) from teacher-ceiling (small gap).
    TR_SUB = min(50_000, len(tr_v))
    tr_sub_v, tr_sub_a = tr_v[:TR_SUB], tr_a[:TR_SUB]

    mse0, corr0, t0m = evaluate(model, va_obs, va_v, va_a)
    print(f"epoch 0 (init): val_mse {mse0:.4f} val_corr {corr0:.4f} top1 {t0m:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)
    n = len(tr_v)
    best = (float("inf"), None)
    history = [dict(epoch=0, val_mse=mse0, val_corr=corr0, top1=t0m)]
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        vls, pls = [], []
        for i in range(0, n - args.batch + 1, args.batch):
            idx = np.sort(perm[i : i + args.batch])
            model, opt_state, vl, pl = train_step(
                model, opt_state,
                jnp.asarray(unpack(tr_obs[idx])), jnp.asarray(tr_v[idx]), jnp.asarray(tr_a[idx]),
            )
            vls.append(float(vl)); pls.append(float(pl))
        mse, corr, top1 = evaluate(model, va_obs, va_v, va_a)
        tr_mse, tr_corr, tr_top1 = evaluate(model, tr_obs, tr_sub_v, tr_sub_a)
        print(f"epoch {ep}: train v_mse {np.mean(vls):.4f} p_ce {np.mean(pls):.4f} | "
              f"val_corr {corr:.4f} top1 {top1:.4f} | train_corr {tr_corr:.4f} train_top1 {tr_top1:.4f} "
              f"gap_corr {tr_corr-corr:+.4f} gap_top1 {tr_top1-top1:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        history.append(dict(epoch=ep, train_v=float(np.mean(vls)), train_p=float(np.mean(pls)),
                            val_mse=mse, val_corr=corr, top1=top1,
                            train_corr=tr_corr, train_top1=tr_top1))
        score = mse - top1  # combined selection: low value error, high agreement
        if score < best[0]:
            best = (score, ep)
            with open(os.path.join(args.out, "ws2.ckpt"), "wb") as f:
                pickle.dump(dict(
                    model=jax.device_get(model),
                    opt_state=None,  # loop re-inits fresh adam from init_ckpt path
                    iteration=0, rng_key=jax.random.PRNGKey(0),
                    config=dict(num_channels=args.num_channels, num_layers=args.num_layers,
                                resnet_v2=True, num_actions=2187, arch=args.arch,
                                d_model=args.d_model, n_heads=args.n_heads, ff_mult=args.ff_mult),
                    epoch=ep, val_mse=mse, val_corr=corr, top1=top1,
                    kind="WS2 value+policy distill", data=os.path.abspath(args.data_dir),
                ), f)
    with open(os.path.join(args.out, "ws2_history.json"), "w") as f:
        json.dump(dict(best_epoch=best[1], history=history, args=vars(args)), f, indent=2)
    print(f"WS2_DONE best_epoch={best[1]}", flush=True)


if __name__ == "__main__":
    main()
