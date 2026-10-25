# Task 077 M2 step 1: value-only distillation of AZNet from teacher labels.
#
# Loss = MSE(value, tanh(cp/1200)) ONLY. The policy head parameters receive
# structurally zero gradients (they are downstream of the trunk and do not feed the
# value path), so "policy stays at random init" falls out of the loss shape — no
# optax.multi_transform/masked freezing needed. Note: BatchNorm STATISTICS in the
# policy head do still update (forward runs with is_training=True through both heads).
#
# Data: m2_data/{train,val}_obs.npy (uint8 packbits memmap, 1205 B/pos) +
#       {train,val}_v.npy (float32). Unpack per minibatch.

import argparse
import json
import os
import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from network import AZNet, TransformerNet

OBS_BITS = 9 * 9 * 119


def make_forward(num_channels, num_layers, resnet_v2=True, num_actions=2187,
                 arch="cnn", d_model=256, n_heads=8, ff_mult=4):
    """Build the (logits, v) forward. arch='cnn' -> AZNet (ResNet), 'transformer'
    -> TransformerNet. num_layers = ResNet blocks (cnn) or encoder layers (transformer).
    Both share the apply signature so the distill loop / search / bench are unchanged."""
    def forward_fn(x, is_eval=False):
        if arch == "transformer":
            net = TransformerNet(num_actions=num_actions, d_model=d_model,
                                 num_blocks=num_layers, n_heads=n_heads, ff_mult=ff_mult)
        else:
            net = AZNet(num_actions=num_actions, num_channels=num_channels,
                        num_blocks=num_layers, resnet_v2=resnet_v2)
        return net(x, is_training=not is_eval, test_local_stats=False)
    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def unpack(rows):
    b = np.unpackbits(rows, axis=1)[:, :OBS_BITS]
    return b.reshape(-1, 9, 9, 119).astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2_ckpt"))
    ap.add_argument("--num-channels", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr_obs = np.load(os.path.join(args.data_dir, "train_obs.npy"), mmap_mode="r")
    tr_v = np.load(os.path.join(args.data_dir, "train_v.npy"))
    va_obs = np.load(os.path.join(args.data_dir, "val_obs.npy"), mmap_mode="r")
    va_v = np.load(os.path.join(args.data_dir, "val_v.npy"))
    print(f"train {len(tr_v)}  val {len(va_v)}  target std {tr_v.std():.3f}")

    forward = make_forward(args.num_channels, args.num_layers)
    optimizer = optax.adam(args.lr)

    dummy = jnp.zeros((2, 9, 9, 119), dtype=bool)
    model = forward.init(jax.random.PRNGKey(args.seed), dummy)
    opt_state = optimizer.init(model[0])

    def loss_fn(params, state, obs, v_tgt):
        (_logits, v), state = forward.apply(params, state, obs, is_eval=False)
        return jnp.mean((v - v_tgt) ** 2), state

    @jax.jit
    def train_step(model, opt_state, obs, v_tgt):
        params, state = model
        (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, obs, v_tgt)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, state), opt_state, loss

    @jax.jit
    def predict(model, obs):
        params, state = model
        (_logits, v), _ = forward.apply(params, state, obs, is_eval=True)
        return v

    def evaluate(model):
        preds = []
        for i in range(0, len(va_v), args.batch):
            preds.append(np.asarray(predict(model, jnp.asarray(unpack(va_obs[i : i + args.batch])))))
        p = np.concatenate(preds)
        mse = float(np.mean((p - va_v) ** 2))
        corr = float(np.corrcoef(p, va_v)[0, 1])
        return mse, corr

    # baseline: untrained net
    mse0, corr0 = evaluate(model)
    print(f"epoch 0 (init): val_mse {mse0:.4f} val_corr {corr0:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)
    n = len(tr_v)
    best = (float("inf"), None)
    history = [dict(epoch=0, val_mse=mse0, val_corr=corr0)]
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n - args.batch + 1, args.batch):
            idx = np.sort(perm[i : i + args.batch])  # sorted memmap reads
            obs = jnp.asarray(unpack(tr_obs[idx]))
            v_tgt = jnp.asarray(tr_v[idx])
            model, opt_state, loss = train_step(model, opt_state, obs, v_tgt)
            losses.append(float(loss))
        mse, corr = evaluate(model)
        dt = time.time() - t0
        print(f"epoch {ep}: train_mse {np.mean(losses):.4f} val_mse {mse:.4f} val_corr {corr:.4f} ({dt:.0f}s)", flush=True)
        history.append(dict(epoch=ep, train_mse=float(np.mean(losses)), val_mse=mse, val_corr=corr))
        if mse < best[0]:
            best = (mse, ep)
            with open(os.path.join(args.out, "value_distill_best.ckpt"), "wb") as f:
                pickle.dump(dict(
                    model=jax.device_get(model),
                    config=dict(num_channels=args.num_channels, num_layers=args.num_layers,
                                resnet_v2=True, num_actions=2187),
                    epoch=ep, val_mse=mse, val_corr=corr,
                    target="tanh(cp/1200)", data=os.path.abspath(args.data_dir),
                ), f)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(dict(best_epoch=best[1], history=history, args=vars(args)), f, indent=2)
    print(f"DISTILL_DONE best_epoch={best[1]} val_mse={best[0]:.4f}")


if __name__ == "__main__":
    main()
