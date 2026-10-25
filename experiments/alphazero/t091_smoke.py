#!/usr/bin/env python3
"""Fast invariance/round-trip smoke for Task 091's dual model."""

import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from m2_usi import load_model
from t091_model import KIND, clone_as_dual, make_base_forward, make_dual_forward, tree_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="t087_wst/ws2.ckpt")
    parser.add_argument("--data-dir", default="m4_data")
    parser.add_argument("--out", default="/tmp/binbo-task091-smoke.ckpt")
    args = parser.parse_args()
    with open(args.ckpt, "rb") as f:
        base_ckpt = pickle.load(f)
    network_config = base_ckpt["config"]
    base_forward = make_base_forward(network_config)
    dual_forward = make_dual_forward(network_config)
    model = clone_as_dual(base_ckpt["model"])
    obs_mm = np.load(os.path.join(args.data_dir, "val_obs.npy"), mmap_mode="r")
    rows = np.unpackbits(obs_mm[:4], axis=1)[:, :9 * 9 * 119]
    obs = jnp.asarray(rows.reshape(-1, 9, 9, 119).astype(bool))
    actions = jnp.asarray(np.load(os.path.join(args.data_dir, "val_a.npy"))[:4], dtype=jnp.int32)

    (base_logits, base_value), _ = base_forward.apply(
        base_ckpt["model"][0], base_ckpt["model"][1], obs, is_eval=True
    )
    (dual_logits, dual_value), _ = dual_forward.apply(model[0], model[1], obs, is_eval=True)
    np.testing.assert_array_equal(np.asarray(base_logits), np.asarray(dual_logits))
    np.testing.assert_array_equal(np.asarray(base_value), np.asarray(dual_value))
    frozen_digest = tree_digest((model[0]["value"], model[1]["value"]))

    def loss(policy_params, policy_state):
        (logits, _), policy_state = base_forward.apply(
            policy_params, policy_state, obs, is_eval=False
        )
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, actions)), policy_state

    (loss_value, policy_state), grads = jax.value_and_grad(loss, has_aux=True)(
        model[0]["policy"], model[1]["policy"]
    )
    optimizer = optax.adam(1e-4)
    updates, _ = optimizer.update(grads, optimizer.init(model[0]["policy"]))
    policy_params = optax.apply_updates(model[0]["policy"], updates)
    updated = (
        {"policy": policy_params, "value": model[0]["value"]},
        {"policy": policy_state, "value": model[1]["value"]},
    )
    (updated_logits, updated_value), _ = dual_forward.apply(
        updated[0], updated[1], obs, is_eval=True
    )
    if np.array_equal(np.asarray(updated_logits), np.asarray(base_logits)):
        raise AssertionError("policy logits did not change after one update")
    np.testing.assert_array_equal(np.asarray(updated_value), np.asarray(base_value))
    if tree_digest((updated[0]["value"], updated[1]["value"])) != frozen_digest:
        raise AssertionError("value tower changed")

    with open(args.out, "wb") as f:
        pickle.dump(dict(
            kind=KIND, model=jax.device_get(updated), network_config=network_config,
            config={}, iteration=1, metrics={"value_corr_teacher": None},
        ), f)
    loaded_forward, loaded_model, info = load_model(args.out)
    (loaded_logits, loaded_value), _ = loaded_forward.apply(
        loaded_model[0], loaded_model[1], obs, is_eval=True
    )
    np.testing.assert_array_equal(np.asarray(loaded_logits), np.asarray(updated_logits))
    np.testing.assert_array_equal(np.asarray(loaded_value), np.asarray(updated_value))
    print(
        f"T091_SMOKE_OK loss={float(loss_value):.6f} "
        f"policy_max_delta={float(jnp.max(jnp.abs(updated_logits-base_logits))):.6g} "
        f"value_max_delta={float(jnp.max(jnp.abs(updated_value-base_value))):.6g} "
        f"digest={frozen_digest} info={info}",
        flush=True,
    )


if __name__ == "__main__":
    main()
