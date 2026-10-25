"""Shared model helpers for Task 091's frozen-value, trainable-policy network."""

from __future__ import annotations

import hashlib

import jax
import numpy as np

from m2_distill_train import make_forward


KIND = "T091FrozenValuePolicy"


class FrozenValuePolicyForward:
    """AZ-compatible apply wrapper over independent policy and value towers."""

    def __init__(self, base_forward):
        self.base_forward = base_forward

    def apply(self, params, state, obs, is_eval=False):
        (logits, _policy_value), policy_state = self.base_forward.apply(
            params["policy"], state["policy"], obs, is_eval=is_eval
        )
        (_value_logits, value), value_state = self.base_forward.apply(
            params["value"], state["value"], obs, is_eval=True
        )
        return (logits, value), {"policy": policy_state, "value": value_state}


def make_base_forward(network_config: dict):
    return make_forward(
        network_config.get("num_channels", 64),
        network_config["num_layers"],
        network_config.get("resnet_v2", True),
        network_config.get("num_actions", 2187),
        arch=network_config.get("arch", "cnn"),
        d_model=network_config.get("d_model", 256),
        n_heads=network_config.get("n_heads", 8),
        ff_mult=network_config.get("ff_mult", 4),
    )


def make_dual_forward(network_config: dict):
    return FrozenValuePolicyForward(make_base_forward(network_config))


def clone_as_dual(base_model):
    params, state = base_model
    return (
        {"policy": jax.tree_util.tree_map(lambda x: x, params),
         "value": jax.tree_util.tree_map(lambda x: x, params)},
        {"policy": jax.tree_util.tree_map(lambda x: x, state),
         "value": jax.tree_util.tree_map(lambda x: x, state)},
    )


def value_tower(model):
    return model[0]["value"], model[1]["value"]


def policy_tower(model):
    return model[0]["policy"], model[1]["policy"]


def tree_digest(tree) -> str:
    digest = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest.update(str(treedef).encode("utf-8"))
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
