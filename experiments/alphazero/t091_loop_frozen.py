"""Task 091: Gumbel-AZ loop with an exactly frozen WS-T value tower.

The live model has two independent copies of WS-T. MCTS reads logits from the
trainable policy tower and values from the immutable value tower. Only policy CE
against MCTS visit weights is optimized; game outcomes are intentionally not a
training target in this experiment.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel, ConfigDict

from t091_model import KIND, clone_as_dual, make_base_forward, make_dual_forward, tree_digest


devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: pgx.EnvId = "shogi"
    seed: int = 20260714
    max_num_iters: int = 100
    selfplay_batch_size: int = 256
    states_per_iter: int = 131_072
    num_simulations: int = 32
    max_num_steps: int = 512
    training_batch_size: int = 256
    learning_rate: float = 0.0001
    init_ckpt: str = "t087_wst/ws2.ckpt"
    resume: str = "auto"
    ckpt_dir: str = "t091_ckpt"
    ckpt_interval: int = 5
    monitor_interval: int = 5
    expected_iter_s: float = 500.0
    teacher_val_dir: str = "m4_data"
    monitor_batch_size: int = 256

config = Config(**OmegaConf.from_cli())
print(config, flush=True)
if config.selfplay_batch_size % num_devices != 0:
    raise ValueError("selfplay_batch_size must be divisible by number of devices")
if config.training_batch_size % num_devices != 0:
    raise ValueError("training_batch_size must be divisible by number of devices")
frames_per_round = config.selfplay_batch_size * config.max_num_steps
if config.states_per_iter % frames_per_round != 0:
    raise ValueError(
        f"states_per_iter={config.states_per_iter} must be divisible by "
        f"selfplay_batch_size*max_num_steps={frames_per_round}"
    )
selfplay_rounds = config.states_per_iter // frames_per_round

env = pgx.make(config.env_id)

with open(config.init_ckpt, "rb") as f:
    _init_ckpt = pickle.load(f)
network_config = _init_ckpt["config"]
if network_config.get("arch") != "transformer":
    raise ValueError(f"Task 091 requires Transformer WS-T, got {network_config}")
base_forward = make_base_forward(network_config)
dual_forward = make_dual_forward(network_config)
optimizer = optax.adam(config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key
    params, model_state = model
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    (logits, value), _ = dual_forward.apply(params, model_state, state.observation, is_eval=True)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = jnp.where(state.terminated, 0.0, -jnp.ones_like(value))
    return mctx.RecurrentFnOutput(
        reward=reward, discount=discount, prior_logits=logits, value=value
    ), state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key):
        key1, key2 = jax.random.split(key)
        observation = state.observation
        (logits, value), _ = dual_forward.apply(
            params, model_state, observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        return state, SelfplayOutput(
            obs=observation,
            terminated=state.terminated,
            action_weights=policy_output.action_weights,
        )

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)
    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    return Sample(obs=data.obs, policy_tgt=data.action_weights)


def loss_fn(policy_params, policy_state, samples: Sample):
    (logits, _unused_value), policy_state = base_forward.apply(
        policy_params, policy_state, samples.obs, is_eval=False
    )
    cross_entropy = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(cross_entropy)
    target_entropy = -jnp.mean(
        jnp.sum(samples.policy_tgt * jnp.log(jnp.maximum(samples.policy_tgt, 1e-12)), axis=-1)
    )
    visit_agree = jnp.mean(
        jnp.argmax(logits, axis=-1) == jnp.argmax(samples.policy_tgt, axis=-1)
    )
    return policy_loss, (policy_state, target_entropy, visit_agree)


@partial(jax.pmap, axis_name="i")
def train(policy_model, opt_state, data: Sample):
    params, model_state = policy_model
    (policy_loss, (policy_state, target_entropy, visit_agree)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params, model_state, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    policy_params = optax.apply_updates(params, updates)
    return (policy_params, policy_state), opt_state, policy_loss, target_entropy, visit_agree


_teacher_val = None
_base_values = None


def value_to_cp(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.clip(1200.0 * np.arctanh(values), -5000.0, 5000.0)


def val_metrics(model_0):
    global _teacher_val, _base_values
    if _teacher_val is None:
        obs = np.load(os.path.join(config.teacher_val_dir, "val_obs.npy"), mmap_mode="r")
        values = np.load(os.path.join(config.teacher_val_dir, "val_v.npy"))
        actions = np.load(os.path.join(config.teacher_val_dir, "val_a.npy")).astype(np.int32)
        _teacher_val = (obs, values, actions)
    obs_mm, values, actions = _teacher_val

    @jax.jit
    def predict_dual(model, obs):
        (logits, value), _ = dual_forward.apply(model[0], model[1], obs, is_eval=True)
        return jnp.argmax(logits, axis=-1), value

    preds = []
    hits = 0
    bs = config.monitor_batch_size
    for start in range(0, len(values), bs):
        rows = np.unpackbits(obs_mm[start:start + bs], axis=1)[:, : 9 * 9 * 119]
        obs = jnp.asarray(rows.reshape(-1, 9, 9, 119).astype(bool))
        argmax, pred = predict_dual(model_0, obs)
        preds.append(np.asarray(pred))
        hits += int((np.asarray(argmax) == actions[start:start + len(pred)]).sum())
    pred = np.concatenate(preds)[:len(values)]
    if _base_values is None:
        _base_values = pred.copy()
    max_abs = float(np.max(np.abs(pred - _base_values)))
    corr = float(np.corrcoef(pred, values)[0, 1])
    return {
        "value_corr_teacher": corr,
        "value_cp_std": float(value_to_cp(pred).std()),
        "value_max_abs_wst": max_abs,
        "top1_teacher": hits / len(values),
    }


def latest_ckpt(ckpt_dir):
    candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(re.search(r"(\d+)\.ckpt$", path).group(1)))


def checkpoint_payload(model_0, opt_state_0, iteration, rng_key, frozen_digest, metrics):
    return dict(
        kind=KIND,
        model=jax.device_get(model_0),
        opt_state=jax.device_get(opt_state_0),
        iteration=iteration,
        rng_key=rng_key,
        config=config.model_dump(),
        network_config=network_config,
        frozen_value_digest=frozen_digest,
        metrics=metrics,
    )


def main():
    os.makedirs(config.ckpt_dir, exist_ok=True)
    iteration = 0
    rng_key = jax.random.PRNGKey(config.seed)
    resume_path = latest_ckpt(config.ckpt_dir) if config.resume == "auto" else None

    if resume_path:
        with open(resume_path, "rb") as f:
            checkpoint = pickle.load(f)
        if checkpoint.get("kind") != KIND:
            raise ValueError(f"unexpected checkpoint kind: {checkpoint.get('kind')}")
        model_0 = checkpoint["model"]
        opt_state_0 = checkpoint["opt_state"]
        iteration = checkpoint["iteration"]
        rng_key = checkpoint["rng_key"]
        frozen_digest = checkpoint["frozen_value_digest"]
        actual_digest = tree_digest((model_0[0]["value"], model_0[1]["value"]))
        if actual_digest != frozen_digest:
            raise AssertionError("frozen value digest mismatch on resume")
        print(f"t091: resumed from {resume_path} at iteration {iteration}", flush=True)
    else:
        model_0 = clone_as_dual(_init_ckpt["model"])
        frozen_digest = tree_digest((model_0[0]["value"], model_0[1]["value"]))
        opt_state_0 = optimizer.init(model_0[0]["policy"])
        metrics = val_metrics(model_0)
        if metrics["value_max_abs_wst"] > 1e-7:
            raise AssertionError(f"value differs at iter0: {metrics}")
        path = os.path.join(config.ckpt_dir, "000000.ckpt")
        with open(path, "wb") as f:
            pickle.dump(checkpoint_payload(
                model_0, opt_state_0, 0, rng_key, frozen_digest, metrics
            ), f)
        print(f"t091: initialized dual tower from {config.init_ckpt}", flush=True)
        print(f"t091 monitor iter=0 {metrics} frozen_digest={frozen_digest}", flush=True)

    model, opt_state = jax.tree_util.tree_map(
        lambda value: jnp.stack([value] * num_devices), (model_0, opt_state_0)
    )
    frames = iteration * config.states_per_iter
    print(
        f"t091: sims={config.num_simulations} selfplay_batch={config.selfplay_batch_size} "
        f"rounds={selfplay_rounds} states_per_iter={config.states_per_iter} "
        f"training_batch={config.training_batch_size} expected_iter_s={config.expected_iter_s}",
        flush=True,
    )

    while iteration < config.max_num_iters:
        iteration += 1
        started = time.time()
        selfplay_seconds = 0.0
        episodes = 0
        policy_losses = []
        target_entropies = []
        visit_agreements = []

        for _round in range(selfplay_rounds):
            round_started = time.time()
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            data = jax.block_until_ready(selfplay(model, keys))
            selfplay_seconds += time.time() - round_started
            episodes += int(jax.device_get(data.terminated).sum())
            samples = jax.device_get(compute_loss_input(data))
            del data
            samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
            rng_key, subkey = jax.random.split(rng_key)
            order = np.asarray(jax.device_get(
                jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
            ))
            samples = jax.tree_util.tree_map(lambda x: x[order], samples)
            num_updates = samples.obs.shape[0] // config.training_batch_size
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
            )
            for update in range(num_updates):
                batch = jax.tree_util.tree_map(lambda x: x[update], minibatches)
                policy_model = (model[0]["policy"], model[1]["policy"])
                policy_model, opt_state, loss, entropy, agree = train(
                    policy_model, opt_state, batch
                )
                model = (
                    {"policy": policy_model[0], "value": model[0]["value"]},
                    {"policy": policy_model[1], "value": model[1]["value"]},
                )
                policy_losses.append(float(loss.mean()))
                target_entropies.append(float(entropy.mean()))
                visit_agreements.append(float(agree.mean()))

        frames += config.states_per_iter
        metrics = None
        if config.monitor_interval > 0 and iteration % config.monitor_interval == 0:
            current_model = jax.tree_util.tree_map(lambda x: x[0], model)
            metrics = val_metrics(current_model)
            digest = tree_digest((current_model[0]["value"], current_model[1]["value"]))
            if digest != frozen_digest:
                raise AssertionError(f"frozen value params changed at iter {iteration}")
            if metrics["value_max_abs_wst"] > 1e-7:
                raise AssertionError(f"frozen value output changed at iter {iteration}: {metrics}")

        elapsed = time.time() - started
        visit_kl = float(np.mean(policy_losses) - np.mean(target_entropies))
        metric_text = "" if metrics is None else (
            f" value_corr={metrics['value_corr_teacher']:.4f}"
            f" value_cp_std={metrics['value_cp_std']:.1f}"
            f" value_max_abs={metrics['value_max_abs_wst']:.3g}"
            f" top1={metrics['top1_teacher']:.4f}"
        )
        print(
            f"t091 iter={iteration} iter_s={elapsed:.1f} selfplay_s={selfplay_seconds:.1f} "
            f"states_s={config.states_per_iter/max(selfplay_seconds, 1e-9):.0f} "
            f"episodes={episodes} policy_ce={np.mean(policy_losses):.4f} "
            f"visit_entropy={np.mean(target_entropies):.4f} visit_kl={visit_kl:.4f} "
            f"visit_argmax_agree={np.mean(visit_agreements):.4f} frames={frames}{metric_text}",
            flush=True,
        )
        if elapsed > 2 * config.expected_iter_s:
            print(
                f"WARN_SLOW iter={iteration} took {elapsed:.0f}s "
                f"(expected ~{config.expected_iter_s:.0f}s)", flush=True
            )

        if iteration % config.ckpt_interval == 0 or iteration == config.max_num_iters:
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            if metrics is None:
                metrics = val_metrics(model_0)
            digest = tree_digest((model_0[0]["value"], model_0[1]["value"]))
            if digest != frozen_digest or metrics["value_max_abs_wst"] > 1e-7:
                raise AssertionError(f"freeze invariant failed at checkpoint {iteration}")
            path = os.path.join(config.ckpt_dir, f"{iteration:06d}.ckpt")
            with open(path, "wb") as f:
                pickle.dump(checkpoint_payload(
                    model_0, opt_state_0, iteration, rng_key, frozen_digest, metrics
                ), f)
            print(f"t091 ckpt iter={iteration} path={path} metrics={metrics}", flush=True)

    print("T091_LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
