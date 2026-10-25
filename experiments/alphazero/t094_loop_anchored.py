"""Task 094: frozen-value T2 loop with supervised policy replay.

This is the Task 091 loop with one experimental variable: a fixed fraction of
policy optimizer updates use the WS-T supervised best-move data instead of MCTS
visit distributions. Self-play generation, value freezing, optimizer, and the
total number of optimizer updates remain unchanged.
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
    max_num_iters: int = 25
    selfplay_batch_size: int = 128
    states_per_iter: int = 65_536
    num_simulations: int = 32
    max_num_steps: int = 512
    training_batch_size: int = 128
    learning_rate: float = 0.0001
    init_ckpt: str = "t087_wst/ws2.ckpt"
    resume: str = "auto"
    ckpt_dir: str = "t094_ckpt"
    ckpt_interval: int = 5
    monitor_interval: int = 5
    expected_iter_s: float = 720.0
    anchor_data_dir: str = "m4_data"
    anchor_ratio: float
    anchor_seed: int = 20260714
    monitor_batch_size: int = 256


config = Config(**OmegaConf.from_cli())
print(config, flush=True)
if config.anchor_ratio not in (0.25, 0.5, 0.75, 1.0):
    raise ValueError(
        "anchor_ratio must be 0.25/0.50 (Task 094), 1.00 (Task 095 control) "
        "or 0.75 (Task 096 dose-curve point)"
    )
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
updates_per_iter = config.states_per_iter // config.training_batch_size
anchor_updates_per_iter = round(updates_per_iter * config.anchor_ratio)
if anchor_updates_per_iter != updates_per_iter * config.anchor_ratio:
    raise ValueError("anchor_ratio must produce an exact integer update count")
num_visit_updates = updates_per_iter - anchor_updates_per_iter
# Task 095: at anchor_ratio=1.0 no update ever consumes self-play data, so
# generating it is pure waste (~552s of a 582s iteration). Skipping is provably
# a no-op: the anchor batches are drawn from np.random.default_rng(SeedSequence(
# [anchor_seed, iteration])) -- independent of the JAX rng_key that self-play
# consumes -- and self-play only READS `model`, never touching it or opt_state.
# T094_FORCE_SELFPLAY=1 generates anyway, so t095_equiv.py can verify that claim
# by digest. It is an env var, not a Config field: a new field would change
# config.model_dump(), which assert_resume_config compares, breaking resume of
# every existing Task 094 checkpoint.
FORCE_SELFPLAY = os.environ.get("T094_FORCE_SELFPLAY", "") == "1"
run_selfplay = num_visit_updates > 0 or FORCE_SELFPLAY


def mean_or_nan(values):
    """np.mean over a possibly-empty list: nan instead of a RuntimeWarning.

    The visit-statistic lists are empty at anchor_ratio=1.0 (no visit updates).
    """
    return float(np.mean(values)) if values else float("nan")

env = pgx.make(config.env_id)

with open(config.init_ckpt, "rb") as file:
    _init_ckpt = pickle.load(file)
network_config = _init_ckpt["config"]
if network_config.get("arch") != "transformer":
    raise ValueError(f"Task 094 requires Transformer WS-T, got {network_config}")
base_forward = make_base_forward(network_config)
dual_forward = make_dual_forward(network_config)
optimizer = optax.adam(config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key
    params, model_state = model
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    (logits, value), _ = dual_forward.apply(
        params, model_state, state.observation, is_eval=True
    )
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


class VisitSample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray


class AnchorSample(NamedTuple):
    obs: jnp.ndarray
    action_tgt: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> VisitSample:
    return VisitSample(obs=data.obs, policy_tgt=data.action_weights)


def visit_loss_fn(policy_params, policy_state, samples: VisitSample):
    (logits, _), policy_state = base_forward.apply(
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


def anchor_loss_fn(policy_params, policy_state, samples: AnchorSample):
    (logits, _), policy_state = base_forward.apply(
        policy_params, policy_state, samples.obs, is_eval=False
    )
    cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
        logits, samples.action_tgt
    )
    policy_loss = jnp.mean(cross_entropy)
    top1 = jnp.mean(jnp.argmax(logits, axis=-1) == samples.action_tgt)
    return policy_loss, (policy_state, top1)


@partial(jax.pmap, axis_name="i")
def train_visit(policy_model, opt_state, data: VisitSample):
    params, model_state = policy_model
    (loss, (model_state, entropy, agree)), grads = jax.value_and_grad(
        visit_loss_fn, has_aux=True
    )(params, model_state, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    return (
        (optax.apply_updates(params, updates), model_state),
        opt_state,
        loss,
        entropy,
        agree,
    )


@partial(jax.pmap, axis_name="i")
def train_anchor(policy_model, opt_state, data: AnchorSample):
    params, model_state = policy_model
    (loss, (model_state, top1)), grads = jax.value_and_grad(
        anchor_loss_fn, has_aux=True
    )(params, model_state, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    return (optax.apply_updates(params, updates), model_state), opt_state, loss, top1


_anchor_train = None
_anchor_val = None
_base_values = None
_base_actions = None


def unpack_obs(rows):
    bits = np.unpackbits(rows, axis=1)[:, : 9 * 9 * 119]
    return bits.reshape(-1, 9, 9, 119).astype(bool)


def anchor_data():
    global _anchor_train, _anchor_val
    if _anchor_train is None:
        train_obs = np.load(
            os.path.join(config.anchor_data_dir, "train_obs.npy"), mmap_mode="r"
        )
        train_actions = np.load(
            os.path.join(config.anchor_data_dir, "train_a.npy"), mmap_mode="r"
        )
        val_obs = np.load(
            os.path.join(config.anchor_data_dir, "val_obs.npy"), mmap_mode="r"
        )
        val_actions = np.load(
            os.path.join(config.anchor_data_dir, "val_a.npy"), mmap_mode="r"
        )
        if len(train_obs) != len(train_actions) or len(val_obs) != len(val_actions):
            raise ValueError("anchor observation/action count mismatch")
        _anchor_train = (train_obs, train_actions)
        _anchor_val = (val_obs, val_actions)
    return _anchor_train, _anchor_val


def value_to_cp(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.clip(1200.0 * np.arctanh(values), -5000.0, 5000.0)


def val_metrics(model_0):
    global _base_values, _base_actions
    _, (obs_mm, actions) = anchor_data()

    @jax.jit
    def predict_dual(model, obs):
        (logits, value), _ = dual_forward.apply(model[0], model[1], obs, is_eval=True)
        return logits, value

    @jax.jit
    def predict_base(model, obs):
        (logits, _), _ = base_forward.apply(model[0], model[1], obs, is_eval=True)
        return logits

    preds = []
    current_actions = []
    base_actions = []
    entropy_sum = 0.0
    ce_sum = 0.0
    hits = 0
    bs = config.monitor_batch_size
    for start in range(0, len(actions), bs):
        end = min(start + bs, len(actions))
        obs = jnp.asarray(unpack_obs(obs_mm[start:end]))
        logits, pred = predict_dual(model_0, obs)
        base_logits = predict_base(_init_ckpt["model"], obs)
        logits_np = np.asarray(logits)
        action_np = np.argmax(logits_np, axis=-1)
        base_actions.append(np.argmax(np.asarray(base_logits), axis=-1))
        current_actions.append(action_np)
        preds.append(np.asarray(pred))
        labels = np.asarray(actions[start:end], dtype=np.int32)
        hits += int((action_np == labels).sum())
        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        entropy_sum += float(
            (-np.sum(probs * np.log(np.maximum(probs, 1e-30)), axis=1)).sum()
        )
        ce_sum += float((-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-30))).sum())

    pred = np.concatenate(preds)[:len(actions)]
    action_pred = np.concatenate(current_actions)[:len(actions)]
    base_action_pred = np.concatenate(base_actions)[:len(actions)]
    if _base_values is None:
        _base_values = pred.copy()
        _base_actions = base_action_pred.copy()
    max_abs = float(np.max(np.abs(pred - _base_values)))
    return {
        "value_cp_std": float(value_to_cp(pred).std()),
        "value_max_abs_wst": max_abs,
        "top1_teacher": hits / len(actions),
        "supervised_policy_ce": ce_sum / len(actions),
        "raw_policy_entropy": entropy_sum / len(actions),
        "policy_argmax_change_wst": float(np.mean(action_pred != _base_actions)),
    }


def latest_ckpt(ckpt_dir):
    candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(re.search(r"(\d+)\.ckpt$", path).group(1)))


def checkpoint_payload(
    model_0, opt_state_0, iteration, rng_key, frozen_digest, initial_policy_digest, metrics
):
    return dict(
        kind=KIND,
        model=jax.device_get(model_0),
        opt_state=jax.device_get(opt_state_0),
        iteration=iteration,
        rng_key=rng_key,
        config=config.model_dump(),
        network_config=network_config,
        frozen_value_digest=frozen_digest,
        initial_policy_digest=initial_policy_digest,
        metrics=metrics,
    )


def assert_resume_config(saved):
    mutable = {"max_num_iters", "resume", "ckpt_interval", "monitor_interval", "expected_iter_s"}
    current = config.model_dump()
    for key, value in current.items():
        if key not in mutable and saved.get(key) != value:
            raise ValueError(
                f"resume config mismatch for {key}: saved={saved.get(key)!r} current={value!r}"
            )


def anchor_slots():
    return {
        update
        for update in range(updates_per_iter)
        if int((update + 1) * config.anchor_ratio) > int(update * config.anchor_ratio)
    }


def main():
    os.makedirs(config.ckpt_dir, exist_ok=True)
    train_data, _ = anchor_data()
    iteration = 0
    rng_key = jax.random.PRNGKey(config.seed)
    resume_path = latest_ckpt(config.ckpt_dir) if config.resume == "auto" else None

    if resume_path:
        with open(resume_path, "rb") as file:
            checkpoint = pickle.load(file)
        if checkpoint.get("kind") != KIND:
            raise ValueError(f"unexpected checkpoint kind: {checkpoint.get('kind')}")
        assert_resume_config(checkpoint["config"])
        model_0 = checkpoint["model"]
        opt_state_0 = checkpoint["opt_state"]
        iteration = checkpoint["iteration"]
        rng_key = checkpoint["rng_key"]
        frozen_digest = checkpoint["frozen_value_digest"]
        initial_policy_digest = checkpoint["initial_policy_digest"]
        actual_digest = tree_digest((model_0[0]["value"], model_0[1]["value"]))
        if actual_digest != frozen_digest:
            raise AssertionError("frozen value digest mismatch on resume")
        print(f"t094: resumed from {resume_path} at iteration {iteration}", flush=True)
    else:
        model_0 = clone_as_dual(_init_ckpt["model"])
        frozen_digest = tree_digest((model_0[0]["value"], model_0[1]["value"]))
        initial_policy_digest = tree_digest((model_0[0]["policy"], model_0[1]["policy"]))
        opt_state_0 = optimizer.init(model_0[0]["policy"])
        metrics = val_metrics(model_0)
        if metrics["value_max_abs_wst"] > 1e-7:
            raise AssertionError(f"value differs at iter0: {metrics}")
        path = os.path.join(config.ckpt_dir, "000000.ckpt")
        with open(path, "wb") as file:
            pickle.dump(
                checkpoint_payload(
                    model_0,
                    opt_state_0,
                    0,
                    rng_key,
                    frozen_digest,
                    initial_policy_digest,
                    metrics,
                ),
                file,
            )
        print(f"t094: initialized dual tower from {config.init_ckpt}", flush=True)
        print(f"t094 monitor iter=0 {metrics} frozen_digest={frozen_digest}", flush=True)

    model, opt_state = jax.tree_util.tree_map(
        lambda value: jnp.stack([value] * num_devices), (model_0, opt_state_0)
    )
    frames = iteration * config.states_per_iter
    slots = anchor_slots()
    if len(slots) != anchor_updates_per_iter:
        raise AssertionError("anchor update schedule count mismatch")
    print(
        f"t094: sims={config.num_simulations} selfplay_batch={config.selfplay_batch_size} "
        f"rounds={selfplay_rounds} states_per_iter={config.states_per_iter} "
        f"training_batch={config.training_batch_size} updates={updates_per_iter} "
        f"anchor_updates={anchor_updates_per_iter} anchor_ratio={config.anchor_ratio:.2f} "
        f"visit_updates={num_visit_updates} selfplay={'on' if run_selfplay else 'off'} "
        f"force_selfplay={FORCE_SELFPLAY} "
        f"schedule=batch_interleave expected_iter_s={config.expected_iter_s}",
        flush=True,
    )

    while iteration < config.max_num_iters:
        iteration += 1
        started = time.time()
        selfplay_seconds = 0.0
        episodes = 0
        visit_losses = []
        target_entropies = []
        visit_agreements = []
        anchor_losses = []
        anchor_top1s = []
        all_visit_batches = []

        visits = None
        if run_selfplay:
            for _round in range(selfplay_rounds):
                round_started = time.time()
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, num_devices)
                data = jax.block_until_ready(selfplay(model, keys))
                selfplay_seconds += time.time() - round_started
                episodes += int(jax.device_get(data.terminated).sum())
                samples = jax.device_get(compute_loss_input(data))
                del data
                samples = jax.tree_util.tree_map(
                    lambda value: value.reshape((-1, *value.shape[3:])), samples
                )
                all_visit_batches.append(samples)

            visits = jax.tree_util.tree_map(
                lambda *values: np.concatenate(values, axis=0), *all_visit_batches
            )
            rng_key, subkey = jax.random.split(rng_key)
            order = np.asarray(
                jax.device_get(jax.random.permutation(subkey, jnp.arange(visits.obs.shape[0])))
            )
            visits = jax.tree_util.tree_map(lambda value: value[order], visits)
            if num_visit_updates > 0:
                visits = jax.tree_util.tree_map(
                    lambda value: value[: num_visit_updates * config.training_batch_size].reshape(
                        (num_visit_updates, num_devices, -1) + value.shape[1:]
                    ),
                    visits,
                )
            else:
                # T094_FORCE_SELFPLAY diagnostic run: generated, then discarded.
                # Left as None so a scheduling bug that reached the visit branch
                # raises TypeError instead of silently indexing unshaped data.
                visits = None

        anchor_rng = np.random.default_rng(
            np.random.SeedSequence([config.anchor_seed, iteration])
        )
        anchor_indices = anchor_rng.choice(
            len(train_data[1]),
            size=anchor_updates_per_iter * config.training_batch_size,
            replace=False,
        ).reshape(anchor_updates_per_iter, config.training_batch_size)

        visit_index = 0
        anchor_index = 0
        for update in range(updates_per_iter):
            policy_model = (model[0]["policy"], model[1]["policy"])
            if update in slots:
                indices = anchor_indices[anchor_index]
                obs = unpack_obs(train_data[0][indices]).reshape(
                    num_devices, -1, 9, 9, 119
                )
                actions = np.asarray(train_data[1][indices], dtype=np.int32).reshape(
                    num_devices, -1
                )
                batch = AnchorSample(obs=jnp.asarray(obs), action_tgt=jnp.asarray(actions))
                policy_model, opt_state, loss, top1 = train_anchor(
                    policy_model, opt_state, batch
                )
                anchor_losses.append(float(loss.mean()))
                anchor_top1s.append(float(top1.mean()))
                anchor_index += 1
            else:
                batch = jax.tree_util.tree_map(lambda value: value[visit_index], visits)
                policy_model, opt_state, loss, entropy, agree = train_visit(
                    policy_model, opt_state, batch
                )
                visit_losses.append(float(loss.mean()))
                target_entropies.append(float(entropy.mean()))
                visit_agreements.append(float(agree.mean()))
                visit_index += 1
            model = (
                {"policy": policy_model[0], "value": model[0]["value"]},
                {"policy": policy_model[1], "value": model[1]["value"]},
            )

        if visit_index != num_visit_updates or anchor_index != anchor_updates_per_iter:
            raise AssertionError("optimizer update accounting mismatch")

        if run_selfplay:
            frames += config.states_per_iter
        metrics = None
        if config.monitor_interval > 0 and iteration % config.monitor_interval == 0:
            current_model = jax.tree_util.tree_map(lambda value: value[0], model)
            metrics = val_metrics(current_model)
            digest = tree_digest((current_model[0]["value"], current_model[1]["value"]))
            if digest != frozen_digest:
                raise AssertionError(f"frozen value params changed at iter {iteration}")
            if metrics["value_max_abs_wst"] > 1e-7:
                raise AssertionError(f"frozen value output changed at iter {iteration}: {metrics}")

        elapsed = time.time() - started
        visit_kl = mean_or_nan(visit_losses) - mean_or_nan(target_entropies)
        states_per_s = config.states_per_iter / selfplay_seconds if selfplay_seconds > 0 else 0.0
        metric_text = "" if metrics is None else (
            f" value_cp_std={metrics['value_cp_std']:.1f}"
            f" value_max_abs={metrics['value_max_abs_wst']:.3g}"
            f" val_top1={metrics['top1_teacher']:.4f}"
            f" val_ce={metrics['supervised_policy_ce']:.4f}"
            f" val_raw_entropy={metrics['raw_policy_entropy']:.4f}"
            f" argmax_change={metrics['policy_argmax_change_wst']:.4f}"
        )
        print(
            f"t094 iter={iteration} iter_s={elapsed:.1f} selfplay_s={selfplay_seconds:.1f} "
            f"states_s={states_per_s:.0f} "
            f"episodes={episodes} visit_ce={mean_or_nan(visit_losses):.4f} "
            f"visit_entropy={mean_or_nan(target_entropies):.4f} visit_kl={visit_kl:.4f} "
            f"visit_argmax_agree={mean_or_nan(visit_agreements):.4f} "
            f"anchor_ce={mean_or_nan(anchor_losses):.4f} "
            f"anchor_top1={mean_or_nan(anchor_top1s):.4f} "
            f"anchor_fraction={anchor_index/updates_per_iter:.4f} frames={frames}{metric_text}",
            flush=True,
        )
        if elapsed > 2 * config.expected_iter_s:
            print(
                f"WARN_SLOW iter={iteration} took {elapsed:.0f}s "
                f"(expected ~{config.expected_iter_s:.0f}s)",
                flush=True,
            )

        if iteration % config.ckpt_interval == 0 or iteration == config.max_num_iters:
            model_0, opt_state_0 = jax.tree_util.tree_map(
                lambda value: value[0], (model, opt_state)
            )
            if metrics is None:
                metrics = val_metrics(model_0)
            digest = tree_digest((model_0[0]["value"], model_0[1]["value"]))
            if digest != frozen_digest or metrics["value_max_abs_wst"] > 1e-7:
                raise AssertionError(f"freeze invariant failed at checkpoint {iteration}")
            path = os.path.join(config.ckpt_dir, f"{iteration:06d}.ckpt")
            with open(path, "wb") as file:
                pickle.dump(
                    checkpoint_payload(
                        model_0,
                        opt_state_0,
                        iteration,
                        rng_key,
                        frozen_digest,
                        initial_policy_digest,
                        metrics,
                    ),
                    file,
                )
            print(f"t094 ckpt iter={iteration} path={path} metrics={metrics}", flush=True)

    print("T094_LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
