"""Task 098: shared-trunk T2 loop with a WS-T value distillation regulariser.

Tasks 091-097 ran a FROZEN-value two-tower loop: `t091_model.clone_as_dual` makes
two complete independent copies of the network, the policy CE reaches only the
policy copy, and the value copy is re-attached by reference every update. That
established the arc's first external positive transfer (+26 Elo, replicated in
097) -- but the winning net's value tower is byte-identical to WS-T, so as a
VALUE teacher it is still WS-T. To carry the gain into the product NNUE the value
must move.

Simply unfreezing the two-tower value and adding `lambda * MSE(v, v_WS-T)` is a
provable no-op: with no shared trunk the policy CE has no path to the value
params, and the MSE is identically zero at initialisation (the value copy IS
WS-T), so its gradient is zero too. The value would never move.

So the one variable here is the VALUE TOWER'S TREATMENT: two-tower frozen ->
SHARED TRUNK with a distillation regulariser. A single parameter set carries both
heads; the loss is

    policy_CE(visit | anchor)  +  lambda * mean((v - v_WS-T)^2)

The policy CE moves the trunk, the value head reads that trunk, and lambda
constrains how far the value may drift from WS-T. Everything else is identical to
the Task 096 r=0.50 arm (r=0.50 anchor, sims 32, 65,536 states/iter, lr 1e-4,
batch 128, 25 iterations, seed = anchor_seed = 20260714).

Two consequences worth stating plainly:

  * Self-play gets ~2x cheaper. The two-tower `FrozenValuePolicyForward.apply`
    evaluated the network TWICE at every MCTS node (once per tower); a shared
    trunk evaluates it once. At initialisation both towers held identical
    parameters, so this changes the cost, not the search behaviour.
  * The value digest is now EXPECTED to change. The freeze assertions become
    measurements: value_std_ratio, value_corr_teacher, value_mae_wst.

Pre-registered stop rule: if `value_std_ratio` drops below 0.80 the run halts.
Task 082's gen100 incident collapsed the value std from 1471 to 698 (ratio 0.475)
over 100 iterations of training value on self-play returns, and Task 084 then
measured that teacher at -145 Elo against the pre-loop net. 0.80 has 2.4x
headroom over that failure, so it catches a gen100-class collapse decisively and
says nothing about milder drift.

The checkpoint is deliberately shaped like a plain WS-T checkpoint (`config`
holds the NETWORK config, `model` is a single (params, state) pair) so that
`m2_usi.load_model`, `m2_match.py` and `az_rescore.py` consume it unmodified.
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

from t091_model import make_base_forward, tree_digest

KIND = "T098SharedTrunkValueDistill"

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
    ckpt_dir: str = "t098_ckpt"
    ckpt_interval: int = 5
    monitor_interval: int = 5
    expected_iter_s: float = 420.0
    anchor_data_dir: str = "m4_data"
    anchor_ratio: float = 0.5
    anchor_seed: int = 20260714
    monitor_batch_size: int = 256
    value_lambda: float
    value_std_ratio_stop: float = 0.80


config = Config(**OmegaConf.from_cli())
print(config, flush=True)
if config.anchor_ratio != 0.5:
    raise ValueError("Task 098 fixes anchor_ratio at 0.50 (the Task 096 arm)")
if config.value_lambda < 0.0:
    raise ValueError("value_lambda must be >= 0")
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


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


env = pgx.make(config.env_id)

with open(config.init_ckpt, "rb") as file:
    _init_ckpt = pickle.load(file)
network_config = _init_ckpt["config"]
if network_config.get("arch") != "transformer":
    raise ValueError(f"Task 098 requires Transformer WS-T, got {network_config}")
base_forward = make_base_forward(network_config)
optimizer = optax.adam(config.learning_rate)

# The WS-T reference net. It is never trained; it supplies the distillation
# target and every drift diagnostic. Replicated once for the pmapped train step.
_ref_model = _init_ckpt["model"]
_ref_replicated = jax.tree_util.tree_map(
    lambda value: jnp.stack([value] * num_devices), _ref_model
)
WST_VALUE_DIGEST = tree_digest(_ref_model)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key
    params, model_state = model
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    # Shared trunk: ONE forward gives both heads (the two-tower loop needed two).
    (logits, value), _ = base_forward.apply(
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
        (logits, value), _ = base_forward.apply(
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


def _distill(value, v_ref):
    """Plain MSE, NOT optax.l2_loss (which is half of it).

    lambda is laddered logarithmically, so the factor-of-two matters for
    reproducing the selected value.
    """
    return jnp.mean((value - v_ref) ** 2)


def visit_loss_fn(params, model_state, samples: VisitSample, v_ref):
    (logits, value), model_state = base_forward.apply(
        params, model_state, samples.obs, is_eval=False
    )
    cross_entropy = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(cross_entropy)
    value_loss = _distill(value, v_ref)
    target_entropy = -jnp.mean(
        jnp.sum(samples.policy_tgt * jnp.log(jnp.maximum(samples.policy_tgt, 1e-12)), axis=-1)
    )
    visit_agree = jnp.mean(
        jnp.argmax(logits, axis=-1) == jnp.argmax(samples.policy_tgt, axis=-1)
    )
    total = policy_loss + config.value_lambda * value_loss
    return total, (model_state, policy_loss, value_loss, target_entropy, visit_agree)


def anchor_loss_fn(params, model_state, samples: AnchorSample, v_ref):
    (logits, value), model_state = base_forward.apply(
        params, model_state, samples.obs, is_eval=False
    )
    cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
        logits, samples.action_tgt
    )
    policy_loss = jnp.mean(cross_entropy)
    value_loss = _distill(value, v_ref)
    top1 = jnp.mean(jnp.argmax(logits, axis=-1) == samples.action_tgt)
    total = policy_loss + config.value_lambda * value_loss
    return total, (model_state, policy_loss, value_loss, top1)


def _reference_value(ref_model, obs):
    """WS-T's value on the same observations -- the distillation target.

    Computed online for both update kinds. Precomputing the whole 2M anchor pool
    offline would cost MORE (2M forwards once vs 25*256*128 = 819k actually
    drawn), so there is nothing to cache.
    """
    (_, v_ref), _ = base_forward.apply(ref_model[0], ref_model[1], obs, is_eval=True)
    return jax.lax.stop_gradient(v_ref)


@partial(jax.pmap, axis_name="i")
def train_visit(model, ref_model, opt_state, data: VisitSample):
    params, model_state = model
    v_ref = _reference_value(ref_model, data.obs)
    (_, aux), grads = jax.value_and_grad(visit_loss_fn, has_aux=True)(
        params, model_state, data, v_ref
    )
    model_state, policy_loss, value_loss, entropy, agree = aux
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    return (
        (optax.apply_updates(params, updates), model_state),
        opt_state, policy_loss, value_loss, entropy, agree,
    )


@partial(jax.pmap, axis_name="i")
def train_anchor(model, ref_model, opt_state, data: AnchorSample):
    params, model_state = model
    v_ref = _reference_value(ref_model, data.obs)
    (_, aux), grads = jax.value_and_grad(anchor_loss_fn, has_aux=True)(
        params, model_state, data, v_ref
    )
    model_state, policy_loss, value_loss, top1 = aux
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    return (
        (optax.apply_updates(params, updates), model_state),
        opt_state, policy_loss, value_loss, top1,
    )


_anchor_train = None
_anchor_val = None


def unpack_obs(rows):
    bits = np.unpackbits(rows, axis=1)[:, : 9 * 9 * 119]
    return bits.reshape(-1, 9, 9, 119).astype(bool)


def anchor_data():
    """m4_data train (2M anchor pool) and val (50k monitor set).

    `val_v.npy` is the teacher value target tanh(gikou_cp/1200); Task 091 used it
    for `value_corr_teacher` and Task 094 dropped it. Task 098 needs it back.
    """
    global _anchor_train, _anchor_val
    if _anchor_train is None:
        def load(name):
            return np.load(os.path.join(config.anchor_data_dir, name), mmap_mode="r")

        train_obs, train_actions = load("train_obs.npy"), load("train_a.npy")
        val_obs, val_actions, val_values = load("val_obs.npy"), load("val_a.npy"), load("val_v.npy")
        if len(train_obs) != len(train_actions) or len(val_obs) != len(val_actions):
            raise ValueError("anchor observation/action count mismatch")
        if len(val_values) != len(val_actions):
            raise ValueError("anchor value/action count mismatch")
        _anchor_train = (train_obs, train_actions)
        _anchor_val = (val_obs, val_actions, val_values)
    return _anchor_train, _anchor_val


def value_to_cp(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.clip(1200.0 * np.arctanh(values), -5000.0, 5000.0)


def val_metrics(model_0):
    """Monitor on m4_data val (50k), disjoint from run2m_c.

    Unlike Task 094's version, the WS-T reference value is recomputed from the
    WS-T checkpoint on every call rather than cached from the live model's first
    output. Task 094 could get away with that because the value was frozen; here
    the value moves, and a resumed run would otherwise baseline against an
    already-drifted model.
    """
    _, (obs_mm, actions, teacher_values) = anchor_data()

    @jax.jit
    def predict(model, obs):
        (logits, value), _ = base_forward.apply(model[0], model[1], obs, is_eval=True)
        return logits, value

    preds, wst_preds, current_actions, base_actions = [], [], [], []
    entropy_sum = ce_sum = 0.0
    hits = 0
    bs = config.monitor_batch_size
    for start in range(0, len(actions), bs):
        end = min(start + bs, len(actions))
        obs = jnp.asarray(unpack_obs(obs_mm[start:end]))
        logits, pred = predict(model_0, obs)
        base_logits, base_pred = predict(_ref_model, obs)
        logits_np = np.asarray(logits)
        action_np = np.argmax(logits_np, axis=-1)
        base_actions.append(np.argmax(np.asarray(base_logits), axis=-1))
        current_actions.append(action_np)
        preds.append(np.asarray(pred))
        wst_preds.append(np.asarray(base_pred))
        labels = np.asarray(actions[start:end], dtype=np.int32)
        hits += int((action_np == labels).sum())
        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        entropy_sum += float((-np.sum(probs * np.log(np.maximum(probs, 1e-30)), axis=1)).sum())
        ce_sum += float((-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-30))).sum())

    n = len(actions)
    pred = np.concatenate(preds)[:n]
    wst = np.concatenate(wst_preds)[:n]
    action_pred = np.concatenate(current_actions)[:n]
    base_action_pred = np.concatenate(base_actions)[:n]
    pred_cp, wst_cp = value_to_cp(pred), value_to_cp(wst)
    teacher = np.asarray(teacher_values[:n], dtype=np.float64)
    wst_std = float(wst_cp.std())
    return {
        "value_cp_std": float(pred_cp.std()),
        "value_wst_cp_std": wst_std,
        # Ratio against WS-T measured on the SAME positions in the same call, so
        # it never depends on a hard-coded constant.
        "value_std_ratio": float(pred_cp.std() / wst_std) if wst_std > 0 else float("nan"),
        # Mean |dcp| vs WS-T -- directly comparable to Task 092-0's finding that a
        # ~10cp MAE difference was not resolvable in Elo at 400 games.
        "value_mae_wst": float(np.abs(pred_cp - wst_cp).mean()),
        "value_max_abs_wst": float(np.max(np.abs(pred - wst))),
        # Correlation against the gikou teacher target; WS-T sits at 0.9357.
        "value_corr_teacher": float(np.corrcoef(pred, teacher)[0, 1]),
        "value_corr_wst": float(np.corrcoef(pred, wst)[0, 1]),
        "top1_teacher": hits / n,
        "supervised_policy_ce": ce_sum / n,
        "raw_policy_entropy": entropy_sum / n,
        "policy_argmax_change_wst": float(np.mean(action_pred != base_action_pred)),
    }


def latest_ckpt(ckpt_dir):
    candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(re.search(r"(\d+)\.ckpt$", path).group(1)))


def checkpoint_payload(model_0, opt_state_0, iteration, rng_key, metrics):
    """Shaped like a plain WS-T checkpoint on purpose.

    `config` holds the NETWORK config and `model` a single (params, state) pair,
    so `m2_usi.load_model`'s non-T091 branch loads it unmodified -- which means
    `m2_match.py` and `az_rescore.py` work with no changes. The run settings live
    under `run_config`.
    """
    return dict(
        kind=KIND,
        model=jax.device_get(model_0),
        opt_state=jax.device_get(opt_state_0),
        iteration=iteration,
        rng_key=rng_key,
        config=network_config,
        run_config=config.model_dump(),
        network_config=network_config,
        wst_value_digest=WST_VALUE_DIGEST,
        metrics=metrics,
    )


def assert_resume_config(saved):
    mutable = {"max_num_iters", "resume", "ckpt_interval", "monitor_interval",
               "expected_iter_s"}
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
        assert_resume_config(checkpoint["run_config"])
        if checkpoint.get("wst_value_digest") != WST_VALUE_DIGEST:
            raise AssertionError("WS-T reference digest differs from the resumed run")
        model_0 = checkpoint["model"]
        opt_state_0 = checkpoint["opt_state"]
        iteration = checkpoint["iteration"]
        rng_key = checkpoint["rng_key"]
        print(f"t098: resumed from {resume_path} at iteration {iteration}", flush=True)
    else:
        model_0 = jax.tree_util.tree_map(lambda x: x, _ref_model)
        opt_state_0 = optimizer.init(model_0[0])
        metrics = val_metrics(model_0)
        # At iteration 0 the shared-trunk net IS WS-T, so the drift metrics must
        # read exactly zero. This is the wiring check for the reference path.
        if metrics["value_max_abs_wst"] > 1e-7 or abs(metrics["value_std_ratio"] - 1.0) > 1e-6:
            raise AssertionError(f"iter0 is not WS-T: {metrics}")
        path = os.path.join(config.ckpt_dir, "000000.ckpt")
        with open(path, "wb") as file:
            pickle.dump(checkpoint_payload(model_0, opt_state_0, 0, rng_key, metrics), file)
        print(f"t098: initialized shared trunk from {config.init_ckpt}", flush=True)
        print(f"t098 monitor iter=0 {metrics} wst_digest={WST_VALUE_DIGEST}", flush=True)

    model, opt_state = jax.tree_util.tree_map(
        lambda value: jnp.stack([value] * num_devices), (model_0, opt_state_0)
    )
    frames = iteration * config.states_per_iter
    slots = anchor_slots()
    if len(slots) != anchor_updates_per_iter:
        raise AssertionError("anchor update schedule count mismatch")
    print(
        f"t098: sims={config.num_simulations} selfplay_batch={config.selfplay_batch_size} "
        f"rounds={selfplay_rounds} states_per_iter={config.states_per_iter} "
        f"training_batch={config.training_batch_size} updates={updates_per_iter} "
        f"anchor_updates={anchor_updates_per_iter} anchor_ratio={config.anchor_ratio:.2f} "
        f"visit_updates={num_visit_updates} trunk=shared "
        f"value_lambda={config.value_lambda} std_ratio_stop={config.value_std_ratio_stop} "
        f"schedule=batch_interleave expected_iter_s={config.expected_iter_s}",
        flush=True,
    )

    while iteration < config.max_num_iters:
        iteration += 1
        started = time.time()
        selfplay_seconds = 0.0
        episodes = 0
        visit_losses, target_entropies, visit_agreements = [], [], []
        anchor_losses, anchor_top1s, value_losses = [], [], []
        all_visit_batches = []

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
        visits = jax.tree_util.tree_map(
            lambda value: value[: num_visit_updates * config.training_batch_size].reshape(
                (num_visit_updates, num_devices, -1) + value.shape[1:]
            ),
            visits,
        )

        anchor_rng = np.random.default_rng(
            np.random.SeedSequence([config.anchor_seed, iteration])
        )
        anchor_indices = anchor_rng.choice(
            len(train_data[1]),
            size=anchor_updates_per_iter * config.training_batch_size,
            replace=False,
        ).reshape(anchor_updates_per_iter, config.training_batch_size)

        visit_index = anchor_index = 0
        for update in range(updates_per_iter):
            if update in slots:
                indices = anchor_indices[anchor_index]
                obs = unpack_obs(train_data[0][indices]).reshape(num_devices, -1, 9, 9, 119)
                actions = np.asarray(train_data[1][indices], dtype=np.int32).reshape(
                    num_devices, -1
                )
                batch = AnchorSample(obs=jnp.asarray(obs), action_tgt=jnp.asarray(actions))
                model, opt_state, loss, value_loss, top1 = train_anchor(
                    model, _ref_replicated, opt_state, batch
                )
                anchor_losses.append(float(loss.mean()))
                anchor_top1s.append(float(top1.mean()))
                value_losses.append(float(value_loss.mean()))
                anchor_index += 1
            else:
                batch = jax.tree_util.tree_map(lambda value: value[visit_index], visits)
                model, opt_state, loss, value_loss, entropy, agree = train_visit(
                    model, _ref_replicated, opt_state, batch
                )
                visit_losses.append(float(loss.mean()))
                target_entropies.append(float(entropy.mean()))
                visit_agreements.append(float(agree.mean()))
                value_losses.append(float(value_loss.mean()))
                visit_index += 1

        if visit_index != num_visit_updates or anchor_index != anchor_updates_per_iter:
            raise AssertionError("optimizer update accounting mismatch")

        frames += config.states_per_iter
        metrics = None
        if config.monitor_interval > 0 and iteration % config.monitor_interval == 0:
            current_model = jax.tree_util.tree_map(lambda value: value[0], model)
            metrics = val_metrics(current_model)

        elapsed = time.time() - started
        visit_kl = mean_or_nan(visit_losses) - mean_or_nan(target_entropies)
        states_per_s = config.states_per_iter / selfplay_seconds if selfplay_seconds > 0 else 0.0
        metric_text = "" if metrics is None else (
            f" value_cp_std={metrics['value_cp_std']:.1f}"
            f" value_std_ratio={metrics['value_std_ratio']:.4f}"
            f" value_mae_wst={metrics['value_mae_wst']:.1f}"
            f" value_corr_teacher={metrics['value_corr_teacher']:.4f}"
            f" value_max_abs={metrics['value_max_abs_wst']:.3g}"
            f" val_top1={metrics['top1_teacher']:.4f}"
            f" val_ce={metrics['supervised_policy_ce']:.4f}"
            f" argmax_change={metrics['policy_argmax_change_wst']:.4f}"
        )
        print(
            f"t098 iter={iteration} iter_s={elapsed:.1f} selfplay_s={selfplay_seconds:.1f} "
            f"states_s={states_per_s:.0f} "
            f"episodes={episodes} visit_ce={mean_or_nan(visit_losses):.4f} "
            f"visit_entropy={mean_or_nan(target_entropies):.4f} visit_kl={visit_kl:.4f} "
            f"visit_argmax_agree={mean_or_nan(visit_agreements):.4f} "
            f"anchor_ce={mean_or_nan(anchor_losses):.4f} "
            f"anchor_top1={mean_or_nan(anchor_top1s):.4f} "
            f"value_mse={mean_or_nan(value_losses):.6f} "
            f"anchor_fraction={anchor_index/updates_per_iter:.4f} frames={frames}{metric_text}",
            flush=True,
        )
        if elapsed > 2 * config.expected_iter_s:
            print(
                f"WARN_SLOW iter={iteration} took {elapsed:.0f}s "
                f"(expected ~{config.expected_iter_s:.0f}s)",
                flush=True,
            )

        stop_now = False
        if metrics is not None and metrics["value_std_ratio"] < config.value_std_ratio_stop:
            # Pre-registered halt: a gen100-class value collapse (Task 082 hit
            # ratio 0.475) makes the run scientifically void, so stop rather than
            # burn the remaining iterations.
            print(
                f"T098_VALUE_STOP iter={iteration} "
                f"value_std_ratio={metrics['value_std_ratio']:.4f} "
                f"< {config.value_std_ratio_stop}",
                flush=True,
            )
            stop_now = True

        if iteration % config.ckpt_interval == 0 or iteration == config.max_num_iters or stop_now:
            model_0, opt_state_0 = jax.tree_util.tree_map(
                lambda value: value[0], (model, opt_state)
            )
            if metrics is None:
                metrics = val_metrics(model_0)
            path = os.path.join(config.ckpt_dir, f"{iteration:06d}.ckpt")
            with open(path, "wb") as file:
                pickle.dump(
                    checkpoint_payload(model_0, opt_state_0, iteration, rng_key, metrics), file
                )
            print(f"t098 ckpt iter={iteration} path={path} metrics={metrics}", flush=True)
        if stop_now:
            return 2

    print("T098_LOOP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
