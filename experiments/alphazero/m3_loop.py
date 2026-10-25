# Task 078 M3: Gumbel-AZ self-play loop on full shogi, warm-started from the M2
# value-distilled checkpoint. Derived from train_m1.py (itself upstream train.py +
# fixes); differences:
#   - init_ckpt: load model (params+state) from the M2 distill ckpt; adam is fresh.
#   - resume=auto: continue from the latest m3_ckpt/*.ckpt (model/opt_state/iteration/
#     rng_key), so a watchdog kill costs at most ckpt_interval iterations.
#   - anomaly detection: expected_iter_s stated up front; WARN_SLOW marker when an
#     iteration exceeds 2x; one flushed log line per iteration for tail -f.
#   - val_corr_m2 at every checkpoint: agreement of the current value head with the
#     M2 distillation val set (drop expected as value learns real outcomes; an EARLY
#     crash would indicate the warm-start being wiped).
#   - learning signal: policy = MCTS action_weights, value = discounted game outcome
#     (value_mask for truncated tails). No teacher labels.
# No wandb, no baseline eval (shogi has none).

import glob
import os
import pickle
import re
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network import AZNet

devices = jax.local_devices()
num_devices = len(devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "shogi"
    seed: int = 0
    max_num_iters: int = 150
    # network params
    num_channels: int = 64
    num_layers: int = 5
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 256
    num_simulations: int = 32
    max_num_steps: int = 512
    # training params
    training_batch_size: int = 1024
    learning_rate: float = 0.001
    # m3
    init_ckpt: str = "m2_ckpt/value_distill_best.ckpt"
    resume: str = "auto"  # "auto" | "" (fresh from init_ckpt)
    ckpt_dir: str = "m3_ckpt"
    ckpt_interval: int = 10
    corr_interval: int = 10  # Task 079: val_corr_m2 in the per-iter line every N iters (1 = every iter)
    expected_iter_s: float = 290.0
    m2_val_dir: str = "m2_data"

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config, flush=True)

env = pgx.make(config.env_id)


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    return net(x, is_training=not is_eval, test_local_stats=False)


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key
    model_params, model_state = model
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)
    return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value), state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation
        (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
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
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
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
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(config.max_num_steps))
    value_tgt = value_tgt[::-1, :]
    return Sample(obs=data.obs, policy_tgt=data.action_weights, value_tgt=value_tgt, mask=value_mask)


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(model_params, model_state, samples.obs, is_eval=False)
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, samples.policy_tgt))
    value_loss = jnp.mean(optax.l2_loss(value, samples.value_tgt) * samples.mask)
    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return (model_params, model_state), opt_state, policy_loss, value_loss


# ---- teacher-val agreement monitor (value corr + optional policy top-1) --------
# Task 080: if {m2_val_dir}/val_a.npy exists (m4_data), top-1 teacher-move agreement
# is computed in the same pass; otherwise top1 is None (m2_data behaviour unchanged).
_m2val = None


def val_metrics(model_0):
    global _m2val
    if _m2val is None:
        obs = np.load(os.path.join(config.m2_val_dir, "val_obs.npy"), mmap_mode="r")
        v = np.load(os.path.join(config.m2_val_dir, "val_v.npy"))
        a_path = os.path.join(config.m2_val_dir, "val_a.npy")
        a = np.load(a_path).astype(np.int32) if os.path.exists(a_path) else None
        _m2val = (obs, v, a)
    obs_mm, v, a = _m2val

    @jax.jit
    def predict(model, o):
        (logits, val), _ = forward.apply(model[0], model[1], o, is_eval=True)
        return jnp.argmax(logits, axis=-1), val

    preds, hits = [], 0
    bs = 4096
    for i in range(0, len(v), bs):
        rows = np.unpackbits(obs_mm[i : i + bs], axis=1)[:, : 9 * 9 * 119]
        am, val = predict(model_0, jnp.asarray(rows.reshape(-1, 9, 9, 119).astype(bool)))
        preds.append(np.asarray(val))
        if a is not None:
            hits += int((np.asarray(am) == a[i : i + bs]).sum())
    p = np.concatenate(preds)[: len(v)]
    corr = float(np.corrcoef(p, v)[0, 1])
    return corr, (hits / len(v) if a is not None else None)


def val_corr_m2(model_0):
    return val_metrics(model_0)[0]


def latest_ckpt(ckpt_dir):
    cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not cands:
        return None
    return max(cands, key=lambda p: int(re.search(r"(\d+)\.ckpt$", p).group(1)))


if __name__ == "__main__":
    os.makedirs(config.ckpt_dir, exist_ok=True)
    iteration = 0
    rng_key = jax.random.PRNGKey(config.seed)

    resume_path = latest_ckpt(config.ckpt_dir) if config.resume == "auto" else None
    if resume_path:
        with open(resume_path, "rb") as f:
            ck = pickle.load(f)
        model_0, opt_state_0 = ck["model"], ck["opt_state"]
        iteration, rng_key = ck["iteration"], ck["rng_key"]
        print(f"m3: resumed from {resume_path} at iteration {iteration}", flush=True)
    else:
        with open(config.init_ckpt, "rb") as f:
            ck = pickle.load(f)
        assert ck["config"]["num_channels"] == config.num_channels
        assert ck["config"]["num_layers"] == config.num_layers
        model_0 = ck["model"]
        opt_state_0 = optimizer.init(model_0[0])  # fresh adam
        print(f"m3: initialized from {config.init_ckpt} (val_corr {ck.get('val_corr')}); fresh adam", flush=True)
        # save the starting point as iteration 0 for the curve
        with open(os.path.join(config.ckpt_dir, "000000.ckpt"), "wb") as f:
            pickle.dump(dict(model=model_0, opt_state=opt_state_0, iteration=0, rng_key=rng_key,
                             config=config.model_dump()), f)

    model, opt_state = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * num_devices), (model_0, opt_state_0)
    )
    frames = 0
    print(f"m3: expected_iter_s={config.expected_iter_s} (WARN_SLOW at >2x), "
          f"ckpt every {config.ckpt_interval} iters -> {config.ckpt_dir}/", flush=True)

    while iteration < config.max_num_iters:
        iteration += 1
        st = time.time()

        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        data = jax.block_until_ready(data)
        sp_et = time.time()
        samples: Sample = compute_loss_input(data)

        samples = jax.device_get(samples)
        episodes = int(jax.device_get(data.terminated).sum())
        mask_frac = float(samples.mask.mean())
        frames_delta = samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        frames += frames_delta
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        policy_losses, value_losses = [], []
        for i in range(num_updates):
            mb = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, pl, vl = train(model, opt_state, mb)
            policy_losses.append(pl.mean().item())
            value_losses.append(vl.mean().item())

        corr_now = top1_now = None
        if config.corr_interval > 0 and iteration % config.corr_interval == 0:
            corr_now, top1_now = val_metrics(jax.tree_util.tree_map(lambda x: x[0], model))

        et = time.time()
        iter_s = et - st
        corr_str = f" corr_m2={corr_now:.4f}" if corr_now is not None else ""
        if top1_now is not None:
            corr_str += f" top1={top1_now:.4f}"
        print(
            f"m3 iter={iteration} iter_s={iter_s:.1f} selfplay_s={sp_et-st:.1f} "
            f"states_s={frames_delta/(sp_et-st):.0f} mask={mask_frac:.3f} episodes={episodes} "
            f"policy_loss={np.mean(policy_losses):.4f} value_loss={np.mean(value_losses):.4f} "
            f"frames={frames}{corr_str}",
            flush=True,
        )
        if iter_s > 2 * config.expected_iter_s:
            print(f"WARN_SLOW iter={iteration} took {iter_s:.0f}s (expected ~{config.expected_iter_s:.0f}s)", flush=True)

        if iteration % config.ckpt_interval == 0 or iteration == config.max_num_iters:
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            if corr_now is None:
                corr_now, top1_now = val_metrics((model_0[0], model_0[1]))
            with open(os.path.join(config.ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                pickle.dump(dict(model=jax.device_get(model_0), opt_state=jax.device_get(opt_state_0),
                                 iteration=iteration, rng_key=rng_key, config=config.model_dump(),
                                 val_corr_m2=corr_now, top1=top1_now), f)
            corr = corr_now
            print(f"m3 ckpt iter={iteration} val_corr_m2={corr:.4f}", flush=True)

    print("M3_LOOP_DONE", flush=True)
