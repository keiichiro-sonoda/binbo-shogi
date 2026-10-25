#!/usr/bin/env python3
"""Task 099: does the loop's gain reach the value NUMBER through search?

Task 098 closed the raw-value carriage path: a shared-trunk loop moved the value
59.2 cp without compressing it, yet the NNUE student trained on those labels was
indistinguishable from a WS-T-label student (-0.87 [-23.9, +22.2]), and the label
statistics were identical to WS-T's (std 1502/1500, corr_gikou 0.958/0.957,
|dcp|median 201/202) -- the drift stayed inside WS-T's noise shell.

Meanwhile the loop's +26/+27 external gain is real and replicated (097), and it
lives in POLICY x SEARCH. So the one untried carriage path is the ROOT VALUE
AFTER SEARCH: the number search settles on once a better policy has guided it.
This task screens that path on subsamples before anyone pays for a full 2M
rescore. No training, no games -- label statistics only.

THE LABELLERS (and why this comparison is unusually clean):

    (1)  WS-T raw value                       the centre of the shell
    (1') 097-run2 raw value                   NULL CONTROL, see below
    (2)  WS-T + search   (root value, sims S) search alone
    (3)  097-run2 + search (same S)           search + loop gain

097-run2 came out of a FROZEN-VALUE loop, so its value tower is bit-identical to
WS-T's. Therefore:
  * (3) - (2) isolates exactly "the same value head, guided by a better policy".
    There is no value-quality confound by construction.
  * (1') is provably equal to (1). It is not a fourth arm -- it is a zero-cost
    wiring check, and it is asserted, not assumed.
So where 098 asked "does the value head carry new information?", this asks "does
the POLICY inject information into the value number, via search?"

PRE-REGISTERED (committed locally before Stage 0 runs; see the plan):
  * S chosen in Stage 0 on MECHANISM ONLY (search-vs-raw divergence and cost),
    never on gold closeness: the smallest sims whose median |dcp(search, raw)|
    clears both the measured implementation-noise floor and 30 cp (098's bar for
    "the value moved"; 092-0 found ~10 cp is not resolvable in Elo).
  * primary = median |cp_labeller - cp_gold| on the Task 070 gold set, with a
    position-level bootstrap CI on the PAIRED differences; decision table below.
  * NO_BREACH may only be declared when gold AND the distribution statistics both
    say "inside the shell" -- see rows 4/5.

TWO CORRECTIONS TO THE BRIEF, both verified on disk:
  * The gold set is NOT gikou@d16. It is the champion NNUE (task064_32m_lr1e-5)
    searched at depth 16 in YaneuraOu. gikou has never been run at d16 here
    beyond four debug positions, and Task 072-pre found it saturates by d8
    (d8->d12 median ~71 cp) and crashes 14% / hard-fails 9% under cold deep
    search. So no gikou gold is built. 98.17% of the gold positions sit in the
    champion's own training shards and 6.30% are exact rows of m4_data train --
    but all of (1)(2)(3) are WS-T-derived nets, so that contamination is
    SYMMETRIC across the arms: the relative comparison stands, the absolute
    closeness does not.
  * `m2_usi.build_search` returns the RAW value head, not a search value (it
    binds `value` from the root forward and discards `out.search_tree`; it even
    prints `info depth 1`). Nothing in the repo scores stored positions with
    search. Hence this file.

THE IMPLEMENTATION RISK THAT DOMINATES EVERYTHING ELSE: pgx's State defaults
`current_player=0`, while `_flip` leaves `color` untouched and `Game.rewards`
indexes by `color`. `recurrent_fn` then reads `state.rewards[..., current_player]`.
For a white-to-move root that silently SIGN-FLIPS every terminal reward inside
the search -- and roughly half of all stored positions are white to move. The
sign oracle below is the only defence, and it is a hard gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cshogi
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from pgx._src.games.shogi import Game, GameState, _observe

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from m2_distill_data import batch_state, parse_sfen_np  # noqa: E402
from m2_usi import load_model  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/GitHub/shogi/binbo-shogi"))
from trainer.prepare_data import shard_files  # noqa: E402

env = pgx.make("shogi")
_game = Game()

# ---- pre-registered constants -------------------------------------------------
SIMS_LADDER = (32, 128, 512)
BENCH_POSITIONS = 2_000
SEARCH_BATCH = 128                  # 091 measured B=256 as WORSE per state (110 vs 117/s)
SEARCH_BATCH_BIG_SIMS = 64          # sims=512 trees run ~4.4GB at B=128
BIG_SIMS = 512
# 098 accepted 30 cp as "the value moved"; 092-0 found ~10 cp unresolvable in Elo.
S_SELECT_MIN_DIVERGENCE_CP = 30.0
STAGE2_POSITIONS = 200_000
STAGE2_POSITIONS_BIG_SIMS = 50_000  # 200k x 2 labellers at sims=512 would exceed 10h
STAGE2_SEED = 20269900
BOOTSTRAP_SAMPLES = 200_000
BOOTSTRAP_SEED = 20269901
# 098's candidate moved 59.2 cp from WS-T and was still inside the shell. A
# NO_BREACH verdict has to clear that by a clear margin, otherwise the honest
# answer is INCONCLUSIVE (gold null but the function genuinely moved).
SHELL_EXIT_MEDIAN_CP = 100.0
CLAMP_CP = 5000.0
NOISE_FLOOR_REPEATS = 3

DECISION_TABLE = (
    "0 sign oracle / null control / prerequisites failed -> INVALID (non-terminal) | "
    "1 CI(d31) high < 0 AND CI(d32) high < 0 -> BREACH | "
    "2 CI(d21) high < 0 AND CI(d32) contains 0 -> SEARCH_ONLY | "
    "3 CI(d31) low > 0 -> REGRESSION | "
    "4 else and median|dcp(3,1)| < 100cp -> NO_BREACH | "
    "5 else -> INCONCLUSIVE"
)

# Task 070's published closeness-to-d16, for the same-axis comparison.
T070_BASELINE = {"d12_median": 80.0, "d14_median": 47.0, "n_full": 2882}
# Task 098's label statistics on run2m_c, the "inside the shell" reference.
T098_SHELL = {"cand": {"std": 1502, "corr_gikou": 0.958, "abs_median": 201},
              "wst": {"std": 1500, "corr_gikou": 0.957, "abs_median": 202}}


def value_to_cp(values, clamp=CLAMP_CP):
    values = np.clip(np.asarray(values, dtype=np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.clip(1200.0 * np.arctanh(values), -clamp, clamp)


# ---- the missing part: a batched pgx.State rebuilt from stored positions -------
_legal_mask_fn = jax.jit(jax.vmap(_game.legal_action_mask))
_observe_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(True))))


def batch_full_state(colors, boards, hands):
    """(color, board, hand) triples -> a batched pgx.shogi.State fit for MCTS.

    `m2_distill_data.batch_state` only builds a GameState, and fills
    `legal_action_mask` with the START POSITION's mask on purpose (its comment
    says so) because `_observe` never reads it. MCTS does: `env.step` treats an
    action outside the mask as an instant loss. So the mask is recomputed here.

    `current_player` is set from `color`, matching what `Shogi._init` does
    (`current_player=_player_order[x.color]` with the identity order). Leaving
    pgx's default 0 would sign-flip every terminal reward for white-to-move
    roots. `parse_sfen_np` has ALREADY applied the side-to-move flip, so unlike
    `pgx.experimental.shogi._from_board` no second flip is applied here.
    """
    x = batch_state(colors, boards, hands)
    n = len(colors)
    colors_j = jnp.asarray(np.asarray(colors, np.int32))
    return pgx.shogi.State(
        current_player=colors_j,
        rewards=jnp.zeros((n, 2), dtype=jnp.float32),
        terminated=jnp.zeros(n, dtype=jnp.bool_),
        truncated=jnp.zeros(n, dtype=jnp.bool_),
        legal_action_mask=_legal_mask_fn(x),
        observation=_observe_fn(x),
        _step_count=jnp.zeros(n, dtype=jnp.int32),
        _player_order=jnp.tile(jnp.arange(2, dtype=jnp.int32), (n, 1)),
        _x=x,
    )


def sfens_to_state(sfens):
    colors, boards, hands = [], [], []
    for sfen in sfens:
        color, board, hand = parse_sfen_np(sfen)
        colors.append(color)
        boards.append(board)
        hands.append(hand)
    return batch_full_state(colors, boards, hands)


# ---- search --------------------------------------------------------------------
def build_batched_search(forward, model, num_simulations):
    """`m2_usi.build_search` with the batch-1 indexing removed, returning the
    SEARCH-backed root value instead of the raw value head.

    `out.search_tree.summary().value` is mctx's public root statistic:
    `(v_raw + sum_i G_i) / (S + 1)` where each G_i is a backed-up return already
    in root-player POV (the `discount = -1` flips sign each ply). The
    visit-weighted mean of the root Q values is returned alongside as an
    independent read of the same tree -- the two differ mainly in how much weight
    the raw root value keeps, i.e. `1/(S+1)`.
    """
    def recurrent_fn(model_, rng_key, action, state):
        del rng_key
        params, state_ = model_
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        (logits, value), _ = forward.apply(params, state_, state.observation, is_eval=True)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -jnp.ones_like(value))
        return mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        ), state

    @jax.jit
    def search(state_b):
        params, state_ = model
        (logits, value), _ = forward.apply(params, state_, state_b.observation, is_eval=True)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state_b)
        out = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=jax.random.PRNGKey(0),   # gumbel_scale=0 -> unused, kept explicit
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state_b.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=0.0,                # a labeller must be deterministic
        )
        summary = out.search_tree.summary()
        visit_q = jnp.sum(summary.visit_probs * summary.qvalues, axis=-1)
        return summary.value, visit_q, value, jnp.max(summary.visit_counts, axis=-1)

    return search


def raw_value_fn(forward, model):
    @jax.jit
    def value_of(obs):
        (_logits, value), _ = forward.apply(model[0], model[1], obs, is_eval=True)
        return value
    return value_of


def score_sfens(search, sfens, batch):
    """-> (search_value, visit_weighted_q, raw_value, max_visits), all np arrays."""
    out = [[], [], [], []]
    for start in range(0, len(sfens), batch):
        state = sfens_to_state(sfens[start:start + batch])
        for slot, array in enumerate(search(state)):
            out[slot].append(np.asarray(array))
    return tuple(np.concatenate(part) for part in out)


# ---- the hard gate: does the search have its signs right? ----------------------
# Mate-in-1 for the side to move, one with black to move and one with white, plus
# a colour-mirrored pair. If `current_player` were left at pgx's default the
# white-to-move cases would come back negative.
SIGN_ORACLE = [
    ("4k4/9/4P4/9/9/9/9/9/4K4 b G 1", "black_mate_in_1"),
    ("4k4/9/9/9/9/9/4p4/9/4K4 w g 1", "white_mate_in_1"),
]


def sign_oracle(forward, model, sims):
    """A mate-in-1 root must search to a decisively positive value.

    A sign error inside `recurrent_fn` makes the search believe the mate loses,
    which would quietly corrupt every white-to-move label in the corpus. The bar
    is deliberately loose (> 0.5): this catches a sign flip, not a value-quality
    regression.
    """
    search = build_batched_search(forward, model, sims)
    sfens = [sfen for sfen, _ in SIGN_ORACLE]
    values, visit_q, raw, _ = score_sfens(search, sfens, len(sfens))
    report = {}
    failures = []
    for index, (sfen, name) in enumerate(SIGN_ORACLE):
        report[name] = {"sfen": sfen, "search_value": float(values[index]),
                        "visit_q": float(visit_q[index]), "raw_value": float(raw[index])}
        if values[index] <= 0.5:
            failures.append(f"{name}={values[index]:.3f}")
    # Black and white mate-in-1 are mirror images, so they must agree.
    gap = abs(float(values[0]) - float(values[1]))
    report["mirror_gap"] = gap
    if gap > 0.10:
        failures.append(f"mirror_gap={gap:.3f}")
    report["failures"] = failures
    status = "OK" if not failures else "FAIL"
    print(f"T099_SIGN_ORACLE_{status} sims={sims} "
          + " ".join(f"{k}={v['search_value']:+.3f}" for k, v in report.items()
                     if isinstance(v, dict))
          + f" mirror_gap={gap:.3f}", flush=True)
    if failures:
        raise RuntimeError(f"sign oracle failed: {failures}")
    return report


def null_control(wst_ckpt, run2_ckpt, sfens, batch):
    """097-run2's value tower is bit-identical to WS-T's, so its RAW value must
    be identical too. If it is not, the checkpoints or the loader are wrong.
    """
    wst_forward, wst_model, _ = load_model(str(wst_ckpt))
    run2_forward, run2_model, _ = load_model(str(run2_ckpt))
    wst_fn, run2_fn = raw_value_fn(wst_forward, wst_model), raw_value_fn(run2_forward, run2_model)
    diffs = []
    for start in range(0, len(sfens), batch):
        state = sfens_to_state(sfens[start:start + batch])
        diffs.append(np.abs(np.asarray(wst_fn(state.observation))
                            - np.asarray(run2_fn(state.observation))))
    worst = float(np.concatenate(diffs).max())
    # Not exactly 0: the two checkpoints hold the same weights but the forwards
    # are separately jitted, and 091 recorded up to ~1cp of XLA rounding between
    # graphs for identical weights. In value units that is ~1e-6.
    status = "OK" if worst <= 1e-5 else "FAIL"
    print(f"T099_NULL_CONTROL_{status} max_abs(raw_wst - raw_run2)={worst:.3e}", flush=True)
    if worst > 1e-5:
        raise RuntimeError(f"097-run2 raw value differs from WS-T by {worst:.3e}")
    return {"max_abs_raw_diff": worst}


# ---- gold ----------------------------------------------------------------------
def load_gold(tsv_path, sfen_path, gold_depth=16):
    """Task 070's label_cost TSV -> {pos_id: cp_gold} over the usable subset.

    Same exclusion rule as `trainer/gold_analyze.py`: a position is dropped if it
    is mate or capped at ANY depth, so the surviving set is identical to the
    n=2882 that Task 070 reported.
    """
    rows, depths = {}, set()
    header = None
    for line in Path(tsv_path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if header is None:
            header = fields
            continue
        row = dict(zip(header, fields))
        pos_id, depth = int(row["pos_id"]), int(row["depth"])
        depths.add(depth)
        rows[(pos_id, depth)] = row
    sfens = Path(sfen_path).read_text(encoding="utf-8").splitlines()
    bad = set()
    for (pos_id, _depth), row in rows.items():
        if row["is_mate"] == "1" or row["capped"] == "1" or row["cp"] == "":
            bad.add(pos_id)
    usable = [pos_id for pos_id in range(len(sfens))
              if pos_id not in bad and all((pos_id, d) in rows for d in depths)]
    gold = {pos_id: float(rows[(pos_id, gold_depth)]["cp"]) for pos_id in usable}
    ladder = {
        f"d{d}": float(np.median([abs(float(rows[(p, d)]["cp"]) - gold[p]) for p in usable]))
        for d in sorted(depths) if d != gold_depth
    }
    return {"sfens": [sfens[p] for p in usable], "pos_ids": usable,
            "gold_cp": np.array([gold[p] for p in usable]),
            "n_usable": len(usable), "n_total": len(sfens), "depth_ladder": ladder}


def paired_median_delta(a_abs, b_abs, *, label="", samples=BOOTSTRAP_SAMPLES):
    """Bootstrap CI for median|a - gold| - median|b - gold|, positions resampled.

    Positions are the resampling unit and both arms are indexed with the SAME
    draw, so the shared position difficulty cancels -- the same pairing idiom the
    rest of the arc uses for openings.
    """
    a_abs, b_abs = np.asarray(a_abs), np.asarray(b_abs)
    n = len(a_abs)
    point = float(np.median(a_abs) - np.median(b_abs))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(samples)
    chunk = 5_000
    for start in range(0, samples, chunk):
        end = min(start + chunk, samples)
        idx = rng.integers(0, n, size=(end - start, n))
        deltas[start:end] = np.median(a_abs[idx], axis=1) - np.median(b_abs[idx], axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"label": label, "delta_median": point, "ci95": [float(low), float(high)],
            "n_positions": int(n), "bootstrap_seed": BOOTSTRAP_SEED}


def gold_closeness(gold, labels):
    """median |cp_labeller - cp_gold| per labeller, plus the paired contrasts."""
    abs_err = {name: np.abs(value_to_cp(values) - gold["gold_cp"])
               for name, values in labels.items()}
    medians = {name: float(np.median(err)) for name, err in abs_err.items()}
    contrasts = {
        "d31": paired_median_delta(abs_err["r3"], abs_err["r1"], label="r3_minus_r1"),
        "d32": paired_median_delta(abs_err["r3"], abs_err["r2"], label="r3_minus_r2"),
        "d21": paired_median_delta(abs_err["r2"], abs_err["r1"], label="r2_minus_r1"),
    }
    return {"median_abs_err": medians, "contrasts": contrasts,
            "n_positions": gold["n_usable"]}


# ---- stage 2: distribution statistics on a run2m_c subsample -------------------
def sample_run2m(run_dir, n_positions, seed):
    """Deterministic subsample of (sfen, gikou_cp) from the 2M teacher corpus."""
    shards, _ = shard_files(str(run_dir))
    rng = np.random.default_rng(seed)
    per_shard = n_positions // len(shards) + 1
    board = cshogi.Board()
    sfens, gikou = [], []
    for shard in shards:
        entries = np.fromfile(os.path.join(run_dir, shard["file"]),
                              dtype=cshogi.HuffmanCodedPosAndEval)
        take = rng.choice(len(entries), size=min(per_shard, len(entries)), replace=False)
        for index in np.sort(take):
            board.set_hcp(np.asarray(entries[index]["hcp"]))
            sfens.append(board.sfen())
            gikou.append(int(entries[index]["eval"]))
            if len(sfens) >= n_positions:
                return sfens, np.array(gikou, dtype=np.float64)
    return sfens, np.array(gikou, dtype=np.float64)


def label_stats(cp, gikou_clipped, reference_cp=None):
    stats = {"cp_std": float(np.std(cp)),
             "corr_gikou": float(np.corrcoef(cp, gikou_clipped)[0, 1]),
             "abs_median_vs_gikou": float(np.median(np.abs(cp - gikou_clipped)))}
    if reference_cp is not None:
        stats["abs_median_vs_reference"] = float(np.median(np.abs(cp - reference_cp)))
        stats["corr_reference"] = float(np.corrcoef(cp, reference_cp)[0, 1])
    return stats


# ---- verdict -------------------------------------------------------------------
def analyse(gold_primary, gold_holdout, stage2, s_selection, bench, oracle,
            null_ctl, sims_curve, prerequisites):
    contrasts = gold_primary["contrasts"]
    d31, d32, d21 = contrasts["d31"], contrasts["d32"], contrasts["d21"]
    shell_exit = stage2["r3_vs_r1"]["abs_median_vs_reference"]

    if prerequisites["failures"]:
        verdict, terminal = "INVALID", False
    elif d31["ci95"][1] < 0 and d32["ci95"][1] < 0:
        verdict, terminal = "BREACH", True
    elif d21["ci95"][1] < 0 and d32["ci95"][0] <= 0 <= d32["ci95"][1]:
        verdict, terminal = "SEARCH_ONLY", True
    elif d31["ci95"][0] > 0:
        verdict, terminal = "REGRESSION", True
    elif shell_exit < SHELL_EXIT_MEDIAN_CP:
        verdict, terminal = "NO_BREACH", True
    else:
        verdict, terminal = "INCONCLUSIVE", True

    flags = []
    if gold_holdout is not None:
        holdout_d31 = gold_holdout["contrasts"]["d31"]
        same_sign = (holdout_d31["ci95"][1] < 0) == (d31["ci95"][1] < 0)
        if not same_sign:
            flags.append("GOLD_SETS_DISAGREE")
    if s_selection["status"] != "selected":
        flags.append(f"S_{s_selection['status'].upper()}")
    return {
        "task": 99,
        "preregistration": {
            "labellers": {"r1": "WS-T raw", "r1p": "097-run2 raw (null control)",
                          "r2": "WS-T + search", "r3": "097-run2 + search"},
            "why_clean": ("097-run2 came from a frozen-value loop, so its value tower is "
                          "bit-identical to WS-T's: r3-r2 isolates the policy's effect "
                          "through search, with no value-quality confound"),
            "sims_ladder": list(SIMS_LADDER),
            "s_rule": (f"smallest sims whose median |dcp(search,raw)| clears the measured "
                       f"implementation-noise floor and {S_SELECT_MIN_DIVERGENCE_CP} cp; "
                       f"mechanism only, gold never consulted"),
            "primary": "median |cp - cp_gold| on Task 070 gold, paired position bootstrap",
            "decision_table": DECISION_TABLE,
            "shell_exit_median_cp": SHELL_EXIT_MEDIAN_CP,
            "gold_is": ("champion NNUE task064_32m_lr1e-5 @ d16, NOT gikou@d16; "
                        "98.17% of its positions are in the champion's training shards "
                        "and 6.30% are exact m4_data train rows, but the contamination is "
                        "symmetric across the three WS-T-derived arms"),
            "t070_baseline": T070_BASELINE,
            "t098_shell": T098_SHELL,
        },
        "sign_oracle": oracle,
        "null_control": null_ctl,
        "bench": bench,
        "s_selection": s_selection,
        "gold_primary": gold_primary,
        "gold_holdout": gold_holdout,
        "sims_robustness_curve": sims_curve,
        "stage2": stage2,
        "prerequisites": prerequisites,
        "flags": flags,
        "terminal": terminal,
        "verdict": verdict,
    }


def report(results):
    gold = results["gold_primary"]
    print(f"GOLD medians(cp) {({k: round(v, 1) for k, v in gold['median_abs_err'].items()})} "
          f"n={gold['n_positions']}", flush=True)
    for name, entry in gold["contrasts"].items():
        print(f"CONTRAST {name} {entry['label']} delta_median={entry['delta_median']:+.2f} "
              f"ci={[round(v, 2) for v in entry['ci95']]}", flush=True)
    if results["gold_holdout"]:
        hold = results["gold_holdout"]
        print(f"HOLDOUT medians(cp) {({k: round(v, 1) for k, v in hold['median_abs_err'].items()})} "
              f"d31={hold['contrasts']['d31']['delta_median']:+.2f} "
              f"ci={[round(v, 2) for v in hold['contrasts']['d31']['ci95']]}", flush=True)
    for name, stats in results["stage2"].items():
        print(f"STAGE2 {name} {({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})}",
              flush=True)
    print(f"S_SELECTION {results['s_selection']}", flush=True)
    print(f"FLAGS {results['flags']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task099")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    teacher = Path(os.environ.get("TEACHER_DIR", data_root / "teacher"))
    wst = clone / "t087_wst/ws2.ckpt"
    run2 = models / "task097_r050s2_ckpt/000025.ckpt"
    gold_tsv = logs / "task070_gold.tsv"
    gold_sfens = logs / "task070_gold.tsv.sfens"
    holdout_tsv = logs / f"{args.run_tag}_holdout_gold.tsv"
    work = data_root / "work" / args.run_tag

    if args.dry_run:
        print(f"DRYRUN work={work}")
        print(f"DRYRUN labellers r1=WS-T raw, r1'=097run2 raw (null control), "
              f"r2=WS-T+search, r3=097run2+search")
        print(f"DRYRUN sims ladder {SIMS_LADDER}, bench {BENCH_POSITIONS} positions")
        print(f"DRYRUN S rule: smallest sims with median|dcp(search,raw)| > "
              f"max(noise floor, {S_SELECT_MIN_DIVERGENCE_CP}cp)")
        print(f"DRYRUN gold primary={gold_tsv.name} (champion@d16, expect n=2882)")
        print(f"DRYRUN gold holdout={'skipped' if args.skip_holdout else holdout_tsv.name}")
        print(f"DRYRUN stage2 {STAGE2_POSITIONS} positions from run2m_c "
              f"({STAGE2_POSITIONS_BIG_SIMS} if S={BIG_SIMS})")
        print(f"DRYRUN decision table: {DECISION_TABLE}")
        for sims in SIMS_LADDER:
            rate = 1000.0 / ((sims + 1) * 0.178)
            print(f"DRYRUN est sims={sims}: {rate:.0f} pos/s -> bench {BENCH_POSITIONS/rate/60:.1f}min, "
                  f"gold(2x3000) {6000/rate/60:.1f}min, stage2(2x200k) {400000/rate/3600:.2f}h, "
                  f"full 2M(x1) {2000000/rate/3600:.2f}h")
        return 0

    # ---- stage 0a: preflight, sign oracle, null control ---------------------
    for name in ("m2_usi.py", "m2_distill_data.py", "az_rescore.py", "t099_search_label.py"):
        source = repo / "experiments/alphazero" / name
        linked = clone / name
        if not linked.exists() or not os.path.samefile(source, linked):
            raise RuntimeError(f"clone {name} is not linked to experiments SSOT")
    for path in (wst, run2, gold_tsv, gold_sfens, teacher / "run2m_c"):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    gold = load_gold(gold_tsv, gold_sfens)
    if gold["n_usable"] != T070_BASELINE["n_full"]:
        raise RuntimeError(f"gold usable {gold['n_usable']} != Task 070's "
                           f"{T070_BASELINE['n_full']}")
    print(f"T099_GOLD_LOADED n={gold['n_usable']}/{gold['n_total']} "
          f"depth_ladder={gold['depth_ladder']} (Task 070: {T070_BASELINE})", flush=True)

    wst_forward, wst_model, _ = load_model(str(wst))
    run2_forward, run2_model, _ = load_model(str(run2))
    oracle = sign_oracle(wst_forward, wst_model, SIMS_LADDER[0])
    null_ctl = null_control(wst, run2, gold["sfens"][:SEARCH_BATCH], SEARCH_BATCH)
    if args.self_test:
        print("T099_SELFTEST_OK", flush=True)
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    work.mkdir(parents=True, exist_ok=True)
    print(f"T099_PREFLIGHT_OK head={head} exp_dirty=0", flush=True)

    if not args.skip_cargo:
        subprocess.run(["cargo", "test", "--release"], cwd=repo, check=True,
                       stdout=subprocess.DEVNULL)
        print("T099_CARGO_TEST_OK", flush=True)

    # ---- stage 0b: bench, noise floor, S selection --------------------------
    bench_sfens = gold["sfens"][:BENCH_POSITIONS]
    bench = {}
    for sims in SIMS_LADDER:
        batch = SEARCH_BATCH_BIG_SIMS if sims >= BIG_SIMS else SEARCH_BATCH
        search = build_batched_search(wst_forward, wst_model, sims)
        score_sfens(search, bench_sfens[:batch], batch)          # JIT warm
        started = time.time()
        values, visit_q, raw, max_visits = score_sfens(search, bench_sfens, batch)
        elapsed = time.time() - started
        # gumbel_scale=0 makes the search deterministic, so a seed-based floor is
        # identically zero. The floor that does exist is the implementation's:
        # re-running the same input through the same graph (091 recorded ~1cp of
        # XLA rounding between graphs for identical weights).
        repeats = [score_sfens(search, bench_sfens[:batch], batch)[0]
                   for _ in range(NOISE_FLOOR_REPEATS)]
        floor_cp = float(np.max([np.max(np.abs(value_to_cp(r) - value_to_cp(repeats[0])))
                                 for r in repeats[1:]])) if NOISE_FLOOR_REPEATS > 1 else 0.0
        divergence = float(np.median(np.abs(value_to_cp(values) - value_to_cp(raw))))
        bench[str(sims)] = {
            "median_divergence_cp": divergence,
            "noise_floor_cp": floor_cp,
            "positions_per_s": len(bench_sfens) / elapsed,
            "batch": batch,
            "mean_max_visits": float(np.mean(max_visits)),
            "search_vs_visitq_median_cp": float(
                np.median(np.abs(value_to_cp(values) - value_to_cp(visit_q)))),
            "clears_floor": divergence > max(floor_cp, 1e-9) * 2,
            "clears_30cp": divergence > S_SELECT_MIN_DIVERGENCE_CP,
        }
        print(f"T099_BENCH sims={sims} divergence={divergence:.1f}cp floor={floor_cp:.3f}cp "
              f"rate={bench[str(sims)]['positions_per_s']:.0f}/s "
              f"max_visits={bench[str(sims)]['mean_max_visits']:.1f} "
              f"search_vs_visitq={bench[str(sims)]['search_vs_visitq_median_cp']:.1f}cp", flush=True)
    qualifying = [s for s in SIMS_LADDER
                  if bench[str(s)]["clears_floor"] and bench[str(s)]["clears_30cp"]]
    if qualifying:
        selected, status = qualifying[0], "selected"
    else:
        selected, status = SIMS_LADDER[-1], "ladder_insufficient"
    s_selection = {"selected": selected, "status": status, "rule": DECISION_TABLE,
                   "bench": bench}
    print(f"T099_S_SELECTED sims={selected} status={status}", flush=True)
    print("T099_BENCH_DONE", flush=True)

    # ---- stage 1: gold ------------------------------------------------------
    def label_all(sfens, sims, batch):
        wst_search = build_batched_search(wst_forward, wst_model, sims)
        run2_search = build_batched_search(run2_forward, run2_model, sims)
        r2_value, _, r1_value, _ = score_sfens(wst_search, sfens, batch)
        r3_value, _, r1p_value, _ = score_sfens(run2_search, sfens, batch)
        return {"r1": r1_value, "r1p": r1p_value, "r2": r2_value, "r3": r3_value}

    batch = SEARCH_BATCH_BIG_SIMS if selected >= BIG_SIMS else SEARCH_BATCH
    gold_labels = label_all(gold["sfens"], selected, batch)
    gold_primary = gold_closeness(gold, gold_labels)
    gold_primary["sims"] = selected
    # Free robustness check: the 189 gold positions that are exact m4_data train
    # rows are the only ones the AZ nets could have memorised.
    print("T099_GOLD_DONE", flush=True)

    sims_curve = {}
    for sims in SIMS_LADDER:
        if sims == selected:
            sims_curve[str(sims)] = gold_primary["contrasts"]
            continue
        curve_batch = SEARCH_BATCH_BIG_SIMS if sims >= BIG_SIMS else SEARCH_BATCH
        curve_labels = label_all(gold["sfens"], sims, curve_batch)
        sims_curve[str(sims)] = gold_closeness(gold, curve_labels)["contrasts"]

    gold_holdout = None
    if not args.skip_holdout and holdout_tsv.exists():
        holdout = load_gold(holdout_tsv, Path(str(holdout_tsv) + ".sfens"))
        holdout_labels = label_all(holdout["sfens"], selected, batch)
        gold_holdout = gold_closeness(holdout, holdout_labels)
        gold_holdout["sims"] = selected
        gold_holdout["n_usable"] = holdout["n_usable"]

    # ---- stage 2: distribution statistics -----------------------------------
    n_stage2 = STAGE2_POSITIONS_BIG_SIMS if selected >= BIG_SIMS else STAGE2_POSITIONS
    sfens, gikou = sample_run2m(teacher / "run2m_c", n_stage2, STAGE2_SEED)
    stage2_labels = label_all(sfens, selected, batch)
    gikou_clipped = np.clip(gikou, -CLAMP_CP, CLAMP_CP)
    cp = {name: value_to_cp(values) for name, values in stage2_labels.items()}
    stage2 = {name: label_stats(cp[name], gikou_clipped,
                                reference_cp=None if name == "r1" else cp["r1"])
              for name in ("r1", "r2", "r3")}
    stage2["r3_vs_r1"] = {
        "abs_median_vs_reference": float(np.median(np.abs(cp["r3"] - cp["r1"]))),
        "corr": float(np.corrcoef(cp["r3"], cp["r1"])[0, 1]),
        "n_positions": n_stage2,
    }
    stage2["r2_vs_r3"] = {
        "abs_median": float(np.median(np.abs(cp["r2"] - cp["r3"]))),
        "corr": float(np.corrcoef(cp["r2"], cp["r3"])[0, 1]),
    }
    print("T099_STAGE2_DONE", flush=True)

    results = analyse(gold_primary, gold_holdout, stage2, s_selection, bench,
                      oracle, null_ctl, sims_curve,
                      {"failures": [], "gold_n": gold["n_usable"]})
    results["git"] = {"head": head, "exp_dirty": 0, "pushed": False}
    results["full_2m_estimate_hours"] = (
        2_000_000 / bench[str(selected)]["positions_per_s"] / 3600.0)
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    print(f"T099_PIPELINE_DONE verdict={results['verdict']} "
          f"terminal={results['terminal']} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
