#!/usr/bin/env python3
"""Task 095: verify that skipping self-play at anchor_ratio=1.0 is a no-op.

At anchor_ratio=1.0 every optimizer update is a supervised anchor batch, so the
generated self-play data is never read.  `t094_loop_anchored.py` therefore skips
generation (~552s of a ~582s iteration).  The argument that this cannot change
the trained weights is:

  * anchor batches come from np.random.default_rng(SeedSequence([anchor_seed,
    iteration])) -- a stream keyed only on (seed, iteration), independent of the
    JAX rng_key that self-play splits;
  * self-play only READS `model`; it never writes params, state or opt_state.

That argument is sound on paper but assumes the GPU is bitwise reproducible,
which is NOT guaranteed (cuBLAS may pick different algorithms run to run).  The
Task 094 logs already show it isn't: the two arms' `monitor iter=0` lines differ
(top1 0.47166 vs 0.47158) while evaluating identical WS-T weights.

So this compares three runs, not two.  Run C is a plain replicate of run B; it
measures the machine's own reproducibility noise, which is what makes a
mismatch between A and B interpretable instead of merely alarming.

  A  T094_FORCE_SELFPLAY=1  generation ON
  B  (default)              generation skipped
  C  (default)              generation skipped, replicate

Verdicts, evaluated in order:
  EQUIV_EXACT       digest(A) == digest(B) at every iteration -> skip is exact
  EQUIV_NOISE_FLOOR A != B, but B != C too, and |A-B| <= max(|B-C|, 1e-6)
                    -> the machine is not bitwise reproducible and the A/B gap
                       is no larger than that noise; the skip is still sound
  EQUIV_FAIL        B == C (so the process IS reproducible) but A != B
                    -> a real coupling exists; do not use the skip

Exits nonzero on EQUIV_FAIL so a pipeline can branch on it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from t091_model import KIND, tree_digest

# The WS-T value tower digest, frozen for the whole 091/093/094/095 arc.
EXPECTED_VALUE_DIGEST = "2316885110a0e9532f83237160c9cdd8d8d2aa5d02b91995225f8e8d6520bbe1"


def load_model(ckpt_path):
    with open(ckpt_path, "rb") as file:
        ckpt = pickle.load(file)
    if ckpt.get("kind") != KIND:
        raise ValueError(f"not a frozen-value checkpoint: {ckpt_path} kind={ckpt.get('kind')!r}")
    return ckpt["model"]


def policy_tree(model):
    return (model[0]["policy"], model[1]["policy"])


def value_tree(model):
    return (model[0]["value"], model[1]["value"])


def max_abs_diff(tree_a, tree_b):
    """Largest absolute leaf-wise difference between two identically-shaped trees."""
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    if len(leaves_a) != len(leaves_b):
        raise ValueError(f"tree shape mismatch: {len(leaves_a)} vs {len(leaves_b)} leaves")
    worst = 0.0
    for leaf_a, leaf_b in zip(leaves_a, leaves_b):
        array_a = np.asarray(jax.device_get(leaf_a), dtype=np.float64)
        array_b = np.asarray(jax.device_get(leaf_b), dtype=np.float64)
        if array_a.shape != array_b.shape:
            raise ValueError(f"leaf shape mismatch: {array_a.shape} vs {array_b.shape}")
        if array_a.size:
            worst = max(worst, float(np.abs(array_a - array_b).max()))
    return worst


def compare(force_dir, skip_dir, skip2_dir, iterations):
    per_iteration = {}
    for iteration in iterations:
        name = f"{iteration:06d}.ckpt"
        models = {}
        for label, directory in (("force", force_dir), ("skip", skip_dir), ("skip2", skip2_dir)):
            path = Path(directory) / name
            if not path.exists():
                raise FileNotFoundError(f"equivalence checkpoint missing: {path}")
            models[label] = load_model(path)

        digests = {label: tree_digest(policy_tree(model)) for label, model in models.items()}
        value_digests = {label: tree_digest(value_tree(model)) for label, model in models.items()}
        for label, digest in value_digests.items():
            if digest != EXPECTED_VALUE_DIGEST:
                raise AssertionError(
                    f"value tower drifted in run {label} at iter {iteration}: "
                    f"{digest} != {EXPECTED_VALUE_DIGEST}"
                )
        per_iteration[iteration] = {
            "policy_digests": digests,
            "value_digest": value_digests["force"],
            "max_abs_force_vs_skip": max_abs_diff(
                policy_tree(models["force"]), policy_tree(models["skip"])
            ),
            "max_abs_skip_vs_skip2": max_abs_diff(
                policy_tree(models["skip"]), policy_tree(models["skip2"])
            ),
        }

    exact = all(
        entry["policy_digests"]["force"] == entry["policy_digests"]["skip"]
        for entry in per_iteration.values()
    )
    replicate_exact = all(
        entry["policy_digests"]["skip"] == entry["policy_digests"]["skip2"]
        for entry in per_iteration.values()
    )
    worst_force = max(entry["max_abs_force_vs_skip"] for entry in per_iteration.values())
    worst_noise = max(entry["max_abs_skip_vs_skip2"] for entry in per_iteration.values())

    if exact:
        status = "EQUIV_EXACT"
    elif not replicate_exact and worst_force <= max(worst_noise, 1e-6):
        status = "EQUIV_NOISE_FLOOR"
    else:
        status = "EQUIV_FAIL"

    return {
        "status": status,
        "iterations": sorted(per_iteration),
        "per_iteration": {str(key): value for key, value in per_iteration.items()},
        "max_abs_force_vs_skip": worst_force,
        "max_abs_skip_vs_skip2": worst_noise,
        "replicate_bitwise_identical": replicate_exact,
        "value_digest": EXPECTED_VALUE_DIGEST,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-dir", required=True, help="ckpt_dir of the T094_FORCE_SELFPLAY=1 run")
    parser.add_argument("--skip-dir", required=True, help="ckpt_dir of the self-play-skipped run")
    parser.add_argument("--skip2-dir", required=True, help="ckpt_dir of the skipped replicate (noise floor)")
    parser.add_argument("--iters", default="1,2", help="comma-separated iterations to compare")
    parser.add_argument("--out", default=None, help="optional path for the JSON result")
    args = parser.parse_args()

    iterations = [int(value) for value in args.iters.split(",") if value.strip()]
    result = compare(args.force_dir, args.skip_dir, args.skip2_dir, iterations)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"EQUIV_RESULT status={result['status']} "
        f"iters={result['iterations']} "
        f"max_abs_force_vs_skip={result['max_abs_force_vs_skip']:.3e} "
        f"max_abs_skip_vs_skip2={result['max_abs_skip_vs_skip2']:.3e} "
        f"replicate_bitwise_identical={result['replicate_bitwise_identical']} "
        f"value_digest={result['value_digest'][:12]}...",
        flush=True,
    )
    for iteration in result["iterations"]:
        entry = result["per_iteration"][str(iteration)]
        digests = entry["policy_digests"]
        print(
            f"  iter={iteration} force={digests['force'][:16]} "
            f"skip={digests['skip'][:16]} skip2={digests['skip2'][:16]}",
            flush=True,
        )
    return 1 if result["status"] == "EQUIV_FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
