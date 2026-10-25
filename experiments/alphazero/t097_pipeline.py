#!/usr/bin/env python3
"""Task 097: does the r=0.50 RECIPE produce positive transfer, or did one run?

Task 096 established `POSITIVE_TRANSFER_CONFIRMED` for r=0.50 vs WS-T (+28.65,
CI [+3.79, +53.51], 1,224 clean pairs). Its limitation ④: `config.seed` drives
self-play only, so each arm is a SINGLE training trajectory and the CI covers
match noise, not training-seed variance. The claim was licensed only as "THIS net
beats its own initialisation".

This task trains a second net changing ONLY the training seeds and asks whether
the claim can be promoted to the recipe.

PRE-REGISTERED (committed locally before the training run; see the plan):
  * one variable = `seed` and `anchor_seed`, 20260714 -> 20269700. Everything
    else identical to the Task 094 r=0.50 arm. Fresh ckpt_dir (the seeds are not
    in assert_resume_config's mutable set).
  * primary = pooled two-run paired ΔElo: run1's 1,224 clean pairs (096, blocks
    2-11) + run2's 1,280 pairs (blocks 12-21) = 2,504 pairs. Predicted
    half-width ±17.4.
  * run2 alone is NOT required to be significant (winner's curse would otherwise
    manufacture false replication failures in the true-effect band +15…+25).
  * decision table, evaluated in precedence order (see DECISION_TABLE below).
  * DISCLOSED ERROR RATES: row 3 is effectively a sign test on Δ̂₂. With
    SE₂≈12.4 the terminal NOT_REPLICATED fires wrongly with probability 5.4% /
    11.3% / 21% at true effects +20 / +15 / +10. The INCONCLUSIVE band is only
    Δ̂₂ ∈ (0, +6.7] — about 7 Elo wide.
  * REPLICATED licenses only the qualitative part of limitation ④ (the effect is
    not one trajectory's artifact). It does NOT give a recipe-level interval:
    between-seed variance has 1 df at k=2 and is not estimable, so no
    random-effects interval is computed. See SCOPE_STATEMENT.
  * effect-size headline is Δ̂₂ (the only selection-uncontaminated estimate of the
    recipe); the decision statistic is pooled, which inherits run1's
    "significant, therefore this task exists" conditioning at ~half weight.
  * between-run difference, heterogeneity and the trajectory comparison are
    DESCRIPTIVE ONLY and never enter the verdict.

ESTIMATOR CORRECTION (found while planning 097): `t096_pipeline.stratified_delta`
resamples blocks and THEN pairs within each drawn block, double-counting the
within-cluster sampling error that the drawn block means already carry. Measured
bootstrap SDs (20k resamples):

    data                          delta-method   pooled   cluster   two-stage
    real blocks 2-11                    12.641   12.738    11.954    17.396
    shuffled -> 10 blocks (no effect)   12.633   12.633     8.302    15.016

The two-stage estimator inflates by 1.19x on data with ZERO block structure, so
096's `STRATIFIED_DISAGREES_WITH_POOLED` — reported there as the result's main
weakness — is an artifact of the estimator, consistent with its own I²=0.0.
Pooled matches the delta method to 0.3%, so 096's verdict is unaffected.
The cluster bootstrap is NOT a drop-in replacement either: at k=10 its variance
estimate has 9 df and it came in 34% low on the null control. So Task 097 retires
the two-stage estimator, corroborates pooled with an analytic delta-method CI,
and reports the cluster bootstrap descriptively with its instability stated.
`t096_pipeline.py` is left byte-identical (artifact of record).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from t094_pipeline import elo_from_score, load_tsv, run_logged  # noqa: E402
from t095_gate import (  # noqa: E402
    BLOCK_SEEDS,
    assert_blocks_disjoint,
    block_pair_scores,
    paired_delta_general,
    self_test as gate_self_test,
)
from t095_pipeline import VALUE_DIGEST, complete_tsv, parse_iter_metrics  # noqa: E402
from t096_pipeline import heterogeneity, quarantine_partial  # noqa: E402

# ---- pre-registered constants -------------------------------------------------
RUN2_SEED = 20269700                 # both `seed` and `anchor_seed`
RUN1_CLEAN_BLOCKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
RUN2_BLOCKS = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
RUN2_BLOCK_GAMES = 256
TARGET_PAIRS_RUN2 = 1280
TARGET_PAIRS_POOLED = 2504
EXPECTED_HALF_WIDTH = 17.4
# Task 096's published primary -- the pooled estimate REUSES these pairs, so the
# judge must be shown not to have moved before anything is added to them.
RUN1_PRIMARY = (28.647362873260647, [3.7931765042756638, 53.507581070411106])
# Task 094 r=0.50 iteration 1, for the seed-change guard.
RUN1_SEED = 20260714
RUN1_ITER1 = {"episodes": 414.0, "visit_ce": 1.3001, "anchor_ce": 1.4772}
SEED_DIVERGENCE_TOL = 0.005          # the arc's REGRESSION_TOL
ANCHOR_UPDATES_PER_ITER = 256        # round(512 * 0.50)
ANCHOR_BATCH = 128
BOOTSTRAP_SAMPLES = 200_000
CLUSTER_SEED = 20269703
BETWEEN_RUN_SEED = 20269702
NULL_CONTROL_SAMPLES = 20_000
NOT_REPLICATED_ERROR_RATES = {"+20": 0.054, "+15": 0.113, "+10": 0.21, "0": 0.50}
INCONCLUSIVE_BAND = [0.0, 6.7]

SCOPE_STATEMENT = (
    "REPLICATED licenses: two independently seeded training runs of the r=0.50 "
    "recipe each produced a network beating its own initialisation WS-T "
    "externally, and their pooled advantage is the reported interval over the "
    "clean pairs. It does NOT license a confidence interval for the recipe's "
    "expected effect over training seeds: the pooled interval is conditional on "
    "the two realised networks, and between-seed variance has 1 df at k=2 and is "
    "not estimable (a random-effects interval would have half-width "
    "12.71*|d1-d2|/2, so none is computed). Limitation 4 of Task 096 is resolved "
    "qualitatively (the effect is not one trajectory's artifact) but NOT "
    "quantitatively (a recipe-level effect size would need k>=5 runs)."
)

# Run 1's final-5-iteration behaviour, for the descriptive trajectory tripwire.
RUN1_TRAJECTORY_BAND = {
    "visit_ce": (0.9550, 0.0122), "visit_entropy": (0.6974, 0.0097),
    "visit_argmax_agree": (0.7868, 0.0039), "anchor_ce": (1.3849, 0.0075),
    "anchor_top1": (0.5568, 0.0016), "episodes": (495.6, 10.3),
}
# Run 1's OVERFIT_SUSPECTED margins -- it escapes by 0.0005 on one arm, so the
# flag is a coin toss for r=0.50 and must be printed with run1's numbers beside it.
RUN1_OVERFIT_MARGINS = {"val_top1_delta": -0.0015, "anchor_top1_delta": +0.0129}

DECISION_TABLE = (
    "0 prerequisites failed -> INSUFFICIENT_SAMPLE (non-terminal) | "
    "1 pooled CI low > 0 -> REPLICATED | "
    "2 pooled CI high < 0 -> NOT_REPLICATED_REVERSED | "
    "3 d2 <= 0 and pooled straddles 0 -> NOT_REPLICATED | "
    "4 otherwise -> INCONCLUSIVE (non-terminal)"
)


def run2_block_order(block):
    """ABBA within each block so drift is not aligned with one series."""
    return ("wst", "r050s2") if block % 2 == 0 else ("r050s2", "wst")


def _elo_slope(score):
    return 400.0 / (math.log(10.0) * score * (1.0 - score))


def delta_method_ci(candidate_blocks, baseline_blocks, *, label=""):
    """Analytic CI for the paired Elo difference -- corroborates the bootstrap.

    Resampling-free, so it cannot share a bug with the bootstrap; on Task 096's
    data it agreed with the pooled bootstrap to 0.3%.
    """
    cand = np.concatenate([block_pair_scores(load_tsv(p)) for p in candidate_blocks])
    base = np.concatenate([block_pair_scores(load_tsv(p)) for p in baseline_blocks])
    n = len(cand)
    c_mean, b_mean = float(cand.mean()), float(base.mean())
    gc, gb = _elo_slope(c_mean), _elo_slope(b_mean)
    variance = (
        (gc * cand.std(ddof=1)) ** 2
        + (gb * base.std(ddof=1)) ** 2
        - 2.0 * float(np.cov(cand, base)[0, 1]) * gc * gb
    )
    se = math.sqrt(variance / n)
    point = elo_from_score(c_mean) - elo_from_score(b_mean)
    return {
        "label": label, "paired_delta_elo": point, "se": se,
        "ci95": [point - 1.96 * se, point + 1.96 * se],
        "half_width": 1.96 * se, "n_pairs": int(n), "estimator": "delta method",
    }


def cluster_delta(candidate_blocks, baseline_blocks, *, label="", samples=BOOTSTRAP_SAMPLES):
    """Cluster bootstrap: resample whole blocks, no within-block resampling.

    DESCRIPTIVE ONLY. At small k its variance estimate has k-1 df and it came in
    34% low on the shuffled null control, so it is reported but never decides.
    """
    cand = [block_pair_scores(load_tsv(p)) for p in candidate_blocks]
    base = [block_pair_scores(load_tsv(p)) for p in baseline_blocks]
    k = len(cand)
    c_sum = np.array([x.sum() for x in cand])
    b_sum = np.array([x.sum() for x in base])
    counts = np.array([len(x) for x in cand], dtype=np.float64)
    rng = np.random.default_rng(CLUSTER_SEED)
    picks = rng.integers(0, k, size=(samples, k))
    totals = counts[picks].sum(axis=1)
    c_scores = np.clip(c_sum[picks].sum(axis=1) / totals, 1e-9, 1 - 1e-9)
    b_scores = np.clip(b_sum[picks].sum(axis=1) / totals, 1e-9, 1 - 1e-9)
    deltas = (400.0 * np.log10(c_scores / (1 - c_scores))
              - 400.0 * np.log10(b_scores / (1 - b_scores)))
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = elo_from_score(float(np.concatenate(cand).mean())) - elo_from_score(
        float(np.concatenate(base).mean()))
    return {
        "label": label, "paired_delta_elo": point, "ci95": [float(low), float(high)],
        "n_blocks": k, "bootstrap_seed": CLUSTER_SEED, "bootstrap_samples": samples,
        "estimator": "cluster bootstrap (blocks only)",
        "caveat": f"variance estimate has {k - 1} df; ran 34% low on the null control",
    }


def between_run_delta(run1_pairs, run2_pairs, *, samples=BOOTSTRAP_SAMPLES):
    """Δ̂₂ − Δ̂₁ with an independent bootstrap on each run. DESCRIPTIVE ONLY.

    run1 and run2 use disjoint blocks and disjoint games, so the two estimates
    are independent and can be resampled separately.
    """
    rng = np.random.default_rng(BETWEEN_RUN_SEED)
    draws = []
    for (cand, base) in (run1_pairs, run2_pairs):
        n = len(cand)
        out = np.empty(samples)
        chunk = 10_000
        for start in range(0, samples, chunk):
            end = min(start + chunk, samples)
            idx = rng.integers(0, n, size=(end - start, n))
            cs = np.clip(cand[idx].mean(axis=1), 1e-9, 1 - 1e-9)
            bs = np.clip(base[idx].mean(axis=1), 1e-9, 1 - 1e-9)
            out[start:end] = (400.0 * np.log10(cs / (1 - cs))
                              - 400.0 * np.log10(bs / (1 - bs)))
        draws.append(out)
    diff = draws[1] - draws[0]
    low, high = np.percentile(diff, [2.5, 97.5])
    point = (elo_from_score(float(run2_pairs[0].mean())) - elo_from_score(float(run2_pairs[1].mean()))
             - (elo_from_score(float(run1_pairs[0].mean())) - elo_from_score(float(run1_pairs[1].mean()))))
    return {
        "label": "d2_minus_d1", "delta_elo": point, "ci95": [float(low), float(high)],
        "half_width": float(high - low) / 2, "excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_seed": BETWEEN_RUN_SEED,
        "reading": "|d1-d2| beyond the half-width means genuine seed sensitivity; "
                   "below it the data cannot distinguish the two runs",
    }


def null_control_selftest(logs):
    """Evidence that retiring the two-stage estimator is a correction, not taste.

    Shuffling the pairs into arbitrary blocks destroys any block structure, so the
    truth is the iid value. The two-stage estimator still inflates; pooled does not.
    """
    cand_paths, base_paths = run1_clean_paths(logs)
    cand = [block_pair_scores(load_tsv(p)) for p in cand_paths]
    base = [block_pair_scores(load_tsv(p)) for p in base_paths]
    flat_c, flat_b = np.concatenate(cand), np.concatenate(base)
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(flat_c))
    shuffled_c = np.array_split(flat_c[perm], 10)
    shuffled_b = np.array_split(flat_b[perm], 10)

    def boot_sd(kind, blocks_c, blocks_b, seed):
        gen = np.random.default_rng(seed)
        k = len(blocks_c)
        cc, bb = np.concatenate(blocks_c), np.concatenate(blocks_b)
        out = np.empty(NULL_CONTROL_SAMPLES)
        for t in range(NULL_CONTROL_SAMPLES):
            if kind == "pooled":
                idx = gen.integers(0, len(cc), len(cc))
                x, y = cc[idx], bb[idx]
            else:  # two-stage, as implemented in t096_pipeline
                picks = gen.integers(0, k, k)
                xs, ys = [], []
                for p in picks:
                    m = len(blocks_c[p])
                    idx = gen.integers(0, m, m)
                    xs.append(blocks_c[p][idx])
                    ys.append(blocks_b[p][idx])
                x, y = np.concatenate(xs), np.concatenate(ys)
            out[t] = elo_from_score(x.mean()) - elo_from_score(y.mean())
        return float(out.std(ddof=1))

    analytic = delta_method_ci(cand_paths, base_paths)["se"]
    pooled_sd = boot_sd("pooled", cand, base, 5)
    shuf_pooled = boot_sd("pooled", shuffled_c, shuffled_b, 6)
    shuf_two_stage = boot_sd("twostage", shuffled_c, shuffled_b, 7)

    if abs(pooled_sd - analytic) / analytic > 0.05:
        raise AssertionError(
            f"pooled bootstrap disagrees with the delta method: {pooled_sd:.3f} vs {analytic:.3f}")
    if shuf_two_stage / shuf_pooled < 1.15:
        raise AssertionError(
            f"two-stage inflation not reproduced on the null control: "
            f"{shuf_two_stage:.3f}/{shuf_pooled:.3f} = {shuf_two_stage/shuf_pooled:.2f}")
    result = {
        "delta_method_se": analytic, "pooled_se": pooled_sd,
        "shuffled_pooled_se": shuf_pooled, "shuffled_two_stage_se": shuf_two_stage,
        "two_stage_inflation_on_null": shuf_two_stage / shuf_pooled,
        "samples": NULL_CONTROL_SAMPLES, "passed": True,
    }
    print(f"T097_NULL_CONTROL_OK delta={analytic:.3f} pooled={pooled_sd:.3f} "
          f"null_pooled={shuf_pooled:.3f} null_two_stage={shuf_two_stage:.3f} "
          f"inflation={result['two_stage_inflation_on_null']:.2f}x", flush=True)
    return result


def assert_anchor_stream_changed(anchor_data, iterations=(1, 2, 3, 25)):
    """`anchor_seed` must actually change which rows the anchor batches draw.

    This replaces a statistical proxy that misfired on the first launch. The
    original guard required |Δanchor_ce| > 0.005 between run1 and run2 at
    iteration 1, borrowing the arc's REGRESSION_TOL. That tolerance is calibrated
    for detecting a CODE regression under an identical configuration; it is the
    wrong scale here. anchor_ce is a mean over 256*128 = 32,768 supervised
    samples, so two genuinely different draws from the same 2M pool differ by
    only ~0.012 (SD), and the observed 0.0045 sat comfortably inside that --
    i.e. the proxy had a large false-alarm rate by construction.

    The batch indices are a pure deterministic function of (anchor_seed,
    iteration), so comparing the index streams tests the mechanism itself with
    zero noise, and does so at preflight before any GPU time is spent. It is
    strictly stronger than the statistic it replaces: it cannot false-alarm, and
    it also cannot be fooled by a model-trajectory difference masquerading as an
    anchor-stream difference.
    """
    pool = int(np.load(Path(anchor_data) / "train_a.npy", mmap_mode="r").shape[0])
    size = ANCHOR_UPDATES_PER_ITER * ANCHOR_BATCH
    report_lines = []
    for iteration in iterations:
        run1 = np.random.default_rng(
            np.random.SeedSequence([RUN1_SEED, iteration])).choice(pool, size=size, replace=False)
        run2 = np.random.default_rng(
            np.random.SeedSequence([RUN2_SEED, iteration])).choice(pool, size=size, replace=False)
        if np.array_equal(run1, run2):
            raise RuntimeError(
                f"SEED_DID_NOT_CHANGE: anchor index stream identical at iteration "
                f"{iteration} for seeds {RUN1_SEED} and {RUN2_SEED}")
        positional = int((run1 == run2).sum())
        overlap = int(np.intersect1d(run1, run2).size)
        report_lines.append({"iteration": iteration, "positional_matches": positional,
                             "set_overlap": overlap,
                             "chance_overlap": size * size / pool})
    print(f"T097_ANCHOR_STREAM_OK pool={pool} size={size} "
          + " ".join(f"it{r['iteration']}:pos={r['positional_matches']},ovl={r['set_overlap']}"
                     for r in report_lines), flush=True)
    return {"pool": pool, "draw_size": size, "iterations": report_lines}


def run1_clean_paths(logs):
    """Task 096's clean candidate/baseline TSVs (blocks 2-11), unchanged."""
    logs = Path(logs)
    cand, base = [], []
    for block in RUN1_CLEAN_BLOCKS:
        seed = BLOCK_SEEDS[block]
        games = 200 if block in (2, 3) else 256
        prefix = "task095" if block in (2, 3) else "task096"
        cand.append(logs / f"{prefix}_r050_iter025_vs_gikou6_s128_b100_seed{seed}_n{games}.tsv")
        base.append(logs / f"task096_wst_vs_gikou6_s128_b100_seed{seed}_n{games}.tsv")
    return cand, base


def replay_096_check(logs):
    """The pooled primary reuses 096's pairs -- prove the judge has not moved."""
    cand, base = run1_clean_paths(logs)
    got = paired_delta_general(cand, base, label="run1_replay")
    want_point, want_ci = RUN1_PRIMARY
    if got["paired_delta_elo"] != want_point or got["paired_ci95"] != want_ci:
        raise AssertionError(
            f"096 replay mismatch: {got['paired_delta_elo']!r} {got['paired_ci95']!r} "
            f"!= {want_point!r} {want_ci!r}")
    print("T097_REPLAY_096_OK primary reproduces task096_results.json exactly", flush=True)
    return got


def analyse(tsv_path, monitor, trajectory, cargo_test, match_rates, prerequisites, logs):
    """Stage 6, isolated so the decision table can be exercised without a 15h run."""
    def paths(series, blocks):
        return [tsv_path[(b, series)] for b in blocks]

    run1_cand, run1_base = run1_clean_paths(logs)
    run2_cand = paths("r050s2", RUN2_BLOCKS)
    run2_base = paths("wst", RUN2_BLOCKS)

    primary = paired_delta_general(run1_cand + run2_cand, run1_base + run2_base,
                                   label="pooled_two_run_minus_wst")
    run2_alone = paired_delta_general(run2_cand, run2_base, label="run2_minus_wst")
    run1_alone = paired_delta_general(run1_cand, run1_base, label="run1_minus_wst")

    pooled_low, pooled_high = primary["paired_ci95"]
    d2 = run2_alone["paired_delta_elo"]

    # Decision table, precedence order. Compute every predicate so a conflict is
    # disclosed rather than hidden by short-circuiting.
    ok = not prerequisites["failures"]
    row1 = ok and pooled_low > 0.0
    row2 = ok and pooled_high < 0.0
    row3 = ok and d2 <= 0.0 and pooled_low <= 0.0 <= pooled_high
    flags = []
    if not ok:
        verdict, terminal = "INSUFFICIENT_SAMPLE", False
    elif row1:
        verdict, terminal = "REPLICATED", True
    elif row2:
        verdict, terminal = "NOT_REPLICATED_REVERSED", True
    elif row3:
        verdict, terminal = "NOT_REPLICATED", True
    else:
        verdict, terminal = "INCONCLUSIVE", False
    if sum((row1, row2, row3)) > 1:
        flags.append("RULE_CONFLICT")

    per_block = {}
    for series, blocks, cand_paths, base_paths in (
        ("r050", RUN1_CLEAN_BLOCKS, run1_cand, run1_base),
        ("r050s2", RUN2_BLOCKS, run2_cand, run2_base),
    ):
        for block, cand_path, base_path in zip(blocks, cand_paths, base_paths):
            entry = paired_delta_general([cand_path], [base_path], label=f"block{block}")
            low, high = entry["paired_ci95"]
            per_block[str(block)] = {
                "delta_elo": entry["paired_delta_elo"], "ci95": entry["paired_ci95"],
                "se": (high - low) / (2 * 1.96), "n_pairs": entry["n_pairs"],
                "run": 1 if series == "r050" else 2,
                "order": list(run2_block_order(block)) if series == "r050s2" else None,
            }
    run_level = {
        "1": {"delta_elo": run1_alone["paired_delta_elo"],
              "se": (run1_alone["paired_ci95"][1] - run1_alone["paired_ci95"][0]) / (2 * 1.96)},
        "2": {"delta_elo": d2,
              "se": (run2_alone["paired_ci95"][1] - run2_alone["paired_ci95"][0]) / (2 * 1.96)},
    }

    run1_pairs = (np.concatenate([block_pair_scores(load_tsv(p)) for p in run1_cand]),
                  np.concatenate([block_pair_scores(load_tsv(p)) for p in run1_base]))
    run2_pairs = (np.concatenate([block_pair_scores(load_tsv(p)) for p in run2_cand]),
                  np.concatenate([block_pair_scores(load_tsv(p)) for p in run2_base]))

    # Session-rate skew: automate the identification Task 096 made by hand, namely
    # that the slower series is slower BECAUSE it wins more and winning takes
    # longer. The strength ordering is a property of the SERIES, so it is tested
    # pooled; per block only the mechanical identities and the rate ratio are
    # checked. (Testing the ordering per block would fire on any block whose
    # effect happens to land negative, which is noise, not a confound.)
    rate_skew = {"per_block": {}}
    totals = {s: {"score": 0.0, "n": 0, "rate": []} for s in ("wst", "r050s2")}
    identity_ok = True
    for block in RUN2_BLOCKS:
        entry = {}
        for series in ("wst", "r050s2"):
            rows = load_tsv(tsv_path[(block, series)])
            scores = [row["score"] for row in rows]
            reasons = [row["reason"] for row in rows]
            rate = match_rates.get(f"block{block}_{series}")
            entry[series] = {
                "rate": rate, "score": sum(scores) / len(scores),
                # resign = gikou resigned = an AZ win; mate = the AZ side was mated.
                "resign_eq_wins": reasons.count("resign") == scores.count(1.0),
                "mate_eq_losses": reasons.count("mate") == scores.count(0.0),
            }
            totals[series]["score"] += sum(scores)
            totals[series]["n"] += len(scores)
            if rate:
                totals[series]["rate"].append(rate)
            if not (entry[series]["resign_eq_wins"] and entry[series]["mate_eq_losses"]):
                identity_ok = False
        rates = [entry[s]["rate"] for s in ("wst", "r050s2")]
        if all(rates):
            entry["ratio"] = abs(rates[0] - rates[1]) / max(rates)
            entry["above_threshold"] = entry["ratio"] > 0.15
            entry["slower"] = "wst" if rates[0] < rates[1] else "r050s2"
        rate_skew["per_block"][str(block)] = entry

    pooled_scores = {s: totals[s]["score"] / totals[s]["n"] for s in totals if totals[s]["n"]}
    pooled_rates = {s: (sum(totals[s]["rate"]) / len(totals[s]["rate"]))
                    for s in totals if totals[s]["rate"]}
    if len(pooled_rates) == 2 and len(pooled_scores) == 2:
        slower = min(pooled_rates, key=pooled_rates.get)
        stronger = max(pooled_scores, key=pooled_scores.get)
        # ABBA cross-check: the skew must have the same sign whichever series went
        # first, otherwise it is an ordering effect rather than a strength effect.
        signs = set()
        for block in RUN2_BLOCKS:
            entry = rate_skew["per_block"][str(block)]
            if "slower" in entry:
                signs.add((run2_block_order(block)[0], entry["slower"]))
        consistent = len({slow for _, slow in signs}) == 1
        rate_skew["pooled"] = {
            "rates": pooled_rates, "scores": pooled_scores,
            "slower_series": slower, "stronger_series": stronger,
            "slower_is_stronger": slower == stronger,
            "identity_resign_eq_wins_and_mate_eq_losses": identity_ok,
            "abba_sign_consistent": consistent,
        }
        explained = identity_ok and slower == stronger and consistent
        rate_skew["classification"] = ("EXPLAINED_BY_STRENGTH" if explained
                                       else "UNEXPLAINED")
        if not explained:
            flags.append("SESSION_RATE_SKEW_UNEXPLAINED")

    # Descriptive trajectory tripwire -- a crude misconfiguration detector, not a test.
    tripwire = {}
    if trajectory:
        last = max(trajectory)
        for metric, (mean, sd) in RUN1_TRAJECTORY_BAND.items():
            value = trajectory[last].get(metric)
            if value is None:
                continue
            inside = abs(value - mean) <= 4 * sd
            tripwire[metric] = {"run1_mean": mean, "run1_sd": sd, "run2": value,
                                "within_4sd": inside}
            if not inside:
                flags.append(f"TRAJECTORY_OUTSIDE_RUN1_BAND_{metric}")

    return {
        "task": 97,
        "preregistration": {
            "one_variable": f"training seeds only: seed=anchor_seed 20260714 -> {RUN2_SEED}",
            "primary": "pooled two-run paired dElo vs WS-T, contaminated block1 excluded",
            "run1_ref": {"task": 96, "delta": RUN1_PRIMARY[0], "ci": RUN1_PRIMARY[1],
                         "n_pairs": 1224},
            "run2_blocks": list(RUN2_BLOCKS), "run2_block_games": RUN2_BLOCK_GAMES,
            "target_pairs_run2": TARGET_PAIRS_RUN2,
            "target_pairs_pooled": TARGET_PAIRS_POOLED,
            "expected_half_width_elo": EXPECTED_HALF_WIDTH,
            "decision_table": DECISION_TABLE,
            "not_replicated_error_rates": NOT_REPLICATED_ERROR_RATES,
            "inconclusive_band_elo": INCONCLUSIVE_BAND,
            "descriptive_only": ["between_run", "heterogeneity", "cluster",
                                 "trajectory", "per_block", "rate_skew"],
        },
        "prerequisites": prerequisites,
        "primary": primary,
        "primary_delta_method": delta_method_ci(run1_cand + run2_cand, run1_base + run2_base,
                                                label="pooled_delta_method"),
        "run2_alone": run2_alone,
        "run1_alone": run1_alone,
        "run1_replay_exact": run1_alone["paired_delta_elo"] == RUN1_PRIMARY[0],
        "effect_size_headline": {
            "value": d2, "ci95": run2_alone["paired_ci95"],
            "why": "run2 is the only selection-uncontaminated estimate of the recipe; "
                   "run1 is conditioned on 096 having been significant and pooled "
                   "inherits that upward bias at about half weight",
        },
        "between_run": between_run_delta(run1_pairs, run2_pairs),
        "heterogeneity": {
            "run_level": dict(heterogeneity(run_level), df_note="k=2, df=1; I2 is not "
                              "interpretable as a magnitude here"),
            "run1_blocks": heterogeneity({k: v for k, v in per_block.items() if v["run"] == 1}),
            "run2_blocks": heterogeneity({k: v for k, v in per_block.items() if v["run"] == 2}),
            "all_blocks": dict(heterogeneity(per_block), caveat="conflates block "
                               "heterogeneity with the between-run difference; "
                               "reported for completeness only"),
        },
        "cluster_descriptive": cluster_delta(run1_cand + run2_cand, run1_base + run2_base,
                                             label="pooled_cluster"),
        "two_stage_estimator": "RETIRED -- mis-specified (double-counts within-cluster "
                               "error); see null_control in the preflight record",
        "per_block_delta": per_block,
        "session_rate_skew": rate_skew,
        "trajectory_tripwire": tripwire,
        "run2_monitor": monitor,
        "run2_trajectory": {str(k): v for k, v in trajectory.items()},
        "match_rates_games_per_min": match_rates,
        "cargo_test": cargo_test,
        "scope_limitation": SCOPE_STATEMENT,
        "limitation4_resolved": ("qualitatively" if verdict == "REPLICATED" else "no"),
        "terminal": terminal,
        "flags": flags,
        "verdict": verdict,
    }


def report(results):
    primary, run2, run1 = results["primary"], results["run2_alone"], results["run1_alone"]
    analytic = results["primary_delta_method"]
    print(f"GATE primary pooled n_pairs={primary['n_pairs']} "
          f"delta={primary['paired_delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in primary['paired_ci95']]}", flush=True)
    print(f"GATE primary_delta_method delta={analytic['paired_delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in analytic['ci95']]} hw={analytic['half_width']:.2f}",
          flush=True)
    print(f"GATE run2 (effect-size headline) delta={run2['paired_delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in run2['paired_ci95']]} n_pairs={run2['n_pairs']}",
          flush=True)
    print(f"GATE run1 delta={run1['paired_delta_elo']:+.2f} "
          f"replay_exact={results['run1_replay_exact']}", flush=True)
    between = results["between_run"]
    print(f"BETWEEN_RUN d2-d1={between['delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in between['ci95']]} "
          f"excludes_zero={between['excludes_zero']}", flush=True)
    print(f"CLUSTER (descriptive) {results['cluster_descriptive']['paired_delta_elo']:+.2f} "
          f"ci={[round(v, 2) for v in results['cluster_descriptive']['ci95']]}", flush=True)
    het = results["heterogeneity"]
    print(f"HETEROGENEITY run_level Q={het['run_level']['Q']} I2={het['run_level']['I2']} | "
          f"run1_blocks I2={het['run1_blocks']['I2']} | run2_blocks I2={het['run2_blocks']['I2']}",
          flush=True)
    print("PER_BLOCK " + " ".join(
        f"{b}:{e['delta_elo']:+.0f}" for b, e in results["per_block_delta"].items()), flush=True)
    print(f"FLAGS {results['flags']}", flush=True)
    print(f"LIMITATION4 resolved={results['limitation4_resolved']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task097")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    args = parser.parse_args()

    repo = Path(os.path.realpath(__file__)).parents[2]
    clone = Path.home() / "task074_frameworks/pgx/examples/alphazero"
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    logs = Path(os.environ.get("LOGS_DIR", data_root / "logs"))
    models = Path(os.environ.get("MODELS_DIR", data_root / "models"))
    teacher = Path(os.environ.get("TEACHER_DIR", data_root / "teacher"))
    torch_py = Path(os.environ.get("VENV_PY", data_root / "venv/bin/python"))
    pgx_py = Path.home() / "task074_frameworks/venv_pgx/bin/python"
    anchor_data = clone / "m4_data"
    wst = clone / "t087_wst/ws2.ckpt"
    ckpt_dir = models / f"{args.run_tag}_r050s2_ckpt"
    run2_ckpt = ckpt_dir / f"{args.iters:06d}.ckpt"
    work_root = data_root / "work" / args.run_tag
    run1_loop_log = logs / "task094_r050_loop_to_005.log"

    tsv_path = {}
    for block in RUN2_BLOCKS:
        for series in ("wst", "r050s2"):
            tag = "" if series == "wst" else f"_iter{args.iters:03d}"
            tsv_path[(block, series)] = logs / (
                f"{args.run_tag}_{series}{tag}_vs_gikou6_s128_b100_"
                f"seed{BLOCK_SEEDS[block]}_n{RUN2_BLOCK_GAMES}.tsv")
    schedule = [(b, s) for b in RUN2_BLOCKS for s in run2_block_order(b)]

    if args.dry_run:
        print(f"DRYRUN work_root={work_root} ckpt_dir={ckpt_dir}")
        print(f"DRYRUN one variable: seed=anchor_seed 20260714 -> {RUN2_SEED}")
        for block, series in schedule:
            print(f"  block{block:>3} seed={BLOCK_SEEDS[block]} {series:>7} "
                  f"games={RUN2_BLOCK_GAMES} order={run2_block_order(block)}")
        run2_pairs = len(RUN2_BLOCKS) * RUN2_BLOCK_GAMES // 2
        games = len(schedule) * RUN2_BLOCK_GAMES
        print(f"DRYRUN run2 pairs={run2_pairs} (target {TARGET_PAIRS_RUN2}) "
              f"pooled={1224 + run2_pairs} (target {TARGET_PAIRS_POOLED}) "
              f"predicted_half_width={EXPECTED_HALF_WIDTH}")
        print(f"DRYRUN new games={games} match_hours={games * 7.95 / 3600:.2f} "
              f"loop_hours={568 * args.iters / 3600:.2f} "
              f"total={games * 7.95 / 3600 + 568 * args.iters / 3600:.2f}h")
        print(f"DRYRUN decision table: {DECISION_TABLE}")
        print(f"DRYRUN not_replicated error rates: {NOT_REPLICATED_ERROR_RATES}")
        return 0

    # ---- stage 0: preflight (analysis checks first, so they work pre-commit) ----
    for name in ("t094_loop_anchored.py", "t094_pipeline.py", "t095_gate.py",
                 "t095_pipeline.py", "t096_pipeline.py", "t097_pipeline.py"):
        source = repo / "experiments/alphazero" / name
        linked = clone / name
        if not linked.exists() or not os.path.samefile(source, linked):
            raise RuntimeError(f"clone {name} is not linked to experiments SSOT")
    gate_self_test(logs)
    replay_096_check(logs)
    null_control = null_control_selftest(logs)
    anchor_stream = assert_anchor_stream_changed(anchor_data)
    if args.self_test:
        print("T097_SELFTEST_OK", flush=True)
        return 0

    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    for path in (pgx_py, torch_py, anchor_data / "train_obs.npy", wst, run1_loop_log):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    logs.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})

    existing = ckpt_dir / "000000.ckpt"
    if existing.exists():
        # Unpickling a checkpoint needs optax, which lives in the pgx venv, not
        # the torch venv this orchestrator runs under -- so read it out of process.
        # (The loop's own assert_resume_config would also catch a seed mismatch;
        # this just fails earlier and says why.)
        probe = subprocess.run(
            [str(pgx_py), "-c",
             "import pickle,sys,json;c=pickle.load(open(sys.argv[1],'rb'))['config'];"
             "print(json.dumps({'seed':c.get('seed'),'anchor_seed':c.get('anchor_seed')}))",
             str(existing)],
            capture_output=True, text=True, env=env)
        if probe.returncode != 0:
            raise RuntimeError(f"could not read {existing}: {probe.stderr.strip()[-400:]}")
        saved = json.loads(probe.stdout.strip().splitlines()[-1])
        if saved.get("seed") != RUN2_SEED or saved.get("anchor_seed") != RUN2_SEED:
            raise RuntimeError(f"{ckpt_dir} holds a run with seed {saved}, "
                               f"expected {RUN2_SEED}; use a fresh ckpt_dir")
        print(f"T097_CKPT_SEED_OK {saved}", flush=True)
    print(f"T097_PREFLIGHT_OK head={head} exp_dirty=0 seed={RUN2_SEED}", flush=True)

    # ---- stage 1: cargo test (fail before 15h, not after) ------------------
    cargo_test = {"status": "SKIPPED"}
    if not args.skip_cargo:
        cargo_log = logs / f"{args.run_tag}_cargo_test.log"
        run_logged(["cargo", "test", "--release"], cargo_log, cwd=repo, env=env)
        cargo_test = {"status": "PASS", "log": str(cargo_log)}
        print("T097_CARGO_TEST_OK", flush=True)

    def loop_command(target, ckpt_interval=5):
        return [
            pgx_py, "t094_loop_anchored.py",
            f"max_num_iters={target}", "num_simulations=32",
            "selfplay_batch_size=128", "states_per_iter=65536",
            "training_batch_size=128", "learning_rate=0.0001",
            f"init_ckpt={wst}", f"ckpt_dir={ckpt_dir}", "resume=auto",
            f"ckpt_interval={ckpt_interval}", "monitor_interval=5",
            "expected_iter_s=720", f"anchor_data_dir={anchor_data}",
            "anchor_ratio=0.5", f"seed={RUN2_SEED}", f"anchor_seed={RUN2_SEED}",
        ]

    # ---- stage 2: smoke + the seed-change guard ----------------------------
    smoke_log = logs / f"{args.run_tag}_run2_smoke.log"
    if not run2_ckpt.exists() and not (ckpt_dir / "000002.ckpt").exists():
        run_logged(loop_command(2, ckpt_interval=1), smoke_log, cwd=clone, env=env)
    if smoke_log.exists():
        text = smoke_log.read_text(encoding="utf-8")
        header = [line for line in text.splitlines() if line.startswith("t094: sims=")]
        expected = ("anchor_updates=256", "anchor_ratio=0.50", "visit_updates=256",
                    "selfplay=on", "force_selfplay=False")
        if not header or not all(token in header[0] for token in expected):
            raise RuntimeError(f"r=0.50 run2 loop header unexpected: {header}")
        config_line = [line for line in text.splitlines() if line.startswith("env_id=")]
        if not config_line or f"seed={RUN2_SEED}" not in config_line[0] \
                or f"anchor_seed={RUN2_SEED}" not in config_line[0]:
            raise RuntimeError(f"SEED_DID_NOT_CHANGE: config line is {config_line}")
        got = parse_iter_metrics(smoke_log, 1)
        # Self-play side: `seed` drives the MCTS scan, whose output (episode count,
        # visit CE) is coarse enough that a changed seed shows up plainly.
        selfplay_moved = (got.get("episodes") != RUN1_ITER1["episodes"]
                          or abs(got.get("visit_ce", 0) - RUN1_ITER1["visit_ce"]) > SEED_DIVERGENCE_TOL)
        if not selfplay_moved:
            raise RuntimeError(
                f"SEED_DID_NOT_CHANGE: iter1 self-play reproduces run1 "
                f"(episodes {got.get('episodes')} vs {RUN1_ITER1['episodes']}, "
                f"visit_ce {got.get('visit_ce')} vs {RUN1_ITER1['visit_ce']})")
        # Anchor side is NOT gated on anchor_ce: it is a mean over 32,768 samples,
        # so two genuinely different draws differ by only ~0.012 and a 0.005
        # threshold false-alarms. The mechanism is tested deterministically at
        # preflight instead (assert_anchor_stream_changed); this is descriptive.
        anchor_delta = abs(got.get("anchor_ce", 0) - RUN1_ITER1["anchor_ce"])
        if VALUE_DIGEST not in text:
            raise RuntimeError("frozen value digest missing from the smoke log")
        print(f"T097_RUN2_SMOKE_OK seed={RUN2_SEED} selfplay_moved={selfplay_moved} "
              f"anchor_ce_delta={anchor_delta:.4f} (descriptive; stream checked at preflight)",
              flush=True)

    # ---- stage 3: run2 loop -------------------------------------------------
    run2_log = logs / f"{args.run_tag}_run2_loop_to_{args.iters:03d}.log"
    if not run2_ckpt.exists():
        run_logged(loop_command(args.iters), run2_log, cwd=clone, env=env)
    print(f"T097_RUN2_LOOP_DONE iters={args.iters}", flush=True)

    # ---- stage 4: rescore ---------------------------------------------------
    cache = work_root / "cache_2m"
    baseline_rescore = work_root / "rescore_wst"
    run2_rescore = work_root / "rescore_run2"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "meta.json").exists():
        run_logged([torch_py, "-m", "trainer.prepare_data", teacher / "run2m_c", cache,
                    "--canonical-shared", "--val-shards", "4"],
                   logs / f"{args.run_tag}_prepare.log", cwd=repo, env=env)
    if not (baseline_rescore / "rescore_metrics.json").exists():
        run_logged([pgx_py, "t091_rescore.py", "--cache-dir", cache, "--ckpt", wst,
                    "--out-dir", baseline_rescore],
                   logs / f"{args.run_tag}_rescore_wst.log", cwd=clone, env=env)
    if not (run2_rescore / "rescore_metrics.json").exists():
        run_logged([pgx_py, "t091_rescore.py", "--cache-dir", cache, "--ckpt", run2_ckpt,
                    "--out-dir", run2_rescore,
                    "--reference-value-dir", baseline_rescore,
                    "--reference-policy-dir", baseline_rescore],
                   logs / f"{args.run_tag}_rescore_run2.log", cwd=clone, env=env)
    monitor = json.loads((run2_rescore / "rescore_metrics.json").read_text(encoding="utf-8"))
    print(f"T097_RUN2_RESCORE_OK digest_ok={monitor.get('value_model_digest') == VALUE_DIGEST}",
          flush=True)

    # ---- stage 5: matches (no Elo, no CI printed) ---------------------------
    series_ckpt = {"wst": wst, "r050s2": run2_ckpt}
    match_rates = {}
    for block, series in schedule:
        path = tsv_path[(block, series)]
        quarantine_partial(path, RUN2_BLOCK_GAMES)
        if complete_tsv(path, RUN2_BLOCK_GAMES):
            print(f"SKIP existing match TSV {path.name}", flush=True)
        else:
            started = time.time()
            run_logged([pgx_py, "m2_match.py",
                        "--az-ckpt", series_ckpt[series], "--az-sims", "128",
                        "--opponent", "gikou", "--games", str(RUN2_BLOCK_GAMES),
                        "--byoyomi", "100", "--opening-plies", "6", "--max-plies", "320",
                        "--seed", str(BLOCK_SEEDS[block]), "--tsv", path],
                       Path(str(path).replace(".tsv", ".console.log")), cwd=clone, env=env)
            rate = RUN2_BLOCK_GAMES / ((time.time() - started) / 60.0)
            match_rates[f"block{block}_{series}"] = rate
            print(f"T097_MATCH_RATE block={block} series={series} "
                  f"games_per_min={rate:.2f}", flush=True)
    print(f"T097_MATCHES_DONE pairs={len(RUN2_BLOCKS) * RUN2_BLOCK_GAMES // 2}", flush=True)

    # ---- stage 6: the gate, exactly once ------------------------------------
    trajectory = {}
    for iteration in range(5, args.iters + 1, 5):
        try:
            trajectory[iteration] = parse_iter_metrics(run2_log, iteration)
        except (ValueError, FileNotFoundError):
            continue

    failures = []
    for block, series in schedule:
        try:
            if not complete_tsv(tsv_path[(block, series)], RUN2_BLOCK_GAMES):
                failures.append(f"missing_tsv_block{block}_{series}")
        except RuntimeError as error:
            failures.append(str(error))
    if not run2_ckpt.exists():
        failures.append("run2_checkpoint_missing")
    if monitor.get("value_model_digest") != VALUE_DIGEST:
        failures.append("value_digest_mismatch")
    illegal = sum(
        1 for (_, _), path in tsv_path.items() if Path(path).exists()
        for row in load_tsv(path) if row["reason"].startswith(("illegal:", "engine_fail:")))
    if illegal:
        failures.append(f"illegal_or_engine_fail_{illegal}")
    prerequisites = {"failures": failures, "illegal_or_engine_fail": illegal,
                     "null_control": null_control, "anchor_stream": anchor_stream}

    for series, blocks in (("r050s2", RUN2_BLOCKS), ("wst", RUN2_BLOCKS)):
        assert_blocks_disjoint([tsv_path[(b, series)] for b in blocks])

    results = analyse(tsv_path, monitor, trajectory, cargo_test, match_rates,
                      prerequisites, logs)
    results["git"] = {"head": head, "exp_dirty": 0, "pushed": False,
                      "note": "pre-registration committed locally before launch; no "
                              "third-party timestamp -- the audit trail is the commit "
                              "DAG plus RUN_META HEAD recorded at launch"}
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    print(f"T097_PIPELINE_DONE verdict={results['verdict']} terminal={results['terminal']} "
          f"results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
