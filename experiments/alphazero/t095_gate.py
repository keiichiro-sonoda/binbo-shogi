#!/usr/bin/env python3
"""Task 095 gate: multi-block paired bootstrap over gikou match TSVs.

Generalises `t094_pipeline.paired_delta` along three axes that Task 095 needs:

  1. arbitrary game count (094 hard-asserted exactly 200 rows per series),
  2. several opening *blocks* concatenated into one estimate (each block is a
     separate `m2_match.py --seed S` run; openings depend only on the seed, so a
     new seed yields fresh opening pairs that can be pooled with the old ones),
  3. pairing ANY two series, not just candidate-vs-baseline.  Task 095's primary
     estimand is r=0.50 vs r=1.00 directly, where the WS-T baseline cancels.

The bootstrap procedure, constants and RNG call sequence are copied verbatim
from `t094_pipeline.paired_delta` so that a single 100-pair block reproduces the
published Task 094 numbers bit-for-bit -- `self_test()` asserts exactly that.
`load_tsv` / `elo_from_score` are imported rather than reimplemented so there is
only ever one TSV parser in the arc.

numpy only (no jax): runs under either venv, self-test finishes in seconds.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from t094_pipeline import elo_from_score, load_tsv

# Same constants as Task 094 -- required for the self-test to reproduce exactly.
BOOTSTRAP_SAMPLES = 200_000
BOOTSTRAP_SEED = 20260724
CHUNK = 10_000

# Pre-registered opening blocks. Blocks 1-3 are the Task 093/094/095 sets; blocks
# 4-11 are Task 096's decisive-measurement blocks; blocks 12-21 are Task 097's
# replication blocks. The 096/097 seeds read 2026-96-NN / 2026-97-NN, i.e. "task
# NNN, block NN" -- there is no month 96 or 97, so they cannot be misread as dates
# and cannot collide with the date-shaped seeds already in use.
BLOCK_SEEDS = {
    1: 20260702, 2: 20260729, 3: 20260730,
    4: 20269604, 5: 20269605, 6: 20269606, 7: 20269607,
    8: 20269608, 9: 20269609, 10: 20269610, 11: 20269611,
    12: 20269712, 13: 20269713, 14: 20269714, 15: 20269715, 16: 20269716,
    17: 20269717, 18: 20269718, 19: 20269719, 20: 20269720, 21: 20269721,
}
if len(set(BLOCK_SEEDS.values())) != len(BLOCK_SEEDS):
    raise AssertionError(f"duplicate opening-block seed in BLOCK_SEEDS: {BLOCK_SEEDS}")

# Two blocks that share openings would be double-counted as independent evidence.
# Coincidental overlap between genuinely different seeds is astronomically
# unlikely over 6-ply uniform-random openings, so a handful of shared lines is
# reported but tolerated; this many means the seeds are not actually distinct.
OPENING_OVERLAP_ABORT = 5


def _openings(rows):
    return tuple(row["opening"] for row in rows)


def validate_block(rows, path):
    """Structural checks on one series' block, independent of any comparison."""
    count = len(rows)
    if count < 2 or count % 2 != 0:
        raise ValueError(f"block must hold an even number of games >= 2: {path} ({count})")
    # `m2_match.py` opens the TSV in append mode but always restarts `g` at 0, so
    # an interrupted-then-rerun match yields duplicated game indices. A row-count
    # check alone (t094's ensure_complete_tsv) cannot see that; this can.
    games = [row["game"] for row in rows]
    if games != list(range(count)):
        raise ValueError(
            f"block game column is not 0..{count - 1} (appended or truncated TSV?): {path}"
        )
    for index in range(0, count, 2):
        if rows[index]["opening"] != rows[index + 1]["opening"]:
            raise ValueError(f"opening pair mismatch at rows {index}/{index + 1}: {path}")
        colors = (rows[index]["color"], rows[index + 1]["color"])
        if colors != ("b", "w"):
            raise ValueError(f"expected colors ('b','w') at rows {index}/{index+1}: {path} {colors}")


def validate_block_pair(a_rows, b_rows, a_path, b_path):
    """Both series must have played the same openings, same colours, same order."""
    if len(a_rows) != len(b_rows):
        raise ValueError(f"block size mismatch: {a_path} ({len(a_rows)}) vs {b_path} ({len(b_rows)})")
    for a_row, b_row in zip(a_rows, b_rows):
        for key in ("game", "color", "opening"):
            if a_row[key] != b_row[key]:
                raise ValueError(
                    f"paired structure mismatch for {key}: {a_path} {a_row} != {b_path} {b_row}"
                )


def assert_blocks_disjoint(paths):
    """No two blocks may draw from the same opening stream.

    Equality is not a sufficient test once blocks have different lengths: the
    openings come from `np.random.default_rng(seed)` consumed sequentially by
    m2_match.py, so the same seed run with --games 200 and --games 256 yields
    tuples where one is a strict PREFIX of the other. Task 095 only ever used
    200-game blocks, so equality sufficed there; Task 096 mixes sizes.
    """
    openings = [(path, _openings(load_tsv(path))) for path in paths]
    for index, (path_a, a) in enumerate(openings):
        for path_b, b in openings[index + 1:]:
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if longer[: len(shorter)] == shorter:
                raise ValueError(
                    "blocks draw from the same opening stream "
                    f"(identical or prefix-nested -- reused seed?): {path_a} and {path_b}"
                )
            shared = len(set(a) & set(b))
            if shared:
                print(
                    f"T096_OPENING_OVERLAP {Path(path_a).name} {Path(path_b).name} n={shared}",
                    flush=True,
                )
                if shared >= OPENING_OVERLAP_ABORT:
                    raise ValueError(
                        f"blocks share {shared} openings (>= {OPENING_OVERLAP_ABORT}): "
                        f"{path_a} and {path_b}"
                    )


def block_pair_scores(rows):
    """Mean of the two colour-flipped games -> one score per opening pair."""
    return np.asarray([
        (rows[index]["score"] + rows[index + 1]["score"]) / 2.0
        for index in range(0, len(rows), 2)
    ])


def paired_delta_general(
    candidate_blocks,
    baseline_blocks,
    *,
    label="",
    allow_identical_blocks=False,
):
    """Paired bootstrap of Elo(candidate) - Elo(baseline) over pooled opening pairs.

    `candidate_blocks` / `baseline_blocks` are equal-length sequences of TSV
    paths; block i of one series must share block i's openings with the other.
    Resampling unit is the opening pair; the SAME index draw is applied to both
    series, which is what removes the common opening-difficulty variance.
    """
    candidate_blocks = [Path(path) for path in candidate_blocks]
    baseline_blocks = [Path(path) for path in baseline_blocks]
    if not candidate_blocks:
        raise ValueError("at least one block is required")
    if len(candidate_blocks) != len(baseline_blocks):
        raise ValueError(
            f"block count mismatch: {len(candidate_blocks)} vs {len(baseline_blocks)}"
        )

    candidate_rows, baseline_rows = [], []
    candidate_pairs_by_block, baseline_pairs_by_block = [], []
    for cand_path, base_path in zip(candidate_blocks, baseline_blocks):
        cand = load_tsv(cand_path)
        base = load_tsv(base_path)
        validate_block(cand, cand_path)
        validate_block(base, base_path)
        validate_block_pair(cand, base, cand_path, base_path)
        candidate_rows.extend(cand)
        baseline_rows.extend(base)
        candidate_pairs_by_block.append(block_pair_scores(cand))
        baseline_pairs_by_block.append(block_pair_scores(base))

    # Distinct blocks must not share openings -- a reused --seed would silently
    # double-count the same positions as independent evidence.
    if not allow_identical_blocks:
        assert_blocks_disjoint(candidate_blocks)
        assert_blocks_disjoint(baseline_blocks)

    candidate_pairs = np.concatenate(candidate_pairs_by_block)
    baseline_pairs = np.concatenate(baseline_pairs_by_block)
    n_pairs = len(candidate_pairs)
    candidate_score = float(candidate_pairs.mean())
    baseline_score = float(baseline_pairs.mean())
    point = elo_from_score(candidate_score) - elo_from_score(baseline_score)

    # RNG call sequence identical to t094_pipeline.paired_delta -- do not
    # reorder, vectorise away the chunk loop, or switch to rng.choice.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, CHUNK):
        end = min(start + CHUNK, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, n_pairs, size=(end - start, n_pairs))
        cand_scores = candidate_pairs[indices].mean(axis=1)
        base_scores = baseline_pairs[indices].mean(axis=1)
        cand_scores = np.clip(cand_scores, 1e-9, 1.0 - 1e-9)
        base_scores = np.clip(base_scores, 1e-9, 1.0 - 1e-9)
        deltas[start:end] = (
            400.0 * np.log10(cand_scores / (1.0 - cand_scores))
            - 400.0 * np.log10(base_scores / (1.0 - base_scores))
        )
    low, high = np.percentile(deltas, [2.5, 97.5])

    scores = np.asarray([row["score"] for row in candidate_rows])
    blocks = []
    for path, cand_pairs, base_pairs in zip(
        candidate_blocks, candidate_pairs_by_block, baseline_pairs_by_block
    ):
        blocks.append({
            "candidate": str(path),
            "n_pairs": int(len(cand_pairs)),
            "block_delta_elo": elo_from_score(float(cand_pairs.mean()))
            - elo_from_score(float(base_pairs.mean())),
        })
    return {
        "label": label,
        "wins": int((scores == 1.0).sum()),
        "draws": int((scores == 0.5).sum()),
        "losses": int((scores == 0.0).sum()),
        "score": candidate_score,
        "baseline_score": baseline_score,
        "absolute_elo": elo_from_score(candidate_score),
        "paired_delta_elo": point,
        "paired_ci95": [float(low), float(high)],
        "n_games": int(len(candidate_rows)),
        "n_pairs": int(n_pairs),
        "n_blocks": int(len(candidate_blocks)),
        "blocks": blocks,
        "candidate_paths": [str(path) for path in candidate_blocks],
        "baseline_paths": [str(path) for path in baseline_blocks],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def sign_class(result):
    """Axis 1 of the Task 095 verdict table: is the paired CI clear of zero?"""
    low, high = result["paired_ci95"]
    if low > 0.0:
        return "POS"
    if high < 0.0:
        return "NEG"
    return "NULL"


def pairs_for_80pct_power(result):
    """Opening pairs needed for 80% power at the observed effect size.

    Half-width scales as 1/sqrt(n); 80% power at alpha=0.05 needs the effect to
    sit ~2.80 SE from zero rather than the 1.96 SE that bounds the CI.
    """
    point = result["paired_delta_elo"]
    low, high = result["paired_ci95"]
    half_width = (high - low) / 2.0
    if point == 0.0 or half_width <= 0.0:
        return None
    return int(np.ceil(result["n_pairs"] * (half_width / abs(point)) ** 2 * (2.80 / 1.96) ** 2))


# Published Task 094 values -- the regression target for the refactor.
T094_EXPECTED = {
    "r050": (79.12175914215098, [-11.06857063914336, 174.21020311016701]),
    "r025": (21.341868924478604, [-72.81715962980718, 118.59518534959426]),
}


def self_test(logs_dir):
    """Reproduce Task 094's published numbers exactly, plus plumbing checks."""
    logs_dir = Path(logs_dir)
    wst = [logs_dir / "task093_wst_vs_gikou6_s128_b100_seed20260702_n200.tsv"]
    r025 = [logs_dir / "task094_r025_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv"]
    r050 = [logs_dir / "task094_r050_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv"]
    for path in wst + r025 + r050:
        if not path.exists():
            raise FileNotFoundError(f"self-test needs the Task 093/094 TSV: {path}")

    results = {}
    for name, blocks in (("r050", r050), ("r025", r025)):
        got = paired_delta_general(blocks, wst, label=f"{name}_vs_wst_block1")
        want_point, want_ci = T094_EXPECTED[name]
        # Exact equality on purpose: the claim is bit-reproduction of a published
        # number. A NumPy upgrade that moves it is itself a finding (the judge
        # changed), so it must fail loudly rather than be absorbed by a tolerance.
        if got["paired_delta_elo"] != want_point:
            raise AssertionError(
                f"{name} paired_delta_elo regression: got {got['paired_delta_elo']!r} "
                f"want {want_point!r} (delta {got['paired_delta_elo'] - want_point:+.6e})"
            )
        if got["paired_ci95"] != want_ci:
            raise AssertionError(
                f"{name} paired_ci95 regression: got {got['paired_ci95']!r} want {want_ci!r}"
            )
        results[name] = got

    # Multi-block plumbing: duplicating a block must preserve the point estimate
    # and strictly narrow the CI. Exercises concat + n_pairs=200 without new games.
    duplicated = paired_delta_general(
        r050 * 2, wst * 2, label="r050_dup", allow_identical_blocks=True
    )
    if duplicated["n_pairs"] != 200 or duplicated["n_blocks"] != 2:
        raise AssertionError(f"duplicated-block shape wrong: {duplicated['n_pairs']}")
    if abs(duplicated["paired_delta_elo"] - results["r050"]["paired_delta_elo"]) > 1e-9:
        raise AssertionError("duplicating a block moved the point estimate")
    single = results["r050"]["paired_ci95"]
    if (duplicated["paired_ci95"][1] - duplicated["paired_ci95"][0]) >= (single[1] - single[0]):
        raise AssertionError("duplicating a block did not narrow the CI")

    # Point estimates are exactly additive (each is a difference of elo(mean)).
    direct = paired_delta_general(r050, r025, label="r050_vs_r025")
    residual = (
        results["r050"]["paired_delta_elo"]
        - results["r025"]["paired_delta_elo"]
        - direct["paired_delta_elo"]
    )
    if abs(residual) > 1e-9:
        raise AssertionError(f"point-estimate additivity violated: residual {residual:+.3e}")

    # The reused-seed guard must actually fire.
    try:
        paired_delta_general(r050 * 2, wst * 2, label="should_raise")
    except ValueError:
        pass
    else:
        raise AssertionError("identical-opening blocks were not rejected")

    # Task 096 additions -------------------------------------------------------
    # (a) seed registry is injective (also asserted at import).
    if len(set(BLOCK_SEEDS.values())) != len(BLOCK_SEEDS):
        raise AssertionError("BLOCK_SEEDS contains a duplicate seed")

    # (b) prefix-nested blocks must raise. Equality alone would miss this, and it
    #     is exactly the same-seed/different---games failure mode that becomes
    #     reachable once blocks have different lengths.
    with tempfile.TemporaryDirectory() as tmp:
        truncated = Path(tmp) / "truncated.tsv"
        rows = Path(r050[0]).read_text(encoding="utf-8").splitlines()
        truncated.write_text("\n".join(rows[:100]) + "\n", encoding="utf-8")
        try:
            assert_blocks_disjoint([r050[0], truncated])
        except ValueError as error:
            if "prefix-nested" not in str(error):
                raise AssertionError(f"wrong rejection reason: {error}") from error
        else:
            raise AssertionError("prefix-nested blocks were not rejected")

        # (c) unequal block sizes pool correctly: n_pairs adds up and the point
        #     estimate is the difference of the pooled-mean Elos.
        wst_truncated = Path(tmp) / "wst_truncated.tsv"
        wst_rows = Path(wst[0]).read_text(encoding="utf-8").splitlines()
        wst_truncated.write_text("\n".join(wst_rows[:100]) + "\n", encoding="utf-8")
        mixed = paired_delta_general(
            [r050[0], truncated], [wst[0], wst_truncated],
            label="mixed_sizes", allow_identical_blocks=True,
        )
        if mixed["n_pairs"] != 150 or mixed["n_games"] != 300:
            raise AssertionError(f"unequal-size pooling shape wrong: {mixed['n_pairs']}")
        pooled_cand = np.concatenate([
            block_pair_scores(load_tsv(r050[0])), block_pair_scores(load_tsv(truncated))
        ]).mean()
        pooled_base = np.concatenate([
            block_pair_scores(load_tsv(wst[0])), block_pair_scores(load_tsv(wst_truncated))
        ]).mean()
        expected = elo_from_score(float(pooled_cand)) - elo_from_score(float(pooled_base))
        if abs(mixed["paired_delta_elo"] - expected) > 1e-9:
            raise AssertionError(
                f"unequal-size point estimate wrong: {mixed['paired_delta_elo']} vs {expected}"
            )

    print(
        "T095_GATE_SELFTEST_OK "
        f"r050={results['r050']['paired_delta_elo']:.6f} "
        f"r025={results['r025']['paired_delta_elo']:.6f} "
        f"dup_pairs={duplicated['n_pairs']} additivity_residual={residual:+.2e} "
        f"blocks={len(BLOCK_SEEDS)} prefix_guard=ok mixed_sizes=ok",
        flush=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--logs-dir",
        default=os.environ.get(
            "LOGS_DIR", str(Path.home() / "binbo-shogi-data" / "logs")
        ),
    )
    args = parser.parse_args()
    if args.self_test:
        self_test(args.logs_dir)
        return 0
    parser.error("nothing to do: pass --self-test (this module is imported by t095_pipeline)")


if __name__ == "__main__":
    raise SystemExit(main())
