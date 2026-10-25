#!/usr/bin/env python3
"""Task 096: decisive measurement of the r=0.50 mixture, plus the r=0.75 curve point.

Task 095 found an inverted-U dose response in the anchor ratio r (paired dElo vs
WS-T on opening block 1): r=0 -74.0, r=0.25 +21.3, r=0.50 +79.1, r=1.00 -7.6.
Pure supervised is exhausted, pure self-play forgets, only the mixture rises --
but the selection-free primary was NULL (+37.8, CI [-27.0,+103.5]).

PRE-REGISTERED (committed before any of this runs; see the plan and the report):
  * decisive arm = r=0.50, fixed. Phase A cannot change it.
  * primary estimand = r=0.50 - WS-T, paired bootstrap, contaminated block 1 EXCLUDED.
  * sample = 1,224 pairs/series (>= the 1,215 computed in 095). Half-width ~26.5 Elo.
  * decision rule, all three branches TERMINAL -- the single-machine loop line
    closes here whatever happens:
        CI lower > 0  -> POSITIVE_TRANSFER_CONFIRMED
        CI upper < 0  -> POSITIVE_TRANSFER_REJECTED
        straddles 0   -> LOOP_LINE_CLOSED_EFFECT_BOUNDED, reporting the ACTUAL
                         upper bound U (the pre-registered "at most +30" wording
                         is false over most of the NULL region, so the report
                         states +U and only adds the "<+30" clause when U < 30).
  * power caveat: 1,215 was sized for Delta_selfplay=+37.8. Against WS-T the
    selection-free prior is +30.2 (=+37.8 + (r1.00-WS-T=-7.6)), where power is
    ~61%. Accepted by human decision; it goes in the verdict sentence.
  * r=0.75 consistency window [-57,+129] (interpolation +36 +/- the +/-93
    single-block half-width). Descriptive only, never used for selection.

Phase B runs BEFORE Phase A: Phase A cannot change any decision, so an
interruption should cost the descriptive part, not the primary. A 19-minute
r=0.75 smoke sits at the front so a broken whitelist edit fails fast; its
iterations land in the real checkpoint dir and are resumed later, so nothing is
wasted. Stages 3-6 print no Elo and no CI -- the gate runs exactly once, at
stage 7, which mechanically forecloses optional stopping.
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
    pairs_for_80pct_power,
    self_test as gate_self_test,
    sign_class,
)
from t095_pipeline import VALUE_DIGEST, complete_tsv, parse_iter_metrics  # noqa: E402

# ---- pre-registered constants -------------------------------------------------
DECISIVE_ARM = "r050"
COMPARATOR = "wst"
CONTAMINATED_BLOCKS = (1,)          # where the r=0.50 arm was selected in 094
EXISTING_CLEAN_BLOCKS = (2, 3)      # r=0.50 already measured; WS-T is not
NEW_BLOCKS = (4, 5, 6, 7, 8, 9, 10, 11)
NEW_BLOCK_GAMES = 256               # 128 pairs each
LEGACY_BLOCK_GAMES = 200            # blocks 1-3
TARGET_PAIRS = 1215
EXPECTED_HALF_WIDTH = 26.5
R075_WINDOW = (-57.0, 129.0)
R075_BLOCKS = (1, 2)
STRATIFIED_SEED = 20260725          # distinct from the primary's 20260724
STRATIFIED_SAMPLES = 200_000
DOSE_SERIES = ("r000", "r025", "r050", "r075", "r100")
FLAT_POINT_MIN = 20.0               # dose-label tolerance, as in 095
ILLEGAL_ABORT = 5
RATE_SKEW_FLAG = 0.15

# ABBA: alternating which series plays first within a block, so that any
# within-block drift is not aligned with one series.
def block_order(block):
    return ("wst", "r050") if block % 2 == 0 else ("r050", "wst")


def block_games(block):
    return NEW_BLOCK_GAMES if block in NEW_BLOCKS else LEGACY_BLOCK_GAMES


def quarantine_partial(path, expected_games):
    """Move an interrupted match TSV aside so the block can be replayed.

    m2_match.py opens the TSV in append mode but restarts its game counter at 0
    and has no offset/resume, so a partial file can only be discarded, never
    continued. A file whose game column is a strict prefix of range(expected) is
    an interrupted run and is safe to quarantine; anything else has been
    appended to and needs a human.
    """
    path = Path(path)
    if not path.exists():
        return False
    rows = load_tsv(path)
    if len(rows) == expected_games:
        return False
    games = [row["game"] for row in rows]
    if games != list(range(len(rows))):
        raise RuntimeError(
            f"match TSV is not a clean prefix (appended?): {path} ({len(rows)} rows)"
        )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = path.with_suffix(f".tsv.partial-{stamp}")
    path.rename(moved)
    print(f"T096_QUARANTINE {path.name} -> {moved.name} rows={len(rows)}", flush=True)
    return True


def stratified_delta(candidate_blocks, baseline_blocks, *, label=""):
    """Two-level bootstrap: resample blocks, then pairs within each drawn block.

    Secondary/descriptive. The primary pooled bootstrap treats pairs as
    exchangeable across blocks and under-covers if a genuine block-level shift
    exists; this one does not. The verdict still follows the pooled primary --
    swapping estimands after seeing the data is exactly 095's lesson.
    """
    cand = [block_pair_scores(load_tsv(p)) for p in candidate_blocks]
    base = [block_pair_scores(load_tsv(p)) for p in baseline_blocks]
    point = elo_from_score(float(np.concatenate(cand).mean())) - elo_from_score(
        float(np.concatenate(base).mean())
    )
    rng = np.random.default_rng(STRATIFIED_SEED)
    deltas = np.empty(STRATIFIED_SAMPLES, dtype=np.float64)
    n_blocks = len(cand)
    for sample in range(STRATIFIED_SAMPLES):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        cand_draw, base_draw = [], []
        for pick in picks:
            size = len(cand[pick])
            idx = rng.integers(0, size, size=size)
            cand_draw.append(cand[pick][idx])
            base_draw.append(base[pick][idx])
        cs = float(np.clip(np.concatenate(cand_draw).mean(), 1e-9, 1 - 1e-9))
        bs = float(np.clip(np.concatenate(base_draw).mean(), 1e-9, 1 - 1e-9))
        deltas[sample] = 400.0 * math.log10(cs / (1 - cs)) - 400.0 * math.log10(bs / (1 - bs))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "label": label,
        "paired_delta_elo": point,
        "paired_ci95": [float(low), float(high)],
        "n_blocks": n_blocks,
        "bootstrap_samples": STRATIFIED_SAMPLES,
        "bootstrap_seed": STRATIFIED_SEED,
    }


def heterogeneity(per_block):
    """Cochran's Q and I-squared over the per-block deltas (descriptive)."""
    deltas = np.array([entry["delta_elo"] for entry in per_block.values()])
    ses = np.array([entry["se"] for entry in per_block.values()])
    if len(deltas) < 2 or not np.all(ses > 0):
        return {"k": int(len(deltas)), "Q": None, "df": None, "I2": None}
    weights = 1.0 / ses**2
    mean = float((weights * deltas).sum() / weights.sum())
    q = float((weights * (deltas - mean) ** 2).sum())
    df = len(deltas) - 1
    return {
        "k": int(len(deltas)),
        "Q": q,
        "df": df,
        "I2": float(max(0.0, (q - df) / q)) if q > 0 else 0.0,
        "weighted_mean_delta": mean,
    }


def dose_label_5(curve):
    """Descriptive shape label for the 5-point dose response. Never a verdict."""
    values = [curve[name] for name in DOSE_SERIES]
    peak = int(np.argmax(values))
    if peak == 0 or peak == len(values) - 1:
        return "MONOTONE_INCREASING" if peak == len(values) - 1 else "MONOTONE_DECREASING"
    rising = all(values[i] < values[i + 1] for i in range(peak))
    falling = all(values[i] > values[i + 1] for i in range(peak, len(values) - 1))
    if rising and falling:
        return f"INTERIOR_OPTIMUM_AT_{DOSE_SERIES[peak].upper()}"
    return "NON_MONOTONE"


def null_branch_wording(low, high):
    """Pre-registered NULL text, made data-conditional so it cannot be false."""
    text = (
        f"CI=[{low:.1f}, {high:.1f}] は 0 を跨いだため正の転写は本標本では確立しなかった。"
        f"本研究の分解能(半幅±{EXPECTED_HALF_WIDTH})で言えるのは上限が +{high:.1f} Elo であることのみ。"
    )
    if high < 30.0:
        return text + " かつ +30 未満 = 094 の block1 点推定 +79.1 は明確に棄却された。"
    return (
        text
        + " +79.1 は棄却されたが、+30 前後の効果は本標本では棄却も確認もできない"
        "(θ=+30.2 における検出力は約61%)。"
    )


def replay_095_check(logs):
    """The gate edits must not have disturbed Task 095's published numbers."""
    import t095_pipeline

    stored_path = Path(logs) / "task095_results.json"
    if not stored_path.exists():
        raise FileNotFoundError(stored_path)
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    tsv_path = {}
    for key, path in stored["series_paths"].items():
        block, series = key.split("_", 1)
        tsv_path[(int(block.replace("block", "")), series)] = Path(path)
    replay = t095_pipeline.analyse(
        tsv_path, 3, stored["r100_monitor"], {},
        stored["regression"], stored["equivalence"],
    )
    for series, want in stored["vs_wst_block1"].items():
        got = replay["vs_wst_block1"][series]
        for field in ("paired_delta_elo", "paired_ci95"):
            if got[field] != want[field]:
                raise AssertionError(
                    f"095 replay mismatch {series}.{field}: {got[field]!r} != {want[field]!r}"
                )
    for key, want in stored["delta_selfplay"].items():
        got = replay["delta_selfplay"][key]
        for field in ("paired_delta_elo", "paired_ci95"):
            if got[field] != want[field]:
                raise AssertionError(
                    f"095 replay mismatch delta_selfplay.{key}.{field}: "
                    f"{got[field]!r} != {want[field]!r}"
                )
    print("T096_REPLAY_095_OK all paired estimates reproduce task095_results.json", flush=True)


def analyse(tsv_path, monitor, trajectory, cargo_test, match_rates, equivalence_note):
    """Stage 7, isolated so it can be exercised without the 15h run."""
    def blocks_for(series, block_ids):
        return [tsv_path[(block, series)] for block in block_ids]

    clean_blocks = list(EXISTING_CLEAN_BLOCKS) + list(NEW_BLOCKS)
    same_session = list(NEW_BLOCKS)

    primary = paired_delta_general(
        blocks_for(DECISIVE_ARM, clean_blocks), blocks_for(COMPARATOR, clean_blocks),
        label="r050_minus_wst_clean",
    )
    primary["contains_selected_block"] = False
    axis = sign_class(primary)
    low, high = primary["paired_ci95"]
    if axis == "POS":
        verdict = "POSITIVE_TRANSFER_CONFIRMED"
    elif axis == "NEG":
        verdict = "POSITIVE_TRANSFER_REJECTED"
    else:
        verdict = "LOOP_LINE_CLOSED_EFFECT_BOUNDED"
    if primary["n_pairs"] < TARGET_PAIRS:
        verdict = "INSUFFICIENT_SAMPLE"

    sensitivity = {
        "same_session_only": paired_delta_general(
            blocks_for(DECISIVE_ARM, same_session), blocks_for(COMPARATOR, same_session),
            label="r050_minus_wst_same_session",
        ),
        "with_block1": paired_delta_general(
            blocks_for(DECISIVE_ARM, [1] + clean_blocks),
            blocks_for(COMPARATOR, [1] + clean_blocks),
            label="r050_minus_wst_with_block1",
        ),
        "block1_only": paired_delta_general(
            blocks_for(DECISIVE_ARM, [1]), blocks_for(COMPARATOR, [1]),
            label="r050_minus_wst_block1",
        ),
        "stratified": stratified_delta(
            blocks_for(DECISIVE_ARM, clean_blocks), blocks_for(COMPARATOR, clean_blocks),
            label="r050_minus_wst_stratified",
        ),
    }
    sensitivity["with_block1"]["contains_selected_block"] = True
    sensitivity["block1_only"]["contains_selected_block"] = True

    per_block = {}
    for block in clean_blocks:
        entry = paired_delta_general(
            blocks_for(DECISIVE_ARM, [block]), blocks_for(COMPARATOR, [block]),
            label=f"block{block}",
        )
        block_low, block_high = entry["paired_ci95"]
        per_block[str(block)] = {
            "delta_elo": entry["paired_delta_elo"],
            "ci95": entry["paired_ci95"],
            "se": (block_high - block_low) / (2 * 1.96),
            "n_pairs": entry["n_pairs"],
            "order": list(block_order(block)) if block in NEW_BLOCKS else ["r050", "wst"],
            "same_session": block in NEW_BLOCKS,
        }

    # Dose response: all five points on block 1, the shared axis. Descriptive only.
    wst_block1 = blocks_for("wst", [1])
    dose = {}
    absolute = {}
    for series in DOSE_SERIES:
        entry = paired_delta_general(
            blocks_for(series, [1]), wst_block1, label=f"{series}_vs_wst_block1"
        )
        dose[series] = entry["paired_delta_elo"]
        absolute[series] = {
            "wins": entry["wins"], "draws": entry["draws"], "losses": entry["losses"],
            "score": entry["score"], "absolute_elo": entry["absolute_elo"],
            "paired_ci95": entry["paired_ci95"],
        }
    r075_replication = paired_delta_general(
        blocks_for("r075", [2]), blocks_for("r100", [2]), label="r075_minus_r100_block2"
    )["paired_delta_elo"]
    r075_in_window = R075_WINDOW[0] <= dose["r075"] <= R075_WINDOW[1]

    flags = []
    illegal = 0
    reasons = {}
    for (block, series), path in sorted(tsv_path.items()):
        histogram = {}
        for row in load_tsv(path):
            histogram[row["reason"]] = histogram.get(row["reason"], 0) + 1
            if row["reason"].startswith(("illegal:", "engine_fail:")):
                illegal += 1
        reasons[f"block{block}_{series}"] = histogram
    if illegal:
        flags.append(f"ILLEGAL_OR_ENGINE_FAILURE_{illegal}")
    if not r075_in_window:
        flags.append("R075_OUTSIDE_CONSISTENCY_WINDOW")
    if monitor.get("value_model_digest") != VALUE_DIGEST:
        flags.append("VALUE_DIGEST_MISMATCH")
    if sign_class(sensitivity["stratified"]) != axis:
        flags.append("STRATIFIED_DISAGREES_WITH_POOLED")
    if sign_class(sensitivity["same_session_only"]) != axis:
        flags.append("SAME_SESSION_DISAGREES_WITH_POOLED")
    for block in NEW_BLOCKS:
        rates = [match_rates.get(f"block{block}_{s}") for s in ("wst", "r050")]
        if all(rate for rate in rates):
            if abs(rates[0] - rates[1]) / max(rates) > RATE_SKEW_FLAG:
                flags.append(f"SESSION_RATE_SKEW_block{block}")
    if trajectory:
        first, last = min(trajectory), max(trajectory)
        val_first, val_last = trajectory[first].get("val_top1"), trajectory[last].get("val_top1")
        anc_first, anc_last = trajectory[first].get("anchor_top1"), trajectory[last].get("anchor_top1")
        if None not in (val_first, val_last, anc_first, anc_last):
            if val_last < val_first - 0.002 and anc_last > anc_first + 0.01:
                flags.append("OVERFIT_SUSPECTED")

    return {
        "task": 96,
        "preregistration": {
            "arm": DECISIVE_ARM, "comparator": COMPARATOR,
            "excluded_blocks": list(CONTAMINATED_BLOCKS),
            "target_pairs": TARGET_PAIRS,
            "expected_half_width_elo": EXPECTED_HALF_WIDTH,
            "power_at_30p2": 0.61, "power_at_37p8": 0.80, "power_at_53": 0.98,
            "decision_rule": "CI lower>0 POSITIVE / upper<0 REJECTED / else CLOSED (all terminal)",
            "r075_window": list(R075_WINDOW),
        },
        "config": {
            "loop_iters": 25, "sims": 32, "states_per_iter": 65_536,
            "updates_per_iter": 512, "seed": 20260714, "anchor_seed": 20260714,
            "external_sims": 128, "external_byoyomi_ms": 100, "opening_plies": 6,
            "max_plies": 320, "gikou_depth_limit": 6,
            "blocks": {
                str(block): {"seed": BLOCK_SEEDS[block], "games": block_games(block),
                             "order": list(block_order(block)) if block in NEW_BLOCKS else None}
                for block in [1] + clean_blocks
            },
            "bootstrap_samples": primary["bootstrap_samples"],
            "bootstrap_seed": primary["bootstrap_seed"],
        },
        "series_paths": {
            f"block{block}_{series}": str(path) for (block, series), path in tsv_path.items()
        },
        "primary": primary,
        "sign_class": axis,
        "sensitivity": sensitivity,
        "per_block_delta": per_block,
        "heterogeneity": heterogeneity(per_block),
        "dose_response": {
            "curve": dose,
            "label": dose_label_5(dose),
            "r075_block2_replication_vs_r100": r075_replication,
            "r075_consistency_window": list(R075_WINDOW),
            "r075_in_window": r075_in_window,
            "caveat": "descriptive only; +/-93 Elo per single-block point",
        },
        "absolute_block1": absolute,
        "r075_monitor": monitor,
        "r075_trajectory": {str(k): v for k, v in trajectory.items()},
        "match_rates_games_per_min": match_rates,
        "termination_reasons": reasons,
        "cargo_test": cargo_test,
        "equivalence_note": equivalence_note,
        "pairs_for_80pct_power": pairs_for_80pct_power(primary),
        "null_branch_statement": null_branch_wording(low, high) if axis == "NULL" else None,
        "scope_limitation": (
            "config.seed drives self-play only, so each arm is a SINGLE training "
            "trajectory. The CI covers match noise only, not training-seed variance. "
            "A confirmed result licenses 'this r=0.50 net beats its own initialisation', "
            "not 'the r=0.50 recipe yields this in general'."
        ),
        "flags": flags,
        "verdict": verdict,
    }


def report(results):
    primary = results["primary"]
    print(
        f"GATE primary {primary['label']} n_pairs={primary['n_pairs']} "
        f"n_games={primary['n_games']} delta={primary['paired_delta_elo']:.2f} "
        f"ci={[round(v, 2) for v in primary['paired_ci95']]} class={results['sign_class']}",
        flush=True,
    )
    for key, entry in results["sensitivity"].items():
        print(
            f"GATE sensitivity {key} delta={entry['paired_delta_elo']:.2f} "
            f"ci={[round(v, 2) for v in entry['paired_ci95']]}",
            flush=True,
        )
    curve = {k: round(v, 1) for k, v in results["dose_response"]["curve"].items()}
    print(f"DOSE {results['dose_response']['label']} curve={curve} "
          f"r075_in_window={results['dose_response']['r075_in_window']}", flush=True)
    print("PER_BLOCK " + " ".join(
        f"{b}:{e['delta_elo']:+.1f}" for b, e in results["per_block_delta"].items()), flush=True)
    het = results["heterogeneity"]
    print(f"HETEROGENEITY k={het['k']} Q={het['Q']} df={het['df']} I2={het['I2']}", flush=True)
    print(f"FLAGS {results['flags']}", flush=True)
    if results["null_branch_statement"]:
        print(f"NULL_STATEMENT {results['null_branch_statement']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="task096")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    parser.add_argument("--self-test", action="store_true", help="gate + 095 replay, then exit")
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
    r075_ckpt_dir = models / f"{args.run_tag}_r075_ckpt"
    # /tmp is wiped when the WSL VM stops -- a live risk across a 15h job.
    work_root = data_root / "work" / args.run_tag

    series_ckpt = {
        "wst": wst,
        "r000": clone / "t091_s32_b128_ckpt/000025.ckpt",
        "r025": models / "task094_r025_ckpt/000025.ckpt",
        "r050": models / "task094_r050_ckpt/000025.ckpt",
        "r075": r075_ckpt_dir / f"{args.iters:06d}.ckpt",
        "r100": models / "task095_r100_ckpt/000025.ckpt",
    }
    block1 = {
        "wst": logs / "task093_wst_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r000": logs / "task093_iter25_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r025": logs / "task094_r025_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r050": logs / "task094_r050_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv",
        "r100": logs / "task095_r100_iter025_vs_gikou6_s128_b100_seed20260702_n200.tsv",
    }

    def tsv_for(block, series):
        """Existing artefacts keep their original names; new ones get task096."""
        if block == 1 and series in block1:
            return block1[series]
        if block in (2, 3) and series in ("r050", "r100"):
            return logs / (f"task095_{series}_iter025_vs_gikou6_s128_b100_"
                           f"seed{BLOCK_SEEDS[block]}_n200.tsv")
        tag = "" if series == "wst" else f"_iter{args.iters:03d}"
        return logs / (f"{args.run_tag}_{series}{tag}_vs_gikou6_s128_b100_"
                       f"seed{BLOCK_SEEDS[block]}_n{block_games(block)}.tsv")

    # Phase B: WS-T on the two existing clean blocks, then both series on the new
    # blocks in ABBA order. Within a block the two series run back-to-back so any
    # machine-state drift is common to both.
    phase_b = [(block, "wst") for block in EXISTING_CLEAN_BLOCKS]
    for block in NEW_BLOCKS:
        phase_b.extend((block, series) for series in block_order(block))
    phase_a = [(block, "r075") for block in R075_BLOCKS]

    tsv_path = {}
    for block in [1] + list(EXISTING_CLEAN_BLOCKS) + list(NEW_BLOCKS):
        for series in ("wst", "r000", "r025", "r050", "r075", "r100"):
            if block == 1 and series in block1:
                tsv_path[(block, series)] = block1[series]
    for block, series in phase_b + phase_a:
        tsv_path[(block, series)] = tsv_for(block, series)
    for block in EXISTING_CLEAN_BLOCKS:
        tsv_path[(block, "r050")] = tsv_for(block, "r050")
        tsv_path[(block, "r100")] = tsv_for(block, "r100")

    if args.dry_run:
        total_new = 0
        print(f"DRYRUN work_root={work_root}")
        print(f"DRYRUN phase_b matches ({len(phase_b)}):")
        for block, series in phase_b:
            games = block_games(block)
            total_new += games
            print(f"  block{block:>2} seed={BLOCK_SEEDS[block]} {series:>4} games={games} "
                  f"order={block_order(block) if block in NEW_BLOCKS else '(existing r050)'}")
        print(f"DRYRUN phase_a matches ({len(phase_a)}):")
        for block, series in phase_a:
            total_new += block_games(block)
            print(f"  block{block:>2} seed={BLOCK_SEEDS[block]} {series:>4} "
                  f"games={block_games(block)}")
        clean = list(EXISTING_CLEAN_BLOCKS) + list(NEW_BLOCKS)
        pairs = sum(block_games(b) // 2 for b in clean)
        print(f"DRYRUN primary pairs={pairs} (target {TARGET_PAIRS}) "
              f"blocks={clean}")
        print(f"DRYRUN new games={total_new} match_hours={total_new * 7.95 / 3600:.2f} "
              f"loop_hours={568 * args.iters / 3600:.2f} "
              f"total_hours={total_new * 7.95 / 3600 + 568 * args.iters / 3600:.2f}")
        return 0

    # ---- stage 0: preflight -------------------------------------------------
    for name in ("t094_loop_anchored.py", "t094_pipeline.py", "t095_gate.py",
                 "t095_pipeline.py", "t096_pipeline.py"):
        source = repo / "experiments/alphazero" / name
        linked = clone / name
        if not linked.exists() or not os.path.samefile(source, linked):
            raise RuntimeError(f"clone {name} is not linked to experiments SSOT")
    # The analysis-side checks run first and without the cleanliness gate, so
    # they can be used to validate the code BEFORE it is committed.
    gate_self_test(logs)
    replay_095_check(logs)
    if args.self_test:
        print("T096_SELFTEST_OK", flush=True)
        return 0
    # Provenance: a real run must come from committed source, so RUN_META's
    # HEAD identifies exactly what produced the numbers.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "experiments/alphazero"],
        cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiments/alphazero is dirty; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    for path in [pgx_py, torch_py, anchor_data / "train_obs.npy", *series_ckpt.values(),
                 *block1.values()]:
        if "r075" in str(path):
            continue  # produced by this run
        if not Path(path).exists():
            raise FileNotFoundError(path)
    logs.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})
    print(f"T096_PREFLIGHT_OK head={head} exp_dirty=0 pairs_target={TARGET_PAIRS}", flush=True)

    # ---- stage 1: cargo test ------------------------------------------------
    cargo_test = {"status": "SKIPPED"}
    if not args.skip_cargo:
        cargo_log = logs / f"{args.run_tag}_cargo_test.log"
        run_logged(["cargo", "test", "--release"], cargo_log, cwd=repo, env=env)
        cargo_test = {"status": "PASS", "log": str(cargo_log)}
        print("T096_CARGO_TEST_OK", flush=True)

    def loop_command(target):
        return [
            pgx_py, "t094_loop_anchored.py",
            f"max_num_iters={target}", "num_simulations=32",
            "selfplay_batch_size=128", "states_per_iter=65536",
            "training_batch_size=128", "learning_rate=0.0001",
            f"init_ckpt={wst}", f"ckpt_dir={r075_ckpt_dir}", "resume=auto",
            "ckpt_interval=5", "monitor_interval=5", "expected_iter_s=720",
            f"anchor_data_dir={anchor_data}", "anchor_ratio=0.75",
            "anchor_seed=20260714",
        ]

    # ---- stage 2: r=0.75 smoke (iterations land in the real ckpt dir) -------
    smoke_log = logs / f"{args.run_tag}_r075_smoke.log"
    if not (r075_ckpt_dir / "000002.ckpt").exists() and not series_ckpt["r075"].exists():
        command = loop_command(2)
        command[command.index("ckpt_interval=5")] = "ckpt_interval=1"
        run_logged(command, smoke_log, cwd=clone, env=env)
        header = [line for line in smoke_log.read_text(encoding="utf-8").splitlines()
                  if line.startswith("t094: sims=")]
        expected = ("anchor_updates=384", "anchor_ratio=0.75", "visit_updates=128",
                    "selfplay=on", "force_selfplay=False")
        if not header or not all(token in header[0] for token in expected):
            raise RuntimeError(f"r=0.75 loop header unexpected: {header}")
        print(f"T096_R075_SMOKE_OK {' '.join(expected)}", flush=True)
    else:
        print("SKIP r=0.75 smoke (checkpoint exists)", flush=True)

    # ---- stage 3: Phase B matches ------------------------------------------
    match_rates = {}

    def run_match(block, series):
        path = tsv_path[(block, series)]
        games = block_games(block)
        quarantine_partial(path, games)
        console = Path(str(path).replace(".tsv", ".console.log"))
        if complete_tsv(path, games):
            print(f"SKIP existing match TSV {path.name}", flush=True)
        else:
            started = time.time()
            run_logged([
                pgx_py, "m2_match.py",
                "--az-ckpt", series_ckpt[series], "--az-sims", "128",
                "--opponent", "gikou", "--games", str(games), "--byoyomi", "100",
                "--opening-plies", "6", "--max-plies", "320",
                "--seed", str(BLOCK_SEEDS[block]), "--tsv", path,
            ], console, cwd=clone, env=env)
            elapsed = time.time() - started
            rate = games / (elapsed / 60.0)
            match_rates[f"block{block}_{series}"] = rate
            print(f"T096_MATCH_RATE block={block} series={series} "
                  f"games_per_min={rate:.2f}", flush=True)
        print(f"T096_MATCH_DONE block={block} series={series} "
              f"seed={BLOCK_SEEDS[block]} n={games}", flush=True)

    for block, series in phase_b:
        run_match(block, series)
    clean_blocks = list(EXISTING_CLEAN_BLOCKS) + list(NEW_BLOCKS)
    pairs = sum(block_games(block) // 2 for block in clean_blocks)
    print(f"T096_PHASEB_MATCHES_DONE pairs={pairs}", flush=True)

    # ---- stage 4: Phase A loop ---------------------------------------------
    r075_log = logs / f"{args.run_tag}_r075_loop_to_{args.iters:03d}.log"
    if not series_ckpt["r075"].exists():
        run_logged(loop_command(args.iters), r075_log, cwd=clone, env=env)
    print(f"T096_R075_LOOP_DONE iters={args.iters}", flush=True)

    # ---- stage 5: rescore monitor ------------------------------------------
    cache = work_root / "cache_2m"
    baseline_rescore = work_root / "rescore_wst"
    r075_rescore = work_root / "rescore_r075"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "meta.json").exists():
        run_logged([torch_py, "-m", "trainer.prepare_data", teacher / "run2m_c", cache,
                    "--canonical-shared", "--val-shards", "4"],
                   logs / f"{args.run_tag}_prepare.log", cwd=repo, env=env)
    if not (baseline_rescore / "rescore_metrics.json").exists():
        run_logged([pgx_py, "t091_rescore.py", "--cache-dir", cache, "--ckpt", wst,
                    "--out-dir", baseline_rescore],
                   logs / f"{args.run_tag}_rescore_wst.log", cwd=clone, env=env)
    if not (r075_rescore / "rescore_metrics.json").exists():
        run_logged([pgx_py, "t091_rescore.py", "--cache-dir", cache,
                    "--ckpt", series_ckpt["r075"], "--out-dir", r075_rescore,
                    "--reference-value-dir", baseline_rescore,
                    "--reference-policy-dir", baseline_rescore],
                   logs / f"{args.run_tag}_rescore_r075.log", cwd=clone, env=env)
    monitor = json.loads((r075_rescore / "rescore_metrics.json").read_text(encoding="utf-8"))
    digest_ok = monitor.get("value_model_digest") == VALUE_DIGEST
    print(f"T096_R075_RESCORE_OK digest_ok={digest_ok}", flush=True)

    # ---- stage 6: Phase A matches ------------------------------------------
    for block, series in phase_a:
        run_match(block, series)
    print("T096_PHASEA_MATCHES_DONE", flush=True)

    # ---- stage 7: the gate, run exactly once --------------------------------
    trajectory = {}
    for iteration in range(5, args.iters + 1, 5):
        try:
            trajectory[iteration] = parse_iter_metrics(r075_log, iteration)
        except (ValueError, FileNotFoundError):
            continue
    for series in (DECISIVE_ARM, COMPARATOR):
        assert_blocks_disjoint([tsv_path[(block, series)] for block in clean_blocks])

    results = analyse(
        tsv_path, monitor, trajectory, cargo_test, match_rates,
        equivalence_note=("self-play skip optimisation unvalidated and unused; "
                          "all arms ran the canonical generation-ON path"),
    )
    results["git"] = {"head": head, "exp_dirty": 0}
    output = logs / f"{args.run_tag}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    print(f"T096_PIPELINE_DONE verdict={results['verdict']} results={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
