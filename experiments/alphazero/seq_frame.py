#!/usr/bin/env python3
"""Task 137: paired sequential frame proposal; offline replay and simulation only.

Eight balanced looks, 100 games per anchor per look. Opening colour pairs stay
intact. A Pocock boundary of 2.6 has Gaussian eight-look one-sided crossing
probability about .02, leaving margin below .025. Percentile bootstrap bounds are on
the PAIRED SCORE difference; its sign is the sign of the pooled Elo difference.
Bootstrap calibration is approximate, as in the fixed frame, not an exact
finite-sample coverage theorem. See reports/task137_sequential_frame.md.

No engine imports or subprocesses. Drivers pass normalized per-game rows as
{anchor: rows}, in chunk order, to sequential_looks(); stop at its first terminal
decision. It accepts balanced prefixes too, so no future games are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

ANCHORS = {"tanuki": 20_000, "aoba": 10_000}
SEEDS = tuple(range(20272001, 20272009))
CHUNK, LOOKS = 100, 8
BOUNDARY = 2.6
TAIL_ALPHA = (.5 * math.erfc(BOUNDARY / math.sqrt(2)),) * LOOKS
SIM_SEED = 20273701

# Actual preregistered comparators, not today's champion. 32M experiments used
# matched recipe controls. Provenance is checked against each script below.
COMPARISONS = {
    "task128_student_qsrelc": ("task125_student_factor", "t128_qs_relabel.py", "CONTROL", "arms.C.pooled"),
    "task128_student_qsreld": ("task125_student_factor", "t128_qs_relabel.py", "CONTROL", "arms.D.pooled"),
    "task129_student_64m_qsrel_e12": ("task126_student_64m_factor_e24", "t129_qs_relabel_64m.py", "ADOPTED", "primary.pooled"),
    "task130_student_calib": ("task128_student_qsrelc", "t130_label_calibration.py", "CONTROL", "frame.pooled"),
    "task131_student_power25": ("task128_student_qsrelc", "t131_power_loss.py", "CONTROL", "frame.pooled"),
    "task132_student_vm": ("task128_student_qsrelc", "t132_halfkp_vm.py", "CONTROL", "frame.pooled"),
    "task134_student_sharp32": ("task129_student_64m_qsrel_e12", "t134_sharp_relabel_s32.py", "ADOPTED", "primary.pooled"),
    "task135_student_fast": ("task129_student_64m_qsrel_e12", "t135_fast_retrain.py", "ADOPTED", "primary.pooled"),
}
REFERENCE_ONLY = {"task125_student_factor", "task126_student_64m_factor_e24"}
FILE_RE = re.compile(r"(std|conf\d+)_(.+)_(tanuki|aoba)(\d+)_c(\d+)_seed(\d+)_n100\.tsv")


def elo(score):
    """Task 137: the fixed frame's pooled score-to-Elo transform."""
    score = np.clip(score, 1e-9, 1 - 1e-9)
    return 400 * np.log10(score / (1 - score))


def verdict(bounds):
    """Task 137: neutral means not distinguished, never equivalence."""
    return "better" if bounds[0] > 0 else "worse" if bounds[1] < 0 else "no_difference"


def paired_scores(candidate, base):
    """Task 137: validate aligned colour pairs; return {anchor: (pairs, 2)}."""
    if set(candidate) != set(ANCHORS) or set(base) != set(ANCHORS):
        raise ValueError("exactly the two current anchors are required")
    lengths = {len(candidate[a]) for a in ANCHORS} | {len(base[a]) for a in ANCHORS}
    if len(lengths) != 1 or not lengths <= set(range(CHUNK, CHUNK * LOOKS + 1, CHUNK)):
        raise ValueError("need equal 100..800 game prefixes, in complete chunks, at both anchors")
    output = {}
    for anchor in ANCHORS:
        ca, ba = candidate[anchor], base[anchor]
        for rows in (ca, ba):
            for i, row in enumerate(rows):
                if row["score"] not in (0, .5, 1):
                    raise ValueError("score must be 0, .5 or 1")
                if row["color"] != ("b" if i % 2 == 0 else "w"):
                    raise ValueError("opening pairs must have colours b,w")
                if str(row.get("reason", "")).startswith(("illegal", "engine_fail", "error")):
                    raise ValueError("invalid engine result")
                if i % 2 and any(row[k] != rows[i - 1][k] for k in ("opening_seed", "opening")):
                    raise ValueError("broken opening colour pair")
        for a, b in zip(ca, ba):
            if any(a[k] != b[k] for k in ("opening_seed", "opening", "color")):
                raise ValueError(f"{anchor}: candidate/base openings differ")
        output[anchor] = np.stack([
            np.array([r["score"] for r in rows]).reshape(-1, 2).mean(1) for rows in (ca, ba)
        ], axis=1)
    return output


def difference_counts(pairs):
    """Task 137: nine possible paired score differences on the quarter lattice."""
    values = np.rint(4 * (pairs[..., 0] - pairs[..., 1])).astype(int) + 4
    return np.stack([(values == j).sum(axis=-1) for j in range(9)], axis=-1)


def bootstrap_distribution(counts):
    """Task 137: exact conditional bootstrap sum distribution, using FFT powers.

    A pair difference is j/4, j=-4..4. The n-fold empirical convolution is
    exactly what resampling n intact pairs computes, without bootstrap noise.
    This is an exact bootstrap calculation, NOT an exact frequentist test.
    """
    counts = np.asarray(counts)
    n_values = counts.sum(axis=-1)
    n = int(n_values.flat[0])
    if n < 1 or not np.all(n_values == n) or np.any(counts < 0):
        raise ValueError("counts must be nonnegative with equal positive totals")
    size = 1 << (8 * n).bit_length()   # strictly larger than maximum degree 8*n
    spectrum = np.fft.rfft(counts / n, n=size, axis=-1)
    pmf = np.fft.irfft(spectrum ** n, n=size, axis=-1)[..., :8 * n + 1]
    pmf = np.maximum(pmf, 0)
    pmf /= pmf.sum(axis=-1, keepdims=True)
    return pmf


def bootstrap_bounds(counts, alpha):
    """Task 137: percentile bounds on mean paired score difference (not Elo)."""
    if not 0 < alpha < .5:
        raise ValueError("tail alpha must be between 0 and .5")
    pmf = bootstrap_distribution(counts)
    n = int(np.asarray(counts).sum(axis=-1).flat[0])
    cdf = pmf.cumsum(axis=-1)
    cdf[..., -1] = 1
    low = ((cdf >= alpha).argmax(axis=-1) - 4 * n) / (4 * n)
    high = ((cdf >= 1 - alpha).argmax(axis=-1) - 4 * n) / (4 * n)
    return np.stack((low, high), axis=-1)


def sequential_looks(candidate, base):
    """Task 137: yield estimate, Pocock bounds and decision; stop on crossing.

    Bounds are in score units; delta_elo is descriptive, not a repeated 95% CI.
    Terminal outcomes: better / worse / no_difference. Earlier null looks say
    continue. Only balanced prefixes enter a decision; no within-chunk peeking.
    """
    pairs = paired_scores(candidate, base)
    available = len(candidate["tanuki"]) // CHUNK
    for look in range(1, available + 1):
        pooled = np.concatenate([pairs[a][:look * CHUNK // 2] for a in ANCHORS])
        bounds = bootstrap_bounds(difference_counts(pooled), TAIL_ALPHA[look - 1]).tolist()
        decision = verdict(bounds)
        if decision == "no_difference" and look < LOOKS:
            decision = "continue"
        yield {"look": look, "candidate_games": 2 * CHUNK * look, "n_pairs": len(pooled),
               "delta_elo": float(elo(pooled[:, 0].mean()) - elo(pooled[:, 1].mean())),
               "score_delta": float((pooled[:, 0] - pooled[:, 1]).mean()),
               "bounds": bounds, "bounds_scale": "paired_score", "tail_alpha": TAIL_ALPHA[look - 1],
               "decision": decision}
        if decision != "continue":
            return


def load_chunk(path, anchor, seed):
    """Task 137: read and validate one current-yardstick TSV without engine code."""
    content = path.read_text(encoding="utf-8")
    if (f"seed={seed} max_games=100" not in content or
            f"go_a='go nodes 100000' go_b='go nodes {ANCHORS[anchor]}'" not in content or
            "elapsed_ms=" not in content or "opening_plies=6 max_plies=320" not in content or
            "extra_a=[] extra_b=[]" not in content):
        raise ValueError(f"{path}: incomplete or nonstandard protocol")
    rows = []
    for line in content.splitlines():
        fields = line.split("\t")
        if not fields[0].isdigit():
            continue
        if len(fields) != 21:
            raise ValueError(f"{path}: malformed game row")
        rows.append({"game": int(fields[0]) - 1, "color": {"black": "b", "white": "w"}[fields[1]],
                     "opening_seed": fields[2], "score": {"win": 1., "draw": .5, "loss": 0.}[fields[3]],
                     "reason": fields[4], "opening": fields[20]})
    if [r["game"] for r in rows] != list(range(CHUNK)):
        raise ValueError(f"{path}: expected games 1..100 exactly")
    return rows


def discover(logs):
    """Task 137: inventory all frame files; never mix old node budgets or seeds."""
    groups, old = {}, 0
    for path in sorted(logs.glob("*.tsv")):
        m = FILE_RE.fullmatch(path.name)
        if not m:
            continue
        prefix, name, anchor, nodes, chunk, seed = m.groups()
        if int(nodes) != ANCHORS[anchor]:
            old += 1
            continue
        key = (prefix, name)
        slots = groups.setdefault(key, {})
        slot = (anchor, int(chunk))
        if slot in slots:
            raise ValueError(f"duplicate frame slot: {path}")
        slots[slot] = (int(seed), path)
    return groups, old


def frame_seeds(slots):
    """Task 137: identify complete, balanced eight-chunk frames."""
    if set(slots) != {(a, i) for a in ANCHORS for i in range(LOOKS)}:
        return None
    seeds = tuple(slots[("tanuki", i)][0] for i in range(LOOKS))
    if any(slots[(a, i)][0] != seeds[i] for a in ANCHORS for i in range(LOOKS)):
        raise ValueError("anchor chunk seeds differ")
    return seeds


def read_frame(slots, seeds):
    """Task 137: retain chunk order and make every game's provenance auditable."""
    rows = {a: [] for a in ANCHORS}
    manifest = []
    for a in ANCHORS:
        for i, seed in enumerate(seeds):
            actual_seed, path = slots[(a, i)]
            if actual_seed != seed:
                raise ValueError(f"{path}: unexpected seed")
            rows[a].extend(load_chunk(path, a, seed))
            manifest.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows, manifest


def comparator(name):
    """Task 137: fail closed on unknown historical comparators; verify sources."""
    base, script, constant, result_path = COMPARISONS[name]
    source = Path(__file__).with_name(script)
    if not re.search(rf'^{constant} = "{re.escape(base)}"', source.read_text(), re.MULTILINE):
        raise ValueError(f"historical comparator changed: {source}")
    return base, f"{script}:{constant}", result_path


def complete_comparisons(logs):
    """Task 137: enumerate every complete current frame and explicit exclusions."""
    groups, old = discover(logs)
    comparisons, exclusions = [], []
    for (prefix, name), slots in sorted(groups.items()):
        seeds = frame_seeds(slots)
        if seeds is None:
            exclusions.append({"prefix": prefix, "eval": name, "reason": "incomplete frame"})
            continue
        if prefix == "std" and seeds != SEEDS:
            raise ValueError(f"{name}: nonstandard seeds under std prefix")
        if name in REFERENCE_ONLY:
            exclusions.append({"prefix": prefix, "eval": name, "reason": "reference/control only, no current-yardstick comparison"})
            continue
        if name not in COMPARISONS:
            raise ValueError(f"unknown comparator for complete frame {prefix}/{name}; add sourced mapping")
        base, provenance, result_path = comparator(name)
        base_slots = groups.get((prefix, base))
        if base_slots is None or frame_seeds(base_slots) != seeds:
            raise ValueError(f"missing complete matched base: {prefix}/{name} vs {base}")
        ca, ca_manifest = read_frame(slots, seeds)
        ba, ba_manifest = read_frame(base_slots, seeds)
        paired_scores(ca, ba)
        comparisons.append({"prefix": prefix, "candidate": name, "base": base, "seeds": seeds,
                            "provenance": provenance, "result_path": result_path,
                            "candidate_rows": ca, "base_rows": ba, "manifest": ca_manifest + ba_manifest})
    return comparisons, {"excluded": exclusions, "old_yardstick_files_excluded": old}


def replay(logs):
    """Task 137: compare against the unchanged 200,000-draw fixed-design oracle."""
    # Import the existing statistic lazily; its main/engine drivers are never called.
    from experiments.alphazero.t103_absolute import paired_series_delta

    comparisons, inventory = complete_comparisons(logs)
    print(f"Excluded old-yardstick files: {inventory['old_yardstick_files_excluded']}", flush=True)
    for excluded in inventory["excluded"]:
        print(f"Excluded {excluded['prefix']} {excluded['eval']}: {excluded['reason']}", flush=True)
    results = []
    for item in comparisons:
        ca, ba = item["candidate_rows"], item["base_rows"]
        fixed = paired_series_delta(sum((ca[a] for a in ANCHORS), []),
                                    sum((ba[a] for a in ANCHORS), []), label=item["candidate"])
        task = re.match(r"task\d+", item["candidate"])[0]
        reference = json.loads((logs / f"{task}_results.json").read_text())
        result_path = "confirm.pooled" if item["prefix"] != "std" else item["result_path"]
        for key in result_path.split("."):
            reference = reference[key]
        if not np.allclose([fixed["delta_elo"], *fixed["ci95"]],
                           [reference["delta_elo"], *reference["ci95"]], atol=1e-9, rtol=0):
            raise ValueError(f"{task}/{item['prefix']}: replay disagrees with stored fixed results")
        looks = list(sequential_looks(ca, ba))
        stopped = looks[-1]
        disagreement = None
        fixed_verdict = verdict(fixed["ci95"])
        pooled = np.concatenate(list(paired_scores(ca, ba).values()))
        fixed_score_bounds = bootstrap_bounds(difference_counts(pooled), .025).tolist()
        if verdict(fixed_score_bounds) != fixed_verdict:
            raise ValueError("finite-draw fixed bootstrap and exact-convolution verdict differ")
        if stopped["decision"] != fixed_verdict:
            if stopped["decision"] == "no_difference":
                disagreement = (f"At look 8 the sequential paired-score bounds {stopped['bounds']} "
                                f"include zero; fixed score bounds {fixed_score_bounds} exclude zero. "
                                f"Repeated-look correction uses tail {TAIL_ALPHA[-1]:.8f}, vs fixed .025.")
            else:
                disagreement = (f"An early crossing at look {stopped['look']} gave {stopped['decision']}; "
                                f"the later completed frame gave {fixed_verdict}. Later data are not "
                                "available at the stopping time, so the terminal decision stays frozen.")
        out = {k: v for k, v in item.items() if not k.endswith("_rows")}
        out.update(fixed=fixed, fixed_verdict=fixed_verdict, sequential=stopped["decision"],
                   stop_look=stopped["look"], games_saved=1600 - stopped["candidate_games"],
                   looks=looks, disagreement=disagreement, stored_fixed_verified=True,
                   fixed_exact_score_bounds=fixed_score_bounds, fixed_bootstrap_sign_agrees=True)
        results.append(out)
        print(f"{item['prefix']} {item['candidate']} - {item['base']}: "
              f"fixed={fixed_verdict} {fixed['delta_elo']:+.2f} {fixed['ci95']} "
              f"seq={stopped['decision']} look={stopped['look']} saved={out['games_saved']}", flush=True)
        if disagreement:
            print(f"  DISAGREEMENT: {disagreement}", flush=True)
    return {"method": method_metadata(), "inventory": inventory, "comparisons": results}


def method_metadata():
    """Task 137: freeze the proposal independently of replay outcomes."""
    return {"boundary": BOUNDARY, "looks": LOOKS, "chunk_games_per_anchor": CHUNK,
            "tail_alpha": TAIL_ALPHA, "gaussian_one_side_crossing": gaussian_crossing(),
            "bounds_scale": "paired_score", "bootstrap": "exact empirical convolution",
            "calibration": "approximate percentile bootstrap, not a finite-sample guarantee"}


def gaussian_crossing(order=200):
    """Task 137: Pocock one-sided crossing via deterministic Gaussian quadrature.

    Canonical Z[k] has corr(Z[j], Z[k]) = sqrt(j/k). Propagate the
    unabsorbed density below +BOUNDARY, ignoring the opposite boundary so this
    conservatively upper-bounds first crossing of that side in a two-sided test.
    The omitted lower tail below -9 has probability < 8 * Phi(-9).
    """
    points, weights = np.polynomial.legendre.leggauss(order)
    width = BOUNDARY + 9
    points = -9 + (points + 1) * width / 2
    weights = weights * width / 2

    def normal(x):
        return np.exp(-x * x / 2) / math.sqrt(2 * math.pi)

    density = normal(points)
    for k in range(2, LOOKS + 1):
        rho, sd = math.sqrt((k - 1) / k), math.sqrt(1 / k)
        kernel = normal((points[:, None] - rho * points[None, :]) / sd) / sd
        density = kernel @ (density * weights)
    return float(1 - density @ weights)


def simulation_pool(comparisons):
    """Task 137: intact shared-opening blocks across BOTH anchors, standard only."""
    blocks = []
    for item in comparisons:
        if item["prefix"] != "std":
            continue
        pairs = paired_scores(item["candidate_rows"], item["base_rows"])
        a, b = pairs.values()
        for anchor in ANCHORS:
            if any(x[k] != y[k] for x, y in zip(item["candidate_rows"]["tanuki"],
                                               item["candidate_rows"][anchor])
                   for k in ("opening_seed", "opening", "color")):
                raise ValueError("simulation requires shared openings across anchors")
        blocks.append(np.stack((a, b), axis=1))  # (400 openings, 2 anchors, candidate/base)
    if not blocks:
        raise ValueError("no standard comparisons for simulation")
    raw = np.concatenate(blocks)
    # Swap the ENTIRE candidate/base block, preserving anchor and colour-pair
    # dependence. Each support point still comes from actual paired outcomes.
    symmetric = np.concatenate((raw, raw[..., ::-1]))
    support, counts = np.unique(symmetric.reshape(-1, 4), axis=0, return_counts=True)
    return support.reshape(-1, 2, 2), counts / counts.sum(), len(raw)


def tilted_pool(support, probabilities, target):
    """Task 137: exponential reweighting to exactly the requested pooled Elo delta."""
    d = (support[..., 0] - support[..., 1]).mean(axis=1)

    def at(lam):
        logw = np.log(probabilities) + lam * d
        weights = np.exp(logw - logw.max())
        weights /= weights.sum()
        means = (weights[:, None, None] * support).sum(axis=0).mean(axis=0)
        return weights, float(elo(means[0]) - elo(means[1])), means

    lo, hi = -64., 64.
    if not at(lo)[1] <= target <= at(hi)[1]:
        raise ValueError("requested delta outside empirical support")
    for _ in range(65):
        mid = (lo + hi) / 2
        if at(mid)[1] < target:
            lo = mid
        else:
            hi = mid
    weights, actual, means = at((lo + hi) / 2)
    return weights, {"lambda": (lo + hi) / 2, "actual_delta_elo": actual,
                     "candidate_mean": float(means[0]), "base_mean": float(means[1])}


def rate_summary(decisions, games):
    """Task 137: rates, Wilson uncertainty and sample-size uncertainty."""
    n = len(decisions)
    output = {}
    for name, code in (("better", 1), ("worse", -1), ("no_difference", 0)):
        p = float((decisions == code).mean())
        den = 1 + 1.96 ** 2 / n
        center = (p + 1.96 ** 2 / (2 * n)) / den
        half = 1.96 * math.sqrt(p * (1 - p) / n + 1.96 ** 2 / (4 * n * n)) / den
        output[name] = p
        output[f"{name}_ci95"] = [center - half, center + half]
    output["mean_games"] = float(games.mean())
    output["mean_games_se"] = float(games.std(ddof=1) / math.sqrt(n))
    return output


def simulate(logs, trials, seed, targets, batch_size=128):
    """Task 137: resample real blocks; evaluate both designs on identical paths."""
    comparisons, inventory = complete_comparisons(logs)
    support, probabilities, n_blocks = simulation_pool(comparisons)
    rng = np.random.default_rng(seed)
    categories = np.rint(4 * (support[..., 0] - support[..., 1])).astype(int) + 4
    results = []
    for target in targets:
        weights, population = tilted_pool(support, probabilities, target)
        seq_decisions, fixed_decisions = np.zeros(trials, int), np.zeros(trials, int)
        games = np.full(trials, 1600)
        stop_hist = np.zeros(LOOKS, int)
        for start in range(0, trials, batch_size):
            end = min(start + batch_size, trials)
            size = end - start
            # One block has 2 anchor colour pairs = 4 candidate games.
            selected = rng.choice(len(support), size=(size, LOOKS, CHUNK // 2), p=weights)
            cats = categories[selected].reshape(size, LOOKS, CHUNK)
            counts = np.zeros((size, LOOKS, 9), int)
            for j in range(9):
                counts[..., j] = (cats == j).sum(axis=-1)
            counts = counts.cumsum(axis=1)
            final_bounds = bootstrap_bounds(counts[:, -1], .025)
            fixed_decisions[start:end] = np.where(final_bounds[:, 0] > 0, 1,
                                                  np.where(final_bounds[:, 1] < 0, -1, 0))
            active = np.arange(size)
            for k in range(LOOKS):
                if len(active) == 0:
                    break
                bounds = bootstrap_bounds(counts[active, k], TAIL_ALPHA[k])
                decisions = np.where(bounds[:, 0] > 0, 1, np.where(bounds[:, 1] < 0, -1, 0))
                crossed = decisions != 0
                done = np.ones(len(active), bool) if k == LOOKS - 1 else crossed
                idx = start + active[done]
                seq_decisions[idx] = decisions[done]
                games[idx] = 200 * (k + 1)
                stop_hist[k] += int(done.sum())
                active = active[~done]
        fixed = rate_summary(fixed_decisions, np.full(trials, 1600))
        seq = rate_summary(seq_decisions, games)
        if target:
            code = 1 if target > 0 else -1
            fixed["power"] = float((fixed_decisions == code).mean())
            seq["power"] = float((seq_decisions == code).mean())
        row = {"target_delta_elo": target, "population": population, "fixed": fixed,
               "sequential": seq, "stop_histogram": stop_hist.tolist(),
               "mean_games_saved": 1600 - seq["mean_games"],
               "design_disagreement_rate": float((seq_decisions != fixed_decisions).mean())}
        results.append(row)
        print(f"delta={target:+g} fixed better/worse={fixed['better']:.4f}/{fixed['worse']:.4f} "
              f"seq={seq['better']:.4f}/{seq['worse']:.4f} games={seq['mean_games']:.1f}", flush=True)
    return {"method": method_metadata(), "trials_per_delta": trials, "seed": seed,
            "batch_size": batch_size, "resampling": "symmetric exponentially tilted shared-opening blocks",
            "pool_blocks": n_blocks, "pool_support_points": len(support), "inventory": inventory,
            "pool_comparisons": [{k: x[k] for k in ("prefix", "candidate", "base", "manifest")}
                                 for x in comparisons if x["prefix"] == "std"],
            "fixed_design": "exact percentile bootstrap sign; infinite-draw limit of paired_series_delta",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    logs = Path(os.environ.get("LOGS_DIR", Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data")) / "logs"))
    for name in ("replay", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--logs", type=Path, default=logs)
        p.add_argument("--output", type=Path, help="optional JSON, inside the worktree")
        if name == "simulate":
            p.add_argument("--trials", type=int, default=20_000)
            p.add_argument("--seed", type=int, default=SIM_SEED)
            p.add_argument("--deltas", type=float, nargs="+", default=[0, 20, -20, 40, -40, 80, -80])
    args = parser.parse_args()
    if args.output and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("output must be inside this worktree")
    if args.command == "simulate" and args.trials < 2:
        parser.error("need at least two trials")
    result = replay(args.logs) if args.command == "replay" else simulate(args.logs, args.trials, args.seed, args.deltas)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
