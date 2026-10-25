#!/usr/bin/env python3
"""Task 176: read-only timing audit and paired sequential-design simulation.

Only numpy is needed. No engine, torch, training, or existing experiment module
is imported. Inputs are never opened for writing. Output directories must be new.
Run after sourcing scripts/env.sh; see reports/task176_judge_cost.md.

The main population symmetrizes observed candidate/baseline opening outcomes,
then exponentially tilts their probabilities to an exact pooled Elo effect.
This is not a claim to know the true effects of old nets. An opening is sampled
with both colors AND both anchors intact. A donor net is chosen once per
replicate, retaining heterogeneity in the observed variances. Centered Elo
influence-function and whole-chunk resampling are additional sensitivities.
"""

from __future__ import annotations

import os

# Enforce a small CPU footprint even when the invoking environment requests more.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

SETS = ("s280", "s281", "s282")
ANCHORS = ("tanuki20000", "aoba10000")
BASE129 = "task129_student_64m_qsrel_e12"
BASE142 = "task142_finetune"
BASE154 = "task154_R_result30"
PATTERN = re.compile(r"(s28[012]|conf[^_]*)_(.+)_(tanuki20000|aoba10000|micro11000)"
                     r"_c(\d+)_seed(\d+)_n100.tsv$")
LOOKS = np.arange(200, 1201, 200)  # opening clusters; each costs four candidate games
RULES = ("fixed_current", "fixed_cluster", "naive_peek", "obf", "obf_cp10",
         "hp_cp10", "futility_only", "sprt20")


def elo(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 400 / math.log(10) * np.log(p / (1 - p))


def derivative(p):
    return 400 / math.log(10) / (p * (1 - p))


def read_chunk(path):
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    rows = [s.split("\t") for s in lines if len(s.split("\t")) == 21 and s.split("\t")[0].isdigit()]
    summary = next((float(s.split("=", 1)[1]) for s in lines if s.startswith("elapsed_ms=")), None)
    valid = (len(rows) == 100 and summary is not None and "status=ok" in lines
             and [int(r[0]) for r in rows] == list(range(1, 101))
             and all(r[3] in ("win", "draw", "loss") for r in rows))
    if valid:
        for a, b in zip(rows[::2], rows[1::2]):
            valid &= a[1] == "black" and b[1] == "white" and a[2] == b[2] and a[20] == b[20]
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), rows=rows,
                elapsed_ms=summary, valid=bool(valid), mtime=path.stat().st_mtime)


def inventory(logs, snapshot=None):
    chunks, rejected, manifest = {}, [], []
    frozen = json.loads(snapshot.read_text()) if snapshot else None
    paths = [Path(x["path"]) for x in frozen if x["valid"]] if frozen else sorted(logs.glob("*.tsv"))
    expected = {x["path"]: x["sha256"] for x in frozen} if frozen else {}
    for path in paths:
        m = PATTERN.fullmatch(path.name)
        if not m:
            continue
        s, net, anchor, c, seed = m.groups()
        obj = read_chunk(path)
        if frozen and obj["sha256"] != expected[str(path)]:
            raise ValueError(f"snapshot changed: {path}")
        manifest.append({k: obj[k] for k in ("path", "sha256", "valid")})
        key = (net, s, anchor, int(c))
        if obj["valid"]:
            obj["seed"] = int(seed)
            chunks[key] = obj
        else:
            rejected.append(path.name)
    return chunks, rejected, manifest


def full_nets(chunks):
    return [n for n in sorted({k[0] for k in chunks})
            if all((n, s, a, c) in chunks for s in SETS for a in ANCHORS for c in range(8))]


def matrix(chunks, net):
    # [set, chunk, opening, anchor, color]
    scores = np.empty((3, 8, 50, 2, 2))
    keys = []
    for si, s in enumerate(SETS):
        for c in range(8):
            for ai, a in enumerate(ANCHORS):
                obj = chunks[net, s, a, c]
                rows = obj["rows"]
                expected_seed = 20280000 + si * 1000 + 100 * c
                if obj["seed"] != expected_seed:
                    raise ValueError(f"unexpected seed: {obj['path']}")
                scores[si, c, :, ai] = np.array([
                    {"win": 1., "draw": .5, "loss": 0.}[r[3]] for r in rows]).reshape(50, 2)
                keys.append(tuple((r[2], r[1], r[20]) for r in rows))
                if ai and keys[-1] != keys[-2]:
                    raise ValueError(f"anchors do not share openings: {net}, {s}, {c}")
    return scores, keys


def baseline(net):
    task = int(re.match(r"task(\d+)", net)[1])
    if task <= 142:
        return BASE129
    if task == 145:
        return BASE129  # primary historical comparison; Task 142 was also reported
    return BASE142 if task <= 154 else BASE154


def moments(cand, base):
    ca, ba = cand.mean(axis=-1).reshape(-1, 2), base.mean(axis=-1).reshape(-1, 2)
    pc, pb = ca.mean(), ba.mean()
    psi = derivative(pc) * (ca - pc) - derivative(pb) * (ba - pb)
    delta = float(elo(pc) - elo(pb))
    se_current = float(np.std(psi, ddof=1) / math.sqrt(psi.size))
    se_cluster = float(np.std(psi.mean(axis=1), ddof=1) / math.sqrt(len(psi)))
    diff = (cand - base).reshape(-1, 2, 2)
    return dict(delta=delta, se_current=se_current, se_cluster=se_cluster,
                anchor_correlation=float(np.corrcoef(psi.T)[0, 1]),
                color_pair_variance_ratio=float(2 * np.var(diff.mean(axis=-1), ddof=1)
                                                / np.var(diff, ddof=1)),
                covariance_variance_ratio=(se_cluster / se_current) ** 2), psi


def list_schedule(durations, parallel=8):
    slots = [0.] * parallel
    for duration in durations:
        end = heapq.heappop(slots) + duration
        heapq.heappush(slots, end)
    return max(slots)


def timing(chunks, nets):
    profiles, groups = [], defaultdict(list)
    for (net, s, anchor, c), obj in chunks.items():
        if s not in SETS or anchor not in ANCHORS or net not in nets:
            continue
        rows = np.asarray([[float(r[i]) for i in range(5, 15)] for r in obj["rows"]])
        game_ms = rows[:, 1].sum()
        a_ms, b_ms = np.sum(rows[:, 2] * rows[:, 5]), np.sum(rows[:, 6] * rows[:, 9])
        groups[anchor].append([100, rows[:, 0].sum(), game_ms, a_ms, b_ms,
                               rows[:, 2].sum(), rows[:, 6].sum(),
                               np.sum(rows[:, 2] * rows[:, 4]), np.sum(rows[:, 6] * rows[:, 8]),
                               obj["elapsed_ms"] - game_ms])
    for net in nets:
        objs = [chunks[net, s, a, c] for s in SETS for a in ANCHORS for c in range(8)]
        ds = [x["elapsed_ms"] / 1000 for x in objs]
        wave = sum(max(ds[k:k+8]) for k in range(0, 48, 8))
        queue = list_schedule(ds)
        # Summary excludes engine initialization; mtime is merely an interval proxy.
        ends = [x["mtime"] for x in objs]
        span = max(ends) - min(e - d for e, d in zip(ends, ds))
        interleaved = [chunks[net, s, a, c]["elapsed_ms"] / 1000
                       for c in range(8) for s in SETS for a in ANCHORS]
        profiles.append(dict(net=net, wave_s=wave, queue_s=queue,
                             queue_saving=1-queue/wave, idle_fraction=1-sum(ds)/(8*wave),
                             ideal_s=sum(ds)/8, mtime_span_s=span,
                             interleaved_wave_s=sum(max(interleaved[k:k+8]) for k in range(0, 48, 8))))
        gaps = [min(x["mtime"]-x["elapsed_ms"]/1000 for x in objs[k:k+8])
                - max(x["mtime"] for x in objs[k-8:k]) for k in range(8, 48, 8)]
        profiles[-1]["interwave_gap_sum_s"] = sum(gaps)
    aggregate = {}
    for anchor, vals in groups.items():
        x = np.sum(vals, axis=0)
        n, plies, game, a, b, am, bm, an, bn, residual = x
        aggregate[anchor] = dict(games=int(n), plies=plies/n, game_ms=game/n,
                                cand_ms=a/n, anchor_ms=b/n, outside_think_ms=(game-a-b)/n,
                                cand_ms_per_move=a/am, anchor_ms_per_move=b/bm,
                                cand_nodes_per_move=an/am, anchor_nodes_per_move=bn/bm,
                                summary_minus_rows_ms_per_game=residual/n)
    return aggregate, profiles


def calibrate(seed, size=300000):
    rng = np.random.default_rng(seed)
    t = LOOKS / LOOKS[-1]
    z = np.cumsum(rng.standard_normal((size, 6)) * np.sqrt(np.diff(np.r_[0, t])), axis=1)
    return float(np.quantile(z.max(axis=1), .975))


def hp_calibration(seed, size=300000):
    rng = np.random.default_rng(seed)
    t = LOOKS/1200
    z = np.cumsum(rng.standard_normal((size, 6))*np.sqrt(np.diff(np.r_[0, t])), axis=1)/np.sqrt(t)
    early = (z[:, :-1] >= 3.5).any(axis=1)
    alpha_left = (.025-early.mean())/(1-early.mean())
    return float(np.quantile(z[~early, -1], 1-alpha_left))


def decide(means, ses, sigmas, c):
    # All arrays [replicate, look]; ses are opening-cluster standard errors.
    n = LOOKS[None, :]
    z = means / ses
    efficacy = z >= c / np.sqrt(n / 1200)
    remaining = 1200 - n
    cp_boundary = (c * sigmas * math.sqrt(1200) - 10 * remaining
                   + NormalDist().inv_cdf(.01) * sigmas * np.sqrt(remaining))
    futility = (means * n <= cp_boundary) & (remaining > 0)
    llr = 20 * n * (means - 10) / (sigmas ** 2)
    upper, lower = math.log(.9/.025), math.log(.1/.975)
    out = {}
    for rule in RULES[2:]:
        if rule == "naive_peek":
            yes, no = z >= 1.96, z <= -1.96
        elif rule == "obf":
            yes, no = efficacy, z <= -c / np.sqrt(n / 1200)
        elif rule == "obf_cp10":
            yes, no = efficacy, futility
        elif rule in ("hp_cp10", "futility_only"):
            final = 1.97 if rule == "hp_cp10" else 1.96
            upper_z = np.array([3.5]*5+[final]) if rule == "hp_cp10" else np.array([np.inf]*5+[final])
            bound = (final*sigmas*math.sqrt(1200)-10*remaining
                     + NormalDist().inv_cdf(.01)*sigmas*np.sqrt(remaining))
            yes, no = z >= upper_z, (means*n <= bound) & (remaining > 0)
        else:
            yes, no = llr >= upper, llr <= lower
        stopped = yes | no
        stopped[:, -1] = True
        first = np.argmax(stopped, axis=1)
        ix = np.arange(len(first))
        out[rule] = (LOOKS[first] * 4, yes[ix, first], no[ix, first])
    return out


def simulate(donors, reps, seed, c, block=False):
    rng = np.random.default_rng(seed)
    results = {effect: {rule: [0., 0, 0, 0, 0] for rule in RULES} for effect in (0, 10, 20, 30)}
    for start in range(0, reps, 256):
        count = min(256, reps-start)
        which = rng.integers(0, len(donors), count)
        # Donors are centered globally, preserving between-set heterogeneity.
        source = donors[which]
        sums = np.zeros((count, 2))
        sq = np.zeros((count, 2))
        cross = np.zeros(count)
        means, sec, sei = [], [], []
        for n in LOOKS:
            if block:
                # Sensitivity: entire 50-opening chunks sampled as clusters.
                ids = rng.integers(0, 24, (count, 4))
                samples = source.reshape(count, 24, 50, 2)[np.arange(count)[:, None], ids].reshape(count, 200, 2)
            else:
                ids = rng.integers(0, 1200, (count, 200))
                samples = source[np.arange(count)[:, None], ids]
            sums += samples.sum(axis=1)
            sq += (samples ** 2).sum(axis=1)
            cross += (samples[:, :, 0] * samples[:, :, 1]).sum(axis=1)
            means.append(sums.mean(axis=1)/n)
            vm = ((sq.sum(axis=1)+2*cross)/4 - (sums.mean(axis=1)**2)/n)/(n-1)
            vi = (sq.sum(axis=1) - sums.sum(axis=1)**2/(2*n))/(2*n-1)
            sec.append(np.sqrt(vm/n))
            sei.append(np.sqrt(vi/(2*n)))
        means, sec, sei = np.array(means).T, np.array(sec).T, np.array(sei).T
        sigma = sec * np.sqrt(LOOKS)
        for effect in results:
            m = means + effect
            decisions = decide(m, sec, sigma, c)
            for rule, se in (("fixed_current", sei), ("fixed_cluster", sec)):
                z = m[:, -1] / se[:, -1]
                decisions[rule] = (np.full(count, 4800), z > 1.96, z < -1.96)
            for rule, (games, yes, no) in decisions.items():
                v = results[effect][rule]
                v[0] += games.sum()
                v[1] += yes.sum()
                v[2] += no.sum()
                v[3] += ((games < 4800) & yes).sum()
                v[4] += ((games < 4800) & no).sum()
    return [dict(effect=effect, rule=rule, reps=reps, expected_games=float(v[0]/reps),
                 promote=float(v[1]/reps), reject=float(v[2]/reps),
                 inconclusive=float(1-(v[1]+v[2])/reps),
                 miss_positive=float(1-v[1]/reps) if effect else None,
                 early_promote=float(v[3]/reps), early_reject=float(v[4]/reps))
            for effect, rules in results.items() for rule, v in rules.items()]


def tilted_population(cand, base, effect):
    ca, ba = cand.mean(axis=-1).reshape(1200, 2), base.mean(axis=-1).reshape(1200, 2)
    pop = np.concatenate((np.c_[ca, ba], np.c_[ba, ca]))
    diff = (pop[:, :2]-pop[:, 2:]).mean(axis=1)
    def tilted(lam):
        logw = lam*diff
        prob = np.exp(logw-logw.max())
        prob /= prob.sum()
        avg = prob @ pop
        return prob, float(elo(avg[:2].mean())-elo(avg[2:].mean()))
    lo, hi = -100., 100.
    for _ in range(60):
        mid = (lo+hi)/2
        if tilted(mid)[1] < effect:
            lo = mid
        else:
            hi = mid
    prob, actual = tilted((lo+hi)/2)
    if abs(actual-effect) > 1e-8:
        raise ValueError("tilted effect calibration failed")
    return pop, np.cumsum(prob), actual


def simulate_discrete(pairs, reps, seed, c):
    rng = np.random.default_rng(seed)
    output, audit = [], []
    for effect in (0, 10, 20, 30):
        populations = [tilted_population(ca, ba, effect) for ca, ba in pairs]
        audit.append(dict(effect=effect, actual_min=min(x[2] for x in populations),
                          actual_max=max(x[2] for x in populations)))
        totals = {r: np.zeros(5) for r in RULES}
        for start in range(0, reps, 256):
            count = min(256, reps-start)
            which = rng.integers(0, len(pairs), count)
            sums, second = np.zeros((count, 4)), np.zeros((count, 4, 4))
            means, sec, sei = [], [], []
            for n in LOOKS:
                samples = np.empty((count, 200, 4))
                for di in np.unique(which):
                    mask = which == di
                    pop, cdf, _ = populations[di]
                    ids = np.searchsorted(cdf, rng.random((mask.sum(), 200)))
                    samples[mask] = pop[ids]
                sums += samples.sum(axis=1)
                second += np.einsum("nki,nkj->nij", samples, samples)
                pc, pb = sums[:, :2].mean(axis=1)/n, sums[:, 2:].mean(axis=1)/n
                dc, db = derivative(pc), derivative(pb)
                coef = np.stack((dc, dc, -db, -db), axis=1)/2
                centered = second - sums[:, :, None]*sums[:, None, :]/n
                vc = np.einsum("ni,nij,nj->n", coef, centered, coef)/(n-1)
                # Existing bootstrap treats the 2*n anchor-specific pairs as IID.
                uncentered_sum = (dc**2*(second[:, 0, 0]+second[:, 1, 1])
                                  + db**2*(second[:, 2, 2]+second[:, 3, 3])
                                  - 2*dc*db*(second[:, 0, 2]+second[:, 1, 3]))
                vi = (uncentered_sum-2*n*(dc*pc-db*pb)**2)/(2*n-1)
                means.append(elo(pc)-elo(pb))
                sec.append(np.sqrt(vc/n))
                sei.append(np.sqrt(vi/(2*n)))
            means, sec, sei = np.array(means).T, np.array(sec).T, np.array(sei).T
            decisions = decide(means, sec, sec*np.sqrt(LOOKS), c)
            for rule, se in (("fixed_current", sei), ("fixed_cluster", sec)):
                z = means[:, -1]/se[:, -1]
                decisions[rule] = (np.full(count, 4800), z > 1.96, z < -1.96)
            for rule, (games, yes, no) in decisions.items():
                totals[rule] += [games.sum(), yes.sum(), no.sum(),
                                 ((games < 4800) & yes).sum(), ((games < 4800) & no).sum()]
        for rule, v in totals.items():
            output.append(dict(effect=effect, rule=rule, reps=reps,
                               expected_games=float(v[0]/reps), promote=float(v[1]/reps),
                               reject=float(v[2]/reps), inconclusive=float(1-(v[1]+v[2])/reps),
                               miss_positive=float(1-v[1]/reps) if effect else None,
                               early_promote=float(v[3]/reps), early_reject=float(v[4]/reps)))
    return output, audit


def replay(cand, base, c):
    # Preserve each full 100-game chunk and its internal game order. Interleave
    # sets, then both anchors. Four set/chunk units make an 800-game look.
    ids = np.array([si*400+ci*50+i for ci in range(8) for si in range(3) for i in range(50)])
    ca = cand.mean(axis=-1).reshape(1200, 2)[ids]
    ba = base.mean(axis=-1).reshape(1200, 2)[ids]
    means, ses = [], []
    for n in LOOKS:
        pc, pb = ca[:n].mean(), ba[:n].mean()
        psi = derivative(pc)*(ca[:n]-pc)-derivative(pb)*(ba[:n]-pb)
        means.append(float(elo(pc)-elo(pb)))
        ses.append(float(psi.mean(axis=1).std(ddof=1)/math.sqrt(n)))
    m, se = np.array([means]), np.array([ses])
    return {k: dict(games=int(v[0][0]), promote=bool(v[1][0]), reject=bool(v[2][0]))
            for k, v in decide(m, se, se*np.sqrt(LOOKS), c).items()}


def ordering_trace(cand, base):
    old_c = cand.transpose(0, 3, 1, 2, 4).reshape(2400, 2).mean(axis=1)
    old_b = base.transpose(0, 3, 1, 2, 4).reshape(2400, 2).mean(axis=1)
    new_c = cand.transpose(1, 0, 2, 3, 4).reshape(2400, 2).mean(axis=1)
    new_b = base.transpose(1, 0, 2, 3, 4).reshape(2400, 2).mean(axis=1)
    return dict(old_delta=[float(elo(old_c[:n*2].mean())-elo(old_b[:n*2].mean())) for n in LOOKS],
                interleaved_delta=[float(elo(new_c[:n*2].mean())-elo(new_b[:n*2].mean())) for n in LOOKS])


def bootstrap_check(cand, base, rng, reps=20000):
    # Exact estimator and percentile-bootstrap definition from paired_series_delta,
    # aggregated to its 25 possible (candidate pair mean, base pair mean) cells.
    a, b = cand.mean(axis=-1).ravel(), base.mean(axis=-1).ravel()
    cell = (np.rint(4*a).astype(int)*5 + np.rint(4*b).astype(int))
    probs = np.bincount(cell, minlength=25)/len(cell)
    counts = rng.multinomial(len(cell), probs, reps)
    grid = np.arange(25)
    ds = elo(counts @ (grid//5/4) / len(cell)) - elo(counts @ (grid%5/4) / len(cell))
    return list(map(float, np.quantile(ds, [.025, .975])))


def confirmation_audit(chunks):
    out = []
    for prefix, cand, base in (("conf154", BASE154, BASE142),
                               ("conf168", "task166_S_R2", BASE154)):
        for anchors in (ANCHORS, ("micro11000",)):
            ca, ba = [], []
            for anchor in anchors:
                for ci in range(8):
                    key = (cand, prefix, anchor, ci)
                    other = (base, prefix, anchor, ci)
                    if key not in chunks or other not in chunks:
                        break
                    x, y = chunks[key], chunks[other]
                    if [(r[2], r[1], r[20]) for r in x["rows"]] != [(r[2], r[1], r[20]) for r in y["rows"]]:
                        raise ValueError("confirmation alignment mismatch")
                    for dst, obj in ((ca, x), (ba, y)):
                        dst.extend({"win": 1., "draw": .5, "loss": 0.}[r[3]] for r in obj["rows"])
            if len(ca) == len(anchors)*800:
                out.append(dict(prefix=prefix, candidate=cand, anchors=list(anchors),
                                games_per_arm=len(ca), delta=float(elo(np.mean(ca))-elo(np.mean(ba)))))
    return out


def hardware(logs, chunks):
    # Restrict to task166's first full frame: a contiguous observed eight-way run.
    objs = [v for k, v in chunks.items() if k[0] == "task166_S_R1" and k[1] in SETS and k[2] in ANCHORS]
    if len(objs) != 48:
        return {}
    intervals = [(v["mtime"]-v["elapsed_ms"]/1000, v["mtime"]) for v in objs]
    lo, hi = min(a for a, b in intervals), max(b for a, b in intervals)
    values = []
    for path in sorted((logs/"hw").glob("hw_*.jsonl")):
        # Filename's local date is only a coarse filter; timestamps decide membership.
        if path.name < "hw_20261022.jsonl" or path.name > "hw_20261024.jsonl":
            continue
        for line in path.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if lo <= row["t"] <= hi and sum(a <= row["t"] <= b for a, b in intervals) == 8:
                values.append(row)
    def stats(key):
        x = [r[key] for r in values if isinstance(r.get(key), (int, float))]
        return dict(mean=float(np.mean(x)), p95=float(np.quantile(x, .95)), max=float(max(x))) if x else None
    return dict(window_epoch=[lo, hi], samples=len(values), cpu_util=stats("cpu_util"),
                ram_used_gb=stats("ram_used_gb"), swap_used_gb=stats("swap_used_gb"),
                cpu_temp=stats("cpu_temp"))


def write_csv(path, rows):
    if rows:
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data = Path(os.environ.get("DATA_ROOT", Path.home()/"binbo-shogi-data"))
    p.add_argument("--logs", type=Path, default=Path(os.environ.get("LOGS_DIR", data/"logs")))
    p.add_argument("--out", type=Path, required=True, help="new directory (never overwrite)")
    p.add_argument("--reps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=176202610)
    p.add_argument("--manifest", type=Path, help="freeze valid inputs from an earlier output manifest")
    args = p.parse_args()
    if args.reps < 1000:
        p.error("--reps must be >=1000")
    args.out.mkdir(parents=True, exist_ok=False)
    chunks, rejected, manifest = inventory(args.logs, args.manifest)
    nets = full_nets(chunks)
    mats = {n: matrix(chunks, n) for n in nets}
    aggregate, profiles = timing(chunks, nets)
    raw_c = calibrate(args.seed)
    c = 2.06  # round conservatively; this is the pre-registerable constant
    comparisons, donors, donor_names, donor_pairs = [], [], [], []
    rng = np.random.default_rng(args.seed+1)
    for net in nets:
        if not net.startswith("task") or net == BASE129:
            continue
        b = baseline(net)
        ca, ck = mats[net]
        ba, bk = mats[b]
        if ck != bk:
            raise ValueError(f"opening/color mismatch: {net}, {b}")
        st, psi = moments(ca, ba)
        ci = bootstrap_check(ca, ba, rng)
        st.update(net=net, base=b, bootstrap_ci95=ci, replay=replay(ca, ba, c),
                  ordering=ordering_trace(ca, ba))
        comparisons.append(st)
        if abs(st["delta"]) <= 40:
            # psi has expectation zero by construction; flattening pools six strata.
            donors.append(psi-psi.mean())
            donor_names.append(net)
            donor_pairs.append((ca, ba))
    sims, tilt_audit = simulate_discrete(donor_pairs, args.reps, args.seed+2, c)
    influence = simulate(np.array(donors), args.reps, args.seed+3, c)
    sensitivity = simulate(np.array(donors), args.reps, args.seed+4, c, block=True)
    summary = dict(seed=args.seed, reps=args.reps, threads=1, full_nets=nets,
                   numpy_version=np.__version__, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   input_manifest=str(args.manifest) if args.manifest else None,
                   valid_chunks=len(chunks), rejected=rejected,
                   donor_names=donor_names, obf_c=c, raw_calibrated_c=raw_c, tilt_audit=tilt_audit,
                   raw_hp_final_z=hp_calibration(args.seed+5), hp_z=[3.5]*5+[1.97],
                   obf_z=list(c/np.sqrt(LOOKS/1200)), timing=aggregate,
                   hardware=hardware(args.logs, chunks),
                   comparisons=comparisons, confirmation=confirmation_audit(chunks),
                   model="symmetrized opening outcomes exponentially tilted to exact pooled Elo effects",
                   limitations=["true effects imposed, not observed labels", "stationary opening population",
                                "normal/delta-method CI; exact historical percentile CI audited separately",
                                "SPRT is normal plug-in approximation, capped with no fallback acceptance",
                                "mtime interval proxy excludes startup, gaps can be resumes",
                                "no engine or concurrency benchmark performed"])
    (args.out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_csv(args.out/"simulation.csv", sims)
    write_csv(args.out/"influence_sensitivity.csv", influence)
    write_csv(args.out/"chunk_sensitivity.csv", sensitivity)
    write_csv(args.out/"timing.csv", profiles)
    print(json.dumps(dict(full_nets=len(nets), donors=len(donors), out=str(args.out),
                          obf_c=c, rejected=len(rejected)), ensure_ascii=False))


if __name__ == "__main__":
    main()
