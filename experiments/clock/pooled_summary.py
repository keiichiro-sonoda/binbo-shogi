#!/usr/bin/env python3
"""Preregistered stratified paired analysis; run only after all machines finish.

Usage: pooled_summary.py MACHINE_ROOT MACHINE_ROOT [...] --out pooled.json
The CLI requires exactly seeds 2026180000..2026183999 and always uses
20000 bootstrap replicates, numpy default_rng(180). No engine is launched.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path

# Keep numerical libraries to one thread even when inherited settings differ.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from paired_summary import summarize

ARMS = ('default', 'cap30', 'slow70')
REPS = 20000
SEED = 180
REGISTERED = range(2026180000, 2026184000)
RESOURCE_KEYS = ('threads', 'hash_mb', 'black_base_ms', 'white_base_ms',
                 'black_inc_ms', 'white_inc_ms', 'network_margin_ms',
                 'charge_quantum_ms', 'opening_plies', 'max_plies', 'a_color')


def portable_config(config):
    """Ignore machine-local paths, but retain command flags and option values."""
    config = copy.deepcopy(config)
    for spec in config.values():
        spec.pop('cwd', None)
        spec['argv'] = ['<path>' if i == 0 or Path(v).is_absolute() else v
                        for i, v in enumerate(spec['argv'])]
        if 'EvalDir' in spec.get('options', {}):
            spec['options']['EvalDir'] = '<path>'
    return config


def asset_roles(meta):
    """Compare hashes by side and role, never by basename or machine path."""
    assets = meta.get('assets_sha256')
    if assets is None:
        return None
    roles = {}
    for side, spec in meta['config'].items():
        paths = {'engine': Path(spec['argv'][0])}
        if 'EvalDir' in spec.get('options', {}):
            paths['eval'] = Path(spec['options']['EvalDir']) / 'nn.bin'
        for role, path in paths.items():
            # prepare() records the literal config paths; do not access assets.
            if str(path) not in assets or not assets[str(path)]:
                raise ValueError(f'missing recorded asset hash: {side}/{role}')
            roles[f'{side}/{role}'] = assets[str(path)]
    return roles


def signature(meta):
    args = meta['args']
    return dict(config=portable_config(meta['config']), assets=asset_roles(meta),
                resources={k: args[k] for k in RESOURCE_KEYS},
                intervention={k: args[k] for k in
                              ('rule', 'cap_ms', 'byoyomi_ms', 'k', 'a_option', 'opening_file')},
                script_sha256=meta['script_sha256'],
                ponder=meta.get('ponder'), clock=meta.get('clock'),
                cshogi_version=meta.get('cshogi_version'))


def stratified_bootstrap(strata, rng, reps=REPS, cluster_ids=None):
    """Each row is one seeded opening's paired differences for both arms.

    As in paired_summary, repeated opening_ids form one cluster within a
    machine. Resample each machine's original number of clusters, calculate
    its ratio mean, then weight it by its original number of seeded openings.
    Thus each machine retains its registered contribution even if sampled
    clusters have unequal sizes. Batching bounds temporary memory.
    """
    total = sum(len(values) for values in strata)
    clusters = []
    for m, values in enumerate(strata):
        ids = range(len(values)) if cluster_ids is None else cluster_ids[m]
        grouped = {}
        for key, value in zip(ids, values):
            grouped.setdefault(key, []).append(value)
        clusters.append((np.array([np.sum(v, axis=0) for v in grouped.values()]),
                         np.array([len(v) for v in grouped.values()]), len(values)))
    boot = np.empty((reps, 2))
    for start in range(0, reps, 100):
        n = min(100, reps - start)
        sums = np.zeros((n, 2))
        for totals, sizes, weight in clusters:
            ix = rng.integers(len(totals), size=(n, len(totals)))
            sums += weight * totals[ix].sum(axis=1) / sizes[ix].sum(axis=1)[:, None]
        boot[start:start+n] = sums / total
    return boot


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError('missing or nonfinite timing data')
    return dict(n=len(values), mean=float(values.mean()), min=float(values.min()),
                p25=float(np.quantile(values, .25)), median=float(np.median(values)),
                p75=float(np.quantile(values, .75)), max=float(values.max()))


def descriptive(rows):
    minima, first15, counts = [], [], []
    for row in rows:
        times = json.loads(row['black_ms'])
        remains = json.loads(row['black_remaining_trace'])
        if (not times or len(times) != len(remains)
                or not np.isfinite(times).all() or not np.isfinite(remains).all()
                or any(t < 0 for t in times)):
            raise ValueError('invalid timing arrays')
        minima.append(min(remains))
        first15.append(sum(times[:15]))
        counts.append(min(15, len(times)))
    return dict(wdl={key: sum(r['result'] == value for r in rows) for key, value in
                     [('wins', '1-0'), ('draws', '1/2-1/2'), ('losses', '0-1')]},
                time_forfeits=sum(r['reason'] == 'time_forfeit' for r in rows),
                minimum_remaining_ms=distribution(minima),
                sente_first15_own_moves_ms=distribution(first15),
                first15_observed_responses=distribution(counts))


def analyze(roots, *, registered_seeds=REGISTERED):
    """registered_seeds is injectable for synthetic unit tests only."""
    if len(roots) < 2:
        raise ValueError('at least two machine roots required')
    machines, strata, cluster_ids, signatures = [], [], [], None
    seen = set()
    pooled_rows = {arm: [] for arm in ARMS}
    scores = {'1-0': 1., '1/2-1/2': .4, '0-1': 0.}
    for root in map(Path, roots):
        paths = [root / arm for arm in ARMS]
        # Use the existing validator and return its exact descriptive output.
        paired = summarize(paths, REPS, SEED)
        groups, metas = {}, {}
        for arm, path in zip(ARMS, paths):
            meta = json.loads((path / 'run.json').read_text())
            if meta['args']['arm'] != arm:
                raise ValueError('subdirectory/arm mismatch')
            metas[arm] = meta
            group = {}
            for file in sorted(path.glob('*.tsv')):
                with file.open() as stream:
                    row = next(csv.DictReader(stream, delimiter='\t'))
                if row['result'] not in scores or float(row['black_score']) != scores[row['result']]:
                    raise ValueError('result/score mismatch')
                group[int(row['seed'])] = row
            groups[arm] = group
            pooled_rows[arm].extend(group.values())
        seeds = set(groups['default'])
        args = metas['default']['args']
        if seeds != set(range(args['seed'], args['seed'] + args['games'])):
            raise ValueError('run seed range mismatch')
        if seen & seeds:
            raise ValueError('overlapping machine opening seeds')
        seen.update(seeds)
        current = {arm: signature(metas[arm]) for arm in ARMS}
        matrix = root / 'matrix.json'
        current['matrix_script_sha256'] = (json.loads(matrix.read_text())['script_sha256']
                                           if matrix.exists() else None)
        if signatures is not None and current != signatures:
            raise ValueError('cross-machine configuration/resource/harness/asset mismatch')
        signatures = current
        strata.append(np.array([[float(groups[arm][seed]['black_score']) -
                                 float(groups['default'][seed]['black_score'])
                                 for arm in ARMS[1:]] for seed in sorted(seeds)]))
        cluster_ids.append([groups['default'][seed]['opening_id'] for seed in sorted(seeds)])
        machines.append(dict(root=str(root), openings=len(seeds), paired_summary=paired,
                             arms={arm: descriptive(list(groups[arm].values())) for arm in ARMS}))
    if seen != set(registered_seeds):
        raise ValueError('opening seed union is not exactly the registered set')
    boot = stratified_bootstrap(strata, np.random.default_rng(SEED), cluster_ids=cluster_ids)
    differences = np.concatenate(strata)
    comparisons = {}
    for i, arm in enumerate(ARMS[1:]):
        ci = np.quantile(boot[:, i], [.0125, .9875]).tolist()
        comparisons[arm] = dict(pairs=len(seen), delta=float(differences[:, i].mean()),
                               ci95=np.quantile(boot[:, i], [.025, .975]).tolist(),
                               ci97_5_bonferroni_two_arms=ci, **{'pass': ci[0] > 0})
    return dict(machines=machines, pooled=dict(
        openings=len(seen), comparisons=comparisons,
        arms={arm: descriptive(pooled_rows[arm]) for arm in ARMS}),
        method=dict(bootstrap_reps=REPS, bootstrap_seed=SEED,
                    sampling='opening_id clusters within machine; ratio means weighted by '
                             'fixed registered seeded-opening counts per machine',
                    score={'win': 1, 'draw': .4, 'loss': 0},
                    timing='sente only; ms; post-response remaining; first 15 searched responses '
                           'after the opening, including terminal responses; short games use all'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('machines', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        output = analyze(args.machines)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'analysis rejected: {exc}\n')
    with args.out.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('scope                 arm      pairs     delta        95% CI            97.5% CI       pass')
    for machine in output['machines']:
        for arm, result in machine['paired_summary'].items():
            print_line(Path(machine['root']).name, arm, result, 'descriptive')
    for arm, result in output['pooled']['comparisons'].items():
        print_line('pooled', arm, result, str(result['pass']))
    print(f'W/D/L, forfeits and timing distributions: {args.out}')


def print_line(scope, arm, result, decision):
    lo, hi = result['ci95']
    blo, bhi = result['ci97_5_bonferroni_two_arms']
    print(f'{scope:21} {arm:7} {result["pairs"]:5} {result["delta"]:+9.5f} '
          f'[{lo:+.5f},{hi:+.5f}] [{blo:+.5f},{bhi:+.5f}] {decision}')


if __name__ == '__main__':
    main()
