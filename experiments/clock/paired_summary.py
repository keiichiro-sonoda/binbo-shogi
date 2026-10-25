#!/usr/bin/env python3
"""Strict paired comparison, cluster percentile bootstrap by opening; stdlib + numpy."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

def summarize(paths, reps=20000, seed=180):
    runs = {}
    rows = defaultdict(dict)
    for directory in paths:
        directory = Path(directory)
        meta = json.loads((directory/'run.json').read_text())
        arm = meta['args']['arm']
        if arm in runs:
            raise ValueError('one complete directory per arm required')
        runs[arm] = meta
        for p in sorted(directory.glob('*.tsv')):
            with p.open() as f:
                data = list(csv.DictReader(f, delimiter='\t'))
            if len(data) != 1:
                raise ValueError('expected one game: '+str(p))
            r = data[0]
            if r['result'] == 'INVALID' or r['black'] != 'A' or r['arm'] != arm:
                raise ValueError('invalid game/color/arm: '+str(p))
            key = int(r['seed'])
            if key in rows[arm]:
                raise ValueError('duplicate seed: '+str(p))
            rows[arm][key] = r
        if len(rows[arm]) != meta['args']['games']:
            raise ValueError('incomplete run: '+str(directory))
    if 'default' not in rows:
        raise ValueError('default arm missing')
    base = rows['default']
    reference = runs['default']
    matches = ['threads','hash_mb','black_base_ms','white_base_ms','black_inc_ms','white_inc_ms',
               'network_margin_ms','charge_quantum_ms','opening_plies','max_plies','a_color']
    result = {}
    rng = np.random.default_rng(seed)
    for arm, group in rows.items():
        if arm == 'default':
            continue
        if set(group) != set(base):
            raise ValueError('missing/unmatched seeds: '+arm)
        if any(runs[arm]['args'][k] != reference['args'][k] for k in matches):
            raise ValueError('clock/resource mismatch: '+arm)
        if runs[arm]['config'] != reference['config']:
            raise ValueError('engine/eval configuration mismatch: '+arm)
        if runs[arm].get('assets_sha256') != reference.get('assets_sha256'):
            raise ValueError('engine/eval binary hash mismatch: '+arm)
        if runs[arm]['script_sha256'] != reference['script_sha256']:
            raise ValueError('harness version mismatch: '+arm)
        clusters = defaultdict(list)
        for k, r in group.items():
            if r['opening'] != base[k]['opening'] or r['opening_id'] != base[k]['opening_id']:
                raise ValueError('opening mismatch: '+str(k))
            clusters[r['opening_id']].append(float(r['black_score'])-float(base[k]['black_score']))
        totals = np.array([sum(v) for v in clusters.values()])
        sizes = np.array([len(v) for v in clusters.values()])
        boot = np.empty(reps)
        for start in range(0,reps,100):
            n = min(100,reps-start)
            ix = rng.integers(len(totals),size=(n,len(totals)))
            boot[start:start+n] = totals[ix].sum(axis=1)/sizes[ix].sum(axis=1)
        diffs = np.array([x for v in clusters.values() for x in v])
        result[arm] = {'pairs':len(group), 'opening_clusters':len(clusters),
                       'default_score':float(np.mean([float(r['black_score']) for r in base.values()])),
                       'arm_score':float(np.mean([float(r['black_score']) for r in group.values()])),
                       'delta':float(diffs.mean()), 'paired_variance':float(diffs.var(ddof=1)) if len(diffs)>1 else None,
                       'ci95':np.quantile(boot,[.025,.975]).tolist(),
                       'ci97_5_bonferroni_two_arms':np.quantile(boot,[.0125,.9875]).tolist(),
                       'forfeits_default':sum(r['reason']=='time_forfeit' for r in base.values()),
                       'forfeits_arm':sum(r['reason']=='time_forfeit' for r in group.values()),
                       'wdl_default':{key:sum(r['result']==value for r in base.values())
                                      for key,value in [('wins','1-0'),('draws','1/2-1/2'),('losses','0-1')]},
                       'wdl_arm':{key:sum(r['result']==value for r in group.values())
                                  for key,value in [('wins','1-0'),('draws','1/2-1/2'),('losses','0-1')]},
                       'bootstrap_reps':reps, 'bootstrap_seed':seed}
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+')
    p.add_argument('--out',required=True)
    p.add_argument('--reps',type=int,default=20000)
    p.add_argument('--seed',type=int,default=180)
    a = p.parse_args()
    if a.reps < 100:
        p.error('at least 100 bootstrap draws')
    output = summarize(a.runs,a.reps,a.seed)
    with open(a.out,'x') as f:
        json.dump(output,f,indent=2)
        f.write('\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':
    main()
