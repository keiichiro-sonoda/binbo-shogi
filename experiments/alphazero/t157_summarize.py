#!/usr/bin/env python3
"""Task 157: engine comparisons, paired bootstrap, and portable numbers JSON."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
from t156_selfplay_audit import save
from t157_collect import DATA, CATEGORIES
from t157_outcomes import ci

NAMES = ('task142_finetune', 'task154_R_result30', 'micro')


def regression(sample, evaluations, reference, target, static=False):
    field = 'static_value' if static else 'cp'
    ok = np.array([evaluations[reference][i]['kind'] == 'cp' and evaluations[target][i]['kind'] == 'cp'
                   for i in range(len(sample))])
    rows = [s for i, s in enumerate(sample) if ok[i]]
    old = np.array([s[field] for i, s in enumerate(evaluations[reference]) if ok[i]])
    new = np.array([s[field] for i, s in enumerate(evaluations[target]) if ok[i]])
    x = old/400
    rook = np.array([s['rook'] for s in rows])
    king = np.array([s['king'] for s in rows])
    wing_advantage = (king[:, 0] == 2).astype(float) - (king[:, 1] == 2)
    plies = np.array([s['ply'] for s in rows])
    X = np.column_stack([np.ones(len(x)), x, x*abs(x), plies == 60, plies == 80,
                         rook[:, 0] == 1, rook[:, 0] == 2,
                         rook[:, 1] == 1, rook[:, 1] == 2, wing_advantage])
    def fit(ix):
        beta = np.linalg.lstsq(X[ix], new[ix], rcond=None)[0]
        # Native micro units differ. Normalize by the near-equality slope.
        return beta[-1], beta[-1]*400/beta[1], beta[1]/400
    estimate = fit(np.arange(len(x)))
    rng = np.random.default_rng(20271576)
    boot = np.array([fit(rng.integers(0, len(x), len(x))) for _ in range(2000)])
    return dict(n=len(x), unit='internal Value' if static else 'displayed cp',
                corr=float(np.corrcoef(old, new)[0,1]),
                target_wing_coefficient=ci(estimate[0], boot[:, 0]),
                reference_scale_wing_coefficient=ci(estimate[1], boot[:, 1]),
                target_per_reference_slope=ci(estimate[2], boot[:, 2]),
                design=['intercept','reference/400','reference*abs(reference)/160000',
                        'ply60','ply80','own_rook_furi','own_rook_other',
                        'opp_rook_furi','opp_rook_other','own_wing-opp_wing'])


def summarize(a):
    sample = json.loads((a.out/'engine_sample.json').read_text())
    engines = {name: json.loads((a.out/f'engine_{name}.json').read_text()) for name in NAMES}
    ev = {name: report['probes'] for name, report in engines.items()}
    assert len(set(s['game_id'] for s in sample)) == len(sample)
    for name in NAMES:
        assert [(r['game_id'],r['ply']) for r in ev[name]] == [(r['game_id'],r['ply']) for r in sample]
    rng = np.random.default_rng(20271577)
    central = np.array([r['king'][0] < 2 for r in sample])
    policy = {}
    for name in NAMES:
        rows = [r for i,r in enumerate(ev[name]) if central[i]]
        policy[name] = dict(n=len(rows))
        for key in ('king', 'outward', 'inward'):
            y = np.array([r[key] for r in rows], float)
            boot = y[rng.integers(0, len(y), (2000, len(y)))].mean(axis=1)
            policy[name][key] = dict(count=int(y.sum()), **ci(y.mean(), boot))
    for first, second in ((NAMES[0], NAMES[1]), (NAMES[0], 'micro')):
        y = np.array([ev[second][i]['outward']-ev[first][i]['outward']
                      for i in range(len(sample)) if central[i]], float)
        policy[f'{second}_minus_{first}_outward'] = ci(y.mean(), y[rng.integers(0,len(y),(2000,len(y)))].mean(axis=1))
    rollout = {}
    arrays = {}
    for name in NAMES[:2]:
        rows = engines[name]['rollouts']
        arrays[name] = np.array([r['king'] for r in rows])
        rollout[name] = dict(games=len(rows), reached60=sum(len(r['moves']) == 60 for r in rows),
                             searches=sum(r['searches'] for r in rows),
                             mean_opening_plies=float(np.mean([len(r['opening']) for r in rows])),
                             categories={cat: int(np.sum(arrays[name] == j)) for j,cat in enumerate(CATEGORIES)},
                             mean_king_moves=float(np.mean([r['king_moves'] for r in rows])))
    old, new = (engines[n]['rollouts'] for n in NAMES[:2])
    assert [r['opening'] for r in old] == [r['opening'] for r in new]
    common60 = np.array([len(o['moves']) == len(n['moves']) == 60 for o, n in zip(old, new)])
    rollout['common_reached60_games'] = int(common60.sum())
    # Fixed-denominator intention-to-treat: a stopped game uses its final position.
    # Report reached60; if either arm stops, this is not a pure ply60 distribution.
    for j, cat in enumerate(CATEGORIES):
        change = ((arrays[NAMES[1]] == j).mean(axis=1)-(arrays[NAMES[0]] == j).mean(axis=1))
        boot = change[rng.integers(0, len(change), (10000, len(change)))].mean(axis=1)
        rollout[f'delta_{cat}'] = ci(change.mean(), boot)
        common = change[common60]
        b = common[rng.integers(0,len(common),(10000,len(common)))].mean(axis=1)
        rollout[f'delta_{cat}_common60'] = ci(common.mean(), b)
    for name in NAMES[:2]:
        rollout[name]['common60_categories'] = {cat: int(np.sum(arrays[name][common60] == j))
                                               for j,cat in enumerate(CATEGORIES)}
    resource_summary = {}
    for name in NAMES:
        r = engines[name]
        nodes = np.array([s['final_info_nodes'] for s in r['probes']])
        depths = np.array([s['depth'] for s in r['probes']])
        resource_summary[name] = dict(config=r['config'], oracle=r['oracle'],
            queries=len(r['probes'])+1+sum(s['searches'] for s in r['rollouts']),
            probe_final_info_nodes=dict(total=int(nodes.sum()), min=int(nodes.min()),
                                       median=float(np.median(nodes)), max=int(nodes.max())),
            depth=dict(min=int(depths.min()), median=float(np.median(depths)), max=int(depths.max())),
            controller_peak_rss_kib=r['controller_peak_rss_kib'], engine_peak_rss_kib=r['engine_peak_rss_kib'],
            seconds_since_start=r['seconds_since_start'])
    regression_result = {}
    for reference, target in ((NAMES[0], 'micro'), (NAMES[1], 'micro'), (NAMES[0], NAMES[1])):
        regression_result[f'{target}_vs_{reference}'] = regression(sample, ev, reference, target)
        regression_result[f'{target}_vs_{reference}_static'] = regression(sample, ev, reference, target, static=True)
    result = dict(task=157, collection=json.loads((a.out/'collect.json').read_text()),
                  outcomes=json.loads((a.out/'outcomes.json').read_text()),
                  validation=json.loads((a.out/'validation.json').read_text()),
                  engines=dict(sample_n=len(sample), population_representative=False,
                               probe_nodes=engines[NAMES[0]]['config']['limit']['nodes'], rollout_nodes=2000,
                               regressions=regression_result, policy=policy, rollout=rollout, resources=resource_summary),
                  method=dict(bootstrap_unit='game; paired opening seed for rollout',
                              score_view='black STM; white uses its negation, never a second search',
                              outcome_expected='empirical cell mean, re-estimated in every bootstrap',
                              outcome_controls=['ply','side','100cp bin','own rook','opponent rook','opponent king'],
                              overlap='n_central*n_wing/(n_central+n_wing); fixed full-data support >=20 each',
                              inference='observational association; no causal strength/Elo claim'),
                  verdict='unclear; small outcome calibration deficit for central kings; no evidence of increased habit after one wring',
                  suggestions=dict(
                      monitor=dict(action='keep current eval; repeat calibration and paired openings after next wring',
                                   openings=128, evals=2, nodes=2000, max_plies=60,
                                   max_queries=128*2*(60-6), max_nodes=128*2*(60-6)*2000,
                                   estimated_seconds=[120,300], measured=False),
                      data_pilot=dict(action='score/rook-matched wing rows weighted 2x, cap changed batch fraction at 10%; one future training candidate',
                                      rows=1_000_000, weight=2, fraction_cap=.1,
                                      cpu_extract_estimated_seconds=[60,180], measured=False,
                                      training_cost='one normal wring candidate; not run in this CPU-only task'),
                      intervention=dict(action='normal vs legal king-guided first 12 own turns; independent micro opponent; fresh opening pairs',
                                        openings=64, colors=2, arms=2, games=256, nodes=4000,
                                        max_plies=320, max_game_queries=256*320,
                                        max_guidance_queries=128*12*2,
                                        max_nodes=(256*320+128*12*2)*4000,
                                        estimated_seconds=[1200,3600], measured=False,
                                        guidance='compare unrestricted micro and king-only search; allow forced king move only within 50 micro cp; no legal king move => normal',
                                        note='screening only; adopt only after the normal independent strength protocol')),
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob('t157_*.py'))})
    save(a.repo/'reports/task157_king_center.json', result)
    print(json.dumps(dict(regressions=regression_result, policy=policy, rollout=rollout,
                          resources=resource_summary), ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DATA/'work/task157_king')
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    summarize(ap.parse_args())


if __name__ == '__main__':
    main()
