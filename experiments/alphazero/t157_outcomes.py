#!/usr/bin/env python3
"""Task 157: score-matched calibration and game-cluster bootstrap, CPU only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
from t156_selfplay_audit import save
from t157_collect import DATA, CATEGORIES, PLIES, ROOKS


def ci(value, draws):
    return dict(estimate=float(value), ci95=list(map(float, np.percentile(draws, [2.5, 97.5]))))


def prepare(roots, width):
    # Two views of the same root. White's score is negated, not a second search.
    n = len(roots)
    own = np.r_[roots['king_b'], roots['king_w']]
    opp = np.r_[roots['king_w'], roots['king_b']]
    rook = np.r_[roots['rook_b'], roots['rook_w']]
    orook = np.r_[roots['rook_w'], roots['rook_b']]
    cp = np.r_[roots['cp'], -roots['cp']]
    side = np.repeat([0, 1], n)
    ply = np.tile(np.searchsorted(PLIES, roots['ply']), 2)
    yb = np.where(roots['winner'] == -1, .5, (roots['winner'] == 0).astype(float))
    y = np.r_[yb, 1-yb]
    gid = np.tile(roots['gid'], 2)
    nb = 4000 // width + 1
    sb = (cp + 2000) // width
    # Baseline controls ply, colour, signed score, BOTH rooks, opposite king.
    cell = (((((ply * 2 + side) * 3 + rook) * 3 + orook) * nb + sb) * 4 + opp)
    nc = 3 * 2 * 3 * 3 * nb * 4
    key = cell * 4 + own
    labels = np.indices((3, 2, 3, 3, nb, 4)).reshape(6, -1)
    return dict(own=own, opp=opp, rook=rook, orook=orook, cp=cp, side=side,
                ply=ply, y=y, gid=gid, cell=cell, key=key, nc=nc, labels=labels)


def aggregate(d, weights=None):
    if weights is None:
        weights = np.ones(len(d['y']))
    shape = (d['nc'], 4)
    count = np.bincount(d['key'], weights=weights, minlength=d['nc']*4).reshape(shape)
    wins = np.bincount(d['key'], weights=weights*d['y'], minlength=d['nc']*4).reshape(shape)
    return count, wins


def metrics(d, count, wins, support=None):
    total = count.sum(axis=1)
    expected = wins.sum(axis=1) / np.maximum(1, total)
    residual = wins - count * expected[:, None]
    result = {}
    masks = {'all': np.ones(d['nc'], bool)}
    masks.update({f'ply{p}': d['labels'][0] == j for j, p in enumerate(PLIES)})
    masks.update({name: d['labels'][2] == j for j, name in enumerate(ROOKS[:2])})
    masks.update({name: d['labels'][1] == j for j, name in enumerate(('black', 'white'))})
    central_n = count[:, :2].sum(axis=1)
    central_w = wins[:, :2].sum(axis=1)
    wing_n, wing_w = count[:, 2], wins[:, 2]
    if support is None:
        support = (central_n >= 20) & (wing_n >= 20)
    # Overlap weights give comparable score/ply/rook/opponent distributions.
    overlap = central_n * wing_n / np.maximum(1, central_n + wing_n)
    diff = wing_w / np.maximum(1, wing_n) - central_w / np.maximum(1, central_n)
    for name, mask in masks.items():
        for j, cat in enumerate(CATEGORIES):
            num = count[mask, j].sum()
            result[f'{name}/residual/{cat}'] = residual[mask, j].sum() / max(1, num)
        selected = mask & support
        w = overlap[selected]
        result[f'{name}/wing_minus_central_adjacent'] = np.sum(w * diff[selected]) / max(1, w.sum())
    return result, support


def loss_rate(d, weights=None, support=None):
    # At ply 60: conditional central prevalence among losses versus wins.
    keep = (d['ply'] == 1) & (d['y'] != .5)
    c = d['cell'][keep]
    lost = (d['y'][keep] == 0).astype(int)
    central = (d['own'][keep] < 2).astype(float)
    w = np.ones(len(c)) if weights is None else weights[keep]
    key = c*2 + lost
    n = np.bincount(key, weights=w, minlength=d['nc']*2).reshape(-1, 2)
    z = np.bincount(key, weights=w*central, minlength=d['nc']*2).reshape(-1, 2)
    if support is None:
        support = (n[:, 0] >= 20) & (n[:, 1] >= 20)
    rate = z / np.maximum(1, n)
    overlap = n[:, 0]*n[:, 1] / np.maximum(1, n.sum(axis=1))
    weight = overlap * support
    value = np.sum(weight * (rate[:, 1]-rate[:, 0])) / max(1, weight.sum())
    raw = z.sum(axis=0) / n.sum(axis=0)
    return value, support, dict(winners_central=float(raw[0]), losers_central=float(raw[1]),
                                observations=n.sum(axis=0).astype(int).tolist(),
                                support_cells=int(support.sum()))


def run(roots, width, reps, seed, max_gid):
    d = prepare(roots, width)
    count, wins = aggregate(d)
    point, support = metrics(d, count, wins)
    loss, loss_support, loss_raw = loss_rate(d)
    draws = {k: [] for k in point}
    loss_draws = []
    rng = np.random.default_rng(seed)
    for i in range(reps):
        # Ordinary resampling of whole games, including games missing a ply.
        game_weights = np.bincount(rng.integers(0, max_gid+1, max_gid+1), minlength=max_gid+1)
        w = game_weights[d['gid']]
        cb, wb = aggregate(d, w)
        b, _ = metrics(d, cb, wb, support)
        for k, v in b.items():
            draws[k].append(v)
        loss_draws.append(loss_rate(d, w, loss_support)[0])
        if (i+1) % 200 == 0:
            print(f'outcomes width={width} bootstrap={i+1}/{reps}', flush=True)
    table = {}
    for name, mask in [('all', np.ones(d['nc'], bool))] + [
            (f'ply{p}', d['labels'][0] == j) for j, p in enumerate(PLIES)]:
        table[name] = {}
        for j, cat in enumerate(CATEGORIES):
            n = count[mask, j].sum()
            table[name][cat] = dict(n=int(n), result=float(wins[mask, j].sum()/max(1, n)),
                                   **ci(point[f'{name}/residual/{cat}'], draws[f'{name}/residual/{cat}']))
    coverage = {}
    for j, p in enumerate(PLIES):
        mask = support & (d['labels'][0] == j)
        coverage[str(p)] = dict(cells=int(mask.sum()), central_adjacent=int(count[mask, :2].sum()),
                                wing=int(count[mask, 2].sum()))
    return dict(width_cp=width, bootstrap=reps, table=table, coverage=coverage,
                metrics={k: ci(v, draws[k]) for k, v in point.items()},
                loss_central_prevalence=dict(**ci(loss, loss_draws), **loss_raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DATA/'work/task157_king')
    ap.add_argument('--bootstrap', type=int, default=1000)
    a = ap.parse_args()
    start = time.monotonic()
    data = np.load(a.out/'positions.npy')
    max_gid = int(data['gid'].max())
    good = ~data['mate'] & ~data['terminal'] & (abs(data['cp']) <= 2000)
    roots = data[good]
    distribution = {}
    for p in PLIES:
        sub = data[data['ply'] == p]
        distribution[str(p)] = dict(roots=len(sub), **{
            color: {cat: int(np.sum(sub['king_'+color] == j)) for j, cat in enumerate(CATEGORIES)}
            for color in ('b', 'w')})
    result = dict(task=157, units='USI displayed cp; probabilities in 0..1',
                  exclusions=dict(all_roots=len(data), primary_roots=len(roots),
                                  mate=int(data['mate'].sum()), terminal=int(data['terminal'].sum()),
                                  abs_cp_gt2000=int((abs(data['cp']) > 2000).sum())),
                  distribution=distribution,
                  primary=run(roots, 100, a.bootstrap, 20271571, max_gid),
                  sensitivity50=run(roots, 50, 400, 20271572, max_gid),
                  sensitivity200=run(roots, 200, 400, 20271573, max_gid))
    deep = roots.copy()
    for color in ('b', 'w'):
        shallow_wing = (deep['king_'+color] == 2) & np.isin(deep['file_'+color], [3, 7])
        deep['king_'+color][shallow_wing] = 3
    result['deep_wing'] = run(deep, 100, 400, 20271575, max_gid)
    result['decisive_only'] = run(roots[roots['winner'] != -1], 100, 400, 20271578, max_gid)
    # Compact, descriptive score-band table. Main inference uses narrower bins.
    d = prepare(roots, 100)
    broad = np.digitize(d['cp'], [-800, -400, -100, 100, 400, 800])
    score_table = []
    for ply_index, ply in enumerate(PLIES):
        for rook_index, rook in enumerate(ROOKS):
            for band in range(7):
                for cat_index, cat in enumerate(CATEGORIES):
                    keep = (d['ply'] == ply_index) & (d['rook'] == rook_index) & (broad == band) & (d['own'] == cat_index)
                    if np.any(keep):
                        score_table.append(dict(ply=ply, rook=rook, score_band=band, king=cat,
                                                n=int(keep.sum()), result=float(d['y'][keep].mean()),
                                                mean_cp=float(d['cp'][keep].mean())))
    result['descriptive_score_bands'] = dict(edges_cp=[-800,-400,-100,100,400,800], rows=score_table)
    result.update(seconds=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(a.out/'outcomes.json', result)
    print(f'outcomes done {result["seconds"]:.1f}s RSS={result["peak_rss_kib"]} KiB', flush=True)


if __name__ == '__main__':
    main()
