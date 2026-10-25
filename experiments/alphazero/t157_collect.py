#!/usr/bin/env python3
"""Task 157: stream Task 154 games; reuse Task 156's SFEN parser. No engines."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from t156_selfplay_audit import position, save
import numpy as np

DATA = Path(os.environ.get('DATA_ROOT', Path.home() / 'binbo-shogi-data'))
PLIES = (40, 60, 80)
CATEGORIES = ('centre', 'adjacent', 'wing_home', 'other')
ROOKS = ('ibisha', 'furibisha', 'other')


def placement(sfen):
    pieces, _ = position(sfen)
    kings, rooks, squares = [], [], []
    for side, king, rook in ((0, 'K', 'R'), (1, 'k', 'r')):
        _, f, rank, _ = next(x for x in pieces if x[0] == king)
        home = rank >= 7 if side == 0 else rank <= 3
        cat = (0 if f == 5 else 1 if f in (4, 6) else 2) if home else 3
        rr = [x for x in pieces if x[0] == rook and not x[3]
              and (x[2] >= 6 if side == 0 else x[2] <= 4)]
        st = (0 if (rr[0][1] <= 3 if side == 0 else rr[0][1] >= 7) else 1) if len(rr) == 1 else 2
        kings.append(cat)
        rooks.append(st)
        squares.append((f, rank))
    return kings, rooks, squares


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, default=DATA / 'work/task154')
    ap.add_argument('--out', type=Path, default=DATA / 'work/task157_king')
    ap.add_argument('--seed', type=int, default=20271570)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    meta = json.loads((a.work / 'cache/meta.json').read_text())
    ids = [s['game_id'] for s in meta['sources']]
    dtype = [('gid', 'i4'), ('ply', 'i2'), ('cp', 'i4'), ('mate', '?'), ('winner', 'i1'),
             ('king_b', 'i1'), ('king_w', 'i1'), ('rook_b', 'i1'), ('rook_w', 'i1'),
             ('file_b', 'i1'), ('file_w', 'i1'), ('rank_b', 'i1'), ('rank_w', 'i1'),
             ('terminal', '?')]
    data = np.empty(len(ids) * len(PLIES), dtype=dtype)
    n = 0
    reservoirs, seen = defaultdict(list), Counter()
    rng = random.Random(a.seed)
    checks = Counter()
    # Stratified reservoirs: four central/wing combinations, three plies.
    for j, gid in enumerate(ids):
        g = json.loads((a.work / 'games' / f'game_{gid:012d}.json').read_text())
        for r in g['rows']:
            if r['ply'] not in PLIES:
                continue
            assert r['turn'] == 0 and r['sfen'].split()[1] == 'b'
            assert r['result'] == (0 if g['winner'] == -1 else 1 if g['winner'] == 0 else -1)
            cats, rooks, sq = placement(r['sfen'])
            cp = (32000 if int(r['raw']) > 0 else -32000) if r['mate'] else int(r['raw'])
            terminal = r['ply'] == len(g['moves'])
            data[n] = (gid, r['ply'], cp, r['mate'], g['winner'], *cats, *rooks,
                       sq[0][0], sq[1][0], sq[0][1], sq[1][1], terminal)
            n += 1
            checks['roots'] += 1
            checks['terminal_roots'] += terminal
            checks['mate_roots'] += r['mate']
            if not r['mate'] and abs(cp) <= 800 and all(c != 3 for c in cats):
                key = (r['ply'], int(cats[0] == 2), int(cats[1] == 2))
                seen[key] += 1
                value = dict(game_id=gid, ply=r['ply'], sfen=r['sfen'], cp=cp,
                             king=cats, rook=rooks, winner=g['winner'], moves=g['moves'][:r['ply']])
                reservoir = reservoirs[key]
                if len(reservoir) < 128:
                    reservoir.append(value)
                else:
                    k = rng.randrange(seen[key])
                    if k < len(reservoir):
                        reservoir[k] = value
        if (j + 1) % 25000 == 0:
            print(f'collect {j+1}/{len(ids)} {time.monotonic()-started:.1f}s', flush=True)
    data = data[:n]
    np.save(a.out / 'positions.npy', data)
    # Check legal-board semantics using cshogi, without changing the SFEN parser.
    import cshogi
    sample, used = [], set()
    for key in sorted(reservoirs):
        pool = reservoirs[key]
        rng.shuffle(pool)
        chosen = 0
        for r in pool:
            if r['game_id'] in used:
                continue
            b = cshogi.Board(r['sfen'])
            if b.is_check() or b.is_game_over():
                continue
            replay = cshogi.Board()
            for move in r['moves']:
                legal = {cshogi.move_to_usi(m): m for m in replay.legal_moves}
                assert move in legal
                replay.push(legal[move])
            assert replay.sfen() == r['sfen']
            used.add(r['game_id'])
            sample.append(r)
            chosen += 1
            if chosen == 32:
                break
        assert chosen == 32, (key, chosen)
    save(a.out / 'engine_sample.json', sample)
    report = dict(task=157, seed=a.seed, games=len(ids), plies=PLIES, categories=CATEGORIES,
                  rooks=ROOKS, checks=checks, engine_sample=len(sample),
                  sample_stratum_population={str(k): v for k, v in seen.items()},
                  source_manifest_sha256=hashlib.sha256((a.work/'manifest.json').read_bytes()).hexdigest(),
                  positions_sha256=hashlib.sha256((a.out/'positions.npy').read_bytes()).hexdigest(),
                  seconds=time.monotonic()-started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(a.out / 'collect.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
