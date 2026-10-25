#!/usr/bin/env python3
"""Task 157: independent board/replay checks and a synthetic matching oracle."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import cshogi
import numpy as np
from t156_selfplay_audit import save
from t157_collect import DATA, placement
from t157_outcomes import prepare, aggregate, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DATA/'work/task157_king')
    a = ap.parse_args()
    sample = json.loads((a.out/'engine_sample.json').read_text())
    board_checks, engine_moves, rollout_moves = 0, 0, 0
    for r in sample:
        b = cshogi.Board(r['sfen'])
        cats, rooks, squares = placement(r['sfen'])
        for c in (0, 1):
            f, rank = divmod(b.king_square(c), 9)
            assert squares[c] == (f+1,rank+1)
        assert cats == r['king'] and rooks == r['rook']
        board_checks += 1
    for name in ('task142_finetune','task154_R_result30','micro'):
        report = json.loads((a.out/f'engine_{name}.json').read_text())
        for r, result in zip(sample, report['probes']):
            b = cshogi.Board(r['sfen'])
            legal = {cshogi.move_to_usi(m): m for m in b.legal_moves}
            assert result['move'] in legal
            engine_moves += 1
        for row in report['rollouts']:
            b = cshogi.Board()
            for mv in row['moves']:
                legal = {cshogi.move_to_usi(m): m for m in b.legal_moves}
                assert mv in legal
                b.push(legal[mv])
                rollout_moves += 1
            assert b.sfen() == row['sfen']
            assert placement(b.sfen())[0] == row['king']
    # A synthetic equal-score population: central black wins 40%, wing black
    # wins 60%, with a fixed central white king. Matched difference must be .20.
    original = np.load(a.out/'positions.npy')
    synthetic = np.zeros(200, dtype=original.dtype)
    synthetic['gid'] = np.arange(200)
    synthetic['ply'] = 60
    synthetic['king_b'][100:] = 2
    synthetic['winner'] = 1
    synthetic['winner'][:40] = 0
    synthetic['winner'][100:160] = 0
    d = prepare(synthetic, 100)
    counts, wins = aggregate(d)
    m, _ = metrics(d,counts,wins)
    assert abs(m['all/wing_minus_central_adjacent']-.2) < 1e-12
    assert abs((d['y'][:200]+d['y'][200:]-1)).max() == 0
    assert np.all(d['cp'][:200] == -d['cp'][200:])
    save(a.out/'validation.json', dict(task=157, sample_board_checks=board_checks,
         engine_bestmoves_legal=engine_moves, rollout_moves_legal=rollout_moves,
         synthetic_matched_difference=m['all/wing_minus_central_adjacent'],
         expected_synthetic_difference=.2, result='PASS'))
    print('Task 157 validation PASS')


if __name__ == '__main__':
    main()
