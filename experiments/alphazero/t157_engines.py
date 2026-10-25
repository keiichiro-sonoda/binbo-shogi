#!/usr/bin/env python3
"""Task 157: serial single-thread USI probes and paired short opening rollouts.

One Python controller and at most one engine. No GPU. Output stays in --out.
Static `e` and engine options are checked against the installed upstream source.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
import cshogi
import numpy as np
from t146_self_relabel import Engine, OPTIONS, digest
from t156_selfplay_audit import save
from t157_collect import DATA, placement


def config(name, nodes):
    micro = name == 'micro'
    binary = DATA / ('engines/yaneuraou980/source/YaneuraOu-SFNN_ka2_2048_8_64_progress8'
                     if micro else 'engines/yaneuraou/source/YaneuraOu-by-gcc')
    evaluation = next((DATA/'public/tsuyouraou11').glob('TuyouraOu11micro-*/')) if micro else DATA/'eval_dirs'/name
    return dict(engine=str(binary), options=dict(OPTIONS, EvalDir=str(evaluation),
                FV_SCALE=24 if micro else 16, USI_Hash=16), limit=dict(nodes=nodes), timeout=60,
                engine_sha256=digest(binary), eval_sha256=digest(evaluation/'nn.bin'))


def move_kind(board, move):
    if move in ('win', 'resign'):
        return dict(king=False, outward=False, inward=False, king_usi=None)
    legal = {cshogi.move_to_usi(m): m for m in board.legal_moves}
    assert move in legal, (board.sfen(), move)
    kings = [x for x in placement(board.sfen())[2]]
    f, rank = kings[board.turn]
    origin = f'{f}{chr(96+rank)}'
    king = move[:2] == origin
    delta = abs(int(move[2])-5)-abs(f-5) if king else 0
    return dict(king=king, outward=king and delta > 0, inward=king and delta < 0,
                king_usi=move if king else None)


def probe(engine, sample):
    results = []
    for i, row in enumerate(sample):
        engine.reset()  # independent TT/history per root
        engine.send('position startpos moves ' + ' '.join(row['moves']))
        engine.send('e')
        line = engine.until('eval = ')[-1]
        static = int(line.split('=')[1])
        # search() would replace full history by SFEN. Keep the game history.
        engine.send('go nodes ' + str(engine.cfg['limit']['nodes']))
        deadline, score = time.monotonic()+60, None
        final_nodes = 0
        from t146_self_relabel import SCORE_RE
        while True:
            line = engine.readline(deadline)
            tokens = line.split()
            if 'nodes' in tokens and line.startswith('info ') and 'string' not in tokens:
                final_nodes = int(tokens[tokens.index('nodes')+1])
            if line.startswith('bestmove '):
                move = tokens[1]
                break
            match = SCORE_RE.search(line)
            if match and not {'upperbound', 'lowerbound'}.intersection(tokens):
                kind, value = match.groups()
                score = dict(kind=kind, cp=int(value),
                             depth=int(tokens[tokens.index('depth')+1]) if 'depth' in tokens else 0,
                             exact_nodes=int(tokens[tokens.index('nodes')+1]) if 'nodes' in tokens else 0)
        assert score is not None
        board = cshogi.Board(row['sfen'])
        results.append(dict(game_id=row['game_id'], ply=row['ply'], static_value=static,
                            move=move, **score, final_info_nodes=final_nodes, **move_kind(board, move)))
        if (i+1) % 64 == 0:
            print(f'probe {i+1}/{len(sample)}', flush=True)
    return results


def rollouts(engine, prefixes):
    result = []
    for j, prefix in enumerate(prefixes):
        engine.reset()
        board, moves = cshogi.Board(), list(prefix['moves'])
        for mv in moves:
            legal = {cshogi.move_to_usi(m): m for m in board.legal_moves}
            assert mv in legal
            board.push(legal[mv])
        king_moves, searches = [0, 0], 0
        reason = 'ply60'
        seen = Counter()
        while len(moves) < 60:
            if board.is_game_over() or board.is_nyugyoku():
                reason = 'terminal'
                break
            key = ' '.join(board.sfen().split()[:3])
            seen[key] += 1
            if seen[key] == 4:
                reason = 'repetition'
                break
            engine.send('position startpos moves ' + ' '.join(moves))
            engine.send('go nodes 2000')
            move = engine.until('bestmove ')[-1].split()[1]
            searches += 1
            if move in ('win', 'resign'):
                reason = move
                break
            kind = move_kind(board, move)
            king_moves[board.turn] += kind['king']
            legal = {cshogi.move_to_usi(m): m for m in board.legal_moves}
            board.push(legal[move])
            moves.append(move)
        cats, rook, squares = placement(board.sfen())
        result.append(dict(game_id=prefix['game_id'], opening=prefix['moves'], moves=moves,
                           sfen=board.sfen(), king=cats, rook=rook, squares=squares,
                           king_moves=king_moves, searches=searches, reason=reason))
        print(f'rollout {j+1}/{len(prefixes)} ply={len(moves)}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DATA/'work/task157_king')
    ap.add_argument('--nodes', type=int, default=4000)
    ap.add_argument('--rollout-games', type=int, default=24)
    ap.add_argument('--only', choices=['task142_finetune', 'task154_R_result30', 'micro'])
    a = ap.parse_args()
    start = time.monotonic()
    sample = json.loads((a.out/'engine_sample.json').read_text())
    # These are actual uniform-legal random prefixes, before any engine move.
    rng = random.Random(20271574)
    gids = rng.sample(range(150000), a.rollout_games)
    prefixes = []
    for gid in gids:
        g = json.loads((DATA/'work/task154/games'/f'game_{gid:012d}.json').read_text())
        prefixes.append(dict(game_id=gid, moves=g['moves'][:g['opening_plies']]))
    save(a.out/'rollout_prefixes.json', prefixes)
    for name in ([a.only] if a.only else ['task142_finetune', 'task154_R_result30', 'micro']):
        path = a.out/f'engine_{name}.json'
        cfg = config(name, a.nodes)
        if path.exists():
            cached = json.loads(path.read_text())
            assert cached['config'] == cfg, 'cached engine configuration differs; choose a fresh --out'
            assert [(r['game_id'],r['ply']) for r in cached['probes']] == [(r['game_id'],r['ply']) for r in sample]
            if name != 'micro':
                assert [r['opening'] for r in cached['rollouts']] == [r['moves'] for r in prefixes]
            print(f'exists, reuse {path}', flush=True)
            continue
        engine = Engine(cfg)
        try:
            # Protocol/score sign oracle: winning one-move mate for side to move.
            oracle = '4k4/9/4P4/9/9/9/9/9/4K4 b G 1'
            b = cshogi.Board(oracle)
            assert not b.is_check()
            assert b.mate_move_in_1ply()
            sign = engine.search(oracle)
            assert sign[3] and sign[0] > 0, sign
            moves = move_kind(b, sign[-1])
            probes = probe(engine, sample)
            rolls = rollouts(engine, prefixes) if name != 'micro' else []
            report = dict(task=157, name=name, config=cfg, oracle=dict(sfen=oracle, result=sign, **moves),
                            probes=probes, rollouts=rolls,
                            seconds_since_start=time.monotonic()-start,
                            controller_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        finally:
            engine.close()
        report['engine_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        save(path, report)
        print(f'{name} done {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
