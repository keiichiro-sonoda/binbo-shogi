#!/usr/bin/env python3
"""Task 156: <=240 real QS roots plus small shared/holdout assembly fixtures.

Run only after the two bulk audit processes finish. All writes stay in --out.
Source scripts/env.sh. No GPU, no game generation, no training.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import t153_selfplay_gen as t
import cshogi
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--work', type=Path, default=t.DATA/'work/task154')
    args = p.parse_args()
    out, work = args.out, args.work
    out.mkdir(parents=True, exist_ok=True)
    checks, examples = Counter(), []
    engine, budget = None, 240
    qm = t.read_json(work/'qs/manifest.json')
    # Independent rerun of full games preserves the same QS reset/ordering boundary.
    try:
        for gid in range(2):
            game = t.read_json(t.game_path(work, gid))
            rows = game['rows']
            assert len(rows) <= budget
            budget -= len(rows)
            inp = np.zeros(len(rows), dtype=t.qs.PSV)
            b = cshogi.Board()
            for i, row in enumerate(rows):
                b.set_sfen(row['sfen'])
                b.to_psfen(inp[i]['sfen'])
                inp[i]['gamePly'] = row['ply']+1
            inp['score'] = 1
            source, dest = out/f'qs_{gid}_input.psv', out/f'qs_{gid}_output.psv'
            inp.tofile(source)
            if engine is None:
                engine = t.old.Engine(dict(qm['config']['engine'], timeout=60))
            else:
                engine.reset()
            engine.send(f'qsearch_psv {source} {dest} 1')
            line = engine.until('info string qsearch_psv ')[-1]
            assert 'done:' in line, line
            leaf = np.fromfile(dest, dtype=t.qs.PSV)
            assert len(leaf) == len(rows)
            selected = []
            with np.load(work/'qs'/f'game_{gid:012d}.npz') as z:
                for i, entry in enumerate(leaf):
                    b.set_psfen(entry['sfen'])
                    root = rows[i]
                    changed = bool(np.any(inp[i]['sfen'] != entry['sfen']))
                    odd = int(entry['gamePly'])-int(inp[i]['gamePly']) & 1
                    checks['qs_roots'] += 1
                    checks['changed'] += changed
                    checks['odd_leaf'] += odd
                    checks['marker_parity_mismatch'] += int(entry['score']) != (-1 if odd else 1)
                    checks['leaf_turn_mismatch'] += b.turn != (root['turn'] ^ odd)
                    root_result = 0 if game['winner'] == -1 else 1 if root['turn'] == game['winner'] else -1
                    expected_result = 0 if game['winner'] == -1 else 1 if b.turn == game['winner'] else -1
                    checks['inherited_result_mismatch'] += root_result*(-1 if odd else 1) != expected_result
                    if odd and len(examples) < 4:
                        examples.append(dict(game_id=gid, source_row=i, winner=game['winner'],
                                             root_turn=root['turn'], leaf_turn=b.turn,
                                             root_score=root['score'], inherited_leaf_score=-root['score'],
                                             root_result=root_result, inherited_leaf_result=-root_result,
                                             root_sfen=root['sfen'], leaf_sfen=b.sfen(),
                                             leaf_plies=int(entry['gamePly'])-int(inp[i]['gamePly']), stored=False))
                    if not changed and not b.is_check():
                        selected.append(i)
                checks['selected_list_mismatch'] += selected != z['source_rows'].tolist()
    finally:
        if engine is not None:
            engine.close()
    # One game ended before write-minply. Its terminal search was never stored.
    meta = t.read_json(work/'cache/meta.json')
    early = []
    cfg = t.read_json(work/'manifest.json')['config']['engine']
    for item in meta['sources']:
        if item['kept']:
            continue
        game = t.read_json(t.game_path(work, item['game_id']))
        if game['rows'] or game['reason'] != 'eval_limit':
            continue
        e = t.GameEngine(dict(cfg, timeout=60))
        try:
            e.reset()
            move, score = e.search_game(game['moves'])
            b = cshogi.Board(game['final_sfen'])
            early.append(dict(game_id=item['game_id'], ply=len(game['moves']),
                              winner=game['winner'], move=move, search=score,
                              winner_agrees=game['winner'] == (b.turn if score['score'] > 0 else 1-b.turn)))
        finally:
            e.close()
    # Replay every repetition draw and distinguish perpetual check independently.
    draws = Counter()
    perpetual_ids = []
    for item in meta['sources']:
        if item['reason'] != 'repetition_draw':
            continue
        game = t.read_json(t.game_path(work, item['game_id']))
        b = cshogi.Board()
        final_key = ' '.join(game['final_sfen'].split()[:3])
        occurrences = [0] if ' '.join(b.sfen().split()[:3]) == final_key else []
        gave_check = []
        for ply, move in enumerate(game['moves']):
            legal = {cshogi.move_to_usi(m): m for m in b.legal_moves}
            assert move in legal
            mover = b.turn
            b.push(legal[move])
            gave_check.append((mover, b.is_check()))
            if ' '.join(b.sfen().split()[:3]) == final_key:
                occurrences.append(ply+1)
        assert len(occurrences) == 4 and b.sfen() == game['final_sfen']
        perpetual = [side for side in (0, 1) if all(check for mover, check in gave_check[occurrences[0]:] if mover == side)]
        draws['games'] += 1
        draws['perpetual_check_games'] += bool(perpetual)
        if perpetual:
            perpetual_ids.append(item['game_id'])
    # Exercise current val_shards branches using existing, immutable game/QS parts.
    fixture = out/'assembly_input'
    (fixture/'games').mkdir(parents=True)
    (fixture/'qs').mkdir()
    manifest = t.read_json(work/'manifest.json')
    manifest['ranges'] = [[0, 8]]
    t.json_write(fixture/'manifest.json', manifest)
    t.json_write(fixture/'qs/manifest.json', qm)
    for gid in range(8):
        t.game_path(fixture, gid).symlink_to(t.game_path(work, gid))
        (fixture/'qs'/f'game_{gid:012d}.npz').symlink_to(work/'qs'/f'game_{gid:012d}.npz')
    assemblies = {}
    for name, fraction in [('shared', 0), ('holdout', .5)]:
        a = argparse.Namespace(work=fixture, out=out/f'fixture_{name}', holdout_fraction=fraction,
                               split_seed=20271531, val_cache=t.DATA/'caches/shared_64m',
                               val_labels=t.DATA/'work/task142/labels_public')
        meta = t.assemble(a)
        tr = np.load(a.out/'train_indices.npy')
        va = np.load(a.out/'val_indices.npy')
        actual_val = {r.tobytes() for r in va}
        val_mirrors = {np.asarray(t.kp.canonical_shared_row(cshogi.Board(t.mirror.flip_sfen(
                       t.kp.canonical_shared_row_to_sfen(r)))), np.int32).tobytes() for r in va}
        train_keys = {r.tobytes() for r in tr}
        # This is the exact required trainer meta lookup. Do not allocate a model or train.
        trainer_val_shards = json.loads((a.out/'meta.json').read_text())['val_shards']
        assemblies[name] = dict(counts=meta['counts'], val_shards=trainer_val_shards,
                                train_val_overlap=len(train_keys & actual_val),
                                train_val_mirror_overlap=len(train_keys & val_mirrors),
                                train_duplicates=len(tr)-len(train_keys))
        if fraction:
            tg = set(np.load(a.out/'train_game_ids.npy').tolist())
            vg = set(np.load(a.out/'val_game_ids.npy').tolist())
            assemblies[name]['game_overlap'] = sorted(tg & vg)
            assemblies[name]['train_games'] = sorted(tg)
            assemblies[name]['val_games'] = sorted(vg)
            assert trainer_val_shards == [] and not tg & vg
        else:
            assert trainer_val_shards == t.read_json(a.val_cache/'meta.json')['val_shards']
        assert not train_keys & actual_val and not train_keys & val_mirrors
    result = dict(checks=checks, odd_leaf_examples=examples, assemblies=assemblies,
                  early_unrecorded_terminal=early,
                  repetition_replay=draws, perpetual_check_game_ids=perpetual_ids)
    t.json_write(out/'spot.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
