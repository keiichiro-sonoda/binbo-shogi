#!/usr/bin/env python3
"""Task 156: read-only, single-process self-play audit; source scripts/env.sh.

raw: all JSON games, with seeded whole-game sample for QS/replay/phase checks.
cache: full exact validation comparison, hash-sort duplicate candidates with
byte comparison, and sampled compact decoding. Outputs only to --out.
"""
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
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cshogi
import numpy as np
from trainer import kp, mirror


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def summary(values):
    a = np.asarray(values)
    return dict(n=len(a), mean=float(a.mean()),
                quantiles=dict(zip(('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'),
                                   map(float, np.percentile(a, [0, 10, 25, 50, 75, 90, 95, 99, 100])))))


def position(sfen):
    """Task 156: independent text parser; files/ranks are physical 1..9."""
    pieces = []
    for rank, row in enumerate(sfen.split()[0].split('/'), 1):
        file, promoted = 9, False
        for ch in row:
            if ch.isdigit():
                file -= int(ch)
            elif ch == '+':
                promoted = True
            else:
                pieces.append((ch, file, rank, promoted))
                file -= 1
                promoted = False
        assert file == 0
    hand, number = 0, ''
    for ch in sfen.split()[2]:
        if ch.isdigit():
            number += ch
        elif ch != '-':
            hand += int(number or 1)
            number = ''
    return pieces, hand


def bucket(value, edges):
    return str(next((hi for hi in edges if value < hi), 'inf'))


def raw(args):
    work = args.work
    meta = read(work / 'cache/meta.json')
    ids = [s['game_id'] for s in meta['sources']]
    sample = set(random.Random(args.seed).sample(ids, min(args.sample, len(ids))))
    np.save(args.out / 'sample_game_ids.npy', np.array(sorted(sample), np.int64))
    reasons, winners, checks, mate_distance, opening_lengths = Counter(), Counter(), Counter(), Counter(), Counter()
    lengths, mate_reads, qs_counts = [], [], Counter()
    openings = {p: Counter() for p in (16, 24, 32)}
    kings = {p: {str(c): Counter() for c in (0, 1)} for p in (30, 40, 60)}
    structures = defaultdict(Counter)
    bias = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # dropped, kept
    examples = []
    started = time.monotonic()
    for number, gid in enumerate(ids):
        path = work / 'games' / f'game_{gid:012d}.json'
        g = read(path)
        rows, end = g['rows'], len(g['moves'])
        reasons[g['reason']] += 1
        winners[str(g['winner'])] += 1
        lengths.append(end)
        opening_lengths[str(g['opening_plies'])] += 1
        checks['raw_rows'] += len(rows)
        checks['raw_mate_rows'] += sum(r['mate'] for r in rows)
        checks['node_ge_cap_rows'] += sum(r['nodes'] >= 10_000_000 for r in rows)
        checks['depth_below9_nonmate_rows'] += sum(r['depth'] < 9 and not r['mate'] for r in rows)
        checks['raw_score_sign_errors'] += sum(r['score'] != ((-1 if r['raw'].startswith('-') else 1) *
            (32000 if r['mate'] else (abs(int(r['raw'])) * 90 + 99) // 100)) for r in rows)
        if g['reason'] == 'eval_limit':
            r = rows[-1] if rows else None
            valid = (r is not None and r['ply'] == end and r['sfen'] == g['final_sfen'] and
                     abs(r['score']) == 32000 and g['winner'] == (r['turn'] if r['score'] > 0 else 1-r['turn']))
            checks['unverified_eval_terminal'] += not valid
            checks['nonmate_eval_terminal'] += valid and not r['mate']
            if r and r['mate']:
                mate_reads.append(abs(int(r['raw'])))
                mate_distance[str(r['raw'])] += 1
        if g['reason'] == 'max_ply_draw':
            checks['bad_maxply'] += end != 320 or g['winner'] != -1
        if g['reason'] == 'declaration_27_win':
            b = cshogi.Board(g['final_sfen'])
            checks['bad_declaration'] += not b.is_nyugyoku() or b.turn != g['winner']
        by_ply = {r['ply']: r['sfen'] for r in rows}
        by_ply[end] = g['final_sfen']
        for p, count in openings.items():
            if p in by_ply:
                count[' '.join(by_ply[p].split()[:3])] += 1
        for p, sides in kings.items():
            if p not in by_ply:
                continue
            pieces, _ = position(by_ply[p])
            for side, symbol, rook in ((0, 'K', 'R'), (1, 'k', 'r')):
                _, f, rank, _ = next(x for x in pieces if x[0] == symbol)
                sides[str(side)][f'{f},{rank}'] += 1
                home = rank >= 7 if side == 0 else rank <= 3
                left = f >= 7 if side == 0 else f <= 3
                right = f <= 3 if side == 0 else f >= 7
                rr = [x for x in pieces if x[0] == rook and not x[3] and (x[2] >= 6 if side == 0 else x[2] <= 4)]
                structure = ('ibisha' if (rr[0][1] <= 3 if side == 0 else rr[0][1] >= 7)
                             else 'furibisha') if len(rr) == 1 else 'other'
                st = structures[f'{p}:{side}:{structure}']
                st['n'] += 1
                st['left_wing'] += left
                st['right_wing'] += right
                st['either_wing_home3'] += home and (left or right)
                st['opposite_rook_wing_home3'] += home and (left if structure == 'ibisha' else right) if structure != 'other' else 0
        if gid in sample:
            payload = {k: v for k, v in g.items() if k != 'content_sha256'}
            canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            checks['game_content_hash_mismatch'] += hashlib.sha256(canonical.encode()).hexdigest() != g['content_sha256']
            with np.load(work / 'qs' / f'game_{gid:012d}.npz', allow_pickle=False) as z:
                selected = set(map(int, z['source_rows']))
                checks['sample_games'] += 1
                checks['sample_rows'] += len(rows)
                checks['sample_qs_rows'] += len(selected)
                checks['qs_source_hash_mismatch'] += str(z['source_sha256']) != hashlib.sha256(path.read_bytes()).hexdigest()
                for key in ('input_rows', 'changed_rows', 'flipped_rows', 'in_check'):
                    qs_counts[key] += int(z[key])
                for j, source in enumerate(z['source_rows']):
                    r = rows[int(source)]
                    leaf = kp.canonical_shared_row_to_sfen(z['features'][j])
                    turn = 0 if leaf.split()[1] == 'b' else 1
                    sign = -1 if turn != r['turn'] else 1
                    checks['kept_changed'] += int(z['changed'][j])
                    checks['kept_flip'] += int(z['flip'][j])
                    checks['leaf_differs'] += leaf.split()[:3] != r['sfen'].split()[:3]
                    checks['qs_eval_mismatch'] += int(z['scores'][j]) != sign * r['score']
                    checks['qs_result_mismatch'] += int(z['results'][j]) != (0 if g['winner'] == -1 else 1 if turn == g['winner'] else -1)
                for i, r in enumerate(rows):
                    kept = int(i in selected)
                    pieces, hand = position(r['sfen'])
                    nonking = len(pieces)-2
                    for key, val in [('ply_upper', bucket(r['ply'], [32, 64, 96, 128, 192])),
                                     ('abs_score_upper', bucket(abs(r['score']), [100, 300, 600, 1200, 2000, 4000, 32000])),
                                     ('board_nonking_upper', bucket(nonking, [24, 32])),
                                     ('hand_upper', bucket(hand, [4, 8, 12])),
                                     ('promotion_upper', bucket(sum(x[3] for x in pieces), [1, 3, 5])),
                                     ('result', str(r['result'])),
                                     ('distance_to_end_upper', bucket(end-r['ply'], [1, 8, 16, 32, 64]))]:
                        bias[key][val][kept] += 1
                    if len(examples) < 6 and r['ply'] == 60:
                        examples.append(dict(game_id=gid, winner=g['winner'], reason=g['reason'],
                                             ply=r['ply'], sfen=r['sfen'], score=r['score'], result=r['result'], kept=bool(kept)))
            b = cshogi.Board()
            roots = {r['ply']: r for r in rows}
            seen = Counter({' '.join(b.sfen().split()[:3]): 1})
            for p in range(end+1):
                legal = {cshogi.move_to_usi(m): m for m in b.legal_moves}
                if p in roots:
                    r = roots[p]
                    checks['replayed_roots'] += 1
                    checks['root_sfen_mismatch'] += r['sfen'] != b.sfen()
                    checks['root_turn_mismatch'] += r['turn'] != b.turn
                    checks['root_wdl_mismatch'] += r['result'] != (0 if g['winner'] == -1 else 1 if b.turn == g['winner'] else -1)
                    checks['illegal_selected_move'] += r['move'] not in legal
                if p < end:
                    move = g['moves'][p]
                    assert move in legal, (gid, p, move)
                    if p in roots:
                        checks['played_move_mismatch'] += move != roots[p]['move']
                    b.push(legal[move])
                    seen[' '.join(b.sfen().split()[:3])] += 1
            checks['final_sfen_mismatch'] += b.sfen() != g['final_sfen']
            if g['reason'] == 'repetition_draw':
                checks['replayed_repetition_games'] += 1
                checks['bad_repetition'] += seen[' '.join(b.sfen().split()[:3])] != 4
        if (number+1) % 10000 == 0:
            print(f'raw {number+1}/{len(ids)} {time.monotonic()-started:.1f}s', flush=True)
    return dict(mode='raw', seed=args.seed, sample_ids_file='sample_game_ids.npy', games=len(ids),
                termination=reasons, winners=winners, length=summary(lengths), length_histogram=Counter(map(str, lengths)),
                opening_lengths=opening_lengths, mate_read_length=summary(mate_reads), mate_read_signed_histogram=mate_distance,
                openings={str(p): dict(n=sum(c.values()), distinct=len(c), max_multiplicity=max(c.values()),
                            duplicate_rows=sum(c.values())-len(c)) for p, c in openings.items()},
                kings=kings, structures=structures, checks=checks, qs_sample=qs_counts, bias=bias, examples=examples,
                generation_summary=read(work/'last_gen.json'), qs_summary=read(work/'qs/last_qs.json'),
                cache_counts=meta['counts'])


def cache(args):
    work = args.work
    folder = work/'cache'
    x = np.load(folder/'train_indices.npy', mmap_mode='r')
    gids = np.load(folder/'train_game_ids.npy', mmap_mode='r')
    plies = np.load(folder/'train_plies.npy', mmap_mode='r')
    evals = np.load(folder/'train_evals.npy', mmap_mode='r')
    wdl = np.load(folder/'train_wdl.npy', mmap_mode='r')
    val = np.load(folder/'val_indices.npy', mmap_mode='r')
    exact = {row.tobytes() for row in val}
    reflected = set()
    for row in val:
        sfen = kp.canonical_shared_row_to_sfen(row)
        reflected.add(np.asarray(kp.canonical_shared_row(cshogi.Board(mirror.flip_sfen(sfen))), dtype='<i4').tobytes())
    # One 16-byte digest and an 8-byte index per row; compare full bytes on every collision candidate.
    hashes = np.empty(len(x), dtype='V16')
    checks, outcomes, sha = Counter(), Counter(), hashlib.sha256()
    for lo in range(0, len(x), 65536):
        block = x[lo:lo+65536]
        sha.update(block.tobytes())
        for j, row in enumerate(block):
            key = row.tobytes()
            hashes[lo+j] = hashlib.blake2b(key, digest_size=16).digest()
            checks['val_exact_overlap'] += key in exact
            checks['val_mirror_overlap'] += key in reflected
        outcomes.update(map(str, wdl[lo:lo+65536].tolist()))
        if lo % (65536*32) == 0:
            print(f'cache {lo}/{len(x)}', flush=True)
    order = np.argsort(hashes)
    equal = np.flatnonzero(hashes[order[1:]] == hashes[order[:-1]])
    for i in equal:
        checks['hash_equal_pairs'] += 1
        checks['train_duplicate_pairs'] += np.array_equal(x[order[i]], x[order[i+1]])
    del hashes, order
    # Check assembly alignment for whole sampled games. Dedup omissions must match earlier cache rows or val.
    selected_ids = np.load(args.out/'sample_game_ids.npy').tolist()
    removed = Counter()
    for gid in selected_ids:
        lo, hi = np.searchsorted(gids, [gid, gid+1])
        g = read(work/'games'/f'game_{gid:012d}.json')
        with np.load(work/'qs'/f'game_{gid:012d}.npz') as z:
            lookup = {int(p): int(lo+j) for j, p in enumerate(plies[lo:hi])}
            for j, source in enumerate(z['source_rows']):
                p = g['rows'][int(source)]['ply']
                if p not in lookup:
                    key = z['features'][j].tobytes()
                    removed['total'] += 1
                    removed['val_exact'] += key in exact
                    removed['val_mirror'] += key in reflected
                    continue
                k = lookup[p]
                checks['aligned_rows'] += 1
                checks['cache_feature_mismatch'] += not np.array_equal(x[k], z['features'][j])
                checks['cache_eval_mismatch'] += evals[k] != z['scores'][j]
                checks['cache_wdl_mismatch'] += wdl[k] != z['results'][j]
    # Verify all omitted sample rows by exact lookup in earlier cache, in bounded chunks.
    missing = {}
    for gid in selected_ids:
        lo, hi = np.searchsorted(gids, [gid, gid+1])
        g = read(work/'games'/f'game_{gid:012d}.json')
        present = set(plies[lo:hi].tolist())
        with np.load(work/'qs'/f'game_{gid:012d}.npz') as z:
            for j, src in enumerate(z['source_rows']):
                if g['rows'][int(src)]['ply'] not in present:
                    key = z['features'][j].tobytes()
                    if key not in exact and key not in reflected:
                        missing[key] = gid
    found = set()
    if missing:
        for lo in range(0, len(x), 65536):
            for j, row in enumerate(x[lo:lo+65536]):
                key = row.tobytes()
                if key in missing and gids[lo+j] <= missing[key]:
                    found.add(key)
    removed['nonval_unique_keys'] = len(missing)
    removed['found_in_earlier_train'] = len(found)
    removed['unexplained_unique_keys'] = len(missing.keys()-found)
    from trainer.compact_cache import decode_chunk
    for split in ('train', 'val'):
        comp = work/'cache_compact'
        orig = np.load(folder/f'{split}_indices.npy', mmap_mode='r')
        k = np.load(comp/f'{split}_kings.npy', mmap_mode='r')
        c = np.load(comp/f'{split}_codes.npy', mmap_mode='r')
        rows = np.array(random.Random(args.seed).sample(range(len(orig)), min(10000, len(orig))))
        checks[f'compact_{split}_sample_rows'] = len(rows)
        checks[f'compact_{split}_mismatch_values'] = int(np.count_nonzero(decode_chunk(k[rows], c[rows]) != orig[rows]))
    return dict(mode='cache', checks=checks, sampled_removals=removed, rows=len(x), result_counts=outcomes,
                source_sha256=sha.hexdigest(), compact_manifest=read(work/'cache_compact/compact.json'),
                mate_rows=int(np.count_nonzero(np.abs(evals.astype(np.int32))==32000)),
                clamped_rows=int(np.count_nonzero(np.abs(evals.astype(np.int32))>2000)),
                val_links={str(p.relative_to(folder)): str(p.resolve()) for p in folder.rglob('*.npy') if p.is_symlink()},
                val_shards=read(folder/'meta.json')['val_shards'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['raw', 'cache'])
    p.add_argument('--work', type=Path, default=Path(os.environ.get('DATA_ROOT', Path.home()/'binbo-shogi-data'))/'work/task154')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sample', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20271560)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = globals()[args.mode](args)
    result['seconds'] = time.monotonic()-started
    result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(args.out/f'{args.mode}.json', result)
    print(f'{args.mode} done: {result["seconds"]:.1f}s, peak RSS {result["peak_rss_kib"]} KiB', flush=True)


if __name__ == '__main__':
    main()
