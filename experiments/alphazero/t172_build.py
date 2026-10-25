#!/usr/bin/env python3
"""Task 172: immutable label overlays, exact feature audit, and multi-cache manifests.

Run with scripts/env.sh sourced. No source file is modified. Outputs are published
only after the full audit; interrupted staging directories are never accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from trainer import kp  # noqa: E402
from trainer.compact_cache import decode_chunk, source_stamp  # noqa: E402
from trainer.mirror import build_mirror_perm_shared  # noqa: E402
from trainer.multi_cache import FORMAT_MULTI, Manifest, checked_part, file_stamp  # noqa: E402

CHUNK = 65536


def env_paths():
    """Read canonical defaults from env.sh, including DATA_ROOT (not exported there)."""
    script = 'source scripts/env.sh\nexport DATA_ROOT\nenv -0'
    result = subprocess.run(['bash', '-c', script], cwd=REPO, check=True, capture_output=True)
    return dict(item.decode().split('=', 1) for item in result.stdout.split(b'\0') if b'=' in item)


def dump(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def sources(data):
    return [dict(name='A', expected_rows=63_950_000, cache=str(data / 'caches/shared_64m_qs129'),
                 compact=str(data / 'caches/compact_qs129'),
                 labels=str(data / 'work/task129/labels/train_dlshogi_evals.npy')),
            *[dict(name=n, expected_rows=count, cache=str(data / f'work/{task}/cache'),
                   compact=str(data / f'work/{task}/cache_compact'),
                   labels=str(data / f'work/{task}/cache/train_evals.npy'))
              for n, task, count in [('B1', 'task154', 10_282_217), ('B2', 'task159', 9_829_926)]]]


def row_hash(rows):
    # Candidate filter only. Every match is confirmed against all 77 int32 columns.
    mult = np.random.default_rng(20261720).integers(1, 2**63, 77, dtype=np.uint64) | np.uint64(1)
    return (rows.astype(np.uint64) * mult).sum(axis=1, dtype=np.uint64)


def mirror_rows(rows, perm):
    mirrored = perm[rows].astype(np.int32)
    mirrored[:, :39].sort(axis=1)
    mirrored[:, 39:].sort(axis=1)
    return mirrored


def premix(raw, wdl):
    """Mix the RAW, unclamped value; round to nearest integer, then int16.

    Draw/unknown rows retain the pure raw eval byte for byte. The trainer applies
    the usual +/-2000 clamp to those rows, and lambda=1 prevents double mixing.
    """
    out = np.asarray(raw, dtype=np.int16).copy()
    known = np.asarray(wdl) != 0
    x = np.asarray(raw)[known].astype(np.float64) / 600
    p = 1 / (1 + np.exp(-x))
    p = .7 * p + .3 * (np.asarray(wdl)[known] > 0)
    # At int16's endpoints sigmoid may round to 1. Clip before logit, then cp clamp.
    p = np.clip(p, 1e-15, 1 - 1e-15)
    out[known] = np.rint(np.clip(600 * (np.log(p) - np.log1p(-p)), -2000, 2000)).astype(np.int16)
    return out


def audit_parts(parts, val, work, chunk=CHUNK, hasher=row_hash):
    """Full scan; A internal duplicates are counted and retained, B first wins.

    Memory: hashes + stable argsort (~1.3GB at 84M), plus sort scratch and masks.
    Features stay mmap-backed; no 26GB materialized concatenation.
    """
    arrays = [np.load(Path(p['cache']) / 'train_indices.npy', mmap_mode='r') for p in parts]
    labels = [np.load(p['labels'], mmap_mode='r') for p in parts]
    counts = [len(a) for a in arrays]
    for a, lab in zip(arrays, labels):
        if a.dtype != np.int32 or a.shape[1:] != (77,) or lab.shape != (len(a),) or lab.dtype != np.int16:
            raise ValueError('invalid feature/label shape or dtype')
    offsets = np.cumsum([0, *counts], dtype=np.int64)
    hashes = np.lib.format.open_memmap(work / 'audit_hashes.npy', mode='w+', dtype=np.uint64,
                                       shape=(int(offsets[-1]),))
    masks = [np.ones(n, dtype=bool) for n in counts]
    v = np.load(Path(val['cache']) / 'val_indices.npy', mmap_mode='r')
    for p in parts:
        other_val = np.load(Path(p['cache']) / 'val_indices.npy', mmap_mode='r')
        if not np.array_equal(v, other_val):
            raise ValueError(f"{p['name']}: validation features differ from shared val")
    perm = build_mirror_perm_shared()
    valrows = np.concatenate([v, mirror_rows(v, perm)])
    vh = hasher(valrows)
    vo = np.argsort(vh, kind='stable')
    vh = vh[vo]
    overlap = []
    result = dict(parts={}, duplicates={}, hash_collision_groups=0,
                  a_internal_duplicates_retained=0, label_difference=dict(count=0, abs_sum=0, max_abs=0),
                  validation=dict(rows=len(v), mirror_rows=len(v), raw_overlaps={}, retained_overlaps={}))
    for i, (p, a) in enumerate(zip(parts, arrays)):
        hits = 0
        compact = Path(p['compact'])
        kings = np.load(compact / 'train_kings.npy', mmap_mode='r')
        codes = np.load(compact / 'train_codes.npy', mmap_mode='r')
        if kings.shape != (len(a), 2) or codes.shape != a.shape:
            raise ValueError('compact rows differ from raw')
        for start in range(0, len(a), chunk):
            rows = a[start:start + chunk]
            if np.any(rows < 0) or np.any(rows >= kp.CANONICAL_SHARED_TOTAL):
                raise ValueError('feature outside shared HalfKP band')
            if not np.array_equal(decode_chunk(kings[start:start + len(rows)], codes[start:start + len(rows)]), rows):
                raise ValueError(f"{p['name']}: compact round-trip differs from raw (77 columns)")
            h = hasher(rows)
            hashes[offsets[i] + start:offsets[i] + start + len(rows)] = h
            lo, hi = np.searchsorted(vh, h, 'left'), np.searchsorted(vh, h, 'right')
            for j in np.flatnonzero(lo < hi):
                # Hash collisions never remove unequal rows.
                if any(np.array_equal(rows[j], valrows[vo[k]]) for k in range(lo[j], hi[j])):
                    masks[i][start + j] = False
                    hits += 1
            if start // chunk % 32 == 0 or start + chunk >= len(a):
                print(f"T172_AUDIT_SCAN {p['name']} {min(start + chunk, len(a))}/{len(a)}", flush=True)
        overlap.append(hits)
        result['validation']['raw_overlaps'][p['name']] = hits
    hashes.flush()
    order = np.argsort(hashes, kind='stable')
    sorted_h = hashes[order]
    # Visit only repeated-hash runs. Full comparisons distinguish all collision classes.
    edges = np.flatnonzero(sorted_h[1:] == sorted_h[:-1])
    at = 0
    while at < len(edges):
        start, end = int(edges[at]), int(edges[at]) + 2
        at += 1
        while at < len(edges) and edges[at] == end - 1:
            end += 1
            at += 1
        seen = {}
        for gid in order[start:end]:
            i = int(np.searchsorted(offsets[1:], gid, side='right'))
            row = int(gid - offsets[i])
            key = arrays[i][row].tobytes()  # exact 77-column comparison after hash match
            if key not in seen:
                seen[key] = (i, row)
                continue
            prev_i, prev_row = seen[key]
            pair = f"{parts[prev_i]['name']}-{parts[i]['name']}"
            result['duplicates'][pair] = result['duplicates'].get(pair, 0) + 1
            if i == 0:
                result['a_internal_duplicates_retained'] += 1
            else:
                masks[i][row] = False
                diff = abs(int(labels[i][row]) - int(labels[prev_i][prev_row]))
                stats = result['label_difference']
                stats['count'] += 1
                stats['abs_sum'] += diff
                stats['max_abs'] = max(stats['max_abs'], diff)
        result['hash_collision_groups'] += int(len(seen) > 1)
    for i, p in enumerate(parts):
        result['parts'][p['name']] = dict(rows=counts[i], kept=int(masks[i].sum()),
                                        excluded=int((~masks[i]).sum()))
        # All raw val hits were removed by exact equality; dedup only removes more.
        result['validation']['retained_overlaps'][p['name']] = 0
    result['rows'] = sum(int(m.sum()) for m in masks)
    result['b_rows'] = sum(int(m.sum()) for m in masks[1:])
    result['b_expected_before_additional_audit'] = 20_112_062
    del hashes, sorted_h, order
    (work / 'audit_hashes.npy').unlink()
    return result, masks


def part_record(cache, compact, labels, rows, split, mask=None, offset=0):
    p = dict(cache=str(Path(cache).resolve()), compact=str(Path(compact).resolve()),
             labels=str(Path(labels).resolve()), rows=rows, offset=offset,
             source_stamp=source_stamp(cache, split), labels_stamp=file_stamp(labels),
             wdl_stamp=file_stamp(Path(cache) / f'{split}_wdl.npy'),
             compact_stamp=file_stamp(Path(compact) / 'compact.json'))
    if mask is not None:
        p.update(mask=str(Path(mask).resolve()), mask_stamp=file_stamp(mask))
    return p


def build(parts, val, output, chunk=CHUNK):
    output = Path(output).resolve()
    if output.exists():
        if (output / 'audit.json').exists():
            old_audit = json.loads((output / 'audit.json').read_text())
            if not old_audit.get('complete'):
                raise ValueError('incomplete audit')
            for item in old_audit['source_stamps']:
                for key in ('labels', 'wdl'):
                    if file_stamp(item[key]['path']) != item[key]:
                        raise ValueError('audited raw label/WDL changed')
            if file_stamp(old_audit['val_stamps']['labels']['path']) != old_audit['val_stamps']['labels']:
                raise ValueError('audited val labels changed')
            for arm in ('C', 'R', 'W'):
                Manifest(output / f'{arm}.json')
                if old_audit['parts_manifest'][arm] != file_stamp(output / f'{arm}.json'):
                    raise ValueError('audited manifest changed')
            print(f'T172_BUILD_REUSED {output}', flush=True)
            return json.loads((output / 'audit.json').read_text())
        # A pre-created task directory is normal. Never replace previous artifacts.
        if any(f.name.endswith(('_mask.npy', '_result30.npy')) or f.name in ('C.json', 'R.json', 'W.json')
               or f.name.startswith('.building-') for f in output.iterdir()):
            raise FileExistsError(f'{output}: incomplete build; use a new --work directory')
    else:
        output.mkdir(parents=True)
    stage = output / f'.building-{uuid.uuid4().hex}'
    stage.mkdir()
    for p in parts:
        n = len(np.load(p['labels'], mmap_mode='r'))
        checked_part(part_record(p['cache'], p['compact'], p['labels'], n, 'train'), 'train')
    vn = len(np.load(val['labels'], mmap_mode='r'))
    checked_part(part_record(val['cache'], val['compact'], val['labels'], vn, 'val'), 'val')
    if parts[0].get('expected_rows') is not None and vn != 50_000:
        raise ValueError('expected the shared 50k validation')
    before = [dict(indices=source_stamp(p['cache'], 'train'), labels=file_stamp(p['labels']),
                   wdl=file_stamp(Path(p['cache']) / 'train_wdl.npy')) for p in parts]
    val_before = dict(indices=source_stamp(val['cache'], 'val'), labels=file_stamp(val['labels']))
    audit, masks = audit_parts(parts, val, stage, chunk)
    records = []
    for p, mask in zip(parts, masks):
        cache = Path(p['cache'])
        raw = np.load(p['labels'], mmap_mode='r')
        wdl = np.load(cache / 'train_wdl.npy', mmap_mode='r')
        if wdl.dtype != np.int8 or wdl.shape != raw.shape or not np.isin(wdl, [-1, 0, 1]).all():
            raise ValueError('invalid WDL')
        mp = output / f"{p['name']}_mask.npy"
        np.save(stage / mp.name, mask)
        lp = output / f"{p['name']}_result30.npy"
        mixed = np.lib.format.open_memmap(stage / lp.name, mode='w+', dtype=np.int16, shape=raw.shape)
        for start in range(0, len(raw), chunk):
            mixed[start:start + chunk] = premix(raw[start:start + chunk], wdl[start:start + chunk])
        mixed.flush()
        del mixed
        if p.get('expected_rows', len(raw)) != len(raw):
            raise ValueError(f"{p['name']}: expected row count mismatch")
        audit['parts'][p['name']]['compact_mismatches'] = 0
        audit['parts'][p['name']]['wdl'] = {str(k): int((wdl == k).sum()) for k in (-1, 0, 1)}
        audit['parts'][p['name']]['kept_wdl'] = {str(k): int(((wdl == k) & mask).sum()) for k in (-1, 0, 1)}
        records.append((p, len(raw), mp, lp))
    after = [dict(indices=source_stamp(p['cache'], 'train'), labels=file_stamp(p['labels']),
                  wdl=file_stamp(Path(p['cache']) / 'train_wdl.npy')) for p in parts]
    val_after = dict(indices=source_stamp(val['cache'], 'val'), labels=file_stamp(val['labels']))
    if before != after or val_before != val_after:
        raise ValueError('source changed during audit/mixing')
    audit['source_stamps'] = before
    audit['val_stamps'] = val_before
    # Publish immutable arrays before manifests. An interruption leaves no audit success marker.
    for f in list(stage.iterdir()):
        os.link(f, output / f.name)  # exclusive publication; never replace an existing file
        f.unlink()
    va = np.load(Path(val['cache']) / 'val_indices.npy', mmap_mode='r')
    vr = part_record(val['cache'], val['compact'], val['labels'], len(va), 'val')
    for arm in ('C', 'R', 'W'):
        selected, offset = [], 0
        for i, (p, rows, mask, mixed) in enumerate(records):
            if arm != 'R' and i > 0:
                continue
            labels_path = mixed if (arm == 'W' or (arm == 'R' and i > 0)) else Path(p['labels'])
            rec = part_record(p['cache'], p['compact'], labels_path, rows, 'train', mask, offset)
            selected.append(rec)
            offset += rows
        path = output / f'{arm}.json'
        dump(path, dict(format=FORMAT_MULTI, rows=offset, parts=selected, val=vr,
                        arm=arm, provenance='A: Gikou positions + WS-T/QS labels; B1: task142 selfplay; B2: task154 selfplay (both include public light finish)',
                        result_lambda_cli=1, result_mix=.3 if arm != 'C' else 0))
        Manifest(path)  # enforce the exact loader contract before accepting this build
    audit.update(task=172, complete=True, parts_manifest={a: file_stamp(output / f'{a}.json') for a in ('C', 'R', 'W')},
                 mixing='raw/600 sigmoid -> 0.7 eval + 0.3 win/loss -> 600 logit -> clamp +/-2000 -> rint int16; draws unchanged',
                 a_changed=bool(audit['parts']['A']['excluded']))
    total = audit['rows']
    audit['natural_ratio'] = {p['name']: audit['parts'][p['name']]['kept'] / total for p in parts}
    audit['steps_per_epoch'] = {a: (sum(audit['parts'][p['name']]['kept'] for p in
                                      (parts if a == 'R' else parts[:1])) + 1023) // 1024 for a in ('C', 'R', 'W')}
    audit['base_update_cap'] = 749_424
    dump(output / 'audit.json', audit)
    stage.rmdir()
    print(f"T172_BUILD_DONE rows={audit['rows']} B={audit['b_rows']} audit={output / 'audit.json'}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--chunk-rows', type=int, default=CHUNK)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error('--chunk-rows must be positive')
    env = env_paths()
    data = Path(env['DATA_ROOT'])
    parts = sources(data)
    val = dict(cache=parts[0]['cache'], compact=parts[0]['compact'],
               labels=str(data / 'work/task129/labels/val_dlshogi_evals.npy'))
    build(parts, val, args.work or data / 'work/task172', args.chunk_rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
