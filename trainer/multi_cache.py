"""Verified, zero-escape shared HalfKP parts with a single resident allocation.

Offsets address physical rows, including masked rows. Shuffle uses the combined
keep mask; labels, WDL and indices therefore always share the same row space.
Paths in a manifest are absolute, or relative to its parent directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch

try:
    from . import kp
    from .compact_cache import CompactCache, FORMAT, source_stamp
except ImportError:  # train_nnue.py invoked directly (trainer/ on sys.path)
    import kp
    from compact_cache import CompactCache, FORMAT, source_stamp

FORMAT_MULTI = 'shared-halfkp-parts-v1'


def file_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=digest.hexdigest())


def checked_part(part, split):
    cache, compact = Path(part['cache']), Path(part['compact'])
    meta = json.loads((cache / 'meta.json').read_text())
    if (meta['features_per_position'] != 77 or
            meta['total_features'] != kp.CANONICAL_SHARED_TOTAL or
            not meta.get('canonical_shared') or meta.get('canonical_shared_ka') or
            meta.get('canonical_shared_kav2') or meta.get('kpp_zone')):
        raise ValueError(f'{cache}: requires shared HalfKP (77 columns)')
    idx = np.load(cache / f'{split}_indices.npy', mmap_mode='r')
    rows = part['rows']
    if idx.shape != (rows, 77) or idx.dtype != np.int32 or meta[f'n_{split}'] != rows:
        raise ValueError(f'{cache}: row count / feature dtype mismatch')
    cm = json.loads((compact / 'compact.json').read_text())
    info = cm['splits'][split]
    if (cm['format'] != FORMAT or Path(cm['source']).resolve() != cache.resolve() or
            info['rows'] != rows or info['escapes'] != 0 or info['mismatches'] != 0 or
            info['source_sha256'] != info['decoded_sha256'] or
            info['source_stamp'] != source_stamp(cache, split)):
        raise ValueError(f'{compact}: unverified, stale, escaped or incompatible compact')
    for key, dtype, shape in [('kings', np.uint8, (rows, 2)), ('codes', np.uint16, (rows, 77))]:
        arr = np.load(compact / f'{split}_{key}.npy', mmap_mode='r')
        if arr.dtype != dtype or arr.shape != shape:
            raise ValueError(f'{compact}: bad {key} format')
    for key in ('labels', 'mask'):
        if key not in part:
            continue
        arr = np.load(part[key], mmap_mode='r')
        if arr.shape != (rows,) or arr.dtype != (np.int16 if key == 'labels' else np.bool_):
            raise ValueError(f'{cache}: bad {key} shape/dtype')
        if file_stamp(part[key]) != part[f'{key}_stamp']:
            raise ValueError(f'{cache}: changed {key}')
    if file_stamp(compact / 'compact.json') != part['compact_stamp']:
        raise ValueError(f'{cache}: changed compact manifest')
    if source_stamp(cache, split) != part['source_stamp']:
        raise ValueError(f'{cache}: changed source')
    wdl = np.load(cache / f'{split}_wdl.npy', mmap_mode='r')
    if wdl.shape != (rows,) or wdl.dtype != np.int8 or not np.isin(wdl, [-1, 0, 1]).all():
        raise ValueError(f'{cache}: bad WDL')
    if file_stamp(cache / f'{split}_wdl.npy') != part['wdl_stamp']:
        raise ValueError(f'{cache}: changed WDL')
    return meta


class Manifest:
    def __init__(self, path):
        path = Path(path).resolve()
        self.identity = hashlib.sha256(path.read_bytes()).hexdigest()
        self.spec = json.loads(path.read_text())
        if self.spec['format'] != FORMAT_MULTI:
            raise ValueError('unsupported multi-cache manifest')
        self.parts, self.val = self.spec['parts'], self.spec['val']
        for part in [*self.parts, self.val]:
            for key in ('cache', 'compact', 'labels', 'mask'):
                if key in part:
                    part[key] = str((path.parent / part[key]).resolve())
        offset = 0
        for part in self.parts:
            if part['offset'] != offset:
                raise ValueError('non-contiguous part offset')
            checked_part(part, 'train')
            offset += part['rows']
        self.rows = offset
        if not self.parts or offset != self.spec['rows']:
            raise ValueError('manifest row count mismatch')
        self.meta = checked_part(self.val, 'val')

    def vector(self, key):
        dtype = np.bool_ if key == 'mask' else np.float32
        out = np.empty(self.rows, dtype=dtype)
        for p in self.parts:
            path = p.get(key) if key != 'wdl' else str(Path(p['cache']) / 'train_wdl.npy')
            out[p['offset']:p['offset'] + p['rows']] = np.load(path, mmap_mode='r') if path else True
        return out

    def resident(self, device, chunk_rows=65536):
        # Do not construct per-part GPU caches and cat: that doubles peak VRAM.
        result = CompactCache.__new__(CompactCache)
        result.device = torch.device(device)
        result.shape = (self.rows, 77)
        result.kings = torch.empty((self.rows, 2), dtype=torch.uint8, device=device)
        result.codes = torch.empty((self.rows, 77), dtype=torch.int16, device=device)
        result.bases = result.escapes = None
        result.half = torch.tensor([0] * 39 + [1] * 38, dtype=torch.long, device=device)
        for p in self.parts:
            for key in ('kings', 'codes'):
                source = np.load(Path(p['compact']) / f'train_{key}.npy', mmap_mode='r')
                if key == 'codes':
                    source = source.view(np.int16)
                for start in range(0, p['rows'], chunk_rows):
                    end = min(start + chunk_rows, p['rows'])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        block = torch.from_numpy(source[start:end])
                    getattr(result, key)[p['offset'] + start:p['offset'] + end].copy_(block)
        return result
