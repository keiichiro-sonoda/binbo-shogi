"""Task188a CPU fixtures; run with scripts/env.sh's VENV_PY, <=2 threads."""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'trainer'), str(REPO / 'experiments/alphazero')]
from trainer import train_nnue as trainer
from trainer import chunked_cache
from trainer.chunked_cache import chunk_targets
from trainer.multi_cache import FORMAT_MULTI, Manifest, file_stamp
from t172_test import cache, rows
from t172_build import part_record

torch.set_num_threads(2)


class ChunkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='t188a-')
        self.root = Path(self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.parts = [cache(self.root, 'A', rows()[:7], rows()[16:]),
                          cache(self.root, 'B', rows()[7:12], rows()[16:])]
        p = self.parts[0]
        self.val = part_record(p['cache'], p['compact'], Path(p['cache']) / 'val_evals.npy', 2, 'val')
        self.paths = []
        for i, p in enumerate(self.parts):
            mask = np.ones(7 if i == 0 else 5, bool)
            mask[[0, 2] if i == 0 else [4]] = False
            mask_path = self.root / f'{i}_mask.npy'; np.save(mask_path, mask)
            part = part_record(p['cache'], p['compact'], p['labels'], len(mask), 'train', mask_path)
            path = self.root / f'{i}.json'
            path.write_text(json.dumps(dict(format=FORMAT_MULTI, parts=[part], val=self.val, rows=len(mask))))
            self.paths.append(path)
        spec = json.loads(self.paths[0].read_text())
        spec['parts'][0].pop('mask'); spec['parts'][0].pop('mask_stamp')
        self.all_true = self.root / 'all.json'; self.all_true.write_text(json.dumps(spec))

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, out, extra=(), main=None, clock=True):
        argv = ['train_nnue', self.parts[0]['cache'], '--out', str(out), '--qat', '--qat-tie',
                '--factorize', '--feature-dim', '2', '--hidden-dim', '2', '--batch-size', '2',
                '--epochs', '2', '--seed', '188', '--device', 'cpu', '--lr', '6e-5',
                '--augment-mirror', '--mirror-cache', str(self.root / 'mirror.npy'), *map(str, extra)]
        with contextlib.ExitStack() as stack:
            log = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(log))
            stack.enter_context(patch.object(sys, 'argv', argv))
            # Wall time and output path are the only nondeterministic legacy
            # history fields. Freeze time and reuse out for the byte-level gate.
            if clock:
                stack.enter_context(patch.object(trainer.time, 'time', return_value=1000.0))
            try:
                result = (main or trainer.main)()
            except SystemExit as exc:
                result = exc.code
            return result, log.getvalue()

    def ok(self, *a, **kw):
        code, log = self.invoke(*a, **kw)
        self.assertEqual(code, 0, log)
        return log

    def artifacts(self, out):
        return {p: (out / p).read_bytes() for p in
                ('nnue_final.pt', 'nnue_best.pt', 'model_config.json', 'history.json')}

    def history(self, out):
        h = json.loads((out / 'history.json').read_text())
        h['run_identity'].pop('out')
        return h

    def test_one_chunk_and_existing_paths_byte_identical_to_head(self):
        old = self.root / 'old.py'
        old.write_text(subprocess.run(['git', 'show', 'HEAD:trainer/train_nnue.py'], cwd=REPO,
                                     capture_output=True, text=True, check=True).stdout)
        old_main = runpy.run_path(str(old))['main']
        out = self.root / 'same_out'
        # Includes plain/compact/manifest, masked and all-true shuffles, fused Adam,
        # async loss, mirror, dropout, scheduling and a global update cap.
        for fast in (False, True):
            common = ['--dropout', '.2', '--lr-schedule', 'cosine', '--max-steps', '5']
            if fast:
                common += ['--fast']
            variants = [(['--compact-cache', self.parts[0]['compact']] if fast else [])]
            variants += [['--manifest', p] for p in (self.all_true, self.paths[0])]
            for extra in variants:
                with self.subTest(fast=fast, source=extra):
                    self.ok(out, [*extra, *common], main=old_main)
                    before = self.artifacts(out)
                    self.ok(out, [*extra, *common])
                    self.assertEqual(before, self.artifacts(out))
                    if '--manifest' in extra:
                        self.ok(out, ['--chunk-manifests', extra[1], *common])
                        self.assertEqual(before, self.artifacts(out))

    def test_orders_targets_release_and_validation_once_per_epoch(self):
        seen, refs, calls, resident_bytes = [], [], [], []
        resident0, batches0 = Manifest.resident, trainer.run_batches
        expected_gen = torch.Generator().manual_seed(188)
        expected = []
        # Independent oracle for legacy masked randperm and mirror RNG draws.
        for _ in range(2):
            for path in self.paths:
                m = Manifest(path)
                torch.randperm(m.rows, generator=expected_gen)
                keep = torch.from_numpy(np.flatnonzero(m.vector('mask')))
                order = keep[torch.randperm(len(keep), generator=expected_gen)]
                expected.append(order.tolist())
                for n in range(0, len(order), 2):
                    torch.rand(min(2, len(order) - n), generator=expected_gen)

        def resident(m, *a, **kw):
            self.assertTrue(all(ref() is None for ref in refs), 'previous resident tensor is live')
            value = resident0(m, *a, **kw)
            resident_bytes.append(sum(t.numel() * t.element_size() for t in
                                      (value.codes, value.kings, value.half)))
            refs.extend([weakref.ref(value), weakref.ref(value.codes), weakref.ref(value.kings)])
            return value

        def batches(*a, **kw):
            calls.append(kw['train'])
            if kw['train']:
                seen.append(a[5].tolist())
                refs.append(weakref.ref(a[2]))  # target must also be released
            return batches0(*a, **kw)

        out = self.root / 'orders'
        with patch.object(Manifest, 'resident', resident), patch.object(trainer, 'run_batches', batches):
            self.ok(out, ['--chunk-manifests', ','.join(map(str, self.paths)), '--fast', '--resume'])
        self.assertEqual(seen, expected)
        self.assertEqual(resident_bytes, [1708, 1396, 1708, 1396])
        self.assertEqual(calls, [False, True, True, False, True, True, False])
        self.assertTrue(all(ref() is None for ref in refs))
        h = self.history(out)
        self.assertEqual(h['updates'], 10)
        self.assertEqual(h['n_train'], 12)
        self.assertEqual(h['chunks'], [file_stamp(p) for p in self.paths])
        for e in h['history']:
            self.assertEqual([c['rows'] for c in e['chunks']], [5, 4])
            self.assertFalse(e['partial_epoch'])
            self.assertEqual(e['train_loss'], sum(c['train_loss'] * c['rows'] for c in e['chunks']) / 9)
        state = torch.load(out / 'train_state.pt', weights_only=False)
        self.assertTrue(all(s['step'].item() == 10 for s in state['optimizer']['state'].values()))
        self.assertTrue(torch.equal(state['generator'], expected_gen.get_state()))

    def test_global_caps_mid_chunk_and_on_boundaries(self):
        for cap, expected_rows, partial in [(1, [[2]], True), (3, [[5]], True),
                                            (5, [[5, 4]], False), (6, [[5, 4], [2]], True)]:
            out = self.root / f'cap{cap}'
            self.ok(out, ['--chunk-manifests', ','.join(map(str, self.paths)), '--fast',
                          '--max-steps', cap, '--resume'])
            h = self.history(out)
            self.assertEqual(h['updates'], cap)
            self.assertEqual([[c['rows'] for c in e['chunks']] for e in h['history']], expected_rows)
            self.assertEqual(h['history'][-1]['partial_epoch'], partial)

    def test_cuda_allocator_lifecycle_without_gpu(self):
        # Exercise the CUDA branch with CPU-backed fake targets/passes. The
        # integration test above checks real resident tensors/targets die; this
        # test checks empty/reset/load/peak/release order without a CUDA driver.
        from types import SimpleNamespace
        events, refs = [], []
        class Target:
            def to(self, device):
                assert device == 'cuda'
                return self
        def resident(m, device):
            self.assertTrue(all(r() is None for r in refs))
            events.append('load')
            value = object.__new__(Target)
            refs.append(weakref.ref(value))
            return value
        def batches(*a, **kw):
            events.append('train')
            for start in range(0, len(a[5]), 2):
                kw['on_batch'](start + 2, 10.)
            return 2., torch.empty(0)
        args = SimpleNamespace(max_steps=None, checkpoint_minutes=0., resume=False, stop_file=None,
                               batch_size=2, objective='cp', scale_cp=600., canonical=False,
                               wdl_weight=0., async_loss=False)
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for name in ('empty_cache', 'reset_peak_memory_stats', 'synchronize',
                         'max_memory_allocated', 'max_memory_reserved'):
                def call(name=name):
                    events.append(name)
                    return 123
                stack.enter_context(patch.object(torch.cuda, name, call))
            stack.enter_context(patch.object(Manifest, 'resident', resident))
            stack.enter_context(patch.object(chunked_cache, 'chunk_targets', side_effect=lambda *a: (Target(), None)))
            result = chunked_cache.train_epoch(
                [Manifest(p) for p in self.paths], args=args, epoch=1, model=None, optimizer=None,
                generator=torch.Generator().manual_seed(188), device='cuda', criterion=None,
                mirror_perm=None, run_batches=batches, checkpoint=None, paused=RuntimeError, updates=0)
        self.assertEqual(events, ['empty_cache', 'reset_peak_memory_stats', 'load', 'train',
                                  'synchronize', 'max_memory_allocated', 'max_memory_reserved', 'empty_cache'] * 2)
        self.assertEqual(result[-1], 5)
        self.assertEqual([c['peak_allocated'] for c in result[2]], [123, 123])

    def test_single_chunk_resume_preserves_dropout_rng(self):
        full, resumed = self.root / 'full', self.root / 'resumed'
        opts = ['--chunk-manifests', self.paths[0], '--fast', '--dropout', '.2',
                '--resume', '--checkpoint-minutes', '1e-12']
        self.ok(full, opts, clock=False)
        step0, count = torch.optim.Adam.step, 0
        def step(opt, *a, **kw):
            nonlocal count
            count += 1
            if count == 2:
                raise SystemExit(76)
            return step0(opt, *a, **kw)
        with patch.object(torch.optim.Adam, 'step', step):
            code, log = self.invoke(resumed, opts, clock=False)
        self.assertEqual(code, 76, log)
        self.ok(resumed, opts, clock=False)
        self.assertEqual((full / 'nnue_final.pt').read_bytes(), (resumed / 'nnue_final.pt').read_bytes())

    def test_resume_boundaries_mid_chunk_cap_and_dropout(self):
        extra = ['--chunk-manifests', ','.join(map(str, self.paths)), '--fast', '--resume',
                 '--dropout', '.2', '--lr-schedule', 'cosine', '--max-steps', '9']
        # 3 updates in A, 2 in B. Boundary-only checkpoints also work with a
        # very long timer; tiny timer covers mid-chunk and non-divisible batches.
        for interval, crash in [('10000', 4), ('10000', 6), ('1e-12', 2), ('1e-12', 5), ('1e-12', 10)]:
            with self.subTest(interval=interval, crash=crash):
                opts = [*extra, '--checkpoint-minutes', interval]
                full, resumed = self.root / f'full{crash}', self.root / f'resume{crash}'
                self.ok(full, opts, clock=False)
                step0, count = torch.optim.Adam.step, 0

                def step(opt, *a, **kw):
                    nonlocal count
                    count += 1
                    if count == crash:
                        raise SystemExit(76)
                    return step0(opt, *a, **kw)

                # crash=10 interrupts validation after the capped 9th update.
                batches0 = trainer.run_batches
                def batches(*a, **kw):
                    if crash == 10 and not kw['train'] and count == 9:
                        raise SystemExit(76)
                    return batches0(*a, **kw)

                with patch.object(torch.optim.Adam, 'step', step), patch.object(trainer, 'run_batches', batches):
                    code, log = self.invoke(resumed, opts, clock=False)
                self.assertEqual(code, 76, log)
                self.ok(resumed, opts, clock=False)
                for name in ('nnue_final.pt', 'nnue_best.pt'):
                    self.assertEqual((full / name).read_bytes(), (resumed / name).read_bytes())
                h1, h2 = self.history(full), self.history(resumed)
                for h in (h1, h2):
                    for e in h['history']:
                        e.pop('train_secs')
                self.assertEqual(h1, h2)
                state1, state2 = [torch.load(p / 'train_state.pt', weights_only=False) for p in (full, resumed)]
                self.assertEqual(state1['updates'], 9)
                self.assertTrue(torch.equal(state1['rng']['torch'], state2['rng']['torch']))
                self.assertTrue(torch.equal(state1['generator'], state2['generator']))
                for key, value in state1['optimizer']['state'].items():
                    for name, tensor in value.items():
                        self.assertTrue(torch.equal(tensor, state2['optimizer']['state'][key][name]), name)
                self.ok(resumed, opts, clock=False)
                self.assertEqual((full / 'nnue_final.pt').read_bytes(), (resumed / 'nnue_final.pt').read_bytes())

    def test_stop_file_and_resume_identity(self):
        stop = self.root / 'stop'
        extra = ['--chunk-manifests', ','.join(map(str, self.paths)), '--resume', '--stop-file', stop]
        full, resumed = self.root / 'full', self.root / 'resumed'
        self.ok(full, extra)
        stop.touch()
        code, log = self.invoke(resumed, extra)
        self.assertEqual(code, 75, log)
        state = torch.load(resumed / 'train_state.pt', weights_only=False)
        self.assertEqual(state['chunk_state']['cursor'], 2)
        stop.unlink()
        self.ok(resumed, extra)
        self.assertEqual((full / 'nnue_final.pt').read_bytes(), (resumed / 'nnue_final.pt').read_bytes())
        for changed in ([*extra, '--lr', '.01'], ['--chunk-manifests', ','.join(map(str, reversed(self.paths))), '--resume']):
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.invoke(resumed, changed)
        # Same content rewritten at the same path changes its stamp and is rejected.
        self.paths[1].write_text(self.paths[1].read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.invoke(resumed, extra)

    def test_cli_validation(self):
        joined = ','.join(map(str, self.paths))
        for extra in (['--chunk-manifests', ''], ['--chunk-manifests', joined + ','],
                      ['--chunk-manifests', joined, '--manifest', self.paths[0]],
                      ['--chunk-manifests', joined, '--compact-cache', self.parts[0]['compact']],
                      ['--chunk-manifests', joined, '--row-mask', 'x'],
                      ['--chunk-manifests', joined, '--eval-npy', 'x'],
                      ['--chunk-manifests', joined, '--limit-train', '1']):
            code, log = self.invoke(self.root / 'bad', extra)
            self.assertEqual(code, 2, log)
        spec = json.loads(self.paths[1].read_text())
        spec['val'] = part_record(self.parts[1]['cache'], self.parts[1]['compact'],
                                  Path(self.parts[1]['cache']) / 'val_evals.npy', 2, 'val')
        self.paths[1].write_text(json.dumps(spec))
        code, log = self.invoke(self.root / 'bad', ['--chunk-manifests', joined])
        self.assertEqual(code, 2, log)
        self.assertIn('same val record', log)

    def test_target_transforms_match_legacy_manifest(self):
        # Compare every supported target recipe with targets observed in the
        # existing manifest loop, before optimizer updates or device transfers.
        from types import SimpleNamespace
        recipes = [[], ['--cp-result-lambda', '.7'], ['--wdl-weight', '.3'],
                   ['--objective', 'winrate'], ['--objective', 'wdl'],
                   ['--objective', 'wdl', '--wdl-lambda', '.7'],
                   ['--objective', 'wdl', '--wdl-lambda', '.7', '--wdl-known-only']]
        for recipe in recipes:
            captured = []
            batches0 = trainer.run_batches
            def batches(*a, **kw):
                if kw['train']:
                    captured.append((a[2].clone(), kw['wdl_target']))
                return batches0(*a, **kw)
            with patch.object(trainer, 'run_batches', batches):
                self.ok(self.root / 'targets', ['--manifest', self.paths[0], '--epochs', '1', *recipe])
            args = SimpleNamespace(objective='cp', wdl_weight=0., wdl_lambda=1., cp_result_lambda=1.,
                                   clamp_cp=2000., scale_cp=600., wdl_known_only=False)
            it = iter(recipe)
            for flag in it:
                key = flag[2:].replace('-', '_')
                setattr(args, key, True if key == 'wdl_known_only' else
                        str(next(it)) if key == 'objective' else float(next(it)))
            target, result = chunk_targets(Manifest(self.paths[0]), args)
            self.assertTrue(torch.equal(target, captured[0][0]), recipe)
            self.assertTrue((result is None and captured[0][1] is None) or
                            torch.equal(result, captured[0][1]), recipe)


if __name__ == '__main__':
    unittest.main()
