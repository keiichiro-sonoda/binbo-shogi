#!/usr/bin/env python3
"""CPU-only Task172 regression/oracle tests, <=4 threads, tiny artificial caches."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
from trainer import kp
from trainer.compact_cache import convert
from trainer.multi_cache import Manifest, file_stamp
from t172_build import audit_parts, build, mirror_rows, premix
from trainer.mirror import build_mirror_perm_shared, flip_sfen
import t172_run

torch.set_num_threads(2)


def rows(n=18):
    a = np.empty((n, 77), dtype=np.int32)
    for i in range(n):
        a[i, :38] = (i + 2) * kp.FEATURES_PER_KING + np.arange(38)
        a[i, 38] = kp.VIEW_DIM + i % 2
        a[i, 39:] = (i + 20) * kp.FEATURES_PER_KING + np.arange(38, 76)
    return a


def cache(root, name, a, val):
    c, comp = root / name, root / (name + '_compact')
    c.mkdir()
    meta = dict(n_train=len(a), n_val=len(val), features_per_position=77,
                total_features=kp.CANONICAL_SHARED_TOTAL, canonical_shared=True, val_shards=[99])
    (c / 'meta.json').write_text(json.dumps(meta))
    for split, idx in [('train', a), ('val', val)]:
        np.save(c / f'{split}_indices.npy', idx)
        np.save(c / f'{split}_evals.npy', (np.arange(len(idx)) * 401 - 2200).astype(np.int16))
        np.save(c / f'{split}_wdl.npy', np.resize(np.array([-1, 0, 1], dtype=np.int8), len(idx)))
    convert(c, comp, chunk_rows=2)
    return dict(name=name, cache=str(c), compact=str(comp), labels=str(c / 'train_evals.npy'))


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='t172-test-')
        self.root = Path(self.tmp.name)
        a = rows()
        self.parts = [cache(self.root, 'A', a[:6], a[16:]),
                      cache(self.root, 'B1', a[6:11], a[16:]),
                      cache(self.root, 'B2', a[11:16], a[16:])]
        self.val = dict(cache=self.parts[0]['cache'], compact=self.parts[0]['compact'],
                        labels=str(Path(self.parts[0]['cache']) / 'val_evals.npy'))
        self.work = self.root / 'work'

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_boundaries_all_columns_labels_wdl_masks_and_checks(self):
        build(self.parts, self.val, self.work, chunk=2)
        m = Manifest(self.work / 'R.json')
        resident = m.resident('cpu', chunk_rows=2)
        expected = np.concatenate([np.load(Path(p['cache']) / 'train_indices.npy') for p in self.parts])
        order = torch.tensor([10, 5, 6, 0, 15, 11, 5, 9, 1])
        np.testing.assert_array_equal(resident[order].numpy(), expected[order.numpy()])
        labels = np.concatenate([np.load(p['labels']) if i == 0 else premix(np.load(p['labels']),
                      np.load(Path(p['cache']) / 'train_wdl.npy')) for i, p in enumerate(self.parts)])
        np.testing.assert_array_equal(m.vector('labels'), labels)
        np.testing.assert_array_equal(m.vector('wdl'), np.concatenate([
            np.load(Path(p['cache']) / 'train_wdl.npy') for p in self.parts]))
        self.assertTrue(m.vector('mask').all())
        masked_spec = json.loads((self.work / 'R.json').read_text())
        expected_mask = np.ones(16, bool); expected_mask[[0, 5, 6, 10, 11, 15]] = False
        offset = 0
        for part in masked_spec['parts']:
            path = self.work / f"mask_{offset}.npy"
            np.save(path, expected_mask[offset:offset + part['rows']])
            part.update(mask=str(path), mask_stamp=file_stamp(path))
            offset += part['rows']
        masked = self.work / 'masked.json'; masked.write_text(json.dumps(masked_spec))
        mm = Manifest(masked)
        np.testing.assert_array_equal(mm.vector('mask'), expected_mask)
        trained = np.flatnonzero(mm.vector('mask'))
        np.testing.assert_array_equal(mm.resident('cpu')[torch.from_numpy(trained)].numpy(), expected[trained])
        # Reuse is verified; modified label and offset cannot silently pass.
        build(self.parts, self.val, self.work, chunk=2)
        spec = json.loads((self.work / 'R.json').read_text())
        spec['parts'][1]['offset'] += 1
        bad = self.work / 'bad.json'; bad.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'offset'):
            Manifest(bad)
        for key in ('rows', 'source_stamp', 'compact_stamp'):
            spec = json.loads((self.work / 'R.json').read_text())
            if key == 'rows':
                spec['parts'][0]['rows'] += 1
            elif key == 'source_stamp':
                spec['parts'][0][key]['mtime_ns'] += 1
            else:
                spec['parts'][0][key]['sha256'] = 'bad'
            bad = self.work / f'bad_{key}.json'; bad.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                Manifest(bad)
        path = Path(m.parts[1]['labels']); label = np.load(path); label[0] += 1; np.save(path, label)
        with self.assertRaisesRegex(ValueError, 'changed labels'):
            Manifest(self.work / 'R.json')

    def test_audit_collision_oracle_duplicates_and_mirror(self):
        # Force ALL hashes to collide, including unequal rows. Independent bytes oracle.
        perm = build_mirror_perm_shared()
        a = rows()
        data = [np.stack([a[0], a[0], a[16], mirror_rows(a[17:18], perm)[0], a[1]]),
                np.stack([a[0], a[2], a[2], a[16], a[3]]), np.stack([a[2], a[4], a[5], a[5], a[17]])]
        for p, arr in zip(self.parts, data):
            np.save(Path(p['cache']) / 'train_indices.npy', arr)
            np.save(p['labels'], np.arange(len(arr), dtype=np.int16))
            import shutil
            shutil.rmtree(p['compact'])
            convert(p['cache'], p['compact'], chunk_rows=2)
        self.work.mkdir()
        audit, masks = audit_parts(self.parts, self.val, self.work, 2,
                                  hasher=lambda x: np.zeros(len(x), dtype=np.uint64))
        np.testing.assert_array_equal(masks[0], [1, 1, 0, 0, 1])
        np.testing.assert_array_equal(masks[1], [0, 1, 0, 0, 1])
        np.testing.assert_array_equal(masks[2], [0, 1, 1, 0, 0])
        self.assertEqual(audit['a_internal_duplicates_retained'], 1)
        self.assertGreater(audit['hash_collision_groups'], 0)
        self.assertEqual(audit['validation']['raw_overlaps'], dict(A=2, B1=1, B2=1))

    def test_mirror_board_oracle(self):
        import cshogi
        board = cshogi.Board()
        rng = np.random.default_rng(172)
        perm = build_mirror_perm_shared()
        for _ in range(20):
            original = np.array([kp.canonical_shared_row(board)], dtype=np.int32)
            flipped = cshogi.Board(flip_sfen(board.sfen()))
            np.testing.assert_array_equal(mirror_rows(original, perm)[0], kp.canonical_shared_row(flipped))
            legal = list(board.legal_moves)
            board.push(legal[int(rng.integers(len(legal)))])

    def test_label_scalar_oracle(self):
        import math
        raw = np.array([-32000, -5000, -1999, -1, 0, 1, 1999, 5000, 32000], np.int16)
        for result in [-1, 0, 1]:
            actual = premix(raw, np.full(len(raw), result, np.int8))
            expected = []
            for x in raw:
                if result == 0:
                    expected.append(x); continue
                p = .7 / (1 + math.exp(-int(x) / 600)) + .3 * (result == 1)
                p = max(1e-15, min(1-1e-15, p))
                expected.append(round(max(-2000, min(2000, 600 * math.log(p / (1-p))))))
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(int(premix(np.array([32000], np.int16), np.array([-1], np.int8))[0]), 508)

    def invoke(self, out, extra=(), script=None, wrapper=None, cache_path=None):
        cmd = [sys.executable, '-B']
        cmd += [str(wrapper)] if wrapper else ([str(script)] if script else ['-m', 'trainer.train_nnue'])
        cmd += [str(cache_path or self.parts[0]['cache']), '--out', str(out), '--qat', '--qat-tie', '--factorize',
                '--feature-dim', '2', '--hidden-dim', '2', '--batch-size', '2', '--epochs', '2',
                '--seed', '20260621', '--device', 'cpu', '--lr', '6e-5', '--augment-mirror',
                '--mirror-cache', str(self.root / 'mirror.npy'), *map(str, extra)]
        env = dict(os.environ, PYTHONPATH=str(REPO) + ':' + str(REPO / 'trainer'))
        return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)

    def test_legacy_byte_identity_and_cap_and_resume(self):
        original = subprocess.run(['git', 'show', 'HEAD:trainer/train_nnue.py'], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout
        old = self.root / 'legacy.py'; old.write_text(original)
        for fast in (False, True):
            extra = ['--fast', '--compact-cache', self.parts[0]['compact']] if fast else []
            old_out, new_out = self.root / f'old{fast}', self.root / f'new{fast}'
            for out, script in [(old_out, old), (new_out, None)]:
                r = self.invoke(out, extra, script)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for name in ('nnue_final.pt', 'nnue_best.pt', 'model_config.json'):
                self.assertEqual((old_out / name).read_bytes(), (new_out / name).read_bytes(), name)
            h1, h2 = [json.loads((out / 'history.json').read_text()) for out in (old_out, new_out)]
            for h in (h1, h2):
                for e in h['history']:
                    e.pop('train_secs')
            self.assertEqual(h1, h2)
        build(self.parts, self.val, self.work, chunk=2)
        # A one-part all-true manifest preserves Task135's legacy shuffle.
        c_out = self.root / 'manifestC'
        r = self.invoke(c_out, ['--manifest', self.work / 'C.json', '--fast'])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.root / 'newTrue/nnue_final.pt').read_bytes(), (c_out / 'nnue_final.pt').read_bytes())
        extra = ['--manifest', self.work / 'R.json', '--fast', '--max-steps', '10', '--resume',
                 '--checkpoint-minutes', '0.000000001']
        full, resumed = self.root / 'full', self.root / 'resumed'
        r = self.invoke(full, extra); self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # Independent materialized concatenation has identical targets, updates,
        # shuffling, and all 77 decoded columns across every part boundary.
        merged = cache(self.root, 'merged', np.concatenate([np.load(Path(p['cache']) / 'train_indices.npy')
                       for p in self.parts]), rows()[16:])
        np.save(merged['labels'], Manifest(self.work / 'R.json').vector('labels').astype(np.int16))
        merge_out = self.root / 'merged_out'
        r = self.invoke(merge_out, ['--compact-cache', merged['compact'], '--fast', '--max-steps', '10'],
                        cache_path=merged['cache'])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((full / 'nnue_final.pt').read_bytes(), (merge_out / 'nnue_final.pt').read_bytes())
        # Crash before optimizer update 4; checkpoint after update 3 regenerates
        # the global shuffled order and mirror RNG, across part and epoch boundaries.
        wrapper = self.root / 'interrupt.py'
        wrapper.write_text('''import torch
from trainer.train_nnue import main
original = torch.optim.Adam.step
calls = 0
def step(self, *a, **kw):
 global calls
 calls += 1
 if calls == 4: raise SystemExit(76)
 return original(self, *a, **kw)
torch.optim.Adam.step = step
raise SystemExit(main())
''')
        r = self.invoke(resumed, extra, wrapper=wrapper)
        self.assertEqual(r.returncode, 76, r.stdout + r.stderr)
        r = self.invoke(resumed, extra); self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((full / 'nnue_final.pt').read_bytes(), (resumed / 'nnue_final.pt').read_bytes())
        hist = json.loads((resumed / 'history.json').read_text())
        self.assertEqual(hist['updates'], 10)
        r = self.invoke(resumed, extra); self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((full / 'nnue_final.pt').read_bytes(), (resumed / 'nnue_final.pt').read_bytes())
        r = self.invoke(resumed, [*extra, '--lr', '1e-5'])
        self.assertNotEqual(r.returncode, 0)
        cap_crash = self.root / 'cap_crash'
        wrapper.write_text('''import torch
import trainer.train_nnue as trainer
step0, batches0 = torch.optim.Adam.step, trainer.run_batches
calls = 0
def step(self, *a, **kw):
 global calls
 calls += 1
 return step0(self, *a, **kw)
def batches(*a, **kw):
 if not kw['train'] and calls == 10: raise SystemExit(77)
 return batches0(*a, **kw)
torch.optim.Adam.step, trainer.run_batches = step, batches
raise SystemExit(trainer.main())
''')
        r = self.invoke(cap_crash, extra, wrapper=wrapper)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        r = self.invoke(cap_crash, extra)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((full / 'nnue_final.pt').read_bytes(), (cap_crash / 'nnue_final.pt').read_bytes())


    def test_export_resume_rebuilds_partial_files_and_gate(self):
        data = self.root / 'data'
        out = self.root / 'model'; out.mkdir()
        logs = self.root / 'export_logs'; logs.mkdir()
        (out / 'nnue_final.pt').write_bytes(b'weights')
        (out / 'model_config.json').write_text('{}')
        for rel in ('engines/yaneuraou/source/YaneuraOu-by-gcc', 'engines/yaneuraou/eval/nn.bin', 'binbo'):
            path = data / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'engine')
        env = dict(BINBO=str(data / 'binbo'))
        ev = data / 'eval_dirs/task172_test/nn.bin'
        gate = logs / 'task172_test_0cp.log'
        def partial(*a, **kw):
            ev.parent.mkdir(parents=True, exist_ok=True)
            ev.write_bytes(b'truncated')
            gate.write_text('partial')
            raise RuntimeError('interrupted')
        with patch.object(t172_run, 'export', side_effect=partial):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                t172_run.export_stage('task172_test', out, data, logs, Path(sys.executable), env)
        def complete(*a, **kw):
            self.assertFalse(ev.exists()); self.assertFalse(gate.exists())
            ev.write_bytes(b'complete')
            (out / 'nnue_final.bnnq').write_bytes(b'bnnq')
            gate.write_text('GREEN GREEN')
            return []
        with patch.object(t172_run, 'export', side_effect=complete) as writer:
            self.assertEqual(t172_run.export_stage('task172_test', out, data, logs, Path(sys.executable), env), [])
            self.assertEqual(t172_run.export_stage('task172_test', out, data, logs, Path(sys.executable), env), [])
            self.assertEqual(writer.call_count, 1)
        gate.write_text('RED')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            t172_run.export_stage('task172_test', out, data, logs, Path(sys.executable), env)

    def test_stage_completion_identity_and_skip(self):
        out = self.root / 'stage_out'
        log = self.root / 'logs'; log.mkdir()
        fake = self.root / 'fake_trainer.py'
        fake.write_text('from pathlib import Path\nimport json\np=Path(' + repr(str(out)) + ')\n'
                        '(p/"nnue_final.pt").write_bytes(b"final")\n'
                        '(p/"history.json").write_text(json.dumps({"updates":3}))\n')
        cmd = [sys.executable, '-B', fake]
        with patch.object(t172_run.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='')):
            result = t172_run.train_stage('test', cmd, out, log, dict(os.environ), 3)
        self.assertEqual(result['steps'], 3)
        with patch.object(t172_run.subprocess, 'Popen', side_effect=AssertionError('must skip')):
            self.assertEqual(t172_run.train_stage('test', cmd, out, log, dict(os.environ), 3), result)
        with self.assertRaisesRegex(ValueError, 'changed'):
            t172_run.train_stage('test', [*cmd, 'changed'], out, log, dict(os.environ), 3)
        (out / 'nnue_final.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            t172_run.train_stage('test', cmd, out, log, dict(os.environ), 3)

    def test_runner_commands_pilot_and_full(self):
        build(self.parts, self.val, self.work, chunk=2)
        env = dict(os.environ, DATA_ROOT=str(self.root), VENV_PY=sys.executable,
                   MODELS_DIR=str(self.root / 'models'), LOGS_DIR=str(self.root / 'logs'))
        commands = []
        def train(name, cmd, out, logs, run_env, expected_steps=None):
            commands.append((name, cmd, out, expected_steps))
            return {'steps': expected_steps}
        with patch.dict(os.environ), patch.object(t172_run, 'env_paths', return_value=env), \
             patch.object(t172_run, 'sources', return_value=self.parts), \
             patch.object(t172_run, 'build', return_value={}), \
             patch.object(t172_run, 'train_stage', side_effect=train), \
             patch.object(t172_run, 'export_stage', return_value=[]) as exp, \
             patch.object(t172_run, 'measure_all', return_value=dict(failures=[], verdict='NO_BETTER')) as frame:
            with patch.object(sys, 'argv', ['t172_run', '--work', str(self.work), '--pilot', '2000']):
                self.assertEqual(t172_run.main(), 0)
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][3], 2000)
            self.assertIn(str(self.work / 'R.json'), list(map(str, commands[0][1])))
            self.assertNotIn('--limit-train', commands[0][1])
            exp.assert_not_called(); frame.assert_not_called()
            commands.clear()
            with patch.object(sys, 'argv', ['t172_run', '--work', str(self.work)]):
                self.assertEqual(t172_run.main(), 0)
            self.assertEqual(len(commands), 6)
            for name, cmd, out, steps in commands[:3]:
                self.assertEqual(steps, 749424)
                self.assertNotIn('--init-weights', cmd)
                self.assertEqual(cmd[cmd.index('--cp-result-lambda') + 1], '1')
            for name, cmd, out, steps in commands[3:]:
                self.assertEqual(cmd[cmd.index('--epochs') + 1], '2')
                self.assertEqual(Path(cmd[cmd.index('--init-weights') + 1]).name, 'nnue_final.pt')
                self.assertEqual(cmd[cmd.index('--lr') + 1], '1e-5')
            self.assertEqual(exp.call_count, 3)
            self.assertEqual(frame.call_count, 1)
            self.assertTrue((Path(env['LOGS_DIR']) / 'task172_results.json').exists())
            exp.return_value = ['0cp RED']
            frame.reset_mock()
            with patch.object(sys, 'argv', ['t172_run', '--work', str(self.work)]):
                self.assertEqual(t172_run.main(), 1)
            frame.assert_not_called()

    def test_runner_frames_and_confirmation(self):
        calls = []
        def measure(name, env, **kw):
            calls.append((name, kw))
            return name, []
        def delta(a, b, label):
            elo = {'task172_C1': 1, 'task172_R1': 3, 'task172_W1': 2}[a]
            return dict(delta_elo=elo, ci95=[1, 4]), {'tanuki': dict(delta_elo=1)}, {}, []
        with patch.object(t172_run.std_frame, 'measure_sets', side_effect=measure), \
             patch.object(t172_run.std_frame, 'delta_sets', side_effect=delta):
            result = t172_run.measure_all({k: f'task172_{k}' for k in ('C1', 'R1', 'W1')}, {})
        self.assertEqual(result['verdict'], 'ADOPT_CANDIDATE')
        self.assertEqual(result['best'], 'task172_R1')
        self.assertEqual(calls[0][1]['sets'], t172_run.std_frame.SEED_SETS)
        self.assertTrue(any(c[1].get('sets') == {'conf172': t172_run.std_frame.spaced_set(20295000)} for c in calls))
        self.assertTrue(any(c[1].get('which') == t172_run.std_frame.CONFIRM_ANCHORS for c in calls))


if __name__ == '__main__':
    unittest.main()
