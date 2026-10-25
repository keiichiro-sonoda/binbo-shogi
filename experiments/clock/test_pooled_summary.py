#!/usr/bin/env python3
"""Synthetic fixtures only: never open an experiment result or metadata file."""
import copy
import csv
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pooled_summary import analyze, main, stratified_bootstrap
import numpy as np


class PooledSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.roots = [Path(self.temp.name) / name for name in ('local', 'remote')]
        for m, root in enumerate(self.roots):
            prefix = f'/synthetic/machine{m}'
            spec = dict(argv=[prefix + '/engine'], cwd=prefix,
                        options={'EvalDir': prefix + '/eval', 'SlowMover': '100'})
            for arm in ('default', 'cap30', 'slow70'):
                path = root / arm
                path.mkdir(parents=True)
                args = dict(arm=arm, games=6, seed=100 + 6*m, threads=1, hash_mb=256,
                            black_base_ms=18000, white_base_ms=60000,
                            black_inc_ms=200, white_inc_ms=200, network_margin_ms=0,
                            charge_quantum_ms=0, opening_plies=4, max_plies=512,
                            a_color='black', rule='cap' if arm == 'cap30' else 'none',
                            cap_ms=3000, byoyomi_ms=300, k=15, opening_file=None,
                            a_option=['SlowMover=70'] if arm == 'slow70' else [],
                            parallel=20 if m == 0 else 40, cpu_budget=20 if m == 0 else 40)
                meta = dict(args=args, config={'A': copy.deepcopy(spec), 'B': copy.deepcopy(spec)},
                            script_sha256={'clock_match.py': 'harness', 'clock_rule.py': 'rule'},
                            assets_sha256={prefix + '/engine': 'engine-hash',
                                           prefix + '/eval/nn.bin': 'eval-hash'},
                            ponder=False, clock='fixture monotonic', cshogi_version='fixture')
                (path / 'run.json').write_text(json.dumps(meta))
                (root / 'matrix.json').write_text(json.dumps({'script_sha256': 'matrix'}))
                for i in range(6):
                    # Machine 0 has +1 / +.4; machine 1 has -1 / -.6.
                    # Stratified replicates MUST be exactly 0 / -.1.
                    score = ([0, 1, .4] if m == 0 else [1, 0, .4])[
                        ('default', 'cap30', 'slow70').index(arm)]
                    row = dict(seed=100 + 6*m + i, arm=arm, black='A', white='B',
                               opening=f'fixture {m} {i}', opening_id=f'{m}-{i}',
                               result={0: '0-1', .4: '1/2-1/2', 1: '1-0'}[score],
                               black_score=score, reason='time_forfeit' if i == 0 else 'fixture',
                               black_ms=json.dumps([1] * 16),
                               black_remaining_trace=json.dumps(list(range(100, 84, -1))))
                    with (path / f'{i}.tsv').open('w') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row), delimiter='\t')
                        writer.writeheader()
                        writer.writerow(row)

    def analyze(self):
        return analyze(self.roots, registered_seeds=range(100, 112))

    def change_meta(self, machine, key, value):
        for arm in ('default', 'cap30', 'slow70'):
            file = self.roots[machine] / arm / 'run.json'
            meta = json.loads(file.read_text())
            if key == 'option':
                for spec in meta['config'].values():
                    spec['options']['SlowMover'] = value
            elif key == 'asset':
                for asset in meta['assets_sha256']:
                    meta['assets_sha256'][asset] = value
            elif key == 'clock':
                meta['args']['black_base_ms'] = value
            else:
                meta[key] = value
            file.write_text(json.dumps(meta))

    def test_descriptions_pooled_and_stratification(self):
        output = self.analyze()
        comparisons = output['pooled']['comparisons']
        for arm, expected in [('cap30', 0), ('slow70', -.1)]:
            result = comparisons[arm]
            self.assertAlmostEqual(result['delta'], expected)
            np.testing.assert_allclose(result['ci95'], [expected, expected], atol=1e-15)
            np.testing.assert_allclose(result['ci97_5_bonferroni_two_arms'],
                                       [expected, expected], atol=1e-15)
            self.assertFalse(result['pass'])
        default = output['pooled']['arms']['default']
        self.assertEqual(default['wdl'], dict(wins=6, draws=0, losses=6))
        self.assertEqual(default['time_forfeits'], 2)
        self.assertEqual(default['minimum_remaining_ms']['min'], 85)
        self.assertEqual(default['sente_first15_own_moves_ms']['median'], 15)
        self.assertEqual(output, self.analyze())

    def test_unequal_stratum_sizes_stay_fixed(self):
        # A nonstratified bootstrap fluctuates; equal machine weighting gives .5.
        values = [np.ones((2, 2)), np.zeros((6, 2))]
        boot = stratified_bootstrap(values, np.random.default_rng(180), 200)
        np.testing.assert_array_equal(boot, np.full((200, 2), .25))

    def test_repeated_openings_remain_one_cluster(self):
        values = [np.array([[0., 0.], [1., 1.], [1., 1.]]), np.zeros((3, 2))]
        boot = stratified_bootstrap(values, np.random.default_rng(180), 200,
                                    cluster_ids=[['a', 'b', 'b'], ['c', 'd', 'e']])
        # Resampling two clusters yields 0, 2/3 or 1 in machine 0.
        # Its fixed 3/6 weight gives 0, 1/3 or 1/2 in the pool.
        np.testing.assert_allclose(np.unique(boot[:, 0]), [0, 1/3, .5])
        np.testing.assert_array_equal(boot[:, 0], boot[:, 1])

    def test_cross_machine_mismatches(self):
        # Consistent within the remote machine, incompatible across machines.
        for key, bad in [('option', '99'), ('asset', 'wrong'), ('clock', 17000),
                         ('script_sha256', {'clock_match.py': 'different'})]:
            with self.subTest(key=key):
                originals = {p: p.read_text() for p in self.roots[1].glob('*/run.json')}
                self.change_meta(1, key, bad)
                with self.assertRaisesRegex(ValueError, 'cross-machine'):
                    self.analyze()
                for path, original in originals.items():
                    path.write_text(original)

    def test_incomplete_rejected(self):
        (self.roots[1] / 'cap30/5.tsv').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete run'):
            self.analyze()

    def test_registered_union_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, 'registered set'):
            analyze(self.roots)  # Tiny fixture must never pass the production gate.

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            analyze([self.roots[0], self.roots[0]], registered_seeds=range(100, 106))

    def test_within_machine_mismatch_rejected(self):
        path = self.roots[0] / 'cap30/run.json'
        meta = json.loads(path.read_text())
        meta['args']['hash_mb'] = 128
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'clock/resource mismatch'):
            self.analyze()

    def test_json_and_short_table(self):
        expected = self.analyze()
        out = Path(self.temp.name) / 'synthetic-pooled.json'
        stdout = io.StringIO()
        with patch('pooled_summary.analyze', return_value=expected), patch(
                'sys.argv', ['pooled_summary.py', *map(str, self.roots), '--out', str(out)]):
            with redirect_stdout(stdout):
                main()
        self.assertEqual(json.loads(out.read_text()), expected)
        self.assertIn('descriptive', stdout.getvalue())
        self.assertIn('pooled', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
