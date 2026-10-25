"""Task 137: offline statistical and pairing oracles (no engines)."""

import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.alphazero import seq_frame as seq


def rows(score, chunks=8):
    return [{"game": i % 100, "color": "b" if i % 2 == 0 else "w", "score": score,
             "opening_seed": str(i // 2), "opening": f"opening-{i // 2}", "reason": "mate"}
            for i in range(chunks * 100)]


class SequentialFrameTests(unittest.TestCase):
    def test_fft_against_exhaustive_bootstrap(self):
        # Four realisable pair differences, with all 4**4 bootstrap samples.
        values = np.array([-4, 0, 1, 4])
        counts = np.bincount(values + 4, minlength=9)
        sums = np.array([sum(x) for x in itertools.product(values, repeat=4)])
        expected = np.bincount(sums + 16, minlength=33) / 4 ** 4
        np.testing.assert_allclose(seq.bootstrap_distribution(counts), expected, atol=2e-15)
        for alpha in (.025, seq.TAIL_ALPHA[0]):
            cdf = expected.cumsum()
            lo = (np.searchsorted(cdf, alpha) - 16) / 16
            hi = (np.searchsorted(cdf, 1 - alpha) - 16) / 16
            np.testing.assert_allclose(seq.bootstrap_bounds(counts, alpha), [lo, hi])

    def test_vectorized_fft_and_swap_symmetry(self):
        counts = np.array([[1, 2, 3, 1, 9, 0, 4, 2, 0], [0, 2, 4, 0, 9, 1, 3, 2, 1]])
        batch = seq.bootstrap_distribution(counts)
        for i in range(2):
            np.testing.assert_allclose(batch[i], seq.bootstrap_distribution(counts[i]), atol=1e-14)
        np.testing.assert_allclose(batch[0], batch[1, ::-1], atol=1e-14)

    def test_gaussian_boundary_against_independent_random_walk(self):
        # Independent Gaussian paths are an external oracle for the quadrature.
        probability = seq.gaussian_crossing(160)
        self.assertAlmostEqual(probability, seq.gaussian_crossing(240), places=11)
        self.assertLess(probability, .0201)
        self.assertGreater(probability, .0198)
        rng = np.random.default_rng(137)
        z = rng.standard_normal((200_000, 8)).cumsum(axis=1) / np.sqrt(np.arange(1, 9))
        observed = float((z.max(axis=1) > seq.BOUNDARY).mean())
        self.assertLess(abs(observed - probability), 4 * math.sqrt(probability * (1 - probability) / len(z)))

    def test_pair_mean_preserves_covariance(self):
        candidate = {a: rows(.5, 1) for a in seq.ANCHORS}
        base = {a: rows(.5, 1) for a in seq.ANCHORS}
        # Every candidate pair is (win,loss), every base pair (draw,draw).
        for a in seq.ANCHORS:
            for i, row in enumerate(candidate[a]):
                row["score"] = 1. if i % 2 == 0 else 0.
        pairs = seq.paired_scores(candidate, base)
        for a in seq.ANCHORS:
            np.testing.assert_array_equal(seq.difference_counts(pairs[a]), [0, 0, 0, 0, 50, 0, 0, 0, 0])
        look = list(seq.sequential_looks(candidate, base))[-1]
        self.assertEqual(look["bounds"], [0., 0.])
        self.assertEqual(look["decision"], "continue")

    def test_terminal_and_prefix_behavior(self):
        candidate = {a: rows(.5) for a in seq.ANCHORS}
        base = {a: rows(.5) for a in seq.ANCHORS}
        looks = list(seq.sequential_looks(candidate, base))
        self.assertEqual([x["decision"] for x in looks], ["continue"] * 7 + ["no_difference"])
        for score, expected in ((1., "better"), (0., "worse")):
            candidate = {a: rows(score, 1) for a in seq.ANCHORS}
            prefix = {a: x[:100] for a, x in base.items()}
            looks = list(seq.sequential_looks(candidate, prefix))
            self.assertEqual(len(looks), 1)
            self.assertEqual(looks[0]["decision"], expected)
            self.assertEqual(looks[0]["candidate_games"], 200)

    def test_malformed_pairing_is_rejected(self):
        mutations = [lambda r: r[0].update(color="w"),
                     lambda r: r[1].update(opening="different"),
                     lambda r: r[0].update(opening_seed="different"),
                     lambda r: r[0].update(score=.75),
                     lambda r: r[0].update(reason="engine_fail"),
                     lambda r: r.pop()]
        for mutate in mutations:
            candidate = {a: rows(.5, 1) for a in seq.ANCHORS}
            base = {a: rows(.5, 1) for a in seq.ANCHORS}
            mutate(candidate["tanuki"])
            with self.assertRaises(ValueError):
                list(seq.sequential_looks(candidate, base))

    def test_old_yardstick_and_missing_chunks_do_not_enter(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            for nodes in (10000, 20000):
                (logs / f"std_task999_tanuki{nodes}_c00_seed20272001_n100.tsv").touch()
            groups, old = seq.discover(logs)
            self.assertEqual(old, 1)
            self.assertIsNone(seq.frame_seeds(groups[("std", "task999")]))
            comparisons, inventory = seq.complete_comparisons(logs)
            self.assertFalse(comparisons)
            self.assertEqual(inventory["excluded"][0]["reason"], "incomplete frame")

    def test_tilting_keeps_valid_real_pairs_and_target_elo(self):
        support = np.array([[[1., .5], [.5, 0.]], [[.5, 1.], [0., .5]],
                            [[1., 1.], [0., 0.]], [[0., 0.], [1., 1.]]])
        for target in (0, 20, -40, 80, -80):
            weights, metadata = seq.tilted_pool(support, np.full(4, .25), target)
            self.assertAlmostEqual(weights.sum(), 1.)
            self.assertAlmostEqual(metadata["actual_delta_elo"], target, places=9)
            self.assertTrue((weights > 0).all())


if __name__ == "__main__":
    unittest.main()
