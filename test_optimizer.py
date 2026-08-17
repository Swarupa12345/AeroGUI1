#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_optimizer.py
===========================================================
Unit tests for optimizer.py -- the core Differential Evolution
engine behind the OPTIMIZER tab.

Fully offline / no xgboost, no .pkl files required: predictor.py's
real boosters are replaced with a tiny deterministic FakeBooster
(linear model over the scaled feature vector) and a real
sklearn MinMaxScaler fit on synthetic data stands in for the
trained scaler. This exercises the actual DE loop, Top-5 heap,
bounds scaling, and per-generation callback/snapshot machinery
end-to-end without needing the trained model artefacts.

Run with:
    python -m pytest tests/test_optimizer.py -v
    python -m unittest tests.test_optimizer -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optimizer as O

try:
    from sklearn.preprocessing import MinMaxScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class FakeBooster:
    """Deterministic stand-in for an XGBoost estimator: predict() is a
    fixed linear function of the (already-scaled) feature matrix, so
    outputs are reproducible and cheap without touching xgboost."""

    def __init__(self, weights, bias):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = bias

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias


def _make_fake_assets():
    n_feat = len(O._FEATURES)
    rng = np.random.RandomState(0)

    # Real MinMaxScaler, fit on synthetic data spanning realistic-ish
    # ranges, so _build_scaled_bounds / _inverse_transform behave
    # exactly as they would against the real trained scaler.
    raw_sample = rng.uniform(0.0, 3000.0, size=(50, n_feat))
    scaler = MinMaxScaler()
    scaler.fit(pd.DataFrame(raw_sample, columns=O._FEATURES))

    boosters = {
        'CL' : FakeBooster(rng.uniform(0.5, 1.5, n_feat), bias=2.0),
        'CD' : FakeBooster(rng.uniform(0.05, 0.2, n_feat), bias=0.5),
        'XCP': FakeBooster(rng.uniform(-1.0, 1.0, n_feat), bias=-5.0),
    }
    top5_indices = list(range(5))  # first 5 features, arbitrary but fixed
    X_test_scaled = rng.uniform(0.0, 1.0, size=(40, n_feat)).astype(np.float32)
    return boosters, scaler, top5_indices, X_test_scaled


BOUNDS_18 = [
    (120, 360), (2400, 3000), (1000, 2000), (150, 250), (110, 190),
    (600, 1500), (15, 25), (5, 11), (0.0, 70.0), (2830, 2910),
    (80, 160), (30, 90), (60, 140), (15, 21), (5, 11),
    (0.2, 0.8), (0, 20), (0, 6000),
]


@unittest.skipUnless(_HAS_SKLEARN, 'scikit-learn not installed')
class OptimizerTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_has_xgb = O._HAS_XGB
        self._orig_get_assets = O._get_predictor_assets
        O._HAS_XGB = True  # bypass the "xgboost not installed" guard

        boosters, scaler, top5_indices, X_test_scaled = _make_fake_assets()
        self._boosters = boosters
        self._scaler = scaler

        def _fake_get_assets():
            return boosters, scaler, 'xgboost', top5_indices, X_test_scaled

        O._get_predictor_assets = _fake_get_assets

    def tearDown(self):
        O._HAS_XGB = self._orig_has_xgb
        O._get_predictor_assets = self._orig_get_assets


class TestGetPredictorAssetsFeatureSelection(unittest.TestCase):
    """Covers the top-N feature-selection logic inside
    optimizer._get_predictor_assets(): nose_len/body_len must always be
    force-included (see O._FORCE_INCLUDE) regardless of where the
    model's feature-importance ranking puts them, so the fuselage is
    never left out of what the DE loop is allowed to move. Exercised
    against a fake predictor module (no real .pkl artefacts needed) so
    it runs in any environment."""

    def setUp(self):
        self._orig_TOP_N = O._TOP_N_FEATURES

    def tearDown(self):
        O._TOP_N_FEATURES = self._orig_TOP_N
        sys.modules.pop('predictor', None)

    def _install_fake_predictor(self, ranked_indices):
        import types as _types
        fake = _types.ModuleType('predictor')
        fake._BOOSTERS = {'CL': object(), 'CD': object(), 'XCP': object()}
        fake._SCALER = object()
        fake.ENSEMBLE_MODE = False
        fake._X_TEST_SCALED = np.zeros((1, len(O._FEATURES)))
        fake.get_top_feature_indices = lambda n: ranked_indices[:n]
        sys.modules['predictor'] = fake
        return fake

    def test_nose_and_body_always_included_even_if_ranked_last(self):
        # Rank everything EXCEPT nose_len(0)/body_len(1) highest, so a
        # naive top-N-by-importance slice would exclude the fuselage
        # entirely (the exact bug being fixed).
        n = len(O._FEATURES)
        ranking = [i for i in range(n) if i not in (0, 1)] + [0, 1]
        self._install_fake_predictor(ranking)

        _, _, _, top_indices, _ = O._get_predictor_assets()

        self.assertIn(0, top_indices)  # nose length
        self.assertIn(1, top_indices)  # body_length
        self.assertLessEqual(len(top_indices), O._TOP_N_FEATURES)

    def test_top_n_size_respected(self):
        n = len(O._FEATURES)
        ranking = list(range(n))
        self._install_fake_predictor(ranking)

        _, _, _, top_indices, _ = O._get_predictor_assets()
        self.assertEqual(len(top_indices), O._TOP_N_FEATURES)

    def test_no_duplicate_indices(self):
        n = len(O._FEATURES)
        ranking = [0, 1] + [i for i in range(n) if i not in (0, 1)]
        self._install_fake_predictor(ranking)

        _, _, _, top_indices, _ = O._get_predictor_assets()
        self.assertEqual(len(top_indices), len(set(top_indices)))

    def test_fixed_flight_condition_indices_excluded(self):
        # MACH/ALPHA/ALT must never occupy an optimized slot.
        n = len(O._FEATURES)
        fixed_idx = {O._FEATURES.index(f) for f in O._FIXED}
        ranking = list(fixed_idx) + [i for i in range(n) if i not in fixed_idx]
        self._install_fake_predictor(ranking)

        _, _, _, top_indices, _ = O._get_predictor_assets()
        self.assertTrue(fixed_idx.isdisjoint(set(top_indices)))


class TestParamNames(unittest.TestCase):
    def test_param_names_length(self):
        self.assertEqual(len(O.PARAM_NAMES), 18)

    def test_param_names_match_features_count(self):
        self.assertEqual(len(O.PARAM_NAMES), len(O._FEATURES))


class TestTop5Heap(unittest.TestCase):
    def test_keeps_only_best_five(self):
        heap = O._Top5Heap(maxsize=5)
        for fitness in [1, 5, 3, 9, 2, 8, 7, 0.5, 6, 4]:
            heap.push(fitness, np.array([fitness]))
        best = heap.best_sorted()
        self.assertEqual(len(best), 5)
        fitnesses = [f for f, _ in best]
        self.assertEqual(fitnesses, sorted(fitnesses, reverse=True))
        self.assertEqual(fitnesses[0], 9)

    def test_ignores_non_finite_fitness(self):
        heap = O._Top5Heap(maxsize=5)
        heap.push(-np.inf, np.array([0.0]))
        heap.push(np.nan, np.array([0.0]))
        heap.push(3.0, np.array([1.0]))
        best = heap.best_sorted()
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0][0], 3.0)


@unittest.skipUnless(_HAS_SKLEARN, 'scikit-learn not installed')
class TestBoundsScaling(unittest.TestCase):
    def test_scaled_bounds_shape(self):
        _, scaler, _, _ = _make_fake_assets()
        scaled = O._build_scaled_bounds(BOUNDS_18, scaler)
        self.assertEqual(scaled.shape, (18, 2))

    def test_scaled_bounds_lo_le_hi(self):
        _, scaler, _, _ = _make_fake_assets()
        scaled = O._build_scaled_bounds(BOUNDS_18, scaler)
        self.assertTrue(np.all(scaled[:, 0] <= scaled[:, 1] + 1e-9))

    def test_inverse_transform_round_trip(self):
        _, scaler, _, _ = _make_fake_assets()
        scaled = O._build_scaled_bounds(BOUNDS_18, scaler)
        midpoint_scaled = (scaled[:, 0] + scaled[:, 1]) / 2.0
        raw = O._inverse_transform(midpoint_scaled, scaler)
        self.assertEqual(len(raw), 18)
        # Midpoint of each raw bound should round-trip close to the
        # midpoint of the original (lo, hi) bound.
        for i, (lo, hi) in enumerate(BOUNDS_18):
            self.assertAlmostEqual(raw[i], (lo + hi) / 2.0, delta=(hi - lo) * 0.05 + 1e-6)


class TestRunOptimizationEndToEnd(OptimizerTestCase):
    """Runs the real DE loop (small pop/gen count for speed) against
    the fake boosters/scaler from setUp -- exercises the full public
    API surface used by app.py's Optimizer tab."""

    def test_returns_expected_types(self):
        gen_events = []
        result, history, elapsed = O.run_optimization(
            BOUNDS_18, maxiter=3, popsize=6, itermax=2,
            constraints=None, out_dir=None,
            gen_callback=lambda gi: gen_events.append(gi),
            geom_snapshot_every=1)

        self.assertTrue(hasattr(result, 'x'))
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'top5_solutions'))
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_history_length_matches_generations(self):
        result, history, elapsed = O.run_optimization(
            BOUNDS_18, maxiter=4, popsize=6, itermax=2,
            constraints=None, out_dir=None, geom_snapshot_every=1)
        self.assertEqual(len(history), 4)

    def test_gen_callback_fires_once_per_generation(self):
        gen_events = []
        O.run_optimization(
            BOUNDS_18, maxiter=4, popsize=6, itermax=2,
            constraints=None, out_dir=None,
            gen_callback=lambda gi: gen_events.append(gi),
            geom_snapshot_every=1)
        self.assertEqual(len(gen_events), 4)
        for i, gi in enumerate(gen_events, 1):
            self.assertEqual(gi['generation'], i)
            for key in ('fitness', 'avg_fitness', 'CL', 'CD', 'XCP', 'CLCD', 'geom'):
                self.assertIn(key, gi)

    def test_geom_snapshot_every_disables_geom(self):
        gen_events = []
        O.run_optimization(
            BOUNDS_18, maxiter=3, popsize=6, itermax=2,
            constraints=None, out_dir=None,
            gen_callback=lambda gi: gen_events.append(gi),
            geom_snapshot_every=0)
        self.assertTrue(all(gi['geom'] is None for gi in gen_events))

    def test_geom_snapshot_every_one_always_populates_geom(self):
        gen_events = []
        O.run_optimization(
            BOUNDS_18, maxiter=3, popsize=6, itermax=2,
            constraints=None, out_dir=None,
            gen_callback=lambda gi: gen_events.append(gi),
            geom_snapshot_every=1)
        self.assertTrue(all(gi['geom'] is not None for gi in gen_events))
        for gi in gen_events:
            self.assertEqual(set(gi['geom'].keys()), set(O.PARAM_NAMES))

    def test_top5_solutions_capped_at_five(self):
        result, _, _ = O.run_optimization(
            BOUNDS_18, maxiter=5, popsize=8, itermax=2,
            constraints=None, out_dir=None, geom_snapshot_every=1)
        self.assertLessEqual(len(result.top5_solutions), 5)
        for sol in result.top5_solutions:
            self.assertEqual(set(sol['params'].keys()), set(O.PARAM_NAMES))

    def test_infeasible_constraints_yield_no_success(self):
        # CD constraint impossible to satisfy given fake boosters'
        # output range -> DE should report failure gracefully rather
        # than raising or returning a bogus "success".
        impossible = {'CD': (-1000.0, -999.0)}
        result, history, elapsed = O.run_optimization(
            BOUNDS_18, maxiter=2, popsize=4, itermax=1,
            constraints=impossible, out_dir=None, geom_snapshot_every=1)
        self.assertFalse(result.success)


if __name__ == '__main__':
    unittest.main(verbosity=2)