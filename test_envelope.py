#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_envelope.py
===========================================================
Unit tests for envelope.py -- the core sweep engine behind the
FLIGHT ENVELOPE tab (alpha_sweep / mach_sweep / altitude_sweep).

These tests never touch the real .pkl model files: they monkeypatch
envelope.aerodynamic_prediction with a small deterministic fake, so
they test the sweep LOGIC (range/step handling, dict shape, CL/CD
ratio, zero-step clamping) in complete isolation from the ML model.

Run with:
    python -m pytest tests/test_envelope.py -v
    python -m unittest tests.test_envelope -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envelope as E


def _fake_prediction(params):
    """
    Deterministic stand-in for predictor.aerodynamic_prediction: CL/CD
    track whichever swept variable is present so tests can assert on
    them directly, and XCP_D is intentionally omitted from the dict
    (mirrors the real predictor.py v10.0 behaviour where XCP_D is not
    always present) to exercise envelope.py's pred.get('XCP_D') fallback.
    """
    swept_val = params.get('alpha', params.get('mach', params.get('alt', 0.0)))
    cl = 1.0 + 0.1 * float(swept_val)
    cd = 0.5 + 0.01 * float(swept_val)
    xcp = -5.0
    return {'CL': cl, 'CD': cd, 'XCP': xcp}


BASE_PARAMS = {
    'nose_len': 300, 'body_len': 2700, 'wing_le': 1500,
    'root_chord': 200, 'tip_chord': 150, 'semi_span': 1000,
    'root_th': 20, 'tip_th': 5, 'wing_sweep': 2.86,
    'tail_le': 2870, 'root_chord1': 120, 'tip_chord1': 60,
    'semi_span1': 100, 'root_th1': 15, 'tip_th1': 5,
    'mach': 0.2, 'alpha': 2, 'alt': 0,
}


class EnvelopeTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_pred = E.aerodynamic_prediction
        E.aerodynamic_prediction = _fake_prediction

    def tearDown(self):
        E.aerodynamic_prediction = self._orig_pred


class TestAlphaSweep(EnvelopeTestCase):
    def test_row_count_covers_full_range(self):
        rows = E.alpha_sweep(BASE_PARAMS, 0, 10, 2)
        # 0, 2, 4, 6, 8, 10 -> 6 rows
        self.assertEqual(len(rows), 6)

    def test_row_keys(self):
        rows = E.alpha_sweep(BASE_PARAMS, 0, 4, 2)
        for r in rows:
            for k in ('alpha', 'CL', 'CD', 'XCP', 'XCP_D', 'CLCD'):
                self.assertIn(k, r)

    def test_alpha_values_are_monotonic_and_bounded(self):
        rows = E.alpha_sweep(BASE_PARAMS, 0, 10, 2)
        alphas = [r['alpha'] for r in rows]
        self.assertEqual(alphas, sorted(alphas))
        self.assertGreaterEqual(alphas[0], 0)
        self.assertLessEqual(alphas[-1], 10)

    def test_clcd_matches_cl_over_cd(self):
        rows = E.alpha_sweep(BASE_PARAMS, 0, 4, 2)
        for r in rows:
            expected = round(r['CL'] / r['CD'], 4) if abs(r['CD']) > 1e-9 else 0.0
            self.assertAlmostEqual(r['CLCD'], expected, places=4)

    def test_missing_xcpd_becomes_none(self):
        # _fake_prediction never returns 'XCP_D' -- envelope.py must
        # fall back to None via pred.get('XCP_D') rather than raising.
        rows = E.alpha_sweep(BASE_PARAMS, 0, 2, 2)
        for r in rows:
            self.assertIsNone(r['XCP_D'])

    def test_zero_or_negative_step_is_clamped(self):
        # step<=0 must not hang / must fall back to a safe default step.
        rows = E.alpha_sweep(BASE_PARAMS, 0, 4, 0)
        self.assertGreater(len(rows), 0)
        rows_neg = E.alpha_sweep(BASE_PARAMS, 0, 4, -3)
        self.assertGreater(len(rows_neg), 0)

    def test_single_point_sweep(self):
        rows = E.alpha_sweep(BASE_PARAMS, 5, 5, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['alpha'], 5)

    def test_does_not_mutate_base_params(self):
        base_copy = dict(BASE_PARAMS)
        E.alpha_sweep(BASE_PARAMS, 0, 4, 2)
        self.assertEqual(BASE_PARAMS, base_copy)


class TestMachSweep(EnvelopeTestCase):
    def test_row_count_and_rounding(self):
        rows = E.mach_sweep(BASE_PARAMS, 0.2, 0.8, 0.2)
        machs = [r['mach'] for r in rows]
        self.assertEqual(machs, [0.2, 0.4, 0.6, 0.8])

    def test_zero_step_uses_default(self):
        rows = E.mach_sweep(BASE_PARAMS, 0.2, 0.8, 0)
        self.assertGreater(len(rows), 0)


class TestAltitudeSweep(EnvelopeTestCase):
    def test_row_count(self):
        rows = E.altitude_sweep(BASE_PARAMS, 0, 4000, 1000)
        alts = [r['alt'] for r in rows]
        self.assertEqual(alts, [0.0, 1000.0, 2000.0, 3000.0, 4000.0])

    def test_zero_step_uses_default(self):
        rows = E.altitude_sweep(BASE_PARAMS, 0, 4000, 0)
        self.assertGreater(len(rows), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
