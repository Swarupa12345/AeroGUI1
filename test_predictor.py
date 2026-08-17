#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_predictor.py
===========================================================
Unit tests for predictor.py -- the core engine behind the
PREDICTION tab.

Two tiers:
  1. Pure-logic tests that need NO model files (rounding
     contract, GUI-key -> model-column mapping, input row
     construction). These always run.
  2. End-to-end prediction tests that need the real
     xgb_model.pkl / minmax_scaler.pkl / metrics.pkl next to
     predictor.py. These are skipped automatically (not
     failed) when those files aren't present, e.g. in a bare
     checkout or CI box without the trained artefacts.

Run with:
    python -m pytest tests/test_predictor.py -v
    python -m unittest tests.test_predictor -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictor as P


class TestParamRounding(unittest.TestCase):
    """PARAM_DECIMALS contract: every GUI param is rounded to its
    documented number of decimal places before it ever reaches the
    model, so cache keys / repeated calls are stable."""

    def test_all_params_have_decimal_rule(self):
        for gui_key in P.PARAM_TO_COL:
            self.assertIn(gui_key, P.PARAM_DECIMALS,
                          f'{gui_key} missing from PARAM_DECIMALS')

    def test_rounding_applied(self):
        raw = {k: 1.23456789 for k in P.PARAM_TO_COL}
        rounded = P._round_params(raw)
        for k, v in rounded.items():
            self.assertEqual(v, round(1.23456789, P.PARAM_DECIMALS[k]))

    def test_rounding_is_idempotent(self):
        raw = {k: 42.987654321 for k in P.PARAM_TO_COL}
        once = P._round_params(raw)
        twice = P._round_params(once)
        self.assertEqual(once, twice)


class TestParamColumnMapping(unittest.TestCase):
    """GUI key <-> model column mapping must be a complete bijection
    onto INPUT_COLS, or the model will silently see garbage/NaN
    columns."""

    def test_mapping_covers_all_input_cols(self):
        mapped_cols = set(P.PARAM_TO_COL.values())
        self.assertEqual(mapped_cols, set(P.INPUT_COLS))

    def test_mapping_is_one_to_one(self):
        cols = list(P.PARAM_TO_COL.values())
        self.assertEqual(len(cols), len(set(cols)),
                         'duplicate model column in PARAM_TO_COL')

    def test_row_has_correct_column_order(self):
        rounded = {k: float(i) for i, k in enumerate(P.PARAM_TO_COL)}
        row_df = P._params_to_row(rounded)
        self.assertEqual(list(row_df.columns), P.INPUT_COLS)
        self.assertEqual(len(row_df), 1)

    def test_row_values_match_gui_inputs(self):
        rounded = {k: float(i) for i, k in enumerate(P.PARAM_TO_COL)}
        row_df = P._params_to_row(rounded)
        for gui_key, col in P.PARAM_TO_COL.items():
            self.assertAlmostEqual(row_df.iloc[0][col], rounded[gui_key])


class TestOutputContract(unittest.TestCase):
    def test_output_cols_fixed(self):
        self.assertEqual(P.OUTPUT_COLS, ['CL', 'CD', 'XCP'])


_MODEL_FILES_PRESENT = all(
    os.path.exists(f) for f in (P.MODEL_FILE, P.SCALER_FILE, P.METRIC_FILE))


@unittest.skipUnless(
    _MODEL_FILES_PRESENT,
    'xgb_model.pkl / minmax_scaler.pkl / metrics.pkl not found next to '
    'predictor.py -- skipping end-to-end prediction tests. Copy the '
    'trained .pkl artefacts alongside predictor.py to enable these.')
class TestEndToEndPrediction(unittest.TestCase):
    """These exercise the real model and only run when the trained
    .pkl artefacts are available (see skip reason above)."""

    SAMPLE_PARAMS = {
        'nose_len': 300, 'body_len': 2700, 'wing_le': 1500,
        'root_chord': 200, 'tip_chord': 150, 'semi_span': 1000,
        'root_th': 20, 'tip_th': 5, 'wing_sweep': 2.86,
        'tail_le': 2870, 'root_chord1': 120, 'tip_chord1': 60,
        'semi_span1': 100, 'root_th1': 15, 'tip_th1': 5,
        'mach': 0.2, 'alpha': 2, 'alt': 0,
    }

    def test_prediction_returns_required_keys(self):
        result = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        for key in ('CL', 'CD', 'XCP', 'XCP_D', 'source', 'mode',
                   'elapsed_ms', 'metrics', 'detailed_metrics',
                   'dataset_match', 'top_features'):
            self.assertIn(key, result)

    def test_prediction_outputs_are_finite_floats(self):
        result = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        for key in ('CL', 'CD', 'XCP'):
            self.assertIsInstance(result[key], float)
            self.assertTrue(result[key] == result[key])  # not NaN

    def test_prediction_is_deterministic(self):
        r1 = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        r2 = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        self.assertEqual(r1['CL'], r2['CL'])
        self.assertEqual(r1['CD'], r2['CD'])
        self.assertEqual(r1['XCP'], r2['XCP'])

    def test_xcpd_is_none_in_pkl_only_mode(self):
        # v10.0 contract: XCP_D always None at runtime (no CSV available).
        result = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        self.assertIsNone(result['XCP_D'])

    def test_dataset_match_always_false(self):
        result = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        self.assertFalse(result['dataset_match'])

    def test_top_features_returns_five(self):
        result = P.aerodynamic_prediction(self.SAMPLE_PARAMS)
        self.assertLessEqual(len(result['top_features']), 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
