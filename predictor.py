#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py  —  DRDL Aerospace AI Platform  v10.0
===========================================================
CHANGES vs v7.0:
  • ZERO CSV dependency at runtime.
    All predictions come purely from the .pkl model files:
      - xgb_model.pkl        (trained XGBoost MultiOutputRegressor)
      - minmax_scaler.pkl    (fitted MinMaxScaler, 18 features)
      - metrics.pkl          (pre-computed MAE/RMSE/R2 per output)
      - ensemble_models.pkl  (RF + GB models, optional)

  • XCP/D removed from runtime output since it requires
    CM/CN columns that only exist in the CSV.
    XCP/D is now returned as None always (no CSV lookup).

  • CSV file is NOT opened, NOT read, NOT required at runtime.
    The .pkl files are the sole source of truth.

  • All other public API (aerodynamic_prediction, get_top_features,
    get_top_feature_indices, ENSEMBLE_MODE) unchanged.
===========================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Tuple, Dict

warnings.filterwarnings("ignore")

# =========================================================
# TOGGLE: set True to use Ensemble, False for XGBoost only
# =========================================================
ENSEMBLE_MODE = False

# =========================================================
# FILE PATHS  (only .pkl files needed at runtime)
# =========================================================
HERE          = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE    = os.path.join(HERE, 'xgb_model.pkl')
SCALER_FILE   = os.path.join(HERE, 'minmax_scaler.pkl')
METRIC_FILE   = os.path.join(HERE, 'metrics.pkl')
ENSEMBLE_FILE = os.path.join(HERE, 'ensemble_models.pkl')

# =========================================================
# FEATURE / OUTPUT DEFINITIONS
# (must match the order used when the model was trained)
# =========================================================
INPUT_COLS = [
    'nose length', 'body_length', 'wing LE', 'root chord', 'tip chord',
    'semi-span', 'root th', 'tip th', 'wing sweep', 'tail LE',
    'root chord.1', 'tip chord.1', 'semi-span.1', 'root th.1',
    'tip th.1', 'MACH', 'ALPHA', 'ALT',
]
OUTPUT_COLS = ['CL', 'CD', 'XCP']

# =========================================================
# GUI KEY → CSV/MODEL COLUMN MAPPING
# =========================================================
PARAM_TO_COL = {
    'nose_len'    : 'nose length',
    'body_len'    : 'body_length',
    'wing_le'     : 'wing LE',
    'root_chord'  : 'root chord',
    'tip_chord'   : 'tip chord',
    'semi_span'   : 'semi-span',
    'root_th'     : 'root th',
    'tip_th'      : 'tip th',
    'wing_sweep'  : 'wing sweep',
    'tail_le'     : 'tail LE',
    'root_chord1' : 'root chord.1',
    'tip_chord1'  : 'tip chord.1',
    'semi_span1'  : 'semi-span.1',
    'root_th1'    : 'root th.1',
    'tip_th1'     : 'tip th.1',
    'mach'        : 'MACH',
    'alpha'       : 'ALPHA',
    'alt'         : 'ALT',
}

# =========================================================
# ROUNDING CONTRACT
# =========================================================
PARAM_DECIMALS = {
    'nose_len'   : 0,  'body_len'   : 0,  'wing_le'    : 0,
    'root_chord' : 0,  'tip_chord'  : 0,  'semi_span'  : 0,
    'root_th'    : 2,  'tip_th'     : 2,  'wing_sweep' : 2,
    'tail_le'    : 0,  'root_chord1': 0,  'tip_chord1' : 0,
    'semi_span1' : 0,  'root_th1'   : 2,  'tip_th1'    : 2,
    'mach'       : 3,  'alpha'      : 2,  'alt'        : 1,
}

FLOAT_TOL = 1e-6

# Module-level cache — loaded once per process
_cache: Dict = {}

# =========================================================
# HELPERS
# =========================================================
def _round_params(params: Dict) -> Dict:
    return {k: round(float(params[k]), PARAM_DECIMALS[k]) for k in params}


def _params_to_row(rounded_params: Dict) -> pd.DataFrame:
    """Convert GUI params dict → single-row DataFrame with model column names."""
    row = {col: float(rounded_params[gui_key])
           for gui_key, col in PARAM_TO_COL.items()}
    return pd.DataFrame([row], columns=INPUT_COLS)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
def _compute_feature_importance(model) -> dict:
    n_feat = len(INPUT_COLS)
    combined = np.zeros(n_feat, dtype=float)
    per_output = {}
    for i, col in enumerate(OUTPUT_COLS):
        fi = model.estimators_[i].feature_importances_
        per_output[col] = fi.tolist()
        combined += fi
    avg = combined / len(OUTPUT_COLS)
    return {
        'features'       : INPUT_COLS,
        'avg_importance' : avg.tolist(),
        'per_output'     : per_output,
    }


def get_top_features(n: int = 5) -> List[Tuple[str, float]]:
    """Return top-n (feature_name, importance_score) tuples, sorted descending."""
    cache = _load_models()
    fi = cache.get('feature_importance', {})
    if not fi:
        return []
    ranked = sorted(zip(fi['features'], fi['avg_importance']),
                    key=lambda x: x[1], reverse=True)
    return ranked[:n]


def get_top_feature_indices(n: int = 5) -> list:
    """Return indices of top-n features (used by optimizer.py)."""
    cache = _load_models()
    fi = cache.get('feature_importance', {})
    if not fi:
        return list(range(n))
    avg_imp = fi['avg_importance']
    return sorted(range(len(avg_imp)), key=lambda i: avg_imp[i], reverse=True)[:n]


# =========================================================
# MODEL LOADER  (PKL ONLY — no CSV)
# =========================================================
def _load_models() -> Dict:
    """
    Load all models from .pkl files. Called once; result cached in _cache.
    Raises RuntimeError with a clear message if any required file is missing.
    """
    global _cache
    if _cache:
        return _cache

    # ── Validate required files exist ────────────────────
    missing = [f for f in [MODEL_FILE, SCALER_FILE, METRIC_FILE]
               if not os.path.exists(f)]
    if missing:
        raise RuntimeError(
            "MISSING PKL FILES — cannot run without these model files:\n" +
            "\n".join(f"  • {m}" for m in missing) +
            "\n\nPlease ensure all .pkl files are in the same folder as app.py."
        )

    # ── Load XGBoost model ────────────────────────────────
    try:
        xgb_model = joblib.load(MODEL_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to load xgb_model.pkl:\n{e}")

    # ── Load scaler ───────────────────────────────────────
    try:
        scaler = joblib.load(SCALER_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to load minmax_scaler.pkl:\n{e}")

    # ── Load pre-computed metrics ─────────────────────────
    try:
        metrics = joblib.load(METRIC_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to load metrics.pkl:\n{e}")

    # ── Load ensemble models (optional) ──────────────────
    ensemble_models = {}
    if os.path.exists(ENSEMBLE_FILE):
        try:
            ens_data = joblib.load(ENSEMBLE_FILE)
            if isinstance(ens_data, dict) and 'models' in ens_data:
                ensemble_models = ens_data['models']
            elif isinstance(ens_data, dict):
                ensemble_models = ens_data
        except Exception:
            ensemble_models = {}

    # ── Feature importance ────────────────────────────────
    feature_importance = _compute_feature_importance(xgb_model)

    # ── Build per-output booster dict for optimizer.py ───
    boosters = {}
    for i, col in enumerate(OUTPUT_COLS):
        key = 'XCP' if col == 'XCP' else col
        boosters[key] = xgb_model.estimators_[i]

    # ── Populate cache ────────────────────────────────────
    _cache['model']             = xgb_model
    _cache['scaler']            = scaler
    _cache['metrics']           = metrics
    _cache['ensemble_models']   = ensemble_models
    _cache['feature_importance']= feature_importance
    _cache['boosters']          = boosters

    # ── Expose assets for optimizer.py ───────────────────
    _expose_assets(_cache)
    _expose_assets_extended(_cache)

    return _cache


# =========================================================
# PREDICT HELPERS
# =========================================================
def _scale(row_df: pd.DataFrame, scaler) -> np.ndarray:
    """Scale a single-row DataFrame using the loaded scaler."""
    return scaler.transform(row_df).astype(np.float32)


def _predict_xgb(x_scaled: np.ndarray, cache: dict) -> Tuple[float, float, float]:
    pred = cache['model'].predict(x_scaled)[0]
    return (round(float(pred[0]), 4),
            round(float(pred[1]), 4),
            round(float(pred[2]), 4))


def _predict_ensemble(x_scaled: np.ndarray, cache: dict) -> Tuple[float, float, float]:
    ens = cache.get('ensemble_models', {})
    if not ens:
        return _predict_xgb(x_scaled, cache)
    preds = np.stack([m.predict(x_scaled) for m in ens.values()])
    avg = preds.mean(axis=0)[0]
    return (round(float(avg[0]), 4),
            round(float(avg[1]), 4),
            round(float(avg[2]), 4))


# =========================================================
# EXPOSE ASSETS FOR optimizer.py
# =========================================================
def _expose_assets(cache):
    mod = sys.modules[__name__]
    mod._SCALER   = cache['scaler']
    mod._BOOSTERS = cache['boosters']


def _expose_assets_extended(cache):
    """
    Pre-build a scaled test-like matrix for the optimizer to use
    without needing the CSV. We use the scaler's own data_min_/data_max_
    to generate a synthetic uniform grid covering the training range.
    """
    mod = sys.modules[__name__]
    if getattr(mod, '_X_TEST_SCALED', None) is not None:
        return

    scaler = cache['scaler']
    n_feat = len(INPUT_COLS)

    # Generate 500 synthetic points uniformly in [0,1] scaled space
    # These cover the full training range without needing the CSV
    np.random.seed(42)
    n_synth = 500
    X_synth_scaled = np.random.uniform(0.0, 1.0, size=(n_synth, n_feat)).astype(np.float32)
    mod._X_TEST_SCALED = X_synth_scaled


# Initialise _X_TEST_SCALED to None until model loads
sys.modules[__name__]._X_TEST_SCALED = None
sys.modules[__name__]._SCALER        = None
sys.modules[__name__]._BOOSTERS      = None


# =========================================================
# PUBLIC API
# =========================================================
def aerodynamic_prediction(params: Dict) -> Dict:
    """
    Predict CL, CD, XCP from 18 input parameters using .pkl models only.

    Parameters
    ----------
    params : dict — 18 GUI parameter names → float values

    Returns
    -------
    dict with keys:
        CL, CD, XCP, XCP_D,
        source, mode, elapsed_ms,
        metrics, detailed_metrics,
        dataset_match, top_features
    """
    t_start = time.perf_counter()

    # Step 1 — canonicalise inputs
    rounded = _round_params(params)

    # Step 2 — load models (cached after first call)
    cache = _load_models()

    # Step 3 — build input row and scale it
    row_df   = _params_to_row(rounded)
    x_scaled = _scale(row_df, cache['scaler'])

    # Step 4 — predict via XGBoost or Ensemble
    if ENSEMBLE_MODE:
        cl, cd, xcp = _predict_ensemble(x_scaled, cache)
        mode = 'ensemble'
    else:
        cl, cd, xcp = _predict_xgb(x_scaled, cache)
        mode = 'xgboost'

    # Step 5 — XCP/D is NOT available without CSV (CM/CN columns)
    # Return None — GUI shows "N/A (model-only mode)"
    xcpd = None

    # Step 6 — metrics from pre-computed metrics.pkl
    m = cache['metrics']
    # SANITY: pkl stores XCP under 'XCP' key — map it to 'XCP'
    _METRIC_ALIAS = {'XCP': 'XCP','CL': 'CL', 'CD': 'CD'}
    m_normalised = {}
    for raw_key, val in m.items():
        mapped = _METRIC_ALIAS.get(raw_key, raw_key)
        m_normalised[mapped] = val

    detailed_metrics = {}
    for col in OUTPUT_COLS:
        if col in m_normalised:
            detailed_metrics[col] = m_normalised[col]

    # Sanity check: warn if any output column is missing metrics
    _missing_met = [c for c in OUTPUT_COLS if c not in detailed_metrics]
    if _missing_met:
        import warnings as _w
        _w.warn(f'[predictor] metrics.pkl missing keys for: {_missing_met}')

    avg_metrics = {
        'MAE' : round(sum(detailed_metrics.get(c, {}).get('MAE',  0) for c in OUTPUT_COLS) / 3, 4),
        'RMSE': round(sum(detailed_metrics.get(c, {}).get('RMSE', 0) for c in OUTPUT_COLS) / 3, 4),
        'R2'  : round(sum(detailed_metrics.get(c, {}).get('R2',   0) for c in OUTPUT_COLS) / 3, 4),
    }

    # Step 7 — top features
    top5 = get_top_features(5)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 3)

    return {
        'CL'              : cl,
        'CD'              : cd,
        'XCP'             : xcp,
        'XCP_D'           : xcpd,          # None — no CSV at runtime
        'source'          : 'xgboost_pkl',
        'mode'            : mode,
        'elapsed_ms'      : elapsed_ms,
        'metrics'         : avg_metrics,
        'detailed_metrics': detailed_metrics,
        'dataset_match'   : False,         # No CSV lookup → always False
        'top_features'    : top5,
    }