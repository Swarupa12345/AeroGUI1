#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================
# envelope.py  —  DRDL Aerospace AI Platform  v9.0
# =========================================================
# CHANGES vs previous:
#   • XCP_D now included in every sweep result dict so
#     render_flight() in app.py can plot it.
#   • Step clamped to always be > 0 (same as before).
# =========================================================

import numpy as np
from predictor import aerodynamic_prediction


# =========================================================
# ALPHA SWEEP
# =========================================================

def alpha_sweep(base_params, alpha_min, alpha_max, step):
    """
    Vary angle-of-attack from alpha_min to alpha_max.

    Returns list of dicts:
        {'alpha', 'CL', 'CD', 'XCP', 'XCP_D', 'CLCD'}
    """
    results = []
    step = step if step > 0 else 1.0

    for alpha in np.arange(alpha_min, alpha_max + step * 0.5, step):
        params = base_params.copy()
        params['alpha'] = round(float(alpha), 4)
        pred = aerodynamic_prediction(params)
        cl, cd, xcp = pred['CL'], pred['CD'], pred['XCP']
        clcd = cl / cd if abs(cd) > 1e-9 else 0.0
        results.append({
            'alpha': round(float(alpha), 4),
            'CL'   : cl,
            'CD'   : cd,
            'XCP'  : xcp,
            'XCP_D': pred.get('XCP_D'),   # may be None for interpolated rows
            'CLCD' : round(clcd, 4),
        })

    return results


# =========================================================
# MACH SWEEP
# =========================================================

def mach_sweep(base_params, mach_min, mach_max, step):
    """
    Vary Mach number from mach_min to mach_max.

    Returns list of dicts:
        {'mach', 'CL', 'CD', 'XCP', 'XCP_D', 'CLCD'}
    """
    results = []
    step = step if step > 0 else 0.1

    for mach in np.arange(mach_min, mach_max + step * 0.5, step):
        params = base_params.copy()
        params['mach'] = round(float(mach), 4)
        pred = aerodynamic_prediction(params)
        cl, cd, xcp = pred['CL'], pred['CD'], pred['XCP']
        clcd = cl / cd if abs(cd) > 1e-9 else 0.0
        results.append({
            'mach' : round(float(mach), 4),
            'CL'   : cl,
            'CD'   : cd,
            'XCP'  : xcp,
            'XCP_D': pred.get('XCP_D'),
            'CLCD' : round(clcd, 4),
        })

    return results


# =========================================================
# ALTITUDE SWEEP
# =========================================================

def altitude_sweep(base_params, alt_min, alt_max, step):
    """
    Vary altitude from alt_min to alt_max.

    Returns list of dicts:
        {'alt', 'CL', 'CD', 'XCP', 'XCP_D', 'CLCD'}
    """
    results = []
    step = step if step > 0 else 1000.0

    for alt in np.arange(alt_min, alt_max + step * 0.5, step):
        params = base_params.copy()
        params['alt'] = round(float(alt), 1)
        pred = aerodynamic_prediction(params)
        cl, cd, xcp = pred['CL'], pred['CD'], pred['XCP']
        clcd = cl / cd if abs(cd) > 1e-9 else 0.0
        results.append({
            'alt'  : round(float(alt), 1),
            'CL'   : cl,
            'CD'   : cd,
            'XCP'  : xcp,
            'XCP_D': pred.get('XCP_D'),
            'CLCD' : round(clcd, 4),
        })

    return results