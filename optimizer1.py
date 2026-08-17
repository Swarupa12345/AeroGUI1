#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py  —  DRDL Aerospace AI Platform  v10.0
===========================================================
CHANGES vs v9.1:
  1. NO CSV FILE REQUIRED.  _build_test_set() now generates
     a synthetic uniform test matrix from the scaler's own
     min/max range rather than loading DRDL_aero_data_final.csv.
     The optimizer always uses _X_TEST_SCALED from predictor.py
     cache (500 synthetic points covering the training range).
     CSV fallback removed entirely.
  2. All other logic (DE loop, top-5 heap, fitness function,
     artefact writers) unchanged — numerically identical.
  3. PARAM_NAMES ordering unchanged; public API unchanged.
===========================================================
"""

import os
import time
import types
import heapq
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import openpyxl        # noqa: F401
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

# =========================================================
# PARAMETER NAMES  (same order as DEFAULTS in app.py)
# =========================================================
PARAM_NAMES = [
    "nose_len", "body_len", "wing_le",
    "root_chord", "tip_chord", "semi_span",
    "root_th", "tip_th", "wing_sweep",
    "tail_le", "root_chord1", "tip_chord1",
    "semi_span1", "root_th1", "tip_th1",
    "mach", "alpha", "alt",
]

_FEATURES = [
    "nose length", "body_length", "wing LE", "root chord", "tip chord",
    "semi-span", "root th", "tip th", "wing sweep", "tail LE",
    "root chord.1", "tip chord.1", "semi-span.1", "root th.1",
    "tip th.1", "MACH", "ALPHA", "ALT",
]

_TARGETS = ["CL", "CD", "XCP"]
_FIXED   = {"MACH", "ALPHA", "ALT"}

# =========================================================
# FITNESS CONSTANTS
# =========================================================
W1          = 0.95
W2          = 0.05
K_FIT       = 0.2464
XCP_TARGET  = -5.15
CD_MIN_HARD = 0.12

_TOP_N_FEATURES = 5

# =========================================================
# BORROW SCALER + BOOSTERS FROM predictor.py
# =========================================================
def _get_predictor_assets():
    """
    Return (boosters_dict, scaler, mode_str, top5_indices,
            X_test_scaled_cached) by importing from predictor.py.
    """
    try:
        import predictor as _pred
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import predictor.py — make sure it is in the "
            "same folder as optimizer.py."
        ) from exc

    boosters = getattr(_pred, '_BOOSTERS', None)
    scaler   = getattr(_pred, '_SCALER',   None)
    mode     = 'ensemble' if getattr(_pred, 'ENSEMBLE_MODE', False) else 'xgboost'

    if boosters is None or scaler is None:
        raise RuntimeError(
            "predictor.py has not finished loading yet.\n"
            "Wait for the 'Model ready' status before running the optimizer."
        )

    # Pull the FULL feature ranking (not just top-5) so we can drop
    # MACH/ALPHA/ALT before slicing to top-5. Those three are pinned
    # to a single fixed value by app.py's optimizer bounds (they are
    # flight conditions, not shape parameters), so letting one of the
    # 5 optimized slots land on them wastes a slot -- it can never
    # move, and it does not affect the 3D aero-body geometry either.
    # Excluding them here guarantees all 5 optimized dimensions are
    # real shape parameters, so the 3D body visibly changes every
    # generation instead of appearing frozen.
    _all_ranked_indices = _pred.get_top_feature_indices(len(_FEATURES))
    _fixed_indices = {_FEATURES.index(f) for f in _FIXED if f in _FEATURES}
    top5_indices = [i for i in _all_ranked_indices if i not in _fixed_indices]
    top5_indices = top5_indices[:_TOP_N_FEATURES]
    X_test_scaled_cached = getattr(_pred, '_X_TEST_SCALED', None)

    return boosters, scaler, mode, top5_indices, X_test_scaled_cached


# =========================================================
# SYNTHETIC TEST SET  (no CSV — uses scaler range)
# =========================================================
def _build_synthetic_test_set(scaler, n_points=500, seed=42):
    """
    Generate a synthetic test matrix covering the full scaled [0,1] space.
    This replaces the CSV-based test set entirely.
    Used only when _X_TEST_SCALED from predictor.py is None.
    """
    np.random.seed(seed)
    n_feat = len(_FEATURES)
    X_scaled = np.random.uniform(0.0, 1.0,
                                 size=(n_points, n_feat)).astype(np.float32)
    return X_scaled


# =========================================================
# BOUNDS HELPERS
# =========================================================
def _build_scaled_bounds(raw_bounds, scaler):
    """
    Convert raw parameter bounds to scaled bounds.
    Uses a named DataFrame so the MinMaxScaler's feature
    alignment never raises a column mismatch.
    """
    n   = len(_FEATURES)
    out = np.empty((n, 2), dtype=float)

    for i in range(n):
        row_lo = pd.DataFrame([[raw_bounds[j][0] if j == i else 0.0
                                for j in range(n)]],
                              columns=_FEATURES)
        row_hi = pd.DataFrame([[raw_bounds[j][1] if j == i else 0.0
                                for j in range(n)]],
                              columns=_FEATURES)
        out[i, 0] = float(scaler.transform(row_lo)[0, i])
        out[i, 1] = float(scaler.transform(row_hi)[0, i])

    return out


def _inverse_transform(vec_scaled, scaler):
    """Inverse-transform a single 18-D scaled vector to raw values."""
    dummy = pd.DataFrame([vec_scaled], columns=_FEATURES)
    return scaler.inverse_transform(dummy)[0]


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def _make_objective(X_test_scaled_base, boosters, top5_indices,
                    fixed_vec, user_constraints):
    """
    Returns (objective_fn, working_matrix).
    working_matrix is a (N, 18) array pre-filled with fixed_vec.
    objective_fn(sub_vec_5d) → float fitness.
    """
    N      = X_test_scaled_base.shape[0]
    n_feat = X_test_scaled_base.shape[1]

    mat = np.empty((N, n_feat), dtype=np.float32)
    mat[:] = fixed_vec   # broadcast (1,18) → (N,18)

    def _objective(sub_vec):
        for rank, col_idx in enumerate(top5_indices):
            mat[:, col_idx] = sub_vec[rank]

        cl  = boosters["CL"].predict(mat)
        cd  = boosters["CD"].predict(mat)
        xcp = boosters["XCP"].predict(mat)

        if np.any(cd < CD_MIN_HARD):
            return -np.inf

        if user_constraints:
            preds = {
                "CL":  float(cl.mean()),
                "CD":  float(cd.mean()),
                "XCP": float(xcp.mean()),
            }
            for metric, (lo, hi) in user_constraints.items():
                v = preds.get(metric)
                if v is not None and not (lo <= v <= hi):
                    return -np.inf

        f1 = float(np.mean(cl / cd))
        f2 = float(np.mean(1.0 / (np.abs(xcp - XCP_TARGET) + 1e-12)))
        return W1 * f1 + W2 * K_FIT * f2

    return _objective, mat


def _evaluate_metrics_sub(sub_vec, mat, boosters, top5_indices):
    for rank, col_idx in enumerate(top5_indices):
        mat[:, col_idx] = sub_vec[rank]
    cl  = boosters["CL"].predict(mat)
    cd  = boosters["CD"].predict(mat)
    xcp = boosters["XCP"].predict(mat)
    return (float(cl.mean()), float(cd.mean()),
            float(xcp.mean()), float((cl / cd).mean()))


# =========================================================
# TOP-5 BEST SOLUTIONS HEAP
# =========================================================
class _Top5Heap:
    """Min-heap of size 5 tracking best unique solutions by fitness."""

    def __init__(self, maxsize=5):
        self._heap    = []
        self._counter = 0
        self._maxsize = maxsize

    def push(self, fitness, sub_vec):
        if not np.isfinite(fitness):
            return
        self._counter += 1
        entry = (fitness, self._counter, sub_vec.copy())
        if len(self._heap) < self._maxsize:
            heapq.heappush(self._heap, entry)
        elif fitness > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def best_sorted(self):
        """Return list of (fitness, sub_vec) sorted best-first."""
        return [(e[0], e[2]) for e in
                sorted(self._heap, key=lambda x: -x[0])]


# =========================================================
# CUSTOM DE LOOP
# =========================================================
def _de_loop(bounds_scaled, X_test_scaled, boosters, scaler,
             generations, popsize, user_constraints, itermax,
             cr_min, cr_max, log_callback, top5_indices, fixed_vec,
             gen_callback=None, geom_snapshot_every=1):

    n_full = bounds_scaled.shape[0]   # 18
    n_opt  = len(top5_indices)        # 5

    sub_bounds = bounds_scaled[top5_indices]  # (5, 2)

    _objective, mat = _make_objective(
        X_test_scaled, boosters, top5_indices, fixed_vec, user_constraints)

    baseline   = (sub_bounds[:, 0] + sub_bounds[:, 1]) / 2.0
    population = np.empty((popsize, n_opt), dtype=float)
    population[0] = baseline
    for i in range(1, popsize):
        population[i] = np.random.uniform(sub_bounds[:, 0], sub_bounds[:, 1])

    best_fitness_hist = []
    avg_fitness_hist  = []
    best_cl_hist      = []
    best_cd_hist      = []
    best_xcp_hist     = []
    best_clcd_hist    = []
    gen_elapsed_hist  = []
    geom_hist         = []

    # Normalise geom_snapshot_every: 0/None disables snapshots entirely.
    snap_every = geom_snapshot_every if geom_snapshot_every else 0

    best_solution = None
    best_score    = -np.inf
    top5_heap     = _Top5Heap(maxsize=5)

    for gen in range(generations):
        gen_t0 = time.perf_counter()

        fitness_pop = np.array([
            _objective(population[i]) for i in range(popsize)
        ])

        for i in range(popsize):
            top5_heap.push(fitness_pop[i], population[i])

        rank = np.argsort(fitness_pop)
        Cr   = np.empty(popsize, dtype=float)
        for i, idx in enumerate(rank):
            Cr[idx] = cr_min + (cr_max - cr_min) * (i / max(popsize - 1, 1))

        # K (pull toward the current best) and F (differential weight)
        # are tuned to stay exploratory across the whole run rather than
        # collapsing onto one design in the first few generations -- a
        # higher K / narrower F used to converge so fast that the live
        # 3D view stopped visibly changing after generation 2-3 even on
        # short (e.g. 10-generation) runs. These values keep the
        # population -- and so the geometry snapshot below -- still
        # meaningfully moving generation-to-generation for most/all of
        # a typical run, while still converging by the end.
        K      = 0.5
        mutant = np.empty_like(population)
        for i in range(popsize):
            a, b, c = np.random.choice(popsize, 3, replace=False)
            F   = np.random.uniform(-0.5, 0.5)
            vec = (population[i]
                   + K * (population[a] - population[i])
                   + F * (population[b] - population[c]))
            mutant[i] = np.clip(vec, sub_bounds[:, 0], sub_bounds[:, 1])

        trial = np.empty_like(population)
        for i in range(popsize):
            mask      = np.random.rand(n_opt) < Cr[i]
            trial_vec = np.where(mask, population[i], mutant[i])

            best_vec = trial_vec.copy()
            best_fit = _objective(best_vec)

            for _ in range(itermax):
                idx  = np.random.randint(0, n_opt)
                cand = best_vec.copy()
                if cand[idx] == population[i][idx]:
                    cand[idx] = mutant[i][idx]
                else:
                    cand[idx] = population[i][idx]

                cand_fit = _objective(cand)
                if cand_fit > best_fit:
                    best_fit, best_vec = cand_fit, cand
                else:
                    break

            trial[i] = best_vec

        new_pop = population.copy()
        for i in range(popsize):
            fit_t = _objective(trial[i])
            if fit_t > fitness_pop[i]:
                new_pop[i]     = trial[i]
                fitness_pop[i] = fit_t
                top5_heap.push(fit_t, trial[i])

        population = new_pop

        gen_elapsed_ms = round((time.perf_counter() - gen_t0) * 1000, 2)
        gen_elapsed_hist.append(gen_elapsed_ms)

        valid    = fitness_pop[np.isfinite(fitness_pop)]
        best_gen = float(fitness_pop.max())
        avg_gen  = float(valid.mean()) if valid.size else -np.inf

        best_fitness_hist.append(best_gen)
        avg_fitness_hist.append(avg_gen)

        gen_best_idx = int(np.argmax(fitness_pop))
        cl_g, cd_g, xcp_g, clcd_g = _evaluate_metrics_sub(
            population[gen_best_idx], mat, boosters, top5_indices)

        best_cl_hist.append(cl_g)
        best_cd_hist.append(cd_g)
        best_xcp_hist.append(xcp_g)
        best_clcd_hist.append(clcd_g)

        if fitness_pop[gen_best_idx] > best_score:
            best_score    = fitness_pop[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        # ── Optional geometry snapshot (every Nth generation) ────────
        # Expands this generation's best 5-D sub-vector to the full
        # 18-D raw parameter dict so the GUI can redraw the 3D aero
        # body live, and so the Undo/Redo history can scrub through
        # generations after the run completes.
        geom_snap = None
        if snap_every and (gen % snap_every == 0):
            full_scaled = _expand_sub_to_full(
                population[gen_best_idx], fixed_vec.ravel(), top5_indices)
            raw = _inverse_transform(full_scaled, scaler)
            geom_snap = {PARAM_NAMES[i]: round(float(raw[i]), 4)
                        for i in range(len(PARAM_NAMES))}
        geom_hist.append(geom_snap)

        msg = (f"Gen {gen+1:02d} – best fitness: {best_score:.6f}"
               f"  [{gen_elapsed_ms:.1f} ms]  "
               f"(optimising {n_opt} of {n_full} dims)")
        print(msg)
        if log_callback:
            log_callback(msg)

        if gen_callback:
            gen_callback({
                "generation" : gen + 1,
                "fitness"    : best_score,
                "avg_fitness": avg_gen,
                "CL"         : cl_g,
                "CD"         : cd_g,
                "XCP"        : xcp_g,
                "CLCD"       : clcd_g,
                "elapsed_ms" : gen_elapsed_ms,
                "geom"       : geom_snap,
            })

    history = [
        {
            "generation"  : g + 1,
            "fitness"     : best_fitness_hist[g],
            "avg_fitness" : avg_fitness_hist[g],
            "CL"          : best_cl_hist[g],
            "CD"          : best_cd_hist[g],
            "XCP"         : best_xcp_hist[g],
            "CLCD"        : best_clcd_hist[g],
            "elapsed_ms"  : gen_elapsed_hist[g] if g < len(gen_elapsed_hist) else 0,
            "geom"        : geom_hist[g] if g < len(geom_hist) else None,
        }
        for g in range(generations)
    ]

    return best_solution, history, top5_heap


# =========================================================
# EXPAND 5-D SUB-VECTOR → 18-D FULL VECTOR
# =========================================================
def _expand_sub_to_full(sub_vec, fixed_vec, top5_indices):
    full = fixed_vec.copy()
    for rank, col_idx in enumerate(top5_indices):
        full[col_idx] = sub_vec[rank]
    return full


# =========================================================
# ARTEFACT WRITERS
# =========================================================
def _write_artefacts(best_scaled_full, history, scaler,
                     boosters, X_test_scaled, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    raw_best = _inverse_transform(best_scaled_full, scaler)

    geom_cols = [f for f in _FEATURES if f not in _FIXED]
    geom_vals = [raw_best[_FEATURES.index(f)] for f in geom_cols]
    geom_df   = pd.DataFrame([geom_vals], columns=geom_cols)
    geom_path = os.path.join(out_dir, "best_geometry.csv")
    geom_df.to_csv(geom_path, index=False)

    X_one  = scaler.transform(pd.DataFrame([raw_best], columns=_FEATURES))
    cl_all = boosters["CL"].predict(X_one)
    cd_all = boosters["CD"].predict(X_one)
    xcp_all= boosters["XCP"].predict(X_one)
    clcd_all = cl_all / np.where(np.abs(cd_all) > 1e-9, cd_all, np.nan)

    perf_df = pd.DataFrame(X_one, columns=_FEATURES)
    perf_df["CL_pred"]    = cl_all
    perf_df["CD_pred"]    = cd_all
    perf_df["XCP_pred"]   = xcp_all
    perf_df["CL/CD_pred"] = clcd_all
    perf_df.to_csv(os.path.join(out_dir, "full_performance.csv"), index=False)

    pd.DataFrame([{
        "CL_mean":   float(cl_all.mean()),   "CL_std":   float(cl_all.std()),
        "CD_mean":   float(cd_all.mean()),   "CD_std":   float(cd_all.std()),
        "CLCD_mean": float(clcd_all.mean()), "CLCD_std": float(clcd_all.std()),
        "XCP_mean":  float(xcp_all.mean()),  "XCP_std":  float(xcp_all.std()),
    }]).to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    evo_df = pd.DataFrame({
        "Generation"  : [h["generation"]  for h in history],
        "BestFitness" : [h["fitness"]     for h in history],
        "AvgFitness"  : [h["avg_fitness"] for h in history],
        "Best_CL"     : [h["CL"]          for h in history],
        "Best_CD"     : [h["CD"]          for h in history],
        "Best_CL/CD"  : [h["CLCD"]        for h in history],
        "Best_XCP"    : [h["XCP"]         for h in history],
        "Gen_Time_ms" : [h["elapsed_ms"]  for h in history],
    })
    if _HAS_OPENPYXL:
        evo_df.to_excel(os.path.join(out_dir, "evolution_history.xlsx"),
                        index=False)
    else:
        evo_df.to_csv(os.path.join(out_dir, "evolution_history.csv"),
                      index=False)

    print(f"All artefacts written to: {out_dir}")
    return raw_best, perf_df


# =========================================================
# PUBLIC API
# =========================================================
def run_optimization(bounds, maxiter=50, popsize=10,
                     constraints=None, out_dir=None,
                     log_callback=None,
                     itermax=5, cr_min=0.3, cr_max=0.9,
                     gen_callback=None, geom_snapshot_every=1):
    """
    Run Differential Evolution optimisation.

    CSV file is NOT required. Uses synthetic test matrix from
    predictor.py (_X_TEST_SCALED) or generates a new one on demand.

    gen_callback : optional callable(gen_info) invoked after every
        generation with a dict of that generation's stats
        (generation, fitness, avg_fitness, CL, CD, XCP, CLCD,
        elapsed_ms, geom). 'geom' is a full 18-key raw-parameter
        dict (PARAM_NAMES → value) on snapshotted generations, else
        None.
    geom_snapshot_every : int — take a geometry snapshot every Nth
        generation (1 = every generation). 0/None disables snapshots
        (geom will always be None, but gen_callback still fires).

    Returns
    -------
    result  : SimpleNamespace (.x, .success, .message, .fun,
                               .perf_df, .mode, .top5_solutions)
    history : list of per-generation dicts
    elapsed : float seconds (total)
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed.")

    t0 = time.perf_counter()

    (boosters, scaler, mode,
     top5_indices, X_test_scaled_cached) = _get_predictor_assets()

    # Use cached synthetic matrix from predictor; generate if None
    if X_test_scaled_cached is not None:
        X_test_scaled = X_test_scaled_cached
    else:
        X_test_scaled = _build_synthetic_test_set(scaler)

    bounds_scaled = _build_scaled_bounds(bounds, scaler)
    fixed_vec = ((bounds_scaled[:, 0] + bounds_scaled[:, 1]) / 2.0).reshape(1, -1)

    best_sub, history, top5_heap = _de_loop(
        bounds_scaled    = bounds_scaled,
        X_test_scaled    = X_test_scaled,
        boosters         = boosters,
        scaler           = scaler,
        generations      = maxiter,
        popsize          = popsize,
        user_constraints = constraints,
        itermax          = itermax,
        cr_min           = cr_min,
        cr_max           = cr_max,
        log_callback     = log_callback,
        top5_indices     = top5_indices,
        fixed_vec        = fixed_vec,
        gen_callback     = gen_callback,
        geom_snapshot_every = geom_snapshot_every,
    )

    elapsed = time.perf_counter() - t0

    # Build top-5 list
    top5_solutions = []
    for rank, (fit, sub_vec) in enumerate(top5_heap.best_sorted(), 1):
        full_scaled = _expand_sub_to_full(sub_vec, fixed_vec.ravel(), top5_indices)
        raw = _inverse_transform(full_scaled, scaler)
        params_dict = {PARAM_NAMES[i]: round(float(raw[i]), 4)
                       for i in range(len(PARAM_NAMES))}

        X1    = scaler.transform(pd.DataFrame([full_scaled], columns=_FEATURES))
        cl_v  = float(boosters["CL"].predict(X1)[0])
        cd_v  = float(boosters["CD"].predict(X1)[0])
        xcp_v = float(boosters["XCP"].predict(X1)[0])
        ld_v  = cl_v / cd_v if abs(cd_v) > 1e-9 else 0.0

        top5_solutions.append({
            'rank'   : rank,
            'fitness': round(fit, 6),
            'CL'     : round(cl_v,  4),
            'CD'     : round(cd_v,  4),
            'XCP'    : round(xcp_v, 4),
            'CLCD'   : round(ld_v,  4),
            'params' : params_dict,
        })

    if best_sub is None:
        result = types.SimpleNamespace(
            x              = np.zeros(len(PARAM_NAMES)),
            success        = False,
            message        = "No feasible solution — all candidates violated constraints.",
            fun            = np.inf,
            perf_df        = None,
            mode           = mode,
            top5_solutions = top5_solutions,
        )
        return result, history, elapsed

    best_full_scaled = _expand_sub_to_full(
        best_sub, fixed_vec.ravel(), top5_indices)
    raw_best = _inverse_transform(best_full_scaled, scaler)
    perf_df  = None

    if out_dir:
        raw_best, perf_df = _write_artefacts(
            best_full_scaled, history, scaler,
            boosters, X_test_scaled, out_dir)

    best_score = history[-1]["fitness"] if history else -np.inf

    result = types.SimpleNamespace(
        x              = raw_best,
        success        = np.isfinite(best_score),
        message        = "DE optimisation complete (no CSV required).",
        fun            = -best_score,
        perf_df        = perf_df,
        mode           = mode,
        top5_solutions = top5_solutions,
    )
    return result, history, elapsed