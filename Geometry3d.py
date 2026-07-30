#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry3d.py — DRDL Aerospace AI Platform
===========================================================
Pure 3-D geometry generation for the aircraft body, wing, and
tail, adapted from the `aero_body.py` specification and mapped
directly onto the 18 parameters already used by predictor.py /
optimizer.py / envelope.py (PARAM_NAMES).

This module intentionally contains NO plotting-window logic,
NO argparse, and NO Plotly/HTML export. Rendering is done by
app.py's existing `_mpl_style()` / `_embed_fig()` / plot-console
pipeline, exactly like the optimisation and sweep figures. This
module only returns numpy geometry arrays and one convenience
Matplotlib Figure builder.

GEOMETRY MODEL
--------------
Body axis = X (fuselage station, matches "body_len").
Cross-sections lie in the Y-Z plane (Y = spanwise, Z = vertical).
Nose + fuselage are surfaces of revolution about X.
Wing and horizontal tail are flat lifting-surface panels
(conventional layout — both extend in +-Y through Z = 0),
with a biconvex thickness distribution in Z.

NOTE ON BODY DIAMETER
----------------------
None of the 18 GUI/optimizer parameters specify fuselage
diameter. BODY_DIAMETER below is a fixed engineering assumption
(mm, same units as nose_len/body_len/etc.). Adjust it to match
your actual vehicle class, or wire it up as a 19th GUI field if
you want it exposed and tunable.
===========================================================
"""

from typing import Tuple, Dict
import math
import numpy as np

# =========================================================
# ENGINEERING CONSTANT (see module docstring)
# =========================================================
BODY_DIAMETER = 300.0   # mm — nominal fuselage diameter


# =========================================================
# VALIDATION HELPERS
# =========================================================
def _validate_positive(value: float, name: str) -> float:
    """Return value as float; raise ValueError if not strictly positive."""
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"'{name}' must be positive, got {value}")
    return value


def _deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return float(deg) * math.pi / 180.0


# =========================================================
# BLUNTED-CONE NOSE PROFILE
# =========================================================
def solve_unblunted_cone(length: float, radius: float, bluntness: float,
                          tol: float = 1e-6) -> Tuple[float, np.ndarray]:
    """
    Solve for the cone/sphere tangency station of a spherically
    blunted conical nose of given overall `length` and base
    `radius`, with bluntness in [0, 1) (0 = sharp point,
    approaching 1 = hemispherical cap).

    Returns
    -------
    (x_tangent, profile) where profile = np.array([x_tangent, y_tangent]).
    """
    length = _validate_positive(length, "length")
    radius = _validate_positive(radius, "radius")
    if not (0.0 <= bluntness < 1.0):
        raise ValueError("bluntness must lie in [0, 1)")

    r_n = bluntness * radius                     # spherical cap radius
    theta = math.atan2(radius, length)           # cone half-angle
    x_o = length - r_n / max(math.sin(theta), 1e-9)   # sphere-centre station

    # Bisection refine so the sphere and cone surfaces agree at the
    # tangent point to within `tol` (guards edge cases near bluntness -> 0/1).
    for _ in range(100):
        x_t = x_o - r_n * math.sin(theta)
        y_t = r_n * math.cos(theta)
        y_cone_at_xt = math.tan(theta) * (x_t - length) + radius
        err = y_t - y_cone_at_xt
        if abs(err) < tol:
            break
        x_o -= err

    return x_t, np.array([x_t, y_t])


def _nose_radius_profile(x: np.ndarray, nose_length: float,
                          nose_radius: float, nose_bluntness: float) -> np.ndarray:
    """Vectorised nose radius r(x) for 0 <= x <= nose_length."""
    x_t, (_, y_t) = solve_unblunted_cone(nose_length, nose_radius, nose_bluntness)
    r_n = nose_bluntness * nose_radius
    theta = math.atan2(nose_radius, nose_length)
    x_o = x_t + r_n * math.sin(theta)

    r = np.empty_like(x)
    cap = x < x_t
    r[cap] = np.sqrt(np.clip(r_n**2 - (x[cap] - x_o) ** 2, 0.0, None))
    r[~cap] = np.tan(theta) * (x[~cap] - nose_length) + nose_radius
    return np.clip(r, 0.0, None)


# =========================================================
# NOSE / BODY SURFACES OF REVOLUTION
# =========================================================
def generate_nose(nose_length: float, nose_diameter: float,
                   nose_bluntness: float,
                   n_x: int = 40, n_theta: int = 32
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) surface-of-revolution mesh for the nose cone."""
    nose_length = _validate_positive(nose_length, "nose_length")
    radius = _validate_positive(nose_diameter, "nose_diameter") / 2.0

    x = np.linspace(0.0, nose_length, n_x)
    r = _nose_radius_profile(x, nose_length, radius, nose_bluntness)
    th = np.linspace(0.0, 2 * math.pi, n_theta)

    X, TH = np.meshgrid(x, th)
    R, _ = np.meshgrid(r, th)
    Y = R * np.cos(TH)
    Z = R * np.sin(TH)
    return X, Y, Z


def generate_body(body_length: float, body_diameter: float, nose_length: float,
                   n_x: int = 20, n_theta: int = 32
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) surface-of-revolution mesh for the cylindrical fuselage."""
    body_length = _validate_positive(body_length, "body_length")
    radius = _validate_positive(body_diameter, "body_diameter") / 2.0
    nose_length = float(nose_length)

    x = np.linspace(nose_length, nose_length + body_length, n_x)
    th = np.linspace(0.0, 2 * math.pi, n_theta)
    X, TH = np.meshgrid(x, th)
    Y = radius * np.cos(TH)
    Z = radius * np.sin(TH)
    return X, Y, Z


# =========================================================
# BICONVEX THICKNESS + FIN PANELS (wing or tail)
# =========================================================
def _biconvex_thickness(x: np.ndarray, root_chord: float, tip_chord: float,
                         root_th: float, tip_th: float) -> np.ndarray:
    """
    Biconvex (parabolic-arc) half-thickness distribution, linearly
    interpolated in chord fraction between root and tip sections.
    `x` is the normalised chordwise coordinate in [0, 1] per station.
    """
    x = np.clip(x, 0.0, 1.0)
    return 2.0 * x * (1.0 - x)   # unit biconvex shape; scaled by caller


def generate_fin_panel(le: float, root_chord: float, tip_chord: float,
                        semi_span: float, hinge_line: float,
                        root_th: float, tip_th: float, sweep: float,
                        body_diameter: float, side: str = "right",
                        n_span: int = 12, n_chord: int = 12
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a flat lifting-surface panel (wing or tail) as an
    (X, Y, Z) mesh, extending outward from the fuselage surface.

    le         : fuselage station (mm) of the panel's root leading edge
    root_chord : root chord length (mm)
    tip_chord  : tip chord length (mm)
    semi_span  : span from body surface to tip (mm)
    hinge_line : chordwise fraction [0,1] used as the sweep reference axis
    root_th/tip_th : max thickness (mm) at root / tip
    sweep      : leading-edge sweep angle (deg)
    body_diameter : fuselage diameter (mm), panel root starts at the surface
    side       : "right" (+Y) or "left" (-Y)
    """
    root_chord = _validate_positive(root_chord, "root_chord")
    tip_chord = _validate_positive(tip_chord, "tip_chord")
    semi_span = _validate_positive(semi_span, "semi_span")
    sign = 1.0 if side == "right" else -1.0
    sweep_rad = _deg2rad(sweep)
    body_r = body_diameter / 2.0

    span_frac = np.linspace(0.0, 1.0, n_span)          # 0 = root, 1 = tip
    chord_len = root_chord + (tip_chord - root_chord) * span_frac
    y_stations = sign * (body_r + span_frac * semi_span)
    le_x = le + span_frac * semi_span * math.tan(sweep_rad)

    chord_frac = np.linspace(0.0, 1.0, n_chord)
    CF, SF = np.meshgrid(chord_frac, span_frac)         # shape (n_span, n_chord)

    X = le_x[:, None] + CF * chord_len[:, None] - hinge_line * chord_len[:, None]
    Y = np.tile(y_stations[:, None], (1, n_chord))

    th_at_station = root_th + (tip_th - root_th) * span_frac
    half_t = th_at_station[:, None] * _biconvex_thickness(CF, root_chord, tip_chord,
                                                           root_th, tip_th) / 2.0
    Z = half_t   # upper surface only; symmetric lower surface omitted for clarity

    return X, Y, Z


# =========================================================
# STATISTICS (fed into any results panel / sidebar)
# =========================================================
def _fuselage_stats(nose_length: float, body_length: float,
                     body_diameter: float) -> Dict[str, float]:
    """Return fuselage length, volume (approx.), and fineness ratio."""
    radius = body_diameter / 2.0
    total_length = nose_length + body_length
    nose_volume = (1.0 / 3.0) * math.pi * radius ** 2 * nose_length  # cone approx.
    body_volume = math.pi * radius ** 2 * body_length
    return {
        "total_length": round(total_length, 2),
        "nose_length": round(nose_length, 2),
        "body_length": round(body_length, 2),
        "body_diameter": round(body_diameter, 2),
        "volume": round(nose_volume + body_volume, 2),
        "fineness_ratio": round(total_length / body_diameter, 3) if body_diameter else 0.0,
    }


def _fin_stats(root_chord: float, tip_chord: float, semi_span: float,
               root_th: float, tip_th: float) -> Dict[str, float]:
    """Return fin planform area, aspect ratio, and taper ratio."""
    area = 0.5 * (root_chord + tip_chord) * semi_span
    aspect_ratio = (2.0 * semi_span) ** 2 / (2.0 * area) if area else 0.0
    taper_ratio = tip_chord / root_chord if root_chord else 0.0
    return {
        "area": round(area, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "taper_ratio": round(taper_ratio, 3),
        "root_chord": round(root_chord, 2),
        "tip_chord": round(tip_chord, 2),
        "semi_span": round(semi_span, 2),
    }


# =========================================================
# GUI PARAMETER ADAPTER
# =========================================================
def params_from_gui(gui_params: Dict[str, float],
                     body_diameter: float = BODY_DIAMETER) -> Dict[str, float]:
    """
    Map the app's 18 PARAM_NAMES onto the geometry generators above.
    `gui_params` is the same dict app.py already builds from PARAMS
    (nose_len, body_len, wing_le, root_chord, tip_chord, semi_span,
    root_th, tip_th, wing_sweep, tail_le, root_chord1, tip_chord1,
    semi_span1, root_th1, tip_th1, mach, alpha, alt).
    """
    return {
        "nose_length": gui_params["nose_len"],
        "body_length": gui_params["body_len"],
        "body_diameter": body_diameter,
        "wing": dict(le=gui_params["wing_le"], root_chord=gui_params["root_chord"],
                     tip_chord=gui_params["tip_chord"], semi_span=gui_params["semi_span"],
                     root_th=gui_params["root_th"], tip_th=gui_params["tip_th"],
                     sweep=gui_params["wing_sweep"]),
        "tail": dict(le=gui_params["tail_le"], root_chord=gui_params["root_chord1"],
                     tip_chord=gui_params["tip_chord1"], semi_span=gui_params["semi_span1"],
                     root_th=gui_params["root_th1"], tip_th=gui_params["tip_th1"],
                     sweep=gui_params["wing_sweep"]),   # no separate tail-sweep param exists
    }


# =========================================================
# FIGURE BUILDER  (the only function app.py needs to call)
# =========================================================
def build_3d_figure(gui_params: Dict[str, float], body_diameter: float = BODY_DIAMETER,
                     title: str = "3-D Aircraft Geometry", figsize=(9, 6)):
    """
    Build a Matplotlib 3-D wireframe Figure of the current geometry,
    styled to match app.py's existing dark theme (caller should have
    already run app.py's _mpl_style() before calling this, exactly
    as it does before building the other optimisation/sweep figures).

    Returns a matplotlib.figure.Figure ready for _embed_fig()/_save_fig().
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

    g = params_from_gui(gui_params, body_diameter)

    nose = generate_nose(g["nose_length"], g["body_diameter"], nose_bluntness=0.15)
    body = generate_body(g["body_length"], g["body_diameter"], g["nose_length"])

    wing_r = generate_fin_panel(**g["wing"], hinge_line=0.25,
                                 body_diameter=g["body_diameter"], side="right")
    wing_l = generate_fin_panel(**g["wing"], hinge_line=0.25,
                                 body_diameter=g["body_diameter"], side="left")
    tail_r = generate_fin_panel(**g["tail"], hinge_line=0.25,
                                 body_diameter=g["body_diameter"], side="right")
    tail_l = generate_fin_panel(**g["tail"], hinge_line=0.25,
                                 body_diameter=g["body_diameter"], side="left")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_wireframe(*nose, color="#06B6D4", linewidth=0.5, rstride=2, cstride=2)
    ax.plot_wireframe(*body, color="#3B82F6", linewidth=0.5, rstride=2, cstride=2)
    for panel, color in ((wing_r, "#10B981"), (wing_l, "#10B981"),
                         (tail_r, "#F59E0B"), (tail_l, "#F59E0B")):
        ax.plot_surface(*panel, color=color, alpha=0.85, linewidth=0)

    total_len = g["nose_length"] + g["body_length"]
    ax.set_xlim(0, total_len)
    ax.set_ylim(-total_len * 0.4, total_len * 0.4)
    ax.set_zlim(-total_len * 0.2, total_len * 0.2)
    ax.set_box_aspect((total_len, total_len * 0.8, total_len * 0.4))
    ax.set_xlabel("X — fuselage station (mm)")
    ax.set_ylabel("Y — span (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title, color="#06B6D4", fontsize=11, fontweight="bold")
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    return fig
