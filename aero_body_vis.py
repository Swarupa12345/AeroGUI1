#!/usr/bin/env python3
"""
Aerodynamic Body 3D Visualization Tool
Generates and plots a 3D aerodynamic body (nose cone, fuselage, wing, and tail fins)
based on geometric parameters.
"""

import os
import argparse
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource, to_rgb

# Try to import plotly for interactive HTML generation
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- Mesh resolution presets ------------------------------------------------
# These only control how many points are sampled when *drawing* a surface --
# they never touch the geometric parameters themselves, so switching between
# them cannot change total_length / wingspan / volume / any other computed
# metric, nor anything fed into the CL/CD/XCP predictors or the optimizer
# (those consume `params` directly, never the mesh). 'preview' is meant for
# live, redrawn-every-frame views (per-generation optimizer playback, alpha
# sweep animation) where many redraws per second matter far more than mesh
# density; 'full' is for the single "final" render of a result.
RES_FULL = dict(body_n_x=50, body_n_theta=50, fin_n_span=30, fin_n_chord=20)
RES_PREVIEW = dict(body_n_x=20, body_n_theta=18, fin_n_span=14, fin_n_chord=10)

# One shared LightSource -- shading direction never changes between calls,
# so there's no reason to allocate a new one on every single facecolor
# computation (was happening ~9-11x per redraw, i.e. every generation).
_SHARED_LIGHT_SOURCE = LightSource(azdeg=315, altdeg=45)


# =========================================================
# FITNESS -> COLOR MAPPING (for live optimizer playback)
# =========================================================
def fitness_to_rgb(t):
    """
    Maps a normalized fitness value t in [0,1] (0=worst seen this run,
    1=best seen this run) to an RGB tuple used for the fitness "glow"
    bezel drawn around the plot: red -> amber -> green, vivid and
    clearly distinguishable at a glance -- a "traffic light" gradient.
    Clamps t into [0,1] so callers don't need to worry about
    out-of-range values.
    """
    t = max(0.0, min(1.0, float(t)))
    # Hue sweeps from red (0.0) to green (1/3) in HSV space; keep
    # saturation/value high so it reads as a clear, vivid tint rather
    # than a washed-out pastel.
    hue = t * (1.0 / 3.0)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (r, g, b)


# Golden-ratio conjugate: stepping a hue by this amount each time gives
# a sequence of colors that stays maximally spread out around the color
# wheel no matter how many steps you take (no two nearby generations
# ever land on similar-looking hues, unlike gen/N which clusters colors
# together for small N and repeats exactly every N).
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def generation_flash_rgb(gen):
    """
    Maps a generation number (1, 2, 3, ...) to a distinct, vivid RGB
    "flash" color -- a different hue for each generation as the DE loop
    plays live, so consecutive generations are visibly distinguishable
    at a glance rather than all looking the same. Uses the golden-ratio
    hue step so the sequence of colors stays well-spread and doesn't
    visibly repeat for a long time. This is intentionally independent
    of fitness_to_rgb()/the fitness glow -- one encodes "which
    generation is this" (this function), the other encodes "how good is
    it" (fitness_to_rgb) -- both are shown together on the same frame.
    """
    gen = max(int(gen), 0)
    hue = (gen * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.80, 1.0)
    return (r, g, b)


def _blend_rgb(rgb, tint_rgb, weight):
    """Linearly blends `rgb` toward `tint_rgb` by `weight` in [0,1]."""
    weight = max(0.0, min(1.0, float(weight)))
    return tuple(c * (1.0 - weight) + t * weight for c, t in zip(rgb, tint_rgb))


def _vividness_scale(rgb, t, dim_floor=0.30):
    """
    Rescales an (r,g,b) color's own saturation/value by t in [0,1]
    without touching its hue -- at t=0 the color reads dull/washed-out,
    at t=1 it's fully vivid. Used to make each aero-body part (nose/
    body/wing/tail) visibly "come alive" as fitness improves while every
    part keeps its own distinct hue throughout (never replaced by a
    shared fitness color -- see fitness_to_rgb(), used only for the glow
    bezel).
    """
    t = max(0.0, min(1.0, float(t)))
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    factor = dim_floor + (1.0 - dim_floor) * t
    s2 = max(0.0, min(1.0, s * factor))
    v2 = max(0.0, min(1.0, v * (0.55 + 0.45 * factor)))
    return colorsys.hsv_to_rgb(h, s2, v2)


def _contrast_color(rgb):
    """
    Returns a hex color ('#0F172A' dark navy or '#FFFFFF' white) chosen
    for readability against a given body tint, via a simple relative
    luminance check. Used for hinge-line overlays so they stay visible
    no matter what the current fitness tint is.
    """
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#0F172A' if lum > 0.6 else '#FFFFFF'


def solve_unblunted_cone(L_n, R, r_b):
    """
    Solves for the unblunted cone length given actual nose length, base radius,
    and tip bluntness radius using a simple binary search.
    """
    if r_b >= R or r_b <= 0:
        return None
    
    # Solve: L_un - r_b * (sqrt(R^2 + L_un^2)/R - 1) - L_n = 0
    low = L_n
    high = L_n + r_b * (R + L_n) / R
    for _ in range(100):
        mid = 0.5 * (low + high)
        sin_theta = R / np.sqrt(R**2 + mid**2)
        f_val = mid - r_b * (1.0 / sin_theta - 1.0) - L_n
        if abs(f_val) < 1e-7:
            return mid
        if f_val < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_nose(L_n, D_n, bluntness, n_x=50, n_theta=50):
    """
    Generates 3D coordinates for the nose section.
    Aligns along the X-axis starting at X=0 to X=L_n.
    """
    # Safeguard bluntness input
    if bluntness > 1.0:
        bluntness = bluntness / 100.0
    bluntness = np.clip(bluntness, 0.0, 1.0)
    
    R = D_n / 2.0
    r_b = bluntness * R
    
    x_coords = np.linspace(0, L_n, n_x)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    theta_grid, x_grid = np.meshgrid(theta, x_coords)
    
    r_profile = np.zeros_like(x_coords)
    
    # Try to solve for spherically blunted cone
    # For spherically blunted cone, the sphere tip radius must be strictly less than the base radius
    L_un = solve_unblunted_cone(L_n, R, r_b) if (0 < r_b < R) else None
    
    if L_un is not None and r_b > 0 and r_b < R:
        # Spherically blunted cone geometry
        theta_cone = np.arctan(R / L_un)
        sin_t = np.sin(theta_cone)
        cos_t = np.cos(theta_cone)
        
        x_c = r_b / sin_t  # Sphere center relative to unblunted tip
        x_start = x_c - r_b
        x_t = x_c - r_b * sin_t  # Tangent point
        
        for i, x_act in enumerate(x_coords):
            x = x_act + x_start
            if x < x_t:
                # Spherical cap (with clipping to prevent negative values in sqrt)
                r_profile[i] = np.sqrt(np.clip(r_b**2 - (x - x_c)**2, 0.0, None))
            else:
                # Conical surface
                r_profile[i] = R * (x / L_un)
    else:
        # Fallback to power-law nose (parabolic profiles for bluntness, cone for sharp)
        # bluntness = 0 -> cone (n=1)
        # bluntness = 1 -> ellipsoid/hemisphere (n=0.5)
        n_power = 1.0 - 0.5 * bluntness
        # Handle case where L_n is zero to avoid division by zero
        L_n_safe = max(L_n, 1e-9)
        r_profile = R * (x_coords / L_n_safe) ** n_power

    # Build 3D mesh
    r_grid = np.tile(r_profile[:, np.newaxis], (1, n_theta))
    X = x_grid
    Y = r_grid * np.cos(theta_grid)
    Z = r_grid * np.sin(theta_grid)
    
    return X, Y, Z


def generate_body(L_b, D_b, start_x, n_x=50, n_theta=50):
    """
    Generates 3D coordinates for the cylindrical body section.
    Aligns along the X-axis from start_x to start_x + L_b.
    """
    R = D_b / 2.0
    x_coords = np.linspace(start_x, start_x + L_b, n_x)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    
    theta_grid, x_grid = np.meshgrid(theta, x_coords)
    r_grid = np.full_like(x_grid, R)
    
    X = x_grid
    Y = r_grid * np.cos(theta_grid)
    Z = r_grid * np.sin(theta_grid)
    
    return X, Y, Z


def generate_fin_panel(le_x, root_chord, tip_chord, semi_span, sweep_deg,
                       root_th, tip_th, phi_deg, hinge_line_frac=None,
                       n_span=30, n_chord=20):
    """
    Generates 3D coordinates for a single wing or fin panel rotated around the X-axis.
    phi_deg: Rotation angle around X-axis (0 = right wing, 90 = top fin, etc.)
    """
    eta = np.linspace(0, 1, n_span)
    s = np.linspace(0, 1, n_chord)
    
    s_grid, eta_grid = np.meshgrid(s, eta)
    
    # Local span coordinate
    y_local = eta_grid * semi_span
    
    # Chord and thickness at each span station
    chord_grid = root_chord + (tip_chord - root_chord) * eta_grid
    thickness_grid = root_th + (tip_th - root_th) * eta_grid
    
    # Leading edge X coordinate at each span station
    le_x_grid = le_x + y_local * np.tan(np.radians(sweep_deg))
    
    # X coordinate of the grid points
    X = le_x_grid + s_grid * chord_grid
    
    # Local Z (thickness) using a biconvex airfoil shape
    # Z_local ranges from -t/2 to +t/2
    Z_local_upper = 0.5 * thickness_grid * 4 * s_grid * (1 - s_grid)
    Z_local_lower = -Z_local_upper
    
    # Rotate around X-axis by phi_deg
    phi_rad = np.radians(phi_deg)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    
    # Upper surface 3D coordinates
    Y_upper = y_local * cos_phi - Z_local_upper * sin_phi
    Z_upper = y_local * sin_phi + Z_local_upper * cos_phi
    
    # Lower surface 3D coordinates
    Y_lower = y_local * cos_phi - Z_local_lower * sin_phi
    Z_lower = y_local * sin_phi + Z_local_lower * cos_phi
    
    # Compute hinge line coordinates if requested
    hinge_line_pts = None
    if hinge_line_frac is not None:
        y_h_local = eta * semi_span
        c_h = root_chord + (tip_chord - root_chord) * eta
        x_le_h = le_x + y_h_local * np.tan(np.radians(sweep_deg))
        
        x_hl = x_le_h + hinge_line_frac * c_h
        t_h = root_th + (tip_th - root_th) * eta
        z_h_upper = 0.5 * t_h * 4 * hinge_line_frac * (1 - hinge_line_frac)
        z_h_lower = -z_h_upper
        
        # Upper hinge line
        y_hl_u = y_h_local * cos_phi - z_h_upper * sin_phi
        z_hl_u = y_h_local * sin_phi + z_h_upper * cos_phi
        
        # Lower hinge line
        y_hl_l = y_h_local * cos_phi - z_h_lower * sin_phi
        z_hl_l = y_h_local * sin_phi + z_h_lower * cos_phi
        
        hinge_line_pts = {
            'upper': (x_hl, y_hl_u, z_hl_u),
            'lower': (x_hl, y_hl_l, z_hl_l)
        }
        
    return X, Y_upper, Z_upper, X, Y_lower, Z_lower, hinge_line_pts


_fig = None
_ax = None


def plot_matplotlib(geometry, hinge_lines, output_file, show=False, block=False):
    """
    Plots the aerodynamic body in 3D using Matplotlib and saves to PNG.
    If the figure is already open, updates it in-place.
    """
    global _fig, _ax
    
    if _fig is not None and plt.fignum_exists(_fig.number):
        _ax.clear()
    else:
        plt.ion()  # Turn on interactive mode
        _fig = plt.figure(figsize=(12, 10))
        _ax = _fig.add_subplot(111, projection='3d')
        
    ax = _ax
    
    # Plot nose
    X, Y, Z = geometry['nose']
    ax.plot_surface(X, Y, Z, color='lightgray', edgecolor='none', alpha=0.9, shade=True)
    
    # Plot body
    X, Y, Z = geometry['body']
    ax.plot_surface(X, Y, Z, color='gray', edgecolor='none', alpha=0.9, shade=True)
    
    # Plot wings and tail fins
    for part_name, surfaces in geometry['wings_tails'].items():
        # Choose colors based on wings vs tails
        color = 'royalblue' if 'wing' in part_name else 'crimson'
        for surface in surfaces:
            X, Y, Z = surface
            ax.plot_surface(X, Y, Z, color=color, edgecolor='none', alpha=0.85, shade=True)
            
    # Plot hinge lines
    for part_name, hl_data in hinge_lines.items():
        if hl_data:
            # Upper hinge line
            xu, yu, zu = hl_data['upper']
            ax.plot(xu, yu, zu, color='yellow', linestyle='--', linewidth=2, label=f'{part_name} Hinge Line' if 'right' in part_name or 'top' in part_name else "")
            # Lower hinge line
            xl, yl, zl = hl_data['lower']
            ax.plot(xl, yl, zl, color='yellow', linestyle='--', linewidth=2)

    # Set labels and aspect ratio
    ax.set_xlabel('X (Length)')
    ax.set_ylabel('Y (Span)')
    ax.set_zlabel('Z (Height)')
    ax.set_title('Aerodynamic Body 3D Visualization')
    
    # Equal aspect ratio trick
    X_all = np.concatenate([geometry['nose'][0].flatten(), geometry['body'][0].flatten()])
    Y_all = np.concatenate([geometry['nose'][1].flatten(), geometry['body'][1].flatten()])
    Z_all = np.concatenate([geometry['nose'][2].flatten(), geometry['body'][2].flatten()])
    
    max_range = np.array([X_all.max()-X_all.min(), Y_all.max()-Y_all.min(), Z_all.max()-Z_all.min()]).max() / 2.0
    mid_x = (X_all.max()+X_all.min()) * 0.5
    mid_y = (Y_all.max()+Y_all.min()) * 0.5
    mid_z = (Z_all.max()+Z_all.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Dark theme or clean grid
    ax.grid(True)
    ax.legend()
    
    # Save file
    _fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Matplotlib static plot saved to: {output_file}")
    
    if show:
        _fig.canvas.draw_idle()
        plt.pause(0.01)
        if block:
            plt.ioff()
            plt.show(block=True)


def plot_plotly(geometry, hinge_lines, params, output_file):
    """
    Plots the aerodynamic body in 3D using Plotly and saves to an interactive HTML dashboard.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is not available. Skipping interactive HTML plot.")
        return
        
    fig = go.Figure()
    
    # Nose Surface
    X, Y, Z = geometry['nose']
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, 'rgb(220,220,220)'], [1, 'rgb(180,180,180)']],
        showscale=False,
        name='Nose Cone',
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.2, specular=0.5)
    ))
    
    # Body Surface
    X, Y, Z = geometry['body']
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, 'rgb(120,120,120)'], [1, 'rgb(100,100,100)']],
        showscale=False,
        name='Fuselage',
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.2, specular=0.5)
    ))
    
    # Wings and Fins
    for part_name, surfaces in geometry['wings_tails'].items():
        is_wing = 'wing' in part_name
        color = 'royalblue' if is_wing else 'crimson'
        
        # Plotly surfaces
        for idx, surface in enumerate(surfaces):
            X, Y, Z = surface
            surf_name = f"{part_name.capitalize()} {'Upper' if idx==0 else 'Lower'}"
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=surf_name,
                lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.3, specular=0.5)
            ))
            
    # Hinge lines
    for part_name, hl_data in hinge_lines.items():
        if hl_data:
            xu, yu, zu = hl_data['upper']
            xl, yl, zl = hl_data['lower']
            
            fig.add_trace(go.Scatter3d(
                x=xu, y=yu, z=zu,
                mode='lines',
                line=dict(color='yellow', width=5, dash='dash'),
                name=f'{part_name.capitalize()} Hinge (Upper)'
            ))
            fig.add_trace(go.Scatter3d(
                x=xl, y=yl, z=zl,
                mode='lines',
                line=dict(color='yellow', width=5, dash='dash'),
                name=f'{part_name.capitalize()} Hinge (Lower)',
                showlegend=False
            ))

    # Layout styling (Dark mode, premium feel)
    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis=dict(title='X (Length)', backgroundcolor='rgb(20,20,20)', gridcolor='gray', showbackground=True),
            yaxis=dict(title='Y (Span)', backgroundcolor='rgb(20,20,20)', gridcolor='gray', showbackground=True),
            zaxis=dict(title='Z (Height)', backgroundcolor='rgb(20,20,20)', gridcolor='gray', showbackground=True),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    
    # Calculate geometric quantities
    L_total = params["nose_length"] + params["body_length"]
    R_b = params["body_diameter"] / 2.0
    V_body = np.pi * (R_b**2) * params["body_length"]
    
    # Blunted cone or power-law volume approximation
    V_nose = np.pi * (R_b**2) * params["nose_length"] * (1.0/3.0 + 0.167 * params["nose_bluntness"])
    V_total = V_body + V_nose
    
    # Wing calculations
    S_wing_one = params["semi_span"] * (params["root_chord"] + params["tip_chord"]) / 2.0
    S_wing = 2.0 * S_wing_one
    AR_wing = (2.0 * params["semi_span"])**2 / S_wing if S_wing > 0 else 0.0
    taper_wing = params["tip_chord"] / params["root_chord"] if params["root_chord"] > 0 else 0.0
    
    # Tail calculations
    num_fins = 4 if params["tail_config"] == 'cruciform' else 3
    S_tail_one = params["tail_semi_span"] * (params["tail_root_chord"] + params["tail_tip_chord"]) / 2.0
    S_tail_total = num_fins * S_tail_one
    AR_tail = (2.0 * params["tail_semi_span"])**2 / (2.0 * S_tail_one) if S_tail_one > 0 else 0.0
    taper_tail = params["tail_tip_chord"] / params["tail_root_chord"] if params["tail_root_chord"] > 0 else 0.0

    # Convert Plotly figure to JSON
    import json
    import plotly.utils
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aerodynamic Design Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: #0b0f19;
            color: #f3f4f6;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        #sidebar {{
            width: 380px;
            background-color: #111827;
            border-right: 1px solid #1f2937;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow-y: auto;
            box-sizing: border-box;
        }}
        #viewer {{
            flex-grow: 1;
            height: 100%;
        }}
        h1 {{
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: #3b82f6;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 2px solid #1f2937;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .card {{
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .card-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #60a5fa;
            text-transform: uppercase;
            border-bottom: 1px solid #374151;
            padding-bottom: 5px;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}
        .stat-label {{
            color: #9ca3af;
        }}
        .stat-value {{
            font-weight: 600;
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Aerodynamic Details</h1>
        
        <div class="card">
            <div class="card-title">Fuselage Specs</div>
            <div class="stat-row">
                <span class="stat-label">Total Length</span>
                <span class="stat-value">{L_total:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Body Length</span>
                <span class="stat-value">{params['body_length']:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Nose Length</span>
                <span class="stat-value">{params['nose_length']:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Body Diameter</span>
                <span class="stat-value">{params['body_diameter']:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Nose Bluntness</span>
                <span class="stat-value">{params['nose_bluntness']*100:.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Fuselage Volume</span>
                <span class="stat-value">{V_body:.2f} units³</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Nose Volume</span>
                <span class="stat-value">{V_nose:.2f} units³</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Volume</span>
                <span class="stat-value">{V_total:.2f} units³</span>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Wing Geometry</div>
            <div class="stat-row">
                <span class="stat-label">Total Area (both)</span>
                <span class="stat-value">{S_wing:.2f} units²</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Semi-span</span>
                <span class="stat-value">{params['semi_span']:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Aspect Ratio</span>
                <span class="stat-value">{AR_wing:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Taper Ratio</span>
                <span class="stat-value">{taper_wing:.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Sweep Angle</span>
                <span class="stat-value">{params['wing_sweep']:.2f}°</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Root Chord</span>
                <span class="stat-value">{params['root_chord']:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Tip Chord</span>
                <span class="stat-value">{params['tip_chord']:.2f} units</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Root / Tip Thickness</span>
                <span class="stat-value">{params['root_th']:.2f} / {params['tip_th']:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Hinge Line Location</span>
                <span class="stat-value">{params['w_hinge_line']*100:.1f}% chord</span>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Tail Geometry</div>
            <div class="stat-row">
                <span class="stat-label">Configuration</span>
                <span class="stat-value">{params['tail_config'].capitalize()} ({num_fins} fins)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Area</span>
                <span class="stat-value">{S_tail_total:.2f} units²</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Aspect Ratio</span>
                <span class="stat-value">{AR_tail:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Taper Ratio</span>
                <span class="stat-value">{taper_tail:.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Sweep Angle</span>
                <span class="stat-value">{params['tail_sweep']:.2f}°</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Hinge Line Location</span>
                <span class="stat-value">{params['f_hinge_line']*100:.1f}% chord</span>
            </div>
        </div>
    </div>
    <div id="viewer"></div>

    <script>
        var figData = {fig_json};
        Plotly.newPlot('viewer', figData.data, figData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Plotly interactive HTML dashboard saved to: {output_file}")


def prompt_parameter(name, default_val):
    """
    Prompts the user for a parameter in the console.
    If the user presses Enter, returns the default value.
    """
    try:
        user_input = input(f"Enter {name} [{default_val}]: ").strip()
        if not user_input:
            return default_val
        if isinstance(default_val, float):
            return float(user_input)
        elif isinstance(default_val, int):
            return int(user_input)
        return user_input
    except (KeyboardInterrupt, EOFError):
        print(f"\nUsing default: {default_val}")
        return default_val
    except ValueError:
        print(f"Invalid input, using default: {default_val}")
        return default_val


def compute_geometry_metrics(params):
    """
    Computes summary aerodynamic-geometry metrics (lengths, areas,
    aspect ratios, taper ratios, volume) from a full vis-parameter dict.
    Used to drive numeric readouts alongside the 3D view.
    """
    L_total = params["nose_length"] + params["body_length"]
    R_b = params["body_diameter"] / 2.0
    V_body = np.pi * (R_b ** 2) * params["body_length"]
    V_nose = np.pi * (R_b ** 2) * params["nose_length"] * (1.0 / 3.0 + 0.167 * params["nose_bluntness"])
    V_total = V_body + V_nose

    S_wing_one = params["semi_span"] * (params["root_chord"] + params["tip_chord"]) / 2.0
    S_wing = 2.0 * S_wing_one
    AR_wing = (2.0 * params["semi_span"]) ** 2 / S_wing if S_wing > 0 else 0.0
    taper_wing = params["tip_chord"] / params["root_chord"] if params["root_chord"] > 0 else 0.0

    num_fins = 4 if params["tail_config"] == 'cruciform' else 3
    S_tail_one = params["tail_semi_span"] * (params["tail_root_chord"] + params["tail_tip_chord"]) / 2.0
    S_tail_total = num_fins * S_tail_one
    AR_tail = (2.0 * params["tail_semi_span"]) ** 2 / (2.0 * S_tail_one) if S_tail_one > 0 else 0.0
    taper_tail = params["tail_tip_chord"] / params["tail_root_chord"] if params["tail_root_chord"] > 0 else 0.0

    return {
        'total_length': L_total,
        'body_diameter': params["body_diameter"],
        'fineness_ratio': L_total / params["body_diameter"] if params["body_diameter"] > 0 else 0.0,
        'volume': V_total,
        'wingspan': 2.0 * params["semi_span"],
        'wing_area': S_wing,
        'wing_aspect_ratio': AR_wing,
        'wing_taper': taper_wing,
        'tail_span': 2.0 * params["tail_semi_span"],
        'tail_area': S_tail_total,
        'tail_aspect_ratio': AR_tail,
        'tail_taper': taper_tail,
        'num_tail_fins': num_fins,
    }


def _pitch_rotate_xz(X, Y, Z, angle_deg, pivot_x, pivot_z=0.0):
    """
    Rotates a mesh (X, Z arrays) about the Y-axis (lateral/span axis) by
    `angle_deg` around the pivot point (pivot_x, pivot_z) in the X-Z
    (length-height) plane. Y (span) is left untouched.

    This is used to visually represent angle of attack (alpha): a
    positive angle pitches the nose up (rotates the body's X axis
    toward +Z) without changing its shape, matching the standard
    convention that angle of attack is the angle between the body's
    longitudinal axis and the oncoming free-stream flow.
    """
    if angle_deg == 0.0:
        return X, Y, Z
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xc = X - pivot_x
    Zc = Z - pivot_z
    X_rot = Xc * cos_t + Zc * sin_t + pivot_x
    Z_rot = -Xc * sin_t + Zc * cos_t + pivot_z
    return X_rot, Y, Z_rot


def _pitch_rotate_hinge(hl_data, angle_deg, pivot_x, pivot_z=0.0):
    """Applies _pitch_rotate_xz to a hinge-line dict's 'upper'/'lower' tuples."""
    if not hl_data or angle_deg == 0.0:
        return hl_data
    out = {}
    for key in ('upper', 'lower'):
        if key in hl_data:
            x, y, z = hl_data[key]
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            z = np.asarray(z, dtype=float)
            xr, yr, zr = _pitch_rotate_xz(x, y, z, angle_deg, pivot_x, pivot_z)
            out[key] = (xr, yr, zr)
        else:
            out[key] = hl_data[key]
    return out


def apply_pitch_to_geometry(geometry, hinge_lines, angle_deg, pivot_x=None):
    """
    Returns new (geometry, hinge_lines) dicts with every surface/line
    pitched by `angle_deg` about the Y-axis, pivoting around the body's
    overall mid-length point (or an explicit `pivot_x`). Does not modify
    the inputs in place.
    """
    if angle_deg == 0.0:
        return geometry, hinge_lines

    if pivot_x is None:
        X_nose, _, _ = geometry['nose']
        X_body, _, _ = geometry['body']
        x_all = np.concatenate([np.asarray(X_nose).flatten(),
                                np.asarray(X_body).flatten()])
        pivot_x = float((x_all.max() + x_all.min()) * 0.5)

    new_geometry = {'wings_tails': {}}
    for part in ('nose', 'body'):
        X, Y, Z = geometry[part]
        new_geometry[part] = _pitch_rotate_xz(
            np.asarray(X, dtype=float), np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float), angle_deg, pivot_x)

    for part_name, surfaces in geometry['wings_tails'].items():
        new_surfaces = []
        for (X, Y, Z) in surfaces:
            new_surfaces.append(_pitch_rotate_xz(
                np.asarray(X, dtype=float), np.asarray(Y, dtype=float),
                np.asarray(Z, dtype=float), angle_deg, pivot_x))
        new_geometry['wings_tails'][part_name] = new_surfaces

    new_hinge_lines = {
        name: _pitch_rotate_hinge(hl, angle_deg, pivot_x)
        for name, hl in hinge_lines.items()
    }

    return new_geometry, new_hinge_lines


def build_geometry(params, resolution=None):
    """
    Generates the full 3D geometry mesh dict (nose, body, wings, tails,
    hinge lines) from a vis-parameter dict, without any plotting.
    Shared by the CLI visualizer and any embedded-canvas renderer.

    `resolution` (optional) is one of RES_FULL / RES_PREVIEW -- it only
    changes how many points are sampled per surface (mesh density for
    drawing). It never changes any of the parameter values themselves,
    so it cannot affect compute_geometry_metrics() or anything the
    predictor/optimizer consume. Defaults to RES_FULL.
    """
    res = resolution or RES_FULL
    params = dict(params)
    for k in ["nose_length", "nose_diameter", "body_length", "body_diameter",
              "wing_le", "root_chord", "tip_chord", "semi_span", "root_th", "tip_th",
              "tail_le", "tail_root_chord", "tail_tip_chord", "tail_semi_span",
              "tail_root_th", "tail_tip_th"]:
        params[k] = abs(params[k])
    params["wing_sweep"] = abs(params["wing_sweep"])
    params["tail_sweep"] = abs(params["tail_sweep"])
    for k in ["w_hinge_line", "f_hinge_line", "nose_bluntness"]:
        if params[k] > 1.0:
            params[k] = params[k] / 100.0
        params[k] = np.clip(params[k], 0.0, 1.0)

    X_nose, Y_nose, Z_nose = generate_nose(
        params["nose_length"], params["nose_diameter"], params["nose_bluntness"],
        n_x=res['body_n_x'], n_theta=res['body_n_theta'])
    X_body, Y_body, Z_body = generate_body(
        params["body_length"], params["body_diameter"], params["nose_length"],
        n_x=res['body_n_x'], n_theta=res['body_n_theta'])

    geometry = {
        'nose': (X_nose, Y_nose, Z_nose),
        'body': (X_body, Y_body, Z_body),
        'wings_tails': {}
    }
    hinge_lines = {}

    w_r_Xu, w_r_Yu, w_r_Zu, w_r_Xl, w_r_Yl, w_r_Zl, w_r_hl = generate_fin_panel(
        params["wing_le"], params["root_chord"], params["tip_chord"], params["semi_span"], params["wing_sweep"],
        params["root_th"], params["tip_th"], 0.0, params["w_hinge_line"],
        n_span=res['fin_n_span'], n_chord=res['fin_n_chord']
    )
    geometry['wings_tails']['wing_right'] = [(w_r_Xu, w_r_Yu, w_r_Zu), (w_r_Xl, w_r_Yl, w_r_Zl)]
    hinge_lines['wing_right'] = w_r_hl

    w_l_Xu, w_l_Yu, w_l_Zu, w_l_Xl, w_l_Yl, w_l_Zl, w_l_hl = generate_fin_panel(
        params["wing_le"], params["root_chord"], params["tip_chord"], params["semi_span"], params["wing_sweep"],
        params["root_th"], params["tip_th"], 180.0, params["w_hinge_line"]
    )
    geometry['wings_tails']['wing_left'] = [(w_l_Xu, w_l_Yu, w_l_Zu), (w_l_Xl, w_l_Yl, w_l_Zl)]
    hinge_lines['wing_left'] = w_l_hl

    if params["tail_config"] == 'cruciform':
        phi_angles = [0.0, 90.0, 180.0, 270.0]
        names = ['tail_right', 'tail_top', 'tail_left', 'tail_bottom']
    else:
        phi_angles = [0.0, 90.0, 180.0]
        names = ['tail_right', 'tail_top', 'tail_left']

    for name, phi in zip(names, phi_angles):
        Xu, Yu, Zu, Xl, Yl, Zl, hl = generate_fin_panel(
            params["tail_le"], params["tail_root_chord"], params["tail_tip_chord"], params["tail_semi_span"],
            params["tail_sweep"], params["tail_root_th"], params["tail_tip_th"], phi, params["f_hinge_line"],
            n_span=res['fin_n_span'], n_chord=res['fin_n_chord']
        )
        geometry['wings_tails'][name] = [(Xu, Yu, Zu), (Xl, Yl, Zl)]
        hinge_lines[name] = hl

    return geometry, hinge_lines, params


def _shaded_facecolors(X, Y, Z, base_rgb, ls_azdeg=315, ls_altdeg=45,
                       lo=0.55, hi=1.0):
    """
    Computes smooth per-facet facecolors using a light source, so the
    surface reads as a solid, gently shaded CAD part instead of a flat
    single-tone (which under Matplotlib's default 3D shading can look
    faceted / hollow, especially at grazing viewing angles).
    """
    ls = (_SHARED_LIGHT_SOURCE if (ls_azdeg, ls_altdeg) == (315, 45)
          else LightSource(azdeg=ls_azdeg, altdeg=ls_altdeg))
    try:
        rgb = ls.shade(Z, plt.cm.Greys, vert_exag=0.0, blend_mode='soft',
                       fraction=1.0)
        # shade() returns an RGBA image keyed off elevation; we only want
        # the intensity channel to modulate our own solid base color.
        intensity = rgb[..., 0]
        intensity = lo + (hi - lo) * (intensity - intensity.min()) / \
            max(intensity.max() - intensity.min(), 1e-9)
    except Exception:
        intensity = np.ones_like(Z)

    r, g, b = base_rgb
    facecolors = np.empty(Z.shape + (4,))
    facecolors[..., 0] = np.clip(r * intensity, 0, 1)
    facecolors[..., 1] = np.clip(g * intensity, 0, 1)
    facecolors[..., 2] = np.clip(b * intensity, 0, 1)
    facecolors[..., 3] = 1.0
    return facecolors


def _add_end_cap(ax, X, Y, Z, color, row=-1):
    """
    Closes off the open end of a tube-like surface (e.g. the aft end of
    the fuselage) with a solid filled disc, so the body reads as a solid
    closed shape instead of a hollow/see-through tube when viewed from
    behind or at an angle.
    """
    x_end = X[row, 0]
    ys = Y[row, :]
    zs = Z[row, :]
    y0, z0 = float(np.mean(ys)), float(np.mean(zs))
    n = len(ys)
    verts = []
    for i in range(n - 1):
        verts.append([
            (x_end, y0, z0),
            (x_end, ys[i], zs[i]),
            (x_end, ys[i + 1], zs[i + 1]),
        ])
    cap = Poly3DCollection(verts, facecolors=color, edgecolors='none',
                           alpha=1.0, shade=False)
    ax.add_collection3d(cap)


def _generation_marker_xyz(full_params, frac, pitch_deg=0.0):
    """
    Computes a 3D point sitting just above the top of the fuselage
    (nose+body) at length-fraction `frac` in [0, 1] -- 0 = nose tip,
    1 = tail end. Used to place a small marker on the body itself that
    visibly moves along it as generations progress, rather than a fixed
    on-screen HUD label.

    Radius at that X is only an approximation (linear taper through the
    nose, full body radius elsewhere) -- it exists purely to place the
    marker visibly just outside the surface, not as an aerodynamic or
    geometric quantity, so it never affects compute_geometry_metrics()
    or anything else.
    """
    frac = max(0.0, min(1.0, float(frac)))
    nose_len = float(full_params.get('nose_length', 0.0))
    body_len = float(full_params.get('body_length', 0.0))
    body_r = float(full_params.get('body_diameter', 0.0)) / 2.0
    nose_r = float(full_params.get('nose_diameter', 0.0)) / 2.0
    total_len = max(nose_len + body_len, 1e-6)

    x = frac * total_len
    if nose_len > 1e-6 and x <= nose_len:
        r = nose_r * (x / nose_len)
    else:
        r = body_r
    z = r * 1.18 + max(body_r, nose_r) * 0.05  # small standoff above the surface
    y = 0.0

    if pitch_deg:
        xa = np.array([x]); ya = np.array([y]); za = np.array([z])
        pivot_x = total_len * 0.5
        xa, ya, za = _pitch_rotate_xz(xa, ya, za, pitch_deg, pivot_x)
        x, y, z = float(xa[0]), float(ya[0]), float(za[0])

    return x, y, z


def render_geometry_on_figure(fig, params, title=None, large=False, bg_color=None,
                              render_mode='solid', pitch_deg=0.0, fast_preview=False,
                              fitness_t=None, overlay_text=None, gen_marker=None,
                              azim_offset=0.0, elev_offset=0.0, show_overlay=True,
                              color_by_progress=True):
    """
    Draws the aerodynamic body geometry described by `params` (a full
    vis-parameter dict) onto an existing Matplotlib `Figure` (e.g. one
    embedded in a Tk canvas via FigureCanvasTkAgg), clearing it first.
    Does NOT call plt.show()/plt.ion() and does NOT touch the module-level
    _fig/_ax used by plot_matplotlib(), so it is safe to use inside a GUI
    event loop alongside the CLI visualizer.

    `large=True` scales up fonts/linewidths for a big popup/zoom view.
    `bg_color` optionally matches the embedding GUI's panel background.
    `render_mode` is either 'solid' (fully covered, CAD-shaded surfaces --
    the default) or 'wireframe' (open mesh lines only, no fill), letting
    the caller flip between a "wired" skeletal view and a fully covered
    view of the same aero body.
    `pitch_deg` visually pitches the whole rigid body (nose/body/wings/
    tails/hinge-lines together, shape unchanged) about its mid-length
    point on the Y-axis -- used to represent angle of attack (alpha)
    live during a Flight Envelope alpha sweep. Leave at 0.0 for the
    normal, unrotated view.
    `fast_preview=True` draws the exact same geometry at a lower mesh
    resolution (RES_PREVIEW instead of RES_FULL) and reuses the existing
    3D axes in place (ax.cla()) instead of tearing down and rebuilding
    the whole Figure -- this is what makes redrawing on every DE
    generation / every alpha-sweep step fast enough to feel live. It
    never changes any parameter value, so total_length/wingspan/volume/
    etc. (and everything the optimizer or predictor sees) are identical
    either way -- only how densely the surface is *drawn*.
    `fitness_t`, if given, is a float in [0,1] (0=worst seen this run,
    1=best seen this run). The body keeps its normal professional CAD
    palette -- nose/body/wing/tail each their own distinct hue, never
    flattened to one shared color -- but `fitness_t` drives two things:
    (1) each part's own hue is dimmed toward gray at t=0 and fully vivid
    at t=1 (see _vividness_scale()), so quality is visible without ever
    losing which part is which; (2) a bold "glow" bezel is drawn around
    the whole plot in the red->green fitness_to_rgb() color, giving an
    at-a-glance quality readout that doesn't touch the body's own colors
    at all. Only applies in 'solid' render_mode. Leave as None for the
    standard, always-fully-vivid, unglowed view.

    `gen_marker['gen']`, when present, ALSO drives a distinct "which
    generation is this" flash color (see generation_flash_rgb()) that's
    independent of fitness_t: a light tint of that generation's color is
    blended into every part before the fitness vividness scaling is
    applied, and the moving on-body marker itself is drawn in that same
    color. Consecutive generations get visibly different colors (the hue
    steps by the golden ratio each generation) so as the optimizer plays
    generation-by-generation you see the body's tint change frame to
    frame, on top of the red->green fitness glow -- one color tells you
    "which generation", the other tells you "how good it is". This is
    for the CURRENT frame only (no trail/history of past generations is
    kept on-screen).
    `overlay_text`, if given and `show_overlay=True`, is drawn as a
    HUD-style label baked directly into the bottom-left corner of the 3D
    plot itself -- e.g. "Gen 07 / 50\nFitness: 0.842311". Pass a
    multi-line string; each line is shown on its own row.
    `show_overlay=False` computes/returns everything as normal but skips
    drawing `overlay_text` onto the plot -- used by live GUI views where
    the same text is shown in a widget beside the canvas instead of baked
    into the rendered image (e.g. so a "Save" button captures a clean
    view). Frame-export call sites (PNG/video generation snapshots) keep
    the default `show_overlay=True` so the exported image is
    self-contained. Leave `overlay_text` as None
    to draw no overlay.

    `color_by_progress=False` skips BOTH the gen_marker "flash" tint
    blend and the fitness_t vividness scaling on the body's own
    surfaces (nose/body/wing/tail keep their normal, fixed, full-vividness
    CAD colors no matter what fitness_t/gen_marker are set to). The
    on-body generation dot + label (see gen_marker below) and the
    glow bezel are unaffected by this flag -- it only controls whether
    the airframe surfaces themselves are recolored. Used by the live 3D
    popup so the body doesn't change color while a run plays out;
    progress there is shown via a separate step counter/progress bar
    instead. Leave True for the normal fitness-tinted behavior.

    `gen_marker`, if given, is a dict {'gen': int, 'total': int,
    'fitness': float} and draws a small yellow dot *directly on the aero
    body itself* (a real 3D point on the fuselage spine, not a flat 2D
    HUD label) labeled "G<gen>\nFit=<fitness>". Its position moves along
    the body from nose (generation 1) to tail (the final generation) as
    `gen`/`total` increases, so the current generation is visibly
    "where" it is on the model as the optimizer runs, in addition to the
    text. Leave as None to draw no marker.

    `azim_offset` / `elev_offset` are added on top of the default 3/4
    CAD viewing angle (elev=22, azim=-58). Used to drive a mouse-free
    "auto-rotate" turntable: the caller advances `azim_offset` by a
    small amount on a timer and re-renders, which spins the body
    continuously without any click-drag needed. Leave both at 0.0 for
    the normal fixed viewing angle.

    Returns the computed geometry metrics dict (see compute_geometry_metrics).
    """
    geometry, hinge_lines, full_params = build_geometry(
        params, resolution=RES_PREVIEW if fast_preview else RES_FULL)

    if pitch_deg:
        geometry, hinge_lines = apply_pitch_to_geometry(geometry, hinge_lines, pitch_deg)

    wireframe = (render_mode == 'wireframe')

    fs_label = 13 if large else 10
    fs_title = 17 if large else 12
    fs_tick = 10 if large else 8
    lw_hinge = 2.2 if large else 1.4
    lw_wire = (1.1 if large else 0.7)
    wire_stride = 2 if large else 3

    face_bg = bg_color or '#EEF1F5'

    # Reuse the same Axes3D across redraws when possible (fast_preview
    # live-update path): ax.cla() clears the drawn artists but keeps the
    # Axes3D object itself, which is considerably cheaper per frame than
    # fig.clf() + add_subplot() rebuilding the whole 3D projection from
    # scratch. Falls back to a fresh axes whenever the figure doesn't
    # already have one set up this way (first render, popup views, or a
    # figure shared with some other kind of plot).
    ax = getattr(fig, '_geo_ax3d', None)
    if fast_preview and ax is not None and ax in fig.axes:
        ax.cla()
    else:
        fig.clf()
        fig.patch.set_facecolor(face_bg)
        ax = fig.add_subplot(111, projection='3d')
        fig._geo_ax3d = ax
    fig.patch.set_facecolor(face_bg)
    ax.set_facecolor(face_bg)

    # --- Clean CAD-style panes: soft neutral fill, subtle edges ---
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
        pane.set_edgecolor((0.75, 0.78, 0.82, 1.0))
        pane.set_linewidth(0.6)
    ax.xaxis._axinfo["grid"]['color'] = (0.82, 0.85, 0.88, 0.6)
    ax.yaxis._axinfo["grid"]['color'] = (0.82, 0.85, 0.88, 0.6)
    ax.zaxis._axinfo["grid"]['color'] = (0.82, 0.85, 0.88, 0.6)

    if wireframe:
        # --- Wired / skeletal view: open mesh lines only, no fill ---
        wire_color = '#3B4652'
        X, Y, Z = geometry['nose']
        ax.plot_wireframe(X, Y, Z, color=wire_color, linewidth=lw_wire,
                          rstride=wire_stride, cstride=wire_stride)

        X, Y, Z = geometry['body']
        ax.plot_wireframe(X, Y, Z, color=wire_color, linewidth=lw_wire,
                          rstride=wire_stride, cstride=wire_stride)

        for part_name, surfaces in geometry['wings_tails'].items():
            is_wing = 'wing' in part_name
            color = '#2563EB' if is_wing else '#DC2626'
            for surface in surfaces:
                X, Y, Z = surface
                ax.plot_wireframe(X, Y, Z, color=color, linewidth=lw_wire,
                                  rstride=wire_stride, cstride=wire_stride)
    else:
        # --- Body surfaces: solid, smoothly-lit CAD-style shading ---
        # Every part always keeps its own fixed hue (nose/body/wing/tail
        # are never flattened to one shared color). fitness_t only scales
        # each part's OWN vividness (dull/grayish at t=0 -> fully vivid at
        # t=1 -- see _vividness_scale()) so quality is visible generation
        # to generation without ever losing which part is which.
        nose_rgb = to_rgb('#D5D9DE')
        body_rgb = to_rgb('#9AA3AD')
        wing_rgb = to_rgb('#2563EB')
        tail_rgb = to_rgb('#DC2626')

        # Per-generation "flash" tint: a light blend of this generation's
        # distinct color into every part, BEFORE fitness vividness scaling
        # -- so each generation frame visibly recolors the body a touch
        # differently as the optimizer plays, current-frame-only (no
        # trail). Kept subtle (18% weight) so nose/body/wing/tail stay
        # instantly identifiable by their base hue at every generation.
        # Skipped entirely when color_by_progress=False -- the body then
        # always keeps its normal, fixed CAD palette regardless of
        # fitness_t/gen_marker (used by the live 3D popup).
        if gen_marker and color_by_progress:
            flash_rgb = generation_flash_rgb(gen_marker.get('gen', 0))
            nose_rgb = _blend_rgb(nose_rgb, flash_rgb, 0.18)
            body_rgb = _blend_rgb(body_rgb, flash_rgb, 0.18)
            wing_rgb = _blend_rgb(wing_rgb, flash_rgb, 0.18)
            tail_rgb = _blend_rgb(tail_rgb, flash_rgb, 0.18)

        if fitness_t is not None and color_by_progress:
            nose_rgb = _vividness_scale(nose_rgb, fitness_t)
            body_rgb = _vividness_scale(body_rgb, fitness_t)
            wing_rgb = _vividness_scale(wing_rgb, fitness_t)
            tail_rgb = _vividness_scale(tail_rgb, fitness_t)

        X, Y, Z = geometry['nose']
        ax.plot_surface(X, Y, Z, facecolors=_shaded_facecolors(X, Y, Z, nose_rgb, lo=0.70, hi=1.0),
                        edgecolor='none', alpha=1.0, shade=False, antialiased=True,
                        rstride=1, cstride=1)

        X, Y, Z = geometry['body']
        ax.plot_surface(X, Y, Z, facecolors=_shaded_facecolors(X, Y, Z, body_rgb, lo=0.45, hi=0.9),
                        edgecolor='none', alpha=1.0, shade=False, antialiased=True,
                        rstride=1, cstride=1)
        # Close the open aft end of the fuselage so it reads as a solid body
        # rather than a hollow tube when viewed from the rear / at an angle.
        end_cap_rgb = tuple(c * 0.75 for c in body_rgb)
        _add_end_cap(ax, X, Y, Z, end_cap_rgb, row=-1)

        for part_name, surfaces in geometry['wings_tails'].items():
            is_wing = 'wing' in part_name
            part_rgb = wing_rgb if is_wing else tail_rgb
            for surface in surfaces:
                X, Y, Z = surface
                ax.plot_surface(X, Y, Z, facecolors=_shaded_facecolors(X, Y, Z, part_rgb, lo=0.6, hi=1.0),
                                edgecolor='none', alpha=1.0, shade=False, antialiased=True,
                                rstride=1, cstride=1)

    hinge_color = '#F59E0B'
    for part_name, hl_data in hinge_lines.items():
        if hl_data:
            xu, yu, zu = hl_data['upper']
            ax.plot(xu, yu, zu, color=hinge_color, linestyle='--', linewidth=lw_hinge)
            xl, yl, zl = hl_data['lower']
            ax.plot(xl, yl, zl, color=hinge_color, linestyle='--', linewidth=lw_hinge)

    ax.set_xlabel('X — Length', fontsize=fs_label, fontweight='bold', color='#334155', labelpad=10)
    ax.set_ylabel('Y — Span', fontsize=fs_label, fontweight='bold', color='#334155', labelpad=10)
    ax.set_zlabel('Z — Height', fontsize=fs_label, fontweight='bold', color='#334155', labelpad=8)
    ax.set_title(title or 'Aerodynamic Body 3D View',
                fontsize=fs_title, fontweight='bold', color='#0F172A', pad=14)
    ax.tick_params(labelsize=fs_tick, colors='#475569')

    # Pleasant 3/4 CAD-style viewing angle, nudged by azim_offset/elev_offset
    # for mouse-free auto-rotate turntable playback (see docstring above).
    ax.view_init(elev=22 + elev_offset, azim=-58 + azim_offset)

    X_all = np.concatenate([geometry['nose'][0].flatten(), geometry['body'][0].flatten()])
    Y_all = np.concatenate([geometry['nose'][1].flatten(), geometry['body'][1].flatten()])
    Z_all = np.concatenate([geometry['nose'][2].flatten(), geometry['body'][2].flatten()])

    max_range = np.array([X_all.max() - X_all.min(),
                          Y_all.max() - Y_all.min(),
                          Z_all.max() - Z_all.min()]).max() / 2.0
    max_range = max(max_range, 1e-6)
    mid_x = (X_all.max() + X_all.min()) * 0.5
    mid_y = (Y_all.max() + Y_all.min()) * 0.5
    mid_z = (Z_all.max() + Z_all.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.grid(True, alpha=0.4)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # tight_layout() re-solves the whole figure's margins from scratch and
    # is one of the pricier calls here; the panel's figsize/labels don't
    # change frame-to-frame during a live preview, so it only needs to run
    # once (on the first/non-preview render) rather than on every generation.
    # Must happen BEFORE the glow bezel below, since the bezel is sized
    # from the axes' final on-figure position.
    if not fast_preview:
        fig.tight_layout()

    # Fitness "glow" bezel: a few concentric rectangles framing the 3D
    # plot with decreasing alpha, faking a soft glow without an expensive
    # per-pixel blur. Red->amber->green (see fitness_to_rgb()) -- reads
    # instantly as "how good is this one" without touching the body's
    # own nose/body/wing/tail colors at all.
    #
    # These are added as FIGURE-level artists (fig.add_artist, in figure-
    # fraction coordinates derived from the axes' own on-screen bounding
    # box), not ax.add_patch() -- a plain 2D Rectangle added directly to a
    # 3D (Axes3D) axes has no do_3d_projection() method, which crashes
    # matplotlib the moment the figure is actually drawn. Living on the
    # Figure instead sidesteps that entirely and still visually frames
    # the 3D plot exactly the same way.
    # NOTE: the fitness "glow" bezel (concentric colored rectangles framed
    # around the plot) used to be drawn here. It has been removed by
    # request -- it read as an unwanted colored border around the 3D live
    # view. fitness_t still drives part-color vividness above; it just no
    # longer draws a border.

    # Per-generation "flash" accent: a single bright, thin ring in THIS
    # generation's distinct color (see generation_flash_rgb()), inset
    # just past the fitness glow above -- a quick visual pulse showing
    # "this is a new generation" independent of how good it scored.
    # Drawn only when a live gen_marker is present (Optimizer playback);
    # never touches the Prediction / Flight Envelope views.
    # NOTE: a per-generation colored "flash" ring used to be drawn here,
    # framing the plot in generation_flash_rgb(). Removed by request along
    # with the fitness glow bezel above -- both read as an unwanted colored
    # border. The on-body generation marker/label below is unaffected.

    # HUD-style text baked onto the plot itself (generation / step counter,
    # live fitness or performance value) -- drawn last, in axes-fraction
    # coordinates, so it stays pinned to the bottom-left corner regardless
    # of how the 3D view is rotated/zoomed. Plain text, no background box
    # (removed by request) -- a fixed dark, readable color is used instead
    # of the fitness tint, since a vivid hue with no box behind it can be
    # hard to read against the light body background.
    # ax.text2D() (unlike ax.add_patch()) is specifically designed to
    # overlay flat 2D content on a 3D axes, so this one's safe as-is.
    if overlay_text and show_overlay:
        ax.text2D(
            0.02, 0.02, overlay_text, transform=ax.transAxes,
            fontsize=max(fs_label - 3, 7), fontweight='bold', color='#111827',
            va='bottom', ha='left', linespacing=1.4, zorder=1001)

    # Live "which generation is this" marker: a small bright-yellow dot
    # anchored to an actual 3D point on the fuselage spine (moves with
    # the body if it's rotated/pitched), plus a compact label naming the
    # generation and its best fitness -- distinct from overlay_text
    # above, which is a fixed on-screen HUD corner label rather than a
    # point living on the model itself.
    if gen_marker:
        gen = int(gen_marker.get('gen', 0))
        total = max(int(gen_marker.get('total', 1)), 1)
        fit_v = gen_marker.get('fitness', 0.0)
        frac = (gen - 1) / (total - 1) if total > 1 else 0.5
        mx, my, mz = _generation_marker_xyz(full_params, frac, pitch_deg=pitch_deg)

        marker_rgb = generation_flash_rgb(gen)
        ax.scatter([mx], [my], [mz], color=marker_rgb, edgecolor='#111827',
                  linewidth=1.1, s=(80 if large else 55), depthshade=False,
                  zorder=2000)
        # Plain text label, no background box (removed by request) -- a
        # fixed dark, readable color instead of the per-generation tint.
        ax.text(mx, my, mz + max(full_params.get('body_diameter', 1.0), 1.0) * 0.06,
               f'G{gen}\nFit={fit_v:.4f}',
               fontsize=(fs_label if large else fs_label - 1), fontweight='bold',
               color='#111827', ha='center', va='bottom', zorder=2001)

    return compute_geometry_metrics(full_params)


def visualize_aero_body(**kwargs):
    """
    Generate and display/save the 3D aerodynamic body visualization.
    If a Matplotlib 3D window is already open, updates it in-place.
    """
    # Standard default values
    defaults = {
        "nose_length": 2.0,
        "nose_diameter": 0.5,
        "nose_bluntness": 0.2,
        "body_length": 5.0,
        "body_diameter": 0.5,
        "wing_le": 3.0,
        "root_chord": 1.5,
        "tip_chord": 0.5,
        "semi_span": 2.0,
        "w_hinge_line": 0.7,
        "root_th": 0.08,
        "tip_th": 0.03,
        "wing_sweep": 30.0,
        "tail_le": 6.0,
        "tail_root_chord": 1.0,
        "tail_tip_chord": 0.3,
        "tail_sweep": 45.0,
        "tail_semi_span": 1.0,
        "f_hinge_line": 0.75,
        "tail_root_th": 0.05,
        "tail_tip_th": 0.02,
        "tail_config": "cruciform",
        "png_out": "aero_body.png",
        "html_out": "aero_body.html",
        "show": True,
        "block": False
    }

    params = defaults.copy()
    params.update(kwargs)

    # Ensure all dimensions/lengths/sweeps are positive and hinge lines/bluntness are fractional
    for k in ["nose_length", "nose_diameter", "body_length", "body_diameter", 
              "wing_le", "root_chord", "tip_chord", "semi_span", "root_th", "tip_th", 
              "tail_le", "tail_root_chord", "tail_tip_chord", "tail_semi_span", 
              "tail_root_th", "tail_tip_th"]:
        params[k] = abs(params[k])
        
    params["wing_sweep"] = abs(params["wing_sweep"])
    params["tail_sweep"] = abs(params["tail_sweep"])
    
    # Hinge lines and bluntness inputs - if > 1.0, assume they entered it as percentage (e.g. 60 instead of 0.6)
    for k in ["w_hinge_line", "f_hinge_line", "nose_bluntness"]:
        if params[k] > 1.0:
            params[k] = params[k] / 100.0
        params[k] = np.clip(params[k], 0.0, 1.0)

    # Generate geometry meshes
    
    # 1. Nose
    X_nose, Y_nose, Z_nose = generate_nose(params["nose_length"], params["nose_diameter"], params["nose_bluntness"])
    
    # 2. Body (starts at the end of the nose)
    X_body, Y_body, Z_body = generate_body(params["body_length"], params["body_diameter"], params["nose_length"])
    
    geometry = {
        'nose': (X_nose, Y_nose, Z_nose),
        'body': (X_body, Y_body, Z_body),
        'wings_tails': {}
    }
    
    hinge_lines = {}
    
    # 3. Wings (Symmetric horizontal, rotation angle = 0 and 180)
    # Right Wing (phi = 0)
    w_r_Xu, w_r_Yu, w_r_Zu, w_r_Xl, w_r_Yl, w_r_Zl, w_r_hl = generate_fin_panel(
        params["wing_le"], params["root_chord"], params["tip_chord"], params["semi_span"], params["wing_sweep"],
        params["root_th"], params["tip_th"], 0.0, params["w_hinge_line"]
    )
    geometry['wings_tails']['wing_right'] = [
        (w_r_Xu, w_r_Yu, w_r_Zu), (w_r_Xl, w_r_Yl, w_r_Zl)
    ]
    hinge_lines['wing_right'] = w_r_hl
    
    # Left Wing (phi = 180)
    w_l_Xu, w_l_Yu, w_l_Zu, w_l_Xl, w_l_Yl, w_l_Zl, w_l_hl = generate_fin_panel(
        params["wing_le"], params["root_chord"], params["tip_chord"], params["semi_span"], params["wing_sweep"],
        params["root_th"], params["tip_th"], 180.0, params["w_hinge_line"]
    )
    geometry['wings_tails']['wing_left'] = [
        (w_l_Xu, w_l_Yu, w_l_Zu), (w_l_Xl, w_l_Yl, w_l_Zl)
    ]
    hinge_lines['wing_left'] = w_l_hl

    # 4. Tail Fins
    if params["tail_config"] == 'cruciform':
        # 4 symmetric fins at 0, 90, 180, 270 degrees
        phi_angles = [0.0, 90.0, 180.0, 270.0]
        names = ['tail_right', 'tail_top', 'tail_left', 'tail_bottom']
    else:
        # Standard aircraft tail: horizontal stabilizers (0, 180) and single vertical stabilizer (90)
        phi_angles = [0.0, 90.0, 180.0]
        names = ['tail_right', 'tail_top', 'tail_left']
        
    for name, phi in zip(names, phi_angles):
        Xu, Yu, Zu, Xl, Yl, Zl, hl = generate_fin_panel(
            params["tail_le"], params["tail_root_chord"], params["tail_tip_chord"], params["tail_semi_span"], params["tail_sweep"],
            params["tail_root_th"], params["tail_tip_th"], phi, params["f_hinge_line"]
        )
        geometry['wings_tails'][name] = [
            (Xu, Yu, Zu), (Xl, Yl, Zl)
        ]
        hinge_lines[name] = hl
        
    # Generate Plots
    plot_matplotlib(geometry, hinge_lines, params["png_out"], show=params["show"], block=params["block"])
    if PLOTLY_AVAILABLE and params["html_out"]:
        plot_plotly(geometry, hinge_lines, params, params["html_out"])


def main():
    print("========================================")
    print("   Aerodynamic Body 3D Visualizer")
    print("========================================")
    
    # Standard default values
    defaults = {
        "nose_length": 2.0,
        "nose_diameter": 0.5,
        "nose_bluntness": 0.2,
        "body_length": 5.0,
        "body_diameter": 0.5,
        "wing_le": 3.0,
        "root_chord": 1.5,
        "tip_chord": 0.5,
        "semi_span": 2.0,
        "w_hinge_line": 0.7,
        "root_th": 0.08,
        "tip_th": 0.03,
        "wing_sweep": 30.0,
        "tail_le": 6.0,
        "tail_root_chord": 1.0,
        "tail_tip_chord": 0.3,
        "tail_sweep": 45.0,
        "tail_semi_span": 1.0,
        "f_hinge_line": 0.75,
        "tail_root_th": 0.05,
        "tail_tip_th": 0.02,
        "tail_config": "cruciform",
        "png_out": "aero_body.png",
        "html_out": "aero_body.html"
    }

    try:
        use_interactive = input("Would you like to input parameters interactively? (y/n) [n]: ").strip().lower() == 'y'
    except (KeyboardInterrupt, EOFError):
        use_interactive = False
        print("\nUsing defaults.")

    params = {}
    if use_interactive:
        print("\nEnter parameter values (press Enter to accept default):")
        params["nose_length"] = prompt_parameter("nose length", defaults["nose_length"])
        params["nose_diameter"] = prompt_parameter("nose diameter", defaults["nose_diameter"])
        params["nose_bluntness"] = prompt_parameter("nose bluntness (0.0=sharp, 1.0=hemisphere)", defaults["nose_bluntness"])
        params["body_length"] = prompt_parameter("body length", defaults["body_length"])
        params["body_diameter"] = prompt_parameter("body diameter", defaults["body_diameter"])
        
        params["wing_le"] = prompt_parameter("wing leading edge (le)", defaults["wing_le"])
        params["root_chord"] = prompt_parameter("wing root chord", defaults["root_chord"])
        params["tip_chord"] = prompt_parameter("wing tip chord", defaults["tip_chord"])
        params["semi_span"] = prompt_parameter("wing semi-span", defaults["semi_span"])
        params["w_hinge_line"] = prompt_parameter("wing hinge line (fraction)", defaults["w_hinge_line"])
        params["root_th"] = prompt_parameter("wing root thickness (root th)", defaults["root_th"])
        params["tip_th"] = prompt_parameter("wing tip thickness (tip th)", defaults["tip_th"])
        params["wing_sweep"] = prompt_parameter("wing sweep (degrees)", defaults["wing_sweep"])
        
        params["tail_le"] = prompt_parameter("tail leading edge (taile le)", defaults["tail_le"])
        params["tail_root_chord"] = prompt_parameter("tail root chord (root chord.1)", defaults["tail_root_chord"])
        params["tail_tip_chord"] = prompt_parameter("tail tip chord (tip chod.1)", defaults["tail_tip_chord"])
        params["tail_sweep"] = prompt_parameter("tail sweep", defaults["tail_sweep"])
        params["tail_semi_span"] = prompt_parameter("tail semi-span (sei-span.1)", defaults["tail_semi_span"])
        params["f_hinge_line"] = prompt_parameter("tail hinge line (f hinge line)", defaults["f_hinge_line"])
        params["tail_root_th"] = prompt_parameter("tail root thickness (root th.1)", defaults["tail_root_th"])
        params["tail_tip_th"] = prompt_parameter("tail tip thickness (tip th.1)", defaults["tail_tip_th"])
        
        try:
            tail_config_input = input(f"tail configuration (cruciform/aircraft) [{defaults['tail_config']}]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            tail_config_input = ""
        params["tail_config"] = tail_config_input if tail_config_input in ['cruciform', 'aircraft'] else defaults['tail_config']
        
        params["png_out"] = defaults["png_out"]
        params["html_out"] = defaults["html_out"]
        params["show"] = True
    else:
        # Fallback to argparse
        parser = argparse.ArgumentParser(description="Aerodynamic Body 3D Visualizer")
        parser.add_argument("--nose-length", type=float, default=defaults["nose_length"])
        parser.add_argument("--nose-diameter", type=float, default=defaults["nose_diameter"])
        parser.add_argument("--nose-bluntness", type=float, default=defaults["nose_bluntness"])
        parser.add_argument("--body-length", type=float, default=defaults["body_length"])
        parser.add_argument("--body-diameter", type=float, default=defaults["body_diameter"])
        parser.add_argument("--wing-le", type=float, default=defaults["wing_le"])
        parser.add_argument("--root-chord", type=float, default=defaults["root_chord"])
        parser.add_argument("--tip-chord", type=float, default=defaults["tip_chord"])
        parser.add_argument("--semi-span", type=float, default=defaults["semi_span"])
        parser.add_argument("--w-hinge-line", type=float, default=defaults["w_hinge_line"])
        parser.add_argument("--root-th", type=float, default=defaults["root_th"])
        parser.add_argument("--tip-th", type=float, default=defaults["tip_th"])
        parser.add_argument("--wing-sweep", type=float, default=defaults["wing_sweep"])
        parser.add_argument("--tail-le", type=float, default=defaults["tail_le"])
        parser.add_argument("--tail-root-chord", type=float, default=defaults["tail_root_chord"])
        parser.add_argument("--tail-tip-chord", type=float, default=defaults["tail_tip_chord"])
        parser.add_argument("--tail-sweep", type=float, default=defaults["tail_sweep"])
        parser.add_argument("--tail-semi-span", type=float, default=defaults["tail_semi_span"])
        parser.add_argument("--f-hinge-line", type=float, default=defaults["f_hinge_line"])
        parser.add_argument("--tail-root-th", type=float, default=defaults["tail_root_th"])
        parser.add_argument("--tail-tip-th", type=float, default=defaults["tail_tip_th"])
        parser.add_argument("--tail-config", type=str, choices=['cruciform', 'aircraft'], default=defaults["tail_config"])
        parser.add_argument("--png-out", type=str, default=defaults["png_out"])
        parser.add_argument("--html-out", type=str, default=defaults["html_out"])
        parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
        
        args = parser.parse_args()
        params = vars(args)
        params["show"] = not args.no_show

    # In standalone CLI execution, we block on plt.show()
    params["block"] = params["show"]
    
    print("\nGenerating and displaying visualizations...")
    visualize_aero_body(**params)
    print("Done!")


if __name__ == "__main__":
    main()