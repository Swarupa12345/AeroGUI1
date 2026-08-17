#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import threading
import queue
import csv
from datetime import datetime
from collections import deque
import gc
#import tkinter as tk
#import tkinter.filedialog as filedialog

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import PySimpleGUI as sg
except ImportError:
    print("ERROR: pip install PySimpleGUI==4.60.5")
    sys.exit(1)

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import tkinter as tk

from predictor import aerodynamic_prediction as _raw_pred, ENSEMBLE_MODE
from optimizer import run_optimization, PARAM_NAMES
from envelope import alpha_sweep, mach_sweep, altitude_sweep

from concurrent.futures import ThreadPoolExecutor

import aero_body_vis


# =========================================================
# 3D GEOMETRY VIEW - PARAMETER ADAPTER
# =========================================================
# aero_body_vis.py uses its own parameter names, and needs a handful of
# shape parameters (diameters, bluntness, tail sweep, hinge lines,
# tail configuration) that this GUI does not expose as inputs. These are
# held fixed at sensible values so the 3D view can be driven purely by
# the existing geometry fields.
# How often (in generations) the Optimizer tab captures the full 3D
# aero-body geometry for Undo/Redo stepping in the 3D popup. The very
# last generation is always captured regardless of this setting.
# Set to 1 so the Optimizer tab's 3D view and Undo/Redo history capture
# every single generation (not just every 5th) -- required for the live
# generation-by-generation aero-body playback.
OPT_GEOM_SNAPSHOT_EVERY = 1

# =========================================================
# 3D VISUALIZATION MODE
# =========================================================
# EMBED_3D_PREVIEW=False: no big 3D canvas sits inside any tab any more.
# Each tab instead gets one compact "VISUALIZE" button (see
# geo_preview_frame()) that opens the same large, rotatable 3D popup
# window that used to be reachable only via double-click / 'POPUP 3D'.
# The geometry (Figure + metrics) is still computed and cached exactly
# as before on every Estimate/Run/generation -- only the always-visible
# embedded Tk canvas widget is removed. Flip back to True to restore the
# old always-on embedded panel behaviour.
EMBED_3D_PREVIEW = False

# canvas_key -> int, bumped every time render_geometry_panel() computes
# a fresh frame for that tab (including fast_preview / live-playback
# frames from the optimizer generation loop and the alpha-sweep
# animation). The open 3D popup window polls this to know when to
# redraw itself live -- this is what makes clicking VISUALIZE while an
# optimization/sweep is running show the body actually morphing in real
# time, generation by generation, instead of a static snapshot.
_geo_live_seq = {}

# Live "animation" pacing: each landed generation/sweep-step is preceded
# by this many linearly-interpolated in-between frames (geometry + color
# + readouts all morph smoothly toward the new frame) so the 3D preview
# reads as continuous motion instead of a slideshow of snapshots.
LIVE_MORPH_STEPS = 6
LIVE_MORPH_STEP_DELAY = 0.045   # seconds between in-between frames

VIS_FIXED_DEFAULTS = {
    'nose_diameter': 250.0,
    'body_diameter': 250.0,
    'nose_bluntness': 0.15,
    'tail_sweep': 40.0,
    'w_hinge_line': 0.7,
    'f_hinge_line': 0.75,
    'tail_config': 'cruciform',
}

# Maps this GUI's PARAMS keys -> aero_body_vis.py's kwarg names
_GEOM_KEY_MAP = {
    'nose_len': 'nose_length',
    'body_len': 'body_length',
    'wing_le': 'wing_le',
    'root_chord': 'root_chord',
    'tip_chord': 'tip_chord',
    'semi_span': 'semi_span',
    'root_th': 'root_th',
    'tip_th': 'tip_th',
    'wing_sweep': 'wing_sweep',
    'tail_le': 'tail_le',
    'root_chord1': 'tail_root_chord',
    'tip_chord1': 'tail_tip_chord',
    'semi_span1': 'tail_semi_span',
    'root_th1': 'tail_root_th',
    'tip_th1': 'tail_tip_th',
}


def geom_dict_to_vis_params(geom):
    """
    Converts a dict keyed by this GUI's PARAMS names (nose_len, body_len,
    root_chord1, ...) into the full kwarg dict expected by
    aero_body_vis.render_geometry_on_figure(), filling in the fixed
    shape parameters the GUI doesn't collect.
    """
    vis_params = dict(VIS_FIXED_DEFAULTS)
    for gui_key, vis_key in _GEOM_KEY_MAP.items():
        if gui_key in geom:
            try:
                vis_params[vis_key] = float(geom[gui_key])
            except (TypeError, ValueError):
                pass
    return vis_params


# ------------------------------------------------------------------
#  Helper – convert a list of tuples to a list of dicts
# ------------------------------------------------------------------
def _sweep_to_dict(rows, var_name):
    if not rows:
        return []

    if isinstance(rows[0], dict):
        return rows

    return [
        {
            var_name: r[0],
            'CL': r[1],
            'CD': r[2],
            'XCP': r[3],
            'XCP_D': r[4],
            'CL_CD': r[5]
        }
        for r in rows
    ]


# =========================================================
# INFERENCE CACHE
# =========================================================
_pred_cache = {}
_CACHE_MAX = 5000
_CACHE_RND = 3


def aerodynamic_prediction(params: dict) -> dict:
    key = tuple(round(float(params[p]), _CACHE_RND) for p in PARAMS)
    if key in _pred_cache:
        return _pred_cache[key]
    result = _raw_pred(params)
    if len(_pred_cache) >= _CACHE_MAX:
        for k in list(_pred_cache)[:400]:
            del _pred_cache[k]
    _pred_cache[key] = result
    return result


# =========================================================
# THEME DEFINITIONS & GLOBALS
# =========================================================
CURRENT_THEME = 'Silver Slate'
FONT_FAMILY = 'Arial'

THEMES = {
    'Silver Slate': {
        'C_BG': '#CBD5E1',  # Gray/silver kind of color
        'C_PANEL': '#E2E8F0',  # Slate-200 panel bg
        'C_INP': '#FFFFFF',  # White input fields
        'C_DARK': '#94A3B8',  # Status bar / progress bg (slate-400)
        'C_BLUE': '#1E3A8A',  # Deep navy blue for primary buttons
        'C_CYAN': '#0F766E',  # Teal/cyan headers
        'C_GREEN': '#047857',  # Dark green metrics
        'C_AMBER': '#B45309',  # Dark amber warnings
        'C_RED': '#B91C1C',  # Red metrics
        'C_WHITE': '#0F172A',  # Slate-900 dark text
        'C_DIM': '#475569',  # Slate-600 dimmed text
        'C_BDR': '#94A3B8',  # Border color
        'C_PURP': '#6D28D9',  # Purple
        'C_HDR': '#1E293B',  # Premium dark header
        'C_INDG': '#4338CA',  # Indigo
        'C_ROSE': '#BE123C',
        'C_SOFT': '#BE123C',
        'SG_THEME': 'SilverSlate',
    },
    'Classic Dark': {
        'C_BG': '#0B0F1A',
        'C_PANEL': '#111827',
        'C_INP': '#1C2333',
        'C_DARK': '#07090F',
        'C_BLUE': '#3B82F6',
        'C_CYAN': '#06B6D4',
        'C_GREEN': '#10B981',
        'C_AMBER': '#F59E0B',
        'C_RED': '#EF4444',
        'C_WHITE': '#F1F5F9',
        'C_DIM': '#94A3B8',
        'C_BDR': '#1E293B',
        'C_PURP': '#8B5CF6',
        'C_HDR': '#0F172A',
        'C_INDG': '#6366F1',
        'C_ROSE': '#FB7185',
        'C_SOFT': '#FB7185',
        'SG_THEME': 'DRDL',
    }
}

# Add PySimpleGUI themes
sg.theme_add_new('SilverSlate', {
    'BACKGROUND': '#CBD5E1',
    'TEXT': '#0F172A',
    'INPUT': '#FFFFFF',
    'TEXT_INPUT': '#0F172A',
    'SCROLL': '#1E3A8A',
    'BUTTON': ('#FFFFFF', '#1E3A8A'),
    'PROGRESS': ('#0F766E', '#94A3B8'),
    'BORDER': 1,
    'SLIDER_DEPTH': 0,
    'PROGRESS_DEPTH': 0,
})

sg.theme_add_new('DRDL', {
    'BACKGROUND': '#0B0F1A',
    'TEXT': '#F1F5F9',
    'INPUT': '#1C2333',
    'TEXT_INPUT': '#F1F5F9',
    'SCROLL': '#3B82F6',
    'BUTTON': ('#F1F5F9', '#3B82F6'),
    'PROGRESS': ('#06B6D4', '#07090F'),
    'BORDER': 1,
    'SLIDER_DEPTH': 0,
    'PROGRESS_DEPTH': 0,
})

# Setup the theme color globals
C_BG = '#CBD5E1'
C_PANEL = '#E2E8F0'
C_INP = '#FFFFFF'
C_DARK = '#94A3B8'
C_BLUE = '#1E3A8A'
C_CYAN = '#0F766E'
C_GREEN = '#047857'
C_AMBER = '#B45309'
C_RED = '#B91C1C'
C_WHITE = '#0F172A'
C_DIM = '#475569'
C_BDR = '#94A3B8'
C_PURP = '#6D28D9'
C_HDR = '#1E293B'
C_INDG = '#4338CA'
C_ROSE = '#BE123C'
C_SOFT = '#BE123C'


def apply_theme(name):
    global C_BG, C_PANEL, C_INP, C_DARK, C_BLUE, C_CYAN, C_GREEN, C_AMBER, C_RED, C_WHITE, C_DIM, C_BDR, C_PURP, C_HDR, C_INDG, C_ROSE, C_SOFT, CURRENT_THEME
    CURRENT_THEME = name
    th = THEMES[name]
    C_BG = th['C_BG']
    C_PANEL = th['C_PANEL']
    C_INP = th['C_INP']
    C_DARK = th['C_DARK']
    C_BLUE = th['C_BLUE']
    C_CYAN = th['C_CYAN']
    C_GREEN = th['C_GREEN']
    C_AMBER = th['C_AMBER']
    C_RED = th['C_RED']
    C_WHITE = th['C_WHITE']
    C_DIM = th['C_DIM']
    C_BDR = th['C_BDR']
    C_PURP = th['C_PURP']
    C_HDR = th['C_HDR']
    C_INDG = th['C_INDG']
    C_ROSE = th['C_ROSE']
    C_SOFT = th['C_SOFT']
    sg.theme(th['SG_THEME'])


# Setup dynamic fonts
F_TITLE = (FONT_FAMILY, 18, 'bold')
F_SUB = (FONT_FAMILY, 16)
F_SEC = (FONT_FAMILY, 12, 'bold')
F_LBL = (FONT_FAMILY, 12)
F_INP = (FONT_FAMILY, 12, 'bold')
F_OUT = (FONT_FAMILY, 13, 'bold')
F_BTN = (FONT_FAMILY, 12, 'bold')
F_STS = (FONT_FAMILY, 12, 'bold')
F_TBL = (FONT_FAMILY, 12)
F_TOP5 = (FONT_FAMILY, 12)
F_CHROME = (FONT_FAMILY, 11, 'bold')
F_ARROW = (FONT_FAMILY, 16, 'bold')
F_PLTLBL = (FONT_FAMILY, 12, 'bold')
F_TABTXT = (FONT_FAMILY, 13, 'bold')


def update_fonts(family):
    global F_TITLE, F_SUB, F_SEC, F_LBL, F_INP, F_OUT, F_BTN, F_STS, F_TBL, F_TOP5, F_CHROME, F_ARROW, F_PLTLBL, F_TABTXT, FONT_FAMILY
    FONT_FAMILY = family
    F_TITLE = (family, 18, 'bold')
    F_SUB = (family, 16)
    F_SEC = (family, 12, 'bold')
    F_LBL = (family, 12)
    F_INP = (family, 12, 'bold')
    F_OUT = (family, 13, 'bold')
    F_BTN = (family, 12, 'bold')
    F_STS = (family, 12, 'bold')
    F_TBL = (family, 12)
    F_TOP5 = (family, 12)
    F_CHROME = (family, 11, 'bold')
    F_ARROW = (family, 16, 'bold')
    F_PLTLBL = (family, 12, 'bold')
    F_TABTXT = (family, 13, 'bold')


# =========================================================
# PARAMETER DEFINITIONS
# =========================================================
DEFAULTS = {
    'nose_len': 300, 'body_len': 2700, 'wing_le': 1500,
    'root_chord': 200, 'tip_chord': 150, 'semi_span': 1000,
    'root_th': 20, 'tip_th': 5, 'wing_sweep': 2.86,
    'tail_le': 2870, 'root_chord1': 120, 'tip_chord1': 60,
    'semi_span1': 100, 'root_th1': 15, 'tip_th1': 5,
    'mach': 0.2, 'alpha': 2, 'alt': 0,
}
LABELS = {
    'nose_len': 'Nose Length ',
    'body_len': 'Body Length ',
    'wing_le': 'Wing LE',
    'root_chord': 'Root Chord',
    'tip_chord': 'Tip Chord ',
    'semi_span': 'Semi-Span ',
    'root_th': 'Root Thickness',
    'tip_th': 'Tip Thickness',
    'wing_sweep': 'Wing Sweep ',
    'tail_le': 'Tail LE ',
    'root_chord1': 'Tail Root Chord',
    'tip_chord1': 'Tail Tip Chord',
    'semi_span1': 'Tail Semi-Span',
    'root_th1': 'Tail Root Thickness',
    'tip_th1': 'Tail Tip Thickness',
    'mach': 'Mach Number',
    'alpha': 'Alpha ',
    'alt': 'Altitude ',
}
PARAMS = list(DEFAULTS.keys())
BOUNDS = {
    'nose_len': (120, 360), 'body_len': (2400, 3000),
    'wing_le': (1000, 2000), 'root_chord': (150, 250),
    'tip_chord': (110, 190), 'semi_span': (600, 1500),
    'root_th': (15, 25), 'tip_th': (5, 11),
    'wing_sweep': (0.0, 70.0), 'tail_le': (2830, 2910),
    'root_chord1': (80, 160), 'tip_chord1': (30, 90),
    'semi_span1': (60, 140), 'root_th1': (15, 21),
    'tip_th1': (5, 11), 'mach': (0.2, 0.8),
    'alpha': (0, 20), 'alt': (0, 6000),
}

apply_theme('Silver Slate')

# =========================================================
# LOGIN
# =========================================================
_VALID_USERS = {'drdl': 'drdl'}


def show_login() -> bool:
    ly = [
        [sg.Text('', background_color=C_BG, pad=(0, 10))],
        [sg.Text('DRDL AEROSPACE AI PLATFORM',
                 font=(FONT_FAMILY, 16, 'bold'),
                 text_color=C_CYAN,
                 background_color=C_BG,
                 justification='center',
                 expand_x=True)],
        [sg.Text('Secure Access..',
                 font=(FONT_FAMILY, 12),
                 text_color=C_DIM,
                 background_color=C_BG,
                 justification='center',
                 expand_x=True, pad=(0, (0, 18)))],
        [sg.Text('Username',
                 size=(12, 1),
                 font=(FONT_FAMILY, 12),
                 text_color=C_DIM,
                 background_color=C_BG),
         sg.Input('', key='LG_USER', size=(22, 1),
                  font=(FONT_FAMILY, 13, 'bold'),
                  background_color=C_INP,
                  text_color=C_WHITE,
                  border_width=1,
                  focus=True)],
        [sg.Text('', size=(0, 1),
                 background_color=C_BG, pad=(0, 3))],
        [sg.Text('Password', size=(12, 1), font=(FONT_FAMILY, 12),
                 text_color=C_DIM, background_color=C_BG),
         sg.Input('', key='LG_PASS', size=(22, 1),
                  font=(FONT_FAMILY, 13, 'bold'), background_color=C_INP,
                  text_color=C_WHITE, border_width=1, password_char='*')],
        [sg.Text('', key='LG_ERR', size=(38, 1), font=(FONT_FAMILY, 12),
                 text_color=C_RED, background_color=C_BG,
                 pad=(0, (6, 10)))],
        [sg.Column([[
            sg.Button('LOGIN', key='LG_OK',
                      font=(FONT_FAMILY, 13, 'bold'),
                      button_color=('#FFFFFF', C_BLUE),
                      border_width=1,
                      bind_return_key=True, pad=(10, 5),
                      mouseover_colors=('#FFFFFF', C_PURP)),
            sg.Button('EXIT', key='LG_EXIT',
                      font=(FONT_FAMILY, 13, 'bold'),
                      button_color=('#FFFFFF', '#991B1B'),
                      border_width=0,
                      mouseover_colors=('#FFFFFF', C_RED)),
        ]], background_color=C_BG, justification='center', expand_x=True)],
        [sg.Text('', background_color=C_BG, pad=(0, 10))],
    ]
    win = sg.Window('DRDL Login', ly, size=(500, 340), finalize=True,
                    background_color=C_BG,
                    element_justification='center',
                    margins=(22, 14), keep_on_top=True)
    attempts = 0
    while True:
        ev, vals = win.read(timeout=100)
        if ev in (sg.WIN_CLOSED, 'LG_EXIT', None):
            win.close()
            return False
        if ev == 'LG_OK':
            u = vals.get('LG_USER', '').strip()
            p = vals.get('LG_PASS', '').strip()
            if _VALID_USERS.get(u) == p:
                win.close()
                return True
            attempts += 1
            win['LG_ERR'].update(
                f' Invalid credentials  (attempt {attempts})')
            win['LG_PASS'].update('')


if not show_login():
    sys.exit(0)

# =========================================================
# RUNTIME STATE
# =========================================================
_model_rdy = True
_opt_run = False
_flt_run = False
_is_max = True
pred_q = queue.Queue()
opt_log_q = queue.Queue()

# Running list of best-fitness values seen so far *this* optimizer run --
# reset each time a new run starts (see 'Run_Opt' handler). Used to
# normalize each generation's fitness into [0,1] (worst..best seen so
# far) for the live red->green body tint (see aero_body_vis.fitness_to_rgb).
_opt_fitness_seen = []

_opt_figs = []
_opt_idx = 0
_opt_agg = None
_last_best_geom = {}
_opt_history = None
_opt_result = None

# -- 3D Geometry Preview state (one panel embedded per tab) --
_last_pred_geom = dict(DEFAULTS)      # last geometry used on Prediction tab

# Rough CL/CD range used only to tint the Tab 1 (Prediction) body on the
# same red->green scale as the Optimizer/Envelope tabs. Unlike those tabs
# there's no run-to-compare-against here, so this is a fixed, adjustable
# reference range rather than a live min/max.
PRED_LD_COLOR_RANGE = (0.0, 12.0)
_last_env_base_geom = dict(DEFAULTS)  # last base geometry used on Flight Envelope tab

_env_figs = []
_env_idx = 0
_env_agg = None
_env_ar = None
_env_mr = None
_env_lr = None
_env_geom_label = ''

t_start_pred = '--:--:--'
_t_start_opt = '--:--:--'
_t_start_env = '--:--:--'

MODEL_FILES = {
    "CL": "cl_xgb.model",
    "CD": "cd_xgb.model",
    "XCP": "xcp_xgb.model",
}
SCALER_FILE = "feature_scaler.pkl"

_OPT_TITLES = [
    'Fitness Evolution',
    'Aero Metrics per Gen',
    'CL/CD vs XCP Scatter',
]
_ENV_TITLES = [
    'Alpha Sweep (CL / CD / XCP)',
    'Mach Sweep (CL / CD / CL/CD)',
    'Altitude Sweep (CL / CD / CL/CD)',
]

_SWEEP_POOL = ThreadPoolExecutor(max_workers=4)


# =========================================================
# LAYOUT HELPERS
# =========================================================
def sf(values, key, default=0.0):
    try:
        return float(values[key])
    except Exception:
        return default


def _ts():
    return datetime.now().strftime('%H:%M:%S')


def sec_hdr(text, color=None):
    if color is None:
        color = C_CYAN
    return [sg.Text(f'| {text}', font=F_SEC, text_color=color,
                    background_color=C_PANEL, pad=(6, (10, 3)), expand_x=True)]


def sec_hdr_rows(text, color=None):
    if color is None:
        color = C_CYAN
    return [sec_hdr(text, color)]


def lbl(text, w=24, **kwargs):
    return sg.Text(
        text,
        size=(w, 1),
        font=F_LBL,
        text_color=C_DIM,
        background_color=C_PANEL,
        pad=((6, 4), (5, 5)),
        **kwargs
    )


def inp(key, value='', w=12, **kwargs):
    return sg.Input(
        str(value),
        key=key,
        size=(w, 1),
        font=F_INP,
        background_color=C_INP,
        text_color=C_WHITE,
        border_width=1,
        pad=((4, 4), (4, 4)),
        **kwargs
    )


def out_field(key, w=20, color=None, **kwargs):
    if color is None:
        color = C_CYAN
    return sg.Input(
        '—',
        key=key,
        size=(w, 1),
        font=F_OUT,
        text_color=color,
        background_color=C_BDR,
        readonly=True,
        border_width=0,
        disabled_readonly_background_color=C_BDR,
        disabled_readonly_text_color=color,
        pad=((4, 4), (4, 4)),
        **kwargs
    )


def action_btn(text, key, bg=None, w=20):
    if bg is None:
        bg = C_BLUE
    return sg.Button(
        text,
        key=key,
        size=(w, 1),
        font=F_BTN,
        button_color=('#FFFFFF', bg),
        border_width=0,
        pad=((6, 6), (6, 6)),
        mouseover_colors=('#FFFFFF', C_PURP))


def prog_row(bk, pk, mk):
    return [
        sg.ProgressBar(
            100,
            orientation='h',
            size=(36, 16),
            key=bk,
            bar_color=(C_CYAN, C_DARK),
            expand_x=True,
            pad=((4, 4), (4, 4))
        ),
        sg.Text(
            ' 0%',
            key=pk,
            size=(5, 1),
            font=F_INP,
            text_color=C_CYAN,
            background_color=C_BG
        ),
        sg.Text(
            '',
            key=mk,
            size=(32, 1),
            font=F_LBL,
            text_color=C_AMBER,
            background_color=C_BG,
            expand_x=True
        ),
    ]


def set_prog(bk, pk, mk, pct, msg=''):
    pct = max(0, min(100, int(pct)))
    window[bk].update(pct)
    window[pk].update(f'{pct:3d}%')
    window[mk].update(msg)
    window.refresh()


# def set_status(msg, elapsed=None, color=C_BLUE):
#     window['STS'].update(msg)
#     try:
#         window['STS'].Widget.config(fg=color)
#     except Exception:
#         pass
#     if elapsed is not None:
#         window['STS_T'].update(f' {elapsed:.3f} s')
#     else:
#         window['STS_T'].update('')
#     window.refresh()

def set_status(msg, elapsed=None, color=C_BLUE):
    window['STS'].update(msg)
    window['STS'].update(text_color=color)
    window.refresh()


def con_clear(key):
    window[key].update('', disabled=False)
    window[key].update('', disabled=True)


def con_append(key, text):
    el = window[key]
    el.update(disabled=False)
    el.print(text)
    el.update(disabled=True)


def styled_frame(title, layout, title_color=C_CYAN):
    return sg.Frame(
        title,
        layout,
        font=F_SEC,
        title_color=title_color,
        background_color=C_PANEL,
        border_width=1,
        relief=sg.RELIEF_GROOVE,
        expand_x=True,
        expand_y=True,
        pad=(8, 8)
    )


def show_validation_error(title, message):
    layout = [
        [sg.Text("Input Validation Error(s):", font=(FONT_FAMILY, 12, 'bold'), text_color=C_RED,
                 background_color=C_BG)],
        [sg.Multiline(message, size=(80, 12), font=('Courier New', 11), background_color=C_INP, text_color=C_WHITE,
                      disabled=True)],
        [sg.Button("OK", key="VAL_OK", size=(12, 1), font=F_BTN, button_color=('#FFFFFF', C_BLUE),
                   bind_return_key=True)]
    ]
    win = sg.Window(title, layout, modal=True, element_justification='center', background_color=C_BG)
    while True:
        # Same defensive guard as the 3D popup: never call .read() again
        # once the window's Tk root is already gone, and never let a
        # single bad read propagate into an unhandled crash.
        if getattr(win, 'TKroot', None) is None:
            break
        try:
            ev, _ = win.read()
        except Exception:
            break
        if ev in (sg.WIN_CLOSED, "VAL_OK", None):
            break
    try:
        win.close()
    except Exception:
        pass


def train_and_save(csv_path: str):
    """Train three XGB regressors + scaler; skip if artefacts already exist."""
    if all(os.path.exists(f) for f in MODEL_FILES.values()) and os.path.exists(SCALER_FILE):
        print("\n=== Models and scaler already exist – skipping training ===")
        return

    print("\n=== Loading & cleaning data ===")
    data = _load_raw_data(csv_path)

    # keep only known features / targets
    feats = [f for f in FEATURES if f in data.columns]
    X = data[feats]
    y = data[TARGETS]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    # MIN-MAX SCALER
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {}
    for tgt in TARGETS:
        mdl = xgb.XGBRegressor(**COMMON_PARAMS)
        mdl.fit(X_tr_s, y_tr[tgt])
        models[tgt] = mdl
        key = tgt.split('.')[0] if '.' in tgt else tgt
        mdl.save_model(MODEL_FILES[key])

    joblib.dump(scaler, SCALER_FILE)


# Load models & scaler

def load_models_and_scaler():
    scaler = joblib.load(SCALER_FILE)
    boosters = {}
    for key, path in MODEL_FILES.items():
        b = xgb.Booster()
        b.load_model(path)
        boosters[key] = b
    return boosters, scaler


def validate_prediction_inputs(values) -> tuple[bool, str]:
    errors = []
    for p in PARAMS:
        val_str = str(values.get(p, '')).strip()
        try:
            val = float(val_str)
            if p in PARAMS[:15]:
                if val < 0:
                    errors.append(f"• {LABELS[p]} ({val_str}) cannot be negative")
            else:
                lo, hi = BOUNDS[p]
                if val < lo or val > hi:
                    errors.append(f"• {LABELS[p]} ({val_str}) must be between {lo} and {hi}")
        except ValueError:
            errors.append(f"• {LABELS[p]} must be a valid number (got '{val_str}')")
    if errors:
        return False, "\n".join(errors)
    return True, ""


def validate_optimizer_inputs(values) -> tuple[bool, str]:
    errors = []
    # 1. Validate the 15 geometry parameter bounds
    for p in PARAMS[:15]:
        low_key = f"{p}_LOW"
        high_key = f"{p}_HIGH"
        low_str = str(values.get(low_key, '')).strip()
        high_str = str(values.get(high_key, '')).strip()

        low_val, high_val = None, None
        try:
            low_val = float(low_str)
            if low_val < 0:
                errors.append(f"• {LABELS[p]} Lower Bound ({low_val}) cannot be negative")
        except ValueError:
            errors.append(f"• {LABELS[p]} Lower Bound must be a number (got '{low_str}')")

        try:
            high_val = float(high_str)
            if high_val < 0:
                errors.append(f"• {LABELS[p]} Upper Bound ({high_val}) cannot be negative")
        except ValueError:
            errors.append(f"• {LABELS[p]} Upper Bound must be a number (got '{high_str}')")

        if low_val is not None and high_val is not None:
            if low_val > high_val:
                errors.append(f"• {LABELS[p]} Lower Bound ({low_val}) cannot be greater than Upper Bound ({high_val})")

    # 2. Validate the 3 flight conditions (read from prediction tab inputs)
    for p in ['mach', 'alpha', 'alt']:
        val_str = str(values.get(p, '')).strip()
        try:
            val = float(val_str)
            lo_limit, hi_limit = BOUNDS[p]
            if val < lo_limit or val > hi_limit:
                errors.append(f"• Baseline {LABELS[p]} ({val}) must be within training limit ({lo_limit}, {hi_limit})")
        except ValueError:
            errors.append(f"• Baseline {LABELS[p]} must be a number (got '{val_str}')")

    # Constraints
    for c in ['CL', 'CD', 'XCP']:
        min_key = f"{c}_MIN"
        max_key = f"{c}_MAX"
        min_str = str(values.get(min_key, '')).strip()
        max_str = str(values.get(max_key, '')).strip()

        min_val, max_val = None, None
        try:
            min_val = float(min_str)
        except ValueError:
            errors.append(f"• {c} Min Constraint must be a number (got '{min_str}')")
        try:
            max_val = float(max_str)
        except ValueError:
            errors.append(f"• {c} Max Constraint must be a number (got '{max_str}')")

        if min_val is not None and max_val is not None:
            if min_val > max_val:
                errors.append(f"• {c} Min Constraint ({min_val}) cannot be greater than Max Constraint ({max_val})")

    # Settings
    for key, label in [('MAXITER', 'Max Generations'), ('POPSIZE', 'Population Size'),
                       ('ITERMAX', 'Max Gene-Swap Steps')]:
        val_str = str(values.get(key, '')).strip()
        try:
            val = int(val_str)
            if val <= 0:
                errors.append(f"• {label} must be a positive integer (got {val})")
        except ValueError:
            errors.append(f"• {label} must be a valid integer (got '{val_str}')")

    if errors:
        return False, "\n".join(errors)
    return True, ""


def validate_envelope_inputs(values) -> tuple[bool, str]:
    errors = []
    # 1. Validate the 15 geometry parameters from the envelope tab (prefix E_)
    for p in PARAMS[:15]:
        key = f"E_{p}"
        val_str = str(values.get(key, '')).strip()
        try:
            val = float(val_str)
            if val < 0:
                errors.append(f"• Base {LABELS[p]} ({val_str}) cannot be negative")
        except ValueError:
            errors.append(f"• Base {LABELS[p]} must be a valid number (got '{val_str}')")

    # 2. Validate baseline flight conditions from the prediction tab (no prefix)
    for p in PARAMS[15:]:
        key = p
        val_str = str(values.get(key, '')).strip()
        try:
            val = float(val_str)
            lo, hi = BOUNDS[p]
            if val < lo or val > hi:
                errors.append(f"• Baseline {LABELS[p]} ({val_str}) must be between {lo} and {hi}")
        except ValueError:
            errors.append(f"• Baseline {LABELS[p]} must be a valid number (got '{val_str}')")

    # Sweep Ranges
    for sweep_name, prefix, lo_bound, hi_bound in [
        ('Alpha Sweep', 'ALPHA', 0.0, 20.0),
        ('Mach Sweep', 'MACH', 0.2, 0.8),
        ('Altitude Sweep', 'ALT', 0.0, 6000.0)
    ]:
        min_str = str(values.get(f"{prefix}_MIN", '')).strip()
        max_str = str(values.get(f"{prefix}_MAX", '')).strip()
        stp_str = str(values.get(f"{prefix}_STP", '')).strip()

        min_val, max_val, stp_val = None, None, None
        try:
            min_val = float(min_str)
            if min_val < lo_bound or min_val > hi_bound:
                errors.append(f"• {sweep_name} Min ({min_val}) must be within training limit ({lo_bound}, {hi_bound})")
        except ValueError:
            errors.append(f"• {sweep_name} Min must be a number (got '{min_str}')")

        try:
            max_val = float(max_str)
            if max_val < lo_bound or max_val > hi_bound:
                errors.append(f"• {sweep_name} Max ({max_val}) must be within training limit ({lo_bound}, {hi_bound})")
        except ValueError:
            errors.append(f"• {sweep_name} Max must be a number (got '{max_str}')")

        try:
            stp_val = float(stp_str)
            if stp_val <= 0:
                errors.append(f"• {sweep_name} Step must be positive (got '{stp_str}')")
        except ValueError:
            errors.append(f"• {sweep_name} Step must be a number (got '{stp_str}')")

        if min_val is not None and max_val is not None:
            if min_val > max_val:
                errors.append(f"• {sweep_name} Min ({min_val}) cannot be greater than Max ({max_val})")
            if stp_val is not None and stp_val > (max_val - min_val) and min_val != max_val:
                errors.append(f"• {sweep_name} Step ({stp_val}) is too large for the range ({min_val} to {max_val})")

    if errors:
        return False, "\n".join(errors)
    return True, ""


# =========================================================
# MATPLOTLIB DARK STYLE
# =========================================================
def _mpl_style():
    plt.rcParams.update({
        'figure.facecolor': C_BG,
        'axes.facecolor': C_INP,
        'axes.edgecolor': C_BDR,
        'axes.labelcolor': C_DIM,
        'axes.titlecolor': C_CYAN,
        'xtick.color': C_DIM,
        'ytick.color': C_DIM,
        'grid.color': C_BDR,
        'grid.linewidth': 1.2,
        'grid.linestyle': '--',
        'text.color': C_WHITE,
        'font.family': 'sans-serif' if FONT_FAMILY in ('Arial', 'Helvetica') else 'serif',
        'font.sans-serif': [FONT_FAMILY, 'DejaVu Sans', 'Arial', 'Helvetica'],
        'font.serif': [FONT_FAMILY, 'DejaVu Serif', 'Times New Roman'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.facecolor': C_PANEL,
        'legend.edgecolor': C_BDR,
    })


# =========================================================
# CANVAS EMBED
# =========================================================
def _embed_fig(fig, canvas_key):
    """Embed a matplotlib Figure into a PySimpleGUI Canvas element.

    This helper is used exclusively by the 3D geometry preview panels, so
    the double-click handler opens the dedicated large 3D popup (rotatable,
    zoomable) rather than the 2D-only `_open_zoom` used by the line-plot
    canvases elsewhere in the app.

    Reuses one FigureCanvasTkAgg per canvas_key across calls. Destroying
    and rebuilding the Tk widget (the old behaviour) on every single call
    was the single biggest cost of the live per-generation optimizer
    preview and the alpha-sweep animation -- far more than the actual
    matplotlib draw -- because it tore down and repacked a Tk widget tree
    on every frame. Now the widget is only built once per canvas; every
    later redraw of the *same* Figure object just repaints it in place,
    so the panel behaves like one persistent live window rather than a
    flicker of new ones, matching the "3D window should never close or
    reopen" requirement.
    """
    agg = _geo_aggs.get(canvas_key)
    if agg is not None and agg.figure is fig:
        agg.draw_idle()
        try:
            agg.get_tk_widget().update_idletasks()
        except Exception:
            pass
        # A fresh Axes3D (or a reused one after ax.cla()) needs its
        # mouse-rotate/zoom handlers (re-)wired to the still-live canvas,
        # but the double-click handler below is already bound once to
        # this same `agg` and must NOT be reconnected each frame -- doing
        # so would stack up duplicate popup triggers, one more per
        # generation, without ever being disconnected.
        for ax in fig.axes:
            if hasattr(ax, 'mouse_init'):
                try:
                    ax.mouse_init()
                except Exception:
                    pass
        return agg

    cv = window[canvas_key].TKCanvas
    try:
        cv.unbind('<Configure>')
    except Exception:
        pass
    for ch in cv.winfo_children():
        ch.destroy()
    agg = FigureCanvasTkAgg(fig, master=cv)
    if not hasattr(fig, '_orig_dpi'):
        fig._orig_dpi = fig.dpi
    if not hasattr(fig, '_orig_size'):
        fig._orig_size = list(fig.get_size_inches())
    fig.dpi = fig._orig_dpi
    fig.set_size_inches(fig._orig_size, forward=False)
    agg.draw()
    agg.get_tk_widget().pack(side='top', fill='both', expand=True)

    # Re-bind the 3D axes' mouse-rotate/zoom handlers to *this* canvas.
    # (Axes3D wires itself to whatever `fig.canvas` was at creation time;
    # since the Figure is created before FigureCanvasTkAgg exists, that
    # binding is stale until we re-run mouse_init() here -- this is what
    # makes the embedded preview a genuinely "live" rotatable 3D view
    # instead of a static snapshot.)
    for ax in fig.axes:
        if hasattr(ax, 'mouse_init'):
            try:
                ax.mouse_init()
            except Exception:
                pass

    agg.mpl_connect('button_press_event',
                    lambda e: (_show_geo_popup(canvas_key)
                        if e.dblclick else None))
    _geo_aggs[canvas_key] = agg
    return agg


def _zoom_ax(ax, factor):
    """Rescales a 3D axes' X/Y/Z limits around their own midpoints by
    `factor` (< 1 zooms in, > 1 zooms out). Used by the popup's
    Zoom In / Zoom Out buttons."""
    try:
        lims = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
        setters = [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]
        for (lo, hi), setter in zip(lims, setters):
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            setter(mid - half, mid + half)
    except Exception:
        pass


def _export_geo_generations(canvas_key, render_mode='solid'):
    """
    Exports every snapshot in canvas_key's Undo/Redo history (for the
    Optimizer tab: one snapshot per DE generation -- see
    _set_geo_history_from_generations; for Prediction/Flight Envelope:
    one snapshot per run/refresh) as its own numbered PNG frame. Each
    frame is tinted along the same red(worst)->green(best) gradient
    aero_body_vis already uses for live fitness feedback (fitness_t =
    its position in the sequence), so paging through the exported frames
    shows -- in colour -- how the body evolved generation by generation.
    Also writes one combined filmstrip PNG (up to 12 evenly-spaced
    frames side by side) as a quick overview. Returns the export folder
    path, or None if there was nothing to export / the user cancelled.
    """
    hist = _geo_history.get(canvas_key) or []
    if not hist:
        sg.popup_quick_message('Nothing to export yet -- run this tab first.')
        return None

    out_dir = sg.popup_get_folder('Choose a folder to export frames into...',
                                  no_window=True)
    if not out_dir:
        return None

    _mpl_style()
    n = len(hist)
    stamp = _ts().replace(':', '-').replace(' ', '_')
    export_dir = os.path.join(out_dir, f'aero_body_export_{stamp}')
    os.makedirs(export_dir, exist_ok=True)

    frame_paths = []
    for i, entry in enumerate(hist):
        t = (i / (n - 1)) if n > 1 else 1.0
        label = entry.get('label', f'Frame {i + 1}')
        frame_fig = Figure(figsize=(9.0, 6.0))
        frame_fig.patch.set_facecolor('#0B0F1A')
        try:
            aero_body_vis.render_geometry_on_figure(
                frame_fig, entry['vis_params'], title=label, large=True,
                bg_color='#EEF1F5', render_mode=render_mode,
                fitness_t=t, overlay_text=label)
        except Exception:
            plt.close(frame_fig)
            continue
        safe_label = ''.join(c if c.isalnum() else '_' for c in label)
        fpath = os.path.join(export_dir, f'{i + 1:03d}_{safe_label}.png')
        try:
            frame_fig.savefig(fpath, dpi=170, facecolor=frame_fig.get_facecolor())
            frame_paths.append(fpath)
        except Exception:
            pass
        plt.close(frame_fig)

    # Combined filmstrip overview: built from the frames already saved
    # above (cheapest and most robust way to lay several 3D renders out
    # side by side without fighting Matplotlib's 3D-subplot camera sync).
    try:
        import matplotlib.image as mpimg
        strip_n = min(n, 12)
        idxs = sorted(set(int(round(k * (n - 1) / max(strip_n - 1, 1)))
                          for k in range(strip_n)))
        strip_fig = Figure(figsize=(3.2 * len(idxs), 3.6))
        strip_fig.patch.set_facecolor('#0B0F1A')
        for col, idx in enumerate(idxs):
            ax = strip_fig.add_subplot(1, len(idxs), col + 1)
            ax.axis('off')
            fp = frame_paths[idx] if idx < len(frame_paths) else None
            if fp and os.path.exists(fp):
                ax.imshow(mpimg.imread(fp))
            ax.set_title(hist[idx].get('label', f'#{idx + 1}'),
                        fontsize=8, color='#CBD5E1')
        strip_path = os.path.join(export_dir, '000_filmstrip_overview.png')
        strip_fig.savefig(strip_path, dpi=150, facecolor=strip_fig.get_facecolor())
        plt.close(strip_fig)
    except Exception:
        pass

    return export_dir


def _render_frame_to_array(vis_params, label, overlay, render_mode, fitness_t,
                           figsize=(9.6, 6.4), dpi=120):
    """
    Renders one aero-body frame off-screen (no Tk canvas involved) and
    returns it as an (H, W, 3) uint8 RGB numpy array -- the shared
    building block for video/GIF export below. Uses FigureCanvasAgg
    (headless), never touching the live popup's own canvas/figure.
    """
    frame_fig = Figure(figsize=figsize, dpi=dpi)
    frame_fig.patch.set_facecolor('#0B0F1A')
    aero_body_vis.render_geometry_on_figure(
        frame_fig, vis_params, title=label, large=True,
        bg_color='#EEF1F5', render_mode=render_mode,
        fitness_t=fitness_t, overlay_text=overlay)
    canvas_agg = FigureCanvasAgg(frame_fig)
    canvas_agg.draw()
    w, h = canvas_agg.get_width_height()
    buf = np.frombuffer(canvas_agg.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    arr = buf[:, :, :3].copy()
    plt.close(frame_fig)
    return arr


def _export_geo_video(canvas_key, render_mode='solid', fps=8):
    """
    Exports canvas_key's Undo/Redo history (one frame per snapshotted
    generation/step -- the same frames the live 3D popup showed as the
    run played out) as a single, real, playable video recording, so the
    user can watch back how the aero body evolved rather than paging
    through separate images.

    Tries, in priority order:
      1. imageio (writes a real .mp4, using its ffmpeg plugin) if the
         'imageio' package is installed.
      2. An animated .gif via Pillow if the 'Pillow' package is installed
         (imageio not available/failed).
      3. Falls back to the existing PNG-frame + filmstrip export
         (_export_geo_generations) if neither video library is present,
         with a message explaining why and what to install for a real
         video file.

    Returns the path to the exported file (or folder, in the PNG
    fallback case), or None if the user cancelled / there was nothing
    to export.
    """
    hist = _geo_history.get(canvas_key) or []
    if not hist:
        sg.popup_quick_message('Nothing to export yet -- run this tab first.')
        return None

    _mpl_style()
    n = len(hist)

    sg.popup_quick_message('Rendering frames for video export...',
                           auto_close=False, non_blocking=True)
    frames = []
    try:
        for i, entry in enumerate(hist):
            t = (i / (n - 1)) if n > 1 else 1.0
            label = entry.get('label', f'Frame {i + 1}')
            overlay = entry.get('overlay') or label
            try:
                frames.append(_render_frame_to_array(
                    entry['vis_params'], label, overlay, render_mode, t))
            except Exception:
                continue
    finally:
        try:
            sg.popup_quick_message('', auto_close=True, auto_close_duration=0)
        except Exception:
            pass

    if not frames:
        sg.popup_error('Could not render any frames to export.')
        return None

    # -- 1) Preferred: a real .mp4 via imageio -----------------------------
    try:
        import imageio.v2 as imageio
        save_path = sg.popup_get_file(
            'Save 3D visualization video as...', save_as=True, no_window=True,
            default_extension='.mp4',
            file_types=(('MP4 Video', '*.mp4'), ('GIF Animation', '*.gif'),
                       ('All Files', '*.*')))
        if not save_path:
            return None
        if save_path.lower().endswith('.gif'):
            imageio.mimsave(save_path, frames, fps=fps)
        else:
            imageio.mimsave(save_path, frames, fps=fps, quality=8, macro_block_size=None)
        sg.popup_quick_message(f'Video saved to:\n{save_path}')
        return save_path
    except ImportError:
        pass
    except Exception as e:
        sg.popup_error(f'Video export via imageio failed:\n{e}\n'
                       f'Falling back to an animated GIF...')

    # -- 2) Fallback: an animated .gif via Pillow --------------------------
    try:
        from PIL import Image
        save_path = sg.popup_get_file(
            'Save 3D visualization animation as...', save_as=True, no_window=True,
            default_extension='.gif',
            file_types=(('GIF Animation', '*.gif'), ('All Files', '*.*')))
        if not save_path:
            return None
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            save_path, save_all=True, append_images=pil_frames[1:],
            duration=int(1000 / fps), loop=0)
        sg.popup_quick_message(f'Animation saved to:\n{save_path}')
        return save_path
    except ImportError:
        pass
    except Exception as e:
        sg.popup_error(f'GIF export via Pillow failed:\n{e}\n'
                       f'Falling back to individual PNG frames...')

    # -- 3) Last resort: individual PNG frames + filmstrip -----------------
    sg.popup_quick_message(
        'No video library found on this system.\n'
        'Install "imageio" (for .mp4) or "Pillow" (for an animated .gif) '
        'to export a real video next time.\n'
        'Exporting individual PNG frames instead...')
    return _export_geo_generations(canvas_key, render_mode)


def _show_geo_popup(canvas_key):
    """
    Opens the current 3D aerodynamic body in a large, dedicated, resizable
    popup window with native click-drag rotation, a Matplotlib navigation
    toolbar, and a dedicated control row: Undo / Redo (steps back and
    forth through this tab's geometry history -- for the Optimizer tab
    that's generation-by-generation; for Prediction/Flight Envelope it's
    each run/refresh), Zoom In / Zoom Out, a Wireframe/Solid toggle, and
    Save (exports the current view as a PNG). This is the "full screen"
    3D view reachable by double-clicking a preview panel or its
    'POPUP 3D' button.
    """
    fig = _geo_figs.get(canvas_key)
    vis_params = getattr(fig, '_geo_vis_params', None) if fig else None
    title = getattr(fig, '_geo_title', None) if fig else None

    if not vis_params:
        sg.popup_quick_message('Nothing to show yet -- run this tab first.')
        return

    _mpl_style()
    win_title = title or 'Aerodynamic Body -- 3D View'

    # Make sure this tab has at least one history entry (the current
    # view) so Undo/Redo are always well-defined once the popup is open.
    hist = _geo_history.get(canvas_key)
    if not hist:
        _fallback_kwargs = getattr(fig, '_geo_render_kwargs', {}) or {}
        hist = [{'vis_params': vis_params, 'label': title or 'Current',
                'overlay': _fallback_kwargs.get('overlay_text')}]
        _geo_history[canvas_key] = hist
        _geo_hist_idx[canvas_key] = 0

    render_mode = _geo_render_mode.get(canvas_key, 'solid')
    zoom_level = [1.0]  # mutable box so inner closures can update it

    # Arrow-key rotation state (replaces the old mouse-drag / idle
    # Auto-Rotate turntable by request): Left/Right nudge azim_state,
    # Up/Down nudge elev_state, and every redraw passes both into
    # render_geometry_on_figure's azim_offset/elev_offset. Rotation now
    # only ever changes in response to an explicit key press -- there is
    # no idle timer advancing these any more.
    azim_state = [0.0]
    elev_state = [0.0]
    ARROW_STEP_DEG = 4.0

    layout = [
        [sg.Text(win_title.upper(), font=(FONT_FAMILY, 15, 'bold'),
                 text_color=C_CYAN, background_color=C_PANEL, pad=(10, 4))],
        [sg.Text('', key='POP_GEO_STEP', font=(FONT_FAMILY, 11, 'italic'),
                 text_color=C_DIM, background_color=C_PANEL, pad=(10, 2))],
        [sg.ProgressBar(max_value=100, orientation='h', size=(60, 14),
                        key='POP_GEO_PROGRESS', bar_color=(C_CYAN, C_BDR),
                        border_width=0, pad=(10, 2))],
        [sg.Canvas(key='POP_CANVAS_GEO', size=(1300, 700),
                   background_color=C_BG, expand_x=True, expand_y=True,
                   border_width=0, pad=(0, 0))],
        [sg.Text('Arrow keys: rotate (\u2190/\u2192 spin, \u2191/\u2193 tilt)    |    '
                 'Toolbar: pan / zoom',
                 font=(FONT_FAMILY, 10, 'italic'), text_color=C_DIM,
                 background_color=C_PANEL, pad=(10, 2))],
        [sg.Push(background_color=C_PANEL),
         sg.Button('\u21B6  Undo', key='POP_GEO_UNDO', size=(10, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#4B5563'),
                   mouseover_colors=('#FFFFFF', '#6B7280'),
                   border_width=0, pad=(4, 8)),
         sg.Button('Redo  \u21B7', key='POP_GEO_REDO', size=(10, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#4B5563'),
                   mouseover_colors=('#FFFFFF', '#6B7280'),
                   border_width=0, pad=(4, 8)),
         sg.VSeparator(color=C_BDR),
         sg.Button('Zoom \u2212', key='POP_GEO_ZOOM_OUT', size=(9, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#1D4ED8'),
                   mouseover_colors=('#FFFFFF', '#2563EB'),
                   border_width=0, pad=(4, 8)),
         sg.Button('Zoom +', key='POP_GEO_ZOOM_IN', size=(9, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#1D4ED8'),
                   mouseover_colors=('#FFFFFF', '#2563EB'),
                   border_width=0, pad=(4, 8)),
         sg.VSeparator(color=C_BDR),
         sg.Button('Wireframe' if render_mode == 'solid' else 'Solid',
                   key='POP_GEO_MODE', size=(11, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#0F766E'),
                   mouseover_colors=('#FFFFFF', '#0D9488'),
                   border_width=0, pad=(4, 8)),
         sg.VSeparator(color=C_BDR),
         sg.Button('Save', key='POP_GEO_SAVE', size=(9, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#047857'),
                   mouseover_colors=('#FFFFFF', '#059669'),
                   border_width=0, pad=(4, 8)),
         sg.Text('', key='POP_GEO_RESULTS', font=(FONT_FAMILY, 10, 'bold'),
                 text_color=C_CYAN, background_color=C_PANEL,
                 size=(40, 3), pad=(10, 4), justification='left'),
         sg.Button('\u2B07 Export Video', key='POP_GEO_EXPORT', size=(17, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#7C3AED'),
                   mouseover_colors=('#FFFFFF', '#8B5CF6'),
                   border_width=0, pad=(4, 8)),
         sg.Button('Close', key='POP_GEO_CLOSE', size=(10, 1), font=F_LBL,
                   button_color=('#FFFFFF', '#4B5563'),
                   mouseover_colors=('#FFFFFF', '#6B7280'),
                   border_width=0, pad=(4, 8)),
         sg.Push(background_color=C_PANEL)],
    ]

    pop_window = sg.Window(win_title, layout, size=(1380, 900),
                           background_color=C_PANEL, modal=True,
                           resizable=True, finalize=True)

    cv = pop_window['POP_CANVAS_GEO'].TKCanvas
    # Kill the default Tk canvas border/highlight -- this is what used to
    # read as a thin colored frame around the 3D view. bd=0 removes the
    # 3D-look bevel, highlightthickness=0 removes the focus-highlight
    # ring, and matching highlightbackground avoids a 1px seam if either
    # ever gets re-enabled by a theme change.
    try:
        cv.configure(bd=0, highlightthickness=0, highlightbackground=C_BG)
    except Exception:
        pass

    fig2 = Figure(figsize=(13.0, 7.2))
    fig2.patch.set_facecolor(C_BG)

    toolbar_frame = tk.Frame(cv, background=C_BG, bd=0, highlightthickness=0)
    toolbar_frame.pack(side='bottom', fill='x')

    agg2 = FigureCanvasTkAgg(fig2, master=cv)
    agg2.get_tk_widget().configure(bd=0, highlightthickness=0)
    agg2.get_tk_widget().pack(side='top', fill='both', expand=True)

    try:
        toolbar = NavigationToolbar2Tk(agg2, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')
    except Exception:
        pass

    # Arrow-key rotation bindings (replaces mouse-drag rotation). Bind
    # both the plain arrow keys and their numpad ("KP_") equivalents so
    # keypad arrows work the same as the main arrow keys.
    try:
        for tk_key, ev_name in (('<Left>',     'POP_GEO_ROT_LEFT'),
                                ('<Right>',    'POP_GEO_ROT_RIGHT'),
                                ('<Up>',       'POP_GEO_ROT_UP'),
                                ('<Down>',     'POP_GEO_ROT_DOWN'),
                                ('<KP_Left>',  'POP_GEO_ROT_LEFT'),
                                ('<KP_Right>', 'POP_GEO_ROT_RIGHT'),
                                ('<KP_Up>',    'POP_GEO_ROT_UP'),
                                ('<KP_Down>',  'POP_GEO_ROT_DOWN')):
            pop_window.bind(tk_key, ev_name)
    except Exception:
        pass

    def _current():
        h = _geo_history.get(canvas_key, hist)
        i = _geo_hist_idx.get(canvas_key, len(h) - 1)
        i = max(0, min(i, len(h) - 1))
        return h, i

    def _update_results_text(text):
        """
        Updates the results readout beside the Save button (POP_GEO_RESULTS)
        with `text` (may be None/empty to clear it). This is the ONLY place
        result text (fitness / CL / CD / XCP / generation info) is shown for
        this popup -- render_geometry_on_figure() is always called with
        show_overlay=False here, so nothing is baked onto the 3D plot itself
        any more. Keeps the "Save" button's exported PNG a clean, unlabeled
        view of the body.
        """
        try:
            pop_window['POP_GEO_RESULTS'].update(text or '')
        except Exception:
            pass

    def _render_step(reset_zoom=False):
        h, i = _current()
        entry = h[i]
        try:
            aero_body_vis.render_geometry_on_figure(
                fig2, entry['vis_params'], title=win_title, large=True,
                bg_color='#EEF1F5', render_mode=render_mode,
                azim_offset=azim_state[0], elev_offset=elev_state[0],
                show_overlay=False, color_by_progress=False)
        except Exception as e:
            sg.popup_error(f'Could not render 3D popup:\n{e}')
            return
        # Rotation is keyboard-driven (arrow keys) now, not mouse-drag --
        # explicitly disable Axes3D's built-in click-drag rotation so the
        # two controls never fight each other.
        for ax in fig2.axes:
            if hasattr(ax, 'disable_mouse_rotation'):
                try:
                    ax.disable_mouse_rotation()
                except Exception:
                    pass
        if reset_zoom:
            zoom_level[0] = 1.0
        elif zoom_level[0] != 1.0:
            for ax in fig2.axes:
                _zoom_ax(ax, zoom_level[0])
        agg2.draw()
        mode_txt = 'Wireframe' if render_mode == 'wireframe' else 'Solid'
        # Results text (fitness/CL/CD/XCP for this history step, if any)
        # now lives beside the Save button rather than on the plot itself.
        _update_results_text(entry.get('overlay'))
        # Not a live tick -- the progress bar only tracks an in-progress
        # run (see _render_live), so it's reset while scrubbing history.
        try:
            pop_window['POP_GEO_PROGRESS'].update_bar(0)
        except Exception:
            pass
        try:
            pop_window['POP_GEO_STEP'].update(
                f'Step {i + 1} / {len(h)}   \u2014   {entry["label"]}   '
                f'|   View: {mode_txt}')
            pop_window['POP_GEO_UNDO'].update(disabled=(i <= 0))
            pop_window['POP_GEO_REDO'].update(disabled=(i >= len(h) - 1))
        except Exception:
            # Window may have been torn down between the loop's
            # close-check and this update -- never let a stale-widget
            # error propagate out of a render helper.
            pass

    def _render_rotate_frame():
        """
        Lightweight redraw fired ONLY in direct response to an arrow-key
        press (see the POP_GEO_ROT_* event handlers below) -- there is no
        idle timer driving this any more, so the body only ever rotates
        when the user actually presses a key. Always uses fast_preview=True
        (low-res mesh, reuses the existing Axes3D via ax.cla() instead of
        fig.clf() + rebuilding a brand-new 3D axis from scratch) so each
        key-press redraw stays snappy.

        This is also the fix for the old "Trying to read a closed window"
        crash: calling the FULL-resolution _render_step() on every single
        rotate frame rebuilds the whole figure (fig.clf() + add_subplot +
        a ~50x50-point body mesh + several 30x20 tail-fin meshes) from
        scratch, which is far too slow to do on every key repeat. Also
        skips the Undo/Redo/step-label widget updates (those only matter
        when the step actually changes, not every rotation frame) to cut
        needless Tk work further.
        """
        h, i = _current()
        entry = h[i]
        try:
            aero_body_vis.render_geometry_on_figure(
                fig2, entry['vis_params'], title=win_title, large=True,
                bg_color='#EEF1F5', render_mode=render_mode,
                fast_preview=True, azim_offset=azim_state[0],
                elev_offset=elev_state[0], show_overlay=False,
                color_by_progress=False)
        except Exception:
            return
        if zoom_level[0] != 1.0:
            for ax in fig2.axes:
                _zoom_ax(ax, zoom_level[0])
        try:
            agg2.draw()
        except Exception:
            pass

    def _render_live():
        """
        Redraws fig2 straight from the source canvas_key's live cached
        Figure (kept fresh by every render_geometry_panel() call, on the
        main window's side, whether or not this popup is open) -- this
        is what makes the popup show the aero body actually morphing
        generation-by-generation / step-by-step while an optimizer run
        or envelope sweep is in progress, rather than a static shot.
        Only fires while the user is looking at the latest ("live")
        history step; scrubbed-back (Undo'd) views are left alone so a
        live update never yanks the view out from under a manual review.

        The body itself is drawn with NO fitness/generation tinting
        (color_by_progress=False, gen_marker/fitness_t stripped before
        the render call) -- progress is communicated entirely through
        the step text, progress bar, and results readout beside Save,
        not by recoloring the airframe. `kwargs` still carries the raw
        gen_marker/fitness_t dicts (from render_geometry_panel's cache)
        purely so this function can read generation/fitness numbers out
        of them for that text + bar.
        """
        src_fig = _geo_figs.get(canvas_key)
        vis_params = getattr(src_fig, '_geo_vis_params', None) if src_fig else None
        if not vis_params:
            return
        kwargs = getattr(src_fig, '_geo_render_kwargs', {}) or {}
        # Results text (fitness / CL / CD / XCP / generation) beside Save.
        live_overlay = kwargs.get('overlay_text')
        # Progress info (generation-by-generation OR sweep-step-by-step)
        # for the step line + progress bar -- see _handle_live_event's
        # OPT_GEN / ENV_STEP handlers, which pass this through
        # render_geometry_panel(progress=...).
        prog = getattr(src_fig, '_geo_progress', None)
        # Render params, minus anything that would tint the body itself.
        render_kwargs = dict(kwargs)
        render_kwargs.pop('gen_marker', None)
        render_kwargs.pop('fitness_t', None)
        try:
            aero_body_vis.render_geometry_on_figure(
                fig2, vis_params, title=win_title, large=True,
                bg_color='#EEF1F5', render_mode=render_mode,
                fast_preview=True, azim_offset=azim_state[0],
                elev_offset=elev_state[0], **render_kwargs,
                show_overlay=False, color_by_progress=False)
        except Exception:
            return
        for ax in fig2.axes:
            if hasattr(ax, 'disable_mouse_rotation'):
                try:
                    ax.disable_mouse_rotation()
                except Exception:
                    pass
        if zoom_level[0] != 1.0:
            for ax in fig2.axes:
                _zoom_ax(ax, zoom_level[0])
        agg2.draw()
        _update_results_text(live_overlay)
        mode_txt = 'Wireframe' if render_mode == 'wireframe' else 'Solid'
        try:
            if prog:
                cur = int(prog.get('current', 0))
                tot = max(int(prog.get('total', 1)), 1)
                pct = max(0, min(100, round(cur / tot * 100)))
                kind = prog.get('kind', 'Generation')
                sub  = prog.get('label', '')
                pop_window['POP_GEO_STEP'].update(
                    f'{kind} {cur} / {tot}   |   {sub}   |   View: {mode_txt}')
                pop_window['POP_GEO_PROGRESS'].update_bar(pct)
            else:
                pop_window['POP_GEO_STEP'].update(
                    'LIVE   \u2014   updating as the run progresses...   '
                    f'|   View: {mode_txt}')
                pop_window['POP_GEO_PROGRESS'].update_bar(0)
        except Exception:
            # Window may have been torn down between the loop's
            # close-check and this update -- never let a stale-widget
            # error propagate out of a render helper.
            pass

    _render_step(reset_zoom=True)
    _last_live_seq = _geo_live_seq.get(canvas_key, 0)

    while True:
        # Defensive guard: if the underlying Tk window has already been
        # torn down for any reason, stop reading instead of calling
        # pop_window.read() again -- repeatedly reading a dead window is
        # exactly what triggers PySimpleGUI's "Trying to read a closed
        # window" safety-net popup after 100 failed attempts.
        if getattr(pop_window, 'TKroot', None) is None:
            break

        # Drain (non-blocking) and process any live events queued for
        # the MAIN window while this popup's own modal loop has control.
        # Without this, the main loop -- previously the only place that
        # ever called _handle_live_event() -- never runs for as long as
        # this popup is open: 'OPT_GEN'/'ENV_STEP' events pile up
        # unprocessed, render_geometry_panel() (and the _geo_live_seq
        # bump it does) never fires, and this "live" popup would just
        # sit frozen on its very first frame -- generations/steps never
        # visibly play out, colors never change -- until the popup is
        # closed and the backlog finally drains all at once. Draining
        # here, every tick of this loop, is what makes the popup
        # actually live. Non-blocking (timeout=0) so it never stalls
        # this loop's own ~150ms redraw cadence.
        while True:
            try:
                win_event, win_values = window.read(timeout=0)
            except Exception:
                break
            if win_event == sg.TIMEOUT_EVENT or win_event is None:
                break
            if win_event in _LIVE_EVENTS:
                try:
                    _handle_live_event(win_event, win_values)
                except Exception:
                    # A single bad live frame should never take the
                    # popup (or the run underneath it) down.
                    pass
            else:
                # Anything else (a click on the main window behind the
                # popup, etc.) is queued for the main loop to handle
                # once this popup closes -- mirrors the main loop's own
                # coalescing behaviour, so nothing is ever dropped.
                _pending_events.append((win_event, win_values))

        try:
            pop_event, _ = pop_window.read(timeout=150)
        except Exception:
            break

        # IMPORTANT: the close check MUST run before the timeout check.
        # In this PySimpleGUI build, sg.WIN_CLOSED is aliased to None,
        # which is the SAME value used to detect "no event this tick"
        # below. If the timeout branch were checked first, a real close
        # event (event=None) would be misread as just another idle tick:
        # the code would try to redraw/update widgets on the now-dead Tk
        # window and loop back to pop_window.read() again -- and again,
        # and again -- which is exactly what produces PySimpleGUI's
        # "Trying to read a closed window ... tried 100 times" popup.
        if pop_event in (sg.WIN_CLOSED, 'POP_GEO_CLOSE', None):
            break

        if pop_event == sg.TIMEOUT_EVENT:
            h, i = _current()
            live_seq = _geo_live_seq.get(canvas_key, 0)
            is_live_now = (i == len(h) - 1)
            try:
                # Only redraw when a NEW generation/step has actually
                # landed -- no idle spinning here any more. This is what
                # makes the body change strictly in response to results
                # being generated, never on its own.
                if is_live_now and live_seq != _last_live_seq:
                    _last_live_seq = live_seq
                    _render_live()
            except Exception:
                # A single bad frame should never take the whole popup
                # down -- just skip this tick and keep the loop alive.
                pass
            continue

        elif pop_event in ('POP_GEO_ROT_LEFT', 'POP_GEO_ROT_RIGHT',
                           'POP_GEO_ROT_UP', 'POP_GEO_ROT_DOWN'):
            # Arrow-key rotation: Left/Right spin azimuth, Up/Down tilt
            # elevation. Single discrete nudge per key press (and per
            # OS key-repeat event while held down) -- no mouse dragging.
            if pop_event == 'POP_GEO_ROT_LEFT':
                azim_state[0] = (azim_state[0] - ARROW_STEP_DEG) % 360.0
            elif pop_event == 'POP_GEO_ROT_RIGHT':
                azim_state[0] = (azim_state[0] + ARROW_STEP_DEG) % 360.0
            elif pop_event == 'POP_GEO_ROT_UP':
                elev_state[0] = max(-89.0, min(89.0, elev_state[0] + ARROW_STEP_DEG))
            elif pop_event == 'POP_GEO_ROT_DOWN':
                elev_state[0] = max(-89.0, min(89.0, elev_state[0] - ARROW_STEP_DEG))
            try:
                _render_rotate_frame()
            except Exception:
                pass

        elif pop_event == 'POP_GEO_UNDO':
            h, i = _current()
            if i > 0:
                _geo_hist_idx[canvas_key] = i - 1
                _render_step()

        elif pop_event == 'POP_GEO_REDO':
            h, i = _current()
            if i < len(h) - 1:
                _geo_hist_idx[canvas_key] = i + 1
                _render_step()

        elif pop_event == 'POP_GEO_ZOOM_IN':
            zoom_level[0] = max(0.15, zoom_level[0] * 0.85)
            for ax in fig2.axes:
                _zoom_ax(ax, 0.85)
            agg2.draw()

        elif pop_event == 'POP_GEO_ZOOM_OUT':
            zoom_level[0] = min(6.0, zoom_level[0] / 0.85)
            for ax in fig2.axes:
                _zoom_ax(ax, 1 / 0.85)
            agg2.draw()

        elif pop_event == 'POP_GEO_MODE':
            render_mode = 'wireframe' if render_mode == 'solid' else 'solid'
            _geo_render_mode[canvas_key] = render_mode
            pop_window['POP_GEO_MODE'].update(
                'Solid' if render_mode == 'wireframe' else 'Wireframe')
            _render_step()

        elif pop_event == 'POP_GEO_SAVE':
            try:
                save_path = sg.popup_get_file(
                    'Save 3D view as...', save_as=True, no_window=True,
                    default_extension='.png',
                    file_types=(('PNG Image', '*.png'), ('All Files', '*.*')))
            except Exception:
                save_path = None
            if save_path:
                try:
                    fig2.savefig(save_path, dpi=200,
                                 facecolor=fig2.get_facecolor())
                    sg.popup_quick_message(f'Saved to {save_path}')
                except Exception as e:
                    sg.popup_error(f'Could not save image:\n{e}')

        elif pop_event == 'POP_GEO_EXPORT':
            try:
                _export_geo_video(canvas_key, render_mode)
            except Exception as e:
                sg.popup_error(f'Could not export video:\n{e}')

    plt.close(fig2)
    pop_window.close()


def _open_zoom(fig):
    _mpl_style()
    fig2 = plt.figure(figsize=(14, 6))
    n_ax = len(fig.axes)
    for i, ax_src in enumerate(fig.axes):
        ax2 = fig2.add_subplot(1, n_ax, i + 1)
        for line in ax_src.get_lines():
            ax2.plot(line.get_xdata(), line.get_ydata(),
                     color=line.get_color(), lw=line.get_linewidth() + 0.5,
                     marker=line.get_marker(),
                     ms=line.get_markersize() + 2,
                     label=line.get_label())
        for coll in ax_src.collections:
            try:
                offs = coll.get_offsets()
                if len(offs):
                    fc = coll.get_facecolor()
                    ax2.scatter(offs[:, 0], offs[:, 1],
                                color=fc[0], s=40, alpha=0.8)
            except Exception:
                pass
        ax2.set_title(ax_src.get_title(), color=C_CYAN, fontsize=11)
        ax2.set_xlabel(ax_src.get_xlabel(), color=C_DIM)
        ax2.set_ylabel(ax_src.get_ylabel(), color=C_DIM)
        ax2.set_facecolor(C_INP)
        ax2.grid(True, lw=0.5, ls='--', color=C_BDR)
        if any(ln.get_label() and not ln.get_label().startswith('_')
               for ln in ax_src.get_lines()):
            ax2.legend()
    fig2.patch.set_facecolor(C_BG)
    fig2.suptitle('ZOOM  (close window to return)',
                  color=C_AMBER, fontsize=11)
    fig2.tight_layout()
    plt.show(block=False)


def _save_fig(fig, out_dir, filename):
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename),
                    dpi=130, bbox_inches='tight', facecolor=C_BG)
    except Exception as ex:
        print(f'[WARN] save {filename}: {ex}')


# =========================================================
# =========================================================
# PLOT CONSOLE SHOW HELPERS
# =========================================================
def _create_opt_fig(idx):
    _mpl_style()
    fig = Figure(figsize=(9, 4))
    fig.patch.set_facecolor(C_BG)
    if not _opt_history:
        return fig
    gens = [h['generation'] for h in _opt_history]
    if idx == 0:
        ax = fig.add_subplot(1, 1, 1)
        bfv = [h['fitness'] for h in _opt_history]
        afv = [h['avg_fitness'] for h in _opt_history]
        ax.plot(gens, bfv, '-o', color=C_GREEN, lw=2, ms=4, label='Best fitness')
        ax.plot(gens, afv, '-s', color=C_AMBER, lw=1.5, ms=3, label='Avg fitness', alpha=0.8)
        ax.fill_between(gens, bfv, alpha=0.08, color=C_GREEN)
        ax.axhline(bfv[-1], color=C_CYAN, lw=1, ls='--', label=f'Final: {bfv[-1]:.4f}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title(f'DE Fitness per Generation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(True)
    elif idx == 1:
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gens, [h['CL'] for h in _opt_history], '-o', color='steelblue', lw=2, ms=4, label='CL')
        ax.plot(gens, [h['CD'] for h in _opt_history], '-s', color='crimson', lw=2, ms=4, label='CD')
        ax.plot(gens, [h['CLCD'] for h in _opt_history], '-^', color='darkgreen', lw=2, ms=4, label='CL/CD')
        ax.plot(gens, [h['XCP'] for h in _opt_history], '-D', color='purple', lw=2, ms=4, label='XCP')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Value')
        ax.set_title('Aerodynamic Metrics per Generation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(True)
    elif idx == 2:
        ax = fig.add_subplot(1, 1, 1)
        perf_df = getattr(_opt_result, 'perf_df', None)
        best_x = _opt_result.x
        best_prm = {p: float(v) for p, v in zip(PARAMS, best_x)}
        best_r = aerodynamic_prediction(best_prm)
        cl = best_r['CL']
        cd = best_r['CD']
        xcp = best_r['XCP']
        ld = cl / cd if abs(cd) > 1e-9 else 0.0
        if perf_df is not None and 'CL/CD_pred' in perf_df.columns:
            ax.scatter(perf_df['CL/CD_pred'], perf_df['XCP_pred'], c='steelblue', alpha=0.7, edgecolor='k', s=28)
            ax.set_xlabel('CL/CD')
            ax.set_ylabel('XCP')
            ax.set_title('Optimised Geometry Performance')
        else:
            ax.scatter([h['CLCD'] for h in _opt_history], [h['XCP'] for h in _opt_history], c='steelblue', alpha=0.7,
                       edgecolor='k', s=28)
            ax.set_xlabel('CL/CD (best/gen)')
            ax.set_ylabel('XCP (best/gen)')
            ax.set_title('CL/CD vs XCP per generation')
        ax.scatter([ld], [xcp], c=C_AMBER, s=100, zorder=5, edgecolor='white', lw=1.5, label=f'Optimal  XCP={xcp:.3f}')
        ax.axvline(ld, color=C_CYAN, lw=0.8, ls='--', alpha=0.6)
        ax.axhline(xcp, color=C_CYAN, lw=0.8, ls='--', alpha=0.6)
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    return fig


def _create_env_fig(idx):
    if not _env_ar or not _env_mr or not _env_lr:
        fig = Figure(figsize=(11, 3.6))
        fig.patch.set_facecolor(C_BG)
        return fig
    if idx == 0:
        return _make_sweep_fig(_env_ar, 'alpha', 'Alpha (deg)', f'Alpha Sweep [{_env_geom_label}]',
                               _ALPHA_CFG,
                               pairs=[('CL', 'CL vs Alpha', C_BLUE),
                                      ('CD', 'CD vs Alpha', C_RED),
                                      ('XCP', 'XCP vs Alpha', C_AMBER)],
                               ylabels=['CL', 'CD', 'XCP'])
    elif idx == 1:
        return _make_sweep_fig(_env_mr, 'mach', 'Mach', f'Mach Sweep [{_env_geom_label}]',
                               _MACH_CFG)
    elif idx == 2:
        return _make_sweep_fig(_env_lr, 'alt', 'Altitude', f'Altitude Sweep [{_env_geom_label}]',
                               _ALT_CFG)
    fig = Figure(figsize=(11, 3.6))
    fig.patch.set_facecolor(C_BG)
    return fig


def _show_opt_plot_popup():
    global _opt_figs
    if not _opt_figs:
        sg.popup_quick_message("No optimization plots available. Run the optimizer first.")
        return

    layout = [
        [sg.Text('OPTIMIZATION EVOLUTION PLOTS', font=(FONT_FAMILY, 13, 'bold'), text_color=C_CYAN,
                 background_color=C_PANEL, pad=(6, 6))],
        [sg.Text('', key='POP_OPT_LBL', font=F_PLTLBL, text_color=C_CYAN, background_color=C_PANEL, size=(45, 1),
                 pad=(6, 2))],
        [sg.Canvas(key='POP_CANVAS_OPT', size=(900, 400), background_color=C_BG, expand_x=True, expand_y=True,
                   pad=(4, 4))],
        [
            sg.Text('', expand_x=True, background_color=C_PANEL),
            sg.Button('<', key='POP_OPT_PREV', size=(5, 1), font=F_ARROW, button_color=('#FFFFFF', '#1E3A5F'),
                      mouseover_colors=('#FFFFFF', '#4F46E5'), border_width=0, pad=(4, 4)),
            sg.Text('', size=(2, 1), background_color=C_PANEL),
            sg.Button('Close', key='POP_OPT_CLOSE', size=(10, 1), font=F_LBL, button_color=('#FFFFFF', '#4B5563'),
                      mouseover_colors=('#FFFFFF', '#6B7280'), border_width=0, pad=(4, 4)),
            sg.Text('', size=(2, 1), background_color=C_PANEL),
            sg.Button('>', key='POP_OPT_NEXT', size=(5, 1), font=F_ARROW, button_color=('#FFFFFF', '#1E3A5F'),
                      mouseover_colors=('#FFFFFF', '#4F46E5'), border_width=0, pad=(4, 4)),
            sg.Text('', expand_x=True, background_color=C_PANEL),
        ]
    ]

    pop_window = sg.Window('Optimization Evolution Plots', layout,
                           size=(950, 560),
                           background_color=C_PANEL, modal=True,
                           resizable=True, finalize=True)

    _pop_idx = 0
    _pop_agg = None

    def draw_pop_plot(idx):
        nonlocal _pop_idx, _pop_agg
        idx = max(0, min(idx, 2))
        _pop_idx = idx
        cv = pop_window['POP_CANVAS_OPT'].TKCanvas
        try:
            cv.pack_propagate(False)
            cv.grid_propagate(False)
        except Exception:
            pass
        try:
            cv.unbind('<Configure>')
        except Exception:
            pass
        for ch in cv.winfo_children():
            ch.destroy()

        fig = _create_opt_fig(idx)
        _pop_agg = FigureCanvasTkAgg(fig, master=cv)
        _pop_agg.draw()
        _pop_agg.get_tk_widget().pack(side='top', fill='both', expand=True)
        _pop_agg.mpl_connect('button_press_event', lambda e: (_open_zoom(fig) if e.dblclick else None))

        title = _OPT_TITLES[idx] if idx < len(_OPT_TITLES) else f'Plot {idx + 1}'
        pop_window['POP_OPT_LBL'].update(f'{idx + 1} / 3  —  {title}')

    draw_pop_plot(0)

    while True:
        event, values = pop_window.read()
        if event in (sg.WIN_CLOSED, 'POP_OPT_CLOSE'):
            break
        elif event == 'POP_OPT_PREV':
            draw_pop_plot(_pop_idx - 1)
        elif event == 'POP_OPT_NEXT':
            draw_pop_plot(_pop_idx + 1)

    pop_window.close()
    import gc
    gc.collect()


def _show_env_plot_popup():
    global _env_figs
    if not _env_figs:
        sg.popup_quick_message("No flight envelope plots available. Run the sweeps first.")
        return

    layout = [
        [sg.Text('FLIGHT ENVELOPE SWEEP PLOTS', font=(FONT_FAMILY, 13, 'bold'), text_color=C_CYAN,
                 background_color=C_PANEL, pad=(6, 6))],
        [sg.Text('', key='POP_ENV_LBL', font=F_PLTLBL, text_color=C_CYAN, background_color=C_PANEL, size=(45, 1),
                 pad=(6, 2))],
        [sg.Canvas(key='POP_CANVAS_ENV', size=(1100, 400), background_color=C_BG, expand_x=True, expand_y=True,
                   pad=(4, 4))],
        [
            sg.Text('', expand_x=True, background_color=C_PANEL),
            sg.Button('<', key='POP_ENV_PREV', size=(5, 1), font=F_ARROW, button_color=('#FFFFFF', '#1E3A5F'),
                      mouseover_colors=('#FFFFFF', '#4F46E5'), border_width=0, pad=(4, 4)),
            sg.Text('', size=(2, 1), background_color=C_PANEL),
            sg.Button('Close', key='POP_ENV_CLOSE', size=(10, 1), font=F_LBL, button_color=('#FFFFFF', '#4B5563'),
                      mouseover_colors=('#FFFFFF', '#6B7280'), border_width=0, pad=(4, 4)),
            sg.Text('', size=(2, 1), background_color=C_PANEL),
            sg.Button('>', key='POP_ENV_NEXT', size=(5, 1), font=F_ARROW, button_color=('#FFFFFF', '#1E3A5F'),
                      mouseover_colors=('#FFFFFF', '#4F46E5'), border_width=0, pad=(4, 4)),
            sg.Text('', expand_x=True, background_color=C_PANEL),
        ]
    ]

    pop_window = sg.Window('Flight Envelope Sweep Plots', layout,
                           size=(1150, 560),
                           background_color=C_PANEL, modal=True,
                           resizable=True, finalize=True)

    _pop_idx = 0
    _pop_agg = None

    def draw_pop_plot(idx):
        nonlocal _pop_idx, _pop_agg
        idx = max(0, min(idx, 2))
        _pop_idx = idx
        cv = pop_window['POP_CANVAS_ENV'].TKCanvas
        try:
            cv.pack_propagate(False)
            cv.grid_propagate(False)
        except Exception:
            pass
        try:
            cv.unbind('<Configure>')
        except Exception:
            pass
        for ch in cv.winfo_children():
            ch.destroy()

        fig = _create_env_fig(idx)
        _pop_agg = FigureCanvasTkAgg(fig, master=cv)
        _pop_agg.draw()
        _pop_agg.get_tk_widget().pack(side='top', fill='both', expand=True)
        _pop_agg.mpl_connect('button_press_event', lambda e: (_open_zoom(fig) if e.dblclick else None))

        title = _ENV_TITLES[idx] if idx < len(_ENV_TITLES) else f'Plot {idx + 1}'
        pop_window['POP_ENV_LBL'].update(f'{idx + 1} / 3  —  {title}')

    draw_pop_plot(0)

    while True:
        event, values = pop_window.read()
        if event in (sg.WIN_CLOSED, 'POP_ENV_CLOSE'):
            break
        elif event == 'POP_ENV_PREV':
            draw_pop_plot(_pop_idx - 1)
        elif event == 'POP_ENV_NEXT':
            draw_pop_plot(_pop_idx + 1)

    pop_window.close()
    import gc
    gc.collect()


# =========================================================
# PLOT CONSOLE WIDGET BUILDER
# =========================================================
def _plot_console(canvas_key, lbl_key, prev_key, next_key,
                  hdr_text, hdr_color, canvas_h=300):
    return [
        # Header
        *sec_hdr_rows(hdr_text, hdr_color),

        # Plot label
        [sg.Text('', key=lbl_key, font=F_PLTLBL,
                 text_color=hdr_color, background_color=C_PANEL,
                 size=(45, 1), pad=(6, 2))],

        # Canvas where the matplotlib figure is embedded
        [sg.Canvas(key=canvas_key, size=(750, canvas_h),
                   background_color=C_BG,
                   expand_x=True, pad=(4, 4))],

        # Navigation buttons + tooltip
        [
            sg.Text('', expand_x=True, background_color=C_PANEL),
            sg.Button('<', key=prev_key, size=(4, 1), font=F_ARROW,
                      button_color=('#FFFFFF', '#1E3A5F'),
                      mouseover_colors=('#FFFFFF', '#4F46E5'),
                      border_width=0, pad=(4, 4),
                      tooltip='Previous plot'),
            sg.Text('', size=(2, 1), background_color=C_PANEL),
            sg.Button('>', key=next_key, size=(4, 1), font=F_ARROW,
                      button_color=('#FFFFFF', '#1E3A5F'),
                      mouseover_colors=('#FFFFFF', '#4F46E5'),
                      border_width=0, pad=(4, 4),
                      tooltip='Next plot'),
            sg.Text('  Double-click plot to zoom',
                    font=('Arial', 11), text_color=C_DIM,
                    background_color=C_PANEL),
            sg.Text('', expand_x=True, background_color=C_PANEL),
        ]
    ]


# =========================================================
# FLIGHT ENVELOPE HELPERS (MOVED UP FOR LAYOUT USE)
# =========================================================
def _load_optimal_base(out_dir):
    import pandas as pd
    feat_to_param = {
        'nose length': 'nose_len', 'body_length': 'body_len',
        'wing LE': 'wing_le', 'root chord': 'root_chord',
        'tip chord': 'tip_chord', 'semi-span': 'semi_span',
        'root th': 'root_th', 'tip th': 'tip_th',
        'wing sweep': 'wing_sweep', 'tail LE': 'tail_le',
        'root chord.1': 'root_chord1', 'tip chord.1': 'tip_chord1',
        'semi-span.1': 'semi_span1', 'root th.1': 'root_th1',
        'tip th.1': 'tip_th1',
    }
    gp = os.path.join(out_dir, 'best_geometry.csv')
    if not os.path.exists(gp):
        raise FileNotFoundError(
            f"best_geometry.csv not found in '{out_dir}'.\n"
            "Run Optimizer first.")
    df = pd.read_csv(gp)
    row = df.iloc[0]
    base = dict(DEFAULTS)
    for feat, param in feat_to_param.items():
        if feat in row.index:
            base[param] = float(row[feat])
        elif param in row.index:
            base[param] = float(row[param])
    return base, gp


def _get_default_env_params():
    try:
        base, _ = _load_optimal_base('de_output')
        return base
    except Exception:
        return DEFAULTS


_env_defaults = _get_default_env_params()


def sweep_frame(title, mk, lk, sk, mv, lv, sv):
    return sg.Frame(title, [[
        sg.Text('Min', size=(4, 1), font=F_LBL,
                text_color=C_DIM, background_color=C_PANEL),
        inp(mk, mv, 7),
        sg.Text('Max', size=(4, 1), font=F_LBL,
                text_color=C_DIM, background_color=C_PANEL),
        inp(lk, lv, 7),
        sg.Text('Step', size=(4, 1), font=F_LBL,
                text_color=C_DIM, background_color=C_PANEL),
        inp(sk, sv, 7),
    ]], font=F_SEC, title_color=C_CYAN,
                    background_color=C_PANEL, border_width=1,
                    relief=sg.RELIEF_FLAT, expand_x=True, pad=(6, 5))


def geo_preview_frame(canvas_key, prefix, refresh_key, title_text, accent_color):
    """
    Builds a compact '3D GEOMETRY' strip for a tab: a row of key metrics
    plus a status line. No 3D canvas is embedded in the tab any more --
    the VISUALIZE button that used to live at the end of this strip now
    sits in the tab's main control-button row (beside RUN/ESTIMATE/
    RESET/ABORT/CLEAR -- see visualize_btn() + build_main_layout()) so
    it reads as one of the tab's primary actions rather than something
    tucked away at the bottom. Clicking it recomputes the geometry from
    this tab's current inputs and opens it directly in the large,
    dedicated, rotatable/zoomable 3D popup window (see _show_geo_popup).
    If a run (optimizer generations / envelope sweep) is currently in
    progress, the popup keeps redrawing itself live as new generations/
    steps arrive, so the body is seen actually changing shape as results
    vary while the run executes, not just as a static snapshot.

    `refresh_key` is kept for API/back-compat (some call sites still
    fire it directly, e.g. after Apply Best Geometry) but no longer has
    its own visible button.
    """
    return [styled_frame(title_text, [
        [lbl('Length', 11), out_field(f'{prefix}_LEN', 10, C_CYAN),
         lbl('Wingspan', 11), out_field(f'{prefix}_WSPAN', 10, C_BLUE),
         lbl('Tail Span', 11), out_field(f'{prefix}_TSPAN', 10, C_RED),
         lbl('L/D', 7), out_field(f'{prefix}_FIN', 8, C_AMBER)],
        [sg.Text('', key=f'{prefix}_STATUS', font=(FONT_FAMILY, 10, 'italic'),
                 text_color=C_DIM, background_color=C_PANEL, pad=(6, 3))],
    ], accent_color)]


def visualize_btn(prefix):
    """
    The VISUALIZE button for a tab, meant to be placed directly inside
    that tab's control-button row (next to RUN/ESTIMATE/RESET/ABORT/
    CLEAR), not below it. Its key ('<prefix>_POPUP') is unchanged, so
    the existing 'P_GEO_POPUP' / 'O_GEO_POPUP' / 'E_GEO_POPUP' event
    handlers keep working without any change on that side.
    """
    return action_btn(' \u2922 VISUALIZE ', f'{prefix}_POPUP',
                      bg='#0F766E', w=16)


def build_main_layout():
    global prediction_tab, optimization_tab, flight_tab, tab_group, HEADER_ROW, STATUS_ROW, layout

    _geo_col1 = ['nose_len', 'body_len', 'wing_le', 'root_chord', 'tip_chord', 'semi_span', 'root_th', 'tip_th',
                 'wing_sweep']
    _geo_col2 = ['tail_le', 'root_chord1', 'tip_chord1', 'semi_span1', 'root_th1', 'tip_th1', 'mach', 'alpha', 'alt']

    nose_wing_layout = [[lbl(LABELS[p], 18), inp(p, DEFAULTS[p], 10)] for p in _geo_col1]
    tail_layout = [[lbl(LABELS[p], 18), inp(p, DEFAULTS[p], 10)] for p in _geo_col2[:6]]
    flight_layout = [[lbl(LABELS[p], 18), inp(p, DEFAULTS[p], 10)] for p in _geo_col2[6:]]

    GEO_COL = [
        [
            sg.Column([[styled_frame('NOSE & WING GEOMETRY', nose_wing_layout, C_BLUE)]],
                      background_color=C_PANEL, pad=(0, 0), expand_x=True, expand_y=True),
            sg.Column([
                [styled_frame('TAIL GEOMETRY', tail_layout, C_GREEN)],
                [styled_frame('FLIGHT CONDITIONS', flight_layout, C_AMBER)]
            ], background_color=C_PANEL, pad=(0, 0), expand_x=True, expand_y=True)
        ]
    ]

    aero_outputs_layout = [
        [lbl('Lift Coefficient  CL', 24), out_field('CL_OUT', 18, C_GREEN)],
        [lbl('Drag Coefficient  CD', 24), out_field('CD_OUT', 18, C_RED)],
        [lbl('Centre of Pressure XCP', 24), out_field('XCP_OUT', 18, C_BLUE)],
        [lbl('XCP/D ', 24), out_field('XCPD_OUT', 18, C_PURP)],
        [lbl('Lift-to-Drag  CL/CD', 24), out_field('LD_OUT', 18, C_AMBER)],
    ]

    metrics_layout = [
        [lbl('Computation Time', 24), out_field('TIME_P', 20)],
        [lbl('Timestamp', 24), out_field('TIME_STAMP_P', 30, C_CYAN)],
        [sg.Input('', key='MET_CL', visible=False)],
        [sg.Input('', key='MET_CD', visible=False)],
        [sg.Input('', key='MET_XCP', visible=False)],
    ]

    OUT_COL = [
        [styled_frame('AERODYNAMIC OUTPUTS', aero_outputs_layout, C_CYAN)],
        [styled_frame('PERFORMANCE & EXECUTION', metrics_layout, C_GREEN)],
        [sg.Input('', key='SRC_OUT', visible=False)],
        [sg.Input('', key='MODE_OUT', visible=False)],
        [sg.Multiline('', key='PRED_CON', visible=False, disabled=True)],
        [sg.Multiline('', key='TOP5_OUT', visible=False, disabled=True)],
    ]

    _prediction_rows = [
        [sg.Column(GEO_COL, background_color=C_PANEL,
                   expand_x=True, expand_y=True,
                   scrollable=False, pad=(8, 6)),
         sg.VSeparator(color=C_BDR, pad=(3, 0)),
         sg.Column(OUT_COL, background_color=C_PANEL,
                   expand_x=True, expand_y=True,
                   scrollable=False, pad=(8, 6))],
        [sg.Column([[
            action_btn(' ESTIMATE ', 'Estimate', bg='#059669', w=22),
            action_btn(' RESET', 'Reset_Pred', bg='#1D4ED8', w=14),
            visualize_btn('P_GEO'),
            sg.Push(background_color=C_BG),
            sg.Text('Progress:', font=F_LBL,
                    text_color=C_DIM, background_color=C_BG),
            *prog_row('PB_P', 'PP_P', 'PM_P'),
        ]], background_color=C_BG, expand_x=True, pad=(6, 6))],
        [sg.HSeparator(color=C_BDR, pad=(4, 6))],
        geo_preview_frame('PRED_GEO_CANVAS', 'P_GEO', 'PRED_GEO_REFRESH',
                          '3D GEOMETRY PREVIEW (current prediction inputs)', C_CYAN),
    ]
    prediction_tab = [[
        sg.Column(_prediction_rows, background_color=C_BG,
                  expand_x=True, expand_y=True, scrollable=True,
                  vertical_scroll_only=True, pad=(0, 0))
    ]]

    bounds_col1_rows = [[
        sg.Text('Parameter', size=(18, 1), font=(FONT_FAMILY, 11, 'bold'),
                text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
        sg.Text('Min Bound', size=(10, 1), font=(FONT_FAMILY, 11, 'bold'),
                text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
        sg.Text('Max Bound', size=(10, 1), font=(FONT_FAMILY, 11, 'bold'),
                text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
    ]]
    bounds_col2_rows = [[
        sg.Text('Parameter', size=(18, 1), font=(FONT_FAMILY, 11, 'bold'),
                text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
        sg.Text('Min Bound', size=(10, 1), font=(FONT_FAMILY, 11, 'bold'),
                text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
        sg.Text('Max Bound', size=(10, 1), font=(FONT_FAMILY, 11, 'bold'),
                text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
    ]]

    for idx, p in enumerate(PARAMS[:15]):
        lo, hi = BOUNDS[p]
        row = [
            lbl(LABELS[p], 18),
            inp(f'{p}_LOW', lo, 10),
            inp(f'{p}_HIGH', hi, 10),
        ]
        if idx < 8:
            bounds_col1_rows.append(row)
        else:
            bounds_col2_rows.append(row)

    PARAMETER_SEARCH_BOUNDS_FRAME = styled_frame(
        'PARAMETER SEARCH BOUNDS (GEOMETRY ONLY)',
        [[
            sg.Column(bounds_col1_rows, background_color=C_PANEL, pad=(0, 0), expand_x=True),
            sg.Column(bounds_col2_rows, background_color=C_PANEL, pad=(0, 0), expand_x=True)
        ]],
        C_AMBER
    )

    output_constraints_layout = [
        [sg.Text('Penalty applied if any constraint is violated', font=(FONT_FAMILY, 11, 'italic'), text_color=C_DIM,
                 background_color=C_PANEL)],
        [lbl('CL  Min', 10), inp('CL_MIN', '-3.723', 8), lbl('CL  Max', 10), inp('CL_MAX', '15.2213', 8)],
        [lbl('CD  Min', 10), inp('CD_MIN', '-1.187', 8), lbl('CD  Max', 10), inp('CD_MAX', '5.7352', 8)],
        [lbl('XCP Min', 10), inp('XCP_MIN', '-12.3114', 8), lbl('XCP Max', 10), inp('XCP_MAX', '-3.5322', 8)]
    ]

    opt_settings_layout = [
        [sg.Text('Hyperparameters configuration for custom DE', font=(FONT_FAMILY, 11, 'italic'), text_color=C_DIM,
                 background_color=C_PANEL)],
        [lbl('Max Generations', 20), inp('MAXITER', '50', 10)],
        [lbl('Population Size', 20), inp('POPSIZE', '10', 10)],
        [lbl('Max Gene-Swap Steps', 20), inp('ITERMAX', '5', 10)]
    ]

    OPT_LEFT = [
        [PARAMETER_SEARCH_BOUNDS_FRAME],
        [
            sg.Column([[styled_frame('OUTPUT CONSTRAINTS', output_constraints_layout, C_RED)]],
                      background_color=C_PANEL, pad=(0, 0), expand_x=True, expand_y=True),
            sg.Column([[styled_frame('OPTIMIZATION SETTINGS', opt_settings_layout, C_BLUE)]],
                      background_color=C_PANEL, pad=(0, 0), expand_x=True, expand_y=True)
        ],
        [sg.Input('de_output', key='OUT_DIR', visible=False)]
    ]

    OPT_RIGHT = [
                    sec_hdr('OPTIMAL RESULT', C_CYAN),
                    [lbl('Best CL', 26), out_field('OPT_CL', 18, C_GREEN)],
                    [lbl('Best CD', 26), out_field('OPT_CD', 18, C_RED)],
                    [lbl('Best XCP', 26), out_field('OPT_XCP', 18, C_BLUE)],
                    [lbl('Best XCP/D', 26), out_field('OPT_XCPD', 18, C_PURP)],
                    [lbl('Max CL/CD', 26), out_field('OPT_LD', 18, C_AMBER)],
                    [lbl('Composite Fitness', 26), out_field('OPT_FIT', 18, C_CYAN)],
                    [sg.Input('', key='OPT_TIME', visible=False)],
                    [sg.Input('', key='OPT_TIME_RANGE', visible=False)],
                    [sg.Input('', key='OPT_MODE', visible=False)],
                ] + sec_hdr_rows('BEST GEOMETRY (15 geometry parameters)', C_AMBER) + [
                    [sg.Multiline('', key='OPT_GEO', size=(60, 22),
                                  font=F_TBL, background_color=C_PANEL,
                                  text_color=C_AMBER, autoscroll=False,
                                  border_width=1, expand_x=True,
                                  disabled=True, pad=(6, 4))],
                    [sg.Multiline('', key='OPT_LOG', size=(58, 4),
                                  visible=False, font=F_TBL,
                                  background_color=C_PANEL,
                                  text_color=C_GREEN, autoscroll=True,
                                  disabled=False)],
                ] + [
                    [sg.Button('  VIEW OPTIMIZATION EVOLUTION PLOTS  ', key='SHOW_OPT_PLOTS', font=F_TABTXT,
                               button_color=('#FFFFFF', C_CYAN), mouseover_colors=('#FFFFFF', '#0891B2'),
                               disabled=True, disabled_button_color=(C_DIM, '#374151'),
                               expand_x=True, pad=(6, 12))]
                ]

    _optimization_rows = [
        [sg.Column(OPT_LEFT, background_color=C_PANEL,
                   expand_x=True, expand_y=True,
                   scrollable=False, pad=(8, 6)),
         sg.VSeparator(color=C_BDR, pad=(2, 0)),
         sg.Column(OPT_RIGHT, background_color=C_PANEL,
                   expand_x=True, expand_y=True,
                   scrollable=False, pad=(8, 6))],
        [sg.Column([[
            action_btn(' RUN OPTIMIZER', 'Run_Opt', bg='#059669', w=22),
            action_btn(' ABORT', 'Abort_Opt', bg='#DC2626', w=12),
            action_btn(' CLEAR', 'Clear_Opt', bg='#1D4ED8', w=12),
            visualize_btn('O_GEO'),
            sg.Push(background_color=C_BG),
            sg.Text('Progress:', font=F_LBL,
                    text_color=C_DIM, background_color=C_BG),
            *prog_row('PB_O', 'PP_O', 'PM_O'),
        ]], background_color=C_BG, expand_x=True, pad=(6, 6))],
        [sg.HSeparator(color=C_BDR, pad=(4, 6))],
        geo_preview_frame('OPT_GEO_CANVAS', 'O_GEO', 'OPT_GEO_REFRESH',
                          '3D GEOMETRY PREVIEW (optimizer best result)', C_AMBER),
    ]
    optimization_tab = [[
        sg.Column(_optimization_rows, background_color=C_BG,
                  expand_x=True, expand_y=True, scrollable=True,
                  vertical_scroll_only=True, pad=(0, 0))
    ]]

    env_base_col1_rows = []
    env_base_col2_rows = []
    for idx, p in enumerate(PARAMS[:15]):
        row = [lbl(LABELS[p], 18), inp(f'E_{p}', _env_defaults[p], 10)]
        if idx < 8:
            env_base_col1_rows.append(row)
        else:
            env_base_col2_rows.append(row)

    env_base_frame = styled_frame(
        'BASE GEOMETRY PARAMETERS',
        [[
            sg.Column(env_base_col1_rows, background_color=C_PANEL, pad=(0, 0), expand_x=True),
            sg.Column(env_base_col2_rows, background_color=C_PANEL, pad=(0, 0), expand_x=True)
        ]],
        C_GREEN
    )

    ENV_LEFT = [
        [styled_frame('GEOMETRY SOURCE', [
            [action_btn(' APPLY BEST GEOMETRY FROM OPTIMIZER', 'APPLY_OPT_GEO', bg='#B45309', w=38)],
            [sg.Text('Loads baseline geometry from previous optimization result', font=(FONT_FAMILY, 11, 'italic'),
                     text_color=C_DIM, background_color=C_PANEL, pad=(6, 4))]
        ], C_AMBER)],
        [styled_frame('SWEEP RANGES', [
            [sweep_frame('ALPHA SWEEP (deg)',
                         'ALPHA_MIN', 'ALPHA_MAX', 'ALPHA_STP',
                         '0', '20', '2')],
            [sweep_frame('MACH NUMBER SWEEP',
                         'MACH_MIN', 'MACH_MAX', 'MACH_STP',
                         '0.2', '0.8', '0.1')],
            [sweep_frame('ALTITUDE SWEEP (m)',
                         'ALT_MIN', 'ALT_MAX', 'ALT_STP',
                         '0', '6000', '1000')]
        ], C_BLUE)],
        [env_base_frame]
    ]

    ENV_RIGHT = (
            [
                sec_hdr('ALPHA SWEEP RESULTS', C_BLUE),
                [sg.Multiline(
                    '',
                    key='ENV_ALPHA',
                    size=(68, 7),
                    font=F_TBL,
                    background_color=C_PANEL,
                    text_color=C_CYAN,
                    autoscroll=True,
                    border_width=1,
                    expand_x=True,
                    disabled=True,
                    pad=(6, 4))],

                sec_hdr('MACH SWEEP RESULTS', C_GREEN),
                [sg.Multiline(
                    '',
                    key='ENV_MACH',
                    size=(68, 7),
                    font=F_TBL,
                    background_color=C_PANEL,
                    text_color=C_GREEN,
                    autoscroll=True,
                    border_width=1,
                    expand_x=True,
                    disabled=True,
                    pad=(6, 4))],

                sec_hdr('ALTITUDE SWEEP RESULTS', C_AMBER),
                [sg.Multiline(
                    '',
                    key='ENV_ALT',
                    size=(68, 7),
                    font=F_TBL,
                    background_color=C_PANEL,
                    text_color=C_AMBER,
                    autoscroll=True,
                    border_width=1,
                    expand_x=True,
                    disabled=True,
                    pad=(6, 4))],

                sec_hdr('SUMMARY STATISTICS', C_RED),
                [sg.Multiline(
                    '',
                    key='ENV_SUM',
                    size=(68, 7),
                    font=F_TBL,
                    background_color=C_PANEL,
                    text_color=C_WHITE,
                    autoscroll=True,
                    border_width=1,
                    expand_x=True,
                    disabled=True,
                    pad=(6, 4))],

                [sg.Input('', key='ENV_TIMESTAMP', visible=False)],
            ] + [
                [sg.Button('  VIEW FLIGHT ENVELOPE SWEEP PLOTS  ', key='SHOW_ENV_PLOTS', font=F_TABTXT,
                           button_color=('#FFFFFF', C_CYAN), mouseover_colors=('#FFFFFF', '#0891B2'),
                           disabled=True, disabled_button_color=(C_DIM, '#374151'),
                           expand_x=True, pad=(6, 12))],
                [sg.Text('  All 3 plots saved as PNG files to the output directory.',
                         font=F_LBL, text_color=C_DIM,
                         background_color=C_PANEL, pad=(8, 5))]
            ]
    )

    _flight_rows = [
        [sg.Column(ENV_LEFT, background_color=C_PANEL,
                   expand_x=True, expand_y=True,
                   scrollable=False, pad=(8, 6)),
         sg.VSeparator(color=C_BDR, pad=(2, 0)),
         sg.Column(ENV_RIGHT, background_color=C_PANEL,
                   expand_x=True, expand_y=True,
                   scrollable=False, pad=(8, 6))],
        [sg.Column([[
            action_btn(' RUN ENVELOPE ', 'Run_Env', bg='#059669', w=22),
            action_btn(' ABORT ', 'Abort_Env', bg='#DC2626', w=12),
            action_btn(' CLEAR ', 'Clear_Env', bg='#1D4ED8', w=14),
            action_btn(' EXPORT CSV ', 'Export_Env', bg='#4B5563', w=14),
            visualize_btn('E_GEO'),
            sg.Push(background_color=C_BG),
            sg.Text('Progress:', font=F_LBL,
                    text_color=C_DIM, background_color=C_BG),
            *prog_row('PB_E', 'PP_E', 'PM_E'),
        ]], background_color=C_BG, expand_x=True, pad=(6, 6))],
        [sg.HSeparator(color=C_BDR, pad=(4, 6))],
        geo_preview_frame('ENV_GEO_CANVAS', 'E_GEO', 'ENV_GEO_REFRESH',
                          '3D GEOMETRY PREVIEW (envelope base geometry)', C_GREEN),
    ]
    flight_tab = [[
        sg.Column(_flight_rows, background_color=C_BG,
                  expand_x=True, expand_y=True, scrollable=True,
                  vertical_scroll_only=True, pad=(0, 0))
    ]]

    HEADER_ROW = [
        sg.Text(
            'OPTIMAL AERODYNAMIC CONFIGURATION DESIGN OF AEROSPACE VEHICLES',
            font=F_TITLE, text_color='#FFFFFF', background_color=C_HDR,
            justification='center', expand_x=True, pad=(10, 10)),
    ]

    TAB_KW = dict(expand_x=True, expand_y=True,
                  pad=(0, 0), background_color=C_BG)

    if CURRENT_THEME == 'Silver Slate':
        sel_txt = '#FFFFFF'
        sel_bg = '#1E3A8A'
        unsel_txt = '#334155'
        unsel_bg = '#E2E8F0'
        bg_grp = '#CBD5E1'
    else:
        sel_txt = '#FFFFFF'
        sel_bg = '#2563EB'
        unsel_txt = '#94A3B8'
        unsel_bg = '#111827'
        bg_grp = '#0B0F1A'

    tab_group = sg.TabGroup([[
        sg.Tab('  PREDICTION  ', prediction_tab, **TAB_KW),
        sg.Tab('  OPTIMIZER   ', optimization_tab, **TAB_KW),
        sg.Tab('  FLIGHT ENVELOPE ', flight_tab, **TAB_KW),
    ]], tab_location='topleft', font=F_TABTXT,
        selected_title_color=sel_txt,
        title_color=unsel_txt,
        selected_background_color=sel_bg,
        background_color=bg_grp,
        tab_background_color=unsel_bg,
        border_width=0,
        expand_x=True, expand_y=True,
        enable_events=True,
        key='TABS')

    STATUS_ROW = [
        sg.Text('*', font=(FONT_FAMILY, 14), text_color=C_GREEN,
                background_color=C_DARK, pad=(10, 4)),
        sg.Text('READY', key='STS', font=F_STS,
                text_color=C_AMBER, background_color=C_DARK, size=(58, 1)),
        sg.Push(background_color=C_DARK),
        sg.Text('Theme:', font=F_LBL, text_color=C_WHITE, background_color=C_DARK),
        sg.Combo(['Silver Slate', 'Classic Dark'], default_value=CURRENT_THEME, key='THEME_SELECT', enable_events=True,
                 font=F_LBL, readonly=True),
        sg.Text('  Font:', font=F_LBL, text_color=C_WHITE, background_color=C_DARK),
        sg.Combo(['Arial', 'Helvetica', 'Times New Roman'], default_value=FONT_FAMILY, key='FONT_SELECT',
                 enable_events=True, font=F_LBL, readonly=True),
        sg.Text(' ', background_color=C_DARK),
    ]

    layout = [
        [sg.Column([HEADER_ROW], background_color=C_HDR,
                   expand_x=True, pad=(0, 0))],
        [tab_group],
        [sg.Column([STATUS_ROW], background_color=C_DARK,
                   expand_x=True, pad=(0, 0))],
    ]
    return layout


def rebuild_window():
    global window
    active_tab = '  PREDICTION  '
    try:
        active_tab = window['TABS'].get()
    except Exception:
        pass

    opt_plots_disabled = True
    env_plots_disabled = True
    try:
        opt_plots_disabled = not bool(_opt_figs)
        env_plots_disabled = not bool(_env_figs)
    except Exception:
        pass

    old_window = window
    layout = build_main_layout()
    window = sg.Window(
        'DRDL Aerospace Platform', layout,
        size=(1600, 980), finalize=True,
        resizable=True, element_justification='left',
        background_color=C_BG, margins=(0, 0),
        return_keyboard_events=True,
        use_custom_titlebar=False,
    )

    # Copy values from old window
    for k in window.AllKeysDict:
        if k not in ('THEME_SELECT', 'FONT_SELECT', 'TABS'):
            try:
                val = old_window[k].get()
                window[k].update(val)
            except Exception:
                pass

    old_window.close()

    try:
        window['TABS'].select_tab(active_tab)
    except Exception:
        pass

    try:
        window['SHOW_OPT_PLOTS'].update(disabled=opt_plots_disabled)
        window['SHOW_ENV_PLOTS'].update(disabled=env_plots_disabled)
    except Exception:
        pass


layout = build_main_layout()
window = sg.Window(
    'DRDL Aerospace Platform', layout,
    size=(1600, 980), finalize=True,
    resizable=True, element_justification='left',
    background_color=C_BG, margins=(0, 0),
    return_keyboard_events=True,
    use_custom_titlebar=False,
)

# Let window open at the default size so that standard minimize/maximize/close buttons work perfectly
_is_max = False


# =========================================================
# FAST TAB-SWITCH SLIDE ANIMATION
# =========================================================
# PySimpleGUI's TabGroup is backed by a ttk.Notebook, which swaps its
# selected child page instantly and (unlike a hand-rolled Frame-based
# tab system) doesn't let you reposition/cross-fade the tab PAGES
# themselves. So instead of faking a full content slide -- which would
# mean abandoning ttk.Notebook's built-in tab bar entirely -- a thin
# accent-colored bar sweeps left-to-right across the tab strip every
# time the active tab changes, completing well under a second (see
# duration_ms below). It gives the same fast, deliberate "sliding into
# the new tab" cue the user is asking for without any visible lag.
_tab_slide_state = {'overlay': None}


def _animate_tab_slide():
    try:
        notebook = window['TABS'].Widget
    except Exception:
        return
    parent = notebook.master
    try:
        x = notebook.winfo_x()
        y = notebook.winfo_y()
        w = notebook.winfo_width()
    except Exception:
        return
    if w <= 1:
        return

    old = _tab_slide_state.get('overlay')
    if old is not None:
        try:
            old.destroy()
        except Exception:
            pass

    bar_h = 4
    bar = tk.Frame(parent, background=C_CYAN, height=bar_h)
    bar.place(x=x, y=y, width=0, height=bar_h)
    _tab_slide_state['overlay'] = bar

    steps = 10
    duration_ms = 220   # total sweep time -- comfortably "within a sec"
    step_delay = max(1, duration_ms // steps)

    def _step(i=0):
        if not bar.winfo_exists():
            return
        frac = (i + 1) / steps
        bar.place_configure(width=int(w * frac))
        if i + 1 < steps:
            parent.after(step_delay, lambda: _step(i + 1))
        else:
            def _fade_out():
                if bar.winfo_exists():
                    bar.destroy()
            parent.after(120, _fade_out)

    _step()


def _bind_tab_slide_fx():
    try:
        notebook = window['TABS'].Widget
        notebook.bind('<<NotebookTabChanged>>', lambda e: _animate_tab_slide())
    except Exception:
        pass


_bind_tab_slide_fx()


# =========================================================
# BACKGROUND MODEL PRELOADER
# =========================================================
def _preload():
    global _model_rdy
    t0 = time.perf_counter()
    aerodynamic_prediction(DEFAULTS)
    for mach in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for alpha in [0, 2, 4, 6, 8, 10]:
            p = dict(DEFAULTS, mach=mach, alpha=alpha)
            aerodynamic_prediction(p)
    elapsed = time.perf_counter() - t0
    _model_rdy = True
    window.write_event_value('MODEL_READY', elapsed)


threading.Thread(target=_preload, daemon=True).start()
set_status('Loading XGBoost model ', color=C_AMBER)
_startup_pct = 0


# =========================================================
# PREDICTION WORKER
# =========================================================
def _pred_worker(params):
    try:
        t0 = time.perf_counter()
        result = aerodynamic_prediction(params)
        elapsed = time.perf_counter() - t0
        window.write_event_value('PRED_DONE', {'ok': True, 'result': result, 'elapsed': elapsed})
    except Exception as e:
        window.write_event_value('PRED_DONE', {'ok': False, 'error': str(e)})


def render_prediction(payload):
    result = payload['result']
    elapsed = payload['elapsed']
    cl = result.get('CL', 0.0)
    cd = result.get('CD', 0.0)
    xcp = result.get('XCP', 0.0)
    xcpd = result.get('XCP_D', None)
    met = result.get('metrics', {})
    det = result.get('detailed_metrics', {})
    # mode    = result.get('mode', 'xgboost')
    ems = result.get('elapsed_ms', elapsed * 1000)
    ld = cl / cd if abs(cd) > 1e-9 else float('inf')

    window['CL_OUT'].update(f'{cl:.6f}')
    window['CD_OUT'].update(f'{cd:.6f}')
    window['XCP_OUT'].update(f'{xcp:.6f}')
    window['LD_OUT'].update(f'{ld:.6f}')
    window['XCPD_OUT'].update(
        f'{xcpd:.6f}' if xcpd is not None else 'N/A')
    window['TIME_P'].update(
        f'{ems:.2f} ms pred / {elapsed * 1000:.2f} ms total')
    window['TIME_STAMP_P'].update(
        f'Start {_t_start_pred}  |  End {_ts()}')

    def fmt(col):
        m = (det.get(col) if det and col in det else met) or {}
        try:
            return (f"MAE={float(m.get('MAE', 0)):.6f}  "
                    f"RMSE={float(m.get('RMSE', 0)):.6f}  "
                    f"R2={float(m.get('R2', 0)):.6f}")
        except:
            return 'N/A'

    window['MET_CL'].update(fmt('CL') if det else fmt('avg'))
    window['MET_CD'].update(fmt('CD') if det else fmt('avg'))
    window['MET_XCP'].update(fmt('XCP') if det else fmt('avg'))

    xcpd = f'{xcpd:.6f}' if xcpd is not None else 'N/A'
    set_prog('PB_P', 'PP_P', 'PM_P', 100, 'Complete')
    set_status("Aerodynamic prediction complete.", elapsed, C_GREEN)

    # Recolor the already-drawn body now that the result is known, on the
    # same red->green CL/CD scale used by the Optimizer/Envelope tabs, and
    # stamp the result directly onto the 3D plot. Parts keep their own
    # hue (nose/body/wing/tail); fitness only drives vividness + the glow
    # bezel around the plot -- see aero_body_vis.render_geometry_on_figure.
    lo, hi = PRED_LD_COLOR_RANGE
    t = 1.0 if ld == float('inf') else max(0.0, min(1.0, (ld - lo) / (hi - lo)))
    ld_txt = 'inf' if ld == float('inf') else f'{ld:.4f}'
    overlay_txt = (
        f'Prediction Result\n'
        f'CL/CD: {ld_txt}\n'
        f'CL: {cl:.4f}   CD: {cd:.4f}   XCP: {xcp:.4f}'
    )
    render_geometry_panel(
        'PRED_GEO_CANVAS', 'P_GEO', _last_pred_geom,
        'Prediction Geometry', push_history=False,
        fitness_t=t, overlay_text=overlay_txt)


# =========================================================
# PREDICTION RUN LAUNCHER  (shared by Estimate and VISUALIZE)
# =========================================================
def _start_prediction_run(values):
    """
    Validates inputs and launches _pred_worker on a background thread
    using the current Prediction tab inputs -- also renders the input
    geometry immediately (before the model has actually returned a
    result) so there's always something on-screen (and cached for the
    popup) the instant this is called.

    Shared by both the 'Estimate' button and the 'P_GEO_POPUP' /
    VISUALIZE button (starts the prediction AND opens the live popup in
    one click) so both entry points use identical validated params.

    Returns True if a prediction was actually started, False otherwise
    (model not ready / inputs failed validation -- a message has
    already been shown to the user in that case).
    """
    global _last_pred_geom, _t_start_pred
    if not _model_rdy:
        sg.popup_quick_message(
            'Model still loading -- please wait...')
        return False
    ok, err_msg = validate_prediction_inputs(values)
    if not ok:
        show_validation_error("Prediction Input Error", err_msg)
        return False
    params = {p: sf(values, p) for p in PARAMS}
    _t_start_pred = _ts()
    window['TIME_STAMP_P'].update(f'Start {_t_start_pred}  |  End --:--:--')
    set_prog('PB_P', 'PP_P', 'PM_P', 15,
             'Running prediction...')
    _last_pred_geom = dict(params)
    render_geometry_panel('PRED_GEO_CANVAS', 'P_GEO', _last_pred_geom,
                          'Prediction Geometry')
    threading.Thread(target=_pred_worker,
                     args=(params,), daemon=True).start()
    return True


# =========================================================
# LIVE MORPH HELPER  (shared by Optimizer + Envelope workers)
# =========================================================
def _emit_morphed_frames(event_name, prev, cur, morph_geom, numeric_keys,
                         abort_check, steps=LIVE_MORPH_STEPS,
                         delay=LIVE_MORPH_STEP_DELAY):
    """
    Emits a short burst of linearly-interpolated frames from `prev` to
    `cur` via window.write_event_value(event_name, frame), landing
    exactly on `cur` at the final step -- this is what turns each
    generation/sweep-step "jump" into smooth, continuous-looking motion
    on the live 3D preview instead of a snap-cut redraw.

    `prev` / `cur` are plain dicts. If `morph_geom` is True, both must
    carry a `'geom'` sub-dict (same keys) which is interpolated
    parameter-by-parameter. `numeric_keys` lists the other top-level
    numeric fields in `cur` to interpolate (e.g. 'fitness', 'CL', 'CD').
    Every other key in `cur` is copied through unchanged on every frame.

    `abort_check()` is polled before every frame; as soon as it returns
    True the burst stops emitting immediately (used so Abort/a new run
    cancels an in-flight morph rather than finishing it).

    If `prev` is None (first frame of a run), `cur` is emitted once,
    unmorphed -- there's nothing to interpolate from yet.
    """
    if prev is None:
        cur = dict(cur)
        cur['is_interp'] = False
        window.write_event_value(event_name, cur)
        return

    for step in range(1, steps + 1):
        if abort_check():
            return
        frac = step / steps
        frame = dict(cur)

        if morph_geom and prev.get('geom') and cur.get('geom'):
            pg, cg = prev['geom'], cur['geom']
            frame['geom'] = {k: pg[k] + (cg[k] - pg[k]) * frac for k in cg}

        for key in numeric_keys:
            pv = prev.get(key, cur.get(key, 0.0))
            cv = cur.get(key, 0.0)
            frame[key] = pv + (cv - pv) * frac

        frame['is_interp'] = (step < steps)
        window.write_event_value(event_name, frame)
        if step < steps:
            time.sleep(delay)


# =========================================================
# OPTIMIZER RUN LAUNCHER  (shared by RUN and VISUALIZE)
# =========================================================
def _start_optimizer_run(values):
    """
    Validates inputs, resets the run's live state, and launches
    _opt_worker on a background thread using the current Optimizer tab
    settings (Max Generations, population, bounds, constraints, etc).

    Shared by both the 'Run_Opt' button (background run, no popup) and
    the 'O_GEO_POPUP' / VISUALIZE button (starts the run AND opens the
    live 3D popup in one click) so the exact same validated bounds/
    constraints/generation-count feed either entry point -- no logic
    duplicated, no risk of the two drifting apart.

    Returns True if a new run was actually started, False if it
    couldn't be (model not ready, a run is already in progress, or
    inputs failed validation -- in the last two cases a popup/status
    message has already been shown to the user, so callers don't need
    to show their own).
    """
    global _opt_run, _t_start_opt
    if not _model_rdy:
        sg.popup_quick_message(
            'Model still loading -- please wait...')
        return False
    if _opt_run:
        # Already running (e.g. VISUALIZE clicked while a background
        # Run_Opt is in progress) -- nothing new to start, caller should
        # just open/attach to the popup and let the existing run's live
        # events keep it updated.
        return False
    ok, err_msg = validate_optimizer_inputs(values)
    if not ok:
        show_validation_error("Optimizer Input Error", err_msg)
        return False
    _opt_run = True
    _opt_fitness_seen.clear()
    _t_start_opt = _ts()
    window['OPT_TIME'].update(
        f'Start {_t_start_opt}  |  End --:--:--')
    geom_bounds = [(sf(values, f'{p}_LOW'),
                    sf(values, f'{p}_HIGH'))
                   for p in PARAMS[:15]]
    mach_val = sf(values, 'mach', 0.2)
    alpha_val = sf(values, 'alpha', 2.0)
    alt_val = sf(values, 'alt', 0.0)
    bounds = geom_bounds + [(mach_val, mach_val), (alpha_val, alpha_val), (alt_val, alt_val)]
    constraints = {
        'CL': (sf(values, 'CL_MIN'), sf(values, 'CL_MAX')),
        'CD': (sf(values, 'CD_MIN'), sf(values, 'CD_MAX')),
        'XCP': (sf(values, 'XCP_MIN'), sf(values, 'XCP_MAX')),
    }
    maxiter = int(sf(values, 'MAXITER', 50))
    popsize = int(sf(values, 'POPSIZE', 10))
    itermax = int(sf(values, 'ITERMAX', 5))
    out_dir = values.get('OUT_DIR', 'de_output').strip()
    con_clear('OPT_GEO')
    con_clear('OPT_LOG')
    set_prog('PB_O', 'PP_O', 'PM_O', 2,
             'Initialising custom DE...')
    con_append('OPT_LOG',
               '=' * 58 + '\n'
                          f'  CUSTOM DE OPTIMIZER\n'
                          f'  Generations:{maxiter}  Pop:{popsize}'
                          f'  GeneSwap:{itermax}\n'
                          f'  Output: {out_dir or "(none)"}\n'
               + '-' * 58)
    set_status('Running Custom DE Optimizer...',
               color=C_AMBER)
    threading.Thread(
        target=_opt_worker,
        args=(bounds, maxiter, popsize,
              itermax, constraints, out_dir),
        daemon=True).start()
    return True


# =========================================================
# OPTIMIZER WORKER
# =========================================================
def _opt_worker(bounds, maxiter, popsize, itermax,
                constraints, out_dir):
    global _opt_run
    try:
        def _log(msg):
            if not _opt_run:
                raise InterruptedError("Aborted")
            opt_log_q.put(msg)
            try:
                gen = int(msg.split()[1])
            except Exception:
                gen = 0
            pct = min(98, int(gen / maxiter * 100)) if maxiter > 0 else 50
            window.write_event_value('OPT_PROG', (pct, msg))

        def _gen_cb(gen_info):
            # Live per-generation callback fired from inside the DE loop
            # itself (see optimizer.py's _de_loop), before the whole run
            # finishes. Streams the generation's best geometry + fitness
            # to the GUI thread so the 3D aero body can redraw in real
            # time, generation by generation -- morphed smoothly from the
            # previous generation's body via _emit_morphed_frames rather
            # than snap-cutting straight to the new one.
            if not _opt_run:
                raise InterruptedError("Aborted")
            gen_info = dict(gen_info)
            gen_info['maxiter'] = maxiter

            if gen_info.get('geom'):
                _emit_morphed_frames(
                    'OPT_GEN', _gen_cb.prev, gen_info,
                    morph_geom=True,
                    numeric_keys=('fitness', 'CL', 'CD', 'XCP', 'CLCD'),
                    abort_check=lambda: not _opt_run)
                _gen_cb.prev = gen_info
            else:
                # No geometry snapshot this generation (geom_snapshot_every
                # > 1) -- nothing to morph the body toward, just forward
                # the fitness update as-is.
                gen_info['is_interp'] = False
                window.write_event_value('OPT_GEN', gen_info)

        _gen_cb.prev = None

        result, history, elapsed = run_optimization(
            bounds=bounds, maxiter=maxiter, popsize=popsize,
            itermax=itermax, constraints=constraints,
            out_dir=out_dir if out_dir.strip() else None,
            log_callback=_log, geom_snapshot_every=OPT_GEOM_SNAPSHOT_EVERY,
            gen_callback=_gen_cb)
        window.write_event_value('OPT_DONE', (result, history, elapsed))
    except InterruptedError:
        pass
    except Exception as e:
        window.write_event_value('OPT_ERR', str(e))
    finally:
        _opt_run = False


def render_optimization(result, history, elapsed):
    global _opt_figs, _opt_idx, _opt_history, _opt_result
    _opt_history = history
    _opt_result = result

    best_x = result.x
    best_prm = {p: round(float(v), 6)
                for p, v in zip(PARAMS, best_x)}
    t0 = time.perf_counter()
    best_r = aerodynamic_prediction(best_prm)
    call_ms = (time.perf_counter() - t0) * 1000

    cl = best_r['CL']
    cd = best_r['CD']
    xcp = best_r['XCP']
    xcpd = best_r.get('XCP_D', None)
    mode = result.mode if hasattr(result, 'mode') else 'xgboost'
    ld = cl / cd if abs(cd) > 1e-9 else 0.0
    comp_fit = -float(result.fun)
    # mode_txt = ('ENSEMBLE (XGB+RF+GB)'
    #             if mode == 'ensemble' else 'XGBOOST ONLY')
    xcpd = f'{xcpd:.6f}' if xcpd is not None else 'N/A'

    window['OPT_CL'].update(f'{cl:.6f}')
    window['OPT_CD'].update(f'{cd:.6f}')
    window['OPT_XCP'].update(f'{xcp:.6f}')
    window['OPT_XCPD'].update(xcpd)
    window['OPT_LD'].update(f'{ld:.6f}')
    window['OPT_FIT'].update(f'{comp_fit:.6f}')
    window['OPT_TIME'].update(
        f'Start {_t_start_opt}  |  End {_ts()}')
    # window['OPT_MODE'].update(mode_txt)

    geo = [
        '  OPTIMAL GEOMETRY -- 15 PARAMETERS',
        '  ' + '-' * 56,
        f'  {"Parameter":<32}  {"Value":>14}',
        '  ' + '-' * 56,
    ]
    for p in PARAMS[:15]:
        v = best_prm[p]
        geo.append(f'  {LABELS[p]:<32}  {v:>14.6f}')
    geo += [
        '  ' + '-' * 56,
    ]
    con_clear('OPT_GEO')
    con_append('OPT_GEO', '\n'.join(geo))

    con_clear('OPT_LOG')
    con_append('OPT_LOG',
               f'Completed {len(history)} generations | '
               f'fitness={comp_fit:.6f} | {elapsed:.3f} s')

    # Build 3 optimisation figures
    _mpl_style()
    plt.close('all')
    _opt_figs.clear()
    _opt_idx = 0
    import gc
    gc.collect()

    if history:
        gens = [h['generation'] for h in history]

        # A: Fitness evolution
        fig_a, ax_a = plt.subplots(figsize=(9, 4))
        fig_a._orig_dpi = fig_a.dpi
        fig_a._orig_size = list(fig_a.get_size_inches())
        fig_a.patch.set_facecolor(C_BG)
        bfv = [h['fitness'] for h in history]
        afv = [h['avg_fitness'] for h in history]
        ax_a.plot(gens, bfv, '-o', color=C_GREEN, lw=2, ms=4,
                  label='Best fitness')
        ax_a.plot(gens, afv, '-s', color=C_AMBER, lw=1.5, ms=3,
                  label='Avg fitness', alpha=0.8)
        ax_a.fill_between(gens, bfv, alpha=0.08, color=C_GREEN)
        ax_a.axhline(bfv[-1], color=C_CYAN, lw=1, ls='--',
                     label=f'Final: {bfv[-1]:.4f}')
        ax_a.set_xlabel('Generation')
        ax_a.set_ylabel('Fitness')
        ax_a.set_title(f'DE Fitness per Generation')
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_a.legend()
        ax_a.grid(True)
        fig_a.tight_layout()
        _opt_figs.append(fig_a)

        # B: Aero metrics per generation
        fig_b, ax_b = plt.subplots(figsize=(9, 4))
        fig_b._orig_dpi = fig_b.dpi
        fig_b._orig_size = list(fig_b.get_size_inches())
        fig_b.patch.set_facecolor(C_BG)
        ax_b.plot(gens, [h['CL'] for h in history],
                  '-o', color='steelblue', lw=2, ms=4, label='CL')
        ax_b.plot(gens, [h['CD'] for h in history],
                  '-s', color='crimson', lw=2, ms=4, label='CD')
        ax_b.plot(gens, [h['CLCD'] for h in history],
                  '-^', color='darkgreen', lw=2, ms=4, label='CL/CD')
        ax_b.plot(gens, [h['XCP'] for h in history],
                  '-D', color='purple', lw=2, ms=4, label='XCP')
        ax_b.set_xlabel('Generation')
        ax_b.set_ylabel('Value')
        ax_b.set_title('Aerodynamic Metrics per Generation')
        ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_b.legend()
        ax_b.grid(True)
        fig_b.tight_layout()
        _opt_figs.append(fig_b)

        # C: CL/CD vs XCP scatter
        perf_df = getattr(result, 'perf_df', None)
        fig_c, ax_c = plt.subplots(figsize=(9, 4))
        fig_c._orig_dpi = fig_c.dpi
        fig_c._orig_size = list(fig_c.get_size_inches())
        fig_c.patch.set_facecolor(C_BG)
        if (perf_df is not None
                and 'CL/CD_pred' in perf_df.columns):
            ax_c.scatter(perf_df['CL/CD_pred'], perf_df['XCP_pred'],
                         c='steelblue', alpha=0.7,
                         edgecolor='k', s=28)
            ax_c.set_xlabel('CL/CD')
            ax_c.set_ylabel('XCP')
            ax_c.set_title('Optimised Geometry Performance')
        else:
            ax_c.scatter([h['CLCD'] for h in history],
                         [h['XCP'] for h in history],
                         c='steelblue', alpha=0.7,
                         edgecolor='k', s=28)
            ax_c.set_xlabel('CL/CD (best/gen)')
            ax_c.set_ylabel('XCP (best/gen)')
            ax_c.set_title('CL/CD vs XCP per generation')
        ax_c.scatter([ld], [xcp], c=C_AMBER, s=100, zorder=5,
                     edgecolor='white', lw=1.5,
                     label=f'Optimal  XCP={xcp:.3f}')
        ax_c.axvline(ld, color=C_CYAN, lw=0.8, ls='--', alpha=0.6)
        ax_c.axhline(xcp, color=C_CYAN, lw=0.8, ls='--', alpha=0.6)
        ax_c.legend()
        ax_c.grid(True)
        fig_c.tight_layout()
        _opt_figs.append(fig_c)

        window['SHOW_OPT_PLOTS'].update(disabled=False)
        out_dir = 'de_output'
        _save_fig(fig_a, out_dir, 'opt_fitness_evolution.png')
        _save_fig(fig_b, out_dir, 'opt_metrics_per_gen.png')
        _save_fig(fig_c, out_dir, 'opt_clcd_vs_xcp.png')

    # Store optimal geometry in memory
    global _last_best_geom
    _last_best_geom = best_prm.copy()

    # Automatically update baseline inputs on the flight envelope screen
    for p in PARAMS[:15]:
        val = best_prm[p]
        try:
            window[f'E_{p}'].update(str(val))
        except Exception:
            pass

    set_prog('PB_O', 'PP_O', 'PM_O', 100, 'Complete')
    set_status("Optimization run complete.", elapsed, C_GREEN)


def _table(rows, var_key, var_label):
    hdr = (f'  {var_label:>10}  {"CL":>12}  {"CD":>12}  '
           f'{"XCP":>14}  {"XCP/D":>14}  {"CL/CD":>12}')
    sep = '  ' + '-' * (len(hdr) - 2)
    lines = [sep, hdr, sep]
    for r in rows:
        v = r[var_key]
        cl = r['CL']
        cd = r['CD']
        xcp = r['XCP']
        ld = cl / cd if abs(cd) > 1e-9 else 0.0
        xcpd = r.get('XCP_D')
        xcpd = f'{xcpd:.6f}' if xcpd is not None else '  N/A  '
        lines.append(
            f'  {v:>10.6f}  {cl:>12.6f}  {cd:>12.6f}  '
            f'{xcp:>14.6f}  {xcpd:>14}  {ld:>12.6f}')
    lines.append(sep)
    return '\n'.join(lines)


def _stats(rows, var_key, label):
    cls = [r['CL'] for r in rows]
    cds = [r['CD'] for r in rows]
    xcps = [r['XCP'] for r in rows]
    lds = [r['CL'] / r['CD']
           for r in rows if abs(r['CD']) > 1e-9]
    n = len(rows)
    if n == 0:
        return f'  {label}  (0 points — no data)'
    return '\n'.join([
        f'  {label}  ({n} points)',
        f'    {"":>4}  {"Min":>14}  {"Max":>14}  {"Mean":>14}',
        f'    {"CL":>4}  {min(cls):>14.6f}  {max(cls):>14.6f}'
        f'  {sum(cls) / n:>14.6f}',
        f'    {"CD":>4}  {min(cds):>14.6f}  {max(cds):>14.6f}'
        f'  {sum(cds) / n:>14.6f}',
        f'    {"XCP":>4}  {min(xcps):>14.6f}  {max(xcps):>14.6f}'
        f'  {sum(xcps) / n:>14.6f}',
        f'    {"L/D":>4}  {min(lds):>14.6f}  {max(lds):>14.6f}'
        f'  {sum(lds) / len(lds):>14.6f}',
    ])


def _make_sweep_fig(rows, var_key, var_label, title, metrics_cfg, pairs=None, ylabels=None, colors=None):
    _mpl_style()
    fig = Figure(figsize=(11, 3.6))
    axes = fig.subplots(1, 3)
    fig._orig_dpi = fig.dpi
    fig._orig_size = list(fig.get_size_inches())
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(title, color=C_CYAN, fontsize=11, fontweight='bold')
    xcpd = [r[var_key] for r in rows]

    # ------------------------------------------------------------------
    # Build default `pairs` and `ylabels` when the caller didn’t provide them
    # ------------------------------------------------------------------
    if pairs is None:
        if metrics_cfg is not None:
            pairs = metrics_cfg
        else:
            # colour order matches the colour constants used elsewhere
            default_colors = [C_BLUE, C_RED, C_AMBER] if colors is None else colors
            pairs = [
                ('CL', f'CL vs {var_label}', default_colors[0]),
                ('CD', f'CD vs {var_label}', default_colors[1]),
                ('XCP', f'XCP vs {var_label}', default_colors[2]),
            ]

    if ylabels is None:
        ylabels = ['CL ', 'CD ', 'XCP']

    # ------------------------------------------------------------------
    # Plot each axis
    # ------------------------------------------------------------------
    x_vals = [r[var_key] for r in rows]

    # for ax, (yk, tit, col) in zip(axes, metrics_cfg,pairs, ylabels):
    #     ys = ([r['CL'] / r['CD'] if abs(r['CD']) > 1e-9 else 0.0
    #            for r in rows]
    #           if yk == 'CLCD' else [r[yk] for r in rows])
    #     ax.plot(xcpd, ys, '-o', color=col, lw=2, ms=4,
    #             markeredgecolor='white', markeredgewidth=0.5)
    #     ax.fill_between(xcpd, ys, alpha=0.12, color=col)
    #     ax.set_title(tit, color=C_CYAN, fontsize=10)
    #     ax.set_xlabel(var_label, color=C_DIM, fontsize=9)
    #     ax.set_ylabel(yk if yk != 'CLCD' else 'CL/CD',
    #                   color=C_DIM, fontsize=9)
    #     ax.grid(True, lw=0.5, ls='--', color=C_BDR)
    #     ax.set_facecolor(C_INP)
    # fig.tight_layout(rect=[0, 0, 1, 0.90])
    # return fig
    for ax, (metric_key, tit, col) in zip(axes, pairs):
        # If the metric is CL/CD we have to compute it on-the-fly
        if metric_key == 'CLCD':
            ys = [r['CL'] / r['CD'] if abs(r['CD']) > 1e-9 else 0.0 for r in rows]
        else:
            ys = [r[metric_key] for r in rows]

        ax.plot(x_vals, ys, '-o', color=col, lw=2, ms=4,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.fill_between(x_vals, ys, alpha=0.12, color=col)

        ax.set_title(tit, color=C_CYAN, fontsize=10)
        ax.set_xlabel(var_label, color=C_DIM, fontsize=9)
        ax.set_ylabel('CL/CD' if metric_key == 'CLCD' else metric_key,
                      color=C_DIM, fontsize=9)
        ax.grid(True, lw=0.5, ls='--', color=C_BDR)
        ax.set_facecolor(C_INP)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


_ALPHA_CFG = [('CL', 'CL vs Alpha', C_BLUE),
              ('CD', 'CD vs Alpha', C_RED),
              ('XCP', 'XCP vs Alpha', C_AMBER)]
_MACH_CFG = [('CL', 'CL vs Mach', C_GREEN),
             ('CD', 'CD vs Mach', C_RED),
             ('CLCD', 'CL/CD vs Mach', C_AMBER)]
_ALT_CFG = [('CL', 'CL vs Altitude', C_CYAN),
            ('CD', 'CD vs Altitude', C_RED),
            ('CLCD', 'CL/CD vs Altitude', C_AMBER)]


# ----------------------------------------------------------------------
#  Inside the function  render_flight( … )
# ----------------------------------------------------------------------
def render_flight(ar, mr, lr, elapsed,
                  geom_label='user-typed base params',
                  base_params=None, out_dir='de_output'):
    global _env_figs, _env_idx, _env_ar, _env_mr, _env_lr, _env_geom_label
    _env_ar = ar
    _env_mr = mr
    _env_lr = lr
    _env_geom_label = geom_label

    # -----------------------------------------------------------------
    # 1??  Build the three result tables (alpha, mach, altitude)
    # -----------------------------------------------------------------
    txt_alpha = _table(ar, 'alpha', 'Alpha')
    txt_mach = _table(mr, 'mach', 'Mach')
    txt_alt = _table(lr, 'alt', 'Altitude')

    con_clear('ENV_ALPHA')
    con_append('ENV_ALPHA', txt_alpha)

    con_clear('ENV_MACH')
    con_append('ENV_MACH', txt_mach)

    con_clear('ENV_ALT')
    con_append('ENV_ALT', txt_alt)

    # -----------------------------------------------------------------
    # 2??  Build a summary statistics block
    # -----------------------------------------------------------------
    stats_alpha = _stats(ar, 'alpha', 'Alpha sweep')
    stats_mach = _stats(mr, 'mach', 'Mach sweep')
    stats_alt = _stats(lr, 'alt', 'Altitude sweep')

    summary = '\n'.join([stats_alpha, '', stats_mach, '', stats_alt])
    con_clear('ENV_SUM')
    con_append('ENV_SUM', summary)

    # -----------------------------------------------------------------
    # 3??  Plot creation (already present in your script)
    # -----------------------------------------------------------------
    _mpl_style()
    plt.close('all')
    _env_figs.clear()
    _env_idx = 0
    import gc
    gc.collect()

    fig1 = _make_sweep_fig(ar, 'alpha', 'Alpha (deg)', f'Alpha Sweep [{geom_label}]',
                           _ALPHA_CFG,
                           pairs=[('CL', 'CL vs Alpha', C_BLUE),
                                  ('CD', 'CD vs Alpha', C_RED),
                                  ('XCP', 'XCP vs Alpha', C_AMBER)],
                           ylabels=['CL', 'CD', 'XCP'])
    _env_figs.append(fig1)
    _save_fig(fig1, out_dir, 'sweep_alpha.png')

    fig2 = _make_sweep_fig(mr, 'mach', 'Mach', f'Mach Sweep [{geom_label}]',
                           _MACH_CFG)
    _env_figs.append(fig2)
    _save_fig(fig2, out_dir, 'sweep_mach.png')

    # ----- patched line -----
    fig3 = _make_sweep_fig(lr, 'alt', 'Altitude', f'Altitude Sweep [{geom_label}]',
                           _ALT_CFG)

    _env_figs.append(fig3)
    _save_fig(fig3, out_dir, 'sweep_altitude.png')

    window['SHOW_ENV_PLOTS'].update(disabled=False)
    set_prog('PB_E', 'PP_E', 'PM_E', 100, 'Sweeps complete')
    set_status(f"Flight envelope sweeps complete. Plots saved to {out_dir}", elapsed, C_GREEN)

    # Keep the Flight Envelope tab's 3D preview in sync with the geometry
    # actually swept
    global _last_env_base_geom
    if base_params:
        _last_env_base_geom = dict(base_params)
    render_geometry_panel('ENV_GEO_CANVAS', 'E_GEO', _last_env_base_geom,
                          'Envelope Base Geometry')


# =========================================================
# 3D GEOMETRY PREVIEW PANELS (one embedded per tab)
# =========================================================
_geo_figs = {}      # canvas_key -> persistent Figure (redrawn in place, no new windows)
_geo_aggs = {}       # canvas_key -> persistent FigureCanvasTkAgg (Tk widget reused in place)
_geo_has_data = {}  # canvas_key -> bool, True once real geometry (not a placeholder) is shown

# =========================================================
# 3D GEOMETRY HISTORY  (per-tab Undo / Redo for the popup window)
# =========================================================
# canvas_key -> list of {'vis_params': dict, 'label': str} snapshots.
# Prediction & Flight Envelope tabs: one entry per Estimate/Run/Refresh.
# Optimizer tab: one entry per snapshotted generation (see
# _set_geo_history_from_generations), so Undo/Redo scrubs through how
# the aero body evolved across the DE run's generations.
_geo_history = {}
# canvas_key -> current index into _geo_history[canvas_key]
_geo_hist_idx = {}
# canvas_key -> 'solid' or 'wireframe', remembered per tab
_geo_render_mode = {}

_GEO_HISTORY_MAX = 200


def _push_geo_history(canvas_key, vis_params, label, overlay=None):
    """
    Appends a new geometry snapshot to canvas_key's Undo/Redo history
    (used by the Prediction and Flight Envelope tabs, which build history
    incrementally as the user runs/refreshes). Skips the push if the
    geometry is identical to the current top entry. Any "future" (redone)
    entries beyond the current index are discarded, matching normal
    undo/redo-stack semantics.

    `overlay`, if given, is the same results text (CL/CD/XCP/fitness,
    etc.) previously baked onto the 3D plot -- it's stored on the entry
    so the popup's results widget (beside the Save button) can show the
    right text for whichever history step is being viewed, including
    right when the popup first opens on a static (non-live) result.
    """
    hist = _geo_history.setdefault(canvas_key, [])
    idx = _geo_hist_idx.get(canvas_key, -1)

    # Drop any redo-able entries past the current position
    del hist[idx + 1:]

    if hist and hist[-1]['vis_params'] == vis_params:
        hist[-1]['overlay'] = overlay
        _geo_hist_idx[canvas_key] = len(hist) - 1
        return

    hist.append({'vis_params': dict(vis_params), 'label': label, 'overlay': overlay})
    if len(hist) > _GEO_HISTORY_MAX:
        del hist[:len(hist) - _GEO_HISTORY_MAX]
    _geo_hist_idx[canvas_key] = len(hist) - 1


def _set_geo_history_from_generations(canvas_key, history):
    """
    Replaces canvas_key's Undo/Redo history wholesale with one entry per
    snapshotted generation from an optimizer `history` list (entries
    whose 'geom' key is not None -- see optimizer.py's
    geom_snapshot_every). Used after an Optimizer run completes so the
    popup's Undo/Redo scrubs through the DE run's generations instead of
    a plain edit history.
    """
    entries = []
    for h in history:
        if h.get('geom'):
            gen  = h['generation']
            fit  = h.get('fitness', 0.0)
            ld_v = h.get('CLCD', 0.0)
            xcp_v = h.get('XCP', 0.0)
            entries.append({
                'vis_params': geom_dict_to_vis_params(h['geom']),
                'label': f"Generation {gen}",
                'overlay': (f'Generation {gen}\n'
                           f'Best Fitness: {fit:.6f}\n'
                           f'CL/CD: {ld_v:.4f}   XCP: {xcp_v:.4f}'),
            })
    if not entries:
        return
    _geo_history[canvas_key] = entries
    _geo_hist_idx[canvas_key] = len(entries) - 1


def _get_geo_fig(canvas_key, figsize=(14.0, 5.9)):
    fig = _geo_figs.get(canvas_key)
    if fig is None:
        fig = Figure(figsize=figsize)
        fig.patch.set_facecolor(C_BG)
        _geo_figs[canvas_key] = fig
    return fig


def render_geometry_placeholder(canvas_key, prefix, title):
    """
    Draws a clean 'nothing computed yet' placeholder into the 3D preview
    canvas -- used at startup and after Reset/Clear so the panel never
    shows a stale or default body before the user actually runs this tab.
    """
    global _geo_has_data

    fig = _get_geo_fig(canvas_key)
    fig._geo_vis_params = None
    fig._geo_title = None
    fig.clf()
    fig.patch.set_facecolor(C_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(C_BG)
    ax.axis('off')
    ax.text(0.5, 0.56, '\u2708', ha='center', va='center',
            fontsize=34, color=C_BDR, transform=ax.transAxes)
    ax.text(0.5, 0.34, f'{title}\nClick RUN to generate the 3D model',
            ha='center', va='center', fontsize=12, color=C_DIM,
            fontweight='bold', linespacing=1.8, transform=ax.transAxes)
    fig.tight_layout()

    if EMBED_3D_PREVIEW:
        try:
            _embed_fig(fig, canvas_key)
        except Exception:
            pass

    _geo_has_data[canvas_key] = False
    for suf in ('_LEN', '_WSPAN', '_TSPAN', '_FIN'):
        try:
            window[f'{prefix}{suf}'].update('---')
        except Exception:
            pass
    try:
        window[f'{prefix}_STATUS'].update('Awaiting run...')
    except Exception:
        pass


def render_geometry_panel(canvas_key, prefix, geom, title, push_history=True,
                          pitch_deg=0.0, status_override=None,
                          fast_preview=False, fitness_t=None,
                          overlay_text=None, gen_marker=None, progress=None):
    """
    Renders the aerodynamic body described by `geom` (a dict keyed by
    this GUI's PARAMS names) into the embedded canvas `canvas_key`, and
    updates that panel's compact metric readouts (keys f'{prefix}_...').
    Safe to call repeatedly; reuses one persistent Figure per canvas.

    `push_history=False` skips adding this render to the tab's
    Undo/Redo popup history -- used for the Optimizer tab, whose history
    is instead built wholesale from the run's generations (see
    _set_geo_history_from_generations) rather than one entry per render.

    `pitch_deg` visually pitches the rendered body about its mid-length
    point (angle of attack representation) -- used for the Flight
    Envelope tab's live alpha-sweep playback. Shape/metrics are computed
    from the unrotated geometry; only the drawn orientation changes.

    `status_override`, if given, replaces the default "Updated <time>"
    text in the panel's status line -- used to show a live readout (e.g.
    "Gen 07/50 | Fitness=0.842311 | CL=... CD=... XCP=...") instead of
    just a timestamp.

    `fast_preview=True` is for the live, redrawn-every-frame call sites
    (per-generation optimizer playback, per-step alpha-sweep animation):
    it draws a lower-resolution mesh and reuses the existing 3D axes
    in place, purely to keep pace with rapid successive redraws. It
    never changes `geom` or any computed metric -- see
    aero_body_vis.render_geometry_on_figure for why that's safe. Final /
    one-off renders (optimizer done, applied geometry, placeholders)
    should leave this False for the crisp full-resolution view.

    `fitness_t` / `overlay_text` are passed straight through to
    aero_body_vis.render_geometry_on_figure() -- see that function's
    docstring. `fitness_t` is a float in [0,1] (0=worst seen, 1=best
    seen); nose/body/wing/tail keep their own distinct hues and only
    their vividness plus a glow bezel around the plot respond to it.
    Used by the live per-generation optimizer playback and the Flight
    Envelope sweep playback. Leave both None for the normal, full-
    vividness, unlabeled view.

    `gen_marker`, if given, is a dict {'gen', 'total', 'fitness'} passed
    straight through to aero_body_vis.render_geometry_on_figure() -- a
    small yellow dot + "G<gen> Fit=<fitness>" label drawn ON the body
    itself (not the HUD corner), moving from nose to tail as the
    generation advances. Used by the Optimizer tab's live playback so
    the current generation is visible directly on the model. Leave None
    for no marker. NOTE: the live 3D popup (_show_geo_popup) always
    renders with color_by_progress=False and strips gen_marker/fitness_t
    before drawing, so the body itself is never tinted during live
    playback there -- fitness_t/gen_marker still drive the compact
    tab-side panels and any exported/static renders that call this
    function directly.

    `progress`, if given, is a dict {'kind': 'Generation'|'Step',
    'current': int, 'total': int, 'label': str} describing the current
    live run's position -- e.g. {'kind': 'Generation', 'current': 7,
    'total': 50, 'label': 'Best Fitness: 0.842311'}. It is NOT forwarded
    to aero_body_vis (it's UI-only): the live 3D popup reads it to drive
    its "Generation 7 / 50 | Best Fitness: ..." step line and progress
    bar, replacing the old generic "LIVE -- updating..." text. Leave
    None outside of live Optimizer/Envelope playback.
    """
    global _geo_figs, _geo_has_data

    vis_params = geom_dict_to_vis_params(geom)
    fig = _get_geo_fig(canvas_key)
    render_mode = _geo_render_mode.get(canvas_key, 'solid')

    try:
        metrics = aero_body_vis.render_geometry_on_figure(
            fig, vis_params, title=title, render_mode=render_mode,
            pitch_deg=pitch_deg, fast_preview=fast_preview,
            fitness_t=fitness_t, overlay_text=overlay_text,
            gen_marker=gen_marker)
    except Exception as e:
        try:
            window[f'{prefix}_STATUS'].update(f'ERROR building geometry: {e}')
        except Exception:
            pass
        return

    # Stash what's needed to reconstruct this exact view in the large
    # popup window (double-click on canvas, or the VISUALIZE button) --
    # including the live-playback extras (pitch/fitness/overlay/marker)
    # so a popup left open during a run can mirror the exact same frame.
    fig._geo_vis_params = vis_params
    fig._geo_title = title
    fig._geo_canvas_key = canvas_key
    fig._geo_render_kwargs = dict(
        pitch_deg=pitch_deg, fitness_t=fitness_t,
        overlay_text=overlay_text, gen_marker=gen_marker)
    fig._geo_progress = progress
    _geo_live_seq[canvas_key] = _geo_live_seq.get(canvas_key, 0) + 1

    if push_history:
        _push_geo_history(canvas_key, vis_params, f'Updated {_ts()}',
                          overlay=overlay_text)

    if EMBED_3D_PREVIEW:
        try:
            _embed_fig(fig, canvas_key)
        except Exception as e:
            try:
                window[f'{prefix}_STATUS'].update(f'ERROR rendering canvas: {e}')
            except Exception:
                pass
            return

    _geo_has_data[canvas_key] = True
    try:
        window[f'{prefix}_LEN'].update(f"{metrics['total_length']:.1f}")
        window[f'{prefix}_WSPAN'].update(f"{metrics['wingspan']:.1f}")
        window[f'{prefix}_TSPAN'].update(f"{metrics['tail_span']:.1f}")
        window[f'{prefix}_FIN'].update(f"{metrics['fineness_ratio']:.2f}")
        window[f'{prefix}_STATUS'].update(
            status_override
            if status_override is not None
            else f'Updated {_ts()}  (double-click to enlarge)')
    except Exception:
        pass


def render_all_geometry_placeholders():
    """Shows the 'awaiting run' placeholder on all three tabs' 3D previews.
    Used once at startup so nothing appears until the user clicks Run."""
    render_geometry_placeholder('PRED_GEO_CANVAS', 'P_GEO', 'Prediction Geometry')
    render_geometry_placeholder('OPT_GEO_CANVAS', 'O_GEO', 'Optimizer Best Geometry')
    render_geometry_placeholder('ENV_GEO_CANVAS', 'E_GEO', 'Envelope Base Geometry')


def render_all_geometry_panels(values=None):
    """
    Redraws all three tabs' 3D previews (used after theme/font rebuilds).
    Any tab that hasn't actually been run yet keeps showing its
    placeholder instead of a default body.
    """
    pred_geom = (
        {p: sf(values, p, DEFAULTS[p]) for p in PARAMS[:15]}
        if values is not None else _last_pred_geom
    )

    if _geo_has_data.get('PRED_GEO_CANVAS'):
        render_geometry_panel('PRED_GEO_CANVAS', 'P_GEO', pred_geom,
                              'Prediction Geometry', push_history=False)
    else:
        render_geometry_placeholder('PRED_GEO_CANVAS', 'P_GEO', 'Prediction Geometry')

    if _geo_has_data.get('OPT_GEO_CANVAS'):
        render_geometry_panel('OPT_GEO_CANVAS', 'O_GEO',
                              _last_best_geom if _last_best_geom else pred_geom,
                              'Optimizer Best Geometry', push_history=False)
    else:
        render_geometry_placeholder('OPT_GEO_CANVAS', 'O_GEO', 'Optimizer Best Geometry')

    if _geo_has_data.get('ENV_GEO_CANVAS'):
        render_geometry_panel('ENV_GEO_CANVAS', 'E_GEO', _last_env_base_geom,
                              'Envelope Base Geometry', push_history=False)
    else:
        render_geometry_placeholder('ENV_GEO_CANVAS', 'E_GEO', 'Envelope Base Geometry')


# =========================================================
# EXPORT CSV
# =========================================================
_last_sweep = {'ar': [], 'mr': [], 'lr': [], 'label': ''}


def export_envelope_csv():
    ar = _last_sweep['ar']
    mr = _last_sweep['mr']
    lr = _last_sweep['lr']
    if not (ar or mr or lr):
        sg.popup_quick_message('Run the sweeps first before exporting.')
        return

    path = sg.popup_get_file(
        'Save sweep results as CSV',
        save_as=True,
        default_path='sweep_results.csv',
        default_extension='.csv',
        file_types=(('CSV Files', '*.csv'), ('All Files', '*.*')),
        no_window=True
    )
    if not path:
        return
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Sweep', 'Variable', 'Value',
                    'CL', 'CD', 'XCP', 'XCP_D', 'CL_CD'])
        for rows, sname, vk in [
            (ar, 'Alpha', 'alpha'),
            (mr, 'Mach', 'mach'),
            (lr, 'Altitude', 'alt')]:
            for r in rows:
                cl = r['CL']
                cd = r['CD']
                xcp = r['XCP']
                ld = cl / cd if abs(cd) > 1e-9 else ''
                xcpd = r.get('XCP_D', '')
                w.writerow([
                    sname,
                    vk,
                    f"{r[vk]:.6f}" if isinstance(r[vk], (int, float)) else r[vk],
                    f"{cl:.6f}" if isinstance(cl, (int, float)) else cl,
                    f"{cd:.6f}" if isinstance(cd, (int, float)) else cd,
                    f"{xcp:.6f}" if isinstance(xcp, (int, float)) else xcp,
                    f"{xcpd:.6f}" if isinstance(xcpd, (int, float)) else xcpd,
                    f"{ld:.6f}" if isinstance(ld, (int, float)) else ld
                ])
    sg.popup_quick_message(f'Exported to:\n{path}')


# =========================================================
# RESET / CLEAR HELPERS
# =========================================================
def reset_pred():
    for p in PARAMS:
        window[p].update(str(DEFAULTS[p]))
    for k in ['CL_OUT', 'CD_OUT', 'XCP_OUT', 'XCPD_OUT', 'LD_OUT',
              'TIME_P', 'MET_CL', 'MET_CD', 'MET_XCP']:
        window[k].update('---')
        window['TIME_STAMP_P'].update('---')

    set_prog('PB_P', 'PP_P', 'PM_P', 0, '')
    set_status('Parameters reset to defaults.')
    _geo_history.pop('PRED_GEO_CANVAS', None)
    _geo_hist_idx.pop('PRED_GEO_CANVAS', None)
    render_geometry_placeholder('PRED_GEO_CANVAS', 'P_GEO', 'Prediction Geometry')


def clear_opt():
    global _opt_figs, _opt_idx
    for k in ['OPT_CL', 'OPT_CD', 'OPT_XCP', 'OPT_XCPD',
              'OPT_LD', 'OPT_FIT', 'OPT_TIME']:
        window[k].update('---')
        window['OPT_TIME'].update('---')

    con_clear('OPT_GEO')
    con_clear('OPT_LOG')
    _opt_figs.clear()
    _opt_idx = 0
    try:
        window['SHOW_OPT_PLOTS'].update(disabled=True)
    except Exception:
        pass
    import gc
    gc.collect()
    set_prog('PB_O', 'PP_O', 'PM_O', 0, '')
    set_status('Optimization results cleared.')
    _geo_history.pop('OPT_GEO_CANVAS', None)
    _geo_hist_idx.pop('OPT_GEO_CANVAS', None)
    render_geometry_placeholder('OPT_GEO_CANVAS', 'O_GEO', 'Optimizer Best Geometry')


def clear_env():
    global _env_figs, _env_idx
    # Cancel any in-flight live sweep-point animation so it doesn't keep
    # writing to a panel that's about to be reset.
    _env_anim_token[0] += 1
    _flt_run_ok[0] = False
    for k in ['ENV_ALPHA', 'ENV_MACH', 'ENV_ALT', 'ENV_SUM']:
        con_clear(k)
        window['ENV_TIMESTAMP'].update('---')

    _last_sweep.update({'ar': [], 'mr': [], 'lr': [], 'label': ''})
    _env_figs.clear()
    _env_idx = 0
    try:
        window['SHOW_ENV_PLOTS'].update(disabled=True)
    except Exception:
        pass
    gc.collect()
    set_prog('PB_E', 'PP_E', 'PM_E', 0, '')
    set_status('Flight envelope cleared.')
    _geo_history.pop('ENV_GEO_CANVAS', None)
    _geo_hist_idx.pop('ENV_GEO_CANVAS', None)
    render_geometry_placeholder('ENV_GEO_CANVAS', 'E_GEO', 'Envelope Base Geometry')


# =========================================================
# PARALLEL SWEEP WORKER
# =========================================================
def _run_sweeps_parallel(base, ac, mc, lc):
    f_alpha = _SWEEP_POOL.submit(alpha_sweep, base, *ac)
    f_mach = _SWEEP_POOL.submit(mach_sweep, base, *mc)
    f_alt = _SWEEP_POOL.submit(altitude_sweep, base, *lc)
    ar = f_alpha.result()
    mr = f_mach.result()
    lr = f_alt.result()
    return ar, mr, lr


# =========================================================
# FLIGHT ENVELOPE — LIVE SWEEP-POINT ANIMATION
# =========================================================
def _env_animate_worker(base_geom, ar, mr, lr, anim_token, ld_lo=0.0, ld_hi=1.0):
    """
    Replays an already-computed sweep result (ar/mr/lr, each a list of
    per-point dicts with keys 'alpha'/'mach'/'alt', 'CL', 'CD', 'XCP')
    one point at a time, live, so the Flight Envelope tab visibly shows
    *which* point is currently active and *how* the result changes --
    instead of only ever showing the final, fully-computed plots. Each
    step morphs smoothly from the previous step via _emit_morphed_frames
    (alpha's pitch angle, and CL/CD/XCP/CLCD for all three sweep types,
    all interpolate frame-to-frame) rather than snap-cutting.

    Alpha sweep: the 3D body is re-rendered with a live pitch rotation
    (angle of attack) for each alpha value.
    Mach / Altitude sweeps: pitch has no geometric meaning for these, so
    the body stays unrotated, but it's still redrawn every step so the
    live color/HUD text track that step's result.

    `ld_lo`/`ld_hi` (the sweep's overall CL/CD range, computed once up
    front by the caller) is stashed into `_env_ld_range` for the GUI-
    thread event handler to normalize each step's CL/CD into a fitness_t
    in [0,1] -- same red->green vividness/glow scheme as the Optimizer
    tab, comparable across all three sweep types within one run.

    `anim_token` lets a newer animation (from a later sweep run) cancel
    an older one still in flight, so two animations never race.
    """
    global _env_ld_range
    _env_ld_range = (ld_lo, ld_hi)

    def _still_current():
        return _env_anim_token[0] == anim_token and _flt_run_ok[0]

    def _abort():
        return not _still_current()

    def _pace(n_points, budget_s=4.0, lo=0.03, hi=0.20):
        if n_points <= 1:
            return 0.0
        return min(hi, max(lo, budget_s / n_points))

    def _ld(r):
        cl, cd = r.get('CL', 0.0), r.get('CD', 0.0)
        return cl / cd if abs(cd) > 1e-9 else 0.0

    def _play(points, kind, value_key):
        # Each sweep type starts its own morph chain fresh (`prev=None`)
        # since jumping from e.g. the last alpha value to the first mach
        # point isn't something that should visually morph.
        prev = None
        step_budget = _pace(len(points))
        sub_delay = max(0.008, step_budget / ENV_MORPH_STEPS)
        for i, r in enumerate(points):
            if not _still_current():
                return
            cur = {
                'kind': kind, 'index': i, 'total': len(points),
                'value': r.get(value_key, 0.0),
                'CL': r.get('CL', 0.0), 'CD': r.get('CD', 0.0),
                'XCP': r.get('XCP', 0.0), 'CLCD': _ld(r),
                'geom': base_geom,
            }
            _emit_morphed_frames(
                'ENV_STEP', prev, cur, morph_geom=False,
                numeric_keys=('value', 'CL', 'CD', 'XCP', 'CLCD'),
                abort_check=_abort, steps=ENV_MORPH_STEPS, delay=sub_delay)
            prev = cur

    try:
        _play(ar, 'alpha', 'alpha')
        _play(mr, 'mach', 'mach')
        _play(lr, 'alt', 'alt')

        if _still_current():
            window.write_event_value('ENV_STEP', {'kind': 'done'})
    except Exception as _anim_exc:
        print(f"[env animation warning] {_anim_exc}")


# Mutable single-slot "cancel token" + run flag so a fresh sweep (or
# Clear_Env) can stop any animation still playing from a previous run.
_env_anim_token = [0]
_flt_run_ok = [True]

# Fewer morph sub-frames than the Optimizer tab (ENV sweeps commonly have
# far more points than optimizer generations, so keep each transition
# cheaper) and the live CL/CD range for the currently-playing sweep, set
# once per run by _env_animate_worker and read by the ENV_STEP handler
# to normalize each step's fitness_t for the body's vividness/glow tint.
ENV_MORPH_STEPS = 4
_env_ld_range = (0.0, 1.0)


# =========================================================
# ENVELOPE SWEEP LAUNCHER  (shared by Run_Env and VISUALIZE)
# =========================================================
def _call_sweep(fn, base, rng, is_aborted):
    """
    Calls a sweep function (alpha_sweep/mach_sweep/altitude_sweep) with
    abort-support if the function accepts it; falls back to calling it
    without that argument if it doesn't (older/local envelope.py
    implementations that pre-date abort support).
    """
    try:
        return fn(base, *rng, is_aborted=is_aborted)
    except TypeError:
        return fn(base, *rng)


def _start_envelope_run(values):
    """
    Validates inputs, resets the run's live state, and launches
    _env_worker on a background thread using the current Flight
    Envelope tab settings (base geometry + alpha/mach/altitude ranges).

    Shared by both the 'Run_Env' button (background sweep, no popup)
    and the 'E_GEO_POPUP' / VISUALIZE button (starts the sweep AND
    opens the live popup in one click), same pattern as
    _start_optimizer_run() / _start_prediction_run() -- one validated
    code path feeds either entry point.

    The sweep itself computes all points up front (fast; it's not an
    iterative search like the optimizer), then ENV_DONE kicks off
    _env_animate_worker, which plays the already-computed points back
    point-by-point on the 3D body (pitching for alpha, recoloring for
    mach/altitude) -- that playback is what an already-open popup
    actually watches "live".

    Returns True if a new sweep was actually started, False otherwise
    (model not ready, a sweep is already running, or inputs failed
    validation -- a message has already been shown to the user in
    those cases).
    """
    global _flt_run, _t_start_env
    if not _model_rdy:
        sg.popup_quick_message(
            'Model still loading -- please wait...')
        return False
    if _flt_run:
        return False
    ok, err_msg = validate_envelope_inputs(values)
    if not ok:
        show_validation_error("Flight Envelope Input Error", err_msg)
        return False
    _flt_run = True
    _t_start_env = _ts()
    window['ENV_TIMESTAMP'].update(
        f'Start {_t_start_env}  |  End --:--:--')
    odv = (values.get('OUT_DIR', 'de_output').strip()
           or 'de_output')
    base = {}
    for p in PARAMS[:15]:
        base[p] = sf(values, f'E_{p}', DEFAULTS[p])
    for p in PARAMS[15:]:
        base[p] = sf(values, p, DEFAULTS[p])
    glbl = 'envelope base parameters'

    ac = (sf(values, 'ALPHA_MIN'),
          sf(values, 'ALPHA_MAX'),
          sf(values, 'ALPHA_STP'))
    mc = (sf(values, 'MACH_MIN'),
          sf(values, 'MACH_MAX'),
          sf(values, 'MACH_STP'))
    lc = (sf(values, 'ALT_MIN'),
          sf(values, 'ALT_MAX'),
          sf(values, 'ALT_STP'))

    for k in ['ENV_ALPHA', 'ENV_MACH', 'ENV_ALT', 'ENV_SUM']:
        con_clear(k)
    set_prog('PB_E', 'PP_E', 'PM_E', 5,
             f'Starting sweeps [{glbl}]...')
    set_status(
        f'Running flight envelope sweeps [{glbl}]...',
        color=C_AMBER)

    def _env_worker(base, ac, mc, lc, label, odir):
        global _flt_run
        try:
            t0 = time.perf_counter()

            is_ab = lambda: not _flt_run

            ar_raw = _call_sweep(alpha_sweep, base, ac, is_ab)
            if not _flt_run:
                return
            mr_raw = _call_sweep(mach_sweep, base, mc, is_ab)
            if not _flt_run:
                return
            lr_raw = _call_sweep(altitude_sweep, base, lc, is_ab)
            if not _flt_run:
                return

            # ---- filter out None / empty rows ----
            ar_raw = [r for r in ar_raw if r]
            mr_raw = [r for r in mr_raw if r]
            lr_raw = [r for r in lr_raw if r]

            # ---- Convert to dicts (keep the original var name) ----
            ar = _sweep_to_dict(ar_raw, 'alpha')
            mr = _sweep_to_dict(mr_raw, 'mach')
            lr = _sweep_to_dict(lr_raw, 'alt')

            elapsed = time.perf_counter() - t0
            window.write_event_value('ENV_DONE',
                                     (ar, mr, lr, elapsed, label, base, odir))
        except InterruptedError:
            pass
        except Exception as e:
            window.write_event_value('ENV_ERR', str(e))
        finally:
            _flt_run = False

    threading.Thread(
        target=_env_worker,
        args=(base, ac, mc, lc, glbl, odv),
        daemon=True).start()
    return True



# =========================================================
# LIVE EVENT HANDLER  (shared by the main window loop AND by
# _show_geo_popup()'s own nested loop -- see call sites below)
# =========================================================
# These are the events the background Prediction/Optimizer/Envelope
# worker threads push via window.write_event_value(): the one-shot
# 'PRED_DONE' result, the per-generation 'OPT_GEN' / terminal 'OPT_DONE'
# /'OPT_ERR' optimizer events, and the per-step 'ENV_STEP' / terminal
# 'ENV_DONE'/'ENV_ERR' envelope-sweep events. Pulling the handling logic
# out into a standalone function (instead of leaving it only reachable
# as inline branches inside the main window's event loop) lets
# _show_geo_popup() process these live while its OWN modal event loop
# is running -- otherwise the main loop (previously the only place that
# drained `window`'s queue) is completely blocked for as long as the
# popup is open, so 'OPT_GEN'/'ENV_STEP' events pile up unread,
# render_geometry_panel() (and the _geo_live_seq bump it does) never
# runs, and the popup's own live-redraw polling has nothing to react
# to -- the body just sits frozen on the first frame until the popup is
# closed and the backlog finally drains. See _show_geo_popup() below.
_LIVE_EVENTS = {'PRED_DONE', 'OPT_PROG', 'OPT_GEN', 'OPT_DONE', 'OPT_ERR',
                'ENV_DONE', 'ENV_STEP', 'ENV_ERR'}


def _handle_live_event(event, values):
    """
    Handles one live/result event pushed by the Prediction, Optimizer,
    or Envelope worker threads. Callers just dispatch on
    `event in _LIVE_EVENTS` and call this -- it does the actual work
    (previously inline in the main loop only). Called from BOTH the
    main window's event loop and _show_geo_popup()'s nested loop, so
    the live 3D popup keeps redrawing generation-by-generation /
    step-by-step even while it, not the main loop, has control.
    """
    global _opt_run, _flt_run

    # Handle prediction queue result via window event value
    if event == 'PRED_DONE':
        msg = values['PRED_DONE']
        if msg['ok']:
            render_prediction(msg)
        else:
            set_prog('PB_P', 'PP_P', 'PM_P', 0, '')
            set_status(
                f'ERROR  Prediction: {msg["error"]}', color=C_RED)
            sg.popup_error(f'Prediction Error:\n{msg["error"]}')
        return

    # Optimizer progress / done / error events
    if event == 'OPT_PROG':
        pct, msg = values['OPT_PROG']
        set_prog('PB_O', 'PP_O', 'PM_O', pct, msg)
        return
    if event == 'OPT_GEN':
        # Live per-generation update: redraw the 3D aero body with THIS
        # generation's best individual immediately, while the DE loop is
        # still running -- generation counter, best fitness, and the
        # CL/CD/XCP/CL-CD panel (the same "Best CL / Best CD / Best XCP /
        # Max CL/CD / Composite Fitness" readout normally only filled in
        # once the run finishes) all update every single generation now,
        # sourced straight from this generation's already-computed
        # gen_callback payload -- nothing here is a new computation, just
        # displaying values optimizer.py was already producing each
        # generation. XCP/D (deflection) isn't part of that per-generation
        # payload, so it still only refreshes once the run completes.
        # Also pushed into the Undo/Redo history for this canvas so the
        # popup viewer can scrub back through every generation.
        gi = values['OPT_GEN']
        geom = gi.get('geom')
        fitness = gi.get('fitness', 0.0)
        is_interp = gi.get('is_interp', False)
        # Track only *landed* generations' fitness (not the in-between
        # morph frames) so the red->green normalization reflects the true
        # best/worst seen so far this run.
        if np.isfinite(fitness) and not is_interp:
            _opt_fitness_seen.append(fitness)
        if geom:
            gen_n   = gi.get('generation', 0)
            tot_n   = gi.get('maxiter', gi.get('total_gens', 0))
            cl_v    = gi.get('CL', 0.0)
            cd_v    = gi.get('CD', 0.0)
            xcp_v   = gi.get('XCP', 0.0)
            ld_v    = gi.get('CLCD',
                             (cl_v / cd_v if abs(cd_v) > 1e-9 else 0.0))
            status_txt = (
                f'Generation : {gen_n} / {tot_n}   |   '
                f'Best Fitness : {fitness:.6f}'
                + ('  (morphing…)' if is_interp else '')
            )
            # Normalize this frame's fitness against the best/worst seen
            # so far this run -> drives part vividness + the glow bezel
            # around the plot (see aero_body_vis.render_geometry_on_figure).
            # Parts keep their own hue; nothing here changes to a flat tint.
            if len(_opt_fitness_seen) >= 2:
                f_lo = min(_opt_fitness_seen)
                f_hi = max(_opt_fitness_seen)
                t = (fitness - f_lo) / (f_hi - f_lo) if f_hi > f_lo else 1.0
            else:
                t = 1.0
            overlay_txt = (
                f'Generation {gen_n} / {tot_n}\n'
                f'Best Fitness: {fitness:.6f}\n'
                f'CL/CD: {ld_v:.4f}   XCP: {xcp_v:.4f}'
            )
            render_geometry_panel(
                'OPT_GEO_CANVAS', 'O_GEO', geom,
                f'Optimizer — Generation {gen_n}',
                push_history=(not is_interp), status_override=status_txt,
                fast_preview=True, fitness_t=t,
                overlay_text=overlay_txt,
                gen_marker={'gen': gen_n, 'total': tot_n, 'fitness': fitness},
                progress={'kind': 'Generation', 'current': gen_n, 'total': tot_n,
                         'label': f'Best Fitness: {fitness:.6f}'
                                  + ('  (morphing…)' if is_interp else '')})
            try:
                window['OPT_CL'].update(f'{cl_v:.6f}')
                window['OPT_CD'].update(f'{cd_v:.6f}')
                window['OPT_XCP'].update(f'{xcp_v:.6f}')
                window['OPT_LD'].update(f'{ld_v:.6f}')
                window['OPT_FIT'].update(f'{fitness:.6f}')
            except Exception:
                pass
        return
    if event == 'OPT_DONE':
        result, hist, elapsed = values['OPT_DONE']
        render_optimization(result, hist, elapsed)
        render_geometry_panel('OPT_GEO_CANVAS', 'O_GEO', _last_best_geom,
                              'Optimizer Best Geometry', push_history=False)
        # Build the popup's Undo/Redo history from this run's
        # snapshotted generations, so the user can scrub through how
        # the aero body evolved generation-by-generation.
        _set_geo_history_from_generations('OPT_GEO_CANVAS', hist)
        return
    if event == 'OPT_ERR':
        e = values['OPT_ERR']
        set_status(f'ERROR  Optimizer: {e}', color=C_RED)
        sg.popup_error(f'Optimization Error:\n{e}')
        set_prog('PB_O', 'PP_O', 'PM_O', 0, 'error')
        _opt_run = False
        return

    # Flight envelope events
    if event == 'ENV_DONE':
        pl = values['ENV_DONE']
        ar, mr, lr, elapsed = pl[:4]
        glbl = pl[4] if len(pl) > 4 else 'user params'
        bp = pl[5] if len(pl) > 5 else None
        odv = pl[6] if len(pl) > 6 else 'de_output'
        _last_sweep.update({'ar': ar, 'mr': mr,
                            'lr': lr, 'label': glbl})
        render_flight(ar, mr, lr, elapsed, glbl,
                      base_params=bp, out_dir=odv)

        # Kick off the live, point-by-point playback of this sweep on
        # the 3D panel: alpha values pitch the body live, mach/altitude
        # values recolor it in place (see _env_animate_worker). All three
        # sweeps share one CL/CD color scale (computed once here, up
        # front) so the tint is comparable across alpha/mach/altitude
        # steps within the same run, not just within one sweep.
        base_geom = bp if bp else \
            {p: sf(values, f'E_{p}', DEFAULTS[p]) for p in PARAMS[:15]}
        all_ld = [r['CL'] / r['CD'] for r in (ar + mr + lr)
                 if abs(r.get('CD', 0.0)) > 1e-9]
        ld_lo = min(all_ld) if all_ld else 0.0
        ld_hi = max(all_ld) if all_ld else 1.0
        _env_anim_token[0] += 1
        _flt_run_ok[0] = True
        threading.Thread(
            target=_env_animate_worker,
            args=(base_geom, ar, mr, lr, _env_anim_token[0], ld_lo, ld_hi),
            daemon=True).start()
        return
    if event == 'ENV_STEP':
        si = values['ENV_STEP']
        kind = si.get('kind')
        if kind == 'done':
            try:
                window['E_GEO_STATUS'].update(
                    'Sweep playback complete  (double-click to enlarge)')
            except Exception:
                pass
            return
        # Mirrors the Optimizer tab's live panel: while the sweep is
        # playing, show only the current step counter and the sweep
        # variable's value alongside the live 3D body. CL/CD/XCP are
        # performance results of that step, not part of what draws the
        # body, and only belong in the tab's plots/tables after the
        # sweep finishes.
        val   = si.get('value', 0.0)
        idx   = si.get('index', 0) + 1
        tot   = si.get('total', 0)
        ld_v  = si.get('CLCD', 0.0)
        xcp_v = si.get('XCP', 0.0)
        geom  = si.get('geom') or {}
        is_interp = si.get('is_interp', False)
        morph_suffix = '  (morphing…)' if is_interp else ''
        # Normalize this step's CL/CD against the sweep's overall CL/CD
        # range (see _env_ld_range, set once per run) -> drives part
        # vividness + the glow bezel, same scheme as the Optimizer tab.
        ld_lo, ld_hi = _env_ld_range
        t = (ld_v - ld_lo) / (ld_hi - ld_lo) if ld_hi > ld_lo else 1.0
        if kind == 'alpha':
            status_txt = f'Step : {idx} / {tot}   |   Alpha : {val:.2f}°' + morph_suffix
            overlay_txt = (
                f'Alpha Sweep — Step {idx} / {tot}\n'
                f'Alpha: {val:.2f}°\n'
                f'CL/CD: {ld_v:.4f}   XCP: {xcp_v:.4f}'
            )
            render_geometry_panel(
                'ENV_GEO_CANVAS', 'E_GEO', geom,
                'Envelope — Alpha Sweep', push_history=False,
                pitch_deg=val, status_override=status_txt,
                fast_preview=True, fitness_t=t,
                overlay_text=overlay_txt,
                progress={'kind': 'Step', 'current': idx, 'total': tot,
                         'label': f'Alpha: {val:.2f}°   CL/CD: {ld_v:.4f}' + morph_suffix})
        elif kind == 'mach':
            status_txt = f'Step : {idx} / {tot}   |   Mach : {val:.3f}' + morph_suffix
            overlay_txt = (
                f'Mach Sweep — Step {idx} / {tot}\n'
                f'Mach: {val:.3f}\n'
                f'CL/CD: {ld_v:.4f}   XCP: {xcp_v:.4f}'
            )
            render_geometry_panel(
                'ENV_GEO_CANVAS', 'E_GEO', geom,
                'Envelope — Mach Sweep', push_history=False,
                pitch_deg=0.0, status_override=status_txt,
                fast_preview=True, fitness_t=t,
                overlay_text=overlay_txt,
                progress={'kind': 'Step', 'current': idx, 'total': tot,
                         'label': f'Mach: {val:.3f}   CL/CD: {ld_v:.4f}' + morph_suffix})
        elif kind == 'alt':
            status_txt = f'Step : {idx} / {tot}   |   Altitude : {val:.1f} m' + morph_suffix
            overlay_txt = (
                f'Altitude Sweep — Step {idx} / {tot}\n'
                f'Altitude: {val:.1f} m\n'
                f'CL/CD: {ld_v:.4f}   XCP: {xcp_v:.4f}'
            )
            render_geometry_panel(
                'ENV_GEO_CANVAS', 'E_GEO', geom,
                'Envelope — Altitude Sweep', push_history=False,
                pitch_deg=0.0, status_override=status_txt,
                fast_preview=True, fitness_t=t,
                overlay_text=overlay_txt,
                progress={'kind': 'Step', 'current': idx, 'total': tot,
                         'label': f'Altitude: {val:.1f} m   CL/CD: {ld_v:.4f}' + morph_suffix})
        return
    if event == 'ENV_ERR':
        e = values['ENV_ERR']
        set_status(f'ERROR  Envelope: {e}', color=C_RED)
        sg.popup_error(f'Flight Envelope Error:\n{e}')
        set_prog('PB_E', 'PP_E', 'PM_E', 0, 'error')
        _flt_run = False
        return


# =========================================================
# MAIN EVENT LOOP
# =========================================================
try:
    render_all_geometry_placeholders()
except Exception as _e:
    print(f'[WARN] initial geometry placeholder render failed: {_e}')

# Events that arrive as a rapid live stream (one per DE generation, one
# per alpha-sweep step) get coalesced below: if several are already
# queued up by the time we get around to handling one, we skip straight
# to the newest and drop the stale in-between frames instead of working
# through a backlog one at a time. This is what keeps the 3D preview
# feeling live/responsive rather than lagging behind and "catching up"
# after the optimizer/sweep has already finished. Any other event type
# encountered while draining is kept, in order, in `_pending_events` so
# nothing else is ever lost.
_pending_events = deque()
_COALESCE_EVENTS = {'OPT_GEN', 'ENV_STEP'}

while True:
    if _pending_events:
        event, values = _pending_events.popleft()
    else:
        event, values = window.read(timeout=300)

    if event in _COALESCE_EVENTS:
        while True:
            nxt_event, nxt_values = window.read(timeout=0)
            if nxt_event == sg.TIMEOUT_EVENT or nxt_event is None:
                break
            if nxt_event == event:
                values = nxt_values
                continue
            _pending_events.append((nxt_event, nxt_values))
            break

    if event in (sg.WINDOW_CLOSED, 'Exit', None):
        break

    if event == 'THEME_SELECT':
        new_theme = values['THEME_SELECT']
        if new_theme != CURRENT_THEME:
            apply_theme(new_theme)
            rebuild_window()
            render_all_geometry_panels()
        continue

    if event == 'FONT_SELECT':
        new_font = values['FONT_SELECT']
        if new_font != FONT_FAMILY:
            update_fonts(new_font)
            rebuild_window()
            render_all_geometry_panels()
        continue

    if event == sg.TIMEOUT_EVENT:
        if not _model_rdy:
            _startup_pct = min(_startup_pct + 2, 92)
            set_prog('PB_P', 'PP_P', 'PM_P', _startup_pct,
                     'Loading + training model...')
        try:
            while True:
                line = opt_log_q.get_nowait()
                con_append('OPT_LOG', line)
        except queue.Empty:
            pass
        continue

    if event == 'WIN_CFG':
        continue

    if event == 'MODEL_READY':
        elapsed = values['MODEL_READY']
        _model_rdy = True
        set_prog('PB_P', 'PP_P', 'PM_P', 100, 'Model ready')
        set_status(
            # f'OK  READY | Loaded in {elapsed:.2f} s | '
            # f'{"ENSEMBLE" if ENSEMBLE_MODE else "XGBoost-only"} | '
            f'ESTIMATE',
            color=C_GREEN)
        continue

    if event in ('F5:116', 'F5:65474', 'F5'):
        event = 'Estimate'

    # Window controls (PySimpleGUI API only)
    if event == 'W_MAX':
        if _is_max:
            try:
                window.normal()
            except Exception:
                pass
            _is_max = False
        else:
            try:
                window.maximize()
            except Exception:
                pass
            _is_max = True
        continue

    if event == 'W_MIN':
        try:
            window.minimize()
        except Exception:
            pass
        continue

    # SHOW_OPT_PLOTS button
    if event == 'SHOW_OPT_PLOTS':
        _show_opt_plot_popup()
        continue

    # APPLY_OPT_GEO button
    if event == 'APPLY_OPT_GEO':
        geom_to_use = _last_best_geom
        if not geom_to_use:
            try:
                geom_to_use, _ = _load_optimal_base('de_output')
            except Exception as ex:
                sg.popup_error(f"Could not load best geometry:\n{ex}")
                continue
        # Update E_{p} inputs
        for p in PARAMS[:15]:
            val = geom_to_use.get(p, DEFAULTS[p])
            window[f'E_{p}'].update(str(val))
        sg.popup_quick_message("Applied optimal geometry to envelope inputs!")
        continue

    # SHOW_ENV_PLOTS button
    if event == 'SHOW_ENV_PLOTS':
        _show_env_plot_popup()
        continue

    if event in _LIVE_EVENTS:
        _handle_live_event(event, values)
        continue


    # -- Tab 1: Prediction ---------------------------------
    if event == 'Estimate':
        _start_prediction_run(values)

    elif event == 'Reset_Pred':
        reset_pred()

    # -- Tab 2: Optimization -------------------------------
    elif event == 'Run_Opt':
        _start_optimizer_run(values)

    elif event == 'Abort_Opt':
        _opt_run = False
        set_status('Optimization aborted.', color=C_RED)
        set_prog('PB_O', 'PP_O', 'PM_O', 0, 'aborted')

    elif event == 'Abort_Env':
        _flt_run = False
        set_status('Flight envelope sweep aborted.', color=C_RED)
        set_prog('PB_E', 'PP_E', 'PM_E', 0, 'aborted')

    elif event == 'Clear_Opt':
        clear_opt()

    # -- Tab 3: Flight Envelope ----------------------------
    elif event == 'Run_Env':
        _start_envelope_run(values)

    elif event == 'Clear_Env':
        clear_env()

    elif event == 'Export_Env':
        export_envelope_csv()

    # -- Manual 3D preview refresh (no longer has its own button, but
    #    still fired internally e.g. right after Apply Best Geometry) --
    elif event == 'PRED_GEO_REFRESH':
        geom = {p: sf(values, p, DEFAULTS[p]) for p in PARAMS[:15]}
        render_geometry_panel('PRED_GEO_CANVAS', 'P_GEO', geom,
                              'Prediction Geometry')

    elif event == 'OPT_GEO_REFRESH':
        geom = _last_best_geom if _last_best_geom else \
            {p: sf(values, p, DEFAULTS[p]) for p in PARAMS[:15]}
        render_geometry_panel('OPT_GEO_CANVAS', 'O_GEO', geom,
                              'Optimizer Best Geometry', push_history=False)

    elif event == 'ENV_GEO_REFRESH':
        geom = {p: sf(values, f'E_{p}', DEFAULTS[p]) for p in PARAMS[:15]}
        render_geometry_panel('ENV_GEO_CANVAS', 'E_GEO', geom,
                              'Envelope Base Geometry')

    # -- VISUALIZE buttons (one per tab): refresh from this tab's
    #    current inputs, THEN open the large 3D popup. If nothing has
    #    been computed yet on this tab, this still gives a live view of
    #    the current input geometry rather than requiring Run first. --
    elif event == 'P_GEO_POPUP':
        # VISUALIZE now runs the prediction itself (using this tab's
        # current inputs) and opens the live 3D popup in the same
        # click. _start_prediction_run() already renders the input
        # geometry immediately (so the popup always has something
        # cached to show right away); once the model finishes,
        # render_prediction() -> render_geometry_panel() bumps this
        # canvas's live-frame counter, and the popup (already open)
        # picks up the finished result automatically.
        started = _start_prediction_run(values)
        if not started:
            # Model not ready / bad inputs -- already told to the user.
            # Fall back to a static view of the current inputs rather
            # than opening an empty/stale popup.
            geom = _last_pred_geom if _last_pred_geom else \
                {p: sf(values, p, DEFAULTS[p]) for p in PARAMS[:15]}
            render_geometry_panel('PRED_GEO_CANVAS', 'P_GEO', geom,
                                  'Prediction Geometry')
        _show_geo_popup('PRED_GEO_CANVAS')

    elif event == 'O_GEO_POPUP':
        # VISUALIZE now starts the optimization run itself (using
        # whatever Max Generations / bounds / constraints are currently
        # set on this tab) and opens the live 3D popup in the same
        # click -- so you watch the body evolve generation-by-generation
        # as it computes, instead of waiting for the run to finish and
        # the result panel to fill in first. If a run is already in
        # progress (e.g. started via RUN, or a previous VISUALIZE
        # click), this just attaches the popup to that run instead of
        # starting a second one.
        started_new_run = _start_optimizer_run(values)
        if started_new_run:
            # Seed an initial frame from the starting geometry right
            # away, before the first generation's live callback has had
            # a chance to land -- otherwise the popup would find nothing
            # cached yet for this canvas and refuse to open at all.
            geom = {p: sf(values, p, DEFAULTS[p]) for p in PARAMS[:15]}
            render_geometry_panel('OPT_GEO_CANVAS', 'O_GEO', geom,
                                  'Optimizer \u2014 Starting\u2026',
                                  push_history=False)
        elif not _opt_run:
            # Couldn't start a new run (model not ready / bad inputs --
            # _start_optimizer_run() already told the user why) and
            # nothing else is running either: fall back to a static view
            # of the last completed result, or the raw current inputs.
            geom = _last_best_geom if _last_best_geom else \
                {p: sf(values, p, DEFAULTS[p]) for p in PARAMS[:15]}
            render_geometry_panel('OPT_GEO_CANVAS', 'O_GEO', geom,
                                  'Optimizer Best Geometry', push_history=False)
        # else: a run was already in progress -- leave the canvas's
        # cached geometry alone; the popup picks up its live frames as
        # they land, same as always.
        _show_geo_popup('OPT_GEO_CANVAS')

    elif event == 'E_GEO_POPUP':
        # VISUALIZE now starts the envelope sweep itself (using the
        # current base geometry + alpha/mach/altitude ranges) and opens
        # the live 3D popup in the same click. Seed an immediate frame
        # of the starting base geometry so the popup always has
        # something cached to show right away; once the sweep's numbers
        # are computed, ENV_DONE kicks off _env_animate_worker, which
        # plays every alpha/mach/altitude point back on the body (see
        # its docstring) -- the popup (already open) follows that
        # playback live via the same _geo_live_seq polling every other
        # tab's popup uses.
        started_new_run = _start_envelope_run(values)
        base_geom = {p: sf(values, f'E_{p}', DEFAULTS[p]) for p in PARAMS[:15]}
        if started_new_run:
            render_geometry_panel('ENV_GEO_CANVAS', 'E_GEO', base_geom,
                                  'Envelope \u2014 Starting\u2026',
                                  push_history=False)
        elif not _flt_run:
            # Couldn't start a new sweep (model not ready / bad inputs
            # -- _start_envelope_run() already told the user why) and
            # nothing else is running: fall back to a static view of
            # the current base geometry.
            render_geometry_panel('ENV_GEO_CANVAS', 'E_GEO', base_geom,
                                  'Envelope Base Geometry')
        # else: a sweep/playback was already in progress -- leave the
        # canvas's cached geometry alone; the popup picks up its live
        # frames as they land, same as always.
        _show_geo_popup('ENV_GEO_CANVAS')

# Clean shutdown
try:
    _SWEEP_POOL.shutdown(wait=False)
except Exception:
    pass

window.close()