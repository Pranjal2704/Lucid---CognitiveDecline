"""
CogTrack Mouse Tracker
───────────────────────
Passive mouse movement analysis based on:
  - Hagler et al. 2011: KDE-based move/pause segmentation
  - Hagler et al. 2014: Cognitive parameter estimation (τ_R, τ_S, b)
  - Seelye et al. 2015: MCI biomarkers from mouse features

Collects mouse (x, y, t) via pynput, segments into moves/pauses,
extracts Fitts' law and curvature features, estimates TMT-A/B scores,
and flags MCI risk markers.

Runs as a background thread alongside eye and speech collectors.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin

try:
    from pynput import mouse as _mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants (Hagler 2011/2014, Seelye 2015)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TMT score calibration (Hagler 2014, eq. 12)
GLOBAL_T0_PLUS_25ALPHA = -9.1       # seconds
GLOBAL_BETA = 2.2
GLOBAL_THETA = 30.0                 # seconds per error
CHI_TILDE_A = 66.0                  # bits (TMT-A motor complexity)
CHI_TILDE_B = 74.0                  # bits (TMT-B motor complexity)
KAPPA_TMT = sum((25 - nu + 1) / 2.0 for nu in range(1, 26))  # 162.5

# Segmentation
V_CUT_SEARCH_LO = 10.0             # px/s
V_CUT_SEARCH_HI = 1000.0           # px/s
V_CUT_FALLBACK = 66.0              # px/s empirical default
MAX_PAUSE_SPEED_PX_S = 1.0         # < 1 px/s over 5s = pause
INTERP_DT = 0.016                  # 16 ms grid
KDE_BANDWIDTH = 0.15

# Recording
MIN_PIXELS_THRESHOLD = 5           # Manhattan distance filter
MIN_MOVES_FOR_ESTIMATION = 25

# Normative scores (Tombaugh 2004; ages 75-79)
NORMATIVE_TMT_A = 42.0
NORMATIVE_TMT_B = 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RawPoint:
    x: float
    y: float
    t: float  # seconds since session start


@dataclass
class MouseMove:
    """One segmented mouse move (Hagler 2011 / Seelye 2015)."""
    points: List[RawPoint]
    delta: float = 0.0          # straight-line distance, px
    D: float = 0.0              # total arc distance, px
    T: float = 0.0              # duration, ms
    K: float = 0.0              # curvature = delta/D (1=straight, 0=loop)
    idle_before: float = 0.0    # inter-move pause, ms
    log2_D_over_W_plus1: float = 0.0  # Fitts' law ID


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mouse Collector (pynput listener)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MouseCollector:
    """Background pynput listener recording (x, y, t) with ≥5px Manhattan filter."""

    def __init__(self):
        if not HAS_PYNPUT:
            raise RuntimeError("pynput not installed — pip install pynput")
        self._lock = threading.Lock()
        self._points: List[RawPoint] = []
        self._listener = None
        self._last_x = None
        self._last_y = None
        self._t0 = 0.0

    def _on_move(self, x, y):
        t = time.perf_counter() - self._t0
        with self._lock:
            if self._last_x is None:
                self._last_x, self._last_y = x, y
                self._points.append(RawPoint(x, y, t))
                return
            if abs(x - self._last_x) + abs(y - self._last_y) >= MIN_PIXELS_THRESHOLD:
                self._points.append(RawPoint(x, y, t))
                self._last_x, self._last_y = x, y

    def start(self):
        self._t0 = time.perf_counter()
        self._last_x = None
        self._points.clear()
        self._listener = _mouse.Listener(on_move=self._on_move)
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def snapshot(self) -> List[RawPoint]:
        with self._lock:
            return list(self._points)

    def point_count(self) -> int:
        with self._lock:
            return len(self._points)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Segmenter (Hagler 2011 KDE method)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Segmenter:
    """KDE-based move/pause segmentation (Hagler 2011)."""

    def segment(self, points: List[RawPoint]) -> Tuple[List[MouseMove], float]:
        if len(points) < 4:
            return [], V_CUT_FALLBACK

        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])
        ts = np.array([p.t for p in points])

        # Interpolate to uniform 16ms grid
        t_uniform = np.arange(ts[0], ts[-1], INTERP_DT)
        if len(t_uniform) < 4:
            return [], V_CUT_FALLBACK
        xi = np.interp(t_uniform, ts, xs)
        yi = np.interp(t_uniform, ts, ys)

        # Instantaneous speed
        dx, dy = np.diff(xi), np.diff(yi)
        speeds = np.sqrt(dx ** 2 + dy ** 2) / INTERP_DT

        # Find v_cut via KDE
        mask = speeds > MAX_PAUSE_SPEED_PX_S
        v_active = speeds[mask]
        v_cut = self._find_v_cut(v_active) if len(v_active) >= 10 else V_CUT_FALLBACK

        # Compute speed on original intervals
        orig_speeds = []
        for i in range(1, len(points)):
            d = math.hypot(points[i].x - points[i - 1].x,
                           points[i].y - points[i - 1].y)
            dt = max(points[i].t - points[i - 1].t, 1e-6)
            orig_speeds.append(d / dt)

        # Partition into moves and pauses
        moves = []
        current_pts = [points[0]]
        last_pause_end_t = None

        for i, spd in enumerate(orig_speeds):
            pt = points[i + 1]
            if spd >= v_cut:
                current_pts.append(pt)
            else:
                if len(current_pts) > 1:
                    move = self._build_move(current_pts)
                    if last_pause_end_t is not None:
                        move.idle_before = (current_pts[0].t - last_pause_end_t) * 1000.0
                    moves.append(move)
                last_pause_end_t = pt.t
                current_pts = [pt]

        if len(current_pts) > 1:
            move = self._build_move(current_pts)
            if last_pause_end_t is not None:
                move.idle_before = (current_pts[0].t - last_pause_end_t) * 1000.0
            moves.append(move)

        moves = [m for m in moves if len(m.points) > 1]
        return moves, v_cut

    def _find_v_cut(self, v_active):
        log_v = np.log10(v_active)
        try:
            kde = gaussian_kde(log_v, bw_method=KDE_BANDWIDTH)
            log_search = np.linspace(math.log10(V_CUT_SEARCH_LO),
                                     math.log10(V_CUT_SEARCH_HI), 500)
            density = kde(log_search)
            mins_idx = argrelmin(density, order=10)[0]
            mins_idx = mins_idx[(mins_idx > 0) & (mins_idx < len(density) - 1)]
            if len(mins_idx) == 0:
                return V_CUT_FALLBACK
            best = mins_idx[np.argmin(density[mins_idx])]
            return float(10 ** log_search[best])
        except Exception:
            return V_CUT_FALLBACK

    @staticmethod
    def _build_move(pts):
        move = MouseMove(points=pts)
        move.delta = math.hypot(pts[-1].x - pts[0].x, pts[-1].y - pts[0].y)
        arc = sum(math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
                  for i in range(1, len(pts)))
        move.D = arc if arc > 0 else 1e-6
        move.T = (pts[-1].t - pts[0].t) * 1000.0
        move.K = min(move.delta / move.D, 1.0)
        W = 40.0  # canonical icon size
        move.log2_D_over_W_plus1 = math.log2(move.D / W + 1.0)
        return move


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cognitive Parameter Estimator (Hagler 2014)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CognitiveParameterEstimator:
    """Fits τ_R+a, τ_S, b from move times (Hagler 2014 eq. 5-8)."""

    MEAN_SEARCH_STEPS = 2.0

    def fit(self, moves):
        if len(moves) < MIN_MOVES_FOR_ESTIMATION:
            return None

        t_vec = np.array([m.T / 1000.0 for m in moves])
        fitts = np.array([m.log2_D_over_W_plus1 for m in moves])
        n_search = np.full(len(moves), self.MEAN_SEARCH_STEPS)

        valid = t_vec < 5.0
        t_vec, fitts, n_search = t_vec[valid], fitts[valid], n_search[valid]

        if len(t_vec) < MIN_MOVES_FOR_ESTIMATION:
            return None

        try:
            from scipy.optimize import nnls
            best_sse = np.inf
            best_c = (0.0, 0.0, 0.0)
            for c0 in np.percentile(t_vec, np.linspace(5, 50, 20)):
                rhs = t_vec - c0
                A_sub = np.column_stack([n_search, fitts])
                coeffs, _ = nnls(A_sub, rhs)
                pred = c0 + A_sub @ coeffs
                sse = float(np.sum((t_vec - pred) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best_c = (c0, float(coeffs[0]), float(coeffs[1]))

            c0, c1, c2 = best_c
            ss_tot = float(np.var(t_vec) * len(t_vec))
            r2 = max(0.0, 1.0 - best_sse / ss_tot) if ss_tot > 0 else 0.0

            return {
                "tau_R_plus_a": max(c0, 0.05),
                "tau_S": c1,
                "b": c2,
                "n_moves": int(len(t_vec)),
                "r2": round(r2, 4),
            }
        except Exception:
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TMT Score Estimation (Hagler 2014 eq. 12)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def estimate_tmt_scores(fit: dict, n_errors_a=0.0, n_errors_b=1.0):
    """Returns (tmt_a, tmt_b) in seconds."""
    base = (GLOBAL_T0_PLUS_25ALPHA
            + 25.0 * GLOBAL_BETA * fit["tau_R_plus_a"]
            + KAPPA_TMT * fit["tau_S"])
    tmt_a = base + (25.0 / 24.0) * CHI_TILDE_A * fit["b"] + GLOBAL_THETA * n_errors_a
    tmt_b = base + (25.0 / 24.0) * CHI_TILDE_B * fit["b"] + GLOBAL_THETA * n_errors_b
    return max(15.0, min(tmt_a, 600.0)), max(15.0, min(tmt_b, 600.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature extraction & MCI flags
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _iqr(arr):
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def compute_mouse_metrics(moves: List[MouseMove], session_duration_s: float,
                          v_cut: float) -> dict:
    """Full metric extraction from segmented moves."""
    if not moves:
        return {"mouse_error": "no_moves"}

    delta_arr = np.array([m.delta for m in moves])
    D_arr = np.array([m.D for m in moves])
    T_arr = np.array([m.T for m in moves])
    K_arr = np.array([m.K for m in moves])
    idle_arr = np.array([m.idle_before for m in moves if m.idle_before > 0])

    metrics = {
        "mouse_n_moves": len(moves),
        "mouse_session_duration_s": round(session_duration_s, 1),
        "mouse_v_cut": round(v_cut, 1),
        # Seelye 2015 features
        "mouse_median_delta": round(float(np.median(delta_arr)), 1),
        "mouse_iqr_delta": round(_iqr(delta_arr), 1),
        "mouse_median_D": round(float(np.median(D_arr)), 1),
        "mouse_iqr_D": round(_iqr(D_arr), 1),
        "mouse_median_T_ms": round(float(np.median(T_arr)), 1),
        "mouse_iqr_T_ms": round(_iqr(T_arr), 1),
        "mouse_median_K": round(float(np.median(K_arr)), 4),
        "mouse_iqr_K": round(_iqr(K_arr), 4),
        "mouse_median_idle_ms": round(float(np.median(idle_arr)), 1) if len(idle_arr) > 0 else 0.0,
        "mouse_iqr_idle_ms": round(_iqr(idle_arr), 1) if len(idle_arr) > 0 else 0.0,
        # Means for longitudinal tracking
        "mouse_mean_delta": round(float(np.mean(delta_arr)), 1),
        "mouse_mean_D": round(float(np.mean(D_arr)), 1),
        "mouse_mean_T_ms": round(float(np.mean(T_arr)), 1),
        "mouse_mean_K": round(float(np.mean(K_arr)), 4),
        "mouse_std_K": round(float(np.std(K_arr)), 4),
    }

    # Cognitive parameter fit
    estimator = CognitiveParameterEstimator()
    fit = estimator.fit(moves)
    if fit:
        metrics["mouse_tau_R_plus_a_ms"] = round(fit["tau_R_plus_a"] * 1000, 1)
        metrics["mouse_tau_S_ms"] = round(fit["tau_S"] * 1000, 1)
        metrics["mouse_b_ms_per_bit"] = round(fit["b"] * 1000, 1)
        metrics["mouse_fit_r2"] = fit["r2"]

        # TMT estimates
        tmt_a, tmt_b = estimate_tmt_scores(fit)
        metrics["mouse_tmt_a_est"] = round(tmt_a, 1)
        metrics["mouse_tmt_b_est"] = round(tmt_b, 1)
        metrics["mouse_tmt_a_vs_norm"] = round(tmt_a - NORMATIVE_TMT_A, 1)
        metrics["mouse_tmt_b_vs_norm"] = round(tmt_b - NORMATIVE_TMT_B, 1)

    # MCI risk flags (Seelye 2015 Table 3)
    flags = []
    if metrics.get("mouse_iqr_K", 0) > 0.145:
        flags.append("high_curvature_variability")
    if metrics.get("mouse_iqr_idle_ms", 0) > 1050:
        flags.append("high_idle_variability")
    if 0 < metrics.get("mouse_median_delta", 999) < 40:
        flags.append("short_movement_distance")
    if 0 < metrics.get("mouse_median_D", 999) < 45:
        flags.append("short_arc_distance")
    metrics["mouse_mci_flags"] = flags
    metrics["mouse_mci_flag_count"] = len(flags)

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# High-level API for PassiveCollector integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MouseTracker:
    """
    Background mouse tracking with periodic scoring.

    Usage (standalone):
        tracker = MouseTracker()
        tracker.start()
        time.sleep(300)
        metrics = tracker.score()
        tracker.stop()

    Integration with PassiveCollector:
        - collector.py calls start/stop/score
        - score() returns a dict of ~25 metrics + TMT estimates
    """

    def __init__(self):
        if not HAS_PYNPUT:
            raise RuntimeError("pynput not installed — pip install pynput")
        self.collector = MouseCollector()
        self.segmenter = Segmenter()
        self._start_time = 0.0

    def start(self):
        self._start_time = time.perf_counter()
        self.collector.start()

    def stop(self):
        self.collector.stop()

    def score(self) -> dict:
        """Segment and score all collected mouse data. Non-destructive."""
        points = self.collector.snapshot()
        duration = time.perf_counter() - self._start_time

        if len(points) < 20:
            return {"mouse_error": "insufficient_data",
                    "mouse_raw_points": len(points)}

        moves, v_cut = self.segmenter.segment(points)
        if len(moves) < MIN_MOVES_FOR_ESTIMATION:
            return {"mouse_error": "insufficient_moves",
                    "mouse_raw_points": len(points),
                    "mouse_n_moves": len(moves)}

        return compute_mouse_metrics(moves, duration, v_cut)

    @property
    def point_count(self) -> int:
        return self.collector.point_count()