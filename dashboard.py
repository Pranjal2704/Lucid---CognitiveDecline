"""
Lucid Dashboard & Chart Generator
─────────────────────────────────────
Reads passive monitoring logs (JSONL segments, daily summaries)
and generates publication-quality charts.

Called automatically after each passive session and via:
    python main.py charts
    python main.py charts --date 2026-04-25
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, date

from config import DASHBOARD_DPI, PASSIVE_LOG_DIR, DAILY_SUMMARY_DIR, SESSION_DIR
from session_manager import SessionManager

# ── Theme ─────────────────────────────────────────────────────
BG      = "#06080d"
SURFACE = "#0f1520"
CARD    = "#141c2b"
BORDER  = "#1e293b"
TEXT    = "#cbd5e1"
DIM     = "#475569"
ACCENT  = "#22d3b7"
ACCENT2 = "#818cf8"
WARN    = "#f87171"
AMBER   = "#fbbf24"
GREEN   = "#34d399"

def _apply_theme(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes.flatten() if hasattr(axes, "flatten") else [axes]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(9)
        ax.title.set_fontweight("bold")


def _load_segments(target_date: str = None) -> list:
    """Load segment metrics from JSONL log for a specific date."""
    if target_date is None:
        target_date = date.today().isoformat()
    log_path = os.path.join(PASSIVE_LOG_DIR, f"{target_date}_segments.jsonl")
    if not os.path.isfile(log_path):
        # Try finding the most recent log
        logs = sorted(glob.glob(os.path.join(PASSIVE_LOG_DIR, "*_segments.jsonl")))
        if not logs:
            return []
        log_path = logs[-1]
        print(f"  Using log: {os.path.basename(log_path)}")

    segments = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    segments.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return segments


def _load_daily_summaries() -> list:
    """Load all daily summaries for longitudinal charts."""
    files = sorted(glob.glob(os.path.join(DAILY_SUMMARY_DIR, "*.json")))
    summaries = []
    for f in files:
        try:
            with open(f) as fh:
                summaries.append(json.load(fh))
        except Exception:
            pass
    return summaries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session Charts (generated after each passive run)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_session_charts(target_date: str = None,
                            output_dir: str = SESSION_DIR) -> list:
    """
    Generate all charts for a session. Returns list of saved file paths.
    """
    segments = _load_segments(target_date)
    if not segments:
        print("  No segment data found.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # Split eye vs speech segments
    eye_segs = [s for s in segments if s.get("activity") in ("READING", "SCANNING", "FOCUSED", "PASSIVE", "IDLE")]
    speech_segs = [s for s in segments if s.get("_segment_type") == "passive_speech"]

    if eye_segs:
        p = _chart_eye_session(eye_segs, output_dir, target_date)
        if p:
            paths.append(p)

    if speech_segs:
        p = _chart_speech_session(speech_segs, output_dir, target_date)
        if p:
            paths.append(p)

    if eye_segs:
        p = _chart_activity_timeline(eye_segs, output_dir, target_date)
        if p:
            paths.append(p)

    # Longitudinal (if multiple days exist)
    summaries = _load_daily_summaries()
    if len(summaries) >= 2:
        p = _chart_longitudinal(summaries, output_dir)
        if p:
            paths.append(p)

    return paths


def _chart_eye_session(segments, output_dir, target_date) -> str:
    """4-panel eye metrics chart for a single session."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Lucid — Eye Metrics  |  {target_date or date.today()}",
                 color=ACCENT, fontsize=13, fontweight="bold", y=0.98)

    # Extract time series
    fix_durs = [(i, s.get("fix_dur_mean", 0)) for i, s in enumerate(segments) if s.get("fix_dur_mean", 0) > 0]
    sac_vels = [(i, s.get("sac_vel_peak_mean", 0)) for i, s in enumerate(segments) if s.get("sac_vel_peak_mean", 0) > 0]
    pupils   = [(i, s.get("pupil_mean", 0)) for i, s in enumerate(segments) if s.get("pupil_mean", 0) > 0]
    activities = [s.get("activity", "?") for s in segments]

    # ── Panel 1: Fixation duration over time ──
    ax = axes[0, 0]
    if fix_durs:
        x, y = zip(*fix_durs)
        ax.fill_between(x, y, alpha=0.15, color=ACCENT)
        ax.plot(x, y, color=ACCENT, linewidth=1.5, marker=".", markersize=3)
        ax.axhline(np.mean(y), color=DIM, linestyle="--", linewidth=0.8)
    ax.set_title("Fixation Duration (ms)")
    ax.set_xlabel("Segment")

    # ── Panel 2: Saccade peak velocity ──
    ax = axes[0, 1]
    if sac_vels:
        x, y = zip(*sac_vels)
        ax.fill_between(x, y, alpha=0.15, color=ACCENT2)
        ax.plot(x, y, color=ACCENT2, linewidth=1.5, marker=".", markersize=3)
        ax.axhline(np.mean(y), color=DIM, linestyle="--", linewidth=0.8)
    ax.set_title("Saccade Peak Velocity (°/s)")
    ax.set_xlabel("Segment")

    # ── Panel 3: Activity breakdown (stacked bar) ──
    ax = axes[1, 0]
    act_colors = {"READING": ACCENT, "SCANNING": ACCENT2, "FOCUSED": AMBER, "PASSIVE": DIM, "IDLE": "#1a1a2e"}
    act_counts = {}
    for a in activities:
        act_counts[a] = act_counts.get(a, 0) + 1
    if act_counts:
        labels = list(act_counts.keys())
        sizes = list(act_counts.values())
        colors = [act_colors.get(l, DIM) for l in labels]
        bars = ax.barh(labels, sizes, color=colors, height=0.6, edgecolor=BORDER)
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8, color=TEXT)
    ax.set_title("Activity Segments")

    # ── Panel 4: Pupil diameter ──
    ax = axes[1, 1]
    if pupils:
        x, y = zip(*pupils)
        ax.fill_between(x, y, alpha=0.15, color=AMBER)
        ax.plot(x, y, color=AMBER, linewidth=1.5, marker=".", markersize=3)
        ax.axhline(np.mean(y), color=DIM, linestyle="--", linewidth=0.8)
    ax.set_title("Pupil Diameter (px)")
    ax.set_xlabel("Segment")

    _apply_theme(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, f"eye_session_{target_date or date.today()}.png")
    fig.savefig(path, dpi=DASHBOARD_DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path


def _chart_speech_session(segments, output_dir, target_date) -> str:
    """4-panel speech metrics chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Lucid — Speech Metrics  |  {target_date or date.today()}",
                 color=ACCENT2, fontsize=13, fontweight="bold", y=0.98)

    f0s = [s.get("f0_mean", 0) for s in segments if s.get("f0_mean", 0) > 0]
    jitters = [s.get("jitter_local", 0) for s in segments]
    hnrs = [s.get("hnr_mean", 0) for s in segments if s.get("hnr_mean", 0) != 0]
    speech_rates = [s.get("speech_rate_syl_per_s", 0) for s in segments if s.get("speech_rate_syl_per_s", 0) > 0]
    pause_rates = [s.get("pause_rate_per_min", 0) for s in segments]

    # Panel 1: Pitch
    ax = axes[0, 0]
    if f0s:
        ax.bar(range(len(f0s)), f0s, color=ACCENT, alpha=0.7, edgecolor=BORDER)
        ax.axhline(np.mean(f0s), color=DIM, linestyle="--", linewidth=0.8)
    ax.set_title("Pitch / F0 (Hz)")
    ax.set_xlabel("Speech Segment")

    # Panel 2: Jitter
    ax = axes[0, 1]
    if jitters:
        colors = [WARN if j > 0.02 else ACCENT for j in jitters]
        ax.bar(range(len(jitters)), [j * 100 for j in jitters], color=colors, alpha=0.7, edgecolor=BORDER)
        ax.axhline(1.04, color=WARN, linestyle=":", linewidth=0.8, label="Healthy threshold")
    ax.set_title("Jitter Local (%)")
    ax.set_xlabel("Speech Segment")

    # Panel 3: HNR
    ax = axes[1, 0]
    if hnrs:
        colors = [WARN if h < 10 else GREEN for h in hnrs]
        ax.bar(range(len(hnrs)), hnrs, color=colors, alpha=0.7, edgecolor=BORDER)
        ax.axhline(20, color=GREEN, linestyle=":", linewidth=0.8, label="Healthy minimum")
    ax.set_title("Harmonics-to-Noise Ratio (dB)")
    ax.set_xlabel("Speech Segment")

    # Panel 4: Speech Rate + Pauses
    ax = axes[1, 1]
    x = range(len(speech_rates))
    if speech_rates:
        ax.bar(x, speech_rates, color=ACCENT2, alpha=0.7, label="Speech rate (syl/s)", edgecolor=BORDER)
    if pause_rates and len(pause_rates) == len(speech_rates):
        ax2 = ax.twinx()
        ax2.plot(range(len(pause_rates)), pause_rates, color=AMBER, marker="o", markersize=4, linewidth=1.5, label="Pause rate")
        ax2.tick_params(colors=AMBER, labelsize=7)
        ax2.spines["right"].set_color(AMBER)
        ax2.set_ylabel("Pauses/min", color=AMBER, fontsize=8)
    ax.set_title("Speech Rate & Pauses")
    ax.set_xlabel("Speech Segment")

    _apply_theme(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, f"speech_session_{target_date or date.today()}.png")
    fig.savefig(path, dpi=DASHBOARD_DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path


def _chart_activity_timeline(segments, output_dir, target_date) -> str:
    """Timeline strip showing activity classification over time."""
    fig, ax = plt.subplots(figsize=(16, 3))
    fig.suptitle(f"Lucid — Activity Timeline  |  {target_date or date.today()}",
                 color=TEXT, fontsize=11, fontweight="bold", y=0.95)

    act_colors = {"READING": ACCENT, "SCANNING": ACCENT2, "FOCUSED": AMBER, "PASSIVE": DIM, "IDLE": "#1a1a2e"}

    for i, seg in enumerate(segments):
        act = seg.get("activity", "PASSIVE")
        color = act_colors.get(act, DIM)
        dur = seg.get("duration_s", 5)
        conf = seg.get("confidence", 0.5)
        ax.barh(0, dur, left=i * 2.5, height=0.6, color=color, alpha=max(0.3, conf), edgecolor=BORDER, linewidth=0.5)

    # Legend
    for act, color in act_colors.items():
        if act != "IDLE":
            ax.barh([], [], color=color, label=act, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, ncol=4)

    ax.set_yticks([])
    ax.set_xlabel("Time (segments × 2.5s stride)", fontsize=8)
    ax.set_xlim(0, len(segments) * 2.5)

    _apply_theme(fig, ax)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    path = os.path.join(output_dir, f"timeline_{target_date or date.today()}.png")
    fig.savefig(path, dpi=DASHBOARD_DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path


def _chart_longitudinal(summaries, output_dir) -> str:
    """Multi-day trend chart from daily summaries."""
    if len(summaries) < 2:
        return None

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle("Lucid — Longitudinal Trends",
                 color=ACCENT, fontsize=13, fontweight="bold", y=0.99)

    dates = [s.get("date", f"Day {i}") for i, s in enumerate(summaries)]
    x = range(len(dates))
    short_dates = [d[-5:] if len(d) > 5 else d for d in dates]  # MM-DD

    # Metrics to track with (key_pattern, title, color, panel_row, panel_col)
    tracked = [
        ("read_fix_dur_mean_mean", "Read: Fix Duration (ms)", ACCENT, 0, 0),
        ("read_sac_vel_peak_mean_mean", "Read: Sac Velocity (°/s)", ACCENT2, 0, 1),
        ("read_read_regression_rate_mean", "Read: Regression Rate", WARN, 0, 2),
        ("scan_sac_vel_peak_mean_mean", "Scan: Sac Velocity (°/s)", ACCENT2, 1, 0),
        ("scan_scan_gaze_path_vel_mean", "Scan: Gaze Path Velocity", GREEN, 1, 1),
        ("read_pupil_load_range_mean", "Pupil Load Range", AMBER, 1, 2),
        ("speech_f0_mean", "Speech: Pitch (Hz)", ACCENT, 2, 0),
        ("speech_jitter_local", "Speech: Jitter", WARN, 2, 1),
        ("speech_speech_rate_syl_per_s", "Speech: Rate (syl/s)", GREEN, 2, 2),
    ]

    for key, title, color, row, col in tracked:
        ax = axes[row, col]
        vals = []
        for s in summaries:
            v = s.get(key, None)
            if v is None:
                # Try without prefix
                for k, val in s.items():
                    if key.split("_", 1)[-1] in k and isinstance(val, (int, float)):
                        v = val
                        break
            vals.append(float(v) if v is not None else np.nan)

        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if valid:
            vx, vy = zip(*valid)
            ax.fill_between(vx, vy, alpha=0.1, color=color)
            ax.plot(vx, vy, color=color, linewidth=2, marker="o", markersize=4)
            # Baseline band (first 3 points or all if fewer)
            n_bl = min(3, len(vy))
            bl_mean = np.mean(vy[:n_bl])
            bl_std = np.std(vy[:n_bl]) if n_bl > 1 else 0
            ax.axhline(bl_mean, color=DIM, linestyle="--", linewidth=0.7)
            if bl_std > 0:
                ax.axhspan(bl_mean - 2 * bl_std, bl_mean + 2 * bl_std, color=DIM, alpha=0.08)

        ax.set_title(title, fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_dates, fontsize=6, rotation=45)

    _apply_theme(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "longitudinal_trends.png")
    fig.savefig(path, dpi=DASHBOARD_DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API for CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_all_charts(target_date: str = None):
    """Entry point for `python main.py charts`."""
    print("\n  ── Generating Charts ──")
    paths = generate_session_charts(target_date)
    if not paths:
        # Fallback: try using session manager data (from full/eye/speech runs)
        print("  No passive logs found. Checking active session data...")
        mgr = SessionManager()
        sessions = mgr.list_sessions()
        if sessions:
            latest = mgr.load_session(sessions[-1]["filepath"])
            metrics = latest.get("metrics", {})
            if metrics:
                path = _chart_from_active_session(metrics)
                if path:
                    paths.append(path)
    if paths:
        print(f"\n  Generated {len(paths)} chart(s).")
    else:
        print("  No data available. Run a session first:")
        print("    python main.py passive --duration 60")
    return paths


def _chart_from_active_session(metrics: dict) -> str:
    """Generate a summary chart from an active session (full/eye/speech run)."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Lucid — Active Session Summary",
                 color=ACCENT, fontsize=13, fontweight="bold", y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Collect available metric groups
    eye_keys = {
        "Fix Duration": "freeview_fix_dur_mean_ms",
        "Sac Peak Vel": "freeview_sac_peak_vel_mean",
        "Blink Rate": "freeview_blink_rate_per_min",
        "Gaze Entropy": "freeview_gaze_entropy_bits",
        "Path Efficiency": "freeview_sac_efficiency_mean",
        "Main Seq Slope": "freeview_sac_main_sequence_slope",
    }
    speech_keys = {
        "Pitch (F0)": "speech_f0_mean",
        "Jitter": "speech_jitter_local",
        "HNR": "speech_hnr_mean",
        "Speech Rate": "speech_speech_rate_syl_per_s",
        "Pause Rate": "speech_pause_rate_per_min",
        "Shimmer": "speech_shimmer_local",
    }

    # Eye bar chart
    ax = fig.add_subplot(gs[0, :2])
    labels, values = [], []
    for label, key in eye_keys.items():
        v = metrics.get(key, 0)
        if v:
            labels.append(label)
            values.append(float(v))
    if values:
        colors = [ACCENT if i % 2 == 0 else ACCENT2 for i in range(len(values))]
        ax.barh(labels, values, color=colors, height=0.5, edgecolor=BORDER)
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.02, i, f"{v:.2f}", va="center", fontsize=8, color=TEXT)
    ax.set_title("Eye Tracking Metrics", fontsize=10)

    # Speech bar chart
    ax = fig.add_subplot(gs[1, :2])
    labels, values = [], []
    for label, key in speech_keys.items():
        v = metrics.get(key, 0)
        labels.append(label)
        values.append(float(v))
    if any(v > 0 for v in values):
        colors = [GREEN if v > 0 else DIM for v in values]
        ax.barh(labels, values, color=colors, height=0.5, edgecolor=BORDER)
        for i, v in enumerate(values):
            ax.text(max(v, 0) + max(max(values), 1) * 0.02, i, f"{v:.4f}" if v < 0.1 else f"{v:.1f}",
                    va="center", fontsize=8, color=TEXT)
    ax.set_title("Speech Metrics", fontsize=10)

    # Task results (if available)
    ax = fig.add_subplot(gs[:, 2])
    task_info = []
    for prefix, name in [("prosaccade", "Prosaccade"), ("antisaccade", "Antisaccade"), ("pursuit", "Pursuit")]:
        acc = metrics.get(f"{prefix}_accuracy_pct", None)
        rt = metrics.get(f"{prefix}_rt_mean_ms", None)
        gain = metrics.get(f"{prefix}_pursuit_gain", None)
        if acc is not None:
            task_info.append(f"{name}: Acc={acc}%")
        if rt is not None:
            task_info.append(f"  RT={rt}ms")
        if gain is not None:
            task_info.append(f"{name}: Gain={gain}")
    if task_info:
        for i, line in enumerate(task_info):
            color = TEXT if not line.startswith(" ") else DIM
            ax.text(0.1, 0.9 - i * 0.08, line, fontsize=9, color=color,
                    transform=ax.transAxes, fontfamily="monospace")
    ax.set_title("Task Results", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    _apply_theme(fig, np.array(fig.axes))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(SESSION_DIR, "active_session_summary.png")
    fig.savefig(path, dpi=DASHBOARD_DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path