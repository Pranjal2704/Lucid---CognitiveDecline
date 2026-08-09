#!/usr/bin/env python3
"""
CogTrack — Passive Multimodal Cognitive Decline Tracker
═══════════════════════════════════════════════════════
Webcam-based eye tracking + speech analysis for longitudinal
monitoring of cognitive function.

Usage:
    python main.py eye                      Eye tracking (free viewing)
    python main.py eye --duration 60        Eye tracking for 60 seconds
    python main.py speech --file audio.wav  Analyze speech recording
    python main.py speech --record 30       Record 30s and analyze
    python main.py task prosaccade          Run prosaccade task
    python main.py task antisaccade         Run antisaccade task
    python main.py task pursuit             Run smooth pursuit task
    python main.py full                     Full assessment (all modalities)
    python main.py dashboard               Generate dashboard from sessions
    python main.py drift                    Run drift analysis
    python main.py sessions                 List past sessions
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Ensure the script's directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║             ▗▖   ▗▖ ▗▖ ▗▄▄▖▗▄▄▄▖▗▄▄▄                  ║")
    print("  ║             ▐▌   ▐▌ ▐▌▐▌     █  ▐▌  █                 ║")
    print("  ║             ▐▌   ▐▌ ▐▌▐▌     █  ▐▌  █                 ║")
    print("  ║             ▐▙▄▄▖▝▚▄▞▘▝▚▄▄▖▗▄█▄▖▐▙▄▄▀                 ║")
    print("  ║                                                       ║")
    print("  ║   Passive Multimodal Cognitive Decline Tracker        ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print()

def print_metrics(metrics: dict, title: str = "Session Metrics"):
    """Pretty-print metrics using rich if available, else plain."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(title=title, box=box.ROUNDED, border_style="cyan",
                      title_style="bold cyan", show_lines=False)
        table.add_column("Metric", style="dim", min_width=30)
        table.add_column("Value", justify="right", style="bold")

        # Group by prefix
        groups = {}
        for key, val in metrics.items():
            if key in ("source", "timestamp", "session_type", "trials", "error"):
                continue
            parts = key.split("_", 1)
            prefix = parts[0] if len(parts) > 1 else "other"
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((key, val))

        for group, items in groups.items():
            for key, val in items:
                if isinstance(val, float):
                    display = f"{val:.4f}" if abs(val) < 0.01 else f"{val:.2f}"
                elif isinstance(val, (list, dict)):
                    continue
                else:
                    display = str(val)
                table.add_row(key, display)

        console.print(table)

    except ImportError:
        # Fallback: plain print
        print(f"\n  ── {title} {'─' * (50 - len(title))}")
        for key, val in sorted(metrics.items()):
            if key in ("source", "timestamp", "session_type", "trials", "error"):
                continue
            if isinstance(val, (list, dict)):
                continue
            if isinstance(val, float):
                display = f"{val:.4f}" if abs(val) < 0.01 else f"{val:.2f}"
            else:
                display = str(val)
            print(f"  {key:40s}  {display:>12s}")
        print()


def cmd_eye(args):
    from eye_tracker import EyeTracker
    from session_manager import SessionManager

    tracker = EyeTracker()
    print("  Starting eye tracking session...")
    print("  Position your face centered in the camera.")
    print("  Press 'q' to stop.\n")

    tracker.start(display=True, duration_s=args.duration)
    metrics = tracker.get_metrics()

    print_metrics(metrics, "Eye Tracking Metrics")

    if not args.no_save:
        mgr = SessionManager()
        path = mgr.save_session(metrics, session_type="eye_tracking")
        print(f"  Session saved → {path}")

    return metrics


def cmd_speech(args):
    from speech_analyzer import SpeechAnalyzer
    from session_manager import SessionManager

    analyzer = SpeechAnalyzer()

    if args.file:
        print(f"  Analyzing: {args.file}")
        metrics = analyzer.analyze_file(args.file)
    else:
        duration = args.record or 30
        save_path = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        metrics = analyzer.record_and_analyze(duration=duration, save_path=save_path)

    print_metrics(metrics, "Speech Analysis Metrics")

    if not args.no_save:
        mgr = SessionManager()
        path = mgr.save_session(metrics, session_type="speech_analysis")
        print(f"  Session saved → {path}")

    return metrics


def cmd_task(args):
    from eye_tracker import EyeTracker
    from prosaccade_task import SaccadeTask, SmoothPursuitTask
    from session_manager import SessionManager

    tracker = EyeTracker()

    if args.task_type == "prosaccade":
        task = SaccadeTask(tracker)
        results = task.run_prosaccade(num_trials=args.trials)
    elif args.task_type == "antisaccade":
        task = SaccadeTask(tracker)
        results = task.run_antisaccade(num_trials=args.trials)
    elif args.task_type == "pursuit":
        task = SmoothPursuitTask(tracker)
        results = task.run(duration_s=args.duration or 30)
    else:
        print(f"  Unknown task type: {args.task_type}")
        return

    # Also get aggregate eye metrics from tracking during task
    eye_metrics = tracker.get_metrics()

    combined = {}
    combined.update({f"task_{k}": v for k, v in results.items() if k != "trials"})
    combined.update(eye_metrics)

    # Print task-specific results
    if "trials" not in results:
        print_metrics(results, f"{args.task_type.title()} Task Results")
    else:
        print_metrics({k: v for k, v in results.items() if k != "trials"},
                      f"{args.task_type.title()} Task Results")

    if not args.no_save:
        mgr = SessionManager()
        full_session = {"task_results": results, "eye_metrics": eye_metrics}
        path = mgr.save_session(combined, session_type=f"task_{args.task_type}")
        print(f"  Session saved → {path}")

    return combined


def cmd_full(args):
    """Full assessment: eye tracking → prosaccade → antisaccade → speech."""
    from eye_tracker import EyeTracker
    from speech_analyzer import SpeechAnalyzer
    from prosaccade_task import SaccadeTask, SmoothPursuitTask
    from session_manager import SessionManager

    print("  ═══ FULL COGNITIVE ASSESSMENT ═══\n")
    all_metrics = {}

    # 1. Free viewing eye tracking
    print("  ── Phase 1: Free Viewing (30s) ──")
    tracker = EyeTracker()
    tracker.start(display=True, duration_s=30)
    eye_metrics = tracker.get_metrics()
    all_metrics.update({f"freeview_{k}": v for k, v in eye_metrics.items()})
    print_metrics(eye_metrics, "Free Viewing Eye Metrics")

    # 2. Prosaccade task
    print("\n  ── Phase 2: Prosaccade Task ──")
    task = SaccadeTask(tracker)
    pro_results = task.run_prosaccade(num_trials=args.trials)
    all_metrics.update({f"prosaccade_{k}": v for k, v in pro_results.items() if k != "trials"})
    print_metrics({k: v for k, v in pro_results.items() if k != "trials"}, "Prosaccade Results")

    # 3. Antisaccade task
    print("\n  ── Phase 3: Antisaccade Task ──")
    task2 = SaccadeTask(tracker)
    anti_results = task2.run_antisaccade(num_trials=args.trials)
    all_metrics.update({f"antisaccade_{k}": v for k, v in anti_results.items() if k != "trials"})
    print_metrics({k: v for k, v in anti_results.items() if k != "trials"}, "Antisaccade Results")

    # 4. Smooth pursuit
    print("\n  ── Phase 4: Smooth Pursuit (20s) ──")
    pursuit = SmoothPursuitTask(tracker)
    pursuit_results = pursuit.run(duration_s=20)
    all_metrics.update({f"pursuit_{k}": v for k, v in pursuit_results.items()})
    print_metrics(pursuit_results, "Smooth Pursuit Results")

    # 5. Speech recording
    print("\n  ── Phase 5: Speech Recording (30s) ──")
    print("  Describe your day, read aloud, or speak freely.")
    analyzer = SpeechAnalyzer()
    speech_metrics = analyzer.record_and_analyze(
        duration=30,
        save_path=f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )
    all_metrics.update({f"speech_{k}": v for k, v in speech_metrics.items()
                        if not isinstance(v, (list, dict))})
    print_metrics(speech_metrics, "Speech Analysis Metrics")

    # Save combined session
    mgr = SessionManager()
    path = mgr.save_session(all_metrics, session_type="full_assessment")
    print(f"\n  Full assessment saved → {path}")
    print(f"  Total metrics extracted: {len(all_metrics)}")

    # Generate charts
    from dashboard import generate_all_charts
    generate_all_charts()

    return all_metrics


def cmd_dashboard(args):
    from dashboard import generate_all_charts
    generate_all_charts()


def cmd_drift(args):
    from session_manager import SessionManager
    from dashboard import generate_all_charts

    mgr = SessionManager()
    drift = mgr.compute_drift()

    if "error" in drift:
        print(f"  {drift['error']}")
        return

    print("\n  ── Cognitive Drift Report ──")
    print(f"  {'Metric':40s}  {'Z-Score':>8s}  {'Direction':>10s}  {'Baseline':>10s}  {'Recent':>10s}")
    print(f"  {'─' * 84}")

    for key, d in drift.items():
        flag = "⚠️ " if abs(d["z_score"]) > 2 else "  "
        print(f"  {flag}{key:38s}  {d['z_score']:>8.2f}  {d['direction']:>10s}"
              f"  {d['baseline_mean']:>10.3f}  {d['recent_mean']:>10.3f}")

    # Generate charts (includes longitudinal if enough data)
    generate_all_charts()


def cmd_sessions(args):
    from session_manager import SessionManager

    mgr = SessionManager()
    sessions = mgr.list_sessions()

    if not sessions:
        print("  No sessions found.")
        return

    print(f"\n  ── {len(sessions)} Sessions ──")
    for s in sessions:
        print(f"  {s['timestamp']:25s}  {s['type']:20s}  ({s['metric_count']} metrics)  {s['filepath']}")
    print()


def cmd_passive(args):
    """Run passive background monitoring."""
    from collector import PassiveCollector

    collector = PassiveCollector(
        show_preview=args.preview,
        enable_speech=not args.no_speech,
        enable_mouse=not args.no_mouse,
    )
    collector.run(duration_s=args.duration)


def cmd_charts(args):
    """Generate charts from session data."""
    from dashboard import generate_all_charts
    generate_all_charts(target_date=args.date)


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="CogTrack — Passive Multimodal Cognitive Decline Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Primary: Passive monitoring ──
    p_passive = subparsers.add_parser("passive", help="Run passive background monitoring (primary mode)")
    p_passive.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: run until Ctrl+C)")
    p_passive.add_argument("--preview", action="store_true", help="Show small webcam preview window")
    p_passive.add_argument("--no-speech", action="store_true", help="Disable speech monitoring")
    p_passive.add_argument("--no-mouse", action="store_true", help="Disable mouse tracking")

    # ── Active tasks (periodic checkpoints) ──
    p_eye = subparsers.add_parser("eye", help="Run eye tracking session")
    p_eye.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    p_eye.add_argument("--no-save", action="store_true", help="Don't save session")

    p_speech = subparsers.add_parser("speech", help="Run speech analysis")
    p_speech.add_argument("--file", type=str, help="Path to audio file")
    p_speech.add_argument("--record", type=float, default=None, help="Record N seconds")
    p_speech.add_argument("--no-save", action="store_true")

    p_task = subparsers.add_parser("task", help="Run oculomotor task")
    p_task.add_argument("task_type", choices=["prosaccade", "antisaccade", "pursuit"])
    p_task.add_argument("--trials", type=int, default=20)
    p_task.add_argument("--duration", type=float, default=None)
    p_task.add_argument("--no-save", action="store_true")

    p_full = subparsers.add_parser("full", help="Full cognitive assessment")
    p_full.add_argument("--trials", type=int, default=15)
    p_full.add_argument("--no-save", action="store_true")

    # ── Analysis ──
    subparsers.add_parser("dashboard", help="Generate dashboard visualizations")
    subparsers.add_parser("drift", help="Run cognitive drift analysis")
    subparsers.add_parser("sessions", help="List past sessions")

    p_charts = subparsers.add_parser("charts", help="Generate visual charts from session data")
    p_charts.add_argument("--date", type=str, default=None, help="Date to chart (YYYY-MM-DD, default: today)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n  Primary mode:")
        print("    python main.py passive              Run background monitoring")
        print("    python main.py passive --preview     With webcam preview")
        print("    python main.py passive --duration 3600  Monitor for 1 hour")
        print()
        return

    commands = {
        "passive": cmd_passive,
        "charts": cmd_charts,
        "eye": cmd_eye,
        "speech": cmd_speech,
        "task": cmd_task,
        "full": cmd_full,
        "dashboard": cmd_dashboard,
        "drift": cmd_drift,
        "sessions": cmd_sessions,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n  Session interrupted.")
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()