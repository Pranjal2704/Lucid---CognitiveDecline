"""
CogTrack Passive Collector
───────────────────────────
Background daemon that continuously captures webcam frames,
runs FaceLandmarker, classifies activity, and extracts metrics.
Designed to run silently while the user works normally.

Architecture:
  Camera → FaceLandmarker → GazeSample stream
    → ActivityClassifier (5s windows) → ActivitySegments
    → MetricExtractor (per-segment) → segment metrics
    → DailySummary (end of day) → longitudinal store

No stimulus presentation. No interruptions. No tasks.
Just watches and measures.
"""

import os
import sys
import time
import json
import signal
import threading
from datetime import datetime, date
from collections import deque
from typing import Optional

import numpy as np
import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    GAZE_HISTORY_MAX, ACTIVITY_WINDOW_S, ACTIVITY_STRIDE_S,
    PASSIVE_LOG_DIR, DAILY_SUMMARY_DIR,
)
from eye_tracker import EyeTracker, GazeSample, create_landmarker, VisionRunningMode
from activity_classifier import ActivityClassifier, ActivitySegment
from metric_extractor import MetricExtractor
from session_manager import SessionManager

# Speech collector is optional (needs sounddevice)
try:
    from speech_collector import SpeechCollector
    HAS_SPEECH = True
except Exception:
    HAS_SPEECH = False

# Mouse tracker is optional (needs pynput)
try:
    from mouse_tracker import MouseTracker
    HAS_MOUSE = True
except Exception:
    HAS_MOUSE = False


class PassiveCollector:
    """
    Background gaze + speech collection and cognitive metric extraction.

    Usage:
        collector = PassiveCollector()
        collector.run()              # blocking — Ctrl+C to stop
        # or
        collector.run(duration_s=3600)  # run for 1 hour
    """

    def __init__(self, show_preview: bool = False, enable_speech: bool = True,
                 enable_mouse: bool = True):
        self.tracker = EyeTracker()
        self.classifier = ActivityClassifier()
        self.extractor = MetricExtractor()
        self.session_mgr = SessionManager()

        self.show_preview = show_preview
        self._running = False
        self._gaze_buffer = deque(maxlen=GAZE_HISTORY_MAX)
        self._segment_metrics = []  # eye segment metrics for the day
        self._speech_metrics = []   # speech segment metrics for the day
        self._mouse_metrics = []    # mouse scoring results for the day
        self._segments_processed = 0
        self._last_classify_ms = 0
        self._last_mouse_score_ms = 0
        self._today = date.today()

        # Speech collector (background thread)
        self._speech_collector = None
        self._enable_speech = enable_speech and HAS_SPEECH

        # Mouse tracker (background thread via pynput)
        self._mouse_tracker = None
        self._enable_mouse = enable_mouse and HAS_MOUSE
        self.MOUSE_SCORE_INTERVAL_MS = 120_000  # score mouse every 2 min

        # Ensure output dirs
        os.makedirs(PASSIVE_LOG_DIR, exist_ok=True)
        os.makedirs(DAILY_SUMMARY_DIR, exist_ok=True)

    def run(self, duration_s: Optional[float] = None):
        """
        Main collection loop. Blocks until Ctrl+C or duration expires.
        """
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not cap.isOpened():
            print("  ERROR: Cannot open camera.")
            return

        landmarker = create_landmarker(VisionRunningMode.VIDEO)
        self._running = True
        start_time = time.time()
        self.tracker.session_start = start_time * 1000
        frame_count = 0
        face_detected_count = 0

        # Handle Ctrl+C gracefully
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda s, f: self._stop())

        print()
        print("  ┌──────────────────────────────────────────────┐")
        print("  │  COGTRACK PASSIVE MONITORING                 │")
        print("  │  Running in background. Ctrl+C to stop.     │")
        print("  │  No tasks. No interruptions. Just observing. │")
        print("  └──────────────────────────────────────────────┘")
        print()

        # Start speech monitoring thread
        if self._enable_speech:
            try:
                self._speech_collector = SpeechCollector()
                self._speech_collector.start()
            except Exception as e:
                print(f"  Speech monitoring unavailable: {e}")
                self._speech_collector = None
        else:
            if not HAS_SPEECH:
                print("  Speech monitoring disabled (sounddevice not installed).")

        # Start mouse tracking
        if self._enable_mouse:
            try:
                self._mouse_tracker = MouseTracker()
                self._mouse_tracker.start()
                print("  Mouse tracking started (pynput listener).")
            except Exception as e:
                print(f"  Mouse tracking unavailable: {e}")
                self._mouse_tracker = None
        else:
            if not HAS_MOUSE:
                print("  Mouse tracking disabled (pynput not installed).")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                frame_count += 1
                now = time.time() * 1000
                timestamp_ms = int(now - self.tracker.session_start)

                # Check duration
                if duration_s and (time.time() - start_time) > duration_s:
                    break

                # Check for day rollover
                if date.today() != self._today:
                    self._end_of_day()
                    self._today = date.today()

                # Run FaceLandmarker
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception:
                    continue

                if result.face_landmarks:
                    face_detected_count += 1
                    landmarks = result.face_landmarks[0]
                    sample = self.tracker._extract_gaze(landmarks, frame.shape, now)
                    if sample:
                        self._gaze_buffer.append(sample)

                # Periodic classification (every stride interval)
                window_ms = ACTIVITY_WINDOW_S * 1000
                stride_ms = ACTIVITY_STRIDE_S * 1000
                if len(self._gaze_buffer) > 30 and (now - self._last_classify_ms) >= stride_ms:
                    self._classify_and_extract()
                    self._drain_speech_metrics()
                    self._score_mouse_if_due(now)
                    self._last_classify_ms = now

                # Optional preview
                if self.show_preview:
                    self._draw_preview(frame, result, now, frame_count, face_detected_count, start_time)
                    cv2.imshow("CogTrack Passive", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Status update every 30s
                elapsed = time.time() - start_time
                if frame_count % (CAMERA_FPS * 30) == 0 and frame_count > 0:
                    face_pct = face_detected_count / frame_count * 100
                    speech_str = ""
                    if self._speech_collector:
                        speaking = "🎤" if self._speech_collector.is_speaking else "  "
                        speech_str = (f"speech_segs={self._speech_collector.segments_analyzed} "
                                      f"{speaking}")
                    mouse_str = ""
                    if self._mouse_tracker:
                        mouse_str = f"mouse_pts={self._mouse_tracker.point_count}"
                    print(f"  [{elapsed / 60:.0f}m] "
                          f"frames={frame_count} "
                          f"face={face_pct:.0f}% "
                          f"gaze={len(self._gaze_buffer)} "
                          f"eye_segs={self._segments_processed} "
                          f"{speech_str}"
                          f"{mouse_str}")

        finally:
            self._running = False
            cap.release()
            landmarker.close()
            if self.show_preview:
                cv2.destroyAllWindows()
            signal.signal(signal.SIGINT, original_sigint)

            # Stop speech collector
            if self._speech_collector:
                self._speech_collector.stop()
                self._drain_speech_metrics()

            # Stop mouse tracker and do final score
            if self._mouse_tracker:
                final_mouse = self._mouse_tracker.score()
                if "mouse_error" not in final_mouse:
                    final_mouse["timestamp"] = datetime.now().isoformat()
                    self._mouse_metrics.append(final_mouse)
                    self._append_segment_log(final_mouse)
                self._mouse_tracker.stop()

            # Final processing
            self._classify_and_extract()
            self._end_of_day()

            elapsed = time.time() - start_time
            print(f"\n  Session ended after {elapsed / 60:.1f} minutes.")
            print(f"  Frames: {frame_count}, Eye segments: {self._segments_processed}")
            if self._speech_collector:
                print(f"  Speech segments: {self._speech_collector.segments_analyzed}, "
                      f"Speech time: {self._speech_collector.total_speech_s:.0f}s")
            if self._mouse_tracker:
                print(f"  Mouse points: {self._mouse_tracker.point_count}, "
                      f"Mouse scores: {len(self._mouse_metrics)}")
            print(f"  Metrics saved to {DAILY_SUMMARY_DIR}/")

    def _stop(self):
        self._running = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Classification & Extraction Pipeline
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _classify_and_extract(self):
        """Classify recent gaze data and extract metrics."""
        if len(self._gaze_buffer) < 30:
            return

        # Get the last window of samples
        window_ms = ACTIVITY_WINDOW_S * 1000
        samples = list(self._gaze_buffer)
        now = samples[-1].t
        window_start = now - window_ms

        window_samples = [s for s in samples if s.t >= window_start]
        if len(window_samples) < 30:
            return

        # Classify
        segment = self.classifier.classify_window(window_samples)
        self._segments_processed += 1

        # Extract metrics
        metrics = self.extractor.extract(window_samples, segment)
        metrics["timestamp"] = datetime.now().isoformat()
        self._segment_metrics.append(metrics)

        # Save raw segment to log (append mode)
        self._append_segment_log(metrics)

    def _append_segment_log(self, metrics: dict):
        """Append a segment's metrics to today's log file."""
        log_path = os.path.join(PASSIVE_LOG_DIR,
                                f"{self._today.isoformat()}_segments.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

    def _drain_speech_metrics(self):
        """Collect any pending speech analysis results from background thread."""
        if not self._speech_collector:
            return
        pending = self._speech_collector.get_pending()
        for m in pending:
            m["timestamp"] = datetime.now().isoformat()
            self._speech_metrics.append(m)
            self._append_segment_log(m)

    def _score_mouse_if_due(self, now_ms):
        """Score mouse data every MOUSE_SCORE_INTERVAL_MS."""
        if not self._mouse_tracker:
            return
        if (now_ms - self._last_mouse_score_ms) < self.MOUSE_SCORE_INTERVAL_MS:
            return
        self._last_mouse_score_ms = now_ms

        metrics = self._mouse_tracker.score()
        if "mouse_error" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
            self._mouse_metrics.append(metrics)
            self._append_segment_log(metrics)

            # Print MCI flags if any
            flags = metrics.get("mouse_mci_flags", [])
            if flags:
                print(f"  ⚠ Mouse MCI markers: {', '.join(flags)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Daily Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _end_of_day(self):
        """Compute and save daily summary with eye + speech + mouse metrics."""
        if not self._segment_metrics and not self._speech_metrics and not self._mouse_metrics:
            return

        # Eye metrics daily summary
        summary = self.extractor.compute_daily_summary(self._segment_metrics) if self._segment_metrics else {}
        summary["date"] = self._today.isoformat()
        summary["eye_segment_count"] = len(self._segment_metrics)

        # Speech metrics daily summary
        if self._speech_metrics:
            summary["speech_segment_count"] = len(self._speech_metrics)
            summary["speech_total_s"] = round(
                sum(m.get("_duration_s", m.get("duration_s", 0)) for m in self._speech_metrics), 1)

            speech_keys = [
                "f0_mean", "f0_std", "f0_cv", "f0_voiced_fraction",
                "jitter_local", "jitter_rap",
                "shimmer_local", "shimmer_local_db",
                "hnr_mean", "nhr_mean",
                "pause_count", "pause_rate_per_min", "pause_mean_s",
                "long_pause_count", "speech_rate_syl_per_s",
                "articulation_rate_syl_per_s", "phonation_ratio",
                "spectral_centroid_mean", "spectral_tilt",
                "rms_mean", "zcr_mean", "energy_entropy",
            ]
            for key in speech_keys:
                vals = [m[key] for m in self._speech_metrics
                        if key in m and isinstance(m[key], (int, float)) and m[key] != 0]
                if vals:
                    summary[f"speech_{key}"] = round(float(np.mean(vals)), 4)

        # Mouse metrics daily summary
        if self._mouse_metrics:
            summary["mouse_score_count"] = len(self._mouse_metrics)
            mouse_keys = [
                "mouse_n_moves", "mouse_v_cut",
                "mouse_median_delta", "mouse_iqr_delta",
                "mouse_median_D", "mouse_iqr_D",
                "mouse_median_T_ms", "mouse_iqr_T_ms",
                "mouse_median_K", "mouse_iqr_K",
                "mouse_median_idle_ms", "mouse_iqr_idle_ms",
                "mouse_tau_R_plus_a_ms", "mouse_tau_S_ms", "mouse_b_ms_per_bit",
                "mouse_fit_r2",
                "mouse_tmt_a_est", "mouse_tmt_b_est",
                "mouse_tmt_a_vs_norm", "mouse_tmt_b_vs_norm",
                "mouse_mci_flag_count",
            ]
            for key in mouse_keys:
                vals = [m[key] for m in self._mouse_metrics
                        if key in m and isinstance(m[key], (int, float))]
                if vals:
                    summary[key] = round(float(np.mean(vals)), 2)

        # Save daily summary
        summary_path = os.path.join(DAILY_SUMMARY_DIR,
                                    f"{self._today.isoformat()}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Also save to session manager for longitudinal tracking
        self.session_mgr.save_session(summary, session_type="passive_daily")

        print(f"\n  ── Daily Summary ({self._today}) ──")
        print(f"  Eye segments: {len(self._segment_metrics)}")
        print(f"  Speech segments: {len(self._speech_metrics)}")
        print(f"  Mouse scores: {len(self._mouse_metrics)}")
        total_dur = summary.get("total_duration_s", 0)
        print(f"  Eye duration: {total_dur / 60:.1f} min")
        if self._speech_metrics:
            print(f"  Speech duration: {summary.get('speech_total_s', 0):.0f}s")
        if self._mouse_metrics:
            tmt_a = summary.get("mouse_tmt_a_est", "—")
            tmt_b = summary.get("mouse_tmt_b_est", "—")
            print(f"  TMT-A est: {tmt_a}s  TMT-B est: {tmt_b}s")
        for key in sorted(summary.keys()):
            if key.startswith("pct_"):
                print(f"  {key}: {summary[key]}%")
        print(f"  Saved → {summary_path}")

        # Auto-generate charts
        try:
            from dashboard import generate_session_charts
            print("\n  ── Generating Charts ──")
            chart_paths = generate_session_charts(self._today.isoformat())
            if chart_paths:
                print(f"  {len(chart_paths)} chart(s) generated.")
        except Exception as e:
            print(f"  Chart generation skipped: {e}")

        # Reset for next day
        self._segment_metrics = []
        self._speech_metrics = []
        self._mouse_metrics = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Preview overlay
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _draw_preview(self, frame, result, now, frame_count, face_count, start_time):
        h, w = frame.shape[:2]

        # Draw iris if detected
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            for idx in [468, 473]:  # iris centers
                pt = (int(lm[idx].x * w), int(lm[idx].y * h))
                cv2.circle(frame, pt, 3, (0, 255, 200), -1)

        # Status bar
        elapsed = time.time() - start_time
        latest_activity = self._segment_metrics[-1].get("activity", "—") if self._segment_metrics else "—"
        status_lines = [
            f"PASSIVE MODE | {elapsed / 60:.0f}m",
            f"Activity: {latest_activity}",
            f"Segments: {self._segments_processed}",
            f"Buffer: {len(self._gaze_buffer)}",
        ]
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)