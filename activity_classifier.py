"""
Lucid Activity Classifier
─────────────────────────────
Segments a continuous gaze stream into cognitive activity types
based on oculomotor signatures. This is the core of passive
monitoring — it determines *what the user is doing* so that
downstream metrics are computed in the correct context.

Activity types:
  READING  — horizontal saccade chains with return sweeps
  SCANNING — wide-field visual search, multi-directional
  FOCUSED  — tight fixation cluster (writing, coding, forms)
  PASSIVE  — slow drift, video watching, idle scrolling
  IDLE     — no face detected or eyes closed

Classification uses a sliding window (default 5s, 2.5s stride)
and is entirely rule-based — no ML dependency, fully interpretable,
tuneable from config.py.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from config import (
    ACTIVITY_WINDOW_S, ACTIVITY_STRIDE_S, ACTIVITY_MIN_SAMPLES,
    READING_HORIZONTAL_RATIO, READING_RETURN_SWEEP_RATIO,
    READING_SACCADE_AMP_MAX, READING_FIX_DUR_RANGE,
    SCANNING_ENTROPY_THRESHOLD, SCANNING_SACCADE_AMP_MIN,
    FOCUSED_DISPERSION_MAX, FOCUSED_FIX_DUR_MIN,
    SACCADE_VELOCITY_THRESHOLD,
    FIXATION_DISPERSION_THRESHOLD, FIXATION_MIN_DURATION_MS,
)


@dataclass
class ActivitySegment:
    """A classified window of gaze data."""
    activity: str          # READING, SCANNING, FOCUSED, PASSIVE, IDLE
    start_ms: float
    end_ms: float
    confidence: float      # 0–1, how clearly this matches the pattern
    n_samples: int
    n_fixations: int
    n_saccades: int
    features: dict = field(default_factory=dict)  # raw features used for classification


class ActivityClassifier:
    """
    Classifies gaze stream segments into cognitive activity types.

    Usage:
        classifier = ActivityClassifier()
        segments = classifier.classify_stream(gaze_samples)
        # gaze_samples: list of GazeSample (from eye_tracker.py)
    """

    def classify_stream(self, samples: list) -> List[ActivitySegment]:
        """
        Classify an entire gaze stream into activity segments.
        samples: list of GazeSample with .t, .x, .y, .ear_l, .ear_r, .pupil_d_l, .pupil_d_r
        """
        if len(samples) < ACTIVITY_MIN_SAMPLES:
            return []

        ts = np.array([s.t for s in samples])
        total_duration_ms = ts[-1] - ts[0]
        window_ms = ACTIVITY_WINDOW_S * 1000
        stride_ms = ACTIVITY_STRIDE_S * 1000

        segments = []
        window_start = ts[0]

        while window_start + window_ms <= ts[-1] + stride_ms:
            window_end = window_start + window_ms

            # Get samples in this window
            mask = (ts >= window_start) & (ts < window_end)
            window_indices = np.where(mask)[0]

            if len(window_indices) < ACTIVITY_MIN_SAMPLES:
                window_start += stride_ms
                continue

            window_samples = [samples[i] for i in window_indices]
            segment = self._classify_window(window_samples)
            segments.append(segment)

            window_start += stride_ms

        # Merge consecutive segments of the same type
        return self._merge_segments(segments)

    def classify_window(self, samples: list) -> ActivitySegment:
        """Classify a single window of gaze data (public API for real-time use)."""
        return self._classify_window(samples)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core classification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _classify_window(self, samples: list) -> ActivitySegment:
        """Classify a single window using gaze features."""
        ts = np.array([s.t for s in samples])
        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])
        ears = np.array([(s.ear_l + s.ear_r) / 2 for s in samples])

        start_ms = float(ts[0])
        end_ms = float(ts[-1])
        n_samples = len(samples)

        # Check for IDLE first (eyes closed or no tracking)
        mean_ear = float(np.mean(ears))
        if mean_ear < 0.15 or n_samples < ACTIVITY_MIN_SAMPLES:
            return ActivitySegment(
                activity="IDLE", start_ms=start_ms, end_ms=end_ms,
                confidence=0.9, n_samples=n_samples,
                n_fixations=0, n_saccades=0,
                features={"mean_ear": mean_ear},
            )

        # Extract saccades and fixations for this window
        features = self._extract_classification_features(ts, xs, ys)

        # Decision tree
        activity, confidence = self._decision_tree(features)

        return ActivitySegment(
            activity=activity, start_ms=start_ms, end_ms=end_ms,
            confidence=confidence, n_samples=n_samples,
            n_fixations=features["n_fixations"],
            n_saccades=features["n_saccades"],
            features=features,
        )

    def _extract_classification_features(self, ts, xs, ys) -> dict:
        """Extract features used for activity classification."""
        n = len(ts)

        # Velocity
        dt = np.diff(ts) / 1000  # seconds
        dt[dt < 1e-6] = 1e-6
        dx = np.diff(xs)
        dy = np.diff(ys)
        vel = np.sqrt(dx ** 2 + dy ** 2) / dt

        # Saccade detection (simplified inline)
        is_saccade = vel > SACCADE_VELOCITY_THRESHOLD
        saccade_count = 0
        saccade_amplitudes = []
        saccade_directions = []  # angle in degrees
        in_sac = False
        sac_start = 0

        for i in range(len(vel)):
            if is_saccade[i] and not in_sac:
                in_sac = True
                sac_start = i
            elif not is_saccade[i] and in_sac:
                in_sac = False
                amp = float(np.sqrt((xs[i + 1] - xs[sac_start]) ** 2 +
                                    (ys[i + 1] - ys[sac_start]) ** 2))
                direction = float(np.degrees(np.arctan2(
                    ys[i + 1] - ys[sac_start],
                    xs[i + 1] - xs[sac_start])))
                if amp > 0.3:  # minimum to count
                    saccade_count += 1
                    saccade_amplitudes.append(amp)
                    saccade_directions.append(direction)

        # Fixation detection (simplified)
        fixation_count = 0
        fixation_durations = []
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                spread = (np.max(xs[i:j + 1]) - np.min(xs[i:j + 1])) + \
                         (np.max(ys[i:j + 1]) - np.min(ys[i:j + 1]))
                if spread > FIXATION_DISPERSION_THRESHOLD:
                    break
                j += 1
            dur = ts[min(j, n - 1)] - ts[i]
            if dur >= FIXATION_MIN_DURATION_MS:
                fixation_count += 1
                fixation_durations.append(float(dur))
            i = j

        # ── Feature computation ──

        # Saccade direction analysis
        dirs = np.array(saccade_directions) if saccade_directions else np.array([0])
        amps = np.array(saccade_amplitudes) if saccade_amplitudes else np.array([0])

        # Horizontal ratio: fraction of saccades within ±30° of horizontal
        horizontal_mask = np.abs(dirs) < 30
        horizontal_ratio = float(np.mean(horizontal_mask)) if len(dirs) > 0 else 0

        # Rightward saccades (reading direction for LTR text)
        rightward_mask = (dirs > -30) & (dirs < 30) & (amps > 0.5)
        rightward_ratio = float(np.mean(rightward_mask)) if len(dirs) > 0 else 0

        # Return sweeps: large leftward saccades (>5° amplitude, direction ~180°)
        return_sweep_mask = (np.abs(dirs) > 150) & (amps > 5.0)
        return_sweep_ratio = float(np.mean(return_sweep_mask)) if len(dirs) > 0 else 0

        # Spatial entropy (how spread out the gaze is)
        bins = 6
        if np.max(xs) - np.min(xs) > 0.1 and np.max(ys) - np.min(ys) > 0.1:
            x_bins = np.linspace(np.min(xs) - 0.1, np.max(xs) + 0.1, bins + 1)
            y_bins = np.linspace(np.min(ys) - 0.1, np.max(ys) + 0.1, bins + 1)
            hist, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
            hist_prob = hist.flatten() / max(np.sum(hist.flatten()), 1)
            from scipy.stats import entropy as sp_entropy
            spatial_entropy = float(sp_entropy(hist_prob + 1e-10, base=2))
        else:
            spatial_entropy = 0.0

        # Dispersion: total spatial spread of gaze
        gaze_dispersion = float(np.max(xs) - np.min(xs)) + float(np.max(ys) - np.min(ys))

        # Mean fixation duration
        mean_fix_dur = float(np.mean(fixation_durations)) if fixation_durations else 0

        # Mean saccade amplitude
        mean_sac_amp = float(np.mean(amps)) if len(amps) > 0 else 0

        return {
            "n_saccades": saccade_count,
            "n_fixations": fixation_count,
            "horizontal_ratio": horizontal_ratio,
            "rightward_ratio": rightward_ratio,
            "return_sweep_ratio": return_sweep_ratio,
            "mean_sac_amplitude": mean_sac_amp,
            "max_sac_amplitude": float(np.max(amps)) if len(amps) > 0 else 0,
            "spatial_entropy": spatial_entropy,
            "gaze_dispersion": gaze_dispersion,
            "mean_fix_duration": mean_fix_dur,
            "saccade_rate": saccade_count / max((ts[-1] - ts[0]) / 1000, 0.01),
            "mean_velocity": float(np.mean(vel)),
        }

    def _decision_tree(self, f: dict) -> tuple:
        """
        Rule-based decision tree for activity classification.
        Returns (activity_label, confidence).

        Priority order matters: READING is checked first because
        its signature is the most specific (horizontal chains + returns).
        """
        # ── READING ──
        # High horizontal saccade ratio + return sweeps + moderate amplitudes
        reading_score = 0.0
        if f["horizontal_ratio"] > READING_HORIZONTAL_RATIO:
            reading_score += 0.35
        if f["return_sweep_ratio"] > READING_RETURN_SWEEP_RATIO:
            reading_score += 0.3
        if f["rightward_ratio"] > 0.3:
            reading_score += 0.15
        if f["mean_sac_amplitude"] < READING_SACCADE_AMP_MAX:
            reading_score += 0.1
        if READING_FIX_DUR_RANGE[0] < f["mean_fix_duration"] < READING_FIX_DUR_RANGE[1]:
            reading_score += 0.1

        if reading_score >= 0.55:
            return ("READING", min(reading_score, 1.0))

        # ── SCANNING ──
        # High spatial entropy + multi-directional + larger saccades
        scanning_score = 0.0
        if f["spatial_entropy"] > SCANNING_ENTROPY_THRESHOLD:
            scanning_score += 0.4
        if f["mean_sac_amplitude"] > SCANNING_SACCADE_AMP_MIN:
            scanning_score += 0.25
        if f["horizontal_ratio"] < 0.5:  # not strongly horizontal
            scanning_score += 0.15
        if f["saccade_rate"] > 2.0:
            scanning_score += 0.2

        if scanning_score >= 0.5:
            return ("SCANNING", min(scanning_score, 1.0))

        # ── FOCUSED ──
        # Tight spatial cluster + long fixations + few saccades
        focused_score = 0.0
        if f["gaze_dispersion"] < FOCUSED_DISPERSION_MAX:
            focused_score += 0.4
        if f["mean_fix_duration"] > FOCUSED_FIX_DUR_MIN:
            focused_score += 0.3
        if f["saccade_rate"] < 2.0:
            focused_score += 0.2
        if f["spatial_entropy"] < 2.5:
            focused_score += 0.1

        if focused_score >= 0.5:
            return ("FOCUSED", min(focused_score, 1.0))

        # ── PASSIVE ──
        # Default: doesn't clearly match any active pattern
        return ("PASSIVE", 0.5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Segment merging
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _merge_segments(self, segments: List[ActivitySegment]) -> List[ActivitySegment]:
        """Merge consecutive segments of the same activity type."""
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.activity == prev.activity:
                # Extend previous segment
                merged[-1] = ActivitySegment(
                    activity=prev.activity,
                    start_ms=prev.start_ms,
                    end_ms=seg.end_ms,
                    confidence=(prev.confidence + seg.confidence) / 2,
                    n_samples=prev.n_samples + seg.n_samples,
                    n_fixations=prev.n_fixations + seg.n_fixations,
                    n_saccades=prev.n_saccades + seg.n_saccades,
                    features=seg.features,  # keep latest features
                )
            else:
                merged.append(seg)

        return merged
