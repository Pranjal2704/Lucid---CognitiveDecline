"""
Lucid Metric Extractor
──────────────────────────
Extracts cognitive-decline-relevant metrics from classified
activity segments. Each activity type yields different metrics
based on what the literature shows discriminates CI from healthy.

Paper-grounded metrics:
  READING:
    - fixation duration ↑ in AD (Fernández et al., 2016)
    - words-per-fixation proxy ↓ in AD
    - regression rate ↑ in AD (re-reading)
    - reading speed ↓ in AD/MCI
    Reference: Matsumoto et al. 2023 (Frontiers Aging Neurosci)

  SCANNING:
    - search efficiency ↓ in AD (more saccades to find target)
    - scanpath entropy ↑ in AD (disorganized search)
    - saccade velocity ↓ in CI (Chan et al., 2024)
    - gaze path velocity ↓ in CI
    Reference: Chan et al. 2024 (Cerebrovasc Dis)

  ALL ACTIVITIES:
    - pupil modulation under cognitive load ↓ in AD
      (locus coeruleus dysfunction, Matsumoto et al. 2023)
    - blink rate changes with cognitive load
    - saccade main sequence (velocity-amplitude relationship)
    - microsaccade rate (↑ in MCI, Kapoula et al. 2014)
    - square-wave jerks (↑ in AD, Nakamagoe et al. 2019)
    Reference: Systematic review PMC12750316 (2025)
"""

import numpy as np
from typing import List, Dict
from scipy.stats import entropy as sp_entropy, linregress

from config import (
    SACCADE_VELOCITY_THRESHOLD, SACCADE_MIN_DURATION_MS,
    SACCADE_MAX_DURATION_MS, SACCADE_MIN_AMPLITUDE_DEG,
    FIXATION_DISPERSION_THRESHOLD, FIXATION_MIN_DURATION_MS,
    BLINK_EAR_THRESHOLD, BLINK_MIN_DURATION_MS, BLINK_MAX_DURATION_MS,
    MICROSACCADE_VEL_THRESHOLD, MICROSACCADE_AMP_MAX,
    SWJ_AMPLITUDE_RANGE, SWJ_INTERSACCADIC_INTERVAL,
    PUPIL_BASELINE_PERCENTILE, PUPIL_SMOOTHING_WINDOW,
)
from activity_classifier import ActivitySegment


class MetricExtractor:
    """
    Extracts cognitive-decline-relevant metrics from gaze data
    within classified activity segments.

    Usage:
        extractor = MetricExtractor()
        metrics = extractor.extract(samples, segment)
        daily = extractor.compute_daily_summary(all_segments_metrics)
    """

    def extract(self, samples: list, segment: ActivitySegment) -> dict:
        """
        Extract all relevant metrics for a classified activity segment.
        samples: gaze samples falling within this segment's time range.
        """
        if len(samples) < 10:
            return {"activity": segment.activity, "error": "insufficient_data"}

        ts = np.array([s.t for s in samples])
        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])
        ears = np.array([(s.ear_l + s.ear_r) / 2 for s in samples])
        pupils = np.array([(s.pupil_d_l + s.pupil_d_r) / 2 for s in samples])
        duration_s = (ts[-1] - ts[0]) / 1000

        # Core events
        fixations = self._detect_fixations(ts, xs, ys)
        saccades = self._detect_saccades(ts, xs, ys)
        blinks = self._detect_blinks(ts, ears)

        # Base metrics (all activities)
        metrics = {
            "activity": segment.activity,
            "duration_s": round(duration_s, 2),
            "n_samples": len(samples),
            "confidence": round(segment.confidence, 3),
        }

        # Universal oculomotor metrics
        metrics.update(self._core_fixation_metrics(fixations, duration_s))
        metrics.update(self._core_saccade_metrics(saccades, duration_s))
        metrics.update(self._blink_metrics(blinks, duration_s))
        metrics.update(self._pupil_metrics(pupils, ts))
        metrics.update(self._microsaccade_metrics(ts, xs, ys, fixations))
        metrics.update(self._square_wave_jerk_metrics(saccades))

        # Activity-specific metrics
        if segment.activity == "READING":
            metrics.update(self._reading_metrics(ts, xs, ys, fixations, saccades))
        elif segment.activity == "SCANNING":
            metrics.update(self._scanning_metrics(ts, xs, ys, fixations, saccades))
        elif segment.activity == "FOCUSED":
            metrics.update(self._focused_metrics(ts, xs, ys, fixations, pupils))

        return metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Event detection (shared)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_fixations(self, ts, xs, ys) -> list:
        fixations = []
        n = len(ts)
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
                fixations.append({
                    "start_ms": float(ts[i]), "end_ms": float(ts[min(j, n - 1)]),
                    "duration_ms": float(dur),
                    "x": float(np.mean(xs[i:j])), "y": float(np.mean(ys[i:j])),
                    "start_idx": i, "end_idx": min(j, n - 1),
                })
            i = j
        return fixations

    def _detect_saccades(self, ts, xs, ys) -> list:
        dt = np.diff(ts) / 1000
        dt[dt < 1e-6] = 1e-6
        dx, dy = np.diff(xs), np.diff(ys)
        vel = np.sqrt(dx ** 2 + dy ** 2) / dt

        saccades = []
        in_sac, start_idx, peak_vel = False, 0, 0.0
        for i in range(len(vel)):
            if vel[i] > SACCADE_VELOCITY_THRESHOLD and not in_sac:
                in_sac, start_idx, peak_vel = True, i, vel[i]
            elif vel[i] > SACCADE_VELOCITY_THRESHOLD and in_sac:
                peak_vel = max(peak_vel, vel[i])
            elif vel[i] <= SACCADE_VELOCITY_THRESHOLD and in_sac:
                in_sac = False
                end_idx = i
                dur = float(ts[end_idx + 1] - ts[start_idx])
                amp = float(np.sqrt((xs[end_idx + 1] - xs[start_idx]) ** 2 +
                                    (ys[end_idx + 1] - ys[start_idx]) ** 2))
                direction = float(np.degrees(np.arctan2(
                    ys[end_idx + 1] - ys[start_idx], xs[end_idx + 1] - xs[start_idx])))
                if SACCADE_MIN_DURATION_MS <= dur <= SACCADE_MAX_DURATION_MS and amp >= SACCADE_MIN_AMPLITUDE_DEG:
                    saccades.append({
                        "start_ms": float(ts[start_idx]), "duration_ms": dur,
                        "amplitude_deg": amp, "peak_velocity": float(peak_vel),
                        "mean_velocity": float(np.mean(vel[start_idx:end_idx + 1])),
                        "direction_deg": direction,
                        "dx": float(xs[end_idx + 1] - xs[start_idx]),
                        "dy": float(ys[end_idx + 1] - ys[start_idx]),
                    })
        return saccades

    def _detect_blinks(self, ts, ears) -> list:
        blinks = []
        in_blink, start = False, 0
        for i in range(len(ears)):
            if ears[i] < BLINK_EAR_THRESHOLD and not in_blink:
                in_blink, start = True, i
            elif ears[i] >= BLINK_EAR_THRESHOLD and in_blink:
                in_blink = False
                dur = float(ts[i] - ts[start])
                if BLINK_MIN_DURATION_MS <= dur <= BLINK_MAX_DURATION_MS:
                    blinks.append({"start_ms": float(ts[start]), "duration_ms": dur})
        return blinks

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core metrics (all activities)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _core_fixation_metrics(self, fixations, duration_s) -> dict:
        if not fixations:
            return {"fix_count": 0, "fix_rate": 0, "fix_dur_mean": 0,
                    "fix_dur_std": 0, "fix_dur_median": 0}
        durs = [f["duration_ms"] for f in fixations]
        return {
            "fix_count": len(fixations),
            "fix_rate": round(len(fixations) / max(duration_s, 0.01), 2),
            "fix_dur_mean": round(float(np.mean(durs)), 1),
            "fix_dur_std": round(float(np.std(durs)), 1),
            "fix_dur_median": round(float(np.median(durs)), 1),
        }

    def _core_saccade_metrics(self, saccades, duration_s) -> dict:
        if not saccades:
            return {"sac_count": 0, "sac_rate": 0, "sac_amp_mean": 0,
                    "sac_vel_mean": 0, "sac_vel_peak_mean": 0,
                    "main_seq_slope": 0, "main_seq_r2": 0}
        amps = [s["amplitude_deg"] for s in saccades]
        pvels = [s["peak_velocity"] for s in saccades]
        mvels = [s["mean_velocity"] for s in saccades]

        ms_slope, ms_r2 = 0.0, 0.0
        if len(amps) > 3:
            log_a = np.log(np.array(amps) + 0.01)
            log_v = np.log(np.array(pvels) + 0.01)
            slope, _, r, _, _ = linregress(log_a, log_v)
            ms_slope, ms_r2 = float(slope), float(r ** 2)

        return {
            "sac_count": len(saccades),
            "sac_rate": round(len(saccades) / max(duration_s, 0.01), 2),
            "sac_amp_mean": round(float(np.mean(amps)), 2),
            "sac_amp_std": round(float(np.std(amps)), 2),
            "sac_vel_mean": round(float(np.mean(mvels)), 1),
            "sac_vel_peak_mean": round(float(np.mean(pvels)), 1),
            "main_seq_slope": round(ms_slope, 3),
            "main_seq_r2": round(ms_r2, 3),
        }

    def _blink_metrics(self, blinks, duration_s) -> dict:
        if not blinks:
            return {"blink_count": 0, "blink_rate_min": 0, "blink_dur_mean": 0}
        durs = [b["duration_ms"] for b in blinks]
        return {
            "blink_count": len(blinks),
            "blink_rate_min": round(len(blinks) / max(duration_s / 60, 0.01), 1),
            "blink_dur_mean": round(float(np.mean(durs)), 1),
        }

    def _pupil_metrics(self, pupils, ts) -> dict:
        """
        Pupil-based cognitive load analysis.
        Matsumoto et al. 2023: AD patients don't show progressive pupil
        dilation across consecutive cognitive effort blocks. We track:
        - absolute diameter (baseline)
        - variability (modulation capacity)
        - trend over session (cognitive load response)
        """
        valid = pupils[pupils > 0]
        if len(valid) < 10:
            return {"pupil_mean": 0, "pupil_std": 0, "pupil_cv": 0,
                    "pupil_load_range": 0, "pupil_trend": 0}

        # Smooth for trend analysis
        kernel = min(PUPIL_SMOOTHING_WINDOW, len(valid) // 3)
        if kernel > 2:
            smoothed = np.convolve(valid, np.ones(kernel) / kernel, mode="valid")
        else:
            smoothed = valid

        # Baseline: low-activity pupil size (20th percentile)
        baseline = float(np.percentile(valid, PUPIL_BASELINE_PERCENTILE))
        peak = float(np.percentile(valid, 80))
        load_range = peak - baseline  # cognitive load modulation capacity

        # Trend: does pupil dilate over time? (positive = increasing effort)
        if len(smoothed) > 5:
            x = np.arange(len(smoothed))
            slope, _, _, _, _ = linregress(x, smoothed)
            pupil_trend = float(slope)
        else:
            pupil_trend = 0.0

        return {
            "pupil_mean": round(float(np.mean(valid)), 2),
            "pupil_std": round(float(np.std(valid)), 2),
            "pupil_cv": round(float(np.std(valid) / np.mean(valid)), 4) if np.mean(valid) > 0 else 0,
            "pupil_load_range": round(load_range, 2),
            "pupil_trend": round(pupil_trend, 6),
        }

    def _microsaccade_metrics(self, ts, xs, ys, fixations) -> dict:
        """
        Detect microsaccades within fixation periods.
        Kapoula et al. 2014: MCI patients exhibit more oblique microsaccades.
        ↑ microsaccade rate during fixation = ↑ attentional instability.
        """
        if not fixations:
            return {"microsac_count": 0, "microsac_rate": 0, "microsac_oblique_ratio": 0}

        microsaccades = []
        total_fix_time = 0

        for fix in fixations:
            si, ei = fix["start_idx"], fix["end_idx"]
            if ei - si < 5:
                continue
            total_fix_time += fix["duration_ms"]

            fix_ts = ts[si:ei + 1]
            fix_xs = xs[si:ei + 1]
            fix_ys = ys[si:ei + 1]

            dt = np.diff(fix_ts) / 1000
            dt[dt < 1e-6] = 1e-6
            dx, dy = np.diff(fix_xs), np.diff(fix_ys)
            vel = np.sqrt(dx ** 2 + dy ** 2) / dt

            in_ms = False
            for i in range(len(vel)):
                if vel[i] > MICROSACCADE_VEL_THRESHOLD and vel[i] < SACCADE_VELOCITY_THRESHOLD and not in_ms:
                    in_ms = True
                    ms_start = i
                elif (vel[i] <= MICROSACCADE_VEL_THRESHOLD or vel[i] >= SACCADE_VELOCITY_THRESHOLD) and in_ms:
                    in_ms = False
                    amp = float(np.sqrt((fix_xs[i] - fix_xs[ms_start]) ** 2 +
                                        (fix_ys[i] - fix_ys[ms_start]) ** 2))
                    if amp < MICROSACCADE_AMP_MAX:
                        direction = np.degrees(np.arctan2(
                            fix_ys[i] - fix_ys[ms_start],
                            fix_xs[i] - fix_xs[ms_start]))
                        microsaccades.append({"amplitude": amp, "direction": float(direction)})

        total_fix_s = total_fix_time / 1000
        oblique_count = sum(1 for m in microsaccades
                           if 20 < abs(m["direction"]) % 90 < 70)

        return {
            "microsac_count": len(microsaccades),
            "microsac_rate": round(len(microsaccades) / max(total_fix_s, 0.01), 2),
            "microsac_oblique_ratio": round(
                oblique_count / max(len(microsaccades), 1), 3),
        }

    def _square_wave_jerk_metrics(self, saccades) -> dict:
        """
        Square-wave jerks: pairs of small saccades in opposite directions
        with 200-400ms interval. ↑ in AD (Nakamagoe et al. 2019).
        """
        swj_count = 0
        for i in range(len(saccades) - 1):
            s1, s2 = saccades[i], saccades[i + 1]
            amp1, amp2 = s1["amplitude_deg"], s2["amplitude_deg"]
            interval = s2["start_ms"] - (s1["start_ms"] + s1["duration_ms"])

            # Both in amplitude range
            if not (SWJ_AMPLITUDE_RANGE[0] <= amp1 <= SWJ_AMPLITUDE_RANGE[1]):
                continue
            if not (SWJ_AMPLITUDE_RANGE[0] <= amp2 <= SWJ_AMPLITUDE_RANGE[1]):
                continue
            # Correct interval
            if not (SWJ_INTERSACCADIC_INTERVAL[0] <= interval <= SWJ_INTERSACCADIC_INTERVAL[1]):
                continue
            # Opposite directions (within 30° of reversal)
            dir_diff = abs(s1["direction_deg"] - s2["direction_deg"])
            if dir_diff > 180:
                dir_diff = 360 - dir_diff
            if dir_diff > 150:
                swj_count += 1

        return {"swj_count": swj_count}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # READING-specific metrics
    # Literature: Fernández 2016, Matsumoto 2023, Hannonen 2022/2026
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _reading_metrics(self, ts, xs, ys, fixations, saccades) -> dict:
        """
        Reading-specific metrics. AD/MCI signatures:
        - ↑ fixation duration (slower processing per word)
        - ↓ forward saccade amplitude (fewer words per jump)
        - ↑ regression rate (more re-reading)
        - ↓ reading speed
        - ↓ skip rate (less predictive skipping)
        """
        if not saccades or not fixations:
            return {}

        # Separate forward (rightward) and regressive (leftward) saccades
        forward_sacs = [s for s in saccades if -45 < s["direction_deg"] < 45]
        regressions = [s for s in saccades if abs(s["direction_deg"]) > 135]
        return_sweeps = [s for s in regressions if s["amplitude_deg"] > 5]
        small_regressions = [s for s in regressions if s["amplitude_deg"] <= 5]

        total_directed = len(forward_sacs) + len(small_regressions)
        regression_rate = len(small_regressions) / max(total_directed, 1)

        # Forward saccade amplitude = proxy for words-per-fixation
        fwd_amps = [s["amplitude_deg"] for s in forward_sacs]
        mean_fwd_amp = float(np.mean(fwd_amps)) if fwd_amps else 0

        # Reading speed proxy: fixations per second
        duration_s = (ts[-1] - ts[0]) / 1000
        fix_rate = len(fixations) / max(duration_s, 0.01)

        # Fixation duration variability (↑ = inconsistent processing)
        fix_durs = [f["duration_ms"] for f in fixations]
        fix_dur_cv = float(np.std(fix_durs) / np.mean(fix_durs)) if np.mean(fix_durs) > 0 else 0

        # Return sweep accuracy: do return sweeps land near the line start?
        # (less accurate returns = worse spatial memory of text layout)
        if len(return_sweeps) > 1:
            landing_xs = []
            for sw in return_sweeps:
                # Find fixation after this return sweep
                for f in fixations:
                    if f["start_ms"] > sw["start_ms"] + sw["duration_ms"]:
                        landing_xs.append(f["x"])
                        break
            return_accuracy = float(np.std(landing_xs)) if len(landing_xs) > 1 else 0
        else:
            return_accuracy = 0

        return {
            "read_forward_sac_count": len(forward_sacs),
            "read_forward_amp_mean": round(mean_fwd_amp, 2),
            "read_regression_rate": round(regression_rate, 3),
            "read_regression_count": len(small_regressions),
            "read_return_sweep_count": len(return_sweeps),
            "read_return_accuracy": round(return_accuracy, 2),
            "read_fix_rate": round(fix_rate, 2),
            "read_fix_dur_cv": round(fix_dur_cv, 3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCANNING-specific metrics
    # Literature: Chan et al. 2024, Matsumoto 2023
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _scanning_metrics(self, ts, xs, ys, fixations, saccades) -> dict:
        """
        Visual search metrics. CI signatures:
        - ↓ gaze path velocity (Chan: cutoff 329.665 px/s)
        - ↓ saccade velocity (Chan: cutoff 2.150 px/ms)
        - ↑ scanpath length (inefficient search)
        - ↓ scanpath efficiency (path/straight ratio)
        - ↑ spatial entropy (disorganized coverage)
        """
        duration_s = (ts[-1] - ts[0]) / 1000

        # Gaze path velocity: total path length / time
        dx, dy = np.diff(xs), np.diff(ys)
        step_lengths = np.sqrt(dx ** 2 + dy ** 2)
        total_path = float(np.sum(step_lengths))
        gaze_path_velocity = total_path / max(duration_s, 0.01)

        # Straight-line distance
        straight = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
        scan_efficiency = straight / max(total_path, 0.01)

        # Spatial coverage entropy
        bins = 8
        if np.ptp(xs) > 0.1 and np.ptp(ys) > 0.1:
            x_bins = np.linspace(np.min(xs) - 0.1, np.max(xs) + 0.1, bins + 1)
            y_bins = np.linspace(np.min(ys) - 0.1, np.max(ys) + 0.1, bins + 1)
            hist, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
            probs = hist.flatten() / max(np.sum(hist.flatten()), 1)
            spatial_entropy = float(sp_entropy(probs + 1e-10, base=2))
        else:
            spatial_entropy = 0

        # Convex hull area (spatial extent of search)
        hull_area = 0.0
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack([xs, ys])
            if len(np.unique(points, axis=0)) >= 3:
                hull_area = float(ConvexHull(points).volume)
        except Exception:
            pass

        # Search revisit rate: how often gaze returns to previously visited regions
        revisit_count = 0
        if fixations and len(fixations) > 3:
            for i in range(1, len(fixations)):
                for j in range(i):
                    dist = np.sqrt((fixations[i]["x"] - fixations[j]["x"]) ** 2 +
                                   (fixations[i]["y"] - fixations[j]["y"]) ** 2)
                    if dist < 1.5:  # within 1.5° = revisit
                        revisit_count += 1
                        break

        return {
            "scan_gaze_path_vel": round(gaze_path_velocity, 1),
            "scan_path_length": round(total_path, 2),
            "scan_efficiency": round(scan_efficiency, 4),
            "scan_spatial_entropy": round(spatial_entropy, 3),
            "scan_hull_area": round(hull_area, 2),
            "scan_revisit_count": revisit_count,
            "scan_revisit_rate": round(revisit_count / max(len(fixations), 1), 3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FOCUSED-specific metrics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _focused_metrics(self, ts, xs, ys, fixations, pupils) -> dict:
        """
        Focused work metrics (coding, writing, form-filling).
        Cognitive load is the key signal here — pupil dilation
        under sustained attention reflects executive function.
        """
        # Sustained fixation: longest continuous fixation
        max_fix_dur = max((f["duration_ms"] for f in fixations), default=0)

        # Gaze stability: mean dispersion within fixations
        dispersions = []
        for f in fixations:
            si, ei = f["start_idx"], f["end_idx"]
            if ei > si:
                d = (np.max(xs[si:ei + 1]) - np.min(xs[si:ei + 1])) + \
                    (np.max(ys[si:ei + 1]) - np.min(ys[si:ei + 1]))
                dispersions.append(d)

        valid_pupils = pupils[pupils > 0]
        pupil_sustained = 0.0
        if len(valid_pupils) > 20:
            # Is pupil diameter stable or fluctuating?
            pupil_sustained = float(np.std(np.diff(valid_pupils)))

        return {
            "focus_max_fix_dur": round(max_fix_dur, 1),
            "focus_mean_dispersion": round(float(np.mean(dispersions)), 3) if dispersions else 0,
            "focus_pupil_stability": round(pupil_sustained, 4),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Daily summary aggregation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_daily_summary(self, segment_metrics: List[dict]) -> dict:
        """
        Aggregate per-segment metrics into a daily summary.
        Groups by activity type and computes means, plus cross-activity metrics.
        """
        if not segment_metrics:
            return {"error": "no_data"}

        summary = {
            "total_segments": len(segment_metrics),
            "total_duration_s": sum(m.get("duration_s", 0) for m in segment_metrics),
        }

        # Activity time distribution
        activity_times = {}
        for m in segment_metrics:
            act = m.get("activity", "UNKNOWN")
            activity_times[act] = activity_times.get(act, 0) + m.get("duration_s", 0)
        total_time = sum(activity_times.values()) or 1
        for act, t in activity_times.items():
            summary[f"time_{act.lower()}_s"] = round(t, 1)
            summary[f"pct_{act.lower()}"] = round(t / total_time * 100, 1)

        # Per-activity aggregation of key metrics
        numeric_keys = set()
        for m in segment_metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)) and k not in ("duration_s", "n_samples", "confidence"):
                    numeric_keys.add(k)

        for activity in ["READING", "SCANNING", "FOCUSED", "PASSIVE"]:
            act_metrics = [m for m in segment_metrics if m.get("activity") == activity]
            if not act_metrics:
                continue
            prefix = activity.lower()[:4]
            for key in numeric_keys:
                vals = [m[key] for m in act_metrics if key in m and isinstance(m[key], (int, float))]
                if vals:
                    summary[f"{prefix}_{key}_mean"] = round(float(np.mean(vals)), 3)

        # Cross-activity: cognitive switching cost
        # How do metrics change between activity transitions?
        if len(segment_metrics) > 1:
            fix_dur_by_activity = {}
            for m in segment_metrics:
                act = m.get("activity", "UNKNOWN")
                fd = m.get("fix_dur_mean", 0)
                if fd > 0:
                    fix_dur_by_activity.setdefault(act, []).append(fd)

            # Variability across activity types = cognitive flexibility proxy
            all_means = [np.mean(v) for v in fix_dur_by_activity.values() if len(v) > 0]
            if len(all_means) > 1:
                summary["cross_activity_fix_dur_range"] = round(
                    float(np.max(all_means) - np.min(all_means)), 1)

        return summary
