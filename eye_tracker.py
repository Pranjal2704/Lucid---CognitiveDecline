"""
Lucid Eye Tracker
────────────────────
Real-time eye tracking using the MediaPipe Tasks FaceLandmarker API
(not the legacy mp.solutions.face_mesh). Uses a downloadable .task
model bundle with 478 landmarks including iris.

API pattern:
  - FaceLandmarker.create_from_options(FaceLandmarkerOptions(...))
  - landmarker.detect_for_video(mp.Image, timestamp_ms)  # VIDEO mode
  - result.face_landmarks[0]  → list[NormalizedLandmark]

Metric groups:
  1. Fixation — duration, count, dispersion, rate
  2. Saccade — velocity, amplitude, duration, main sequence, count
  3. Blink — rate, duration, regularity
  4. Scanpath — length, efficiency, entropy, convex hull
  5. Pupil — diameter estimate, variability
  6. Dynamics — fixation/saccade ratio, intersaccadic interval
"""

import os
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from scipy.spatial import ConvexHull
from scipy.stats import entropy as sp_entropy

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    PIXELS_PER_DEGREE, MODEL_DIR, FACE_LANDMARKER_MODEL, FACE_LANDMARKER_URL,
    SACCADE_VELOCITY_THRESHOLD, SACCADE_MIN_DURATION_MS,
    SACCADE_MAX_DURATION_MS, SACCADE_MIN_AMPLITUDE_DEG,
    FIXATION_DISPERSION_THRESHOLD, FIXATION_MIN_DURATION_MS,
    BLINK_EAR_THRESHOLD, BLINK_MIN_DURATION_MS, BLINK_MAX_DURATION_MS,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
    LEFT_EYE_EAR, RIGHT_EYE_EAR, LEFT_IRIS, RIGHT_IRIS,
    GAZE_HISTORY_MAX,
)

# ── Tasks API imports ─────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def ensure_model():
    """Download the FaceLandmarker .task bundle if it doesn't exist."""
    if os.path.isfile(FACE_LANDMARKER_MODEL):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"  Downloading FaceLandmarker model → {FACE_LANDMARKER_MODEL}")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    print(f"  Download complete ({os.path.getsize(FACE_LANDMARKER_MODEL) / 1e6:.1f} MB).")


def create_landmarker(mode: VisionRunningMode = VisionRunningMode.VIDEO):
    """
    Build a FaceLandmarker instance with the specified running mode.
    Public helper so prosaccade_task.py can create its own landmarker
    without duplicating option setup.
    """
    ensure_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


class GazeSample:
    """Single gaze measurement."""
    __slots__ = ("t", "x", "y", "ear_l", "ear_r", "pupil_d_l", "pupil_d_r")

    def __init__(self, t, x, y, ear_l=0, ear_r=0, pupil_d_l=0, pupil_d_r=0):
        self.t = t          # timestamp (ms)
        self.x = x          # horizontal gaze (degrees from center)
        self.y = y          # vertical gaze (degrees from center)
        self.ear_l = ear_l  # left eye aspect ratio
        self.ear_r = ear_r  # right eye aspect ratio
        self.pupil_d_l = pupil_d_l  # left pupil diameter estimate (px)
        self.pupil_d_r = pupil_d_r  # right pupil diameter estimate (px)


class EyeTracker:
    """
    Real-time eye tracking with MediaPipe Tasks FaceLandmarker.

    Usage:
        tracker = EyeTracker()
        tracker.start()                   # blocking — runs until 'q' pressed
        metrics = tracker.get_metrics()    # aggregate session metrics
    """

    def __init__(self):
        ensure_model()
        self.gaze_buffer = deque(maxlen=GAZE_HISTORY_MAX)
        self.session_start = 0
        self.frame_count = 0
        self._running = False

    def start(self, display: bool = True, duration_s: float = None):
        """
        Start webcam capture and eye tracking.
        Press 'q' to stop, or runs for `duration_s` seconds if specified.
        """
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.gaze_buffer.clear()
        self.session_start = time.time() * 1000  # ms
        self.frame_count = 0
        self._running = True

        print("  Eye tracking started. Press 'q' to stop.")

        landmarker = create_landmarker(VisionRunningMode.VIDEO)

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # mirror
                self.frame_count += 1
                now = time.time() * 1000  # ms
                timestamp_ms = int(now - self.session_start)

                if duration_s and (now - self.session_start) / 1000 > duration_s:
                    break

                # Convert BGR → RGB numpy → mp.Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # Run detection (VIDEO mode — synchronous, requires monotonic timestamps)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]  # first face
                    sample = self._extract_gaze(landmarks, frame.shape, now)
                    if sample:
                        self.gaze_buffer.append(sample)

                    if display:
                        self._draw_overlay(frame, landmarks, sample)

                if display:
                    self._draw_live_metrics(frame)
                    cv2.imshow("Lucid — Eye Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self._running = False
            cap.release()
            landmarker.close()
            if display:
                cv2.destroyAllWindows()

        print(f"  Session ended. {len(self.gaze_buffer)} samples collected over "
              f"{(time.time() * 1000 - self.session_start) / 1000:.1f}s.")

    def process_frame_with_landmarker(self, frame: np.ndarray,
                                      landmarker: FaceLandmarker,
                                      timestamp_ms: int) -> dict:
        """
        Process a single frame using a caller-managed landmarker.
        Used by prosaccade_task.py during stimulus presentation.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            sample = self._extract_gaze(landmarks, frame.shape, time.time() * 1000)
            if sample:
                self.gaze_buffer.append(sample)
                return {"x": sample.x, "y": sample.y, "t": sample.t,
                        "ear": (sample.ear_l + sample.ear_r) / 2,
                        "pupil_d": (sample.pupil_d_l + sample.pupil_d_r) / 2}
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Gaze extraction from NormalizedLandmark list
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_gaze(self, landmarks, frame_shape, t_ms) -> GazeSample:
        """
        Extract gaze direction from FaceLandmarker NormalizedLandmark list.
        landmarks: list of NormalizedLandmark (478 entries) with .x, .y, .z
        """
        h, w = frame_shape[:2]

        def lm(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        # Iris centers (landmarks 468, 473)
        iris_l = lm(LEFT_IRIS_CENTER)
        iris_r = lm(RIGHT_IRIS_CENTER)

        # Eye corners
        l_inner, l_outer = lm(LEFT_EYE_INNER), lm(LEFT_EYE_OUTER)
        r_inner, r_outer = lm(RIGHT_EYE_INNER), lm(RIGHT_EYE_OUTER)

        # Gaze: iris position relative to eye center, normalized by eye width
        l_eye_center = (l_inner + l_outer) / 2
        r_eye_center = (r_inner + r_outer) / 2
        l_eye_width = np.linalg.norm(l_inner - l_outer)
        r_eye_width = np.linalg.norm(r_inner - r_outer)

        if l_eye_width < 1 or r_eye_width < 1:
            return None

        # Normalized iris displacement [-1, 1] where 0 = center of eye
        l_gaze_norm = (iris_l - l_eye_center) / (l_eye_width / 2)
        r_gaze_norm = (iris_r - r_eye_center) / (r_eye_width / 2)

        # Average binocular gaze, convert to approximate degrees
        gaze_norm = (l_gaze_norm + r_gaze_norm) / 2
        gaze_deg_x = float(gaze_norm[0]) * 25  # ±25° horizontal range
        gaze_deg_y = float(gaze_norm[1]) * 20  # ±20° vertical range

        # Eye Aspect Ratio for blink detection
        ear_l = self._compute_ear(landmarks, LEFT_EYE_EAR, w, h)
        ear_r = self._compute_ear(landmarks, RIGHT_EYE_EAR, w, h)

        # Pupil diameter estimate from iris landmark spread
        pupil_d_l = self._estimate_pupil_diameter(landmarks, LEFT_IRIS, w, h)
        pupil_d_r = self._estimate_pupil_diameter(landmarks, RIGHT_IRIS, w, h)

        return GazeSample(t_ms, gaze_deg_x, gaze_deg_y, ear_l, ear_r, pupil_d_l, pupil_d_r)

    def _compute_ear(self, landmarks, eye_dict, w, h) -> float:
        """Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
        def lm(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        p1 = lm(eye_dict["p1"])
        p2 = lm(eye_dict["p2"])
        p3 = lm(eye_dict["p3"])
        p4 = lm(eye_dict["p4"])
        p5 = lm(eye_dict["p5"])
        p6 = lm(eye_dict["p6"])

        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h1 = np.linalg.norm(p1 - p4)

        if h1 < 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h1))

    def _estimate_pupil_diameter(self, landmarks, iris_indices, w, h) -> float:
        """Estimate pupil/iris diameter from iris landmark spread."""
        points = []
        for idx in iris_indices:
            points.append([landmarks[idx].x * w, landmarks[idx].y * h])
        points = np.array(points)
        if len(points) < 3:
            return 0.0
        from scipy.spatial.distance import pdist
        dists = pdist(points)
        return float(np.max(dists)) if len(dists) > 0 else 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Aggregate metric computation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_metrics(self) -> dict:
        """Compute all aggregate metrics from the gaze buffer."""
        if len(self.gaze_buffer) < 10:
            return {"error": "insufficient_data", "samples": len(self.gaze_buffer)}

        samples = list(self.gaze_buffer)
        duration_s = (samples[-1].t - samples[0].t) / 1000

        metrics = {
            "session_duration_s": round(duration_s, 2),
            "total_samples": len(samples),
            "effective_fps": round(len(samples) / max(duration_s, 0.01), 1),
        }

        ts = np.array([s.t for s in samples])
        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])
        ears_l = np.array([s.ear_l for s in samples])
        ears_r = np.array([s.ear_r for s in samples])
        pupils_l = np.array([s.pupil_d_l for s in samples])
        pupils_r = np.array([s.pupil_d_r for s in samples])

        dt = np.diff(ts) / 1000  # seconds
        dx = np.diff(xs)
        dy = np.diff(ys)
        dt[dt < 1e-6] = 1e-6
        vel = np.sqrt(dx ** 2 + dy ** 2) / dt  # deg/s

        fixations = self._detect_fixations(ts, xs, ys)
        metrics.update(self._fixation_metrics(fixations, duration_s))

        saccades = self._detect_saccades(ts, xs, ys, vel, dt)
        metrics.update(self._saccade_metrics(saccades, duration_s))

        ears = (ears_l + ears_r) / 2
        blinks = self._detect_blinks(ts, ears)
        metrics.update(self._blink_metrics(blinks, duration_s))

        metrics.update(self._scanpath_metrics(xs, ys, vel))

        pupils = (pupils_l + pupils_r) / 2
        metrics.update(self._pupil_metrics(pupils))

        metrics.update(self._dynamics_metrics(fixations, saccades, duration_s))

        return metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Fixation Detection (I-DT: Dispersion-Threshold)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_fixations(self, ts, xs, ys) -> list:
        fixations = []
        n = len(ts)
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                window_x = xs[i:j + 1]
                window_y = ys[i:j + 1]
                dispersion = (np.max(window_x) - np.min(window_x)) + \
                             (np.max(window_y) - np.min(window_y))
                if dispersion > FIXATION_DISPERSION_THRESHOLD:
                    break
                j += 1
            duration_ms = ts[min(j, n - 1)] - ts[i]
            if duration_ms >= FIXATION_MIN_DURATION_MS:
                cx = float(np.mean(xs[i:j]))
                cy = float(np.mean(ys[i:j]))
                disp = float((np.max(xs[i:j]) - np.min(xs[i:j])) +
                             (np.max(ys[i:j]) - np.min(ys[i:j])))
                fixations.append({
                    "start_ms": float(ts[i]),
                    "duration_ms": float(duration_ms),
                    "x": cx, "y": cy,
                    "dispersion": disp,
                    "n_samples": j - i,
                })
            i = j
        return fixations

    def _fixation_metrics(self, fixations, duration_s) -> dict:
        if not fixations:
            return {k: 0 for k in [
                "fix_count", "fix_rate_per_s", "fix_dur_mean_ms", "fix_dur_median_ms",
                "fix_dur_std_ms", "fix_dur_min_ms", "fix_dur_max_ms",
                "fix_dispersion_mean_deg", "fix_dispersion_std_deg",
                "fix_total_time_ms", "fix_time_ratio",
            ]}
        durs = [f["duration_ms"] for f in fixations]
        disps = [f["dispersion"] for f in fixations]
        total_fix_time = sum(durs)
        return {
            "fix_count": len(fixations),
            "fix_rate_per_s": round(len(fixations) / max(duration_s, 0.01), 2),
            "fix_dur_mean_ms": round(float(np.mean(durs)), 1),
            "fix_dur_median_ms": round(float(np.median(durs)), 1),
            "fix_dur_std_ms": round(float(np.std(durs)), 1),
            "fix_dur_min_ms": round(float(np.min(durs)), 1),
            "fix_dur_max_ms": round(float(np.max(durs)), 1),
            "fix_dispersion_mean_deg": round(float(np.mean(disps)), 3),
            "fix_dispersion_std_deg": round(float(np.std(disps)), 3),
            "fix_total_time_ms": round(total_fix_time, 1),
            "fix_time_ratio": round(total_fix_time / max(duration_s * 1000, 1), 3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Saccade Detection (Velocity-Threshold)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_saccades(self, ts, xs, ys, vel, dt) -> list:
        saccades = []
        n = len(vel)
        in_saccade = False
        start_idx = 0
        peak_vel = 0.0

        for i in range(n):
            if vel[i] > SACCADE_VELOCITY_THRESHOLD and not in_saccade:
                in_saccade = True
                start_idx = i
                peak_vel = vel[i]
            elif vel[i] > SACCADE_VELOCITY_THRESHOLD and in_saccade:
                peak_vel = max(peak_vel, vel[i])
            elif vel[i] <= SACCADE_VELOCITY_THRESHOLD and in_saccade:
                in_saccade = False
                end_idx = i
                duration_ms = float(ts[end_idx + 1] - ts[start_idx])
                amplitude = float(np.sqrt((xs[end_idx + 1] - xs[start_idx]) ** 2 +
                                          (ys[end_idx + 1] - ys[start_idx]) ** 2))
                if (SACCADE_MIN_DURATION_MS <= duration_ms <= SACCADE_MAX_DURATION_MS and
                        amplitude >= SACCADE_MIN_AMPLITUDE_DEG):
                    path_len = float(np.sum(np.sqrt(
                        np.diff(xs[start_idx:end_idx + 2]) ** 2 +
                        np.diff(ys[start_idx:end_idx + 2]) ** 2)))
                    saccades.append({
                        "start_ms": float(ts[start_idx]),
                        "duration_ms": duration_ms,
                        "amplitude_deg": amplitude,
                        "peak_velocity_deg_s": float(peak_vel),
                        "mean_velocity_deg_s": float(np.mean(vel[start_idx:end_idx + 1])),
                        "path_length_deg": path_len,
                        "efficiency": amplitude / max(path_len, 0.01),
                        "direction_deg": float(np.degrees(np.arctan2(
                            ys[end_idx + 1] - ys[start_idx],
                            xs[end_idx + 1] - xs[start_idx]))),
                    })
        return saccades

    def _saccade_metrics(self, saccades, duration_s) -> dict:
        if not saccades:
            return {k: 0 for k in [
                "sac_count", "sac_rate_per_s",
                "sac_amp_mean_deg", "sac_amp_std_deg", "sac_amp_median_deg",
                "sac_dur_mean_ms", "sac_dur_std_ms",
                "sac_peak_vel_mean", "sac_peak_vel_std", "sac_peak_vel_max",
                "sac_mean_vel_mean",
                "sac_efficiency_mean", "sac_efficiency_std",
                "sac_main_sequence_slope", "sac_main_sequence_r2",
            ]}
        amps = [s["amplitude_deg"] for s in saccades]
        durs = [s["duration_ms"] for s in saccades]
        pvels = [s["peak_velocity_deg_s"] for s in saccades]
        mvels = [s["mean_velocity_deg_s"] for s in saccades]
        effs = [s["efficiency"] for s in saccades]

        ms_slope, ms_r2 = 0.0, 0.0
        if len(amps) > 3:
            from scipy.stats import linregress
            log_a = np.log(np.array(amps) + 0.01)
            log_v = np.log(np.array(pvels) + 0.01)
            slope, _, r, _, _ = linregress(log_a, log_v)
            ms_slope = float(slope)
            ms_r2 = float(r ** 2)

        return {
            "sac_count": len(saccades),
            "sac_rate_per_s": round(len(saccades) / max(duration_s, 0.01), 2),
            "sac_amp_mean_deg": round(float(np.mean(amps)), 2),
            "sac_amp_std_deg": round(float(np.std(amps)), 2),
            "sac_amp_median_deg": round(float(np.median(amps)), 2),
            "sac_dur_mean_ms": round(float(np.mean(durs)), 1),
            "sac_dur_std_ms": round(float(np.std(durs)), 1),
            "sac_peak_vel_mean": round(float(np.mean(pvels)), 1),
            "sac_peak_vel_std": round(float(np.std(pvels)), 1),
            "sac_peak_vel_max": round(float(np.max(pvels)), 1),
            "sac_mean_vel_mean": round(float(np.mean(mvels)), 1),
            "sac_efficiency_mean": round(float(np.mean(effs)), 3),
            "sac_efficiency_std": round(float(np.std(effs)), 3),
            "sac_main_sequence_slope": round(ms_slope, 3),
            "sac_main_sequence_r2": round(ms_r2, 3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Blink Detection (EAR threshold)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_blinks(self, ts, ears) -> list:
        blinks = []
        in_blink = False
        blink_start = 0
        for i in range(len(ears)):
            if ears[i] < BLINK_EAR_THRESHOLD and not in_blink:
                in_blink = True
                blink_start = i
            elif ears[i] >= BLINK_EAR_THRESHOLD and in_blink:
                in_blink = False
                dur = float(ts[i] - ts[blink_start])
                if BLINK_MIN_DURATION_MS <= dur <= BLINK_MAX_DURATION_MS:
                    blinks.append({"start_ms": float(ts[blink_start]), "duration_ms": dur})
        return blinks

    def _blink_metrics(self, blinks, duration_s) -> dict:
        if not blinks:
            return {"blink_count": 0, "blink_rate_per_min": 0,
                    "blink_dur_mean_ms": 0, "blink_dur_std_ms": 0,
                    "blink_regularity_cv": 0}
        durs = [b["duration_ms"] for b in blinks]
        ibis = [blinks[i + 1]["start_ms"] - blinks[i]["start_ms"]
                for i in range(len(blinks) - 1)]
        cv = float(np.std(ibis) / np.mean(ibis)) if ibis and np.mean(ibis) > 0 else 0
        return {
            "blink_count": len(blinks),
            "blink_rate_per_min": round(len(blinks) / max(duration_s / 60, 0.01), 1),
            "blink_dur_mean_ms": round(float(np.mean(durs)), 1),
            "blink_dur_std_ms": round(float(np.std(durs)), 1),
            "blink_regularity_cv": round(cv, 3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Scanpath Metrics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _scanpath_metrics(self, xs, ys, vel) -> dict:
        dx = np.diff(xs)
        dy = np.diff(ys)
        step_lengths = np.sqrt(dx ** 2 + dy ** 2)
        scanpath_length = float(np.sum(step_lengths))

        straight = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
        path_efficiency = straight / max(scanpath_length, 0.01)

        bins = 10
        x_bins = np.linspace(np.min(xs) - 0.1, np.max(xs) + 0.1, bins + 1)
        y_bins = np.linspace(np.min(ys) - 0.1, np.max(ys) + 0.1, bins + 1)
        hist, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
        hist_flat = hist.flatten()
        hist_prob = hist_flat / max(np.sum(hist_flat), 1)
        gaze_entropy = float(sp_entropy(hist_prob + 1e-10, base=2))

        x_idx = np.clip(np.digitize(xs, x_bins) - 1, 0, bins - 1)
        y_idx = np.clip(np.digitize(ys, y_bins) - 1, 0, bins - 1)
        cell_ids = x_idx * bins + y_idx
        transitions = {}
        for i in range(len(cell_ids) - 1):
            key = (cell_ids[i], cell_ids[i + 1])
            transitions[key] = transitions.get(key, 0) + 1
        if transitions:
            trans_counts = np.array(list(transitions.values()), dtype=float)
            trans_probs = trans_counts / np.sum(trans_counts)
            transition_entropy = float(sp_entropy(trans_probs, base=2))
        else:
            transition_entropy = 0.0

        hull_area = 0.0
        try:
            points = np.column_stack([xs, ys])
            if len(np.unique(points, axis=0)) >= 3:
                hull = ConvexHull(points)
                hull_area = float(hull.volume)
        except Exception:
            pass

        vel_mean = float(np.mean(vel)) if len(vel) > 0 else 0
        vel_std = float(np.std(vel)) if len(vel) > 0 else 0

        return {
            "scanpath_length_deg": round(scanpath_length, 2),
            "scanpath_efficiency": round(path_efficiency, 4),
            "gaze_entropy_bits": round(gaze_entropy, 3),
            "gaze_transition_entropy_bits": round(transition_entropy, 3),
            "gaze_convex_hull_area_deg2": round(hull_area, 3),
            "gaze_velocity_mean_deg_s": round(vel_mean, 1),
            "gaze_velocity_std_deg_s": round(vel_std, 1),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Pupil Metrics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _pupil_metrics(self, pupils) -> dict:
        valid = pupils[pupils > 0]
        if len(valid) < 5:
            return {"pupil_diameter_mean_px": 0, "pupil_diameter_std_px": 0,
                    "pupil_diameter_cv": 0}
        return {
            "pupil_diameter_mean_px": round(float(np.mean(valid)), 2),
            "pupil_diameter_std_px": round(float(np.std(valid)), 2),
            "pupil_diameter_cv": round(float(np.std(valid) / np.mean(valid)), 4),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. Fixation-Saccade Dynamics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _dynamics_metrics(self, fixations, saccades, duration_s) -> dict:
        fix_total = sum(f["duration_ms"] for f in fixations) if fixations else 0
        sac_total = sum(s["duration_ms"] for s in saccades) if saccades else 0
        total = fix_total + sac_total
        return {
            "fix_sac_ratio": round(fix_total / max(sac_total, 1), 2),
            "fix_sac_time_coverage": round(total / max(duration_s * 1000, 1), 3),
            "mean_intersaccadic_interval_ms": round(
                float(np.mean(np.diff([s["start_ms"] for s in saccades])))
                if len(saccades) > 1 else 0, 1),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Visualization overlay
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _draw_overlay(self, frame, landmarks, sample):
        h, w = frame.shape[:2]

        def lm_px(idx):
            return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

        for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
            pt = lm_px(idx)
            cv2.circle(frame, pt, 3, (0, 255, 200), -1)
            cv2.circle(frame, pt, 8, (0, 255, 200), 1)

        for idx in [LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER]:
            cv2.circle(frame, lm_px(idx), 2, (100, 100, 100), -1)

        if sample:
            cx, cy = w // 2, h // 2
            gx = int(cx + sample.x * 10)
            gy = int(cy + sample.y * 10)
            cv2.arrowedLine(frame, (cx, 30), (gx, 30), (0, 255, 200), 2)

    def _draw_live_metrics(self, frame):
        if len(self.gaze_buffer) < 30:
            return
        recent = list(self.gaze_buffer)[-60:]
        ts = [s.t for s in recent]
        xs = [s.x for s in recent]
        ys = [s.y for s in recent]

        if len(ts) > 1:
            dt = (ts[-1] - ts[0]) / 1000
            path = sum(np.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2)
                       for i in range(1, len(xs)))
            vel = path / max(dt, 0.01)
        else:
            vel = 0

        ear = (recent[-1].ear_l + recent[-1].ear_r) / 2
        y0 = 20
        lines = [
            f"Gaze Vel: {vel:.1f} deg/s",
            f"EAR: {ear:.2f}",
            f"Samples: {len(self.gaze_buffer)}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (frame.shape[1] - 220, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
