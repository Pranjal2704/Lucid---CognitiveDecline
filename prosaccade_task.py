"""
Lucid Prosaccade / Antisaccade / Smooth Pursuit Tasks
─────────────────────────────────────────────────────────
Standardized oculomotor assessment tasks using MediaPipe Tasks
FaceLandmarker API (not legacy face_mesh).

Prosaccade: look toward peripheral target as fast as possible.
Antisaccade: look away from peripheral target (inhibitory control).
Smooth Pursuit: track a smoothly moving target.
"""

import time
import random
import numpy as np
import cv2
import mediapipe as mp

from bak.config import (
    TASK_NUM_TRIALS, TASK_FIXATION_DURATION_MS, TASK_GAP_DURATION_MS,
    TASK_TARGET_DURATION_MS, TASK_ITI_MS, TASK_TARGET_POSITIONS,
    TASK_EXPRESS_SACCADE_THRESHOLD_MS, TASK_ANTICIPATORY_THRESHOLD_MS,
    SACCADE_VELOCITY_THRESHOLD, PURSUIT_TARGET_FREQ_HZ, PURSUIT_AMPLITUDE_DEG,
    CAMERA_WIDTH, CAMERA_HEIGHT,
)
from bak.eye_tracker import EyeTracker, create_landmarker, VisionRunningMode


class SaccadeTask:
    """
    Runs prosaccade or antisaccade task with live eye tracking.
    Returns per-trial and aggregate metrics.
    """

    def __init__(self, tracker: EyeTracker = None):
        self.tracker = tracker or EyeTracker()
        self.trials = []

    def run_prosaccade(self, num_trials: int = TASK_NUM_TRIALS) -> dict:
        return self._run_task("prosaccade", num_trials)

    def run_antisaccade(self, num_trials: int = TASK_NUM_TRIALS) -> dict:
        return self._run_task("antisaccade", num_trials)

    def _run_task(self, task_type: str, num_trials: int) -> dict:
        """Run the saccade task with OpenCV display and live tracking."""
        screen_w, screen_h = 1280, 720
        win_name = f"Lucid — {task_type.title()} Task"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, screen_w, screen_h)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Create FaceLandmarker via Tasks API (VIDEO mode, monotonic timestamps)
        landmarker = create_landmarker(VisionRunningMode.VIDEO)
        task_start = time.time() * 1000  # epoch for monotonic ts

        self.trials = []
        positions = TASK_TARGET_POSITIONS.copy()

        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  {task_type.upper()} TASK — {num_trials} trials")
        if task_type == "prosaccade":
            print(f"  ║  Look AT the target dot as fast as you can")
        else:
            print(f"  ║  Look AWAY from the target dot")
        print(f"  ╚══════════════════════════════════════════╝\n")

        try:
            for trial_num in range(num_trials):
                target_pos = positions[trial_num % len(positions)]
                target_px = (int(target_pos[0] * screen_w), int(target_pos[1] * screen_h))
                center_px = (screen_w // 2, screen_h // 2)

                # Expected saccade direction
                if task_type == "prosaccade":
                    expected_dir_x = target_pos[0] - 0.5
                else:
                    expected_dir_x = -(target_pos[0] - 0.5)  # opposite

                trial_data = {
                    "trial": trial_num + 1,
                    "type": task_type,
                    "target_x": target_pos[0],
                    "target_y": target_pos[1],
                }

                # ── Phase 1: Fixation ──
                fix_dur = random.uniform(*TASK_FIXATION_DURATION_MS)
                phase_start = time.time() * 1000
                while (time.time() * 1000 - phase_start) < fix_dur:
                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    cv2.line(canvas, (center_px[0] - 15, center_px[1]),
                             (center_px[0] + 15, center_px[1]), (200, 200, 200), 2)
                    cv2.line(canvas, (center_px[0], center_px[1] - 15),
                             (center_px[0], center_px[1] + 15), (200, 200, 200), 2)
                    self._update_tracking(cap, landmarker, task_start, canvas, screen_w, screen_h)
                    cv2.imshow(win_name, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return self._aggregate_results()

                # ── Phase 2: Gap (blank screen) ──
                gap_start = time.time() * 1000
                while (time.time() * 1000 - gap_start) < TASK_GAP_DURATION_MS:
                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    self._update_tracking(cap, landmarker, task_start, canvas, screen_w, screen_h)
                    cv2.imshow(win_name, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return self._aggregate_results()

                # Record gaze at target onset
                gaze_at_onset = self._get_current_gaze(cap, landmarker, task_start)

                # ── Phase 3: Target ──
                target_onset = time.time() * 1000
                saccade_detected = False
                reaction_time = None
                correct_direction = None
                gaze_samples = []

                while (time.time() * 1000 - target_onset) < TASK_TARGET_DURATION_MS:
                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    cv2.circle(canvas, target_px, 14, (0, 100, 255), -1)
                    cv2.circle(canvas, target_px, 18, (0, 70, 180), 2)

                    gaze = self._update_tracking(cap, landmarker, task_start, canvas, screen_w, screen_h)
                    cv2.imshow(win_name, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return self._aggregate_results()

                    if gaze:
                        gaze_samples.append(gaze)

                    # Detect saccade
                    if gaze and not saccade_detected and len(gaze_samples) > 3:
                        recent = gaze_samples[-3:]
                        vel = sum(
                            abs(recent[i]["x"] - recent[i - 1]["x"])
                            for i in range(1, len(recent))
                        ) / len(recent)
                        if vel > 2.0:
                            saccade_detected = True
                            reaction_time = time.time() * 1000 - target_onset
                            saccade_dir = gaze["x"] - (gaze_at_onset["x"] if gaze_at_onset else 0)
                            correct_direction = (saccade_dir * expected_dir_x) > 0

                trial_data["reaction_time_ms"] = round(reaction_time, 1) if reaction_time else None
                trial_data["correct_direction"] = correct_direction
                trial_data["saccade_detected"] = saccade_detected

                if reaction_time:
                    trial_data["is_express"] = reaction_time < TASK_EXPRESS_SACCADE_THRESHOLD_MS
                    trial_data["is_anticipatory"] = reaction_time < TASK_ANTICIPATORY_THRESHOLD_MS
                else:
                    trial_data["is_express"] = False
                    trial_data["is_anticipatory"] = False

                self.trials.append(trial_data)
                status = "✓" if correct_direction else "✗" if correct_direction is False else "—"
                rt_str = f"{reaction_time:.0f}ms" if reaction_time else "timeout"
                print(f"  Trial {trial_num + 1:2d}/{num_trials}  {status}  RT: {rt_str}")

                # ITI
                time.sleep(TASK_ITI_MS / 1000)

        finally:
            cap.release()
            landmarker.close()
            cv2.destroyWindow(win_name)

        return self._aggregate_results()

    def _update_tracking(self, cap, landmarker, task_start_ms,
                         canvas, sw, sh) -> dict:
        """Read frame, run FaceLandmarker, record gaze, draw small preview."""
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)

        # Monotonic timestamp relative to task start
        timestamp_ms = int(time.time() * 1000 - task_start_ms)

        # Delegate to tracker's frame processor
        gaze = self.tracker.process_frame_with_landmarker(
            frame, landmarker, timestamp_ms
        )

        # Draw small webcam preview in corner
        preview = cv2.resize(frame, (160, 120))
        canvas[sh - 130:sh - 10, sw - 170:sw - 10] = preview

        return gaze

    def _get_current_gaze(self, cap, landmarker, task_start_ms) -> dict:
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000 - task_start_ms)
        return self.tracker.process_frame_with_landmarker(
            frame, landmarker, timestamp_ms
        )

    def _aggregate_results(self) -> dict:
        if not self.trials:
            return {"error": "no_trials"}

        valid = [t for t in self.trials if t["reaction_time_ms"] is not None]
        valid_noanticipatory = [t for t in valid if not t["is_anticipatory"]]

        rts = [t["reaction_time_ms"] for t in valid_noanticipatory]
        correct = [t for t in valid_noanticipatory if t["correct_direction"]]
        errors = [t for t in valid_noanticipatory if t["correct_direction"] is False]
        express = [t for t in valid_noanticipatory if t["is_express"]]
        anticipatory = [t for t in valid if t["is_anticipatory"]]
        timeouts = [t for t in self.trials if t["reaction_time_ms"] is None]

        result = {
            "task_type": self.trials[0]["type"],
            "total_trials": len(self.trials),
            "valid_trials": len(valid_noanticipatory),
            "timeout_count": len(timeouts),
            "anticipatory_count": len(anticipatory),
            "trials": self.trials,
        }

        if rts:
            result.update({
                "rt_mean_ms": round(float(np.mean(rts)), 1),
                "rt_median_ms": round(float(np.median(rts)), 1),
                "rt_std_ms": round(float(np.std(rts)), 1),
                "rt_min_ms": round(float(np.min(rts)), 1),
                "rt_max_ms": round(float(np.max(rts)), 1),
                "rt_cv": round(float(np.std(rts) / np.mean(rts)), 3) if np.mean(rts) > 0 else 0,
            })

        result.update({
            "accuracy_pct": round(len(correct) / max(len(valid_noanticipatory), 1) * 100, 1),
            "error_rate_pct": round(len(errors) / max(len(valid_noanticipatory), 1) * 100, 1),
            "express_saccade_pct": round(len(express) / max(len(valid_noanticipatory), 1) * 100, 1),
        })

        if correct:
            correct_rts = [t["reaction_time_ms"] for t in correct]
            result["rt_correct_mean_ms"] = round(float(np.mean(correct_rts)), 1)

        if errors:
            error_rts = [t["reaction_time_ms"] for t in errors]
            result["rt_error_mean_ms"] = round(float(np.mean(error_rts)), 1)

        return result


class SmoothPursuitTask:
    """
    Smooth pursuit assessment: track a smoothly moving dot.
    Measures pursuit gain, position error, and catch-up saccades.
    """

    def __init__(self, tracker: EyeTracker = None):
        self.tracker = tracker or EyeTracker()

    def run(self, duration_s: float = 30.0) -> dict:
        screen_w, screen_h = 1280, 720
        win_name = "Lucid — Smooth Pursuit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, screen_w, screen_h)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Tasks API landmarker
        landmarker = create_landmarker(VisionRunningMode.VIDEO)
        task_start = time.time() * 1000

        print(f"\n  Smooth Pursuit Task — {duration_s}s")
        print(f"  Follow the moving dot with your eyes. Keep your head still.\n")

        target_positions = []
        gaze_positions = []
        start_time = time.time()

        try:
            while (time.time() - start_time) < duration_s:
                elapsed = time.time() - start_time
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

                # Horizontal sinusoidal target
                target_x = 0.5 + 0.35 * np.sin(2 * np.pi * PURSUIT_TARGET_FREQ_HZ * elapsed)
                target_y = 0.5
                target_px = (int(target_x * screen_w), int(target_y * screen_h))

                cv2.circle(canvas, target_px, 12, (0, 200, 255), -1)

                # Track gaze via Tasks API
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    timestamp_ms = int(time.time() * 1000 - task_start)

                    gaze = self.tracker.process_frame_with_landmarker(
                        frame, landmarker, timestamp_ms
                    )
                    if gaze:
                        target_deg_x = (target_x - 0.5) * 50  # approx degrees
                        target_positions.append({"t": elapsed, "x": target_deg_x})
                        gaze_positions.append({"t": elapsed, "x": gaze["x"]})

                    preview = cv2.resize(frame, (160, 120))
                    canvas[screen_h - 130:screen_h - 10, screen_w - 170:screen_w - 10] = preview

                cv2.imshow(win_name, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            landmarker.close()
            cv2.destroyWindow(win_name)

        return self._compute_pursuit_metrics(target_positions, gaze_positions)

    def _compute_pursuit_metrics(self, targets, gazes) -> dict:
        if len(targets) < 20 or len(gazes) < 20:
            return {"error": "insufficient_data"}

        t_x = np.array([t["x"] for t in targets])
        g_x = np.array([g["x"] for g in gazes])
        min_len = min(len(t_x), len(g_x))
        t_x, g_x = t_x[:min_len], g_x[:min_len]

        errors = g_x - t_x
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        dt = np.diff([t["t"] for t in targets[:min_len]])
        dt[dt < 1e-6] = 1e-6
        target_vel = np.abs(np.diff(t_x) / dt)
        gaze_vel = np.abs(np.diff(g_x) / dt)

        valid = target_vel > 1.0
        if np.any(valid):
            gain = float(np.mean(gaze_vel[valid] / target_vel[valid]))
        else:
            gain = 0.0

        catch_up_count = int(np.sum(gaze_vel > SACCADE_VELOCITY_THRESHOLD))

        return {
            "pursuit_duration_s": round(float(targets[-1]["t"]) if targets else 0, 1),
            "pursuit_gain": round(gain, 3),
            "pursuit_rmse_deg": round(rmse, 2),
            "pursuit_position_error_mean_deg": round(float(np.mean(np.abs(errors))), 2),
            "pursuit_position_error_std_deg": round(float(np.std(errors)), 2),
            "pursuit_catch_up_saccade_count": catch_up_count,
            "pursuit_catch_up_saccade_rate": round(
                catch_up_count / max(float(targets[-1]["t"]), 0.01), 2) if targets else 0,
        }