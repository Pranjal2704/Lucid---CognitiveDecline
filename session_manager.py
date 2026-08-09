"""
Lucid Session Manager
─────────────────────────
Persists session data as JSON, computes longitudinal baselines,
and detects metric drift using per-user z-score deviation.
"""

import os
import json
import glob
from datetime import datetime
from typing import Optional
import numpy as np

SESSION_DIR = "sessions"


class SessionManager:
    """
    Manages session persistence and longitudinal analysis.

    Each session is a JSON file: sessions/YYYY-MM-DD_HH-MM-SS.json
    Contains all metrics from eye tracking, speech, and tasks.
    """

    def __init__(self, session_dir: str = SESSION_DIR):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)

    def save_session(self, metrics: dict, session_type: str = "mixed") -> str:
        """Save a session to disk. Returns the filepath."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session = {
            "timestamp": datetime.now().isoformat(),
            "session_type": session_type,
            "metrics": metrics,
        }
        filepath = os.path.join(self.session_dir, f"{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(session, f, indent=2, default=str)
        return filepath

    def load_session(self, filepath: str) -> dict:
        with open(filepath) as f:
            return json.load(f)

    def list_sessions(self) -> list:
        """List all sessions sorted by date."""
        files = sorted(glob.glob(os.path.join(self.session_dir, "*.json")))
        sessions = []
        for f in files:
            try:
                data = self.load_session(f)
                sessions.append({
                    "filepath": f,
                    "timestamp": data.get("timestamp", ""),
                    "type": data.get("session_type", "unknown"),
                    "metric_count": len(data.get("metrics", {})),
                })
            except Exception:
                pass
        return sessions

    def get_all_metrics(self) -> list:
        """Load metrics from all sessions, ordered chronologically."""
        files = sorted(glob.glob(os.path.join(self.session_dir, "*.json")))
        all_metrics = []
        for f in files:
            try:
                data = self.load_session(f)
                entry = {"timestamp": data.get("timestamp", "")}
                entry.update(data.get("metrics", {}))
                all_metrics.append(entry)
            except Exception:
                pass
        return all_metrics

    def compute_baseline(self, n_baseline: int = 5) -> dict:
        """
        Compute per-metric baseline from the first N sessions.
        Returns {metric_name: {"mean": float, "std": float}}.
        """
        all_sessions = self.get_all_metrics()
        if len(all_sessions) < n_baseline:
            n_baseline = len(all_sessions)
        if n_baseline == 0:
            return {}

        baseline_sessions = all_sessions[:n_baseline]

        # Collect all numeric metric keys
        all_keys = set()
        for s in baseline_sessions:
            for k, v in s.items():
                if isinstance(v, (int, float)) and k != "timestamp":
                    all_keys.add(k)

        baseline = {}
        for key in all_keys:
            values = []
            for s in baseline_sessions:
                if key in s and isinstance(s[key], (int, float)):
                    values.append(float(s[key]))
            if len(values) >= 2:
                baseline[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }

        return baseline

    def compute_drift(self, baseline: Optional[dict] = None,
                      n_baseline: int = 5, n_recent: int = 3) -> dict:
        """
        Compare recent sessions to baseline using z-scores.
        Returns metrics sorted by absolute z-score (largest drift first).

        Positive z = metric increased. Negative z = metric decreased.
        Clinical significance typically at |z| > 1.5 to 2.0.
        """
        if baseline is None:
            baseline = self.compute_baseline(n_baseline)
        if not baseline:
            return {"error": "insufficient_baseline_data"}

        all_sessions = self.get_all_metrics()
        if len(all_sessions) < n_baseline + 1:
            return {"error": "need_more_sessions"}

        recent = all_sessions[-n_recent:]

        drift = {}
        for key, bl in baseline.items():
            if bl["std"] < 1e-10:
                continue  # skip constant metrics

            recent_vals = []
            for s in recent:
                if key in s and isinstance(s[key], (int, float)):
                    recent_vals.append(float(s[key]))

            if not recent_vals:
                continue

            recent_mean = float(np.mean(recent_vals))
            z = (recent_mean - bl["mean"]) / bl["std"]
            drift[key] = {
                "baseline_mean": round(bl["mean"], 4),
                "baseline_std": round(bl["std"], 4),
                "recent_mean": round(recent_mean, 4),
                "z_score": round(z, 3),
                "abs_z": round(abs(z), 3),
                "direction": "increased" if z > 0 else "decreased",
            }

        # Sort by absolute z-score
        drift = dict(sorted(drift.items(), key=lambda x: x[1]["abs_z"], reverse=True))
        return drift

    def get_longitudinal_series(self, metric_keys: list) -> dict:
        """
        Extract time series for specified metrics across all sessions.
        Returns {metric_key: [(timestamp, value), ...]}.
        """
        all_sessions = self.get_all_metrics()
        series = {k: [] for k in metric_keys}

        for s in all_sessions:
            ts = s.get("timestamp", "")
            for k in metric_keys:
                if k in s and isinstance(s[k], (int, float)):
                    series[k].append((ts, float(s[k])))

        return series
