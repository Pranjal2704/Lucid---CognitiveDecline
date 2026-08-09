"""
Lucid Speech Collector
──────────────────────────
Background thread that continuously monitors the microphone,
detects speech segments via energy-based VAD, and feeds them
to SpeechAnalyzer for feature extraction.

Captures speech from any source: GMeet calls, phone calls,
dictation, conversations. No manual recording needed.

Pipeline:
  Mic → rolling buffer → VAD (energy threshold)
    → speech segment detection → SpeechAnalyzer
    → segment metrics → daily accumulation

Thread-safe: runs on a background thread, results collected
via get_pending_metrics().
"""

import time
import threading
import numpy as np
from collections import deque
from typing import List, Optional

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

from config import (
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS,
    SPEECH_VAD_ENERGY_THRESHOLD, SPEECH_VAD_MIN_SPEECH_S,
    SPEECH_VAD_MAX_SILENCE_S, SPEECH_VAD_FRAME_S,
    SPEECH_CAPTURE_BUFFER_S, SPEECH_ANALYSIS_MIN_S,
    SPEECH_ANALYSIS_MAX_S,
)


class SpeechCollector:
    """
    Background microphone monitoring with voice activity detection.

    Usage:
        collector = SpeechCollector()
        collector.start()                    # starts background thread
        metrics = collector.get_pending()    # retrieve analyzed segments
        collector.stop()                     # cleanup
    """

    def __init__(self):
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not installed — pip install sounddevice")

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._pending_metrics = []

        # Audio state
        self._buffer = deque(maxlen=int(AUDIO_SAMPLE_RATE * SPEECH_CAPTURE_BUFFER_S))
        self._frame_size = int(AUDIO_SAMPLE_RATE * SPEECH_VAD_FRAME_S)

        # VAD state
        self._in_speech = False
        self._speech_start_sample = 0
        self._silence_start_time = 0
        self._current_speech_samples = []
        self._total_samples_received = 0

        # Stats
        self.segments_analyzed = 0
        self.total_speech_s = 0
        self.total_monitor_s = 0

    def start(self):
        """Start background audio monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("  Speech monitoring started (background mic capture).")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        # Process any remaining speech
        self._finalize_segment()
        print(f"  Speech monitoring stopped. "
              f"Segments: {self.segments_analyzed}, "
              f"Speech: {self.total_speech_s:.0f}s")

    def get_pending(self) -> List[dict]:
        """Retrieve and clear pending analyzed speech segments (thread-safe)."""
        with self._lock:
            results = list(self._pending_metrics)
            self._pending_metrics.clear()
        return results

    @property
    def is_speaking(self) -> bool:
        """Whether voice activity is currently detected."""
        return self._in_speech

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Background thread
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _run(self):
        """Main audio capture loop (runs on background thread)."""
        start_time = time.time()

        def audio_callback(indata, frames, time_info, status):
            """Called by sounddevice for each audio chunk."""
            if status:
                pass  # ignore xruns silently
            samples = indata[:, 0].copy()  # mono
            self._process_audio_chunk(samples)

        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype="float32",
                blocksize=self._frame_size,
                callback=audio_callback,
            ):
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"  Speech capture error: {e}")
        finally:
            self.total_monitor_s = time.time() - start_time

    def _process_audio_chunk(self, samples: np.ndarray):
        """Process an audio chunk: VAD + segment accumulation."""
        self._total_samples_received += len(samples)

        # Compute frame energy (RMS)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        is_voiced = rms > SPEECH_VAD_ENERGY_THRESHOLD
        now = time.time()

        if is_voiced:
            if not self._in_speech:
                # Speech onset
                self._in_speech = True
                self._speech_start_sample = self._total_samples_received
                self._current_speech_samples = []

            self._current_speech_samples.append(samples)
            self._silence_start_time = 0

            # Cap segment length
            current_dur = len(self._current_speech_samples) * SPEECH_VAD_FRAME_S
            if current_dur >= SPEECH_ANALYSIS_MAX_S:
                self._finalize_segment()

        else:
            if self._in_speech:
                # Accumulate short silences within speech
                self._current_speech_samples.append(samples)

                if self._silence_start_time == 0:
                    self._silence_start_time = now

                # Check if silence is long enough to end segment
                silence_dur = now - self._silence_start_time
                if silence_dur >= SPEECH_VAD_MAX_SILENCE_S:
                    self._finalize_segment()

    def _finalize_segment(self):
        """Analyze accumulated speech segment if long enough."""
        if not self._current_speech_samples:
            self._in_speech = False
            return

        # Concatenate all samples
        audio = np.concatenate(self._current_speech_samples)
        duration = len(audio) / AUDIO_SAMPLE_RATE

        self._in_speech = False
        self._current_speech_samples = []
        self._silence_start_time = 0

        # Skip segments that are too short
        if duration < SPEECH_ANALYSIS_MIN_S:
            return

        self.total_speech_s += duration
        self.segments_analyzed += 1

        # Run analysis on background thread (already on background)
        try:
            from speech_analyzer import SpeechAnalyzer
            analyzer = SpeechAnalyzer()
            metrics = analyzer.analyze_array(audio.astype(np.float64), AUDIO_SAMPLE_RATE)
            metrics["_segment_type"] = "passive_speech"
            metrics["_timestamp"] = time.time()
            metrics["_duration_s"] = round(duration, 2)

            with self._lock:
                self._pending_metrics.append(metrics)

        except Exception as e:
            print(f"  Speech analysis error: {e}")