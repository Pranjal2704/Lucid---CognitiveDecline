"""
Lucid Speech Analyzer
─────────────────────────
Extracts 60+ speech/voice metrics from audio using Praat (parselmouth)
and librosa. Every metric has documented clinical relevance to cognitive
or neurological function.

Metric groups:
  1. Fundamental frequency (F0) — prosody, emotional affect, motor control
  2. Jitter (5 variants) — vocal fold regularity → neuromotor integrity
  3. Shimmer (5 variants) — amplitude stability → laryngeal control
  4. Harmonics & noise — voice clarity → neurological health
  5. Voice quality — breaks, unvoiced frames → motor planning
  6. Formants (F1–F3 + bandwidths) — articulatory precision
  7. Temporal dynamics — pauses, speech rate → executive function, word retrieval
  8. Spectral features — centroid, rolloff, flux, flatness, contrast
  9. MFCCs + deltas — full vocal tract characterization
  10. Energy features — RMS, ZCR, entropy
"""

import numpy as np
import parselmouth
from parselmouth import praat
import warnings

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not installed — spectral/MFCC metrics will be skipped")

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO_IO = True
except ImportError:
    HAS_AUDIO_IO = False

from config import (
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_RECORD_SECONDS_DEFAULT,
    PITCH_FLOOR_HZ, PITCH_CEILING_HZ, PITCH_TIME_STEP,
    FORMANT_MAX_FREQUENCY, FORMANT_NUM_FORMANTS, FORMANT_WINDOW_LENGTH,
    INTENSITY_MIN_PITCH,
    PAUSE_INTENSITY_THRESHOLD_DB, PAUSE_MIN_DURATION_S, PAUSE_LONG_THRESHOLD_S,
    SPECTRAL_N_FFT, SPECTRAL_HOP_LENGTH, SPECTRAL_N_MFCC, SPECTRAL_N_MELS,
)


class SpeechAnalyzer:
    """
    Comprehensive speech/voice analysis engine.

    Usage:
        analyzer = SpeechAnalyzer()
        metrics = analyzer.analyze_file("recording.wav")
        # or
        metrics = analyzer.record_and_analyze(duration=30)
    """

    def analyze_file(self, filepath: str) -> dict:
        """Analyze an audio file and return all metrics."""
        sound = parselmouth.Sound(filepath)
        return self._analyze_sound(sound, filepath)

    def analyze_array(self, samples: np.ndarray, sample_rate: int) -> dict:
        """Analyze a numpy array of audio samples."""
        sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
        return self._analyze_sound(sound, source="array")

    def record_and_analyze(self, duration: float = AUDIO_RECORD_SECONDS_DEFAULT,
                           save_path: str = None) -> dict:
        """Record from microphone and analyze."""
        if not HAS_AUDIO_IO:
            raise RuntimeError("sounddevice/soundfile not installed")

        print(f"  Recording {duration}s of audio... speak naturally.")
        audio = sd.rec(int(duration * AUDIO_SAMPLE_RATE),
                       samplerate=AUDIO_SAMPLE_RATE,
                       channels=AUDIO_CHANNELS, dtype="float64")
        sd.wait()
        print("  Recording complete.")

        audio = audio.flatten()

        if save_path:
            sf.write(save_path, audio, AUDIO_SAMPLE_RATE)

        sound = parselmouth.Sound(audio, sampling_frequency=AUDIO_SAMPLE_RATE)
        return self._analyze_sound(sound, source=save_path or "live_recording")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core analysis pipeline
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_sound(self, sound: parselmouth.Sound, source: str = "") -> dict:
        """Run all analysis passes on a Praat Sound object."""
        duration = sound.get_total_duration()
        sr = int(sound.sampling_frequency)

        # Core Praat objects
        pitch = sound.to_pitch_cc(
            time_step=PITCH_TIME_STEP,
            pitch_floor=PITCH_FLOOR_HZ,
            pitch_ceiling=PITCH_CEILING_HZ,
        )
        point_process = praat.call([sound, pitch], "To PointProcess (cc)")
        formant = sound.to_formant_burg(
            time_step=PITCH_TIME_STEP,
            max_number_of_formants=FORMANT_NUM_FORMANTS,
            maximum_formant=FORMANT_MAX_FREQUENCY,
            window_length=FORMANT_WINDOW_LENGTH,
        )
        harmonicity = sound.to_harmonicity_cc(
            time_step=PITCH_TIME_STEP,
            minimum_pitch=PITCH_FLOOR_HZ,
        )
        intensity = sound.to_intensity(
            minimum_pitch=INTENSITY_MIN_PITCH,
            time_step=PITCH_TIME_STEP,
        )

        # Numpy samples for librosa
        samples = sound.values[0]

        # Assemble all metrics
        metrics = {
            "source": source,
            "duration_s": round(duration, 3),
            "sample_rate": sr,
        }

        metrics.update(self._f0_metrics(pitch, duration))
        metrics.update(self._jitter_metrics(point_process))
        metrics.update(self._shimmer_metrics(sound, point_process))
        metrics.update(self._hnr_metrics(harmonicity))
        metrics.update(self._voice_quality_metrics(sound, pitch, point_process))
        metrics.update(self._formant_metrics(formant, pitch, duration))
        metrics.update(self._temporal_metrics(intensity, duration))
        metrics.update(self._intensity_metrics(intensity))

        if HAS_LIBROSA:
            metrics.update(self._spectral_metrics(samples, sr))
            metrics.update(self._mfcc_metrics(samples, sr))
            metrics.update(self._energy_metrics(samples, sr))

        return metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Fundamental Frequency (F0)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _f0_metrics(self, pitch, duration: float) -> dict:
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) == 0:
            return {k: 0.0 for k in [
                "f0_mean", "f0_median", "f0_std", "f0_min", "f0_max",
                "f0_range", "f0_cv", "f0_iqr", "f0_skewness", "f0_kurtosis",
                "f0_slope", "f0_voiced_fraction",
            ]}

        from scipy import stats as sp_stats

        # F0 contour slope via linear regression on voiced frames
        voiced_indices = np.where(f0_values > 0)[0]
        times = voiced_indices * PITCH_TIME_STEP
        if len(times) > 2:
            slope, _, _, _, _ = sp_stats.linregress(times, f0_voiced)
        else:
            slope = 0.0

        return {
            "f0_mean": round(float(np.mean(f0_voiced)), 2),
            "f0_median": round(float(np.median(f0_voiced)), 2),
            "f0_std": round(float(np.std(f0_voiced)), 2),
            "f0_min": round(float(np.min(f0_voiced)), 2),
            "f0_max": round(float(np.max(f0_voiced)), 2),
            "f0_range": round(float(np.max(f0_voiced) - np.min(f0_voiced)), 2),
            "f0_cv": round(float(np.std(f0_voiced) / np.mean(f0_voiced)) * 100, 2) if np.mean(f0_voiced) > 0 else 0,
            "f0_iqr": round(float(np.percentile(f0_voiced, 75) - np.percentile(f0_voiced, 25)), 2),
            "f0_skewness": round(float(sp_stats.skew(f0_voiced)), 3),
            "f0_kurtosis": round(float(sp_stats.kurtosis(f0_voiced)), 3),
            "f0_slope": round(float(slope), 4),
            "f0_voiced_fraction": round(float(len(f0_voiced) / len(f0_values)), 3) if len(f0_values) > 0 else 0,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Jitter — cycle-to-cycle frequency perturbation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _jitter_metrics(self, pp) -> dict:
        def _safe_call(method, *args):
            try:
                v = praat.call(pp, method, *args)
                return round(float(v), 6) if v is not None and not np.isnan(v) else 0.0
            except Exception:
                return 0.0

        return {
            "jitter_local":     _safe_call("Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_local_abs": _safe_call("Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_rap":       _safe_call("Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_ppq5":      _safe_call("Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_ddp":       _safe_call("Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Shimmer — cycle-to-cycle amplitude perturbation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _shimmer_metrics(self, sound, pp) -> dict:
        def _safe_call(method, *args):
            try:
                v = praat.call([sound, pp], method, *args)
                return round(float(v), 6) if v is not None and not np.isnan(v) else 0.0
            except Exception:
                return 0.0

        return {
            "shimmer_local":     _safe_call("Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_local_db":  _safe_call("Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq3":      _safe_call("Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq5":      _safe_call("Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq11":     _safe_call("Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Harmonics-to-Noise Ratio
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _hnr_metrics(self, harmonicity) -> dict:
        hnr_values = harmonicity.values[0]
        hnr_valid = hnr_values[hnr_values != -200]  # Praat uses -200 for unvoiced

        if len(hnr_valid) == 0:
            return {"hnr_mean": 0.0, "hnr_std": 0.0, "hnr_min": 0.0, "nhr_mean": 0.0}

        hnr_mean = float(np.mean(hnr_valid))
        return {
            "hnr_mean": round(hnr_mean, 2),
            "hnr_std": round(float(np.std(hnr_valid)), 2),
            "hnr_min": round(float(np.min(hnr_valid)), 2),
            "nhr_mean": round(1.0 / max(hnr_mean, 0.01), 4),  # noise-to-harmonics
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Voice Quality
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _voice_quality_metrics(self, sound, pitch, pp) -> dict:
        dur = sound.get_total_duration()

        def _safe(method, *args, default=0.0):
            try:
                v = praat.call(pp, method, *args)
                return round(float(v), 4) if v is not None and not np.isnan(v) else default
            except Exception:
                return default

        num_pulses = _safe("Get number of periods", 0, 0, 0.0001, 0.02, 1.3)
        num_voice_breaks = 0
        degree_voice_breaks = 0.0

        try:
            vr = praat.call([pitch], "Count voiced frames")
            uvr = praat.call([pitch], "Count unvoiced frames") if False else 0
            # Voice breaks via Praat's voice report
            report = praat.call([sound, pitch, pp], "Voice report", 0, 0, PITCH_FLOOR_HZ, PITCH_CEILING_HZ, 1.3, 1.6, 0.03, 0.45)
            # Parse report for voice breaks
            for line in report.split("\n"):
                if "Number of voice breaks" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        num_voice_breaks = int(parts[1].strip())
                elif "Degree of voice breaks" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        pct = parts[1].strip().replace("%", "")
                        try:
                            degree_voice_breaks = float(pct)
                        except ValueError:
                            pass
        except Exception:
            pass

        f0_values = pitch.selected_array["frequency"]
        total_frames = len(f0_values)
        unvoiced_frames = np.sum(f0_values == 0)

        return {
            "voice_breaks_count": num_voice_breaks,
            "voice_breaks_degree_pct": round(degree_voice_breaks, 2),
            "unvoiced_fraction": round(float(unvoiced_frames / max(total_frames, 1)), 3),
            "num_pulses": int(num_pulses) if num_pulses else 0,
            "mean_period_s": _safe("Get mean period", 0, 0, 0.0001, 0.02, 1.3),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. Formants
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _formant_metrics(self, formant, pitch, duration: float) -> dict:
        """Extract F1, F2, F3 frequencies and bandwidths at voiced frames."""
        f0_vals = pitch.selected_array["frequency"]
        times = np.arange(len(f0_vals)) * PITCH_TIME_STEP

        results = {}
        for fi in range(1, 4):  # F1, F2, F3
            freqs, bws = [], []
            for t, f0 in zip(times, f0_vals):
                if f0 > 0 and t < duration:
                    try:
                        f = formant.get_value_at_time(fi, t)
                        b = formant.get_bandwidth_at_time(fi, t)
                        if f is not None and not np.isnan(f) and f > 0:
                            freqs.append(f)
                        if b is not None and not np.isnan(b) and b > 0:
                            bws.append(b)
                    except Exception:
                        pass

            prefix = f"f{fi}"
            if len(freqs) > 0:
                results[f"{prefix}_mean"] = round(float(np.mean(freqs)), 1)
                results[f"{prefix}_std"] = round(float(np.std(freqs)), 1)
                results[f"{prefix}_median"] = round(float(np.median(freqs)), 1)
            else:
                results[f"{prefix}_mean"] = 0.0
                results[f"{prefix}_std"] = 0.0
                results[f"{prefix}_median"] = 0.0

            if len(bws) > 0:
                results[f"{prefix}_bandwidth_mean"] = round(float(np.mean(bws)), 1)
            else:
                results[f"{prefix}_bandwidth_mean"] = 0.0

        # Vowel space area approximation: triangle formed by mean (F1, F2)
        # of /a/, /i/, /u/ — we approximate with overall F1×F2 dispersion
        f1m = results.get("f1_mean", 0)
        f2m = results.get("f2_mean", 0)
        f1s = results.get("f1_std", 0)
        f2s = results.get("f2_std", 0)
        results["vowel_space_dispersion"] = round(float(f1s * f2s), 1) if f1s > 0 and f2s > 0 else 0.0
        results["f2_f1_ratio"] = round(f2m / f1m, 3) if f1m > 0 else 0.0

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Temporal Dynamics — pauses, speech rate
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _temporal_metrics(self, intensity, duration: float) -> dict:
        """Detect pauses from intensity contour and compute temporal features."""
        int_values = intensity.values[0]
        int_times = np.linspace(intensity.get_time_from_frame_number(1),
                                intensity.get_time_from_frame_number(len(int_values)),
                                len(int_values))

        if len(int_values) == 0:
            return self._empty_temporal()

        # Adaptive threshold: use percentile-based approach instead of
        # fixed peak-35dB (which fails with noisy laptop mics).
        # The 25th percentile of intensity approximates the speech/silence
        # boundary in most recordings. For clean recordings, fall back to
        # the peak-relative method if it gives a lower threshold.
        percentile_thresh = float(np.percentile(int_values, 25))
        peak_relative_thresh = float(np.max(int_values)) + PAUSE_INTENSITY_THRESHOLD_DB
        threshold = max(percentile_thresh, peak_relative_thresh)

        # Detect pause regions
        is_silent = int_values < threshold
        pauses = []
        in_pause = False
        pause_start = 0.0

        for i, (t, silent) in enumerate(zip(int_times, is_silent)):
            if silent and not in_pause:
                in_pause = True
                pause_start = t
            elif not silent and in_pause:
                in_pause = False
                pause_dur = t - pause_start
                if pause_dur >= PAUSE_MIN_DURATION_S:
                    pauses.append({"start": round(pause_start, 3),
                                   "duration": round(pause_dur, 3)})

        # Handle final pause
        if in_pause:
            pause_dur = duration - pause_start
            if pause_dur >= PAUSE_MIN_DURATION_S:
                pauses.append({"start": round(pause_start, 3),
                               "duration": round(pause_dur, 3)})

        pause_durations = [p["duration"] for p in pauses]
        total_pause = sum(pause_durations)
        total_speech = duration - total_pause
        short_pauses = [d for d in pause_durations if d < PAUSE_LONG_THRESHOLD_S]
        long_pauses = [d for d in pause_durations if d >= PAUSE_LONG_THRESHOLD_S]

        # Speech rate proxy: count intensity peaks above threshold (≈ syllable nuclei)
        # Using Praat's intensity-based approach
        voiced_segments = []
        in_speech = False
        seg_start = 0.0
        for i, (t, silent) in enumerate(zip(int_times, is_silent)):
            if not silent and not in_speech:
                in_speech = True
                seg_start = t
            elif silent and in_speech:
                in_speech = False
                voiced_segments.append(t - seg_start)
        if in_speech:
            voiced_segments.append(duration - seg_start)

        # Estimate syllable count from intensity peaks in voiced segments
        from scipy.signal import find_peaks
        if len(int_values) > 10:
            # Smooth intensity
            kernel_size = max(3, int(0.04 / PITCH_TIME_STEP))  # ~40ms window
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = np.convolve(int_values, np.ones(kernel_size) / kernel_size, mode="same")
            peaks, _ = find_peaks(smoothed, distance=int(0.1 / PITCH_TIME_STEP),  # min 100ms between syllables
                                  height=threshold)
            syllable_count = len(peaks)
        else:
            syllable_count = 0

        speech_rate = syllable_count / duration if duration > 0 else 0
        artic_rate = syllable_count / max(total_speech, 0.01) if total_speech > 0 else 0

        return {
            "pause_count": len(pauses),
            "pause_total_s": round(total_pause, 3),
            "pause_mean_s": round(float(np.mean(pause_durations)), 3) if pause_durations else 0.0,
            "pause_max_s": round(float(np.max(pause_durations)), 3) if pause_durations else 0.0,
            "pause_std_s": round(float(np.std(pause_durations)), 3) if len(pause_durations) > 1 else 0.0,
            "pause_rate_per_min": round(len(pauses) / (duration / 60), 2) if duration > 0 else 0,
            "short_pause_count": len(short_pauses),
            "long_pause_count": len(long_pauses),
            "pause_ratio": round(total_pause / max(duration, 0.01), 3),
            "speech_to_pause_ratio": round(total_speech / max(total_pause, 0.01), 2),
            "phonation_time_s": round(total_speech, 3),
            "phonation_ratio": round(total_speech / max(duration, 0.01), 3),
            "syllable_count_est": syllable_count,
            "speech_rate_syl_per_s": round(speech_rate, 2),
            "articulation_rate_syl_per_s": round(artic_rate, 2),
            "voiced_segment_count": len(voiced_segments),
            "mean_utterance_length_s": round(float(np.mean(voiced_segments)), 3) if voiced_segments else 0.0,
        }

    def _empty_temporal(self) -> dict:
        """Return zeroed temporal metrics when intensity data is empty."""
        return {k: 0 for k in [
            "pause_count", "pause_total_s", "pause_mean_s", "pause_max_s",
            "pause_std_s", "pause_rate_per_min", "short_pause_count",
            "long_pause_count", "pause_ratio", "speech_to_pause_ratio",
            "phonation_time_s", "phonation_ratio", "syllable_count_est",
            "speech_rate_syl_per_s", "articulation_rate_syl_per_s",
            "voiced_segment_count", "mean_utterance_length_s",
        ]}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Intensity metrics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _intensity_metrics(self, intensity) -> dict:
        vals = intensity.values[0]
        valid = vals[vals > 0]  # ignore zero/negative dB
        if len(valid) == 0:
            return {"intensity_mean_db": 0, "intensity_std_db": 0, "intensity_range_db": 0}
        return {
            "intensity_mean_db": round(float(np.mean(valid)), 2),
            "intensity_std_db": round(float(np.std(valid)), 2),
            "intensity_range_db": round(float(np.max(valid) - np.min(valid)), 2),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. Spectral Features (librosa)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _spectral_metrics(self, y: np.ndarray, sr: int) -> dict:
        y = y.astype(np.float32)

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                     n_fft=SPECTRAL_N_FFT,
                                                     hop_length=SPECTRAL_HOP_LENGTH)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                        n_fft=SPECTRAL_N_FFT,
                                                        hop_length=SPECTRAL_HOP_LENGTH)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                    n_fft=SPECTRAL_N_FFT,
                                                    hop_length=SPECTRAL_HOP_LENGTH)[0]
        flatness = librosa.feature.spectral_flatness(y=y,
                                                      n_fft=SPECTRAL_N_FFT,
                                                      hop_length=SPECTRAL_HOP_LENGTH)[0]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                      n_fft=SPECTRAL_N_FFT,
                                                      hop_length=SPECTRAL_HOP_LENGTH)

        # Spectral flux (onset strength envelope)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                                  n_fft=SPECTRAL_N_FFT,
                                                  hop_length=SPECTRAL_HOP_LENGTH)

        # Spectral tilt (slope of log-power spectrum)
        S = np.abs(librosa.stft(y, n_fft=SPECTRAL_N_FFT, hop_length=SPECTRAL_HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=SPECTRAL_N_FFT)
        log_S = np.mean(librosa.power_to_db(S ** 2), axis=1)
        if len(freqs) > 1 and len(log_S) > 1:
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(freqs[1:], log_S[1:])  # skip DC
            spectral_tilt = float(slope)
        else:
            spectral_tilt = 0.0

        return {
            "spectral_centroid_mean": round(float(np.mean(centroid)), 1),
            "spectral_centroid_std": round(float(np.std(centroid)), 1),
            "spectral_bandwidth_mean": round(float(np.mean(bandwidth)), 1),
            "spectral_bandwidth_std": round(float(np.std(bandwidth)), 1),
            "spectral_rolloff_mean": round(float(np.mean(rolloff)), 1),
            "spectral_rolloff_std": round(float(np.std(rolloff)), 1),
            "spectral_flatness_mean": round(float(np.mean(flatness)), 6),
            "spectral_flatness_std": round(float(np.std(flatness)), 6),
            "spectral_contrast_mean": round(float(np.mean(contrast)), 2),
            "spectral_flux_mean": round(float(np.mean(onset_env)), 4),
            "spectral_flux_std": round(float(np.std(onset_env)), 4),
            "spectral_tilt": round(spectral_tilt, 6),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 9. MFCCs + Deltas
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _mfcc_metrics(self, y: np.ndarray, sr: int) -> dict:
        y = y.astype(np.float32)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=SPECTRAL_N_MFCC,
                                      n_fft=SPECTRAL_N_FFT,
                                      hop_length=SPECTRAL_HOP_LENGTH,
                                      n_mels=SPECTRAL_N_MELS)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        results = {}
        for i in range(SPECTRAL_N_MFCC):
            results[f"mfcc_{i}_mean"] = round(float(np.mean(mfccs[i])), 4)
            results[f"mfcc_{i}_std"] = round(float(np.std(mfccs[i])), 4)
            results[f"mfcc_delta_{i}_mean"] = round(float(np.mean(mfcc_delta[i])), 4)
            results[f"mfcc_delta_{i}_std"] = round(float(np.std(mfcc_delta[i])), 4)
            results[f"mfcc_delta2_{i}_mean"] = round(float(np.mean(mfcc_delta2[i])), 4)
            results[f"mfcc_delta2_{i}_std"] = round(float(np.std(mfcc_delta2[i])), 4)

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 10. Energy Features
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _energy_metrics(self, y: np.ndarray, sr: int) -> dict:
        y = y.astype(np.float32)
        rms = librosa.feature.rms(y=y, frame_length=SPECTRAL_N_FFT,
                                   hop_length=SPECTRAL_HOP_LENGTH)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=SPECTRAL_N_FFT,
                                                   hop_length=SPECTRAL_HOP_LENGTH)[0]

        # Energy entropy: split signal into short frames, compute entropy of energy distribution
        frame_length = SPECTRAL_N_FFT
        hop = SPECTRAL_HOP_LENGTH
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop)
        frame_energies = np.sum(frames ** 2, axis=0)
        total_energy = np.sum(frame_energies) + 1e-10
        energy_probs = frame_energies / total_energy
        energy_probs = energy_probs[energy_probs > 0]
        energy_entropy = float(-np.sum(energy_probs * np.log2(energy_probs)))

        return {
            "rms_mean": round(float(np.mean(rms)), 6),
            "rms_std": round(float(np.std(rms)), 6),
            "rms_max": round(float(np.max(rms)), 6),
            "zcr_mean": round(float(np.mean(zcr)), 6),
            "zcr_std": round(float(np.std(zcr)), 6),
            "energy_entropy": round(energy_entropy, 4),
        }
