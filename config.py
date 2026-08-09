"""
Lucid Configuration
All thresholds, landmark indices, and tuneable parameters in one place.
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MediaPipe FaceLandmarker Model (Tasks API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACE_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MediaPipe FaceLandmarker Landmark Indices (478 total)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Iris landmarks (indices 468–477 in the 478-landmark model)
LEFT_IRIS_CENTER = 468
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corner landmarks
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Eye contour landmarks for EAR (Eye Aspect Ratio) blink detection
# Vertical pairs + horizontal pair
LEFT_EYE_EAR = {
    "p1": 33,   # outer corner
    "p2": 160,  # upper lid (left of center)
    "p3": 158,  # upper lid (right of center)
    "p4": 133,  # inner corner
    "p5": 153,  # lower lid (right of center)
    "p6": 145,  # lower lid (left of center)
}
RIGHT_EYE_EAR = {
    "p1": 263,  # outer corner
    "p2": 387,  # upper lid
    "p3": 385,  # upper lid
    "p4": 362,  # inner corner
    "p5": 380,  # lower lid
    "p6": 374,  # lower lid
}

# Nose tip for head pose reference
NOSE_TIP = 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Eye Tracking Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Approximate angular conversion: pixels → degrees
# Assumes ~60cm viewing distance, 640px across ~50° horizontal FOV
PIXELS_PER_DEGREE = 640 / 50  # ~12.8 px/deg

# Saccade detection (velocity-threshold algorithm)
SACCADE_VELOCITY_THRESHOLD = 30.0   # deg/s — onset threshold
SACCADE_MIN_DURATION_MS = 10        # ms — reject noise spikes
SACCADE_MAX_DURATION_MS = 500       # ms — reject tracking loss
SACCADE_MIN_AMPLITUDE_DEG = 1.0     # deg — reject microsaccades (optional)

# Fixation detection (dispersion-threshold, I-DT)
FIXATION_DISPERSION_THRESHOLD = 2.0  # degrees — max spread to count as fixation
FIXATION_MIN_DURATION_MS = 100       # ms — minimum fixation duration

# Blink detection
BLINK_EAR_THRESHOLD = 0.2     # EAR below this = blink
BLINK_MIN_DURATION_MS = 50    # ms — minimum blink duration
BLINK_MAX_DURATION_MS = 500   # ms — maximum (longer = not a blink)

# Gaze buffer
GAZE_BUFFER_SECONDS = 60      # rolling window for live metrics
GAZE_HISTORY_MAX = 10000      # max samples to retain in memory

# Smooth pursuit task
PURSUIT_TARGET_FREQ_HZ = 0.4  # target oscillation frequency
PURSUIT_AMPLITUDE_DEG = 15.0  # target amplitude in degrees

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prosaccade / Antisaccade Task
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK_NUM_TRIALS = 20
TASK_FIXATION_DURATION_MS = (1000, 2000)  # random uniform range
TASK_GAP_DURATION_MS = 200                # gap between fixation offset and target onset
TASK_TARGET_DURATION_MS = 1500            # how long target stays on
TASK_ITI_MS = 500                         # inter-trial interval
TASK_TARGET_ECCENTRICITIES_DEG = [5, 10, 15]  # degrees from center
TASK_EXPRESS_SACCADE_THRESHOLD_MS = 120   # <120ms = express saccade
TASK_ANTICIPATORY_THRESHOLD_MS = 80       # <80ms = anticipatory (invalid)

# Target positions (normalized screen coordinates, center = 0.5, 0.5)
TASK_TARGET_POSITIONS = [
    (0.15, 0.5), (0.85, 0.5),   # horizontal
    (0.5, 0.15), (0.5, 0.85),   # vertical
    (0.2, 0.2), (0.8, 0.2),     # diagonal
    (0.2, 0.8), (0.8, 0.8),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Speech Analysis Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Recording
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_RECORD_SECONDS_DEFAULT = 30

# Pitch extraction
PITCH_FLOOR_HZ = 75
PITCH_CEILING_HZ = 500
PITCH_TIME_STEP = 0.01  # 10ms frames

# Formant extraction
FORMANT_MAX_FREQUENCY = 5500  # Hz — for male voices; 5500 for female
FORMANT_NUM_FORMANTS = 5
FORMANT_WINDOW_LENGTH = 0.025  # 25ms

# Intensity
INTENSITY_MIN_PITCH = 100  # Hz

# Pause detection
PAUSE_INTENSITY_THRESHOLD_DB = -35  # relative to peak; below = silence
PAUSE_MIN_DURATION_S = 0.2          # 200ms minimum pause
PAUSE_LONG_THRESHOLD_S = 1.0        # >1s = long pause

# Spectral analysis (librosa)
SPECTRAL_N_FFT = 2048
SPECTRAL_HOP_LENGTH = 512
SPECTRAL_N_MFCC = 13
SPECTRAL_N_MELS = 128

# Passive speech capture (background mic monitoring)
SPEECH_VAD_ENERGY_THRESHOLD = 0.01     # RMS threshold for voice activity
SPEECH_VAD_MIN_SPEECH_S = 2.0          # minimum voiced segment to analyze
SPEECH_VAD_MAX_SILENCE_S = 1.5         # silence gap to split segments
SPEECH_VAD_FRAME_S = 0.03              # 30ms VAD frames
SPEECH_CAPTURE_BUFFER_S = 120          # rolling buffer: 2 minutes
SPEECH_ANALYSIS_MIN_S = 5.0            # minimum segment length for full analysis
SPEECH_ANALYSIS_MAX_S = 60.0           # cap segment length (memory)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Passive Monitoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Activity classification windows
ACTIVITY_WINDOW_S = 5.0       # seconds of gaze data per classification window
ACTIVITY_STRIDE_S = 2.5       # overlap between windows (50%)
ACTIVITY_MIN_SAMPLES = 30     # minimum gaze samples to classify a window

# Reading detection thresholds
READING_HORIZONTAL_RATIO = 0.6     # >60% of saccades must be horizontal (±30°)
READING_RETURN_SWEEP_RATIO = 0.05  # >5% of saccades should be large leftward returns
READING_SACCADE_AMP_MAX = 12.0     # degrees — reading saccades are typically <8°
READING_FIX_DUR_RANGE = (100, 600) # ms — reading fixations 100–600ms

# Scanning detection
SCANNING_ENTROPY_THRESHOLD = 3.5   # spatial entropy > this = scanning
SCANNING_SACCADE_AMP_MIN = 3.0     # degrees — scanning has larger jumps

# Focused work detection
FOCUSED_DISPERSION_MAX = 8.0       # degrees — gaze stays in tight region
FOCUSED_FIX_DUR_MIN = 250          # ms — longer fixations = deeper processing

# Microsaccade detection (during fixations)
MICROSACCADE_VEL_THRESHOLD = 8.0   # deg/s — below saccade threshold
MICROSACCADE_AMP_MAX = 1.0         # degrees
MICROSACCADE_MIN_DURATION_MS = 6   # ms

# Square-wave jerk detection
SWJ_AMPLITUDE_RANGE = (0.5, 5.0)   # degrees
SWJ_INTERSACCADIC_INTERVAL = (200, 400)  # ms between the two saccade components

# Pupil cognitive load analysis
PUPIL_BASELINE_PERCENTILE = 20     # low-load baseline = 20th percentile of diameter
PUPIL_SMOOTHING_WINDOW = 15        # samples for moving average

# Session persistence
SESSION_DIR = "sessions"
PASSIVE_LOG_DIR = os.path.join(SESSION_DIR, "passive")
DAILY_SUMMARY_DIR = os.path.join(SESSION_DIR, "daily")
DASHBOARD_DPI = 150