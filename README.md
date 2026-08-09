# Lucid: Passive Multimodal Cognitive Decline Monitoring System

## Project Overview

Lucid is a multimodal behavioral monitoring system designed to identify patterns that may be associated with cognitive decline. The system combines eye tracking, speech analysis, and mouse interaction monitoring to collect behavioral data during computer-based activities.

By analyzing these modalities together across multiple sessions, Lucid aims to detect changes from an individual's established behavioral baseline. The project focuses on passive and non-invasive monitoring, reducing dependence on periodic assessments performed only in controlled environments.

---

## Problem Statement

Cognitive decline can influence several aspects of everyday behavior, including visual attention, eye movements, speech characteristics, and motor interactions. Conventional cognitive assessments are generally performed at specific intervals and may not capture subtle changes occurring during normal daily activities.

A system capable of continuously collecting behavioral indicators and comparing them with an individual's historical baseline can provide a more longitudinal view of potential cognitive changes.

---

## Proposed Solution

Lucid provides a unified pipeline for collecting multimodal behavioral data, extracting relevant features, storing session information, and analyzing changes over time.

The system supports both passive monitoring and dedicated active eye-movement assessments. The collected measurements are used to establish individual baselines and calculate longitudinal metric changes.

---

## Eye-Tracking Module

The eye-tracking module uses a standard webcam along with MediaPipe FaceLandmarker and OpenCV. Facial and iris landmarks are processed to estimate gaze behavior and extract features such as:

- Fixations
- Saccades
- Blink rate
- Pupil measurements
- Gaze-path characteristics
- Gaze entropy

The system also classifies gaze behavior into activities such as:

- Reading
- Scanning
- Focused activity
- Passive activity
- Idle states

Active oculomotor assessments including:

- Prosaccade task
- Antisaccade task
- Smooth-pursuit task

are also implemented.

---

## Speech-Analysis Module

The speech module analyzes recorded or passively collected speech segments using:

- Parselmouth/Praat
- Librosa
- NumPy
- SciPy

It extracts acoustic and temporal features including:

- Pitch
- Jitter
- Shimmer
- Harmonics-to-noise ratio
- Intensity
- Speech rate
- Pauses
- Spectral characteristics
- MFCC-based features

These measurements provide quantitative information about speech and vocal behavior.

---

## Mouse-Tracking Module

The mouse module passively records cursor movements using Pynput.

It analyzes:

- Movement duration
- Movement distance
- Movement curvature
- Idle time
- Other movement characteristics

These features help capture behavioral and motor interaction patterns during computer usage.

---

## Project Architecture

```text
                         Lucid System

Webcam ───────> Eye Tracking ────┐
                                 │
Microphone ───> Speech Analysis ─┼──> Feature Extraction
                                 │
Mouse ────────> Mouse Tracking ──┘
                                      |
                                      v
                              Session Management
                                      |
                                      v
                            Baseline & Drift Analysis
                                      |
                                      v
                              Dashboard & Reports
```

---

## Technologies Used

### Programming Language

- Python

### Computer Vision and Eye Tracking

- MediaPipe
- OpenCV

### Speech Processing

- Parselmouth/Praat
- Librosa
- NumPy
- SciPy

### Data Processing and Visualization

- Matplotlib
- Scikit-learn
- JSON

### Input Monitoring

- Pynput

---

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd Lucid-cognitive-decline-main
```

Create a virtual environment:

```bash
python3.13 -m venv .venv
```

Activate the environment:

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The system requires access to:

- Webcam for eye tracking
- Microphone for speech analysis
- Mouse input for movement tracking

---

## How to Run

Start passive multimodal monitoring:

```bash
python main.py passive
```

Other available modes:

### Eye Tracking

```bash
python main.py eye
```

### Speech Analysis

```bash
python main.py speech
```

### Eye Movement Tasks

```bash
python main.py task prosaccade

python main.py task antisaccade

python main.py task pursuit
```

### Complete Assessment

```bash
python main.py full
```

### Dashboard Generation

```bash
python main.py dashboard
```

### Cognitive Drift Analysis

```bash
python main.py drift
```

---

## Outputs and Results

Lucid stores session data and extracted behavioral metrics in JSON format.

The system generates:

- Activity classifications
- Session summaries
- Behavioral metrics
- Baseline statistics
- Longitudinal metric trends
- Cognitive drift scores
- Visualization dashboards

The generated outputs help analyze behavioral changes over time.

These outputs are intended for research and behavioral monitoring purposes and should not be considered a clinical diagnosis.

---

## Team Members

This project was developed as a 2nd semester Experiential Learning project.

- Rishi Rajesh 
- Pranjal Mishra
- Hardik Vats
- Revanasidda Malappa Pujari
- Krishna Ramasimha
