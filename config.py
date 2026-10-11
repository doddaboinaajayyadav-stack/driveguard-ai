"""
config.py — DriveGuard AI Central Configuration
================================================
All tunable parameters are defined here.
Edit this file to adjust thresholds, paths, and server settings
without touching any other source file.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "eye_state_cnn.pth")
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR       = os.path.join(DATASET_DIR, "train")
VAL_DIR         = os.path.join(DATASET_DIR, "val")
ALARM_SOUND     = os.path.join(BASE_DIR, "static", "sounds", "alarm.wav")

# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------
IMG_SIZE        = 24          # Resize eye crops to IMG_SIZE × IMG_SIZE pixels
NUM_CLASSES     = 2           # 0 = open, 1 = closed
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 1e-3
DEVICE          = "cpu"       # "cpu" is the safe default on ARM64 Windows

# ---------------------------------------------------------------------------
# Drowsiness Detection Thresholds
# ---------------------------------------------------------------------------
# Number of consecutive frames both eyes must be CLOSED to trigger an alert
CLOSED_FRAMES_THRESHOLD = 20   # At 20 FPS ≈ 1 second of closure

# Confidence below this value is treated as UNCERTAIN (not OPEN or CLOSED)
CONFIDENCE_THRESHOLD = 0.70

# Severity escalation: ALERT after N frames, WARNING after 2N, DANGER after 3N
ALERT_FRAMES    = 20
WARNING_FRAMES  = 40
DANGER_FRAMES   = 60

# Rolling history window for the eye-closure frequency graph (seconds)
HISTORY_WINDOW_SECONDS = 60

# ---------------------------------------------------------------------------
# OpenCV Face / Eye Detection (Haar Cascade)
# ---------------------------------------------------------------------------
# These cascade classifiers ship with every OpenCV installation
FACE_SCALE_FACTOR   = 1.3
FACE_MIN_NEIGHBORS  = 5
FACE_MIN_SIZE       = (80, 80)

EYE_SCALE_FACTOR    = 1.1
EYE_MIN_NEIGHBORS   = 10
EYE_MIN_SIZE        = (20, 20)

# ---------------------------------------------------------------------------
# Flask / Web Server
# ---------------------------------------------------------------------------
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
FLASK_DEBUG     = False        # Set True only during development

# MJPEG stream quality (1–95; higher = better quality, more bandwidth)
STREAM_JPEG_QUALITY = 70

# How often (ms) the JS dashboard polls for metrics
DASHBOARD_POLL_MS = 500

# ---------------------------------------------------------------------------
# Alarm
# ---------------------------------------------------------------------------
ALARM_VOLUME    = 0.8          # 0.0 – 1.0
ALARM_COOLDOWN  = 5            # Seconds between repeated alarms
