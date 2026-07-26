"""Central tunables and role configuration shared across hand_controller.py,
capture_gestures.py, train_gesture_model.py, and mouse_control.py."""

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Resolution the camera frame is downscaled to before MediaPipe inference
# (the on-screen preview and overlay stay full-resolution -- landmark
# coordinates are normalized 0-1 regardless of input size). Set to None to
# disable downscaling.
DETECTION_SIZE = (480, 360)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# --- Handedness / role mapping -------------------------------------------
# Raw handedness labels as reported by MediaPipe ("Left" / "Right").
# Run hand_controller.py, check the overlay for the label printed next to
# each of your hands, then set these two to match which hand you want doing
# keyboard gestures vs mouse control.
KEYBOARD_HAND_LABEL = "Right"
MOUSE_HAND_LABEL = "Left"

# --- Legacy threshold-based gesture detection (handcontroller/legacy_gestures.py) ---
THRESHOLD = 0.02  # raise if false positives, lower if gestures won't trigger
THUMB_THRESHOLD = 0.06

# Frames a gesture must be seen before its key presses (ON) / absent before
# it releases (OFF). Asymmetric on purpose: a fast press feels responsive
# and flicker there is rare, while flicker on release is what's actually
# noticeable, so it keeps a small guard. REQUIRED_FRAMES_ON=1 is safest with
# a trained classifier (which already gates on CLASSIFIER_CONFIDENCE_THRESHOLD);
# raise it if the noisier legacy_gestures.py fallback path feels twitchy.
REQUIRED_FRAMES_ON = 1
REQUIRED_FRAMES_OFF = 2

# --- Learned gesture classifier -------------------------------------------
KEYBOARD_GESTURE_LABELS = [
    "none",
    "fist",
    "open_hand",
    "pointing",
    "peace_sign",
    "three",
    "four",
    "thumbs_up",
    "finger_gun",
    "ok_sign",
    "middle_finger",
]
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7

# Mouse hand currently uses geometric checks (gesture_features.py), not a
# trained classifier -- this list only exists so capture_gestures.py can
# optionally build a dataset for a future learned mouse-hand model.
MOUSE_GESTURE_LABELS = ["none", "open_hand", "fist"]

GESTURE_LABELS_BY_ROLE = {
    "keyboard": KEYBOARD_GESTURE_LABELS,
    "mouse": MOUSE_GESTURE_LABELS,
}

# --- Mouse control ---------------------------------------------------------
# Used only as a fallback when no calibration (see calibrate_mouse.py) has
# been saved yet -- once calibrated, per-axis sensitivity is derived from
# your own recorded comfortable range instead of this fixed multiplier.
MOUSE_SENSITIVITY = 4.0  # multiplies normalized hand-delta into screen pixels

# Gain applied on top of the calibrated mapping (see calibrate_mouse.py),
# separate per axis since vertical reach tends to fall short more than
# horizontal (shorter/narrower vertical FOV on most webcams -> less
# physical room to move before losing tracking near the top/bottom edge).
# 1.0 = your exact recorded range maps to exactly the full screen; raising
# it lets you reach the screen edge before physically reaching your
# calibrated extreme. Tune independently: raise Y if top/bottom still falls
# short without touching how X already feels.
MOUSE_CALIBRATED_GAIN_X = 1.5
MOUSE_CALIBRATED_GAIN_Y = 2.0

MOUSE_ACCEL_GAIN = 0.0  # 0 = linear movement; >0 adds speed-based acceleration
PINCH_CLOSE_RATIO = 0.35  # pinch_ratio below this counts as "pinched"
PINCH_OPEN_RATIO = 0.5  # pinch_ratio must rise above this to count as "released"
SCROLL_SENSITIVITY = 15.0

# Minimum acceptable left-right / top-bottom spread (normalized units) for a
# calibrate_mouse.py run to be considered valid, rather than saving a
# near-zero range that would make the cursor unusably twitchy.
MIN_CALIBRATION_RANGE = 0.05
