"""Central tunables and role configuration shared across hand_controller.py,
capture_gestures.py, train_gesture_model.py, and mouse_control.py."""

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

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
REQUIRED_FRAMES = 2  # frames a gesture must be held before triggering / releasing

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
MOUSE_SENSITIVITY = 4.0  # multiplies normalized hand-delta into screen pixels
MOUSE_ACCEL_GAIN = 0.0  # 0 = linear movement; >0 adds speed-based acceleration
PINCH_CLOSE_RATIO = 0.35  # pinch_ratio below this counts as "pinched"
PINCH_OPEN_RATIO = 0.5  # pinch_ratio must rise above this to count as "released"
SCROLL_SENSITIVITY = 15.0
