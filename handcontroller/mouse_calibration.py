"""Load/save the user's calibrated mouse-hand range (see calibrate_mouse.py).

Stored in handcontroller/data/ -- personal to the user's camera/setup, same
reasoning as the gitignored gesture datasets."""

import json
import os

import config

CALIBRATION_PATH = os.path.join(config.DATA_DIR, "mouse_calibration.json")


def load_calibration():
    if not os.path.exists(CALIBRATION_PATH):
        return None
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def save_calibration(bounds: dict):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(bounds, f, indent=2)
