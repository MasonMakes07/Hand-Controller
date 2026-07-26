"""Calibrate the mouse hand's comfortable movement range so the cursor can
reach every edge of the screen (see mouse_control.py / mouse_calibration.py).

Usage:
    python calibrate_mouse.py

Controls:
    SPACE  - record the current hand position for this stage, advance
    r      - go back a stage and redo it
    q      - quit without saving
"""

import threading

import cv2 as cv
import mediapipe as mp

import camera
import config
import mouse_calibration
import tracker
from mouse_control import REF_LANDMARK

STAGES = ["left", "right", "top", "bottom"]
PROMPTS = {
    "left": "Move your mouse hand to the LEFTMOST comfortable position",
    "right": "Move your mouse hand to the RIGHTMOST comfortable position",
    "top": "Move your mouse hand to the TOPMOST comfortable position",
    "bottom": "Move your mouse hand to the BOTTOMMOST comfortable position",
}
AXIS = {"left": "x", "right": "x", "top": "y", "bottom": "y"}

latest_landmarks = None
landmarks_lock = threading.Lock()


def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    with landmarks_lock:
        latest_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None


def main():
    landmarker = tracker.create_landmarker(on_result, num_hands=1)
    cam = camera.CameraStream(0).start()
    if not cam.isOpened():
        cam.stop()
        cam = camera.CameraStream(0, backend=cv.CAP_ANY).start()
    if not cam.isOpened():
        print("Unable to access camera")
        return

    recorded = {}
    stage_idx = 0
    timestamp = 0

    print("Press SPACE to record each position, 'r' to redo the previous stage, 'q' to quit.")

    try:
        while cam.isOpened() and stage_idx < len(STAGES):
            success, image = cam.read()
            if not success:
                break
            image = cv.flip(image, 1)
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            timestamp += 1
            landmarker.detect_async(mp_image, timestamp)

            with landmarks_lock:
                landmarks = latest_landmarks

            stage = STAGES[stage_idx]
            cv.putText(image, PROMPTS[stage], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(image, "then press SPACE", (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            hand_status = "hand detected" if landmarks is not None else "no hand detected"
            cv.putText(image, hand_status, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0) if landmarks is not None else (0, 0, 255), 2)

            cv.imshow("Calibrate Mouse Range", image)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Calibration cancelled.")
                return
            elif key == ord("r"):
                stage_idx = max(0, stage_idx - 1)
            elif key == ord(" "):
                if landmarks is None:
                    continue
                axis = AXIS[stage]
                value = landmarks[REF_LANDMARK].x if axis == "x" else landmarks[REF_LANDMARK].y
                recorded[stage] = value
                stage_idx += 1
    finally:
        cam.stop()
        cv.destroyAllWindows()
        landmarker.close()

    if len(recorded) < len(STAGES):
        print("Calibration cancelled.")
        return

    min_x, max_x = sorted((recorded["left"], recorded["right"]))
    min_y, max_y = sorted((recorded["top"], recorded["bottom"]))

    if (max_x - min_x) < config.MIN_CALIBRATION_RANGE or (max_y - min_y) < config.MIN_CALIBRATION_RANGE:
        print("Recorded range is too small to be usable -- move your hand further between "
              "stages and run calibration again.")
        return

    mouse_calibration.save_calibration({"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y})
    print(f"Saved calibration: x range {max_x - min_x:.3f}, y range {max_y - min_y:.3f}")
    print(f"-> {mouse_calibration.CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
