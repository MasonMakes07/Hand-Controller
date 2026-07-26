import argparse
import collections
import os
import threading
import time

import cv2 as cv
import joblib
import mediapipe as mp

import camera
import config
import gesture_features
import legacy_gestures
import tracker
from debounce import Debouncer
from KeyBinds import InputController
from mouse_control import MouseController

KEYBOARD_MODEL_PATH = os.path.join(config.MODELS_DIR, "gesture_classifier_keyboard.joblib")

latest_hands = {"Left": None, "Right": None}
hands_lock = threading.Lock()

# Timing instrumentation (only meaningfully populated when --debug-timing is
# passed, but cheap enough to always maintain). Kept on a separate lock from
# hands_lock so measuring latency never adds contention to that hot path.
dispatch_times = {}
timing_lock = threading.Lock()
mp_latency_samples = collections.deque(maxlen=30)


def on_result(result, output_image, timestamp_ms):
    global latest_hands
    with hands_lock:
        latest_hands = {"Left": None, "Right": None}
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            latest_hands[label] = landmarks

    with timing_lock:
        sent_at = dispatch_times.pop(timestamp_ms, None)
    if sent_at is not None:
        mp_latency_samples.append(time.perf_counter() - sent_at)


def load_classifier():
    if not os.path.exists(KEYBOARD_MODEL_PATH):
        return None
    bundle = joblib.load(KEYBOARD_MODEL_PATH)
    return bundle["model"]


def classify_keyboard_gesture(model, landmarks):
    features = gesture_features.extract_features(landmarks)
    probs = model.predict_proba([features])[0]
    top_idx = probs.argmax()
    top_label = model.classes_[top_idx]
    confidence = probs[top_idx]
    if top_label == "none" or confidence < config.CLASSIFIER_CONFIDENCE_THRESHOLD:
        return set()
    return {top_label}


# Gestures that share the same finger pattern and differ only by thumb
# position can both fire on the same frame due to the thumb_in bug in
# legacy_gestures.py (see its docstring). Rather than patch that frozen
# module, ambiguous pairs are resolved here: the key (winner) suppresses
# the values (losers) whenever both are active at once.
GESTURE_PRIORITY_OVERRIDES = {
    "finger_gun": {"pointing"},
}


def resolve_gesture_conflicts(active: set) -> set:
    resolved = set(active)
    for winner, losers in GESTURE_PRIORITY_OVERRIDES.items():
        if winner in resolved:
            resolved -= losers
    return resolved


def draw_landmarks(image, landmarks, label):
    h, w = image.shape[:2]
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in gesture_features.FINGERTIP_INDICES:
            color = (0, 0, 255)  # red
        elif idx in gesture_features.KNUCKLE_INDICES:
            color = (0, 255, 255)  # yellow
        else:
            color = (255, 255, 255)  # white
        cv.circle(image, (cx, cy), 6, color, -1)
    wrist = landmarks[0]
    cv.putText(image, label, (int(wrist.x * w), int(wrist.y * h) + 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true",
                         help="Force legacy threshold-based gesture detection even if a trained model exists")
    parser.add_argument("--debug-timing", action="store_true",
                         help="Show FPS and MediaPipe inference latency on-screen and in the console")
    args = parser.parse_args()

    print("Starting hand controller")

    keybinds = InputController()
    mouse_ctrl = MouseController()
    if mouse_ctrl.bounds is not None:
        print("Using calibrated mouse range.")
    else:
        print("No mouse calibration found -- using default MOUSE_SENSITIVITY. "
              "Run calibrate_mouse.py if the cursor can't reach every edge of the screen.")

    model = None if args.legacy else load_classifier()
    if model is not None:
        print("Using trained gesture classifier for keyboard hand.")
        gesture_names = [name for name in config.KEYBOARD_GESTURE_LABELS if name != "none"]
    else:
        print("No trained gesture model found (or --legacy passed) -- "
              "using legacy threshold detection. Run capture_gestures.py and "
              "train_gesture_model.py to switch to the classifier.")
        gesture_names = list(legacy_gestures.GESTURE_CHECKS.keys())

    debouncer = Debouncer(gesture_names, config.REQUIRED_FRAMES_ON, config.REQUIRED_FRAMES_OFF)

    landmarker = tracker.create_landmarker(on_result, num_hands=2)

    cam = camera.CameraStream(0).start()
    if not cam.isOpened():
        cam.stop()
        print("CAP_DSHOW failed to open the camera, falling back to CAP_ANY.")
        cam = camera.CameraStream(0, backend=cv.CAP_ANY).start()
    if not cam.isOpened():
        print("Unable to access camera")
        return

    timestamp = 0
    frame_times = collections.deque(maxlen=30)
    last_console_print = time.perf_counter()
    print("Press 'p' to quit.")

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            break
        frame_times.append(time.perf_counter())

        image = cv.flip(image, 1)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        detect_input = (cv.resize(rgb_image, config.DETECTION_SIZE, interpolation=cv.INTER_LINEAR)
                         if config.DETECTION_SIZE else rgb_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_input)
        timestamp += 1
        with timing_lock:
            if len(dispatch_times) > 100:
                dispatch_times.clear()  # a result went missing somewhere; don't leak forever
            dispatch_times[timestamp] = time.perf_counter()
        landmarker.detect_async(mp_image, timestamp)

        with hands_lock:
            keyboard_lm = latest_hands.get(config.KEYBOARD_HAND_LABEL)
            mouse_lm = latest_hands.get(config.MOUSE_HAND_LABEL)
            snapshot = dict(latest_hands)

        if keyboard_lm is not None:
            if model is not None:
                active = classify_keyboard_gesture(model, keyboard_lm)
            else:
                active = {name for name, check in legacy_gestures.GESTURE_CHECKS.items() if check(keyboard_lm)}
        else:
            active = set()

        active = resolve_gesture_conflicts(active)
        latched = debouncer.update(active)
        for name in gesture_names:
            keybinds.set_gestures(name, latched[name])
        if keyboard_lm is None:
            keybinds.release_all()

        mouse_info = mouse_ctrl.update(mouse_lm)

        for label, landmarks in snapshot.items():
            if landmarks is not None:
                draw_landmarks(image, landmarks, label)

        active_names = [name for name, is_active in latched.items() if is_active]
        cv.putText(image, f"Keyboard ({config.KEYBOARD_HAND_LABEL}): {', '.join(active_names) or 'none'}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f"Mouse ({config.MOUSE_HAND_LABEL}): {mouse_info['state']}",
                   (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if "pinch_left_ratio" in mouse_info:
            cv.putText(image,
                       f"pinch L/R: {mouse_info['pinch_left_ratio']:.2f} / {mouse_info['pinch_right_ratio']:.2f}"
                       f"  (close<{config.PINCH_CLOSE_RATIO} open>{config.PINCH_OPEN_RATIO})",
                       (10, 78), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        if args.debug_timing:
            now = time.perf_counter()
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) >= 2 else 0.0
            mp_latency_ms = (sum(mp_latency_samples) / len(mp_latency_samples) * 1000) if mp_latency_samples else 0.0
            timing_text = f"FPS: {fps:.1f}  MP latency: {mp_latency_ms:.0f}ms"
            cv.putText(image, timing_text, (10, 101), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            if now - last_console_print >= 1.0:
                print(timing_text)
                last_console_print = now

        cv.imshow("Hand Controller", image)
        if cv.waitKey(1) & 0xFF == ord("p"):
            break

    cam.stop()
    keybinds.release_all()
    mouse_ctrl.release_all()
    cv.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
