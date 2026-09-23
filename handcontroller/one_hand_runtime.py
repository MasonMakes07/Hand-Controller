"""Separate webcam/input lifecycle for the optional one-hand mode."""

import queue
import threading
import time

import cv2 as cv
import mediapipe as mp
from pynput.keyboard import Key, Listener

import camera
import config
import legacy_gestures
import tracker
from debounce import Debouncer
from KeyBinds import InputController
from one_hand_control import OneHandController


def run(args, load_classifier, classify, draw_landmarks, resolve):
    controls = OneHandController(
        config.ONE_HAND_DEADZONE, config.ONE_HAND_RELEASE_RATIO,
        config.ONE_HAND_CALIBRATION_SECONDS, config.ONE_HAND_CALIBRATION_TOLERANCE,
        config.ONE_HAND_ACTION_GESTURES,
    )
    output = InputController(config.ONE_HAND_BINDINGS)
    model = None if args.legacy else load_classifier()
    gestures = list(config.ONE_HAND_ACTION_GESTURES.values())
    debounce = Debouncer(gestures, config.REQUIRED_FRAMES_ON, config.REQUIRED_FRAMES_OFF)
    events = queue.SimpleQueue()
    lock = threading.Lock()
    result_state = {"landmarks": None, "time": 0.0, "sequence": 0}

    def on_result(result, image, timestamp):
        selected = next((lm for lm, hand in zip(result.hand_landmarks, result.handedness)
                         if hand[0].category_name == args.hand), None)
        with lock:
            result_state.update(landmarks=selected, time=time.monotonic(), sequence=timestamp)

    # Release events avoid keyboard autorepeat repeatedly toggling pause.
    def on_release(key):
        if key in (Key.f8, Key.f9, Key.f10):
            events.put(key)

    cam = landmarker = listener = None
    try:
        landmarker = tracker.create_landmarker(on_result, num_hands=2)
        cam = camera.CameraStream(0).start()
        if not cam.isOpened():
            cam.stop()
            cam = camera.CameraStream(0, backend=cv.CAP_ANY).start()
        if not cam.isOpened():
            print("Unable to access camera")
            return
        listener = Listener(on_release=on_release)
        listener.start()
        print(f"One-hand mode ({args.hand}): hold still for calibration. "
              "F8 pause/resume, F9 recenter, F10 quit (global hotkeys).")
        sequence = processed = 0
        active_gestures = set()
        last_frame = time.monotonic()
        fps = 0.0
        while cam.isOpened():
            while not events.empty():
                event = events.get()
                if event == Key.f10:
                    return
                output.release_all()
                debounce.reset()
                active_gestures = set()
                if event == Key.f8:
                    controls.toggle_pause()
                else:
                    controls.reset()
            success, image = cam.read(timeout=0.1)
            if not success:
                output.release_all()
                debounce.reset()
                controls.update(None, set(), time.monotonic())
                active_gestures = set()
                continue
            now = time.monotonic()
            fps = 1.0 / max(now - last_frame, 1e-6)
            last_frame = now
            image = cv.flip(image, 1)
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            if config.ONE_HAND_DETECTION_SIZE:
                rgb = cv.resize(rgb, config.ONE_HAND_DETECTION_SIZE, interpolation=cv.INTER_LINEAR)
            sequence += 1
            landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), sequence)
            with lock:
                snapshot = dict(result_state)
            lm = snapshot["landmarks"] if now - snapshot["time"] <= config.ONE_HAND_TRACKING_TIMEOUT else None
            if lm is None or controls.paused:
                debounce.reset()
                active_gestures = set()
            elif snapshot["sequence"] != processed:
                detected = classify(model, lm) if model is not None else {
                    name for name, check in legacy_gestures.GESTURE_CHECKS.items() if check(lm)
                }
                latched = debounce.update(resolve(detected))
                active_gestures = {name for name, held in latched.items() if held}
            processed = snapshot["sequence"]
            point = (lm[9].x, lm[9].y) if lm is not None else None
            active, state = controls.update(point, active_gestures, now)
            output.set_active(active)
            if lm is not None:
                draw_landmarks(image, lm, args.hand)
            if controls.center is not None:
                h, w = image.shape[:2]
                cx, cy = controls.center
                dz = controls.deadzone
                cv.rectangle(image, (int((cx-dz)*w), int((cy-dz)*h)),
                             (int((cx+dz)*w), int((cy+dz)*h)), (255, 200, 0), 2)
                cv.putText(image, "Neutral zone", (max(0, int((cx-dz)*w)),
                           max(15, int((cy-dz)*h)-8)), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 200, 0), 1)
            lines = [f"ONE HAND ({args.hand}): {state}",
                     "Inputs: " + (", ".join(sorted(active)) or "none"),
                     "F8: pause/resume | F9: recenter | F10: quit",
                     "Box = rest only; move and gesture anywhere in view"]
            if args.debug_timing:
                lines.append(f"FPS: {fps:.1f} | tracking age: {(now-snapshot['time'])*1000:.0f} ms")
            for i, line in enumerate(lines):
                cv.putText(image, line, (10, 25 + i*25), cv.FONT_HERSHEY_SIMPLEX,
                           0.55, (0, 255, 0), 2)
            cv.imshow("Hand Controller - One Hand", image)
            if cv.waitKey(1) & 0xFF == ord("p"):
                break
            if cv.getWindowProperty("Hand Controller - One Hand", cv.WND_PROP_VISIBLE) < 1:
                break
    finally:
        output.release_all()
        if listener is not None:
            listener.stop()
        if cam is not None:
            cam.stop()
        if landmarker is not None:
            landmarker.close()
        cv.destroyAllWindows()
