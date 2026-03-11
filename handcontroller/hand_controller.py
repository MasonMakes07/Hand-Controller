import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
from dataclasses import dataclass
import math

from KeyBinds import InputController


@dataclass
class Point:
    x: float
    y: float
    z: float


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


fingertip_indices = [4, 8, 12, 16, 20]
knuckle_indices   = [3, 7, 11, 15, 19]

model_path = "handcontroller\hand_landmarker.task"

THRESHOLD = 0.02  # raise if false positives, lower if gestures won't trigger
thumb_THRESHOLD = 0.06
REQUIRED_FRAMES = 2  # raise to reduce flicker, lower for faster response


def finger_up(landmarks, i):
    return landmarks[fingertip_indices[i]].y < landmarks[knuckle_indices[i]].y - THRESHOLD

def finger_down(landmarks, i):
    return landmarks[fingertip_indices[i]].y > landmarks[knuckle_indices[i]].y + THRESHOLD

def thumb_out(landmarks):
    return abs(landmarks[4].x - landmarks[5].x) > thumb_THRESHOLD

def thumb_in(landmarks):
    return abs(landmarks[4].x - landmarks[5].x <= thumb_THRESHOLD)


def is_fist(landmarks):
    return thumb_in(landmarks) and all(finger_down(landmarks, i) for i in range(1, 5))

def is_open_hand(landmarks):
    return thumb_out(landmarks) and all(finger_up(landmarks, i) for i in range(1, 5))

def is_pointing(landmarks): 
    return (thumb_in(landmarks) and
            finger_up(landmarks, 1) and
            finger_down(landmarks, 2) and
            finger_down(landmarks, 3) and
            finger_down(landmarks, 4))

def is_peace_sign(landmarks):
    return (finger_up(landmarks, 1) and
            finger_up(landmarks, 2) and
            finger_down(landmarks, 3) and
            finger_down(landmarks, 4))

def is_three(landmarks):
    return (finger_up(landmarks, 1) and
            finger_up(landmarks, 2) and
            finger_up(landmarks, 3) and
            finger_down(landmarks, 4))

def is_four(landmarks):
    return (thumb_in(landmarks) and
            finger_up(landmarks, 1) and
            finger_up(landmarks, 2) and
            finger_up(landmarks, 3) and
            finger_up(landmarks, 4))

def is_thumb_up(landmarks):
    return (thumb_out(landmarks) and
            finger_down(landmarks, 1) and
            finger_down(landmarks, 2) and
            finger_down(landmarks, 3) and
            finger_down(landmarks, 4))

def is_finger_gun(landmarks):
    return (thumb_out(landmarks) and
            finger_up(landmarks, 1) and
            finger_down(landmarks, 2) and
            finger_down(landmarks, 3) and
            finger_down(landmarks, 4))

def is_ok_sign(landmarks):
    return (thumb_in(landmarks) and
            finger_down(landmarks, 1) and
            finger_up(landmarks, 2) and
            finger_up(landmarks, 3) and
            finger_up(landmarks, 4))

def is_middle_finger(landmarks):
    return (thumb_in(landmarks) and
            finger_down(landmarks, 1) and
            finger_up(landmarks, 2) and
            finger_down(landmarks, 3) and
            finger_down(landmarks, 4))


GESTURE_CHECKS = {
    'fist':          is_fist,
    'open_hand':     is_open_hand,
    'pointing':      is_pointing,
    'peace_sign':    is_peace_sign,
    'three':         is_three,
    'four':          is_four,
    'thumbs_up':     is_thumb_up,
    'finger_gun':    is_finger_gun,
    'ok_sign':       is_ok_sign,
    'middle_finger': is_middle_finger
}

latest_landmarks = None
landmarks_lock   = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    with landmarks_lock:
        if result.hand_landmarks:
            latest_landmarks = result.hand_landmarks[0]
        else:
            latest_landmarks = None


if __name__ == "__main__":
    print("Starting hand controller")

    keybinds = InputController()

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=on_result
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access camera")
        exit()

    gesture_frames = {name: 0 for name in GESTURE_CHECKS}
    timestamp = 0  # must increment every frame or MediaPipe rejects frames

    print("Press 'p' to quit. Make a gesture to start.")

    while cap.isOpened():
        success, image = cap.read()
        image = cv.flip(image, 1)
        if not success:
            break
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        # Read latest landmarks written by on_result
        with landmarks_lock:
            landmarks = latest_landmarks

        if landmarks:
            for name, check_fn in GESTURE_CHECKS.items():
                detected = check_fn(landmarks)
                if detected:
                    gesture_frames[name] = min(gesture_frames[name] + 1, REQUIRED_FRAMES)
                else:
                    gesture_frames[name] = max(gesture_frames[name] - 1, 0)
                keybinds.set_gestures(name, gesture_frames[name] >= REQUIRED_FRAMES)
        else:
            gesture_frames = {name: 0 for name in GESTURE_CHECKS}
            keybinds.release_all()

        #Debugging Overlays
        #Comment out if you want, not needed
        
        #Landmarks for visualizations
        if landmarks:
            h, w = image.shape[:2]
            for idx, lm in enumerate(landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Highlight fingertips in red, knuckles in yellow, rest in white
                if idx in fingertip_indices:
                    color = (0, 0, 255)   # red
                elif idx in knuckle_indices:
                    color = (0, 255, 255) # yellow
                else:
                    color = (255, 255, 255) # white
                cv.circle(image, (cx, cy), 6, color, -1)
                cv.putText(image, str(idx), (cx + 6, cy - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        active = [name for name, count in gesture_frames.items() if count >= REQUIRED_FRAMES]
        cv.putText(image,
                   f"Gesture: {', '.join(active) if active else 'none'}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        

        cv.imshow('Hand Controller', image)
        if cv.waitKey(1) & 0xFF == ord('p'):
            break

    cap.release()
    keybinds.release_all()
    cv.destroyAllWindows()
    landmarker.close()