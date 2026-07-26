"""Record labeled hand-landmark samples from the webcam for training a
gesture classifier (see train_gesture_model.py).

Usage:
    python capture_gestures.py --role keyboard
    python capture_gestures.py --role mouse --out data/gestures_mouse.csv

Controls:
    number/letter keys shown on screen  - select which gesture label you're about to record
    SPACE                                - toggle recording on/off for the current label
    q                                    - quit (flushes any buffered samples first)
"""

import argparse
import csv
import os
import threading
import time

import cv2 as cv
import mediapipe as mp

import config
import tracker

CAPTURE_INTERVAL = 0.1  # seconds between saved samples while recording (~10/sec)
FLUSH_INTERVAL = 3.0  # seconds between buffer flushes to disk

KEY_POOL = list("1234567890twerqyuiopasdfghjklzxcvbnm")

latest_landmarks = None
latest_handedness = None
landmarks_lock = threading.Lock()


def on_result(result, output_image, timestamp_ms):
    global latest_landmarks, latest_handedness
    with landmarks_lock:
        if result.hand_landmarks:
            latest_landmarks = result.hand_landmarks[0]
            latest_handedness = result.handedness[0][0].category_name
        else:
            latest_landmarks = None
            latest_handedness = None


def row_for(label, handedness, landmarks):
    row = [label, handedness or "unknown"]
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=list(config.GESTURE_LABELS_BY_ROLE))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    labels = config.GESTURE_LABELS_BY_ROLE[args.role]
    label_keys = {KEY_POOL[i]: label for i, label in enumerate(labels)}

    out_path = args.out or os.path.join(config.DATA_DIR, f"gestures_{args.role}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    file_is_new = not os.path.exists(out_path) or os.path.getsize(out_path) == 0

    csv_file = open(out_path, "a", newline="")
    writer = csv.writer(csv_file)
    if file_is_new:
        header = ["label", "handedness"] + [f"lm{i}_{axis}" for i in range(21) for axis in "xyz"]
        writer.writerow(header)
        csv_file.flush()

    landmarker = tracker.create_landmarker(on_result, num_hands=1)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access camera")
        return

    current_label = labels[0]
    recording = False
    buffer = []
    session_counts = {label: 0 for label in labels}
    last_capture_time = 0.0
    last_flush_time = time.time()
    timestamp = 0

    print(f"Recording for role='{args.role}' -> {out_path}")
    print("Labels:", ", ".join(f"[{key}]={label}" for key, label in label_keys.items()))
    print("SPACE toggles recording, q quits.")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv.flip(image, 1)
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            timestamp += 1
            landmarker.detect_async(mp_image, timestamp)

            with landmarks_lock:
                landmarks = latest_landmarks
                handedness = latest_handedness

            now = time.time()
            if recording and landmarks and (now - last_capture_time) >= CAPTURE_INTERVAL:
                buffer.append(row_for(current_label, handedness, landmarks))
                session_counts[current_label] += 1
                last_capture_time = now

            if buffer and (now - last_flush_time) >= FLUSH_INTERVAL:
                writer.writerows(buffer)
                csv_file.flush()
                buffer.clear()
                last_flush_time = now

            y = 30
            for key, label in label_keys.items():
                marker = " <-- current" if label == current_label else ""
                color = (0, 255, 0) if label == current_label else (255, 255, 255)
                cv.putText(image, f"[{key}] {label} ({session_counts[label]}){marker}",
                           (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20

            status = f"RECORDING ({current_label})" if recording else "paused"
            cv.putText(image, status, (10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255) if recording else (200, 200, 200), 2)

            cv.imshow("Capture Gestures", image)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
            else:
                key_char = chr(key) if 32 < key < 127 else None
                if key_char in label_keys:
                    current_label = label_keys[key_char]
    finally:
        # Runs on normal quit ('q'), camera failure, or any unexpected
        # exception (e.g. Ctrl+C) so buffered-but-unflushed samples are
        # never silently lost.
        if buffer:
            writer.writerows(buffer)
            csv_file.flush()
        csv_file.close()

        cap.release()
        cv.destroyAllWindows()
        landmarker.close()

    print("Session sample counts:", session_counts)


if __name__ == "__main__":
    main()
