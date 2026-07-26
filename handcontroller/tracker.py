"""Shared MediaPipe HandLandmarker setup, used by both hand_controller.py
(live inference) and capture_gestures.py (data capture), so the two never
drift apart in confidence thresholds / running mode."""

import mediapipe as mp

import config


def create_landmarker(result_callback, num_hands: int = 2):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=result_callback,
    )
    return HandLandmarker.create_from_options(options)
