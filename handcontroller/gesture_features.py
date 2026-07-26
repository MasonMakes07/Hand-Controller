"""Landmark normalization and feature extraction shared by the live inference
loop (hand_controller.py), the data capture tool, and the training script.

Using one shared implementation for train-time and inference-time feature
extraction guarantees the two stay in sync.
"""

import math

import numpy as np

WRIST = 0
MIDDLE_MCP = 9

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
KNUCKLE_INDICES = [3, 7, 11, 15, 19]

# How far (in reference-distance units) a fingertip must clear its knuckle
# to count as "extended" for the geometric checks (mouse engage/disengage).
EXTEND_RATIO_THRESHOLD = 0.4


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe landmark list (or an already-built (21, 3) array) to numpy."""
    if isinstance(landmarks, np.ndarray):
        return landmarks
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


def reference_distance(landmarks) -> float:
    """Raw (un-normalized) wrist-to-middle-knuckle distance.

    Used as a stable "palm length" reference that scales with hand size /
    distance from camera but not with finger pose, so other measurements
    (pinch distance, finger curl) can be divided by it to become scale-invariant.
    """
    arr = landmarks_to_array(landmarks)
    return float(np.linalg.norm(arr[MIDDLE_MCP] - arr[WRIST])) or 1e-6


def normalize_landmarks(landmarks) -> np.ndarray:
    """Translate by the wrist and scale by reference_distance().

    Makes the resulting (21, 3) array robust to hand position in frame and
    distance from the camera. Rotation is NOT corrected for in this v1.
    """
    arr = landmarks_to_array(landmarks)
    translated = arr - arr[WRIST]
    scale = reference_distance(landmarks)
    return translated / scale


def extract_features(landmarks) -> np.ndarray:
    """Flattened normalized landmarks -> 63-dim feature vector for the classifier."""
    return normalize_landmarks(landmarks).flatten()


def finger_curl(landmarks, finger_idx: int) -> float:
    """How far finger `finger_idx` (1=index..4=pinky) is extended.

    Positive and above EXTEND_RATIO_THRESHOLD means extended; near zero or
    negative means curled into the palm. Scale-invariant via normalization.
    """
    normalized = normalize_landmarks(landmarks)
    tip = normalized[FINGERTIP_INDICES[finger_idx]]
    knuckle = normalized[KNUCKLE_INDICES[finger_idx]]
    return float(knuckle[1] - tip[1])


def is_finger_extended(landmarks, finger_idx: int) -> bool:
    return finger_curl(landmarks, finger_idx) > EXTEND_RATIO_THRESHOLD


def thumb_extension(landmarks) -> float:
    """Normalized distance between thumb tip and index knuckle (replaces the
    old thumb_out/thumb_in threshold checks with a single scale-invariant value)."""
    normalized = normalize_landmarks(landmarks)
    return float(math.hypot(normalized[4][0] - normalized[5][0], normalized[4][1] - normalized[5][1]))


def pinch_ratio(landmarks, tip_idx_a: int, tip_idx_b: int) -> float:
    """Distance between two fingertips, normalized by reference_distance().

    ~1.0 means the two tips are about a palm-length apart (hand open);
    values well below that mean the tips are pinched together.
    """
    arr = landmarks_to_array(landmarks)
    raw_dist = float(np.linalg.norm(arr[tip_idx_a] - arr[tip_idx_b]))
    return raw_dist / reference_distance(landmarks)
