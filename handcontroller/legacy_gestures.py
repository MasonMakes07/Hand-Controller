"""Original hardcoded threshold-based gesture detection, kept as a
zero-setup fallback for when no trained classifier model exists yet.

This is frozen reference code moved verbatim out of hand_controller.py -
including the known thumb_in bug (parens wrap the comparison instead of the
distance). The bug is fixed for the *learned* path via gesture_features.py,
not patched here, since this module is only a bootstrap/fallback.
"""

import config

fingertip_indices = [4, 8, 12, 16, 20]
knuckle_indices = [3, 7, 11, 15, 19]

THRESHOLD = config.THRESHOLD
thumb_THRESHOLD = config.THUMB_THRESHOLD


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
