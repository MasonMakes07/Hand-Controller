"""Relative (trackpad-style) mouse control driven by the "mouse hand".

State priority per frame (first match wins), mirroring a touchpad:
    1. Hand closed into a fist              -> paused (cursor frozen, no drag)
    2. Thumb+index pinch, held              -> left button held down for as long as the
                                                pinch is held (quick pinch = click, hold+move = drag)
    3. Thumb+middle pinch, held              -> right button held down for as long as the pinch is held
    4. Index+middle extended, others curled -> two-finger scroll
    5. Otherwise (hand open, engaged)       -> relative cursor move
"""

import ctypes

from pynput.mouse import Button, Controller as PynputMouseController

import config
import gesture_features
import mouse_calibration
from one_euro_filter import OneEuroFilter

NON_THUMB_FINGERS = (1, 2, 3, 4)  # index, middle, ring, pinky
REF_LANDMARK = 9  # middle-finger MCP: stable on the palm regardless of finger curl


class MouseController:
    def __init__(self):
        self.mouse = PynputMouseController()
        self._left_held = False
        self._right_held = False
        self._pinch_left_active = False
        self._pinch_right_active = False
        self._prev_ref_point = None
        self._prev_scroll_y = None
        self._filter_x = OneEuroFilter(min_cutoff=config.MOUSE_FILTER_MIN_CUTOFF, beta=config.MOUSE_FILTER_BETA)
        self._filter_y = OneEuroFilter(min_cutoff=config.MOUSE_FILTER_MIN_CUTOFF, beta=config.MOUSE_FILTER_BETA)
        self.screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        self.bounds = mouse_calibration.load_calibration()

    def release_all(self):
        if self._left_held:
            self.mouse.release(Button.left)
            self._left_held = False
        if self._right_held:
            self.mouse.release(Button.right)
            self._right_held = False
        self._prev_ref_point = None
        self._prev_scroll_y = None

    def update(self, landmarks) -> dict:
        if landmarks is None:
            self.release_all()
            return {"state": "no_hand"}

        is_paused = all(
            gesture_features.finger_curl(landmarks, i) <= gesture_features.EXTEND_RATIO_THRESHOLD
            for i in NON_THUMB_FINGERS
        )

        pinch_left_ratio = gesture_features.pinch_ratio(landmarks, 4, 8)
        pinch_right_ratio = gesture_features.pinch_ratio(landmarks, 4, 12)
        self._pinch_left_active = self._update_hysteresis(self._pinch_left_active, pinch_left_ratio)
        self._pinch_right_active = self._update_hysteresis(self._pinch_right_active, pinch_right_ratio)

        scroll_pose = (
            gesture_features.is_finger_extended(landmarks, 1)
            and gesture_features.is_finger_extended(landmarks, 2)
            and not gesture_features.is_finger_extended(landmarks, 3)
            and not gesture_features.is_finger_extended(landmarks, 4)
        )

        if is_paused:
            self.release_all()
            state = "paused"
        elif self._pinch_left_active:
            if self._right_held:
                self.mouse.release(Button.right)
                self._right_held = False
            if not self._left_held:
                self.mouse.press(Button.left)
                self._left_held = True
            self._prev_scroll_y = None
            self._move_cursor(landmarks)
            state = "left_held"
        elif self._pinch_right_active:
            if self._left_held:
                self.mouse.release(Button.left)
                self._left_held = False
            if not self._right_held:
                self.mouse.press(Button.right)
                self._right_held = True
            self._prev_scroll_y = None
            self._move_cursor(landmarks)
            state = "right_held"
        else:
            if self._left_held:
                self.mouse.release(Button.left)
                self._left_held = False
            if self._right_held:
                self.mouse.release(Button.right)
                self._right_held = False

            if scroll_pose:
                self._prev_ref_point = None
                self._scroll(landmarks)
                state = "scroll"
            else:
                self._prev_scroll_y = None
                self._move_cursor(landmarks)
                state = "move"

        return {
            "state": state,
            "pinch_left_ratio": pinch_left_ratio,
            "pinch_right_ratio": pinch_right_ratio,
        }

    def _update_hysteresis(self, active: bool, ratio: float) -> bool:
        if active:
            return ratio < config.PINCH_OPEN_RATIO
        return ratio < config.PINCH_CLOSE_RATIO

    def _move_cursor(self, landmarks):
        raw_ref = (landmarks[REF_LANDMARK].x, landmarks[REF_LANDMARK].y)

        # 1-euro filter: heavily smooths the landmark position while it's
        # nearly still (kills jitter), but automatically lets more through
        # as estimated velocity rises (stays responsive during real movement).
        smoothed_ref = (self._filter_x.filter(raw_ref[0]), self._filter_y.filter(raw_ref[1]))

        if self._prev_ref_point is None:
            self._prev_ref_point = smoothed_ref
            return

        delta_x = smoothed_ref[0] - self._prev_ref_point[0]
        delta_y = smoothed_ref[1] - self._prev_ref_point[1]
        self._prev_ref_point = smoothed_ref

        # Dead zone: residual jitter that survives filtering still shouldn't
        # move the cursor at all when the hand is essentially holding still.
        if abs(delta_x) < config.MOUSE_DEADZONE:
            delta_x = 0.0
        if abs(delta_y) < config.MOUSE_DEADZONE:
            delta_y = 0.0

        if self.bounds is not None:
            range_x = max(self.bounds["max_x"] - self.bounds["min_x"], 1e-6)
            range_y = max(self.bounds["max_y"] - self.bounds["min_y"], 1e-6)
            dx = delta_x / range_x * self.screen_w * config.MOUSE_CALIBRATED_GAIN_X
            dy = delta_y / range_y * self.screen_h * config.MOUSE_CALIBRATED_GAIN_Y
        else:
            dx = delta_x * self.screen_w * config.MOUSE_SENSITIVITY
            dy = delta_y * self.screen_h * config.MOUSE_SENSITIVITY

        if config.MOUSE_ACCEL_GAIN:
            accel = 1 + config.MOUSE_ACCEL_GAIN * (dx * dx + dy * dy) ** 0.5
            dx *= accel
            dy *= accel

        curr_x, curr_y = self.mouse.position
        new_x = min(max(0, curr_x + dx), self.screen_w - 1)
        new_y = min(max(0, curr_y + dy), self.screen_h - 1)
        self.mouse.position = (new_x, new_y)

    def _scroll(self, landmarks):
        y = (landmarks[8].y + landmarks[12].y) / 2
        if self._prev_scroll_y is None:
            self._prev_scroll_y = y
            return

        dy = y - self._prev_scroll_y
        self._prev_scroll_y = y
        amount = int(-dy * config.SCROLL_SENSITIVITY * 100)
        if amount:
            self.mouse.scroll(0, amount)
