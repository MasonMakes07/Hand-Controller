"""Per-key hysteresis for latching gestures on/off across frames.

Originally a single symmetric frame-counter (extracted from the
gesture_frames pattern inline in hand_controller.py). Redesigned around an
explicit latch + streak-toward-opposite-state so activation and
deactivation can use different thresholds -- fast presses, guarded
releases -- without the two thresholds interfering with each other."""


class Debouncer:
    def __init__(self, keys, required_frames_on: int, required_frames_off: int = None):
        self.required_frames_on = max(1, required_frames_on)
        self.required_frames_off = max(1, required_frames_off if required_frames_off is not None else required_frames_on)
        self._latched = {key: False for key in keys}
        self._streak = {key: 0 for key in keys}

    def update(self, active_keys) -> dict:
        for key, latched in self._latched.items():
            is_active = key in active_keys
            if is_active == latched:
                self._streak[key] = 0
                continue
            self._streak[key] += 1
            threshold = self.required_frames_on if is_active else self.required_frames_off
            if self._streak[key] >= threshold:
                self._latched[key] = is_active
                self._streak[key] = 0
        return dict(self._latched)

    def reset(self):
        for key in self._latched:
            self._latched[key] = False
            self._streak[key] = 0
