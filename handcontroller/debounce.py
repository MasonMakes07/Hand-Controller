"""Generic frame-counter hysteresis, extracted from the gesture_frames pattern
originally inline in hand_controller.py. Shared by keyboard-gesture latching
and mouse pinch/engage state latching."""


class Debouncer:
    def __init__(self, keys, required_frames: int):
        self.required_frames = required_frames
        self.counters = {key: 0 for key in keys}

    def update(self, active_keys) -> dict:
        for key in self.counters:
            if key in active_keys:
                self.counters[key] = min(self.counters[key] + 1, self.required_frames)
            else:
                self.counters[key] = max(self.counters[key] - 1, 0)
        return {key: count >= self.required_frames for key, count in self.counters.items()}

    def reset(self):
        for key in self.counters:
            self.counters[key] = 0
