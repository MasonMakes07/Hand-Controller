"""One-hand directional control. Pure state logic; no camera or OS input."""


class OneHandController:
    def __init__(self, deadzone=0.08, release_ratio=0.65, calibration_seconds=1.0,
                 calibration_tolerance=0.025, action_gestures=None):
        self.deadzone = deadzone
        self.release_ratio = release_ratio
        self.calibration_seconds = calibration_seconds
        self.calibration_tolerance = calibration_tolerance
        self.action_gestures = action_gestures or {"jump": "fist", "interact": "thumbs_up"}
        self.paused = False
        self.reset()

    def reset(self):
        self.center = None
        self._candidate = None
        self._since = None
        self._directions = set()

    def toggle_pause(self):
        self.paused = not self.paused
        self.reset()

    def update(self, point, gestures, now):
        if self.paused:
            return set(), "paused"
        if point is None:
            self.reset()
            return set(), "no hand - input released"
        if self.center is None:
            if self._candidate is None or max(abs(a - b) for a, b in zip(point, self._candidate)) > self.calibration_tolerance:
                self._candidate, self._since = point, now
            if now - self._since >= self.calibration_seconds:
                self.center = point
            # Never emit actions on the frame that completes calibration.
            return set(), "hold still at a comfortable resting position"

        dx, dy = point[0] - self.center[0], point[1] - self.center[1]
        directions = set()
        for name, distance in (("left", -dx), ("right", dx), ("up", -dy), ("down", dy)):
            threshold = self.deadzone * (self.release_ratio if name in self._directions else 1.0)
            if distance > threshold:
                directions.add(name)
        self._directions = directions
        actions = {name for name, gesture in self.action_gestures.items() if gesture in gestures}
        return directions | actions, "active"
