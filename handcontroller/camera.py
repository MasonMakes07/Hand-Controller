"""Background camera-grabbing thread.

cv.VideoCapture(...).read() on Windows can internally queue several frames
(driver/backend dependent), so a single-threaded capture step always risks
processing a frame that's already stale by the time it's read. This thread
grabs frames continuously and exposes only the most recently completed one,
discarding any backlog -- mirroring the hands_lock/latest_hands pattern
already used for MediaPipe results in hand_controller.py.
"""

import threading
import time

import cv2 as cv


class CameraStream:
    def __init__(self, index=0, backend=cv.CAP_DSHOW):
        self.cap = cv.VideoCapture(index, backend)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # advisory; the drain loop below is the real fix

        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._frame = None
        self._running = False
        self._thread = None

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def start(self) -> "CameraStream":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)  # avoid pinning a core if the camera drops out
                continue
            with self._lock:
                self._frame = frame
            self._new_frame.set()

    def read(self, timeout: float = 1.0):
        """Blocks until a fresh frame is ready, then returns (success, frame)."""
        got_new = self._new_frame.wait(timeout)
        self._new_frame.clear()
        with self._lock:
            if self._frame is None:
                return False, None
            return got_new, self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
