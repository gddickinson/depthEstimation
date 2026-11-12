"""Live Kinect access for the web app: one PyKinectCapture behind a lock, never blocking a request for long.

kinect_wrapper (the Cython extension built by setup.py) is imported lazily, so the app runs without it; status()
then says why there is no live stream. get_frames() blocks until libfreenect delivers a frame, so it runs on a single
worker thread with a timeout.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout


def _default_factory():
    import kinect_wrapper  # built by `python setup.py build_ext --inplace`; needs libfreenect
    return kinect_wrapper.PyKinectCapture()


class KinectSource:
    def __init__(self, factory=_default_factory, timeout=3.0):
        self.factory = factory
        self.timeout = timeout
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kinect")
        self.cap = None
        self.state = "idle"  # idle | streaming | no-device | unavailable | stalled
        self.reason = "Not connected yet."
        self.frames = 0

    def status(self):
        return {"state": self.state, "reason": self.reason, "frames": self.frames, "streaming": self.state == "streaming"}

    def connect(self):
        """Open the first Kinect (no-op while streaming)."""
        with self.lock:
            if self.state == "streaming":
                return self.status()
            try:
                cap = self.factory()
            except Exception as e:  # extension missing or built for another Python
                self.state, self.reason = "unavailable", (
                    f"kinect_wrapper could not be loaded ({type(e).__name__}: {e}). Build it with "
                    "`python setup.py build_ext --inplace` (needs libfreenect).")
                return self.status()
            if cap.init():
                self.cap, self.state, self.reason, self.frames = cap, "streaming", "Kinect streaming.", 0
            else:
                cap.close()
                self.state, self.reason = "no-device", "No Kinect found: libfreenect could not open device 0."
            return self.status()

    def frame(self):
        """(rgb, depth) numpy arrays of the next frame, or None when not streaming or the device stalls."""
        with self.lock:
            if self.state != "streaming":
                return None
            fut = self.pool.submit(self.cap.get_frames)
            try:
                rgb, depth = fut.result(timeout=self.timeout)
            except FutureTimeout:
                self.state, self.reason = "stalled", f"No frame from the Kinect within {self.timeout:g} s."
                return None
            if rgb is None or depth is None:
                self.state, self.reason = "no-device", "The Kinect stopped streaming."
                return None
            self.frames += 1
            return rgb, depth

    def close(self):
        with self.lock:
            if self.cap is not None and self.state != "stalled":
                self.cap.close()
            self.cap, self.state, self.reason = None, "idle", "Disconnected."
