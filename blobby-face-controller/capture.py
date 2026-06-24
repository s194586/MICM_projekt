"""Low-latency threaded camera capture that always exposes the newest frame."""

from __future__ import annotations

import sys
import threading
import time

import cv2


def _ewma(previous: float, current: float, alpha: float = 0.2) -> float:
    if previous <= 0.0:
        return current
    return previous + alpha * (current - previous)


class LatestFrameCamera:
    """Continuously read frames and keep only the newest one."""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        camera_backend: str = "auto",
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_backend = camera_backend.strip().lower()

        self._capture = None
        self._thread = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame = None
        self._frame_timestamp = 0.0
        self._sequence = -1
        self._capture_ms = 0.0
        self._error = ""
        self._backend_name = "default"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def error(self) -> str:
        return self._error

    def start(self) -> bool:
        capture = self._open_capture()
        if capture is None:
            return False

        self._capture = capture
        self._thread = threading.Thread(target=self._reader_loop, name="blobby-camera-reader", daemon=True)
        self._thread.start()
        return True

    def latest(self) -> tuple[int, object | None, float, float]:
        with self._lock:
            return self._sequence, self._frame, self._frame_timestamp, self._capture_ms

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _candidate_backends(self) -> list[tuple[int | None, str]]:
        backend_map: dict[str, tuple[int | None, str]] = {
            "default": (None, "default"),
        }
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            backend_map["dshow"] = (cv2.CAP_DSHOW, "dshow")
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_MSMF"):
            backend_map["msmf"] = (cv2.CAP_MSMF, "msmf")

        if self.camera_backend == "auto":
            ordered = ("dshow", "msmf", "default")
            return [backend_map[name] for name in ordered if name in backend_map]

        if self.camera_backend in backend_map:
            return [backend_map[self.camera_backend]]

        raise ValueError("Unsupported camera backend. Expected one of: auto, dshow, msmf, default")

    def _open_capture(self):
        try:
            candidates = self._candidate_backends()
        except ValueError as exc:
            self._error = str(exc)
            return None

        for backend, backend_name in candidates:
            capture = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
            if not capture or not capture.isOpened():
                if capture:
                    capture.release()
                continue

            self._configure_capture(capture)
            self._backend_name = backend_name
            self._error = ""
            return capture

        self._error = f"Cannot open camera index {self.camera_index} with backend {self.camera_backend}"
        return None

    def _configure_capture(self, capture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _reader_loop(self) -> None:
        assert self._capture is not None

        while not self._stop_event.is_set():
            started_at = time.perf_counter()
            ok, frame = self._capture.read()
            finished_at = time.perf_counter()

            if not ok:
                self._error = "Camera read failed"
                time.sleep(0.005)
                continue

            with self._lock:
                self._frame = frame
                self._frame_timestamp = finished_at
                self._sequence += 1
                self._capture_ms = _ewma(self._capture_ms, (finished_at - started_at) * 1000.0)
