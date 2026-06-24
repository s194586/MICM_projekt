"""Detector backends for the model-based non-MediaPipe controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray | None
    backend: str

    @property
    def center_x(self) -> float:
        return float(self.bbox[0] + (self.bbox[2] * 0.5))

    @property
    def center_y(self) -> float:
        return float(self.bbox[1] + (self.bbox[3] * 0.5))

    @property
    def width(self) -> float:
        return float(self.bbox[2])

    @property
    def height(self) -> float:
        return float(self.bbox[3])

    @property
    def face_center_x(self) -> float:
        return self.center_x

    @property
    def face_center_y(self) -> float:
        return self.center_y

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) >= 5

    @property
    def right_eye(self) -> np.ndarray | None:
        return None if not self.has_landmarks else self.landmarks[0]

    @property
    def left_eye(self) -> np.ndarray | None:
        return None if not self.has_landmarks else self.landmarks[1]

    @property
    def nose(self) -> np.ndarray | None:
        return None if not self.has_landmarks else self.landmarks[2]

    @property
    def right_mouth(self) -> np.ndarray | None:
        return None if not self.has_landmarks else self.landmarks[3]

    @property
    def left_mouth(self) -> np.ndarray | None:
        return None if not self.has_landmarks else self.landmarks[4]


class BaseFaceDetector:
    backend_name = "base"

    def detect(self, frame_bgr: np.ndarray) -> tuple[DetectedFace | None, float]:
        raise NotImplementedError


class YuNetFaceDetector(BaseFaceDetector):
    backend_name = "yunet"

    def __init__(
        self,
        model_path: Path,
        input_size: tuple[int, int],
        score_threshold: float = 0.88,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = tuple(int(v) for v in input_size)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing YuNet ONNX model. Expected: {self.model_path}. "
                "Download the official YuNet model and place it there."
            )

        if not hasattr(getattr(cv2, "FaceDetectorYN", object()), "create"):
            raise RuntimeError("This OpenCV build does not expose cv2.FaceDetectorYN.create")

        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            self.input_size,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[DetectedFace | None, float]:
        height, width = frame_bgr.shape[:2]
        if (width, height) != self.input_size:
            self._detector.setInputSize((width, height))
            self.input_size = (width, height)

        tick0 = cv2.getTickCount()
        _, faces = self._detector.detect(frame_bgr)
        elapsed_ms = (cv2.getTickCount() - tick0) * 1000.0 / cv2.getTickFrequency()

        if faces is None or len(faces) == 0:
            return None, float(elapsed_ms)

        faces = np.asarray(faces, dtype=np.float32)
        best_row = max(faces, key=lambda row: float(row[14]))
        bbox = (float(best_row[0]), float(best_row[1]), float(best_row[2]), float(best_row[3]))
        landmarks = best_row[4:14].reshape(5, 2).astype(np.float32)
        detected = DetectedFace(
            bbox=bbox,
            score=float(best_row[14]),
            landmarks=landmarks,
            backend=self.backend_name,
        )
        return detected, float(elapsed_ms)


class HaarFaceDetector(BaseFaceDetector):
    backend_name = "haar"

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        if self._cascade.empty():
            raise RuntimeError("Cannot load haarcascade_frontalface_alt2.xml")

    def detect(self, frame_bgr: np.ndarray) -> tuple[DetectedFace | None, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tick0 = cv2.getTickCount()
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        elapsed_ms = (cv2.getTickCount() - tick0) * 1000.0 / cv2.getTickFrequency()

        if len(faces) == 0:
            return None, float(elapsed_ms)

        best = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = [float(value) for value in best]
        landmarks = np.array(
            [
                [x + w * 0.32, y + h * 0.37],
                [x + w * 0.68, y + h * 0.37],
                [x + w * 0.50, y + h * 0.56],
                [x + w * 0.38, y + h * 0.76],
                [x + w * 0.62, y + h * 0.76],
            ],
            dtype=np.float32,
        )
        detected = DetectedFace(
            bbox=(x, y, w, h),
            score=1.0,
            landmarks=landmarks,
            backend=self.backend_name,
        )
        return detected, float(elapsed_ms)


def recommended_default_detector(model_path: Path | None = None) -> str:
    path = DEFAULT_YUNET_MODEL if model_path is None else Path(model_path)
    if path.exists() and hasattr(getattr(cv2, "FaceDetectorYN", object()), "create"):
        return "yunet"
    return "haar"


def create_face_detector(
    detector_name: str,
    input_size: tuple[int, int],
    model_path: Path | None = None,
    allow_haar_fallback: bool = False,
) -> BaseFaceDetector:
    selected = detector_name.strip().lower()
    path = DEFAULT_YUNET_MODEL if model_path is None else Path(model_path)

    if selected == "yunet":
        try:
            return YuNetFaceDetector(path, input_size=input_size)
        except Exception:
            if not allow_haar_fallback:
                raise
            return HaarFaceDetector()
    if selected == "haar":
        return HaarFaceDetector()
    raise ValueError(f"Unsupported detector backend: {detector_name}")
