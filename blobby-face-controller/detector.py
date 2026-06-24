"""YuNet face detector wrapper used by the final one-player controller."""

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


class YuNetFaceDetector:
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

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing YuNet ONNX model. Expected: {self.model_path}. "
                "Place face_detection_yunet_2023mar.onnx in the models directory."
            )

        if not hasattr(getattr(cv2, "FaceDetectorYN", object()), "create"):
            raise RuntimeError("This OpenCV build does not expose cv2.FaceDetectorYN.create")

        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            self.input_size,
            score_threshold,
            nms_threshold,
            top_k,
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
        return DetectedFace(
            bbox=bbox,
            score=float(best_row[14]),
            landmarks=landmarks,
            backend=self.backend_name,
        ), float(elapsed_ms)


def create_face_detector(
    detector_name: str,
    input_size: tuple[int, int],
    model_path: Path | None = None,
) -> YuNetFaceDetector:
    if detector_name.strip().lower() != "yunet":
        raise ValueError(f"Unsupported detector backend: {detector_name}. Only 'yunet' is supported.")
    return YuNetFaceDetector(DEFAULT_YUNET_MODEL if model_path is None else Path(model_path), input_size=input_size)
