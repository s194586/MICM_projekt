"""Lightweight face landmark detector running on top of YuNet face crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

try:
    import onnxruntime as ort

    ONNXRUNTIME_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency.
    ort = None
    ONNXRUNTIME_IMPORT_ERROR = exc

from detector import DetectedFace


DEFAULT_LANDMARK_MODEL = Path(__file__).resolve().parent / "models" / "pfld_68_face_landmarks.onnx"
LANDMARK_INPUT_SIZE = 112
LANDMARK_CROP_SCALE = 1.10
MOUTH_POINT_INDICES_68 = tuple(range(48, 68))


@dataclass(slots=True)
class LandmarkDetectionResult:
    points: np.ndarray
    inference_ms: float
    backend_name: str

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def mouth_points(self) -> np.ndarray:
        if self.point_count < 68:
            return np.empty((0, 2), dtype=np.float32)
        return self.points[list(MOUTH_POINT_INDICES_68)]


@dataclass(slots=True)
class FaceCrop:
    left: float
    top: float
    size: float


class LandmarkDetector:
    backend_name = "base"

    def detect(self, frame_bgr: np.ndarray, face: DetectedFace) -> LandmarkDetectionResult | None:
        raise NotImplementedError


class PFLDLandmarkDetector(LandmarkDetector):
    """68-point PFLD landmark detector using ONNX Runtime on CPU."""

    backend_name = "onnxruntime"

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing landmark model. Download/place it here: {self.model_path}"
            )
        if ort is None:
            raise RuntimeError(f"onnxruntime is not available: {ONNXRUNTIME_IMPORT_ERROR}")

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray, face: DetectedFace) -> LandmarkDetectionResult | None:
        crop = self._expanded_square_crop(face)
        face_crop = self._extract_face_crop(frame_bgr, crop)
        if face_crop.size == 0:
            return None

        resized = cv2.resize(face_crop, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        started_at = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: blob})
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        landmark_vector = self._select_landmark_output(outputs)
        if landmark_vector is None:
            return None

        landmarks = landmark_vector.reshape(-1, 2).astype(np.float32)
        landmarks[:, 0] = (landmarks[:, 0] * crop.size) + crop.left
        landmarks[:, 1] = (landmarks[:, 1] * crop.size) + crop.top

        return LandmarkDetectionResult(
            points=landmarks,
            inference_ms=float(inference_ms),
            backend_name=self.backend_name,
        )

    @staticmethod
    def _select_landmark_output(outputs: list[np.ndarray]) -> np.ndarray | None:
        for output in outputs:
            if output.ndim == 2 and output.shape[0] == 1 and output.shape[1] in (136, 196):
                return output[0]
        return None

    @staticmethod
    def _expanded_square_crop(face: DetectedFace) -> FaceCrop:
        x, y, w, h = face.bbox
        size = max(w, h) * LANDMARK_CROP_SCALE
        center_x = x + (w * 0.5)
        center_y = y + (h * 0.5)
        return FaceCrop(
            left=float(center_x - (size * 0.5)),
            top=float(center_y - (size * 0.5)),
            size=float(size),
        )

    @staticmethod
    def _extract_face_crop(frame_bgr: np.ndarray, crop: FaceCrop) -> np.ndarray:
        frame_height, frame_width = frame_bgr.shape[:2]
        left = int(np.floor(crop.left))
        top = int(np.floor(crop.top))
        right = int(np.ceil(crop.left + crop.size))
        bottom = int(np.ceil(crop.top + crop.size))

        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - frame_width)
        pad_bottom = max(0, bottom - frame_height)

        left = max(0, left)
        top = max(0, top)
        right = min(frame_width, right)
        bottom = min(frame_height, bottom)

        crop_bgr = frame_bgr[top:bottom, left:right]
        if crop_bgr.size == 0:
            return np.empty((0, 0, 3), dtype=frame_bgr.dtype)

        if any((pad_left, pad_top, pad_right, pad_bottom)):
            crop_bgr = cv2.copyMakeBorder(
                crop_bgr,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
        return crop_bgr


def create_landmark_detector(model_path: Path | None = None) -> LandmarkDetector:
    path = DEFAULT_LANDMARK_MODEL if model_path is None else Path(model_path)
    return PFLDLandmarkDetector(path)
