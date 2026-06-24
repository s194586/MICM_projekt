"""OpenCV-only detection, tracking, and gesture helpers for the lite controller."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


Rect = tuple[int, int, int, int]


def _rect_area(rect: Rect) -> int:
    return rect[2] * rect[3]


def _rect_center(rect: Rect) -> tuple[float, float]:
    x, y, w, h = rect
    return x + (w * 0.5), y + (h * 0.5)


def clamp_rect(rect: Rect, frame_width: int, frame_height: int) -> Rect:
    x, y, w, h = rect
    x = max(0, min(int(round(x)), frame_width - 1))
    y = max(0, min(int(round(y)), frame_height - 1))
    w = max(1, min(int(round(w)), frame_width - x))
    h = max(1, min(int(round(h)), frame_height - y))
    return x, y, w, h


def movement_from_offset(offset_x: float, current_move: str, enter_threshold: float, exit_threshold: float) -> str:
    if current_move == "LEFT":
        if offset_x > enter_threshold:
            return "RIGHT"
        if offset_x > -exit_threshold:
            return "IDLE"
        return "LEFT"
    if current_move == "RIGHT":
        if offset_x < -enter_threshold:
            return "LEFT"
        if offset_x < exit_threshold:
            return "IDLE"
        return "RIGHT"
    if offset_x < -enter_threshold:
        return "LEFT"
    if offset_x > enter_threshold:
        return "RIGHT"
    return "IDLE"


def normalized_offsets(rect: Rect, calibration) -> tuple[float, float]:
    center_x, center_y = _rect_center(rect)
    width_scale = max(calibration.face_width, 1.0)
    height_scale = max(calibration.face_height, 1.0)
    offset_x = (center_x - calibration.center_x) / width_scale
    offset_y = (center_y - calibration.center_y) / height_scale
    return float(offset_x), float(offset_y)


def extract_mouth_patch(gray_frame: np.ndarray, rect: Rect, output_size: tuple[int, int] = (48, 24)) -> np.ndarray | None:
    x, y, w, h = rect
    x1 = x + int(w * 0.18)
    x2 = x + int(w * 0.82)
    y1 = y + int(h * 0.58)
    y2 = y + int(h * 0.88)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return None

    roi = cv2.resize(roi, output_size, interpolation=cv2.INTER_AREA)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    return roi


def mouth_motion_score(gray_frame: np.ndarray, rect: Rect, neutral_patch: np.ndarray | None) -> float:
    if neutral_patch is None:
        return 0.0
    current_patch = extract_mouth_patch(gray_frame, rect, output_size=(neutral_patch.shape[1], neutral_patch.shape[0]))
    if current_patch is None:
        return 0.0
    diff = cv2.absdiff(current_patch, neutral_patch)
    return float(np.mean(diff) / 255.0)


def smile_cascade_score(gray_frame: np.ndarray, rect: Rect, smile_cascade: cv2.CascadeClassifier) -> float:
    x, y, w, h = rect
    x1 = x + int(w * 0.10)
    x2 = x + int(w * 0.90)
    y1 = y + int(h * 0.45)
    y2 = y + int(h * 0.92)
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    min_w = max(18, int(w * 0.18))
    min_h = max(10, int(h * 0.08))
    smiles = smile_cascade.detectMultiScale(
        roi,
        scaleFactor=1.6,
        minNeighbors=18,
        minSize=(min_w, min_h),
    )
    if len(smiles) == 0:
        return 0.0

    largest = max(smiles, key=lambda item: item[2] * item[3])
    return float((largest[2] * largest[3]) / max(roi.shape[0] * roi.shape[1], 1))


def face_up_score(rect: Rect, calibration) -> float:
    _, center_y = _rect_center(rect)
    return float((calibration.center_y - center_y) / max(calibration.face_height, 1.0))


def pick_jump_score(
    gray_frame: np.ndarray,
    rect: Rect,
    calibration,
    jump_mode: str,
    smile_cascade: cv2.CascadeClassifier,
) -> float:
    if jump_mode == "smile_cascade":
        return smile_cascade_score(gray_frame, rect, smile_cascade)
    if jump_mode == "face_up":
        return face_up_score(rect, calibration)
    return mouth_motion_score(gray_frame, rect, calibration.mouth_patch)


@dataclass(slots=True)
class NeutralCalibration:
    center_x: float
    center_y: float
    face_width: float
    face_height: float
    mouth_patch: np.ndarray | None


class NeutralCalibrator:
    """Collect face samples for a short neutral calibration period."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.reset()

    def reset(self, now: float | None = None) -> None:
        self.started_at = time.perf_counter() if now is None else now
        self._centers_x: list[float] = []
        self._centers_y: list[float] = []
        self._widths: list[float] = []
        self._heights: list[float] = []
        self._mouth_sum = None
        self._mouth_count = 0

    def add_sample(self, gray_frame: np.ndarray, rect: Rect) -> None:
        center_x, center_y = _rect_center(rect)
        _, _, width, height = rect
        self._centers_x.append(center_x)
        self._centers_y.append(center_y)
        self._widths.append(width)
        self._heights.append(height)

        patch = extract_mouth_patch(gray_frame, rect)
        if patch is not None:
            patch32 = patch.astype(np.float32)
            if self._mouth_sum is None:
                self._mouth_sum = patch32
            else:
                self._mouth_sum += patch32
            self._mouth_count += 1

    def is_ready(self, now: float) -> bool:
        return len(self._centers_x) >= 6 and (now - self.started_at) >= self.duration_seconds

    def finalize(self) -> NeutralCalibration | None:
        if len(self._centers_x) < 6:
            return None

        mouth_patch = None
        if self._mouth_sum is not None and self._mouth_count > 0:
            mouth_patch = np.clip(self._mouth_sum / float(self._mouth_count), 0, 255).astype(np.uint8)

        return NeutralCalibration(
            center_x=float(np.mean(self._centers_x)),
            center_y=float(np.mean(self._centers_y)),
            face_width=float(np.mean(self._widths)),
            face_height=float(np.mean(self._heights)),
            mouth_patch=mouth_patch,
        )


class FaceTrackerBackend:
    """OpenCV-only face reacquire + fast tracker update."""

    def __init__(
        self,
        detect_every: int = 8,
        search_scale: float = 1.9,
    ) -> None:
        self.detect_every = max(1, detect_every)
        self.search_scale = max(1.2, search_scale)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        if self.face_cascade.empty():
            raise RuntimeError("Cannot load haarcascade_frontalface_alt2.xml")
        if self.smile_cascade.empty():
            raise RuntimeError("Cannot load haarcascade_smile.xml")

        self.tracker = None
        self.tracker_name = self._choose_tracker_name()
        self.last_rect: Rect | None = None
        self.frame_index = 0

    @property
    def backend_name(self) -> str:
        return f"haar+{self.tracker_name}"

    def reset(self) -> None:
        self.tracker = None
        self.last_rect = None

    def update(self, frame_bgr: np.ndarray, gray_frame: np.ndarray) -> tuple[Rect | None, str]:
        self.frame_index += 1
        tracked_rect = None
        tracked_ok = False

        if self.tracker is not None:
            tracked_ok, tracked_box = self.tracker.update(frame_bgr)
            if tracked_ok:
                tracked_rect = clamp_rect(
                    (tracked_box[0], tracked_box[1], tracked_box[2], tracked_box[3]),
                    frame_width=gray_frame.shape[1],
                    frame_height=gray_frame.shape[0],
                )
            else:
                self.tracker = None

        needs_detection = (self.tracker is None) or (self.frame_index % self.detect_every == 0)
        if needs_detection:
            detected_rect = self._detect_face(gray_frame, tracked_rect or self.last_rect)
            if detected_rect is not None:
                self._start_tracker(frame_bgr, detected_rect)
                self.last_rect = detected_rect
                return detected_rect, "detect"

        if tracked_ok and tracked_rect is not None:
            self.last_rect = tracked_rect
            return tracked_rect, "track"

        self.last_rect = None
        self.tracker = None
        return None, "miss"

    def _detect_face(self, gray_frame: np.ndarray, hint_rect: Rect | None) -> Rect | None:
        dynamic_min = max(36, int(min(gray_frame.shape[0], gray_frame.shape[1]) * 0.16))
        min_size = (dynamic_min, dynamic_min)

        if hint_rect is not None:
            roi_rect = self._expand_rect(hint_rect, gray_frame.shape[1], gray_frame.shape[0])
            roi = gray_frame[roi_rect[1] : roi_rect[1] + roi_rect[3], roi_rect[0] : roi_rect[0] + roi_rect[2]]
            faces = self.face_cascade.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_size,
            )
            if len(faces) > 0:
                best = self._choose_face(faces, hint_rect)
                x, y, w, h = best
                return roi_rect[0] + x, roi_rect[1] + y, w, h

        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size,
        )
        if len(faces) == 0:
            return None
        return self._choose_face(faces, hint_rect)

    def _choose_face(self, faces, hint_rect: Rect | None) -> Rect:
        if hint_rect is None:
            best = max(faces, key=lambda item: item[2] * item[3])
            return int(best[0]), int(best[1]), int(best[2]), int(best[3])

        hint_center = _rect_center(hint_rect)
        best = min(
            faces,
            key=lambda item: (
                ((_rect_center((int(item[0]), int(item[1]), int(item[2]), int(item[3])))[0] - hint_center[0]) ** 2)
                + ((_rect_center((int(item[0]), int(item[1]), int(item[2]), int(item[3])))[1] - hint_center[1]) ** 2)
            ),
        )
        return int(best[0]), int(best[1]), int(best[2]), int(best[3])

    def _expand_rect(self, rect: Rect, frame_width: int, frame_height: int) -> Rect:
        x, y, w, h = rect
        center_x, center_y = _rect_center(rect)
        new_w = int(w * self.search_scale)
        new_h = int(h * self.search_scale)
        new_x = int(center_x - (new_w * 0.5))
        new_y = int(center_y - (new_h * 0.5))
        return clamp_rect((new_x, new_y, new_w, new_h), frame_width=frame_width, frame_height=frame_height)

    def _start_tracker(self, frame_bgr: np.ndarray, rect: Rect) -> None:
        tracker = self._create_tracker()
        if tracker is None:
            self.tracker = None
            return
        tracker.init(frame_bgr, rect)
        self.tracker = tracker

    def _choose_tracker_name(self) -> str:
        if hasattr(getattr(cv2, "legacy", object()), "TrackerMOSSE_create"):
            return "mosse"
        if hasattr(cv2, "TrackerKCF_create"):
            return "kcf"
        if hasattr(cv2, "TrackerCSRT_create"):
            return "csrt"
        return "detect-only"

    def _create_tracker(self):
        if hasattr(getattr(cv2, "legacy", object()), "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        return None
