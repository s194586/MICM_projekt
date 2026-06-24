"""Calibration and gesture helpers for the YuNet-only fast controller."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from detector import DetectedFace


def distance(point_a: np.ndarray | None, point_b: np.ndarray | None) -> float:
    if point_a is None or point_b is None:
        return 0.0
    return float(np.linalg.norm(point_a - point_b))


@dataclass(slots=True)
class FaceMetrics:
    center_x: float
    center_y: float
    face_width: float
    face_height: float
    nose_x_ratio: float
    nose_y_ratio: float
    mouth_open_score: float


@dataclass(slots=True)
class NeutralCalibration:
    center_x: float
    center_y: float
    face_width: float
    face_height: float
    nose_x_ratio: float
    nose_y_ratio: float
    mouth_open_score: float


@dataclass(slots=True)
class LiveSignals:
    movement: float
    mouth_open: float
    vertical_head: float
    bonus: float


def mouth_roi_bounds(face: DetectedFace, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    if face.left_mouth is None or face.right_mouth is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    face_x, face_y, face_width, face_height = face.bbox
    mouth_center = (face.left_mouth + face.right_mouth) * 0.5
    mouth_width = max(distance(face.left_mouth, face.right_mouth), face.width * 0.16)

    left = max(face_x + face_width * 0.12, float(mouth_center[0] - mouth_width * 0.82))
    right = min(face_x + face_width * 0.88, float(mouth_center[0] + mouth_width * 0.82))
    top = max(face_y + face_height * 0.52, float(mouth_center[1] - face_height * 0.06))
    bottom = min(face_y + face_height * 0.95, float(mouth_center[1] + face_height * 0.24))

    x0 = max(0, int(math.floor(left)))
    y0 = max(0, int(math.floor(top)))
    x1 = min(frame_width, int(math.ceil(right)))
    y1 = min(frame_height, int(math.ceil(bottom)))

    if (x1 - x0) < 6 or (y1 - y0) < 6:
        return None
    return x0, y0, x1, y1


def mouth_open_score(frame_bgr: np.ndarray, face: DetectedFace) -> float:
    bounds = mouth_roi_bounds(face, frame_bgr.shape)
    if bounds is None:
        return 0.0

    x0, y0, x1, y1 = bounds
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mean_value = float(np.mean(gray))
    std_value = float(np.std(gray))
    dark_threshold = max(18.0, min(90.0, mean_value - (0.55 * std_value)))
    dark_ratio = float(np.mean(gray <= dark_threshold))
    contrast_score = min(std_value / 64.0, 1.0)
    vertical_edges = float(np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0))) / 255.0)
    return float((dark_ratio * 0.72) + (contrast_score * 0.20) + (vertical_edges * 0.08))


def metrics_from_face(face: DetectedFace, frame_bgr: np.ndarray) -> FaceMetrics:
    face_width = max(face.width, 1.0)
    face_height = max(face.height, 1.0)
    center_x = face.center_x
    center_y = face.center_y

    if face.nose is not None:
        nose_x_ratio = float((face.nose[0] - center_x) / face_width)
        nose_y_ratio = float((face.nose[1] - center_y) / face_height)
    else:
        nose_x_ratio = 0.0
        nose_y_ratio = 0.0

    return FaceMetrics(
        center_x=float(center_x),
        center_y=float(center_y),
        face_width=float(face_width),
        face_height=float(face_height),
        nose_x_ratio=float(nose_x_ratio),
        nose_y_ratio=float(nose_y_ratio),
        mouth_open_score=mouth_open_score(frame_bgr, face),
    )


class NeutralCalibrator:
    """Collect a short sequence of detections to define the neutral pose."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.reset()

    def reset(self, now: float | None = None) -> None:
        self.started_at = time.perf_counter() if now is None else now
        self._metrics: list[FaceMetrics] = []

    def add_sample(self, face: DetectedFace, frame_bgr: np.ndarray) -> None:
        self._metrics.append(metrics_from_face(face, frame_bgr))

    def is_ready(self, now: float) -> bool:
        return len(self._metrics) >= 8 and (now - self.started_at) >= self.duration_seconds

    def finalize(self) -> NeutralCalibration | None:
        if len(self._metrics) < 8:
            return None

        return NeutralCalibration(
            center_x=float(np.mean([item.center_x for item in self._metrics])),
            center_y=float(np.mean([item.center_y for item in self._metrics])),
            face_width=float(np.mean([item.face_width for item in self._metrics])),
            face_height=float(np.mean([item.face_height for item in self._metrics])),
            nose_x_ratio=float(np.mean([item.nose_x_ratio for item in self._metrics])),
            nose_y_ratio=float(np.mean([item.nose_y_ratio for item in self._metrics])),
            mouth_open_score=float(np.mean([item.mouth_open_score for item in self._metrics])),
        )


def movement_signal_from_metrics(metrics: FaceMetrics, neutral: NeutralCalibration) -> float:
    relative_center_shift = (metrics.center_x - neutral.center_x) / max(neutral.face_width, 1.0)
    relative_nose_shift = metrics.nose_x_ratio - neutral.nose_x_ratio
    return float((relative_nose_shift * 0.8) + (relative_center_shift * 0.2))


def movement_from_signal(signal: float, current_move: str, enter_threshold: float, exit_threshold: float) -> str:
    if current_move == "LEFT":
        if signal > enter_threshold:
            return "RIGHT"
        if signal > -exit_threshold:
            return "IDLE"
        return "LEFT"
    if current_move == "RIGHT":
        if signal < -enter_threshold:
            return "LEFT"
        if signal < exit_threshold:
            return "IDLE"
        return "RIGHT"
    if signal < -enter_threshold:
        return "LEFT"
    if signal > enter_threshold:
        return "RIGHT"
    return "IDLE"


def mouth_open_signal_from_metrics(metrics: FaceMetrics, neutral: NeutralCalibration) -> float:
    return float(metrics.mouth_open_score - neutral.mouth_open_score)


def vertical_head_signal_from_metrics(metrics: FaceMetrics, neutral: NeutralCalibration) -> float:
    nose_component = neutral.nose_y_ratio - metrics.nose_y_ratio
    center_component = (neutral.center_y - metrics.center_y) / max(neutral.face_height, 1.0)
    return float((nose_component * 0.7) + (center_component * 0.3))


def bonus_signal_from_metrics(metrics: FaceMetrics, neutral: NeutralCalibration) -> float:
    nose_component = metrics.nose_y_ratio - neutral.nose_y_ratio
    center_component = (metrics.center_y - neutral.center_y) / max(neutral.face_height, 1.0)
    return float((nose_component * 0.7) + (center_component * 0.3))


def compute_live_signals(frame_bgr: np.ndarray, face: DetectedFace, neutral: NeutralCalibration) -> LiveSignals:
    metrics = metrics_from_face(face, frame_bgr)
    return LiveSignals(
        movement=movement_signal_from_metrics(metrics, neutral),
        mouth_open=mouth_open_signal_from_metrics(metrics, neutral),
        vertical_head=vertical_head_signal_from_metrics(metrics, neutral),
        bonus=bonus_signal_from_metrics(metrics, neutral),
    )


def select_jump_signal(signals: LiveSignals, jump_mode: str) -> float:
    selected = jump_mode.strip().lower()
    if selected == "mouth_open":
        return signals.mouth_open
    if selected == "vertical_head":
        return signals.vertical_head
    raise ValueError(f"Unsupported jump mode: {jump_mode}")


class HysteresisHold:
    """Hold an action with separate enter and exit thresholds."""

    def __init__(self, enter_threshold: float, exit_threshold: float) -> None:
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.held = False

    def update(self, score: float) -> bool:
        if self.held:
            self.held = score >= self.exit_threshold
        else:
            self.held = score >= self.enter_threshold
        return self.held

    def reset(self) -> None:
        self.held = False


class CooldownTap:
    """One-shot tap gate with a cooldown and re-arm behavior."""

    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.armed = True
        self.last_trigger_time = -999.0

    def update(self, is_active: bool, now: float) -> bool:
        if not is_active:
            self.armed = True
            return False
        if not self.armed:
            return False
        if now - self.last_trigger_time < self.cooldown_seconds:
            return False

        self.armed = False
        self.last_trigger_time = now
        return True

    def cooldown_left(self, now: float) -> float:
        return max(0.0, self.cooldown_seconds - (now - self.last_trigger_time))

    def reset(self) -> None:
        self.armed = True
