"""Calibration and gesture helpers for the model-based fast controller."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from model_fast_detector import DetectedFace


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
    mouth_width_ratio: float
    eye_distance_ratio: float


def metrics_from_face(face: DetectedFace) -> FaceMetrics:
    face_width = max(face.width, 1.0)
    face_height = max(face.height, 1.0)
    center_x = face.face_center_x
    center_y = face.face_center_y

    if face.nose is not None:
        nose_x_ratio = float((face.nose[0] - center_x) / face_width)
        nose_y_ratio = float((face.nose[1] - center_y) / face_height)
    else:
        nose_x_ratio = 0.0
        nose_y_ratio = 0.0

    mouth_width_ratio = distance(face.left_mouth, face.right_mouth) / face_width
    eye_distance_ratio = distance(face.left_eye, face.right_eye) / face_width

    return FaceMetrics(
        center_x=float(center_x),
        center_y=float(center_y),
        face_width=float(face_width),
        face_height=float(face_height),
        nose_x_ratio=float(nose_x_ratio),
        nose_y_ratio=float(nose_y_ratio),
        mouth_width_ratio=float(mouth_width_ratio),
        eye_distance_ratio=float(eye_distance_ratio),
    )


@dataclass(slots=True)
class NeutralCalibration:
    center_x: float
    center_y: float
    face_width: float
    face_height: float
    nose_x_ratio: float
    nose_y_ratio: float
    mouth_width_ratio: float
    eye_distance_ratio: float


class NeutralCalibrator:
    """Collect a short sequence of detections to define the neutral pose."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.reset()

    def reset(self, now: float | None = None) -> None:
        self.started_at = time.perf_counter() if now is None else now
        self._metrics: list[FaceMetrics] = []

    def add_sample(self, face: DetectedFace) -> None:
        self._metrics.append(metrics_from_face(face))

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
            mouth_width_ratio=float(np.mean([item.mouth_width_ratio for item in self._metrics])),
            eye_distance_ratio=float(np.mean([item.eye_distance_ratio for item in self._metrics])),
        )


def movement_signal(face: DetectedFace, neutral: NeutralCalibration) -> float:
    metrics = metrics_from_face(face)
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


def jump_signal(face: DetectedFace, neutral: NeutralCalibration, jump_mode: str) -> float:
    metrics = metrics_from_face(face)
    selected = jump_mode.strip().lower()

    if selected in ("smile_landmarks", "mouth_distance"):
        return float(metrics.mouth_width_ratio - neutral.mouth_width_ratio)
    if selected == "vertical_head":
        nose_component = neutral.nose_y_ratio - metrics.nose_y_ratio
        center_component = (neutral.center_y - metrics.center_y) / max(neutral.face_height, 1.0)
        return float((nose_component * 0.7) + (center_component * 0.3))
    if selected == "keyboard_test":
        return 0.0
    raise ValueError(f"Unsupported jump mode: {jump_mode}")


def bonus_signal(face: DetectedFace, neutral: NeutralCalibration) -> float:
    metrics = metrics_from_face(face)
    nose_component = metrics.nose_y_ratio - neutral.nose_y_ratio
    center_component = (metrics.center_y - neutral.center_y) / max(neutral.face_height, 1.0)
    return float((nose_component * 0.7) + (center_component * 0.3))


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
