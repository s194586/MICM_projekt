"""Feature extraction from MediaPipe Face Mesh landmarks.

The functions in this file are deliberately independent from OpenCV windows,
keyboard control and training code. That makes this module easy to reuse in
Google Colab later for dataset inspection or model training.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


FEATURE_NAMES = [
    "mouth_open_ratio",
    "mouth_width_ratio",
    "left_eye_open_ratio",
    "right_eye_open_ratio",
    "eyebrow_raise_left",
    "eyebrow_raise_right",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_width",
    "face_height",
]

# Face Mesh landmark indices used for lightweight, stable geometric features.
NOSE_TIP = 1
FOREHEAD_TOP = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_BROW = 105

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_BROW = 334

EPS = 1e-6


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe landmark list to an Nx3 float array."""
    raw_landmarks = getattr(landmarks, "landmark", landmarks)
    return np.array([[point.x, point.y, point.z] for point in raw_landmarks], dtype=np.float32)


def face_center_x(landmarks) -> float:
    """Return the horizontal face center used for left-to-right player sorting."""
    points = landmarks_to_array(landmarks)
    return float(np.mean(points[:, 0]))


def face_bbox(landmarks) -> tuple[float, float, float, float]:
    """Return normalized bounding box coordinates: min_x, min_y, max_x, max_y."""
    points = landmarks_to_array(landmarks)
    min_x = float(np.min(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_x = float(np.max(points[:, 0]))
    max_y = float(np.max(points[:, 1]))
    return min_x, min_y, max_x, max_y


def sort_faces_left_to_right(face_landmarks: Iterable) -> list:
    """Sort detected faces so player 1 is on the left and player 2 on the right."""
    return sorted(face_landmarks, key=face_center_x)


def _dist(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first, :2] - points[second, :2]))


def _midpoint(points: np.ndarray, first: int, second: int) -> np.ndarray:
    return (points[first, :2] + points[second, :2]) / 2.0


def extract_feature_dict(landmarks) -> dict[str, float]:
    """Return named normalized features for one face."""
    points = landmarks_to_array(landmarks)

    cheek_width = _dist(points, LEFT_CHEEK, RIGHT_CHEEK)
    face_height = _dist(points, FOREHEAD_TOP, CHIN)
    face_width = max(cheek_width, EPS)
    norm_height = max(face_height, EPS)

    mouth_open = _dist(points, UPPER_LIP, LOWER_LIP)
    mouth_width = _dist(points, MOUTH_LEFT, MOUTH_RIGHT)

    left_eye_open = _dist(points, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
    left_eye_width = _dist(points, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right_eye_open = _dist(points, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
    right_eye_width = _dist(points, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)

    left_eye_center = _midpoint(points, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right_eye_center = _midpoint(points, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
    eye_mid = (left_eye_center + right_eye_center) / 2.0

    # Smaller y means higher in the image. Positive values mean brow raised.
    eyebrow_raise_left = float((left_eye_center[1] - points[LEFT_BROW, 1]) / norm_height)
    eyebrow_raise_right = float((right_eye_center[1] - points[RIGHT_BROW, 1]) / norm_height)

    face_center = _midpoint(points, LEFT_CHEEK, RIGHT_CHEEK)
    nose = points[NOSE_TIP, :2]

    # Practical yaw proxy: nose horizontal offset relative to cheek midpoint.
    # Negative usually means the nose moved toward the left side of the camera image.
    head_yaw = float((nose[0] - face_center[0]) / face_width)
    head_pitch = float((nose[1] - eye_mid[1]) / norm_height)
    head_roll = float(math.degrees(math.atan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])))

    return {
        "mouth_open_ratio": float(mouth_open / max(mouth_width, EPS)),
        "mouth_width_ratio": float(mouth_width / face_width),
        "left_eye_open_ratio": float(left_eye_open / max(left_eye_width, EPS)),
        "right_eye_open_ratio": float(right_eye_open / max(right_eye_width, EPS)),
        "eyebrow_raise_left": eyebrow_raise_left,
        "eyebrow_raise_right": eyebrow_raise_right,
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
        "head_roll": head_roll,
        "face_width": float(face_width),
        "face_height": float(face_height),
    }


def extract_features(landmarks) -> np.ndarray:
    """Return the ordered numeric feature vector used by CSV and ML model."""
    feature_dict = extract_feature_dict(landmarks)
    return np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)


def estimate_smile_score(features: Mapping[str, float]) -> float:
    """Return a simple normalized smile score for rule-based jump detection.

    Mouth width is already normalized by face width, which makes this score
    easier to calibrate across small changes in distance from the camera.
    """
    return float(features["mouth_width_ratio"])
