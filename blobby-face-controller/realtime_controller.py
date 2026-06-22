"""Realtime two-person controller for Blobby Volley Online."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import joblib
import mediapipe as mp
import numpy as np

import config
from feature_extraction import (
    FEATURE_NAMES,
    estimate_smile_score,
    extract_feature_dict,
    extract_features,
    face_bbox,
    sort_faces_left_to_right,
)
from keyboard_utils import resolve_key, tap_key

try:
    from pynput.keyboard import Controller

    PYNPUT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local desktop/session.
    Controller = None
    PYNPUT_IMPORT_ERROR = exc


LEFT = "LEFT"
RIGHT = "RIGHT"
IDLE = "IDLE"


class StableAction:
    """Confirm an action only after it appears for several consecutive frames."""

    def __init__(self, confirm_frames: int, initial: str = IDLE):
        self.confirm_frames = max(1, confirm_frames)
        self.candidate = initial
        self.candidate_frames = 0
        self.stable = initial

    def update(self, action: str) -> str:
        if action == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = action
            self.candidate_frames = 1

        if self.candidate_frames >= self.confirm_frames:
            self.stable = self.candidate
        return self.stable

    def reset(self, value: str = IDLE) -> None:
        self.candidate = value
        self.candidate_frames = 0
        self.stable = value


class GestureDebouncer:
    """Debounce the one-shot bonus gesture."""

    def __init__(self, confirm_frames: int, cooldown_seconds: float):
        self.confirm_frames = max(1, confirm_frames)
        self.cooldown_seconds = cooldown_seconds
        self.true_frames = 0
        self.armed = True
        self.last_trigger_time = -999.0

    def update(self, is_active: bool, now: float) -> bool:
        if is_active:
            self.true_frames += 1
        else:
            self.true_frames = 0
            self.armed = True

        if not self.armed:
            return False
        if self.true_frames < self.confirm_frames:
            return False
        if now - self.last_trigger_time < self.cooldown_seconds:
            return False

        self.last_trigger_time = now
        self.armed = False
        return True

    def cooldown_left(self, now: float) -> float:
        return max(0.0, self.cooldown_seconds - (now - self.last_trigger_time))

    def reset(self) -> None:
        self.true_frames = 0
        self.armed = True


class ConfirmedHold:
    """Hold an action after confirmation and release it immediately when false."""

    def __init__(self, confirm_frames: int):
        self.confirm_frames = max(1, confirm_frames)
        self.true_frames = 0
        self.held = False

    def update(self, is_active: bool) -> bool:
        if not is_active:
            self.true_frames = 0
            self.held = False
            return False
        self.true_frames += 1
        if self.true_frames >= self.confirm_frames:
            self.held = True
        return self.held

    def reset(self) -> None:
        self.true_frames = 0
        self.held = False


class KeyboardManager:
    """Stateful keyboard output that avoids spamming repeated keyDown events."""

    def __init__(self):
        self.controller = None
        self.enabled = False
        self.held_keys: set[object] = set()
        self.error = str(PYNPUT_IMPORT_ERROR) if PYNPUT_IMPORT_ERROR else ""

        if Controller is not None and PYNPUT_IMPORT_ERROR is None:
            try:
                self.controller = Controller()
                self.enabled = True
            except Exception as exc:  # pragma: no cover - desktop dependent.
                self.error = str(exc)

    def press(self, key) -> None:
        if not self.enabled or self.controller is None:
            return
        try:
            self.controller.press(key)
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def release(self, key) -> None:
        if self.controller is None:
            return
        try:
            self.controller.release(key)
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def hold(self, key) -> None:
        if key not in self.held_keys:
            self.press(key)
            self.held_keys.add(key)

    def release_hold(self, key) -> None:
        if key in self.held_keys:
            self.held_keys.remove(key)
            self.release(key)

    def set_movement(self, action: str, left_key, right_key) -> None:
        if action == LEFT:
            self.release_hold(right_key)
            self.hold(left_key)
        elif action == RIGHT:
            self.release_hold(left_key)
            self.hold(right_key)
        else:
            self.release_hold(left_key)
            self.release_hold(right_key)

    def set_hold(self, key, should_hold: bool) -> None:
        if should_hold:
            self.hold(key)
        else:
            self.release_hold(key)

    def release_all(self) -> None:
        for key in set(self.held_keys):
            self.release(key)
        self.held_keys.clear()

    def status(self) -> str:
        if self.enabled:
            return "keyboard OK"
        if self.error:
            return f"keyboard disabled: {self.error[:55]}"
        return "keyboard disabled"


@dataclass
class BonusModel:
    model: object | None
    status: str


def load_bonus_model() -> BonusModel:
    if not config.MODEL_PATH.exists():
        return BonusModel(None, f"model not loaded: missing {config.MODEL_PATH.name}")
    try:
        payload = joblib.load(config.MODEL_PATH)
        model = payload.get("model") if isinstance(payload, dict) else None
    except Exception as exc:
        return BonusModel(None, f"model not loaded: {exc}")

    if model is None:
        return BonusModel(None, "model not loaded: invalid or legacy model file")
    if payload.get("feature_names") != FEATURE_NAMES:
        return BonusModel(None, "model not loaded: feature schema mismatch")
    if payload.get("target_gesture") != config.BONUS_GESTURE_ID:
        return BonusModel(None, "model not loaded: retrain for head-down gesture")
    return BonusModel(model, "model OK")


def draw_face_box(frame, landmarks, label: str, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    min_x, min_y, max_x, max_y = face_bbox(landmarks)
    x1 = int(min_x * width)
    y1 = int(min_y * height)
    x2 = int(max_x * width)
    y2 = int(max_y * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def put_lines(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        y += 25


def movement_from_head_yaw(head_yaw: float, current_move: str = IDLE) -> str:
    """Return movement with separate enter and exit thresholds (hysteresis)."""
    enter = config.HEAD_YAW_ENTER_THRESHOLD
    exit_threshold = config.HEAD_YAW_EXIT_THRESHOLD

    if current_move == LEFT:
        if head_yaw > enter:
            return RIGHT
        if head_yaw > -exit_threshold:
            return IDLE
        return LEFT
    if current_move == RIGHT:
        if head_yaw < -enter:
            return LEFT
        if head_yaw < exit_threshold:
            return IDLE
        return RIGHT
    if head_yaw < -enter:
        return LEFT
    if head_yaw > enter:
        return RIGHT
    return IDLE


def is_jump_gesture(features: dict[str, float]) -> bool:
    """Detect Player 2 jump without involving the trained bonus model."""
    if config.JUMP_MODE == "mouth_open":
        return features["mouth_open_ratio"] >= config.MOUTH_OPEN_THRESHOLD
    return estimate_smile_score(features) >= config.SMILE_THRESHOLD


def assign_player_faces(faces: list, solo_test_mode: bool) -> tuple[object | None, object | None, bool]:
    """Return Player 1, Player 2 and whether single-face solo fallback is active."""
    if len(faces) >= 2:
        return faces[0], faces[1], False
    if len(faces) == 1 and solo_test_mode and config.SOLO_TEST_ROLE == "player2":
        return None, faces[0], True
    return None, None, False


def predict_bonus(model, feature_vector: np.ndarray) -> tuple[bool, float | None]:
    vector = feature_vector.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        classes = list(getattr(model, "classes_", []))
        if config.LABEL_BONUS in classes:
            bonus_index = classes.index(config.LABEL_BONUS)
        else:
            bonus_index = 1 if len(probabilities) > 1 else 0
        bonus_probability = float(probabilities[bonus_index])
        return bonus_probability >= config.BONUS_PROBA_THRESHOLD, bonus_probability

    prediction = int(model.predict(vector)[0])
    return prediction == config.LABEL_BONUS, None


def main() -> int:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {config.CAMERA_INDEX}. Check config.py.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    left_key = resolve_key(config.MOVE_LEFT_KEY)
    right_key = resolve_key(config.MOVE_RIGHT_KEY)
    jump_key = resolve_key(config.JUMP_KEY)
    bonus_key = resolve_key(config.BONUS_KEY)
    keyboard = KeyboardManager()
    bonus_model = load_bonus_model()
    movement_smoother = StableAction(config.ACTION_CONFIRM_FRAMES)
    jump_hold = ConfirmedHold(config.JUMP_CONFIRM_FRAMES)
    bonus_debouncer = GestureDebouncer(config.BONUS_CONFIRM_FRAMES, config.BONUS_COOLDOWN_SECONDS)

    mp_face_mesh = mp.solutions.face_mesh
    prev_time = time.perf_counter()
    fps = 0.0
    bonus_display_until = 0.0
    model_status = bonus_model.status
    solo_test_mode = config.SOLO_TEST_MODE

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        ) as face_mesh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("ERROR: Cannot read frame from camera.")
                    break

                now = time.perf_counter()

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                faces = sort_faces_left_to_right(results.multi_face_landmarks or [])
                player1, player2, solo_active = assign_player_faces(faces, solo_test_mode)

                p1_status = "missing"
                p2_status = "missing"
                p1_action = IDLE
                jump_key_held = False
                bonus_status = "MODEL NOT LOADED" if bonus_model.model is None else "READY"
                p1_head_yaw = None
                p2_smile_score = None
                p2_head_pitch = None
                bonus_probability = None

                if solo_active:
                    p1_status = "missing / solo inactive"
                    p2_status = "solo fallback"
                    draw_face_box(frame, player2, "Player 2 - SOLO TEST", (80, 255, 80))
                elif player1 is not None and player2 is not None:
                    p1_status = "detected"
                    p2_status = "detected"
                    draw_face_box(frame, player1, "Player 1 - movement", (0, 220, 255))
                    draw_face_box(frame, player2, "Player 2 - jump/bonus", (80, 255, 80))

                if player1 is not None:
                    p1_features = extract_feature_dict(player1)
                    p1_head_yaw = p1_features["head_yaw"]
                    raw_movement = movement_from_head_yaw(p1_head_yaw, movement_smoother.stable)
                    p1_action = movement_smoother.update(raw_movement)
                    keyboard.set_movement(p1_action, left_key, right_key)
                else:
                    keyboard.set_movement(IDLE, left_key, right_key)
                    movement_smoother.reset()

                if player2 is not None:
                    p2_features = extract_feature_dict(player2)
                    p2_feature_vector = extract_features(player2)
                    p2_smile_score = estimate_smile_score(p2_features)
                    p2_head_pitch = p2_features["head_pitch"]
                    jump_raw = is_jump_gesture(p2_features)
                    jump_key_held = jump_hold.update(jump_raw)
                    keyboard.set_hold(jump_key, jump_key_held)

                    if bonus_model.model is not None:
                        try:
                            bonus_raw, bonus_probability = predict_bonus(bonus_model.model, p2_feature_vector)
                            if bonus_debouncer.update(bonus_raw, now):
                                tap_key(keyboard, bonus_key, duration=config.KEY_TAP_SECONDS)
                                bonus_display_until = now + 0.35
                            cooldown = bonus_debouncer.cooldown_left(now)
                            if now < bonus_display_until:
                                bonus_status = "ACTIVE"
                            elif cooldown > 0:
                                bonus_status = f"COOLDOWN {cooldown:.1f}s"
                            else:
                                bonus_status = "READY"
                        except Exception as exc:
                            bonus_model = BonusModel(None, f"model not loaded: prediction error: {exc}")
                            model_status = bonus_model.status
                            bonus_status = "MODEL NOT LOADED"
                    else:
                        bonus_debouncer.reset()
                else:
                    keyboard.release_all()
                    keyboard.release(jump_key)
                    keyboard.release(bonus_key)
                    jump_hold.reset()
                    bonus_debouncer.reset()
                    if len(faces) == 1:
                        draw_face_box(frame, faces[0], "Only one player detected", (0, 180, 255))

                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now

                warning = ""
                warning_color = (255, 255, 255)
                if solo_active:
                    warning = "SOLO TEST MODE: single face used as Player 2"
                    warning_color = (80, 255, 80)
                elif len(faces) == 1:
                    warning = "Only one player detected - keys released"
                    warning_color = (0, 180, 255)
                elif len(faces) == 0:
                    warning = "No face detected - keys released"
                    warning_color = (0, 80, 255)

                p1_yaw_text = "n/a" if p1_head_yaw is None else f"{p1_head_yaw:.3f}"
                p2_smile_text = "n/a" if p2_smile_score is None else f"{p2_smile_score:.3f}"
                p2_pitch_text = "n/a" if p2_head_pitch is None else f"{p2_head_pitch:.3f}"
                bonus_prob_text = "n/a" if bonus_probability is None else f"{bonus_probability:.2f}"
                model_loaded_text = "yes" if bonus_model.model is not None else "no"
                if p1_action == LEFT:
                    movement_key_text = "LEFT HELD"
                elif p1_action == RIGHT:
                    movement_key_text = "RIGHT HELD"
                else:
                    movement_key_text = "RELEASED"
                jump_key_text = "HELD" if jump_key_held else "RELEASED"

                lines = [
                    (f"FPS: {fps:.1f}", (255, 255, 255)),
                    (f"Faces detected: {len(faces)}", (255, 255, 255)),
                ]
                if warning:
                    lines.append((warning, warning_color))
                lines.extend(
                    [
                        (f"Player 1 status: {p1_status}", (0, 220, 255) if p1_status == "detected" else (0, 180, 255)),
                        (f"Movement key: {movement_key_text}", (0, 220, 255)),
                        (f"Player 1 head_yaw: {p1_yaw_text}", (0, 220, 255)),
                        (
                            f"Player 2 status: {p2_status}",
                            (80, 255, 80) if p2_status in ("detected", "solo fallback") else (0, 180, 255),
                        ),
                        (f"Jump key: {jump_key_text}", (80, 255, 80)),
                        (f"Player 2 smile_score: {p2_smile_text}", (80, 255, 80)),
                        (f"Player 2 head_pitch: {p2_pitch_text}", (80, 255, 80)),
                        (f"Bonus key: {str(config.BONUS_KEY).upper()}", (255, 220, 80)),
                        (f"Bonus status: {bonus_status} | proba: {bonus_prob_text}", (255, 220, 80)),
                        (f"Model loaded: {model_loaded_text} | {model_status}", (230, 230, 230)),
                        (f"Keyboard: {keyboard.status()}", (230, 230, 230)),
                        ("t = toggle solo test | q = quit", (230, 230, 230)),
                    ]
                )
                put_lines(frame, lines)

                cv2.imshow("Blobby Face Controller - realtime", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("t"):
                    solo_test_mode = not solo_test_mode
                    print(f"Solo test mode: {'ON' if solo_test_mode else 'OFF'}")
                    if not solo_test_mode:
                        keyboard.release_all()
                        movement_smoother.reset()
                        jump_hold.reset()
                        bonus_debouncer.reset()

    finally:
        keyboard.release_all()
        for key in (left_key, right_key, jump_key, bonus_key):
            keyboard.release(key)
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
