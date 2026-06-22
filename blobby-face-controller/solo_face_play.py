"""Experimental single-face controller for playing one full Blobby character."""

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
)
from keyboard_utils import resolve_key, tap_key

try:
    from pynput.keyboard import Controller

    PYNPUT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the local desktop.
    Controller = None
    PYNPUT_IMPORT_ERROR = exc


LEFT = "LEFT"
RIGHT = "RIGHT"
IDLE = "IDLE"
JUMP_CONFIRM_FRAMES = getattr(config, "JUMP_CONFIRM_FRAMES", 2)
BONUS_CONFIRM_FRAMES = getattr(config, "BONUS_CONFIRM_FRAMES", 3)
BONUS_COOLDOWN_SECONDS = getattr(config, "BONUS_COOLDOWN_SECONDS", 1.5)


class StableAction:
    """Accept a movement action after it remains stable for several frames."""

    def __init__(self, confirm_frames: int):
        self.confirm_frames = max(1, confirm_frames)
        self.candidate = IDLE
        self.candidate_frames = 0
        self.stable = IDLE

    def update(self, action: str) -> str:
        if action == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = action
            self.candidate_frames = 1
        if self.candidate_frames >= self.confirm_frames:
            self.stable = self.candidate
        return self.stable

    def reset(self) -> None:
        self.candidate = IDLE
        self.candidate_frames = 0
        self.stable = IDLE


class GestureDebouncer:
    """Confirm a one-shot gesture and prevent repeated key taps."""

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

        if not self.armed or self.true_frames < self.confirm_frames:
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
    """Hold movement/jump keys without repeating keyDown every frame."""

    def __init__(self):
        self.controller = None
        self.enabled = False
        self.error = str(PYNPUT_IMPORT_ERROR) if PYNPUT_IMPORT_ERROR else ""
        self.held_keys: set[object] = set()

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
        return BonusModel(None, "model not loaded")
    try:
        payload = joblib.load(config.MODEL_PATH)
    except Exception as exc:
        return BonusModel(None, f"model not loaded: {exc}")

    if not isinstance(payload, dict):
        return BonusModel(None, "model not loaded: invalid model file")
    model = payload.get("model")
    if model is None:
        return BonusModel(None, "model not loaded: invalid model file")
    if payload.get("feature_names") != FEATURE_NAMES:
        return BonusModel(None, "model not loaded: feature schema mismatch")
    if payload.get("target_gesture") != config.BONUS_GESTURE_ID:
        return BonusModel(None, "model not loaded: retrain head-down model")
    return BonusModel(model, "model OK")


def predict_bonus(model, feature_vector: np.ndarray) -> tuple[bool, float | None]:
    vector = feature_vector.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        classes = list(getattr(model, "classes_", []))
        if config.LABEL_BONUS not in classes:
            prediction = int(model.predict(vector)[0])
            return prediction == config.LABEL_BONUS, None
        bonus_index = classes.index(config.LABEL_BONUS)
        probability = float(probabilities[bonus_index])
        return probability >= config.BONUS_PROBA_THRESHOLD, probability
    prediction = int(model.predict(vector)[0])
    return prediction == config.LABEL_BONUS, None


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


def draw_face_box(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    min_x, min_y, max_x, max_y = face_bbox(landmarks)
    start = (int(min_x * width), int(min_y * height))
    end = (int(max_x * width), int(max_y * height))
    cv2.rectangle(frame, start, end, (80, 255, 80), 2)
    cv2.putText(
        frame,
        "Solo face - full control",
        (start[0], max(24, start[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 255, 80),
        2,
    )


def put_lines(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        y += 25


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
    jump_hold = ConfirmedHold(JUMP_CONFIRM_FRAMES)
    bonus_debouncer = GestureDebouncer(BONUS_CONFIRM_FRAMES, BONUS_COOLDOWN_SECONDS)

    prev_time = time.perf_counter()
    fps = 0.0
    bonus_display_until = 0.0
    mp_face_mesh = mp.solutions.face_mesh

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
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
                face = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

                movement_action = IDLE
                jump_key_held = False
                bonus_status = "MODEL NOT LOADED" if bonus_model.model is None else "READY"
                head_yaw = None
                smile_score = None
                bonus_probability = None

                if face is not None:
                    draw_face_box(frame, face)
                    feature_dict = extract_feature_dict(face)
                    feature_vector = extract_features(face)
                    head_yaw = feature_dict["head_yaw"]
                    smile_score = estimate_smile_score(feature_dict)

                    movement_target = movement_from_head_yaw(head_yaw, movement_smoother.stable)
                    movement_action = movement_smoother.update(movement_target)
                    keyboard.set_movement(movement_action, left_key, right_key)

                    jump_key_held = jump_hold.update(smile_score >= config.SMILE_THRESHOLD)
                    keyboard.set_hold(jump_key, jump_key_held)

                    if bonus_model.model is not None:
                        try:
                            bonus_raw, bonus_probability = predict_bonus(bonus_model.model, feature_vector)
                            if bonus_debouncer.update(bonus_raw, now):
                                tap_key(keyboard, bonus_key, duration=config.KEY_TAP_SECONDS)
                                bonus_display_until = now + 0.35
                            bonus_cooldown = bonus_debouncer.cooldown_left(now)
                            if now < bonus_display_until:
                                bonus_status = "ACTIVE"
                            elif bonus_cooldown > 0:
                                bonus_status = f"COOLDOWN {bonus_cooldown:.1f}s"
                        except Exception as exc:
                            bonus_model = BonusModel(None, f"model not loaded: prediction error: {exc}")
                            bonus_status = "MODEL NOT LOADED"
                    else:
                        bonus_debouncer.reset()
                else:
                    keyboard.release_all()
                    keyboard.release(jump_key)
                    keyboard.release(bonus_key)
                    movement_smoother.reset()
                    jump_hold.reset()
                    bonus_debouncer.reset()

                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now

                yaw_text = "n/a" if head_yaw is None else f"{head_yaw:.3f}"
                smile_text = "n/a" if smile_score is None else f"{smile_score:.3f}"
                probability_text = "n/a" if bonus_probability is None else f"{bonus_probability:.2f}"
                detected_text = "yes" if face is not None else "no - keys released"
                if movement_action == LEFT:
                    movement_key_text = "LEFT HELD"
                elif movement_action == RIGHT:
                    movement_key_text = "RIGHT HELD"
                else:
                    movement_key_text = "RELEASED"
                jump_key_text = "HELD" if jump_key_held else "RELEASED"

                put_lines(
                    frame,
                    [
                        (f"FPS: {fps:.1f}", (255, 255, 255)),
                        ("SOLO FACE PLAY", (80, 255, 80)),
                        (f"Detected face: {detected_text}", (80, 255, 80) if face is not None else (0, 80, 255)),
                        (f"Movement key: {movement_key_text}", (0, 220, 255)),
                        (f"head_yaw: {yaw_text}", (0, 220, 255)),
                        (f"Jump key: {jump_key_text}", (80, 255, 80)),
                        (f"smile_score: {smile_text}", (80, 255, 80)),
                        (f"Bonus key: {str(config.BONUS_KEY).upper()}", (255, 220, 80)),
                        (f"Bonus status: {bonus_status}", (255, 220, 80)),
                        (f"Bonus probability: {probability_text}", (255, 220, 80)),
                        (f"Model: {bonus_model.status}", (230, 230, 230)),
                        (f"Keyboard: {keyboard.status()}", (230, 230, 230)),
                        ("q = quit", (230, 230, 230)),
                    ],
                )

                cv2.imshow("Blobby Face Controller - SOLO FACE PLAY", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        keyboard.release_all()
        for key in (left_key, right_key, jump_key, bonus_key):
            keyboard.release(key)
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
