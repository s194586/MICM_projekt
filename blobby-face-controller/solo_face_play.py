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

try:
    from pynput.keyboard import Controller, Key

    PYNPUT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the local desktop.
    Controller = None
    Key = None
    PYNPUT_IMPORT_ERROR = exc


LEFT = "LEFT"
RIGHT = "RIGHT"
IDLE = "IDLE"
JUMP_CONFIRM_FRAMES = getattr(config, "JUMP_CONFIRM_FRAMES", 2)
JUMP_COOLDOWN_SECONDS = getattr(config, "JUMP_COOLDOWN_SECONDS", 0.35)
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


class KeyboardManager:
    """Hold movement keys and safely release one-shot key taps."""

    def __init__(self):
        self.controller = None
        self.enabled = False
        self.error = str(PYNPUT_IMPORT_ERROR) if PYNPUT_IMPORT_ERROR else ""
        self.held_keys: set[str] = set()
        self.tap_release_at: dict[str, float] = {}

        if Controller is not None and PYNPUT_IMPORT_ERROR is None:
            try:
                self.controller = Controller()
                self.enabled = True
            except Exception as exc:  # pragma: no cover - desktop dependent.
                self.error = str(exc)

    @staticmethod
    def parse_key(key_name: str):
        if Key is not None and hasattr(Key, key_name):
            return getattr(Key, key_name)
        return key_name

    def press(self, key_name: str) -> None:
        if not self.enabled or self.controller is None:
            return
        try:
            self.controller.press(self.parse_key(key_name))
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def release(self, key_name: str) -> None:
        if not self.enabled or self.controller is None:
            return
        try:
            self.controller.release(self.parse_key(key_name))
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def hold(self, key_name: str) -> None:
        if key_name not in self.held_keys:
            self.press(key_name)
            self.held_keys.add(key_name)

    def release_hold(self, key_name: str) -> None:
        if key_name in self.held_keys:
            self.held_keys.remove(key_name)
            if key_name not in self.tap_release_at:
                self.release(key_name)

    def set_movement(self, action: str) -> None:
        if action == LEFT:
            self.release_hold(config.MOVE_RIGHT_KEY)
            self.hold(config.MOVE_LEFT_KEY)
        elif action == RIGHT:
            self.release_hold(config.MOVE_LEFT_KEY)
            self.hold(config.MOVE_RIGHT_KEY)
        else:
            self.release_hold(config.MOVE_LEFT_KEY)
            self.release_hold(config.MOVE_RIGHT_KEY)

    def tap(self, key_name: str, now: float) -> None:
        self.press(key_name)
        release_at = now + config.KEY_TAP_SECONDS
        self.tap_release_at[key_name] = max(self.tap_release_at.get(key_name, 0.0), release_at)

    def update_taps(self, now: float) -> None:
        for key_name, release_at in list(self.tap_release_at.items()):
            if now >= release_at:
                del self.tap_release_at[key_name]
                if key_name not in self.held_keys:
                    self.release(key_name)

    def release_all(self) -> None:
        for key_name in set(self.held_keys) | set(self.tap_release_at):
            self.release(key_name)
        self.held_keys.clear()
        self.tap_release_at.clear()

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


def movement_from_head_yaw(head_yaw: float) -> str:
    if head_yaw < config.HEAD_YAW_LEFT_THRESHOLD:
        return LEFT
    if head_yaw > config.HEAD_YAW_RIGHT_THRESHOLD:
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

    keyboard = KeyboardManager()
    bonus_model = load_bonus_model()
    movement_smoother = StableAction(config.ACTION_CONFIRM_FRAMES)
    jump_debouncer = GestureDebouncer(JUMP_CONFIRM_FRAMES, JUMP_COOLDOWN_SECONDS)
    bonus_debouncer = GestureDebouncer(BONUS_CONFIRM_FRAMES, BONUS_COOLDOWN_SECONDS)

    prev_time = time.perf_counter()
    fps = 0.0
    jump_display_until = 0.0
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
                keyboard.update_taps(now)

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                face = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

                movement_action = IDLE
                jump_status = "READY"
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

                    movement_action = movement_smoother.update(movement_from_head_yaw(head_yaw))
                    keyboard.set_movement(movement_action)

                    if jump_debouncer.update(smile_score > config.SMILE_THRESHOLD, now):
                        keyboard.tap(config.JUMP_KEY, now)
                        jump_display_until = now + 0.25
                    jump_cooldown = jump_debouncer.cooldown_left(now)
                    if now < jump_display_until:
                        jump_status = "JUMP"
                    elif jump_cooldown > 0:
                        jump_status = f"COOLDOWN {jump_cooldown:.1f}s"

                    if bonus_model.model is not None:
                        try:
                            bonus_raw, bonus_probability = predict_bonus(bonus_model.model, feature_vector)
                            if bonus_debouncer.update(bonus_raw, now):
                                keyboard.tap(config.BONUS_KEY, now)
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
                    movement_smoother.reset()
                    jump_debouncer.reset()
                    bonus_debouncer.reset()

                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now

                yaw_text = "n/a" if head_yaw is None else f"{head_yaw:.3f}"
                smile_text = "n/a" if smile_score is None else f"{smile_score:.3f}"
                probability_text = "n/a" if bonus_probability is None else f"{bonus_probability:.2f}"
                detected_text = "yes" if face is not None else "no - keys released"

                put_lines(
                    frame,
                    [
                        (f"FPS: {fps:.1f}", (255, 255, 255)),
                        ("SOLO FACE PLAY", (80, 255, 80)),
                        (f"Detected face: {detected_text}", (80, 255, 80) if face is not None else (0, 80, 255)),
                        (f"Action movement: {movement_action}", (0, 220, 255)),
                        (f"head_yaw: {yaw_text}", (0, 220, 255)),
                        (f"Jump: {jump_status}", (80, 255, 80)),
                        (f"smile_score: {smile_text}", (80, 255, 80)),
                        (f"Bonus: {bonus_status}", (255, 220, 80)),
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
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
