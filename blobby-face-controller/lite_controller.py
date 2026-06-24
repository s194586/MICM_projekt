"""Ultra-light non-MediaPipe experimental controller for Blobby Online."""

from __future__ import annotations

import argparse
import time

import cv2

import config
from lite_capture import LatestFrameCamera
from lite_gestures import FaceTrackerBackend, NeutralCalibrator, movement_from_offset, normalized_offsets, pick_jump_score
from lite_keyboard import KeyboardController


LEFT = "LEFT"
RIGHT = "RIGHT"
IDLE = "IDLE"

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 60
FAST_OVERLAY = True
KEYBOARD_BACKEND = "win32"
JUMP_MODE = "mouth_roi_motion"
BONUS_MODE = "rule"
MOVE_ENTER_THRESHOLD = 0.18
MOVE_EXIT_THRESHOLD = 0.08
JUMP_ENTER_THRESHOLD = 0.085
JUMP_EXIT_THRESHOLD = 0.055
BONUS_ENTER_THRESHOLD = 0.095
BONUS_EXIT_THRESHOLD = 0.050
BONUS_COOLDOWN_SECONDS = 1.0
SPACE_TAP_SECONDS = 0.03
CALIBRATION_SECONDS = 1.2
REDETECT_EVERY_FRAMES = 8


def _ewma(previous: float, current: float, alpha: float = 0.18) -> float:
    if previous <= 0.0:
        return current
    return previous + alpha * (current - previous)


class HysteresisHold:
    """Stateful hold with separate enter and exit thresholds."""

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


class BonusTapState:
    """One-shot tap gate with cooldown."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite non-MediaPipe experimental controller.")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--jump-mode", choices=("mouth_roi_motion", "smile_cascade", "face_up"), default=JUMP_MODE)
    parser.add_argument("--keyboard", choices=("win32", "pynput"), default=KEYBOARD_BACKEND)
    parser.add_argument("--no-overlay", action="store_true")
    return parser


def draw_overlay(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 24
    for text, color in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 19


def main() -> int:
    args = build_parser().parse_args()
    overlay_enabled = FAST_OVERLAY and not args.no_overlay

    keyboard = KeyboardController(preferred_backend=args.keyboard)
    camera = LatestFrameCamera(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        prefer_dshow=True,
    )
    if not camera.start():
        print(f"ERROR: {camera.error}")
        return 1

    tracker = FaceTrackerBackend(detect_every=REDETECT_EVERY_FRAMES)
    calibrator = NeutralCalibrator(duration_seconds=CALIBRATION_SECONDS)
    calibration = None
    movement_state = IDLE
    jump_hold = HysteresisHold(JUMP_ENTER_THRESHOLD, JUMP_EXIT_THRESHOLD)
    bonus_tap = BonusTapState(BONUS_COOLDOWN_SECONDS)

    last_sequence = -1
    last_report_at = 0.0
    fps = 0.0
    latency_ms = 0.0
    capture_ms = 0.0
    processing_ms = 0.0
    face_detected = False
    tracker_mode = "miss"
    jump_score = 0.0

    print(
        f"Lite controller started | capture={camera.backend_name} | detect={tracker.backend_name} "
        f"| jump={args.jump_mode} | keyboard={keyboard.status_text()}"
    )
    print("Keep a neutral face for the first second, or press c later to recalibrate.")
    if not overlay_enabled:
        print("Overlay disabled. Use Ctrl+C to stop.")

    try:
        while True:
            loop_started_at = time.perf_counter()
            keyboard.update(loop_started_at)

            sequence, frame, frame_timestamp, raw_capture_ms = camera.latest()
            if frame is None or sequence == last_sequence:
                if overlay_enabled and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.001)
                continue

            last_sequence = sequence
            capture_ms = _ewma(capture_ms, raw_capture_ms)
            latency_ms = _ewma(latency_ms, max(0.0, (loop_started_at - frame_timestamp) * 1000.0))

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processing_started_at = time.perf_counter()

            rect, tracker_mode = tracker.update(frame, gray)
            face_detected = rect is not None

            if face_detected:
                if calibration is None:
                    calibrator.add_sample(gray, rect)
                    if calibrator.is_ready(loop_started_at):
                        calibration = calibrator.finalize()
                        if calibration is not None:
                            print("Neutral calibration captured.")
                else:
                    offset_x, offset_y = normalized_offsets(rect, calibration)
                    movement_state = movement_from_offset(
                        offset_x,
                        movement_state,
                        enter_threshold=MOVE_ENTER_THRESHOLD,
                        exit_threshold=MOVE_EXIT_THRESHOLD,
                    )
                    keyboard.set_movement(movement_state, config.MOVE_LEFT_KEY, config.MOVE_RIGHT_KEY)

                    jump_score = pick_jump_score(
                        gray,
                        rect,
                        calibration,
                        jump_mode=args.jump_mode,
                        smile_cascade=tracker.smile_cascade,
                    )
                    jump_active = jump_hold.update(jump_score)
                    keyboard.set_hold(config.JUMP_KEY, jump_active)

                    bonus_active = offset_y >= BONUS_ENTER_THRESHOLD
                    if offset_y <= BONUS_EXIT_THRESHOLD:
                        bonus_tap.reset()
                    if bonus_tap.update(bonus_active, loop_started_at):
                        keyboard.tap(config.BONUS_KEY, loop_started_at, SPACE_TAP_SECONDS)
                if calibration is None:
                    movement_state = IDLE
                    jump_hold.reset()
                    keyboard.release_all()
            else:
                movement_state = IDLE
                jump_score = 0.0
                jump_hold.reset()
                bonus_tap.reset()
                keyboard.release_all()

            processing_finished_at = time.perf_counter()
            processing_ms = _ewma(processing_ms, (processing_finished_at - processing_started_at) * 1000.0)
            fps = _ewma(fps, 1.0 / max(processing_finished_at - loop_started_at, 1e-6))

            if calibration is None:
                jump_text = "CAL"
                bonus_text = "CAL"
                move_text = "CAL"
            else:
                jump_text = "HELD" if jump_hold.held else "RELEASED"
                cooldown_left = bonus_tap.cooldown_left(loop_started_at)
                bonus_text = f"COOLDOWN {cooldown_left:.1f}s" if cooldown_left > 0.0 and not bonus_tap.armed else "READY"
                move_text = movement_state

            if overlay_enabled:
                draw_overlay(
                    frame,
                    [
                        (f"FPS: {fps:.1f}", (255, 255, 255)),
                        (f"Backend: {tracker.backend_name}", (220, 220, 220)),
                        (f"Face: {'yes' if face_detected else 'no'} {tracker_mode}", (80, 255, 80) if face_detected else (0, 80, 255)),
                        (f"Move: {move_text}", (0, 220, 255)),
                        (f"Jump: {jump_text}", (80, 255, 80)),
                        (f"Bonus: {bonus_text}", (255, 220, 80)),
                        (f"Jump mode: {args.jump_mode}", (220, 220, 220)),
                        (f"Latency: {latency_ms:.1f} ms", (220, 220, 220)),
                        ("q quit | c recal | o overlay | r reset", (220, 220, 220)),
                    ],
                )
                cv2.imshow("Blobby Lite Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    calibration = None
                    calibrator.reset(loop_started_at)
                    movement_state = IDLE
                    jump_hold.reset()
                    bonus_tap.reset()
                    keyboard.release_all()
                    print("Recalibrating neutral face...")
                if key == ord("o"):
                    overlay_enabled = False
                    cv2.destroyAllWindows()
                    print("Overlay disabled. Use Ctrl+C to stop.")
                if key == ord("r"):
                    tracker.reset()
                    print("Face tracking reset.")
            else:
                if loop_started_at - last_report_at >= 2.0:
                    print(
                        f"fps={fps:5.1f} latency_ms={latency_ms:5.1f} capture_ms={capture_ms:5.1f} "
                        f"proc_ms={processing_ms:5.1f} face={'yes' if face_detected else 'no'} "
                        f"move={move_text:<5} jump={jump_text:<8} bonus={bonus_text}"
                    )
                    last_report_at = loop_started_at

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        keyboard.release_all()
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
