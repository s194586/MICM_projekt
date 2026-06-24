"""One-player low-latency Blobby controller based on OpenCV YuNet."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:  # pragma: no cover - Windows-only interactive path.
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback.
    msvcrt = None

from capture import LatestFrameCamera
from detector import DEFAULT_YUNET_MODEL, create_face_detector
from gestures import CooldownTap, HysteresisHold, NeutralCalibrator, compute_live_signals, movement_from_signal, select_jump_signal
from keyboard_backend import KeyboardController


LEFT = "LEFT"
RIGHT = "RIGHT"
IDLE = "IDLE"
WINDOW_NAME = "Blobby Face Controller - YuNet Fast"

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 60
DEFAULT_CAMERA_BACKEND = "auto"
DEFAULT_DETECTOR = "yunet"
DEFAULT_KEYBOARD_BACKEND = "win32"
DEFAULT_OVERLAY = True
DEFAULT_CALIBRATION_SECONDS = 1.2
DEFAULT_JUMP_MODE = "mouth_open"

DEFAULT_MOVE_ENTER_THRESHOLD = 0.075
DEFAULT_MOVE_EXIT_THRESHOLD = 0.035

MOUTH_OPEN_ENTER_THRESHOLD = 0.060
MOUTH_OPEN_EXIT_THRESHOLD = 0.030
VERTICAL_HEAD_ENTER_THRESHOLD = 0.070
VERTICAL_HEAD_EXIT_THRESHOLD = 0.035

DEFAULT_BONUS_ENTER_THRESHOLD = 0.065
DEFAULT_BONUS_EXIT_THRESHOLD = 0.032
DEFAULT_BONUS_COOLDOWN_SECONDS = 0.9
DEFAULT_SPACE_TAP_SECONDS = 0.03
SUMMARY_INTERVAL_SECONDS = 2.0

MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"
JUMP_KEY = "w"
BONUS_KEY = "space"


def _ewma(previous: float, current: float, alpha: float = 0.18) -> float:
    if previous <= 0.0:
        return current
    return previous + alpha * (current - previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-player low-latency Blobby controller with OpenCV YuNet.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--camera-backend", choices=("auto", "msmf", "dshow", "default"), default=DEFAULT_CAMERA_BACKEND)
    parser.add_argument("--detector", choices=("yunet",), default=DEFAULT_DETECTOR)
    parser.add_argument("--keyboard", choices=("win32", "pynput"), default=DEFAULT_KEYBOARD_BACKEND)
    parser.add_argument("--jump-mode", choices=("mouth_open", "vertical_head"), default=DEFAULT_JUMP_MODE)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_YUNET_MODEL)
    return parser


def jump_thresholds_for_mode(jump_mode: str) -> tuple[float, float]:
    if jump_mode == "mouth_open":
        return MOUTH_OPEN_ENTER_THRESHOLD, MOUTH_OPEN_EXIT_THRESHOLD
    return VERTICAL_HEAD_ENTER_THRESHOLD, VERTICAL_HEAD_EXIT_THRESHOLD


def draw_overlay(frame, lines: list[tuple[str, tuple[int, int, int]]], face) -> None:
    if face is not None:
        x, y, w, h = [int(round(value)) for value in face.bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 255), 1)
        if face.landmarks is not None:
            for point in face.landmarks:
                cv2.circle(frame, (int(round(point[0])), int(round(point[1]))), 1, (80, 255, 80), -1)

    y = 22
    for text, color in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        y += 18


def hide_overlay_window() -> None:
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


def poll_runtime_key(overlay_enabled: bool) -> str | None:
    if overlay_enabled:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            return chr(key).lower()
        return None

    if msvcrt is not None and msvcrt.kbhit():  # pragma: no branch - platform specific.
        return msvcrt.getwch().lower()
    return None


def apply_recalibration(
    calibrator: NeutralCalibrator,
    jump_hold: HysteresisHold,
    bonus_tap: CooldownTap,
    keyboard: KeyboardController,
    now: float,
) -> None:
    calibrator.reset(now)
    jump_hold.reset()
    bonus_tap.reset()
    keyboard.release_all()


def main() -> int:
    args = build_parser().parse_args()
    overlay_enabled = DEFAULT_OVERLAY and not args.no_overlay
    jump_enter_threshold, jump_exit_threshold = jump_thresholds_for_mode(args.jump_mode)

    try:
        detector = create_face_detector(args.detector, input_size=(args.width, args.height), model_path=args.model_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        print(f"Expected YuNet model path: {args.model_path}")
        return 1

    camera = LatestFrameCamera(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_backend=args.camera_backend,
    )
    if not camera.start():
        print(f"ERROR: {camera.error}")
        return 1

    keyboard = KeyboardController(preferred_backend=args.keyboard)
    calibrator = NeutralCalibrator(duration_seconds=DEFAULT_CALIBRATION_SECONDS)
    jump_hold = HysteresisHold(jump_enter_threshold, jump_exit_threshold)
    bonus_tap = CooldownTap(DEFAULT_BONUS_COOLDOWN_SECONDS)

    calibration = None
    current_move = IDLE
    current_face = None
    face_detected = False
    jump_state = False
    mouth_open_delta = 0.0

    last_sequence = -1
    last_report_at = 0.0
    fps = 0.0
    latency_ms = 0.0
    capture_ms = 0.0
    inference_ms = 0.0
    processing_ms = 0.0

    print(
        f"Controller started | capture={camera.backend_name} | detector={detector.backend_name} "
        f"| jump={args.jump_mode} | keyboard={keyboard.status_text()}"
    )
    print(f"YuNet model path: {args.model_path}")
    print("Keep a neutral face with closed mouth for the first 1-2 seconds, or press c later to recalibrate.")
    if not overlay_enabled:
        print("Overlay disabled. Use q/c/o in the console, or Ctrl+C to stop.")

    try:
        while True:
            loop_started_at = time.perf_counter()
            keyboard.update(loop_started_at)

            sequence, frame, frame_timestamp, raw_capture_ms = camera.latest()
            if frame is None or sequence == last_sequence:
                key = poll_runtime_key(overlay_enabled)
                if key == "q":
                    break
                if key == "o":
                    overlay_enabled = not overlay_enabled
                    if not overlay_enabled:
                        hide_overlay_window()
                    print(f"Overlay {'enabled' if overlay_enabled else 'disabled'}.")
                time.sleep(0.001)
                continue

            last_sequence = sequence
            capture_ms = _ewma(capture_ms, raw_capture_ms)
            latency_ms = _ewma(latency_ms, max(0.0, (loop_started_at - frame_timestamp) * 1000.0))

            frame = cv2.flip(frame, 1)
            current_face, raw_inference_ms = detector.detect(frame)
            inference_ms = _ewma(inference_ms, raw_inference_ms)
            face_detected = current_face is not None

            processing_started_at = time.perf_counter()
            move_text = IDLE
            jump_text = "RELEASED"
            bonus_text = "READY"
            calibrated_text = "no"
            mouth_open_delta = 0.0

            if face_detected:
                if calibration is None:
                    calibrator.add_sample(current_face, frame)
                    if calibrator.is_ready(loop_started_at):
                        calibration = calibrator.finalize()
                        if calibration is not None:
                            print("Neutral calibration captured.")
                else:
                    signals = compute_live_signals(frame, current_face, calibration)
                    current_move = movement_from_signal(
                        signals.movement,
                        current_move,
                        enter_threshold=DEFAULT_MOVE_ENTER_THRESHOLD,
                        exit_threshold=DEFAULT_MOVE_EXIT_THRESHOLD,
                    )
                    keyboard.set_movement(current_move, MOVE_LEFT_KEY, MOVE_RIGHT_KEY)

                    mouth_open_delta = signals.mouth_open
                    jump_state = jump_hold.update(select_jump_signal(signals, args.jump_mode))
                    keyboard.set_hold(JUMP_KEY, jump_state)

                    bonus_active = signals.bonus >= DEFAULT_BONUS_ENTER_THRESHOLD
                    if signals.bonus <= DEFAULT_BONUS_EXIT_THRESHOLD:
                        bonus_tap.reset()
                    if bonus_tap.update(bonus_active, loop_started_at):
                        keyboard.tap(BONUS_KEY, loop_started_at, DEFAULT_SPACE_TAP_SECONDS)

                if calibration is None:
                    current_move = IDLE
                    jump_state = False
                    jump_hold.reset()
                    keyboard.release_all()
                calibrated_text = "yes" if calibration is not None else "no"
            else:
                current_move = IDLE
                jump_state = False
                mouth_open_delta = 0.0
                jump_hold.reset()
                bonus_tap.reset()
                keyboard.release_all()

            cooldown_left = bonus_tap.cooldown_left(loop_started_at)
            bonus_text = f"COOLDOWN {cooldown_left:.1f}s" if cooldown_left > 0.0 and not bonus_tap.armed else "READY"
            move_text = current_move if calibration is not None else "CALIBRATING"
            jump_text = "HELD" if jump_state else "RELEASED"

            processing_finished_at = time.perf_counter()
            processing_ms = _ewma(processing_ms, (processing_finished_at - processing_started_at) * 1000.0)
            fps = _ewma(fps, 1.0 / max(processing_finished_at - loop_started_at, 1e-6))

            if overlay_enabled:
                draw_overlay(
                    frame,
                    [
                        (f"FPS: {fps:.1f}", (255, 255, 255)),
                        (f"Detector: {detector.backend_name}", (230, 230, 230)),
                        (f"Inference: {inference_ms:.1f} ms", (230, 230, 230)),
                        (f"Face: {'yes' if face_detected else 'no'}", (80, 255, 80) if face_detected else (0, 80, 255)),
                        (f"Move: {move_text}", (0, 220, 255)),
                        (f"Jump: {jump_text}", (80, 255, 80)),
                        (f"Mouth: {mouth_open_delta:+.3f}", (255, 220, 80)),
                        (f"Bonus: {bonus_text}", (255, 220, 80)),
                        (f"Calibrated: {calibrated_text}", (230, 230, 230)),
                        ("q quit | c recalibrate | o overlay", (230, 230, 230)),
                    ],
                    current_face,
                )
                cv2.imshow(WINDOW_NAME, frame)
            elif loop_started_at - last_report_at >= SUMMARY_INTERVAL_SECONDS:
                print(
                    f"fps={fps:5.1f} infer_ms={inference_ms:5.1f} latency_ms={latency_ms:5.1f} "
                    f"capture_ms={capture_ms:5.1f} proc_ms={processing_ms:5.1f} "
                    f"face={'yes' if face_detected else 'no'} move={move_text:<11} "
                    f"jump={jump_text:<8} mouth={mouth_open_delta:+.3f} bonus={bonus_text:<12} calibrated={calibrated_text}"
                )
                last_report_at = loop_started_at

            key = poll_runtime_key(overlay_enabled)
            if key == "q":
                break
            if key == "c":
                calibration = None
                current_move = IDLE
                jump_state = False
                mouth_open_delta = 0.0
                apply_recalibration(calibrator, jump_hold, bonus_tap, keyboard, loop_started_at)
                print("Recalibrating neutral face...")
            if key == "o":
                overlay_enabled = not overlay_enabled
                if not overlay_enabled:
                    hide_overlay_window()
                print(f"Overlay {'enabled' if overlay_enabled else 'disabled'}.")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        keyboard.release_all()
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
