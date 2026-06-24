"""Final one-player Blobby controller based on YuNet + PFLD mouth landmarks."""

from __future__ import annotations

import argparse
import time

import cv2

try:  # pragma: no cover - Windows-only interactive path.
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback.
    msvcrt = None

from capture import LatestFrameCamera
from detector import DEFAULT_YUNET_MODEL, create_face_detector
from gestures import HysteresisHold, NeutralCalibrator, compute_live_signals, movement_from_signal
from keyboard_backend import KeyboardController
from landmark_detector import DEFAULT_LANDMARK_MODEL, create_landmark_detector


IDLE = "IDLE"
WINDOW_NAME = "Blobby Face Controller - Final"

DEFAULT_WIDTH = 424
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 60
DEFAULT_CAMERA_BACKEND = "auto"
DEFAULT_KEYBOARD_BACKEND = "win32"
DEFAULT_LEFT_KEY = "a"
DEFAULT_RIGHT_KEY = "d"
DEFAULT_JUMP_KEY = "w"
DEFAULT_FOCUS_DELAY_SECONDS = 3.0
DEFAULT_CALIBRATION_SECONDS = 1.0
DEFAULT_OVERLAY_ENABLED = True

MOVE_ENTER_THRESHOLD = 0.075
MOVE_EXIT_THRESHOLD = 0.035
JUMP_ENTER_THRESHOLD = 0.35
JUMP_EXIT_THRESHOLD = 0.20
SUMMARY_INTERVAL_SECONDS = 2.0


def _ewma(previous: float, current: float, alpha: float = 0.18) -> float:
    if previous <= 0.0:
        return current
    return previous + alpha * (current - previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final Blobby controller with YuNet and mouth landmarks.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "msmf", "dshow", "default"), default=DEFAULT_CAMERA_BACKEND)
    parser.add_argument("--keyboard", choices=("win32", "pynput"), default=DEFAULT_KEYBOARD_BACKEND)
    parser.add_argument("--left-key", choices=("a", "left"), default=DEFAULT_LEFT_KEY)
    parser.add_argument("--right-key", choices=("d", "right"), default=DEFAULT_RIGHT_KEY)
    parser.add_argument("--jump-key", choices=("w", "up"), default=DEFAULT_JUMP_KEY)
    parser.add_argument("--focus-delay", type=float, default=DEFAULT_FOCUS_DELAY_SECONDS)
    return parser


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
        if msvcrt is not None and msvcrt.kbhit():  # pragma: no branch - platform specific.
            return msvcrt.getwch().lower()
        return None

    if msvcrt is not None and msvcrt.kbhit():  # pragma: no branch - platform specific.
        return msvcrt.getwch().lower()
    return None


def reset_runtime_state(jump_hold: HysteresisHold, keyboard: KeyboardController) -> None:
    jump_hold.reset()
    keyboard.release_all()


def begin_neutral_calibration(
    calibrator: NeutralCalibrator,
    jump_hold: HysteresisHold,
    keyboard: KeyboardController,
    now: float,
) -> None:
    calibrator.reset(now)
    reset_runtime_state(jump_hold, keyboard)


def main() -> int:
    args = build_parser().parse_args()
    overlay_enabled = DEFAULT_OVERLAY_ENABLED

    try:
        detector = create_face_detector(
            "yunet",
            input_size=(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            model_path=DEFAULT_YUNET_MODEL,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        print(f"Expected YuNet model path: {DEFAULT_YUNET_MODEL}")
        return 1

    try:
        landmark_detector = create_landmark_detector(DEFAULT_LANDMARK_MODEL)
    except Exception as exc:
        print(f"ERROR: {exc}")
        print(f"Missing landmark model. Download/place it here: {DEFAULT_LANDMARK_MODEL}")
        return 1

    camera = LatestFrameCamera(
        camera_index=args.camera_index,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS,
        camera_backend=args.camera_backend,
    )
    if not camera.start():
        print(f"ERROR: {camera.error}")
        return 1

    keyboard = KeyboardController(preferred_backend=args.keyboard)
    neutral_calibrator = NeutralCalibrator(duration_seconds=DEFAULT_CALIBRATION_SECONDS)
    jump_hold = HysteresisHold(JUMP_ENTER_THRESHOLD, JUMP_EXIT_THRESHOLD)

    neutral_calibration = None
    calibration_mode = "neutral"
    current_move = IDLE
    jump_state = False
    face_detected = False
    mouth_ratio = 0.0
    last_sequence = -1
    last_report_at = 0.0
    fps = 0.0
    input_enable_at = time.perf_counter() + max(0.0, float(args.focus_delay))
    input_enabled_announced = False

    print(
        f"Controller started | capture={camera.backend_name} | detector={detector.backend_name} "
        f"| landmark={landmark_detector.backend_name} | keyboard={keyboard.status_text()}"
    )
    print(f"Resolution: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS} fps")
    print("Keep a neutral face with a closed mouth for the first second.")
    if args.focus_delay > 0.0:
        print(f"Click Blobby window now. Input starts in {args.focus_delay:.1f}s.")

    try:
        while True:
            loop_started_at = time.perf_counter()
            keyboard.update(loop_started_at)
            input_enabled = loop_started_at >= input_enable_at
            if input_enabled and not input_enabled_announced:
                print("Input enabled.")
                input_enabled_announced = True

            sequence, frame, _, _ = camera.latest()
            if frame is None or sequence == last_sequence:
                key = poll_runtime_key(overlay_enabled)
                if key == "q":
                    break
                if key == "c":
                    neutral_calibration = None
                    calibration_mode = "neutral"
                    current_move = IDLE
                    jump_state = False
                    begin_neutral_calibration(neutral_calibrator, jump_hold, keyboard, loop_started_at)
                    print("Recalibrating neutral closed mouth...")
                if key == "o":
                    overlay_enabled = not overlay_enabled
                    if not overlay_enabled:
                        hide_overlay_window()
                    print(f"Overlay {'enabled' if overlay_enabled else 'disabled'}.")
                time.sleep(0.001)
                continue

            last_sequence = sequence
            frame = cv2.flip(frame, 1)
            detected_face, _ = detector.detect(frame)
            face_detected = detected_face is not None
            landmark_result = landmark_detector.detect(frame, detected_face) if face_detected else None
            tracking_ready = detected_face is not None and landmark_result is not None

            jump_state = False
            mouth_ratio = 0.0

            if tracking_ready:
                if calibration_mode == "neutral":
                    current_move = IDLE
                    neutral_calibrator.add_sample(detected_face, frame, landmark_result)
                    if neutral_calibrator.is_ready(loop_started_at):
                        neutral_calibration = neutral_calibrator.finalize()
                        calibration_mode = None
                        if neutral_calibration is not None:
                            print("Neutral closed-mouth calibration captured.")
                    reset_runtime_state(jump_hold, keyboard)
                elif neutral_calibration is not None:
                    signals = compute_live_signals(
                        frame,
                        detected_face,
                        neutral_calibration,
                        landmark_result=landmark_result,
                    )
                    current_move = movement_from_signal(
                        signals.movement,
                        current_move,
                        enter_threshold=MOVE_ENTER_THRESHOLD,
                        exit_threshold=MOVE_EXIT_THRESHOLD,
                    )
                    jump_state = jump_hold.update(signals.mouth_landmark_score)
                    mouth_ratio = signals.mouth_landmark_ratio

                    if input_enabled:
                        keyboard.set_movement(current_move, args.left_key, args.right_key)
                        keyboard.set_hold(args.jump_key, jump_state)
                    else:
                        keyboard.release_all()
                else:
                    current_move = IDLE
                    reset_runtime_state(jump_hold, keyboard)
            else:
                current_move = IDLE
                reset_runtime_state(jump_hold, keyboard)

            loop_finished_at = time.perf_counter()
            fps = _ewma(fps, 1.0 / max(loop_finished_at - loop_started_at, 1e-6))
            jump_text = "HELD" if jump_state else "RELEASED"
            click_prompt = "Click Blobby window now" if not input_enabled else ""

            if overlay_enabled:
                overlay_lines = [
                    (f"FPS: {fps:.1f}", (255, 255, 255)),
                    (f"Face: {'yes' if face_detected else 'no'}", (80, 255, 80) if face_detected else (0, 80, 255)),
                    (f"Move: {current_move}", (0, 220, 255)),
                    (f"Jump: {jump_text}", (80, 255, 80)),
                    (f"Mouth ratio: {mouth_ratio:.3f}", (255, 220, 80)),
                    (f"Keyboard: {keyboard.status_text()}", (230, 230, 230)),
                    (f"Last key event: {keyboard.last_event()}", (255, 220, 80)),
                    ("q quit | c recalibrate | o overlay", (230, 230, 230)),
                ]
                if click_prompt:
                    overlay_lines.insert(0, (click_prompt, (0, 220, 255)))
                draw_overlay(frame, overlay_lines, detected_face)
                cv2.imshow(WINDOW_NAME, frame)
            elif loop_started_at - last_report_at >= SUMMARY_INTERVAL_SECONDS:
                print(
                    f"face={'yes' if face_detected else 'no'} move={current_move:<5} jump={jump_text:<8} "
                    f"mouth_ratio={mouth_ratio:.3f} keyboard={keyboard.status_text():<24} "
                    f"last_key={keyboard.last_event():<12}"
                )
                if click_prompt:
                    print(click_prompt)
                last_report_at = loop_started_at

            key = poll_runtime_key(overlay_enabled)
            if key == "q":
                break
            if key == "c":
                neutral_calibration = None
                calibration_mode = "neutral"
                current_move = IDLE
                jump_state = False
                begin_neutral_calibration(neutral_calibrator, jump_hold, keyboard, loop_started_at)
                print("Recalibrating neutral closed mouth...")
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
