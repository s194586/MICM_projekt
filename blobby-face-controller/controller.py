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
from gestures import CooldownTap, HysteresisHold, NeutralCalibrator, SmileCalibrator, clamp, compute_live_signals, movement_from_signal, select_jump_signal
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
DEFAULT_JUMP_MODE = "calibrated_smile"

DEFAULT_MOVE_ENTER_THRESHOLD = 0.075
DEFAULT_MOVE_EXIT_THRESHOLD = 0.035
DEFAULT_JUMP_ENTER_THRESHOLD = 0.55
DEFAULT_JUMP_EXIT_THRESHOLD = 0.35
DEFAULT_JUMP_THRESHOLD_STEP = 0.05

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
    parser.add_argument(
        "--jump-mode",
        choices=("calibrated_smile", "mouth_open", "smile_or_mouth_open", "vertical_head"),
        default=DEFAULT_JUMP_MODE,
    )
    parser.add_argument("--jump-enter", type=float, default=DEFAULT_JUMP_ENTER_THRESHOLD)
    parser.add_argument("--jump-exit", type=float, default=DEFAULT_JUMP_EXIT_THRESHOLD)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_YUNET_MODEL)
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
        return None

    if msvcrt is not None and msvcrt.kbhit():  # pragma: no branch - platform specific.
        return msvcrt.getwch().lower()
    return None


def reset_runtime_state(
    jump_hold: HysteresisHold,
    bonus_tap: CooldownTap,
    keyboard: KeyboardController,
) -> None:
    jump_hold.reset()
    bonus_tap.reset()
    keyboard.release_all()


def begin_neutral_calibration(
    calibrator: NeutralCalibrator,
    jump_hold: HysteresisHold,
    bonus_tap: CooldownTap,
    keyboard: KeyboardController,
    now: float,
) -> None:
    calibrator.reset(now)
    reset_runtime_state(jump_hold, bonus_tap, keyboard)


def begin_smile_calibration(
    calibrator: SmileCalibrator,
    jump_hold: HysteresisHold,
    bonus_tap: CooldownTap,
    keyboard: KeyboardController,
    now: float,
) -> None:
    calibrator.reset(now)
    reset_runtime_state(jump_hold, bonus_tap, keyboard)


def adjust_jump_thresholds(
    jump_hold: HysteresisHold,
    jump_enter_threshold: float,
    jump_exit_threshold: float,
    delta: float,
) -> tuple[float, float]:
    gap = max(0.05, jump_enter_threshold - jump_exit_threshold)
    jump_enter_threshold = clamp(jump_enter_threshold + delta, 0.10, 1.40)
    jump_exit_threshold = clamp(jump_enter_threshold - gap, 0.05, jump_enter_threshold - 0.02)
    jump_hold.set_thresholds(jump_enter_threshold, jump_exit_threshold)
    return jump_enter_threshold, jump_exit_threshold


def main() -> int:
    args = build_parser().parse_args()
    overlay_enabled = DEFAULT_OVERLAY and not args.no_overlay
    jump_enter_threshold = float(args.jump_enter)
    jump_exit_threshold = float(args.jump_exit)
    if jump_exit_threshold >= jump_enter_threshold:
        print("ERROR: --jump-exit must be lower than --jump-enter.")
        return 1

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
    neutral_calibrator = NeutralCalibrator(duration_seconds=DEFAULT_CALIBRATION_SECONDS)
    smile_calibrator = SmileCalibrator(duration_seconds=DEFAULT_CALIBRATION_SECONDS)
    jump_hold = HysteresisHold(jump_enter_threshold, jump_exit_threshold)
    bonus_tap = CooldownTap(DEFAULT_BONUS_COOLDOWN_SECONDS)

    neutral_calibration = None
    smile_calibration = None
    calibration_mode = "neutral"
    current_move = IDLE
    current_face = None
    face_detected = False
    jump_state = False

    last_sequence = -1
    last_report_at = 0.0
    fps = 0.0
    latency_ms = 0.0
    capture_ms = 0.0
    inference_ms = 0.0
    processing_ms = 0.0

    smile_raw = 0.0
    smile_norm = 0.0
    mouth_open_norm = 0.0

    print(
        f"Controller started | capture={camera.backend_name} | detector={detector.backend_name} "
        f"| jump={args.jump_mode} | keyboard={keyboard.status_text()}"
    )
    print(f"YuNet model path: {args.model_path}")
    print("Keep a neutral face for the first 1-2 seconds. Then press m and smile wide for 1-2 seconds.")
    if not overlay_enabled:
        print("Overlay disabled. Use q/c/m/o/[ / ] in the console, or Ctrl+C to stop.")

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
            calibration_text = calibration_mode.upper() if calibration_mode is not None else "READY"
            smile_raw = 0.0
            smile_norm = 0.0
            mouth_open_norm = 0.0

            if face_detected:
                if calibration_mode == "neutral":
                    neutral_calibrator.add_sample(current_face, frame)
                    if neutral_calibrator.is_ready(loop_started_at):
                        neutral_calibration = neutral_calibrator.finalize()
                        calibration_mode = None
                        smile_calibration = None
                        if neutral_calibration is not None:
                            print("Neutral calibration captured. Press m and smile wide to calibrate jump.")
                    current_move = IDLE
                    jump_state = False
                    reset_runtime_state(jump_hold, bonus_tap, keyboard)
                elif calibration_mode == "smile":
                    smile_calibrator.add_sample(current_face, frame)
                    if smile_calibrator.is_ready(loop_started_at):
                        smile_calibration = smile_calibrator.finalize()
                        calibration_mode = None
                        if smile_calibration is not None:
                            print("Smile calibration captured.")
                    current_move = IDLE
                    jump_state = False
                    reset_runtime_state(jump_hold, bonus_tap, keyboard)
                elif neutral_calibration is not None:
                    signals = compute_live_signals(frame, current_face, neutral_calibration, smile_calibration)
                    current_move = movement_from_signal(
                        signals.movement,
                        current_move,
                        enter_threshold=DEFAULT_MOVE_ENTER_THRESHOLD,
                        exit_threshold=DEFAULT_MOVE_EXIT_THRESHOLD,
                    )
                    keyboard.set_movement(current_move, MOVE_LEFT_KEY, MOVE_RIGHT_KEY)

                    smile_raw = signals.smile_raw
                    smile_norm = signals.smile_norm
                    mouth_open_norm = signals.mouth_open_norm
                    jump_state = jump_hold.update(select_jump_signal(signals, args.jump_mode))
                    keyboard.set_hold(JUMP_KEY, jump_state)

                    bonus_active = signals.bonus >= DEFAULT_BONUS_ENTER_THRESHOLD
                    if signals.bonus <= DEFAULT_BONUS_EXIT_THRESHOLD:
                        bonus_tap.reset()
                    if bonus_tap.update(bonus_active, loop_started_at):
                        keyboard.tap(BONUS_KEY, loop_started_at, DEFAULT_SPACE_TAP_SECONDS)
                else:
                    current_move = IDLE
                    jump_state = False
                    reset_runtime_state(jump_hold, bonus_tap, keyboard)
            else:
                current_move = IDLE
                jump_state = False
                smile_raw = 0.0
                smile_norm = 0.0
                mouth_open_norm = 0.0
                reset_runtime_state(jump_hold, bonus_tap, keyboard)

            cooldown_left = bonus_tap.cooldown_left(loop_started_at)
            bonus_text = f"COOLDOWN {cooldown_left:.1f}s" if cooldown_left > 0.0 and not bonus_tap.armed else "READY"
            move_text = current_move if neutral_calibration is not None and calibration_mode is None else "CALIBRATING"
            jump_text = "HELD" if jump_state else "RELEASED"

            processing_finished_at = time.perf_counter()
            processing_ms = _ewma(processing_ms, (processing_finished_at - processing_started_at) * 1000.0)
            fps = _ewma(fps, 1.0 / max(processing_finished_at - loop_started_at, 1e-6))

            neutral_smile_baseline = 0.0 if neutral_calibration is None else neutral_calibration.smile_raw
            smile_baseline = 0.0 if smile_calibration is None else smile_calibration.smile_raw
            neutral_ready_text = "yes" if neutral_calibration is not None else "no"
            smile_ready_text = "yes" if smile_calibration is not None else "no"

            if overlay_enabled:
                draw_overlay(
                    frame,
                    [
                        (f"FPS: {fps:.1f}", (255, 255, 255)),
                        (f"Detector: {detector.backend_name}", (230, 230, 230)),
                        (f"Inference: {inference_ms:.1f} ms", (230, 230, 230)),
                        (f"Face: {'yes' if face_detected else 'no'}", (80, 255, 80) if face_detected else (0, 80, 255)),
                        (f"Move: {move_text}", (0, 220, 255)),
                        (f"Jump mode: {args.jump_mode}", (230, 230, 230)),
                        (f"Jump: {jump_text}", (80, 255, 80)),
                        (f"Smile raw: {smile_raw:.3f}", (255, 220, 80)),
                        (f"Neutral smile: {neutral_smile_baseline:.3f}", (255, 220, 80)),
                        (f"Smile baseline: {smile_baseline:.3f}", (255, 220, 80)),
                        (f"Smile norm: {smile_norm:.3f}", (255, 220, 80)),
                        (f"Mouth norm: {mouth_open_norm:.3f}", (255, 220, 80)),
                        (f"Enter: {jump_enter_threshold:.2f} | Exit: {jump_exit_threshold:.2f}", (230, 230, 230)),
                        (f"Neutral calibrated: {neutral_ready_text}", (230, 230, 230)),
                        (f"Smile calibrated: {smile_ready_text}", (230, 230, 230)),
                        (f"Bonus: {bonus_text}", (255, 220, 80)),
                        (f"Cal mode: {calibration_text}", (230, 230, 230)),
                        ("q quit | c neutral | m smile | [ ] tune | o overlay", (230, 230, 230)),
                    ],
                    current_face,
                )
                cv2.imshow(WINDOW_NAME, frame)
            elif loop_started_at - last_report_at >= SUMMARY_INTERVAL_SECONDS:
                print(
                    f"fps={fps:5.1f} infer_ms={inference_ms:5.1f} latency_ms={latency_ms:5.1f} "
                    f"face={'yes' if face_detected else 'no'} move={move_text:<11} jump_mode={args.jump_mode:<20} "
                    f"jump={jump_text:<8} smile_raw={smile_raw:.3f} neutral_smile={neutral_smile_baseline:.3f} "
                    f"smile_base={smile_baseline:.3f} smile_norm={smile_norm:.3f} enter={jump_enter_threshold:.2f} "
                    f"exit={jump_exit_threshold:.2f} neutral={neutral_ready_text} smile={smile_ready_text}"
                )
                last_report_at = loop_started_at

            key = poll_runtime_key(overlay_enabled)
            if key == "q":
                break
            if key == "c":
                neutral_calibration = None
                smile_calibration = None
                calibration_mode = "neutral"
                current_move = IDLE
                jump_state = False
                begin_neutral_calibration(neutral_calibrator, jump_hold, bonus_tap, keyboard, loop_started_at)
                print("Recalibrating neutral face...")
            if key == "m":
                if neutral_calibration is None:
                    print("Calibrate neutral first with c or wait for startup calibration.")
                else:
                    smile_calibration = None
                    calibration_mode = "smile"
                    current_move = IDLE
                    jump_state = False
                    begin_smile_calibration(smile_calibrator, jump_hold, bonus_tap, keyboard, loop_started_at)
                    print("Calibrating smile... hold a wide smile for 1-2 seconds.")
            if key == "[":
                jump_enter_threshold, jump_exit_threshold = adjust_jump_thresholds(
                    jump_hold,
                    jump_enter_threshold,
                    jump_exit_threshold,
                    -DEFAULT_JUMP_THRESHOLD_STEP,
                )
                print(f"Jump thresholds: enter={jump_enter_threshold:.2f}, exit={jump_exit_threshold:.2f}")
            if key == "]":
                jump_enter_threshold, jump_exit_threshold = adjust_jump_thresholds(
                    jump_hold,
                    jump_enter_threshold,
                    jump_exit_threshold,
                    DEFAULT_JUMP_THRESHOLD_STEP,
                )
                print(f"Jump thresholds: enter={jump_enter_threshold:.2f}, exit={jump_exit_threshold:.2f}")
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
