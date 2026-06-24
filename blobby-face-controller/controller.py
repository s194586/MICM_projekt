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
from landmark_detector import DEFAULT_LANDMARK_MODEL, LandmarkDetectionResult, create_landmark_detector


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
DEFAULT_JUMP_MODE = "mouth_landmarks"
DEFAULT_FOCUS_DELAY_SECONDS = 3.0

DEFAULT_MOVE_ENTER_THRESHOLD = 0.075
DEFAULT_MOVE_EXIT_THRESHOLD = 0.035
DEFAULT_JUMP_ENTER_THRESHOLD = 0.35
DEFAULT_JUMP_EXIT_THRESHOLD = 0.20
DEFAULT_JUMP_THRESHOLD_STEP = 0.05

DEFAULT_BONUS_ENTER_THRESHOLD = 0.065
DEFAULT_BONUS_EXIT_THRESHOLD = 0.032
DEFAULT_BONUS_COOLDOWN_SECONDS = 0.9
DEFAULT_SPACE_TAP_SECONDS = 0.03
SUMMARY_INTERVAL_SECONDS = 2.0

DEFAULT_LEFT_KEY = "a"
DEFAULT_RIGHT_KEY = "d"
DEFAULT_JUMP_KEY = "w"
DEFAULT_BONUS_KEY = "space"
LANDMARK_REQUIRED_MODES = {"mouth_landmarks"}
SMILE_CALIBRATION_MODES = {"calibrated_smile", "smile_or_mouth_open"}


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
    parser.add_argument("--left-key", choices=("a", "left"), default=DEFAULT_LEFT_KEY)
    parser.add_argument("--right-key", choices=("d", "right"), default=DEFAULT_RIGHT_KEY)
    parser.add_argument("--jump-key", choices=("w", "up"), default=DEFAULT_JUMP_KEY)
    parser.add_argument("--bonus-key", choices=("space",), default=DEFAULT_BONUS_KEY)
    parser.add_argument(
        "--jump-mode",
        choices=("mouth_landmarks", "calibrated_smile", "mouth_open", "smile_or_mouth_open", "vertical_head_up"),
        default=DEFAULT_JUMP_MODE,
    )
    parser.add_argument("--jump-enter", type=float, default=DEFAULT_JUMP_ENTER_THRESHOLD)
    parser.add_argument("--jump-exit", type=float, default=DEFAULT_JUMP_EXIT_THRESHOLD)
    parser.add_argument("--focus-delay", type=float, default=DEFAULT_FOCUS_DELAY_SECONDS)
    parser.add_argument("--landmark-model-path", type=Path, default=DEFAULT_LANDMARK_MODEL)
    parser.add_argument("--debug-landmarks", action="store_true")
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


def draw_debug_landmarks(frame, landmark_result: LandmarkDetectionResult | None) -> None:
    if landmark_result is None:
        return
    for point in landmark_result.mouth_points:
        cv2.circle(frame, (int(round(point[0])), int(round(point[1]))), 1, (255, 180, 0), -1)


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
    gap = max(0.03, jump_enter_threshold - jump_exit_threshold)
    jump_enter_threshold = clamp(jump_enter_threshold + delta, 0.05, 1.40)
    jump_exit_threshold = clamp(jump_enter_threshold - gap, 0.02, jump_enter_threshold - 0.01)
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

    landmark_detector = None
    if args.jump_mode in LANDMARK_REQUIRED_MODES or args.debug_landmarks:
        try:
            landmark_detector = create_landmark_detector(args.landmark_model_path)
        except Exception as exc:
            print(f"ERROR: {exc}")
            print(f"Missing landmark model. Download/place it here: {args.landmark_model_path}")
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
    yunet_ms = 0.0
    landmark_ms = 0.0
    total_stage_ms = 0.0

    smile_raw = 0.0
    smile_norm = 0.0
    mouth_open_norm = 0.0
    mouth_ratio = 0.0
    landmark_detected = False
    input_enable_at = time.perf_counter() + max(0.0, float(args.focus_delay))
    input_enabled_announced = False

    print(
        f"Controller started | capture={camera.backend_name} | detector={detector.backend_name} "
        f"| jump={args.jump_mode} | keyboard={keyboard.status_text()}"
    )
    print(f"YuNet model path: {args.model_path}")
    if landmark_detector is not None:
        print(f"Landmark model path: {args.landmark_model_path} | backend={landmark_detector.backend_name}")
    print("Keep a neutral face with closed mouth for the first 1-2 seconds, or press c later to recalibrate.")
    if args.jump_mode in SMILE_CALIBRATION_MODES:
        print("Smile fallback is available, but not required for gameplay. Use m only if you want calibrated_smile.")
    if args.focus_delay > 0.0:
        print(f"Click Blobby window now. Input starts in {args.focus_delay:.1f}s.")
    if not overlay_enabled:
        print("Overlay disabled. Use q/c/[ / ]/o in the console, or Ctrl+C to stop.")

    try:
        while True:
            loop_started_at = time.perf_counter()
            keyboard.update(loop_started_at)
            input_enabled = loop_started_at >= input_enable_at
            if input_enabled and not input_enabled_announced:
                print("Input enabled.")
                input_enabled_announced = True

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
            stage_started_at = time.perf_counter()
            current_face, raw_yunet_ms = detector.detect(frame)
            yunet_ms = _ewma(yunet_ms, raw_yunet_ms)
            face_detected = current_face is not None

            current_landmarks = None
            raw_landmark_ms = 0.0
            if face_detected and landmark_detector is not None:
                current_landmarks = landmark_detector.detect(frame, current_face)
                if current_landmarks is not None:
                    landmark_detected = True
                    raw_landmark_ms = current_landmarks.inference_ms
                else:
                    landmark_detected = False
            else:
                landmark_detected = False

            landmark_ms = _ewma(landmark_ms, raw_landmark_ms)
            total_stage_ms = _ewma(total_stage_ms, (time.perf_counter() - stage_started_at) * 1000.0)

            processing_started_at = time.perf_counter()
            move_text = IDLE
            jump_text = "RELEASED"
            bonus_text = "READY"
            calibration_text = calibration_mode.upper() if calibration_mode is not None else "READY"
            smile_raw = 0.0
            smile_norm = 0.0
            mouth_open_norm = 0.0
            mouth_ratio = 0.0

            if face_detected:
                if calibration_mode == "neutral":
                    neutral_calibrator.add_sample(current_face, frame, current_landmarks)
                    if neutral_calibrator.is_ready(loop_started_at):
                        neutral_calibration = neutral_calibrator.finalize()
                        calibration_mode = None
                        if neutral_calibration is not None:
                            print("Neutral closed-mouth calibration captured.")
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
                    signals = compute_live_signals(
                        frame,
                        current_face,
                        neutral_calibration,
                        smile_calibration,
                        current_landmarks,
                    )
                    current_move = movement_from_signal(
                        signals.movement,
                        current_move,
                        enter_threshold=DEFAULT_MOVE_ENTER_THRESHOLD,
                        exit_threshold=DEFAULT_MOVE_EXIT_THRESHOLD,
                    )
                    if input_enabled:
                        keyboard.set_movement(current_move, args.left_key, args.right_key)
                    else:
                        keyboard.release_all()

                    smile_raw = signals.smile_raw
                    smile_norm = signals.smile_norm
                    mouth_open_norm = signals.mouth_open_norm
                    mouth_ratio = signals.mouth_landmark_ratio
                    jump_state = jump_hold.update(select_jump_signal(signals, args.jump_mode))
                    if input_enabled:
                        keyboard.set_hold(args.jump_key, jump_state)

                    bonus_active = signals.bonus >= DEFAULT_BONUS_ENTER_THRESHOLD
                    if input_enabled:
                        if signals.bonus <= DEFAULT_BONUS_EXIT_THRESHOLD:
                            bonus_tap.reset()
                        if bonus_tap.update(bonus_active, loop_started_at):
                            keyboard.tap(args.bonus_key, loop_started_at, DEFAULT_SPACE_TAP_SECONDS)
                    else:
                        bonus_tap.reset()
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
                mouth_ratio = 0.0
                reset_runtime_state(jump_hold, bonus_tap, keyboard)

            cooldown_left = bonus_tap.cooldown_left(loop_started_at)
            bonus_text = f"COOLDOWN {cooldown_left:.1f}s" if cooldown_left > 0.0 and not bonus_tap.armed else "READY"
            move_text = current_move
            jump_text = "HELD" if jump_state else "RELEASED"

            processing_finished_at = time.perf_counter()
            fps = _ewma(fps, 1.0 / max(processing_finished_at - loop_started_at, 1e-6))

            threshold_text = f"{jump_enter_threshold:.2f}/{jump_exit_threshold:.2f}"
            neutral_ready_text = "yes" if neutral_calibration is not None else "no"
            focus_seconds_left = max(0.0, input_enable_at - loop_started_at)
            click_prompt = f"Click Blobby window now ({focus_seconds_left:.1f}s)" if not input_enabled else ""

            if overlay_enabled:
                if args.debug_landmarks:
                    draw_debug_landmarks(frame, current_landmarks)
                overlay_lines = [
                    (f"Keyboard: {keyboard.status_text()}", (230, 230, 230)),
                    (f"Last key event: {keyboard.last_event()}", (255, 220, 80)),
                    (f"Move: {move_text}", (0, 220, 255)),
                    (f"Jump: {jump_text}", (80, 255, 80)),
                    (f"Bonus: {bonus_text}", (255, 220, 80)),
                    (f"Jump mode: {args.jump_mode}", (230, 230, 230)),
                    (f"Threshold: {threshold_text}", (230, 230, 230)),
                    (f"Mouth ratio: {mouth_ratio:.3f}", (255, 220, 80)),
                    (f"Face: {'yes' if face_detected else 'no'} | Landmark: {'yes' if landmark_detected else 'no'}", (230, 230, 230)),
                    (f"Neutral closed-mouth: {neutral_ready_text} | Cal mode: {calibration_text}", (230, 230, 230)),
                    ("q quit | c closed mouth | [ ] tune | o overlay | m smile fallback", (230, 230, 230)),
                ]
                if click_prompt:
                    overlay_lines.insert(0, (click_prompt, (0, 220, 255)))
                draw_overlay(
                    frame,
                    overlay_lines,
                    current_face,
                )
                cv2.imshow(WINDOW_NAME, frame)
            elif loop_started_at - last_report_at >= SUMMARY_INTERVAL_SECONDS:
                print(
                    f"keyboard={keyboard.status_text():<24} last_key={keyboard.last_event():<12} "
                    f"move={move_text:<5} jump={jump_text:<8} bonus={bonus_text:<12} "
                    f"jump_mode={args.jump_mode:<20} mouth_ratio={mouth_ratio:.3f} threshold={threshold_text} "
                    f"face={'yes' if face_detected else 'no'} landmark={'yes' if landmark_detected else 'no'}"
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
                begin_neutral_calibration(neutral_calibrator, jump_hold, bonus_tap, keyboard, loop_started_at)
                print("Recalibrating neutral closed mouth...")
            if key == "m":
                if args.jump_mode not in SMILE_CALIBRATION_MODES:
                    print("Smile calibration is only used by the calibrated_smile fallback modes.")
                elif neutral_calibration is None:
                    print("Calibrate neutral first with c or wait for startup calibration.")
                else:
                    smile_calibration = None
                    calibration_mode = "smile"
                    current_move = IDLE
                    jump_state = False
                    begin_smile_calibration(smile_calibrator, jump_hold, bonus_tap, keyboard, loop_started_at)
                    print("Calibrating smile fallback... hold a wide smile for 1-2 seconds.")
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
