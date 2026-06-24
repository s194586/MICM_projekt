"""Model-based low-latency controller without MediaPipe FaceMesh."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

import config
from model_fast_capture import LatestFrameCamera
from model_fast_detector import DEFAULT_YUNET_MODEL, create_face_detector, recommended_default_detector
from model_fast_gestures import CooldownTap, HysteresisHold, NeutralCalibrator, bonus_signal, jump_signal, movement_from_signal, movement_signal
from model_fast_keyboard import KeyboardController


LEFT = "LEFT"
RIGHT = "RIGHT"
IDLE = "IDLE"

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 60
DEFAULT_KEYBOARD_BACKEND = "win32"
DEFAULT_OVERLAY = True
DEFAULT_CALIBRATION_SECONDS = 1.2
DEFAULT_JUMP_MODE = "smile_landmarks"
DEFAULT_MOVE_ENTER_THRESHOLD = 0.075
DEFAULT_MOVE_EXIT_THRESHOLD = 0.035
DEFAULT_JUMP_ENTER_THRESHOLD = 0.032
DEFAULT_JUMP_EXIT_THRESHOLD = 0.016
DEFAULT_BONUS_ENTER_THRESHOLD = 0.065
DEFAULT_BONUS_EXIT_THRESHOLD = 0.032
DEFAULT_BONUS_COOLDOWN_SECONDS = 0.9
DEFAULT_SPACE_TAP_SECONDS = 0.03
SUMMARY_INTERVAL_SECONDS = 2.0


def _ewma(previous: float, current: float, alpha: float = 0.18) -> float:
    if previous <= 0.0:
        return current
    return previous + alpha * (current - previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-based low-latency controller without MediaPipe.")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--detector", choices=("yunet", "haar"), default=recommended_default_detector())
    parser.add_argument("--allow-haar-fallback", action="store_true")
    parser.add_argument("--jump-mode", choices=("smile_landmarks", "mouth_distance", "vertical_head", "keyboard_test"), default=DEFAULT_JUMP_MODE)
    parser.add_argument("--keyboard", choices=("win32", "pynput"), default=DEFAULT_KEYBOARD_BACKEND)
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


def main() -> int:
    args = build_parser().parse_args()
    overlay_enabled = DEFAULT_OVERLAY and not args.no_overlay

    try:
        detector = create_face_detector(
            args.detector,
            input_size=(args.width, args.height),
            model_path=args.model_path,
            allow_haar_fallback=args.allow_haar_fallback,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        if args.detector == "yunet":
            print(f"Expected YuNet model path: {args.model_path}")
            print("If you want the crude fallback anyway, rerun with --allow-haar-fallback --detector yunet.")
        return 1

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

    keyboard = KeyboardController(preferred_backend=args.keyboard)
    calibrator = NeutralCalibrator(duration_seconds=DEFAULT_CALIBRATION_SECONDS)
    jump_hold = HysteresisHold(DEFAULT_JUMP_ENTER_THRESHOLD, DEFAULT_JUMP_EXIT_THRESHOLD)
    bonus_tap = CooldownTap(DEFAULT_BONUS_COOLDOWN_SECONDS)

    calibration = None
    current_move = IDLE
    last_sequence = -1
    last_report_at = 0.0

    fps = 0.0
    latency_ms = 0.0
    capture_ms = 0.0
    inference_ms = 0.0
    processing_ms = 0.0
    jump_state = False
    face_detected = False
    current_face = None

    print(
        f"Model fast controller started | capture={camera.backend_name} | detector={detector.backend_name} "
        f"| jump={args.jump_mode} | keyboard={keyboard.status_text()}"
    )
    print(f"YuNet model path: {args.model_path}")
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
            current_face, raw_inference_ms = detector.detect(frame)
            inference_ms = _ewma(inference_ms, raw_inference_ms)
            face_detected = current_face is not None

            processing_started_at = time.perf_counter()
            move_text = IDLE
            jump_text = "RELEASED"
            bonus_text = "READY"
            calibrated_text = "no"

            if face_detected:
                if calibration is None:
                    calibrator.add_sample(current_face)
                    if calibrator.is_ready(loop_started_at):
                        calibration = calibrator.finalize()
                        if calibration is not None:
                            print("Neutral calibration captured.")
                else:
                    move_signal = movement_signal(current_face, calibration)
                    current_move = movement_from_signal(
                        move_signal,
                        current_move,
                        enter_threshold=DEFAULT_MOVE_ENTER_THRESHOLD,
                        exit_threshold=DEFAULT_MOVE_EXIT_THRESHOLD,
                    )
                    keyboard.set_movement(current_move, config.MOVE_LEFT_KEY, config.MOVE_RIGHT_KEY)

                    current_jump_signal = jump_signal(current_face, calibration, args.jump_mode)
                    jump_state = jump_hold.update(current_jump_signal)
                    keyboard.set_hold(config.JUMP_KEY, jump_state)

                    current_bonus_signal = bonus_signal(current_face, calibration)
                    bonus_active = current_bonus_signal >= DEFAULT_BONUS_ENTER_THRESHOLD
                    if current_bonus_signal <= DEFAULT_BONUS_EXIT_THRESHOLD:
                        bonus_tap.reset()
                    if bonus_tap.update(bonus_active, loop_started_at):
                        keyboard.tap(config.BONUS_KEY, loop_started_at, DEFAULT_SPACE_TAP_SECONDS)

                if calibration is None:
                    current_move = IDLE
                    jump_state = False
                    jump_hold.reset()
                    keyboard.release_all()
                calibrated_text = "yes" if calibration is not None else "CALIBRATING"
            else:
                current_move = IDLE
                jump_state = False
                jump_hold.reset()
                bonus_tap.reset()
                keyboard.release_all()
                if calibration is None:
                    calibrated_text = "CALIBRATING"

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
                        (f"Latency: {latency_ms:.1f} ms", (230, 230, 230)),
                        (f"Face: {'yes' if face_detected else 'no'}", (80, 255, 80) if face_detected else (0, 80, 255)),
                        (f"Move: {move_text}", (0, 220, 255)),
                        (f"Jump: {jump_text}", (80, 255, 80)),
                        (f"Bonus: {bonus_text}", (255, 220, 80)),
                        (f"Calibrated: {calibrated_text}", (230, 230, 230)),
                        ("q quit | c recalibrate", (230, 230, 230)),
                    ],
                    current_face,
                )
                cv2.imshow("Blobby Model Fast Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    calibration = None
                    calibrator.reset(loop_started_at)
                    current_move = IDLE
                    jump_state = False
                    jump_hold.reset()
                    bonus_tap.reset()
                    keyboard.release_all()
                    print("Recalibrating neutral face...")
            else:
                if loop_started_at - last_report_at >= SUMMARY_INTERVAL_SECONDS:
                    print(
                        f"fps={fps:5.1f} infer_ms={inference_ms:5.1f} latency_ms={latency_ms:5.1f} "
                        f"capture_ms={capture_ms:5.1f} proc_ms={processing_ms:5.1f} "
                        f"face={'yes' if face_detected else 'no'} move={move_text:<11} "
                        f"jump={jump_text:<8} bonus={bonus_text:<12} calibrated={calibrated_text}"
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
