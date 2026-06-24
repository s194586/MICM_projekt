"""Quick YuNet + landmark benchmark for camera settings and detector latency."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2

from capture import LatestFrameCamera
from detector import DEFAULT_YUNET_MODEL, create_face_detector
from landmark_detector import DEFAULT_LANDMARK_MODEL, create_landmark_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark YuNet detector and optional landmark latency.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--camera-backend", choices=("auto", "msmf", "dshow", "default"), default="auto")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--model-path", default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--landmark-model-path", default=str(DEFAULT_LANDMARK_MODEL))
    return parser


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def benchmark_pipeline(
    camera: LatestFrameCamera,
    duration_seconds: float,
    width: int,
    height: int,
    model_path: str,
    landmark_model_path: str,
) -> tuple[dict[str, float | str] | None, dict[str, float | str] | None]:
    try:
        detector = create_face_detector("yunet", input_size=(width, height), model_path=model_path)
    except Exception as exc:
        print(f"yunet: unavailable ({exc})")
        return None, None

    landmark_detector = None
    try:
        landmark_detector = create_landmark_detector(landmark_model_path)
    except Exception as exc:
        print(f"landmark: unavailable ({exc})")

    yunet_times: list[float] = []
    landmark_times: list[float] = []
    total_times: list[float] = []
    detection_count = 0
    landmark_success_count = 0
    frame_count = 0
    last_sequence = -1
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        sequence, frame, _, _ = camera.latest()
        if frame is None or sequence == last_sequence:
            time.sleep(0.001)
            continue

        last_sequence = sequence
        frame = cv2.flip(frame, 1)

        total_started_at = time.perf_counter()
        face, inference_ms = detector.detect(frame)
        total_after_yunet = time.perf_counter()

        yunet_times.append(float(inference_ms))
        frame_count += 1
        if face is not None:
            detection_count += 1

        if face is not None and landmark_detector is not None:
            landmark_result = landmark_detector.detect(frame, face)
            landmark_elapsed_ms = 0.0 if landmark_result is None else float(landmark_result.inference_ms)
            landmark_times.append(landmark_elapsed_ms)
            if landmark_result is not None:
                landmark_success_count += 1
        elif landmark_detector is not None:
            landmark_times.append(0.0)

        total_times.append((time.perf_counter() - total_started_at) * 1000.0)

    elapsed = max(time.perf_counter() - start, 1e-6)
    yunet_result = {
        "backend": "yunet",
        "fps": frame_count / elapsed,
        "avg_ms": statistics.mean(yunet_times) if yunet_times else 0.0,
        "p95_ms": percentile(yunet_times, 0.95),
        "detection_rate": (detection_count / frame_count) if frame_count else 0.0,
        "frames": float(frame_count),
    }

    combined_result = None
    if landmark_detector is not None:
        detected_frames = max(detection_count, 1)
        combined_result = {
            "backend": "yunet+landmark",
            "fps": frame_count / elapsed,
            "avg_yunet_ms": statistics.mean(yunet_times) if yunet_times else 0.0,
            "avg_landmark_ms": statistics.mean(landmark_times) if landmark_times else 0.0,
            "avg_total_ms": statistics.mean(total_times) if total_times else 0.0,
            "detection_rate": (detection_count / frame_count) if frame_count else 0.0,
            "landmark_success_rate": (landmark_success_count / detected_frames),
        }

    return yunet_result, combined_result


def choose_recommendation(
    yunet_result: dict[str, float | str] | None,
    combined_result: dict[str, float | str] | None,
    width: int,
    height: int,
    camera_backend: str,
) -> str:
    if yunet_result is None:
        return "YuNet could not start. Verify the ONNX model path and OpenCV build first."

    detection_rate = float(yunet_result["detection_rate"])
    fps = float(yunet_result["fps"])
    if detection_rate < 0.70:
        return "Detection rate is low. Improve lighting, keep one face centered, and recalibrate with a neutral pose."
    if combined_result is not None:
        total_ms = float(combined_result["avg_total_ms"])
        landmark_success_rate = float(combined_result["landmark_success_rate"])
        if landmark_success_rate < 0.70:
            return "Landmark success rate is low. Sit closer to the camera and keep the face centered."
        if total_ms > 20.0 or fps < 45.0:
            return (
                "Latency is above the fast target. Prefer "
                "`python controller.py --width 320 --height 240 --camera-backend dshow --no-overlay`."
            )
    return (
        f"Current settings look good. Recommended run: "
        f"`python controller.py --width {width} --height {height} --camera-backend {camera_backend}`."
    )


def main() -> int:
    args = build_parser().parse_args()
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

    try:
        yunet_result, combined_result = benchmark_pipeline(
            camera,
            duration_seconds=args.seconds,
            width=args.width,
            height=args.height,
            model_path=args.model_path,
            landmark_model_path=args.landmark_model_path,
        )

        if yunet_result is not None:
            print(
                f"yunet: fps={float(yunet_result['fps']):.1f} avg_yunet_ms={float(yunet_result['avg_ms']):.2f} "
                f"p95_yunet_ms={float(yunet_result['p95_ms']):.2f} detection_rate={float(yunet_result['detection_rate']):.2%}"
            )
        if combined_result is not None:
            print(
                f"yunet+landmark: avg_yunet_ms={float(combined_result['avg_yunet_ms']):.2f} "
                f"avg_landmark_ms={float(combined_result['avg_landmark_ms']):.2f} "
                f"avg_total_ms={float(combined_result['avg_total_ms']):.2f} "
                f"fps={float(combined_result['fps']):.1f} "
                f"detection_rate={float(combined_result['detection_rate']):.2%} "
                f"landmark_success_rate={float(combined_result['landmark_success_rate']):.2%}"
            )
        print(f"Recommendation: {choose_recommendation(yunet_result, combined_result, args.width, args.height, args.camera_backend)}")
    finally:
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
