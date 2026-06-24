"""Quick YuNet benchmark for camera settings and detector latency."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2

from capture import LatestFrameCamera
from detector import DEFAULT_YUNET_MODEL, create_face_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark YuNet detector latency and face detection rate.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--camera-backend", choices=("auto", "msmf", "dshow", "default"), default="auto")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--model-path", default=str(DEFAULT_YUNET_MODEL))
    return parser


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def benchmark_yunet(
    camera: LatestFrameCamera,
    duration_seconds: float,
    width: int,
    height: int,
    model_path: str,
) -> dict[str, float | str] | None:
    try:
        detector = create_face_detector("yunet", input_size=(width, height), model_path=model_path)
    except Exception as exc:
        print(f"yunet: unavailable ({exc})")
        return None

    detection_times: list[float] = []
    detection_count = 0
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
        face, inference_ms = detector.detect(frame)
        detection_times.append(float(inference_ms))
        frame_count += 1
        if face is not None:
            detection_count += 1

    elapsed = max(time.perf_counter() - start, 1e-6)
    result = {
        "backend": "yunet",
        "fps": frame_count / elapsed,
        "avg_ms": statistics.mean(detection_times) if detection_times else 0.0,
        "p95_ms": percentile(detection_times, 0.95),
        "detections": float(detection_count),
        "frames": float(frame_count),
        "detection_rate": (detection_count / frame_count) if frame_count else 0.0,
    }
    print(
        f"yunet: fps={result['fps']:.1f} avg_ms={result['avg_ms']:.2f} "
        f"p95_ms={result['p95_ms']:.2f} detections={detection_count}/{frame_count}"
    )
    return result


def choose_recommendation(result: dict[str, float | str] | None, width: int, height: int, camera_backend: str) -> str:
    if result is None:
        return "YuNet could not start. Verify the ONNX model path and OpenCV build first."

    fps = float(result["fps"])
    p95_ms = float(result["p95_ms"])
    detection_rate = float(result["detection_rate"])

    if detection_rate < 0.70:
        return "Detection rate is low. Improve lighting, keep one face centered, and recalibrate with a neutral pose."
    if p95_ms > 20.0 or fps < 45.0:
        return (
            "Latency is higher than the fast target. Prefer "
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
        result = benchmark_yunet(
            camera,
            duration_seconds=args.seconds,
            width=args.width,
            height=args.height,
            model_path=args.model_path,
        )
        print(f"Recommendation: {choose_recommendation(result, args.width, args.height, args.camera_backend)}")
    finally:
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
