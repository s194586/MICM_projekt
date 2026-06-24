"""Quick camera benchmark for available detector backends."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2

import config
from model_fast_capture import LatestFrameCamera
from model_fast_detector import DEFAULT_YUNET_MODEL, create_face_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark fast face detector backends.")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--model-path", default=str(DEFAULT_YUNET_MODEL))
    return parser


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def benchmark_backend(
    backend_name: str,
    camera: LatestFrameCamera,
    duration_seconds: float,
    width: int,
    height: int,
    model_path: str,
) -> dict[str, float | str] | None:
    try:
        detector = create_face_detector(
            backend_name,
            input_size=(width, height),
            model_path=model_path,
            allow_haar_fallback=False,
        )
    except Exception as exc:
        print(f"{backend_name}: unavailable ({exc})")
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
        "backend": backend_name,
        "fps": frame_count / elapsed,
        "avg_ms": statistics.mean(detection_times) if detection_times else 0.0,
        "p95_ms": percentile(detection_times, 0.95),
        "detections": float(detection_count),
        "frames": float(frame_count),
        "detection_rate": (detection_count / frame_count) if frame_count else 0.0,
    }
    print(
        f"{backend_name}: fps={result['fps']:.1f} avg_ms={result['avg_ms']:.2f} "
        f"p95_ms={result['p95_ms']:.2f} detections={detection_count}/{frame_count}"
    )
    return result


def choose_recommendation(results: list[dict[str, float | str]]) -> str:
    if not results:
        return "none"

    def score(item: dict[str, float | str]) -> tuple[float, float, float]:
        detection_rate = float(item["detection_rate"])
        avg_ms = float(item["avg_ms"])
        fps = float(item["fps"])
        return (detection_rate, -avg_ms, fps)

    best = max(results, key=score)
    return str(best["backend"])


def main() -> int:
    args = build_parser().parse_args()
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

    try:
        results: list[dict[str, float | str]] = []
        for backend_name in ("yunet", "haar"):
            result = benchmark_backend(
                backend_name,
                camera,
                duration_seconds=args.seconds,
                width=args.width,
                height=args.height,
                model_path=args.model_path,
            )
            if result is not None:
                results.append(result)

        recommendation = choose_recommendation(results)
        print(f"Recommended backend: {recommendation}")
    finally:
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
