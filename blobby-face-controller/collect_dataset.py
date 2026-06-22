"""Collect Player 2 head-down samples for the bonus gesture classifier."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter

import cv2
import mediapipe as mp
import pandas as pd

import config
from feature_extraction import (
    FEATURE_NAMES,
    estimate_smile_score,
    extract_feature_dict,
    extract_features,
    face_bbox,
    sort_faces_left_to_right,
)


CSV_COLUMNS = [*FEATURE_NAMES, "label"]
BURST_SAMPLE_COUNT = 50
BURST_INTERVAL_SECONDS = 0.06


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear gestures.csv and recreate only its header before collection",
    )
    return parser.parse_args(argv)


def reset_dataset_file() -> None:
    config.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.DATASET_PATH.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(CSV_COLUMNS)


def ensure_dataset_file() -> None:
    if not config.DATASET_PATH.exists() or config.DATASET_PATH.stat().st_size == 0:
        reset_dataset_file()
        return

    with config.DATASET_PATH.open("r", newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), [])
    if header != CSV_COLUMNS:
        raise ValueError(
            f"Dataset header does not match the current feature schema: {config.DATASET_PATH}\n"
            f"Expected: {CSV_COLUMNS}\n"
            f"Found:    {header}"
        )


def load_counts() -> Counter:
    ensure_dataset_file()
    try:
        frame = pd.read_csv(config.DATASET_PATH)
    except pd.errors.EmptyDataError:
        return Counter()
    labels = pd.to_numeric(frame.get("label"), errors="coerce").dropna().astype(int)
    return Counter(label for label in labels if label in config.LABEL_NAMES)


def append_sample(features, label: int) -> None:
    ensure_dataset_file()
    with config.DATASET_PATH.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([*features.tolist(), label])


def draw_face_box(frame, landmarks, label: str, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    min_x, min_y, max_x, max_y = face_bbox(landmarks)
    x1 = int(min_x * width)
    y1 = int(min_y * height)
    x2 = int(max_x * width)
    y2 = int(max_y * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def put_lines(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 27
    for text, color in lines:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        y += 25


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset:
        reset_dataset_file()
        print(f"Dataset reset: {config.DATASET_PATH}")

    try:
        counts = load_counts()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {config.CAMERA_INDEX}. Check config.py.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    mp_face_mesh = mp.solutions.face_mesh
    prev_time = time.perf_counter()
    fps = 0.0
    last_saved = "none"
    burst_label: int | None = None
    burst_remaining = 0
    next_burst_sample_at = 0.0

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.MAX_NUM_FACES,
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
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)

                faces = sort_faces_left_to_right(results.multi_face_landmarks or [])
                source_face = faces[1] if len(faces) >= 2 else (faces[0] if len(faces) == 1 else None)
                player2_detected = len(faces) >= 2

                feature_dict = extract_feature_dict(source_face) if source_face is not None else None
                feature_vector = extract_features(source_face) if source_face is not None else None
                smile_score = estimate_smile_score(feature_dict) if feature_dict is not None else None

                if len(faces) >= 1:
                    label = "Player 1" if player2_detected else "Single face - fallback dataset source"
                    draw_face_box(frame, faces[0], label, (0, 220, 255))
                if player2_detected:
                    draw_face_box(frame, faces[1], "Player 2 - dataset source", (80, 255, 80))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                single_labels = {ord("n"): config.LABEL_NEUTRAL, ord("b"): config.LABEL_BONUS}
                burst_labels = {ord("1"): config.LABEL_NEUTRAL, ord("2"): config.LABEL_BONUS}

                if key in single_labels:
                    if feature_vector is None:
                        print("No face detected. Sample not saved.")
                    else:
                        label = single_labels[key]
                        append_sample(feature_vector, label)
                        counts[label] += 1
                        last_saved = config.LABEL_NAMES[label]
                        print(
                            f"Saved {last_saved}: neutral={counts[config.LABEL_NEUTRAL]}, "
                            f"bonus={counts[config.LABEL_BONUS]}"
                        )
                elif key in burst_labels:
                    if feature_vector is None:
                        print("No face detected. Burst not started.")
                    else:
                        burst_label = burst_labels[key]
                        burst_remaining = BURST_SAMPLE_COUNT
                        next_burst_sample_at = now
                        print(f"Started burst: {config.LABEL_NAMES[burst_label]} ({BURST_SAMPLE_COUNT} samples)")

                if burst_label is not None and feature_vector is not None and now >= next_burst_sample_at:
                    append_sample(feature_vector, burst_label)
                    counts[burst_label] += 1
                    last_saved = config.LABEL_NAMES[burst_label]
                    burst_remaining -= 1
                    next_burst_sample_at = now + BURST_INTERVAL_SECONDS
                    if burst_remaining == 0:
                        print(f"Completed burst: {config.LABEL_NAMES[burst_label]}")
                        burst_label = None

                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now

                if player2_detected:
                    source_text = "Dataset source: Player 2 (right face)"
                    source_color = (80, 255, 80)
                elif source_face is not None:
                    source_text = "Dataset source: single-face fallback"
                    source_color = (0, 180, 255)
                else:
                    source_text = "WARNING: no face detected; capture disabled"
                    source_color = (0, 80, 255)

                if burst_label is None:
                    burst_text = "Burst: idle"
                elif source_face is None:
                    burst_text = f"Burst: PAUSED, no face ({burst_remaining} remaining)"
                else:
                    burst_text = f"Burst: {config.LABEL_NAMES[burst_label]} ({burst_remaining} remaining)"

                def show(name: str) -> str:
                    if feature_dict is None:
                        return "n/a"
                    return f"{feature_dict[name]:.3f}"

                smile_text = "n/a" if smile_score is None else f"{smile_score:.3f}"
                put_lines(
                    frame,
                    [
                        (f"FPS: {fps:.1f} | Faces detected: {len(faces)}", (255, 255, 255)),
                        ("Dataset target: bonus model", (255, 220, 80)),
                        ("neutral = normal head position", (230, 230, 230)),
                        ("bonus_gesture = head tilted down / nod down", (230, 230, 230)),
                        ("Jump is NOT trained here. Jump is rule-based smile detection.", (230, 230, 230)),
                        (source_text, source_color),
                        (burst_text, (255, 220, 80)),
                        (
                            f"neutral: {counts[config.LABEL_NEUTRAL]} | "
                            f"bonus_gesture: {counts[config.LABEL_BONUS]} | last: {last_saved}",
                            (255, 255, 255),
                        ),
                        (
                            f"mouth_width_ratio: {show('mouth_width_ratio')} | smile_score: {smile_text}",
                            (200, 255, 200),
                        ),
                        (
                            f"head_pitch: {show('head_pitch')} | head_yaw: {show('head_yaw')} | "
                            f"mouth_open: {show('mouth_open_ratio')}",
                            (200, 255, 200),
                        ),
                        ("n = neutral | b = bonus_gesture", (230, 230, 230)),
                        ("1 = burst neutral | 2 = burst bonus_gesture | q = quit", (230, 230, 230)),
                    ],
                )

                cv2.imshow("Blobby Face Controller - collect bonus dataset", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
