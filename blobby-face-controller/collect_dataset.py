"""Collect feature-based samples for the bonus gesture classifier."""

from __future__ import annotations

import csv
import time
from collections import Counter

import cv2
import mediapipe as mp
import pandas as pd

import config
from feature_extraction import FEATURE_NAMES, extract_features, face_bbox, sort_faces_left_to_right


CSV_COLUMNS = [*FEATURE_NAMES, "label"]


def ensure_dataset_file() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.DATASET_PATH.exists() or config.DATASET_PATH.stat().st_size == 0:
        with config.DATASET_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)
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
        df = pd.read_csv(config.DATASET_PATH)
    except pd.errors.EmptyDataError:
        return Counter()
    if "label" not in df.columns:
        return Counter()
    return Counter(df["label"].dropna().astype(int).tolist())


def append_sample(features, label: int) -> None:
    ensure_dataset_file()
    with config.DATASET_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*features.tolist(), label])


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
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        y += 27


def main() -> int:
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
    current_label_name = "none"
    prev_time = time.perf_counter()
    fps = 0.0

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

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            faces = sort_faces_left_to_right(results.multi_face_landmarks or [])
            player2_face = faces[1] if len(faces) >= 2 else (faces[0] if len(faces) == 1 else None)
            player2_detected = len(faces) >= 2

            if len(faces) >= 1:
                draw_face_box(frame, faces[0], "Player 1" if len(faces) >= 2 else "Single face", (0, 220, 255))
            if len(faces) >= 2:
                draw_face_box(frame, faces[1], "Player 2 - dataset source", (80, 255, 80))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("n"), ord("b")):
                if player2_face is None:
                    print("No face detected. Sample not saved.")
                else:
                    label = config.LABEL_NEUTRAL if key == ord("n") else config.LABEL_BONUS
                    features = extract_features(player2_face)
                    append_sample(features, label)
                    counts[label] += 1
                    current_label_name = config.LABEL_NAMES[label]
                    print(f"Saved {current_label_name}: neutral={counts[config.LABEL_NEUTRAL]}, bonus={counts[config.LABEL_BONUS]}")
            elif key == ord("q"):
                break

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            status_color = (80, 255, 80) if player2_detected else (0, 180, 255)
            status_text = "Dataset source: Player 2 (right face)" if player2_detected else "Dataset source: single-player fallback"
            if player2_face is None:
                status_text = "WARNING: no face detected; sample disabled"
                status_color = (0, 80, 255)

            put_lines(
                frame,
                [
                    (f"FPS: {fps:.1f}", (255, 255, 255)),
                    (f"Faces detected: {len(faces)}", (255, 255, 255)),
                    (status_text, status_color),
                    (f"Current class: {current_label_name}", (255, 255, 255)),
                    (f"neutral: {counts[config.LABEL_NEUTRAL]}", (255, 255, 255)),
                    (f"bonus_gesture: {counts[config.LABEL_BONUS]}", (255, 255, 255)),
                    ("n = save neutral | b = save bonus_gesture | q = quit", (230, 230, 230)),
                ],
            )

            cv2.imshow("Blobby Face Controller - collect dataset", frame)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
