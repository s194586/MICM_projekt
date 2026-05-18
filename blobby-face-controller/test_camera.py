"""Camera and two-face MediaPipe test."""

from __future__ import annotations

import time

import cv2
import mediapipe as mp

import config
from feature_extraction import face_bbox, sort_faces_left_to_right


def draw_face_label(frame, landmarks, label: str, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    min_x, min_y, max_x, max_y = face_bbox(landmarks)
    x1 = int(min_x * width)
    y1 = int(min_y * height)
    x2 = int(max_x * width)
    y2 = int(max_y * height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def main() -> int:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {config.CAMERA_INDEX}. Check config.py.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    mp_face_mesh = mp.solutions.face_mesh
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
            if len(faces) >= 1:
                draw_face_label(frame, faces[0], "Player 1", (0, 220, 255))
            if len(faces) >= 2:
                draw_face_label(frame, faces[1], "Player 2", (80, 255, 80))

            if len(faces) == 1:
                cv2.putText(frame, "Only one player detected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            elif len(faces) == 0:
                cv2.putText(frame, "No face detected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Blobby Face Controller - camera test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
