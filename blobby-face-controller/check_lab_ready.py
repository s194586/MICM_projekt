"""Quick dependency, artifact and camera sanity check for lab machines."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import config


BASE_DIR = Path(__file__).resolve().parent
errors: list[str] = []
warnings: list[str] = []


def report_ok(message: str) -> None:
    print(f"[OK] {message}")


def report_error(message: str) -> None:
    errors.append(message)
    print(f"[ERROR] {message}")


def report_warning(message: str) -> None:
    warnings.append(message)
    print(f"[WARNING] {message}")


def import_dependency(module_name: str, display_name: str | None = None):
    label = display_name or module_name
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        report_error(f"Cannot import {label}: {exc}")
        return None
    version = getattr(module, "__version__", "version unknown")
    report_ok(f"Import {label}: {version}")
    return module


def check_file(path: Path, label: str) -> bool:
    if not path.exists():
        report_error(f"Missing {label}: {path}")
        return False
    report_ok(f"{label}: {path} ({path.stat().st_size} bytes)")
    return True


def main() -> int:
    print("Blobby Face Controller - lab readiness check")
    print(f"Project: {BASE_DIR}")
    print()

    python_version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 10):
        report_error(f"Python {python_version} is too old; Python 3.11 is recommended.")
    else:
        report_ok(f"Python {python_version} ({sys.executable})")

    cv2 = import_dependency("cv2", "opencv-python/cv2")
    mediapipe = import_dependency("mediapipe")
    import_dependency("sklearn", "scikit-learn/sklearn")
    joblib = import_dependency("joblib")
    import_dependency("numpy")
    import_dependency("pandas")
    import_dependency("pynput")

    if mediapipe is not None:
        mp_version = getattr(mediapipe, "__version__", "unknown")
        has_legacy_face_mesh = hasattr(mediapipe, "solutions") and hasattr(
            mediapipe.solutions, "face_mesh"
        )
        if not has_legacy_face_mesh:
            report_error("MediaPipe legacy API mp.solutions.face_mesh is unavailable.")
        elif mp_version == "0.10.21":
            report_ok("MediaPipe 0.10.21 legacy Face Mesh API is available.")
        else:
            report_warning(
                f"MediaPipe {mp_version} exposes legacy Face Mesh, but 0.10.21 is recommended."
            )

    model_exists = check_file(config.MODEL_PATH, "bonus model")
    if model_exists and joblib is not None:
        try:
            payload = joblib.load(config.MODEL_PATH)
            model = payload.get("model") if isinstance(payload, dict) else payload
            if model is None:
                report_error("bonus_model.pkl loaded, but it does not contain a model.")
            else:
                report_ok(f"bonus_model.pkl loads correctly ({type(model).__name__}).")
        except Exception as exc:
            report_error(f"Cannot load bonus_model.pkl: {exc}")

    check_file(config.DATASET_PATH, "gesture dataset")
    check_file(config.REPORTS_DIR / "validation_metrics.txt", "validation metrics")
    check_file(config.REPORTS_DIR / "confusion_matrix.png", "confusion matrix")

    if cv2 is not None:
        camera = None
        try:
            camera = cv2.VideoCapture(config.CAMERA_INDEX)
            if camera.isOpened():
                report_ok(f"Camera index {config.CAMERA_INDEX} opens successfully.")
            else:
                report_error(
                    f"Camera index {config.CAMERA_INDEX} did not open. "
                    "Close other camera apps or adjust CAMERA_INDEX in config.py."
                )
        except Exception as exc:
            report_error(f"Camera check failed: {exc}")
        finally:
            if camera is not None:
                camera.release()

    print()
    if errors:
        print(f"NOT READY: {len(errors)} error(s), {len(warnings)} warning(s).")
        for message in errors:
            print(f" - {message}")
        return 1

    print(f"READY: all required checks passed ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
