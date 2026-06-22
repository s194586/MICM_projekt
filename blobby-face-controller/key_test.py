"""Send configured controls to a focused text field without using a camera."""

from __future__ import annotations

import time

import config
from keyboard_utils import resolve_key, tap_key

try:
    from pynput.keyboard import Controller
except Exception as exc:  # pragma: no cover - depends on the local desktop.
    Controller = None
    PYNPUT_IMPORT_ERROR = exc
else:
    PYNPUT_IMPORT_ERROR = None


def main() -> int:
    if Controller is None:
        print(f"ERROR: pynput keyboard controller is unavailable: {PYNPUT_IMPORT_ERROR}")
        return 1

    try:
        keyboard = Controller()
    except Exception as exc:
        print(f"ERROR: Cannot create pynput keyboard controller: {exc}")
        return 1
    controls = [
        ("MOVE_LEFT_KEY", resolve_key(config.MOVE_LEFT_KEY)),
        ("MOVE_RIGHT_KEY", resolve_key(config.MOVE_RIGHT_KEY)),
        ("JUMP_KEY", resolve_key(config.JUMP_KEY)),
        ("BONUS_KEY", resolve_key(config.BONUS_KEY)),
    ]

    print("Click Notepad or another text field now. Key test starts in 3 seconds...")
    time.sleep(3.0)
    try:
        for name, key in controls:
            print(f"Testing {name}: {key}")
            tap_key(keyboard, key, duration=0.12)
            time.sleep(0.25)
    finally:
        for _, key in controls:
            try:
                keyboard.release(key)
            except Exception:
                pass

    print("Key test complete. Expected default text: adw followed by a space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
