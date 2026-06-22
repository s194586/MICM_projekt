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


def hold_combo(keyboard, label: str, keys: list, duration: float = 1.0) -> None:
    """Hold several keys together and release them in reverse order."""
    print(f"Testing combo: {label} ({duration:.1f}s hold)")
    for key in keys:
        keyboard.press(key)
    try:
        time.sleep(duration)
    finally:
        for key in reversed(keys):
            keyboard.release(key)


def hold_with_bonus_tap(keyboard, label: str, held_keys: list, bonus_key) -> None:
    """Keep movement/jump held while independently tapping the bonus key."""
    print(f"Testing combo: {label}")
    for key in held_keys:
        keyboard.press(key)
    try:
        time.sleep(0.25)
        tap_key(keyboard, bonus_key, duration=0.12)
        time.sleep(0.50)
    finally:
        for key in reversed(held_keys):
            keyboard.release(key)


def main() -> int:
    if Controller is None:
        print(f"ERROR: pynput keyboard controller is unavailable: {PYNPUT_IMPORT_ERROR}")
        return 1

    try:
        keyboard = Controller()
    except Exception as exc:
        print(f"ERROR: Cannot create pynput keyboard controller: {exc}")
        return 1
    left_key = resolve_key(config.MOVE_LEFT_KEY)
    right_key = resolve_key(config.MOVE_RIGHT_KEY)
    jump_key = resolve_key(config.JUMP_KEY)
    bonus_key = resolve_key(config.BONUS_KEY)
    controls = [left_key, right_key, jump_key, bonus_key]

    print("Click Notepad or another text field now. Combo test starts in 3 seconds...")
    time.sleep(3.0)
    try:
        hold_combo(keyboard, "A + W", [left_key, jump_key])
        time.sleep(0.35)
        hold_combo(keyboard, "D + W", [right_key, jump_key])
        time.sleep(0.35)
        hold_with_bonus_tap(keyboard, "A + Space", [left_key], bonus_key)
        time.sleep(0.35)
        hold_with_bonus_tap(keyboard, "D + Space", [right_key], bonus_key)
        time.sleep(0.35)
        hold_with_bonus_tap(keyboard, "A + W + Space", [left_key, jump_key], bonus_key)
        time.sleep(0.35)
        hold_with_bonus_tap(keyboard, "D + W + Space", [right_key, jump_key], bonus_key)
    finally:
        for key in controls:
            try:
                keyboard.release(key)
            except Exception:
                pass

    print("Combo test complete. Verify the printed sequence in the focused application.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
