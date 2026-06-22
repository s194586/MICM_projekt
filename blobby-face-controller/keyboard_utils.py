"""Shared pynput key resolution and short key taps."""

from __future__ import annotations

import time

try:
    from pynput.keyboard import Key
except Exception:  # pragma: no cover - depends on the local desktop.
    Key = None


SPECIAL_KEY_NAMES = {
    "space": "space",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
}


def resolve_key(key_name):
    """Map configured special-key names to pynput Key objects."""
    if not isinstance(key_name, str):
        return key_name

    normalized = key_name.strip().lower()
    special_name = SPECIAL_KEY_NAMES.get(normalized)
    if special_name is not None and Key is not None:
        return getattr(Key, special_name)
    if len(normalized) == 1:
        return normalized
    return key_name


def tap_key(keyboard, key, duration: float = 0.05) -> None:
    """Press and release one resolved key, always attempting the release."""
    keyboard.press(key)
    try:
        time.sleep(duration)
    finally:
        keyboard.release(key)
