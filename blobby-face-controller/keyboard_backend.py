"""Low-latency keyboard backend for the model-based fast controller."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

try:
    from pynput.keyboard import Controller as PynputController
    from pynput.keyboard import Key

    PYNPUT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - desktop dependent.
    PynputController = None
    Key = None
    PYNPUT_IMPORT_ERROR = exc


KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
INPUT_HARDWARE = 2
ULONG_PTR = wintypes.WPARAM

PYNPUT_SPECIAL_KEYS = {
    "space": "space",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
}


@dataclass(frozen=True, slots=True)
class KeySpec:
    scan_code: int
    display_name: str
    extended: bool = False


KEY_SPECS = {
    "a": KeySpec(0x1E, "A"),
    "d": KeySpec(0x20, "D"),
    "w": KeySpec(0x11, "W"),
    "space": KeySpec(0x39, "SPACE"),
    "left": KeySpec(0x4B, "LEFT", extended=True),
    "right": KeySpec(0x4D, "RIGHT", extended=True),
    "up": KeySpec(0x48, "UP", extended=True),
    "down": KeySpec(0x50, "DOWN", extended=True),
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]


@dataclass(slots=True)
class BackendStatus:
    name: str
    enabled: bool
    error: str = ""


class Win32KeyboardBackend:
    """Send key events through Win32 SendInput."""

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._send_input = self._user32.SendInput
        self._send_input.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self._send_input.restype = wintypes.UINT
        self.status = BackendStatus(name="win32", enabled=True, error="")

    def press(self, key_name: str) -> None:
        self._emit(key_name, is_key_up=False)

    def release(self, key_name: str) -> None:
        self._emit(key_name, is_key_up=True)

    def _emit(self, key_name: str, is_key_up: bool) -> None:
        if not self.status.enabled:
            return

        normalized = key_name.lower()
        key_spec = KEY_SPECS.get(normalized)
        if key_spec is None:
            self.status.enabled = False
            self.status.error = f"Unsupported win32 key: {key_name}"
            return

        flags = KEYEVENTF_SCANCODE
        if key_spec.extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        if is_key_up:
            flags |= KEYEVENTF_KEYUP

        event = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, key_spec.scan_code, flags, 0, 0))
        sent = self._send_input(1, ctypes.byref(event), ctypes.sizeof(INPUT))
        if sent != 1:
            self.status.enabled = False
            self.status.error = str(ctypes.WinError(ctypes.get_last_error()))


class PynputKeyboardBackend:
    """Fallback keyboard backend based on pynput."""

    def __init__(self) -> None:
        self._controller = None
        error = str(PYNPUT_IMPORT_ERROR) if PYNPUT_IMPORT_ERROR else ""
        enabled = False
        if PynputController is not None and PYNPUT_IMPORT_ERROR is None:
            try:
                self._controller = PynputController()
                enabled = True
                error = ""
            except Exception as exc:  # pragma: no cover - desktop dependent.
                error = str(exc)
        self.status = BackendStatus(name="pynput", enabled=enabled, error=error)

    def press(self, key_name: str) -> None:
        key = self._resolve(key_name)
        if not self.status.enabled or self._controller is None:
            return
        try:
            self._controller.press(key)
        except Exception as exc:  # pragma: no cover - desktop dependent.
            self.status.enabled = False
            self.status.error = str(exc)

    def release(self, key_name: str) -> None:
        key = self._resolve(key_name)
        if self._controller is None:
            return
        try:
            self._controller.release(key)
        except Exception as exc:  # pragma: no cover - desktop dependent.
            self.status.enabled = False
            self.status.error = str(exc)

    def _resolve(self, key_name: str):
        normalized = key_name.strip().lower()
        special_name = PYNPUT_SPECIAL_KEYS.get(normalized)
        if special_name is not None and Key is not None:
            return getattr(Key, special_name)
        if len(normalized) == 1:
            return normalized
        return key_name


class KeyboardController:
    """Stateful hold/tap helper for low-latency output."""

    def __init__(self, preferred_backend: str = "win32") -> None:
        self._preferred_backend = preferred_backend.strip().lower()
        self._fallback_active = False
        self._failover_warned = False
        self._backend = self._build_backend(self._preferred_backend)
        self._held_keys: set[str] = set()
        self._pending_taps: dict[str, float] = {}
        self._last_event = "NONE"

    def press(self, key_name: str) -> None:
        self._backend.press(key_name)
        self._recover_backend_if_needed()

    def release(self, key_name: str) -> None:
        self._backend.release(key_name)
        self._recover_backend_if_needed()

    def hold(self, key_name: str) -> None:
        if key_name not in self._held_keys:
            self.press(key_name)
            self._held_keys.add(key_name)
            self._last_event = f"{self._event_label(key_name)}_DOWN"

    def release_hold(self, key_name: str) -> None:
        if key_name in self._held_keys:
            self._held_keys.remove(key_name)
            self.release(key_name)
            self._last_event = f"{self._event_label(key_name)}_UP"

    def set_movement(self, action: str, left_key: str, right_key: str) -> None:
        if action == "LEFT":
            self.release_hold(right_key)
            self.hold(left_key)
        elif action == "RIGHT":
            self.release_hold(left_key)
            self.hold(right_key)
        else:
            self.release_hold(left_key)
            self.release_hold(right_key)

    def set_hold(self, key_name: str, should_hold: bool) -> None:
        if should_hold:
            self.hold(key_name)
        else:
            self.release_hold(key_name)

    def tap(self, key_name: str, now: float, duration: float) -> None:
        release_at = now + duration
        current_release = self._pending_taps.get(key_name)
        if current_release is None:
            self.press(key_name)
            self._pending_taps[key_name] = release_at
        elif release_at > current_release:
            self._pending_taps[key_name] = release_at
        self._last_event = f"{self._event_label(key_name)}_TAP"

    def update(self, now: float) -> None:
        due_keys = [key_name for key_name, release_at in self._pending_taps.items() if now >= release_at]
        for key_name in due_keys:
            self._pending_taps.pop(key_name, None)
            if key_name not in self._held_keys:
                self.release(key_name)

    def release_all(self) -> None:
        keys_to_release = set(self._held_keys)
        keys_to_release.update(self._pending_taps)
        self._held_keys.clear()
        self._pending_taps.clear()
        for key_name in keys_to_release:
            self.release(key_name)

    def status_text(self) -> str:
        if self._backend.status.enabled:
            if self._fallback_active:
                return "pynput fallback"
            return self._backend.status.name
        if self._backend.status.error:
            return f"{self._backend.status.name} disabled: {self._backend.status.error[:48]}"
        return f"{self._backend.status.name} disabled"

    def backend_name(self) -> str:
        return self._backend.status.name

    def last_event(self) -> str:
        return self._last_event

    def _event_label(self, key_name: str) -> str:
        key_spec = KEY_SPECS.get(key_name.strip().lower())
        if key_spec is not None:
            return key_spec.display_name
        return key_name.strip().upper()

    def _build_backend(self, preferred_backend: str):
        normalized = preferred_backend.strip().lower()

        if normalized == "win32":
            try:
                backend = Win32KeyboardBackend()
                if backend.status.enabled:
                    return backend
            except Exception as exc:  # pragma: no cover - platform dependent.
                backend = type("FallbackStatus", (), {"status": BackendStatus("win32", False, str(exc))})()
            fallback = PynputKeyboardBackend()
            if fallback.status.enabled:
                self._fallback_active = True
                return fallback
            return backend

        if normalized == "pynput":
            backend = PynputKeyboardBackend()
            if backend.status.enabled:
                return backend
            return backend

        return PynputKeyboardBackend()

    def _recover_backend_if_needed(self) -> None:
        if self._backend.status.enabled:
            return
        if self._preferred_backend != "win32":
            return
        if self._fallback_active:
            return

        fallback = PynputKeyboardBackend()
        if not fallback.status.enabled:
            return

        error_message = self._backend.status.error or "unknown win32 error"
        self._backend = fallback
        self._fallback_active = True
        if not self._failover_warned:
            print(f"Win32 keyboard failed, switching to pynput fallback: {error_message}")
            self._failover_warned = True
