"""Low-latency keyboard backend for the non-MediaPipe lite controller."""

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


KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1
MAPVK_VK_TO_VSC = 0
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

VK_CODES = {
    "a": 0x41,
    "d": 0x44,
    "w": 0x57,
    "space": 0x20,
}

PYNPUT_SPECIAL_KEYS = {
    "space": "space",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
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
        self._map_virtual_key = self._user32.MapVirtualKeyW
        self._map_virtual_key.argtypes = (wintypes.UINT, wintypes.UINT)
        self._map_virtual_key.restype = wintypes.UINT
        self.status = BackendStatus(name="win32", enabled=True, error="")

    def press(self, key_name: str) -> None:
        self._emit(key_name, is_key_up=False)

    def release(self, key_name: str) -> None:
        self._emit(key_name, is_key_up=True)

    def _emit(self, key_name: str, is_key_up: bool) -> None:
        if not self.status.enabled:
            return

        normalized = key_name.lower()
        vk_code = VK_CODES.get(normalized)
        if vk_code is None:
            self.status.enabled = False
            self.status.error = f"Unsupported win32 key: {key_name}"
            return

        scan_code = self._map_virtual_key(vk_code, MAPVK_VK_TO_VSC)
        if scan_code == 0:
            self.status.enabled = False
            self.status.error = f"Cannot map scan code for: {key_name}"
            return

        flags = KEYEVENTF_SCANCODE
        if is_key_up:
            flags |= KEYEVENTF_KEYUP

        event = INPUT(type=INPUT_KEYBOARD, union=INPUTUNION(ki=KEYBDINPUT(0, scan_code, flags, 0, 0)))
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
        self._backend = self._build_backend(preferred_backend)
        self._held_keys: set[str] = set()
        self._pending_taps: dict[str, float] = {}

    def press(self, key_name: str) -> None:
        self._backend.press(key_name)

    def release(self, key_name: str) -> None:
        self._backend.release(key_name)

    def hold(self, key_name: str) -> None:
        if key_name not in self._held_keys:
            self.press(key_name)
            self._held_keys.add(key_name)

    def release_hold(self, key_name: str) -> None:
        if key_name in self._held_keys:
            self._held_keys.remove(key_name)
            self.release(key_name)

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
            return
        if release_at > current_release:
            self._pending_taps[key_name] = release_at

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
            return self._backend.status.name
        if self._backend.status.error:
            return f"{self._backend.status.name} disabled: {self._backend.status.error[:48]}"
        return f"{self._backend.status.name} disabled"

    def _build_backend(self, preferred_backend: str):
        normalized = preferred_backend.strip().lower()
        backends = []

        if normalized == "win32":
            try:
                backend = Win32KeyboardBackend()
                if backend.status.enabled:
                    return backend
                backends.append(backend)
            except Exception as exc:  # pragma: no cover - platform dependent.
                backends.append(type("FallbackStatus", (), {"status": BackendStatus("win32", False, str(exc))})())
        if normalized in ("win32", "pynput"):
            backend = PynputKeyboardBackend()
            if backend.status.enabled:
                return backend
            backends.append(backend)

        if backends:
            return backends[-1]
        return PynputKeyboardBackend()
