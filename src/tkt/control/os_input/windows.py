"""Windows user32 ile işletim sistemine girdi gönderir."""

import ctypes
import sys
from contextlib import ExitStack
from math import isfinite, trunc

# Win32 LONG ve DWORD, Python'un çalıştığı platformdan bağımsız olarak 32 bittir.
_LONG = ctypes.c_int32
_DWORD = ctypes.c_uint32
_WORD = ctypes.c_uint16
_ULONG_PTR = ctypes.c_size_t

_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_BUTTON_FLAGS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010)}
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x1000
_WHEEL_DELTA = 120

_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_VK_TAB = 0x09
_VK_CONTROL = 0x11
_VK_LEFT = 0x25
_VK_RIGHT = 0x27
_VK_LWIN = 0x5B
_VK_H = 0x48
_VK_C = 0x43
_VK_RETURN = 0x0D
_CHORDS = {
    # Sistem diktesi (Windows sesle yazma): Win+H aç/kapat.
    "dictation_start": (_VK_LWIN, _VK_H),
    "dictation_stop": (_VK_LWIN, _VK_H),
    "mission_control": (_VK_LWIN, _VK_TAB),
    "desktop_left": (_VK_LWIN, _VK_CONTROL, _VK_LEFT),
    "desktop_right": (_VK_LWIN, _VK_CONTROL, _VK_RIGHT),
    "key_left": (_VK_LEFT,),
    "key_right": (_VK_RIGHT,),
    "key_tab": (_VK_TAB,),
    "key_c": (_VK_C,),
}


class _Point(ctypes.Structure):
    _fields_ = [("x", _LONG), ("y", _LONG)]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", _LONG),
        ("dy", _LONG),
        ("mouseData", _DWORD),
        ("dwFlags", _DWORD),
        ("time", _DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", _WORD),
        ("wScan", _WORD),
        ("dwFlags", _DWORD),
        ("time", _DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", _DWORD), ("wParamL", _WORD), ("wParamH", _WORD)]


class _InputData(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", _DWORD), ("data", _InputData)]


class WindowsBackend:
    """İmleç, fare tuşları, kaydırma ve Windows kısayolları gönderir."""

    def __init__(self, step_px: float, smooth: bool = True) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsBackend yalnızca Windows'ta çalışır")
        if not isfinite(step_px) or step_px <= 0:
            raise ValueError("scroll.step_px pozitif ve sonlu olmalı")

        self._user32 = ctypes.windll.user32
        self._user32.GetCursorPos.argtypes = (ctypes.POINTER(_Point),)
        self._user32.GetCursorPos.restype = ctypes.c_int
        self._user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
        self._user32.SetCursorPos.restype = ctypes.c_int
        self._user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
        self._user32.GetSystemMetrics.restype = ctypes.c_int
        self._user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(_Input), ctypes.c_int)
        self._user32.SendInput.restype = ctypes.c_uint
        self._step_px = step_px
        self._wheel_quantum = 1 if smooth else _WHEEL_DELTA
        self._wheel_remainder = {_MOUSEEVENTF_HWHEEL: 0.0, _MOUSEEVENTF_WHEEL: 0.0}
        self._pressed_buttons: dict[str, None] = {}
        self._pressed_keys: dict[int, None] = {}
        self.events_sent = 0

    def move_rel(self, dx: int, dy: int) -> None:
        point = _Point()
        if not self._user32.GetCursorPos(ctypes.pointer(point)):
            raise OSError("GetCursorPos başarısız")
        left = self._user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
        top = self._user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
        width = self._user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
        height = self._user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
        x = max(left, min(point.x + dx, left + width - 1))
        y = max(top, min(point.y + dy, top + height - 1))
        if not self._user32.SetCursorPos(x, y):
            raise OSError("SetCursorPos başarısız")
        self.events_sent += 1

    def _send(self, event: _Input) -> None:
        if self._user32.SendInput(1, ctypes.pointer(event), ctypes.sizeof(_Input)) != 1:
            raise OSError("SendInput girdiyi gönderemedi")
        self.events_sent += 1

    def _mouse(self, flags: int, data: int = 0) -> None:
        event = _MouseInput(mouseData=data, dwFlags=flags)
        self._send(_Input(_INPUT_MOUSE, _InputData(mi=event)))

    def button(self, name: str, down: bool) -> None:
        press, release = _BUTTON_FLAGS[name]
        self._mouse(press if down else release)
        if down:
            self._pressed_buttons[name] = None
        else:
            self._pressed_buttons.pop(name, None)

    def scroll(self, dx: float, dy: float) -> None:
        for flag, pixels in ((_MOUSEEVENTF_HWHEEL, -dx), (_MOUSEEVENTF_WHEEL, dy)):
            units = self._wheel_remainder[flag] + pixels / self._step_px * _WHEEL_DELTA
            sent = trunc(units / self._wheel_quantum) * self._wheel_quantum
            if sent:
                self._mouse(flag, sent)
            self._wheel_remainder[flag] = units - sent

    def _key(self, key: int, down: bool) -> None:
        flags = _KEYEVENTF_EXTENDEDKEY if key in (_VK_LWIN, _VK_LEFT, _VK_RIGHT) else 0
        if not down:
            flags |= _KEYEVENTF_KEYUP
        event = _KeyboardInput(wVk=key, dwFlags=flags)
        self._send(_Input(_INPUT_KEYBOARD, _InputData(ki=event)))
        if down:
            self._pressed_keys[key] = None
        else:
            self._pressed_keys.pop(key, None)

    def system_action(self, name: str) -> None:
        if name == "dictation_submit":
            from tkt.control.os_input import dictation

            dictation.submit_later(lambda: self.system_action("dictation_stop"), self._enter)
            return
        with ExitStack() as releases:
            for key in _CHORDS[name]:
                self._key(key, True)
                releases.callback(self._key, key, False)

    def _enter(self) -> None:
        self._key(_VK_RETURN, True)
        self._key(_VK_RETURN, False)

    def release_all(self) -> None:
        for flag in self._wheel_remainder:
            self._wheel_remainder[flag] = 0.0
        errors = []
        for release, pressed in (
            (self._key, self._pressed_keys),
            (self.button, self._pressed_buttons),
        ):
            for key in reversed(tuple(pressed)):
                try:
                    release(key, False)
                except OSError as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("Windows girdi tuşları bırakılamadı", errors)
