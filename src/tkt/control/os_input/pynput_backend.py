"""pynput ile işletim sistemine girdi gönderen backend."""

import sys
from contextlib import ExitStack
from math import isfinite, trunc

# Sistem diktesini aç/kapat eylemleri (ikisi de aynı kısayolu gönderir).
DICTATION_ACTIONS = ("dictation_start", "dictation_stop", "dictation_submit")


class PynputBackend:
    """pynput bağımlılığını yalnızca örnek oluşturulurken yükler."""

    step_px: float | None = None

    def __init__(self) -> None:
        """Girdi denetleyicilerini hazırla."""
        from pynput import keyboard, mouse

        self._mouse = mouse.Controller()
        self._keyboard = keyboard.Controller()
        self._keys = keyboard.Key
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right}
        self._pressed: dict[str, None] = {}
        self._scroll_x = self._scroll_y = 0.0
        self.events_sent = 0

    def move_rel(self, dx: int, dy: int) -> None:
        self._mouse.move(dx, dy)
        self.events_sent += 1

    def button(self, name: str, down: bool) -> None:
        button = self._buttons[name]
        if down:
            self._pressed[name] = None
            self._mouse.press(button)
        else:
            self._mouse.release(button)
            self._pressed.pop(name, None)
        self.events_sent += 1

    def scroll(self, dx: float, dy: float) -> None:
        step = self.step_px
        if step is None or not isfinite(step) or step <= 0:
            raise ValueError("scroll.step_px pozitif ve sonlu olmalı")
        x, y = self._scroll_x + dx, self._scroll_y + dy
        sx, sy = trunc(x / step), trunc(y / step)
        if sx or sy:
            # Windows pozitif yatay tekerlekle görünümü sağa kaydırır.
            self._mouse.scroll(-sx if sys.platform == "win32" else sx, sy)
            self.events_sent += 1
        self._scroll_x, self._scroll_y = x - sx * step, y - sy * step

    def system_action(self, name: str) -> None:
        key = self._keys
        if name == "dictation_submit":
            from tkt.control.os_input import dictation

            dictation.submit_later(lambda: self._dictation_toggle(False), self._enter)
            return
        if name in DICTATION_ACTIONS:
            self._dictation_toggle(name == "dictation_start")
            return
        if sys.platform == "win32":
            chords = {
                "mission_control": (key.cmd, key.tab),
                "desktop_left": (key.cmd, key.ctrl, key.left),
                "desktop_right": (key.cmd, key.ctrl, key.right),
            }
        elif sys.platform == "darwin":
            chords = {
                "mission_control": (key.ctrl, key.up),
                "desktop_left": (key.ctrl, key.left),
                "desktop_right": (key.ctrl, key.right),
            }
        else:
            raise RuntimeError(f"Sistem eylemleri bu platformda desteklenmiyor: {sys.platform}")

        with ExitStack() as releases:
            for key in chords[name]:
                releases.callback(self._key, key, False)
                self._key(key, True)

    def _enter(self) -> None:
        self._key(self._keys.enter, True)
        self._key(self._keys.enter, False)

    def _dictation_toggle(self, start: bool):
        """Sistem diktesini aç/kapat: macOS menüden (Düzen > Dikte), Windows Win+H.

        macOS'ta başlatılan osascript süreci döner (dikte bitirmede beklenir).
        """
        if sys.platform == "darwin":
            from tkt.control.os_input import mac_dictation

            proc = mac_dictation.toggle(start)
            self.events_sent += 1
            return proc
        elif sys.platform == "win32":
            with ExitStack() as releases:
                for k in (self._keys.cmd, "h"):
                    releases.callback(self._key, k, False)
                    self._key(k, True)
        else:
            raise RuntimeError(f"Dikte bu platformda desteklenmiyor: {sys.platform}")

    def _key(self, key, down: bool) -> None:
        send = self._keyboard.press if down else self._keyboard.release
        send(key)
        self.events_sent += 1

    def release_all(self) -> None:
        self._scroll_x = self._scroll_y = 0.0
        errors = []
        for name in tuple(self._pressed):
            try:
                self.button(name, False)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Fare tuşları bırakılamadı", errors)
