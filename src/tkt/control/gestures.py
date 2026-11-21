"""El karelerinden girdi olaylarına geçişi yöneten jest motoru."""

from collections.abc import Mapping
from enum import Enum
from threading import RLock
from typing import Any

from tkt.control.os_input.base import InputBackend
from tkt.control.pointer import PointerMotion
from tkt.control.scroll import ScrollMotion
from tkt.core.types import FINGERS, Finger, HandFrame, SourceInfo


class Mode(Enum):
    """Jest motorunun çalışma modları."""

    NONE = 0
    WAIT = 1
    POINTER = 2
    SCROLL = 3
    SWIPE = 4


_MODES_BY_FINGERS = {
    ("index",): Mode.POINTER,
    ("index", "middle"): Mode.SCROLL,
    ("index", "middle", "ring"): Mode.SWIPE,
    ("index", "middle", "ring", "pinky"): Mode.SWIPE,
}


class GestureEngine:
    """El durumunu izler ve backend'e girdi olayları gönderir."""

    mode: Mode
    left_down: bool

    def __init__(self, cfg: Mapping[str, Any], backend: InputBackend, info: SourceInfo) -> None:
        """Control yapılandırmasını, backend'i ve kaynak bilgisini al."""
        general = cfg["general"]
        mode = cfg["mode"]
        self._confidence_min = general["confidence_min"]
        self._lost_release_s = general["lost_release_ms"] / 1000
        self._lock_s = mode["lock_ms"] / 1000
        self._curl_open = mode["curl_open"]
        self._curl_closed = mode["curl_closed"]
        self._freeze_curl_rate = mode["freeze_curl_rate"]
        self._rollback_ms = cfg["pointer"]["rollback_ms"]
        self._backend = backend
        self._pointer = PointerMotion(cfg["pointer"], info)
        self._scroll = ScrollMotion(cfg["scroll"], info)
        self._lock = RLock()
        self._reset_state()

    def process(self, frame: HandFrame) -> None:
        """Bir el karesini işle."""
        with self._lock:
            if not frame.present or frame.confidence < self._confidence_min:
                self._process_missing_hand(frame.t)
                return

            if self._lost_until is not None and frame.t >= self._lost_until:
                self.release()
            self._lost_until = None
            self._lost_released = False
            index = frame.fingers["index"]
            # Hızlı kıvrılma aynı karede modu kapatsa bile geri almayı gönder.
            curling_fast = self._check_curl_rate(index.curl, frame.t)
            self._update_mode(frame)
            if self.mode is Mode.POINTER:
                self._move_pointer(index, frame.hand_size, frame.t, curling_fast)

    def release(self) -> None:
        """Tuşları bırak ve jest durumunu sıfırla."""
        with self._lock:
            self._backend.release_all()
            self._pointer.reset()
            self._reset_state()

    def _reset_state(self) -> None:
        self.mode = Mode.NONE
        self.left_down = False
        self._open_fingers: tuple[str, ...] = ()
        self._wait_until: float | None = None
        self._lost_until: float | None = None
        self._lost_released = False
        self._previous_index: tuple[float, float] | None = None
        self._pointer_frozen = False
        self._pointer_needs_rebase = True

    def _process_missing_hand(self, t: float) -> None:
        if self._lost_released:
            return
        if self._lost_until is None:
            self._lost_until = t + self._lost_release_s
            self._wait_until = None
            self._previous_index = None
            self._pointer_needs_rebase = True
        if t >= self._lost_until:
            self.release()
            self._lost_released = True

    def _update_mode(self, frame: HandFrame) -> None:
        opened = tuple(
            name
            for name in FINGERS
            if frame.fingers[name].curl < self._curl_open
            or (name in self._open_fingers and frame.fingers[name].curl <= self._curl_closed)
        )
        if not opened:
            if self.mode is Mode.POINTER:
                self._pointer.reset()
            self.mode = Mode.NONE
            self._wait_until = None
            self._pointer_frozen = False
            self._pointer_needs_rebase = True
        elif self.mode in (Mode.NONE, Mode.WAIT):
            if opened != self._open_fingers or self._wait_until is None:
                self._wait_until = frame.t + self._lock_s
            self.mode = Mode.WAIT
            if frame.t >= self._wait_until:
                self.mode = _MODES_BY_FINGERS.get(opened, Mode.WAIT)
        self._open_fingers = opened

    def _check_curl_rate(self, curl: float, t: float) -> bool:
        previous = self._previous_index
        self._previous_index = (curl, t)
        if self.mode is not Mode.POINTER or previous is None:
            return False
        previous_curl, previous_t = previous
        dt = t - previous_t
        curling_fast = dt > 0 and (curl - previous_curl) / dt > self._freeze_curl_rate
        if curling_fast and not self._pointer_frozen:
            self._pointer_frozen = True
            self._pointer_needs_rebase = True
            self._send_move(self._pointer.rollback(self._rollback_ms))
        return curling_fast

    def _move_pointer(self, index: Finger, hand_size: float, t: float, curling_fast: bool) -> None:
        if "index" not in self._open_fingers:
            self._pointer_needs_rebase = True
            return
        if self._pointer_frozen:
            if index.curl >= self._curl_open or curling_fast:
                return
            self._pointer_frozen = False
        if self._pointer_needs_rebase:
            self._pointer.rebase()
            self._pointer_needs_rebase = False
        self._send_move(self._pointer.update(index.img, hand_size, t))

    def _send_move(self, delta: tuple[int, int]) -> None:
        if delta != (0, 0):
            self._backend.move_rel(*delta)
