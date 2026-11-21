"""El karelerinden girdi olaylarına geçişi yöneten jest motoru."""

from collections import deque
from collections.abc import Mapping
from enum import Enum
from math import hypot
from threading import RLock
from typing import Any

from tkt.control.os_input.base import InputBackend
from tkt.control.pointer import PointerMotion
from tkt.control.scroll import ScrollMotion
from tkt.control.tap import TapDetector
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
_TAP_FINGERS = {Mode.POINTER: ("index",), Mode.SCROLL: ("index", "middle")}


class GestureEngine:
    """El durumunu izler ve backend'e girdi olayları gönderir."""

    mode: Mode
    left_down: bool

    def __init__(self, cfg: Mapping[str, Any], backend: InputBackend, info: SourceInfo) -> None:
        """Control yapılandırmasını, backend'i ve kaynak bilgisini al."""
        general = cfg["general"]
        mode = cfg["mode"]
        thumb = cfg["thumb"]
        self._click_method = general.get("click_method", "thumb")
        self._confidence_min = general["confidence_min"]
        self._lost_release_s = general["lost_release_ms"] / 1000
        self._lock_s = mode["lock_ms"] / 1000
        self._curl_open = mode["curl_open"]
        self._curl_closed = mode["curl_closed"]
        self._freeze_curl_rate = mode["freeze_curl_rate"]
        self._thumb_close_threshold = thumb["closed"]
        self._thumb_open_threshold = thumb["open"]
        self._thumb_debounce_s = thumb["debounce_ms"] / 1000
        self._drag_threshold = thumb["drag_threshold"]
        self._still_s = thumb["still_ms"] / 1000
        self._image_aspect = info.height / info.width
        self._rollback_ms = cfg["pointer"]["rollback_ms"]
        self._backend = backend
        self._pointer = PointerMotion(cfg["pointer"], info)
        self._scroll = ScrollMotion(cfg["scroll"], info)
        self._tap = TapDetector(cfg) if self._click_method in ("tap", "both") else None
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
            index = frame.fingers["index"]
            if self._lost_until is not None:
                if self.mode is Mode.SCROLL:
                    self._scroll.rebase()
                if self._drag_origin is not None:
                    self._drag_origin = index.img
            self._lost_until = None
            self._lost_released = False
            # Hızlı kıvrılma aynı karede modu kapatsa bile geri almayı gönder.
            curling_fast = self._check_curl_rate(index.curl, frame.t)
            previous_mode = self.mode
            self._update_mode(frame)
            if self.mode is not previous_mode:
                self._change_mode(previous_mode, frame.t)
            self._update_tap(frame)
            self._record_palm(frame)
            self._update_thumb(frame)
            if self._tap_busy:
                return
            if self.mode is Mode.POINTER:
                self._move_pointer(index, frame.hand_size, frame.t, curling_fast)
            elif self.mode is Mode.SCROLL:
                middle = frame.fingers["middle"]
                pos = ((index.img[0] + middle.img[0]) / 2, (index.img[1] + middle.img[1]) / 2)
                self._send_scroll(self._scroll.update(pos, frame.hand_size, frame.t))
            elif self._scroll_coasting and previous_mode is Mode.NONE:
                self._send_scroll(self._scroll.tick(frame.t))

    def release(self) -> None:
        """Tuşları bırak ve jest durumunu sıfırla."""
        with self._lock:
            self._backend.release_all()
            self._pointer.reset()
            self._stop_scroll()
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
        self._tap_busy = False
        if self._tap is not None:
            self._tap.reset()
        self._thumb_closed = False
        self._thumb_change_at: float | None = None
        self._drag_origin: tuple[float, float] | None = None
        self._palm_history: deque[tuple[float, tuple[float, float]]] = deque()
        self._scroll_coasting = False

    def _process_missing_hand(self, t: float) -> None:
        if self._lost_released:
            return
        if self._lost_until is None:
            self._lost_until = t + self._lost_release_s
            self._wait_until = None
            self._previous_index = None
            self._pointer_needs_rebase = True
            self._thumb_change_at = None
            self._palm_history.clear()
            self._cancel_active_tap()
            self._stop_scroll()
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

    def _change_mode(self, previous_mode: Mode, t: float) -> None:
        if self.left_down:
            self._release_left()
        self._drag_origin = None
        if previous_mode is Mode.SCROLL and not self._tap_busy:
            self._scroll.release(t)
            self._scroll_coasting = True
        elif self._scroll_coasting:
            self._stop_scroll()
        self._cancel_active_tap()
        if self.mode is Mode.SCROLL:
            self._scroll.rebase()

    def _cancel_active_tap(self) -> None:
        if self._tap is not None and self._tap_busy:
            self._tap.reset()
            self._tap_busy = False

    def _update_tap(self, frame: HandFrame) -> None:
        watched = _TAP_FINGERS.get(self.mode)
        if self._tap is None or watched is None or any(
            frame.fingers[name].lift is None for name in watched
        ):
            return
        update = self._tap.update(frame, watched)
        if update.started:
            self._send_move(self._pointer.rollback(self._rollback_ms))
            self._stop_scroll()
        if update.kind is not None and not self.left_down:
            button = "left" if update.kind == "tap1" else "right"
            self._backend.button(button, True)
            self._backend.button(button, False)
        if self._tap_busy and not update.busy:
            self._pointer.rebase()
            self._pointer_needs_rebase = False
            if self.mode is Mode.SCROLL:
                self._scroll.rebase()
        self._tap_busy = update.busy

    def _record_palm(self, frame: HandFrame) -> None:
        self._palm_history.append((frame.t, frame.palm))
        # Pencerenin başlangıcını kapsayan son örneği de tut.
        window_start = frame.t - self._still_s
        while len(self._palm_history) > 1 and self._palm_history[1][0] <= window_start:
            self._palm_history.popleft()

    def _hand_is_still(self, frame: HandFrame) -> bool:
        oldest_t, _ = self._palm_history[0]
        return (
            oldest_t <= frame.t - self._still_s
            and all(
                self._image_distance(palm, frame.palm) <= self._drag_threshold * frame.hand_size
                for _, palm in self._palm_history
            )
        )

    def _update_thumb(self, frame: HandFrame) -> None:
        # Eşiklerin arasında, varsa bekleyen geçiş de histerezisle korunur.
        closed = self._thumb_closed if self._thumb_change_at is None else not self._thumb_closed
        if frame.thumb_out < self._thumb_close_threshold:
            closed = True
        elif frame.thumb_out > self._thumb_open_threshold:
            closed = False
        if closed == self._thumb_closed:
            self._thumb_change_at = None
            return
        if self._thumb_change_at is None:
            self._thumb_change_at = frame.t + self._thumb_debounce_s
        if frame.t < self._thumb_change_at:
            return
        self._thumb_closed = closed
        self._thumb_change_at = None

        if not closed:
            if self.left_down:
                self._release_left()
            elif self._drag_origin is not None:
                self._drag_origin = None
                self._pointer_needs_rebase = True
        elif self.mode is Mode.POINTER:
            self._stop_scroll()
            self._drag_origin = frame.fingers["index"].img
            if self._click_method != "tap":
                self._press_left()
        elif (
            self.mode is Mode.SCROLL
            and self._click_method != "tap"
            and self._hand_is_still(frame)
        ):
            self._stop_scroll()
            self._backend.button("right", True)
            self._backend.button("right", False)

    def _press_left(self) -> None:
        self._backend.button("left", True)
        self.left_down = True

    def _release_left(self) -> None:
        self._backend.button("left", False)
        self.left_down = False
        if self._drag_origin is not None:
            self._drag_origin = None
            self._pointer_needs_rebase = True

    def _image_distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return hypot(a[0] - b[0], (a[1] - b[1]) * self._image_aspect)

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
            if not self.left_down and not self._tap_busy:
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
        if self._drag_origin is not None:
            drag_distance = self._image_distance(index.img, self._drag_origin)
            if drag_distance < self._drag_threshold * hand_size:
                return
            if not self.left_down:
                self._press_left()
            self._drag_origin = None
            self._pointer_needs_rebase = True
        if self._pointer_needs_rebase:
            self._pointer.rebase()
            self._pointer_needs_rebase = False
        self._send_move(self._pointer.update(index.img, hand_size, t))

    def _send_move(self, delta: tuple[int, int]) -> None:
        if delta != (0, 0):
            self._backend.move_rel(*delta)

    def _stop_scroll(self) -> None:
        self._scroll.stop()
        self._scroll_coasting = False

    def _send_scroll(self, delta: tuple[float, float]) -> None:
        if delta != (0, 0):
            self._backend.scroll(*delta)
