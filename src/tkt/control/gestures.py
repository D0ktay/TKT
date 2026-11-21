"""El karelerinden girdi olaylarına geçişi yöneten jest motoru."""

from collections import deque
from collections.abc import Mapping
from enum import Enum
from math import hypot, isfinite
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
    DICTATE = 5


class _IndexPinch(Enum):
    DISARMED = 0
    READY = 1
    CLOSING = 2
    HELD = 3
    CONSUMED = 4


_MODES_BY_FINGERS = {
    ("index",): Mode.POINTER,
    ("index", "middle"): Mode.SCROLL,
    ("index", "middle", "ring"): Mode.SWIPE,
    ("index", "middle", "ring", "pinky"): Mode.SWIPE,
    # Yalnız serçe açık (başparmak serbest, "telefon" işareti): sistem diktesi.
    ("pinky",): Mode.DICTATE,
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
        swipe = cfg["swipe"]
        self._click_method = general.get("click_method", "thumb")
        self._pinch = (
            cfg[self._click_method] if self._click_method in ("pinch", "index_pinch") else None
        )
        self._confidence_min = general["confidence_min"]
        self._lost_release_s = general["lost_release_ms"] / 1000
        self._lock_s = mode["lock_ms"] / 1000
        self._curl_open = mode["curl_open"]
        self._curl_closed = mode["curl_closed"]
        self._freeze_curl_rate = mode["freeze_curl_rate"]
        self._thumb_close_threshold = thumb["closed"]
        self._thumb_open_threshold = thumb["open"]
        self._thumb_debounce_s = thumb["debounce_ms"] / 1000
        self._thumb_freeze_rate = thumb["freeze_rate"]
        self._drag_threshold = thumb["drag_threshold"]
        self._still_s = thumb["still_ms"] / 1000
        self._swipe_distance = swipe["distance"]
        self._swipe_window_s = swipe["window_ms"] / 1000
        self._image_aspect = info.height / info.width
        self._rollback_ms = cfg["pointer"]["rollback_ms"]
        self._backend = backend
        self._pointer = PointerMotion(cfg["pointer"], info)
        self._scroll = ScrollMotion(cfg["scroll"], info)
        self._tap = TapDetector(cfg) if self._click_method in ("tap", "both") else None
        if self._tap is not None:
            self._tap.set_info(info)
        self._lock = RLock()
        self._reset_state()

    def process(self, frame: HandFrame) -> None:
        """Bir el karesini işle."""
        with self._lock:
            reliable = frame.present and frame.confidence >= self._confidence_min
            if self._pinch is not None:
                if reliable and frame.pinch is not None and self._inside(frame):
                    if self._click_method == "index_pinch":
                        self._process_index_pinch(frame, frame.pinch["index"])
                    else:
                        self._process_pinch(frame, frame.pinch)
                elif self.mode is not Mode.NONE:
                    self.release()
                return
            if not reliable and not self._dictation_frame(frame):
                self._process_missing_hand(frame.t)
                return

            index = frame.fingers["index"]
            if self._hand_missing:
                if self._loss_expired(frame.t):
                    self.release()
                else:
                    if self.mode is Mode.SCROLL:
                        self._rebase_scroll()
                    if self._drag_origin is not None:
                        self._drag_origin = index.img
            self._hand_missing = False
            self._last_reliable_t = frame.t
            # Hızlı kıvrılma aynı karede modu kapatsa bile geri almayı gönder.
            curling_fast = self._check_curl_rate(index.curl, frame.t)
            settling = self._check_settle_rate(frame)
            previous_mode = self.mode
            self._update_mode(frame)
            if self.mode is not previous_mode:
                self._change_mode(previous_mode, frame.t)
            scroll_pos = self._scroll_position(frame)
            self._record_stillness(frame, scroll_pos)
            was_dragging = self.left_down
            self._update_thumb(frame)
            self._update_tap(frame, suppress=was_dragging or self.left_down)
            if self.mode is Mode.SCROLL and scroll_pos is None:
                self._scroll_needs_rebase = True
            if self._tap_busy:
                return
            if self.mode is Mode.POINTER and settling:
                self._pointer_needs_rebase = True
            elif self.mode is Mode.POINTER:
                self._move_pointer(index, frame.hand_size, frame.t, curling_fast)
            elif self.mode is Mode.SCROLL and scroll_pos is not None:
                if self._scroll_needs_rebase:
                    self._rebase_scroll()
                self._send_scroll(self._scroll.update(scroll_pos, frame.hand_size, frame.t))
            elif self.mode is Mode.SWIPE:
                self._update_swipe(frame)
            elif self._scroll_coasting and previous_mode is Mode.NONE:
                self._send_scroll(self._scroll.tick(frame.t))

    def release(self) -> None:
        """Tuşları bırak, dikteyi durdur ve jest durumunu sıfırla."""
        with self._lock:
            if self.mode is Mode.DICTATE:
                self._backend.system_action("dictation_stop")
            self._backend.release_all()
            self._pointer.reset()
            self._stop_scroll()
            self._reset_state()

    def _reset_state(self) -> None:
        self.mode = Mode.NONE
        self.left_down = False
        self._index_state = _IndexPinch.DISARMED
        self._index_closed_at = 0.0
        self._pinch_armed = False
        self._pinch_owner: str | None = None
        self._pinch_candidate: str | None = None
        self._pinch_since: float | None = None
        self._pinch_moving = False
        self._open_fingers: tuple[str, ...] = ()
        self._wait_until: float | None = None
        self._last_reliable_t: float | None = None
        self._hand_missing = False
        self._previous_index: tuple[float, float] | None = None
        self._previous_settle: tuple[float, float, float] | None = None
        self._settling = False
        self._pointer_frozen = False
        self._pointer_needs_rebase = True
        self._tap_busy = False
        if self._tap is not None:
            self._tap.reset()
        self._thumb_closed = False
        self._thumb_change_at: float | None = None
        self._drag_origin: tuple[float, float] | None = None
        self._still_history: deque[tuple[float, tuple[float, float], tuple[float, float]]] = deque()
        self._swipe_history: deque[tuple[float, tuple[float, float]]] = deque()
        self._swipe_triggered = False
        self._scroll_coasting = False
        self._scroll_needs_rebase = False

    @staticmethod
    def _inside(frame: HandFrame) -> bool:
        points = [frame.palm, *(f.surface for f in frame.fingers.values() if f.surface is not None)]
        return all(0 <= coordinate <= 1 for point in points for coordinate in point)

    def _process_index_pinch(self, frame: HandFrame, distance: float) -> None:
        cfg = self._pinch
        if self._loss_expired(frame.t):
            self.release()
        self._last_reliable_t = frame.t
        if sum(frame.fingers[name].curl for name in FINGERS) / len(FINGERS) >= cfg["fist_curl"]:
            if self.left_down:
                self._backend.button("left", False)
                self.left_down = False
            self._index_state = _IndexPinch.DISARMED
            self._pinch_since = None
            self._pointer.rebase()
            self.mode = Mode.WAIT
            return

        self.mode = Mode.POINTER
        if self._index_state is _IndexPinch.DISARMED:
            if distance < cfg["open"]:
                return
            self._index_state = _IndexPinch.READY
        if self._index_state is _IndexPinch.READY:
            if distance > cfg["closed"]:
                self._send_move(self._pointer.update(frame.palm, frame.hand_size, frame.t))
                return
            self._index_state = _IndexPinch.CLOSING
            self._index_closed_at = frame.t
            self._drag_origin = frame.palm
            self._pointer.rebase()

        debounce = cfg["debounce_ms"] / 1000
        if self._index_state is _IndexPinch.CLOSING:
            if distance >= cfg["open"]:
                self._index_state = _IndexPinch.READY
                self._pointer.rebase()
                return
            if frame.t < self._index_closed_at + debounce:
                self._pointer.update(self._drag_origin, frame.hand_size, frame.t)
                return
            self._index_state = _IndexPinch.HELD

        self._held_index_pinch(frame, distance)

    def _held_index_pinch(self, frame: HandFrame, distance: float) -> None:
        cfg = self._pinch
        if distance >= cfg["open"] and self._pinch_since is None:
            self._pinch_since = frame.t
        elif distance <= cfg["closed"]:
            self._pinch_since = None
        if self._pinch_since is not None:
            if frame.t >= self._pinch_since + cfg["debounce_ms"] / 1000:
                if self.left_down:
                    self._backend.button("left", False)
                    self.left_down = False
                elif self._index_state is _IndexPinch.HELD:
                    self._backend.button("left", True)
                    self._backend.button("left", False)
                self._index_state = _IndexPinch.READY
                self._pinch_since = None
                self._pointer.rebase()
            return

        if self._index_state is _IndexPinch.CONSUMED:
            return
        if not self.left_down:
            moved = self._image_distance(frame.palm, self._drag_origin)
            if moved > cfg["drag_threshold"] * frame.hand_size:
                self._backend.button("left", True)
                self.left_down = True
            elif frame.t >= self._index_closed_at + cfg["hold_ms"] / 1000:
                self._backend.button("right", True)
                self._backend.button("right", False)
                self._index_state = _IndexPinch.CONSUMED
                return
        pos = frame.palm if self.left_down else self._drag_origin
        self._send_move(self._pointer.update(pos, frame.hand_size, frame.t))

    def _process_pinch(self, frame: HandFrame, distances: dict[str, float]) -> None:
        cfg = self._pinch
        if self._loss_expired(frame.t):
            self.release()
        self._last_reliable_t = frame.t
        all_open = all(distances[name] >= cfg["open"] for name in FINGERS)
        if self._pinch_owner is not None:
            self._held_pinch(frame, distances, all_open)
            return

        mean_curl = sum(frame.fingers[name].curl for name in FINGERS) / len(FINGERS)
        if mean_curl >= cfg["fist_curl"]:
            self._pinch_armed = False
            self._pinch_candidate = None
            self._pinch_since = None
            if self.mode is not Mode.WAIT:
                self._pointer.rebase()
            self.mode = Mode.WAIT
            return

        self.mode = Mode.POINTER
        if all_open:
            self._pinch_armed = True
        winner, runner_up, *_ = sorted(FINGERS, key=distances.__getitem__)
        if (
            self._pinch_armed
            and (distances[winner] <= cfg["closed"]
                 or (winner == self._pinch_candidate and distances[winner] < cfg["open"]))
            and distances[runner_up] - distances[winner] >= cfg["margin"]
        ):
            if winner != self._pinch_candidate:
                self._pinch_candidate = winner
                self._pinch_since = frame.t
                self._drag_origin = frame.palm
                self._pointer.rebase()
                self._scroll.rebase()
            if frame.t >= self._pinch_since + cfg["debounce_ms"] / 1000:
                self._pinch_owner = winner
                self._pinch_candidate = None
                self._pinch_since = None
                self._pinch_armed = False
                if winner == "index":
                    self._backend.button("left", True)
                    self.left_down = True
                self._held_pinch(frame, distances, all_open)
            return
        self._pinch_candidate = None
        self._pinch_since = None
        self._send_move(self._pointer.update(frame.palm, frame.hand_size, frame.t))

    def _held_pinch(self, frame: HandFrame, distances: dict[str, float], all_open: bool) -> None:
        cfg = self._pinch
        owner = self._pinch_owner
        threshold = cfg["drag_threshold"] * frame.hand_size
        if self._image_distance(frame.palm, self._drag_origin) > threshold:
            self._pinch_moving = True

        if distances[owner] >= cfg["open"]:
            if self._pinch_since is None:
                self._pinch_since = frame.t
            if frame.t >= self._pinch_since + cfg["debounce_ms"] / 1000:
                if self.left_down:
                    self._backend.button("left", False)
                    self.left_down = False
                elif owner == "middle" and not self._pinch_moving:
                    self._backend.button("right", True)
                    self._backend.button("right", False)
                self._pinch_owner = None
                self._pinch_armed = all_open
                self._pinch_since = None
                self._pinch_moving = False
                self._drag_origin = None
                self._pointer.rebase()
                self._stop_scroll()
                self.mode = Mode.POINTER
            return

        self._pinch_since = None
        # Keep the filter anchored while waiting so the threshold-crossing frame
        # can emit motion even after a long stationary hold.
        pos = frame.palm if self._pinch_moving else self._drag_origin
        if owner == "index":
            self._send_move(self._pointer.update(pos, frame.hand_size, frame.t))
        elif owner == "middle":
            delta = self._scroll.update(pos, frame.hand_size, frame.t)
            if self._pinch_moving:
                self.mode = Mode.SCROLL
                self._send_scroll(delta)

    def _process_missing_hand(self, t: float) -> None:
        if self._last_reliable_t is None:
            return
        if self._loss_expired(t):
            self.release()
            return
        if not self._hand_missing:
            self._hand_missing = True
            # El kalkınca/çıkınca son kareler kaymış olabilir (tespit gecikmesi): taşımadaki
            # gibi son `rollback_ms` hareketi geri al.
            if self.mode is Mode.POINTER and not self._pointer_frozen:
                self._send_move(self._pointer.rollback_at(self._rollback_ms, t))
            self._wait_until = None
            self._previous_index = None
            self._previous_settle = None
            self._settling = False
            self._pointer_needs_rebase = True
            self._thumb_change_at = None
            self._still_history.clear()
            self._swipe_history.clear()
            self._cancel_active_tap()
            self._stop_scroll()

    def _dictation_frame(self, frame: HandFrame) -> bool:
        """Güveni düşük (ör. el masadan kalkık) ama görünen kare dikte için işlensin mi.

        El havadayken de dikte açılıp kapanabilsin: dikte modundayken (yumruğu görmek için)
        veya mod boştayken yalnız serçe açıksa. Diğer modlarda düşük güvenli kare imleci
        oynatmasın diye "el yok" sayılmaya devam eder.
        """
        if not frame.present or not frame.fingers:
            return False
        if self.mode is Mode.DICTATE:
            return True
        if self.mode not in (Mode.NONE, Mode.WAIT):
            return False
        opened = tuple(n for n in FINGERS if frame.fingers[n].curl < self._curl_open)
        return _MODES_BY_FINGERS.get(opened) is Mode.DICTATE

    def _loss_expired(self, t: float) -> bool:
        last = self._last_reliable_t
        return last is not None and t >= last + self._lost_release_s

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
        if previous_mode is Mode.DICTATE:
            # Dikte modundan yalnız yumrukla çıkılır: söyleneni gönder (dikte dur + Enter).
            self._backend.system_action("dictation_submit")
        if self.mode is Mode.DICTATE:
            self._backend.system_action("dictation_start")
        self._reset_swipe()
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
            self._rebase_scroll()

    def _reset_swipe(self) -> None:
        self._swipe_history.clear()
        self._swipe_triggered = False

    def _update_swipe(self, frame: HandFrame) -> None:
        if self._swipe_triggered or not isfinite(frame.hand_size) or frame.hand_size <= 0:
            return

        history = self._swipe_history
        while history and history[0][0] + self._swipe_window_s < frame.t:
            history.popleft()

        # Her iki farkı da görüntü genişliği biriminde tut; hand_size da aynı ölçekte.
        threshold = self._swipe_distance * frame.hand_size
        x, y = frame.palm
        for _, (old_x, old_y) in history:
            dx = x - old_x
            dy = (y - old_y) * self._image_aspect
            if hypot(dx, dy) < threshold:
                continue
            if abs(dx) >= abs(dy):
                action = "desktop_left" if dx < 0 else "desktop_right"
            elif dy < 0:
                action = "mission_control"
            else:
                continue
            self._swipe_triggered = True
            history.clear()
            self._backend.system_action(action)
            return

        history.append((frame.t, frame.palm))

    def _cancel_active_tap(self) -> None:
        if self._tap is not None and self._tap_busy:
            self._tap.reset()
            self._tap_busy = False

    def _update_tap(self, frame: HandFrame, *, suppress: bool) -> None:
        watched = _TAP_FINGERS.get(self.mode)
        if self._tap is None or watched is None:
            return
        if suppress:
            self._tap.cancel_until_landing(watched)
        update = self._tap.update(frame, watched)
        if update.started:
            self._send_move(self._pointer.rollback_at(self._rollback_ms, frame.t))
            self._stop_scroll()
        if update.kind is not None:
            button = "left" if update.kind == "tap1" else "right"
            self._backend.button(button, True)
            self._backend.button(button, False)
        if self._tap_busy and not update.busy:
            self._pointer.rebase()
            self._pointer_needs_rebase = False
            if self.mode is Mode.SCROLL:
                self._rebase_scroll()
        self._tap_busy = update.busy

    def _scroll_position(self, frame: HandFrame) -> tuple[float, float] | None:
        if "index" not in self._open_fingers or "middle" not in self._open_fingers:
            return None
        ix, iy = frame.fingers["index"].img
        mx, my = frame.fingers["middle"].img
        return (ix + mx) / 2, (iy + my) / 2

    def _record_stillness(self, frame: HandFrame, scroll_pos: tuple[float, float] | None) -> None:
        history = self._still_history
        if scroll_pos is None:
            history.clear()
            return
        history.append((frame.t, frame.palm, scroll_pos))
        # Pencerenin başlangıcını kapsayan son örneği de tut.
        window_start = frame.t - self._still_s
        while len(history) > 1 and history[1][0] <= window_start:
            history.popleft()

    def _hand_is_still(self, frame: HandFrame) -> bool:
        history = self._still_history
        if not history or history[0][0] > frame.t - self._still_s:
            return False
        _, palm, scroll_pos = history[-1]
        threshold = self._drag_threshold * frame.hand_size
        return all(
            self._image_distance(old_palm, palm) <= threshold
            and self._image_distance(old_pos, scroll_pos) <= threshold
            for _, old_palm, old_pos in history
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
            self.mode is Mode.SCROLL and self._click_method != "tap" and self._hand_is_still(frame)
        ):
            self._stop_scroll()
            self._backend.button("right", True)
            self._backend.button("right", False)

    def _press_left(self) -> None:
        if self.mode is Mode.POINTER:
            # Başparmak kapanırken el kaymış olabilir: tık, kaymadan önceki yerde olsun.
            self._send_move(self._pointer.rollback(self._rollback_ms))
        self._backend.button("left", True)
        self.left_down = True

    def _release_left(self) -> None:
        if self.mode is Mode.POINTER:
            # Başparmak açılırken el kaymış olabilir: bırakma, kaymadan önceki yerde olsun.
            self._send_move(self._pointer.rollback(self._rollback_ms))
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

    def _check_settle_rate(self, frame: HandFrame) -> bool:
        """İmleç modunda başparmak hızlı oynuyor veya orta parmak hızla açılıyorsa True.

        Bu sürede imleç donar (el tıklarken/mod değiştirirken kayıyor). Orta parmak
        açılmaya başlayınca son `rollback_ms` hareket de geri alınır; başparmakta geri alma
        tuş basılıp bırakılırken yapılır (normal harekette yanlış geri alma olmasın).
        """
        previous = self._previous_settle
        middle = frame.fingers["middle"].curl
        self._previous_settle = (frame.thumb_out, middle, frame.t)
        if self.mode is not Mode.POINTER or previous is None:
            self._settling = False
            return False
        thumb_prev, middle_prev, t_prev = previous
        dt = frame.t - t_prev
        if dt <= 0:
            return self._settling
        thumb_fast = abs(frame.thumb_out - thumb_prev) / dt > self._thumb_freeze_rate
        middle_opening = (middle_prev - middle) / dt > self._freeze_curl_rate
        # Hızlı kıvırma donması aynı karede zaten geri aldıysa ikinci kez alınmaz.
        if middle_opening and not self._settling and not self._tap_busy \
                and not self._pointer_frozen:
            self._send_move(self._pointer.rollback(self._rollback_ms))
        self._settling = thumb_fast or middle_opening
        return self._settling

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

    def _rebase_scroll(self) -> None:
        self._scroll.rebase()
        self._scroll_needs_rebase = False

    def _send_scroll(self, delta: tuple[float, float]) -> None:
        if delta != (0, 0):
            self._backend.scroll(*delta)
