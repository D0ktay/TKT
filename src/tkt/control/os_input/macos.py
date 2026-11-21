"""macOS'ta Quartz ile işletim sistemine girdi gönderen backend."""

import time
from collections.abc import Callable
from math import hypot, isfinite, trunc
from typing import Any

# NSEvent.doubleClickInterval okunamazsa kullanılan çift tık aralığı (sn).
DOUBLE_CLICK_INTERVAL_S = 0.5
# Çift tık sayılması için iki basış arasındaki en büyük uzaklık (piksel).
DOUBLE_CLICK_RADIUS_PX = 4.0
# Sistem eylemleri: ad -> (sanal tuş kodu). Hepsi Ctrl ile birlikte gider.
_KEY_UP, _KEY_LEFT, _KEY_RIGHT = 126, 123, 124
_SYSTEM_KEYS = {
    "mission_control": _KEY_UP,
    "desktop_left": _KEY_LEFT,
    "desktop_right": _KEY_RIGHT,
}
_MAX_DISPLAYS = 16
# Sistem diktesi: "Control tuşuna iki kez bas" kısayolu (sol Control sanal tuş kodu).
_KEY_CONTROL = 59
_DICTATION_ACTIONS = ("dictation_start", "dictation_stop")
_DICTATION_TAP_GAP_S = 0.05
# Cihaza bağlı sol Control bayrağı (NX_DEVICELCTLKEYMASK); gerçek klavye bunu da gönderir.
_NX_DEVICE_LEFT_CONTROL = 0x1


def post_control_double_tap(q: Any, post: Callable[[Any], None],
                            sleep: Callable[[float], None] | None = None) -> None:
    """Sol Control'e iki kez basıp bırakır (macOS "Control tuşuna iki kez bas" diktesi).

    macOS değiştirici tuşların çift basışını `kCGEventFlagsChanged` olaylarından tanır;
    `CGEventCreateKeyboardEvent` sıradan tuş olayı ürettiği için algılanmaz. Olaylar HID
    sistem durumu kaynağıyla, bayrak değişimi türünde oluşturulur.
    """
    sleep = sleep or time.sleep
    source = q.CGEventSourceCreate(q.kCGEventSourceStateHIDSystemState)
    pressed = q.kCGEventFlagMaskControl | _NX_DEVICE_LEFT_CONTROL
    for i in range(2):
        if i:
            sleep(_DICTATION_TAP_GAP_S)
        for down in (True, False):
            event = q.CGEventCreate(source)
            q.CGEventSetType(event, q.kCGEventFlagsChanged)
            q.CGEventSetIntegerValueField(event, q.kCGKeyboardEventKeycode, _KEY_CONTROL)
            q.CGEventSetFlags(event, pressed if down else 0)
            post(event)


def _system_double_click_interval() -> float:
    try:
        from AppKit import NSEvent  # type: ignore[import-not-found]

        interval = float(NSEvent.doubleClickInterval())
    except Exception:
        return DOUBLE_CLICK_INTERVAL_S
    return interval if isfinite(interval) and interval > 0 else DOUBLE_CLICK_INTERVAL_S


class MacBackend:
    """Quartz'ı yalnızca örnek oluşturulurken yükler; macOS dışında import edilebilir."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        double_click_interval: float | None = None,
    ) -> None:
        """Quartz'ı yükle; yoksa açık hatayla dur."""
        try:
            import Quartz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MacBackend yalnızca macOS'ta ve pyobjc-framework-Quartz kuruluyken çalışır"
            ) from exc
        self._q: Any = Quartz
        self._clock = clock
        self._interval = (
            double_click_interval
            if double_click_interval is not None
            else _system_double_click_interval()
        )
        q = Quartz
        self._buttons = {
            "left": (q.kCGMouseButtonLeft, q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp,
                     q.kCGEventLeftMouseDragged),
            "right": (q.kCGMouseButtonRight, q.kCGEventRightMouseDown, q.kCGEventRightMouseUp,
                      q.kCGEventRightMouseDragged),
        }  # fmt: skip
        self._pressed: dict[str, None] = {}
        # Her tuş için son basışın (zaman, x, y, tık sayısı).
        self._last_down: dict[str, tuple[float, float, float, int]] = {}
        self._click_state: dict[str, int] = {}
        self._scroll_x = self._scroll_y = 0.0
        self.events_sent = 0

    # --- yardımcılar ---

    def _post(self, event: Any) -> None:
        self._q.CGEventPost(self._q.kCGHIDEventTap, event)
        self.events_sent += 1

    def _location(self) -> tuple[float, float]:
        point = self._q.CGEventGetLocation(self._q.CGEventCreate(None))
        return float(point.x), float(point.y)

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        """Noktayı ekranların içine, en yakın ekran dikdörtgenine kırp."""
        q = self._q
        result = q.CGGetActiveDisplayList(_MAX_DISPLAYS, None, None)
        best: tuple[float, float] | None = None
        best_dist = float("inf")
        for display in result[1] or ():
            rect = q.CGDisplayBounds(display)
            left, top = float(rect.origin.x), float(rect.origin.y)
            right, bottom = left + float(rect.size.width), top + float(rect.size.height)
            # Sağ/alt kenar ekranın dışında kalır; bir piksel içeride dur.
            cx = min(max(x, left), max(left, right - 1))
            cy = min(max(y, top), max(top, bottom - 1))
            dist = hypot(cx - x, cy - y)
            if dist < best_dist:
                best, best_dist = (cx, cy), dist
        return best if best is not None else (x, y)

    def _mouse_event(self, kind: int, button: int, x: float, y: float) -> Any:
        return self._q.CGEventCreateMouseEvent(None, kind, (x, y), button)

    # --- InputBackend ---

    def move_rel(self, dx: int, dy: int) -> None:
        q = self._q
        x, y = self._location()
        nx, ny = self._clamp(x + dx, y + dy)
        held = next((n for n in ("left", "right") if n in self._pressed), None)
        if held is not None:
            button, _, _, kind = self._buttons[held]
        else:
            button, kind = q.kCGMouseButtonLeft, q.kCGEventMouseMoved
        event = self._mouse_event(kind, button, nx, ny)
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaX, round(nx - x))
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaY, round(ny - y))
        self._post(event)

    def button(self, name: str, down: bool) -> None:
        q = self._q
        button, down_type, up_type, _ = self._buttons[name]
        x, y = self._location()
        if down:
            if name in self._pressed:
                return
            now = self._clock()
            last = self._last_down.get(name)
            count = 1
            if (
                last is not None
                and now - last[0] <= self._interval
                and hypot(x - last[1], y - last[2]) <= DOUBLE_CLICK_RADIUS_PX
                and last[3] < 3
            ):
                count = last[3] + 1
            self._last_down[name] = (now, x, y, count)
            self._click_state[name] = count
            self._pressed[name] = None
            kind = down_type
        else:
            if name not in self._pressed:
                return
            kind = up_type
        event = self._mouse_event(kind, button, x, y)
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventClickState, self._click_state[name])
        self._post(event)
        if not down:
            self._pressed.pop(name)

    def scroll(self, dx: float, dy: float) -> None:
        if not (isfinite(dx) and isfinite(dy)):
            raise ValueError("scroll değerleri sonlu olmalı")
        x, y = self._scroll_x + dx, self._scroll_y + dy
        sx, sy = trunc(x), trunc(y)
        if sx or sy:
            q = self._q
            # Quartz tekerlek eksenlerini dikey, yatay sırasıyla alır.
            self._post(q.CGEventCreateScrollWheelEvent(None, q.kCGScrollEventUnitPixel, 2, sy, sx))
        self._scroll_x, self._scroll_y = x - sx, y - sy

    def system_action(self, name: str) -> None:
        if name in _DICTATION_ACTIONS:
            self._dictation_toggle()
            return
        try:
            keycode = _SYSTEM_KEYS[name]
        except KeyError:
            raise ValueError(f"Bilinmeyen sistem eylemi: {name}") from None
        q = self._q
        for down in (True, False):
            event = q.CGEventCreateKeyboardEvent(None, keycode, down)
            q.CGEventSetFlags(event, q.kCGEventFlagMaskControl)
            self._post(event)

    def _dictation_toggle(self) -> None:
        """Control'e iki kez basar (macOS Dikte kısayolu bu olmalı)."""
        post_control_double_tap(self._q, self._post)

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
