"""Canlı durum penceresi (--hud): kamera görüntüsü, iskelet ve şu anki hareket.

Vision (görüntü, iskelet) ile control'ü (mod, tuşlar) yalnızca app birleştirir. Pencere ana
thread'de çizilir (macOS'ta `cv2.imshow` ana thread ister). Yazılar ASCII'dir; OpenCV'nin
yazı tipi Türkçe harfleri çizemez.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from tkt.control.gestures import GestureEngine, Mode
from tkt.core.types import HandFrame

WINDOW_NAME = "TKT"
# Tık gibi anlık olayların ekranda kalma süresi (sn).
ACTION_SHOW_S = 0.8
# Basılıp bu süreden önce bırakılan tuş "tık" sayılır; daha uzunsa "sürükleme".
CLICK_MAX_S = 0.35
# Taşıma etiketi: imleç modunda işaret parmağı bu kadar kıvrıksa.
CARRY_CURL = 0.5
_QUIT_KEYS = (ord("q"), 27)
_FINGER_NAMES = (("index", "isaret"), ("middle", "orta"), ("ring", "yuzuk"), ("pinky", "serce"))

# BGR
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (80, 220, 80)
_BLUE = (240, 160, 60)
_YELLOW = (60, 220, 240)
_ORANGE = (40, 150, 250)
_RED = (60, 60, 240)
_GRAY = (150, 150, 150)

_MODE_LABELS = {
    Mode.NONE: ("DINLENME", _GRAY),
    Mode.WAIT: ("BEKLIYOR", _GRAY),
    Mode.POINTER: ("IMLEC", _GREEN),
    Mode.SCROLL: ("KAYDIRMA", _BLUE),
    Mode.SWIPE: ("KAYDIRMA (4 parmak)", _BLUE),
    Mode.DICTATE: ("DIKTE - konus (yumruk: dur)", _RED),
}


_ACTION_LABELS = {
    "dictation_start": "DIKTE ACILDI",
    "dictation_stop": "DIKTE KAPANDI",
    "dictation_submit": "GONDER (Enter)",
}


class HudClosed(Exception):
    """Kullanıcı HUD penceresinde q/ESC'ye bastı."""


class ActionTap:
    """Backend vekili: çağrıları iletir, son anlık hareketi (tık, sürükleme) hatırlar.

    `clock` olay zamanını verir (testte sahte saat). `events_sent` iç backend'den geçer.
    """

    def __init__(self, backend: Any, clock: Callable[[], float]) -> None:
        self._backend = backend
        self._clock = clock
        self._down_at: dict[str, float] = {}
        self.action: str = ""
        self.action_t: float = -1e9

    @property
    def events_sent(self) -> Any:
        return getattr(self._backend, "events_sent", None)

    def _set(self, label: str) -> None:
        self.action, self.action_t = label, self._clock()

    def move_rel(self, dx: int, dy: int) -> None:
        self._backend.move_rel(dx, dy)

    def button(self, name: str, down: bool) -> None:
        self._backend.button(name, down)
        now = self._clock()
        side = "SOL" if name == "left" else "SAG" if name == "right" else name.upper()
        if down:
            self._down_at[name] = now
            return
        start = self._down_at.pop(name, None)
        if start is not None and now - start <= CLICK_MAX_S:
            self._set(f"{side} TIK")
        else:
            self._set(f"{side} TUS BIRAKILDI")

    def scroll(self, dx: float, dy: float) -> None:
        self._backend.scroll(dx, dy)

    def system_action(self, name: str) -> None:
        self._backend.system_action(name)
        self._set(_ACTION_LABELS.get(name, f"SISTEM: {name}"))

    def release_all(self) -> None:
        self._backend.release_all()
        self._down_at.clear()

    def held(self, now: float) -> str:
        """Şu an CLICK_MAX_S'den uzun basılı tutulan tuş varsa etiketi."""
        for name, start in self._down_at.items():
            if now - start > CLICK_MAX_S:
                return "SURUKLEME" if name == "left" else f"{name.upper()} BASILI"
        return ""


@dataclass(frozen=True)
class Status:
    """Ekranda gösterilen durum."""
    label: str
    color: tuple[int, int, int]
    action: str            # son anlık olay (boş olabilir)
    fingers: str           # parmak özeti


def status(frame: HandFrame, engine: GestureEngine, tap: ActionTap, now: float,
           confidence_min: float, paused: bool) -> Status:
    """Kare, jest motoru ve son olaylardan ekrandaki etiketi üretir."""
    action = tap.action if now - tap.action_t <= ACTION_SHOW_S else ""
    fingers = ""
    if frame.present:
        parts = [f"{short}:{'acik' if frame.fingers[name].curl < 0.5 else 'kivrik'}"
                 for name, short in _FINGER_NAMES if name in frame.fingers]
        parts.append(f"basparmak:{'acik' if frame.thumb_out >= 0.5 else 'kapali'}")
        fingers = "  ".join(parts)
    if paused:
        return Status("DURAKLATILDI (Ctrl+Alt+P)", _ORANGE, action, fingers)
    if not frame.present:
        return Status("EL YOK", _RED, action, fingers)
    if engine.mode is Mode.DICTATE:            # el havadayken de dikte sürer
        label, color = _MODE_LABELS[Mode.DICTATE]
        return Status(label, color, action, fingers)
    if frame.confidence <= 0.0:
        return Status("EL KALKIK", _ORANGE, action, fingers)
    if frame.confidence < confidence_min:
        return Status("GUVEN DUSUK", _ORANGE, action, fingers)
    held = tap.held(now)
    if held:
        return Status(held, _YELLOW, action, fingers)
    label, color = _MODE_LABELS.get(engine.mode, (engine.mode.name, _WHITE))
    if engine.mode is Mode.POINTER and frame.fingers["index"].curl >= CARRY_CURL:
        label, color = "TASIMA (imlec durdu)", _YELLOW
    return Status(label, color, action, fingers)


def draw(image: np.ndarray, hand: Any, st: Status) -> np.ndarray:
    """Görüntünün üstüne iskelet, büyük durum etiketi, son olay ve parmak özetini çizer."""
    import cv2

    from tkt.vision.debug_view import draw_skeleton, to_bgr

    out = to_bgr(image)
    h, w = out.shape[:2]
    if hand is not None:
        draw_skeleton(out, hand)
    band = max(40, h // 7)
    cv2.rectangle(out, (0, 0), (w, band), _BLACK, -1)
    scale = band / 45.0
    cv2.putText(out, st.label, (12, int(band * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                st.color, max(2, int(scale * 2)), cv2.LINE_AA)
    if st.action:
        size, _ = cv2.getTextSize(st.action, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = max(12, w - size[0] - 12)
        cv2.putText(out, st.action, (x, int(band * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    _YELLOW, max(2, int(scale * 2)), cv2.LINE_AA)
    if st.fingers:
        fs = max(0.4, h / 1000)
        cv2.rectangle(out, (0, h - int(30 * fs * 2)), (w, h), _BLACK, -1)
        cv2.putText(out, st.fingers, (10, h - int(18 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    _WHITE, 1, cv2.LINE_AA)
    return out


class HudSource:
    """Kaynağı sarar: `frames()` kareleri verir, son görüntü ve ham eli saklar.

    Kaynakta `frames_raw` yoksa (kayıt, sentetik) görüntü siyah zemin olur.
    """

    def __init__(self, source: Any) -> None:
        self._source = source
        self.info = source.info
        self.image: np.ndarray | None = None
        self.hand: Any = None

    def frames(self) -> Iterator[HandFrame]:
        raw = getattr(self._source, "frames_raw", None)
        if raw is None:
            blank = np.zeros((max(1, self.info.height), max(1, self.info.width), 3), np.uint8)
            for frame in self._source.frames():
                self.image, self.hand = blank, None
                yield frame
            return
        for frame, hand, image in raw():
            self.image, self.hand = image, hand
            yield frame

    def close(self) -> None:
        self._source.close()


class Hud:
    """Her kare `show` ile pencereyi günceller; q/ESC `HudClosed` yükseltir."""

    def __init__(self, source: HudSource, tap: ActionTap, confidence_min: float,
                 show: Callable[[str, np.ndarray], None] | None = None,
                 poll_key: Callable[[], int] | None = None) -> None:
        import cv2

        self._source = source
        self._tap = tap
        self._confidence_min = confidence_min
        self._own_window = show is None
        self._show = show or cv2.imshow
        self._poll_key = poll_key or (lambda: cv2.waitKey(1) & 0xFF)

    def show(self, frame: HandFrame, engine: GestureEngine, now: float, paused: bool) -> None:
        if self._source.image is None:
            return
        st = status(frame, engine, self._tap, now, self._confidence_min, paused)
        self._show(WINDOW_NAME, draw(self._source.image, self._source.hand, st))
        if self._poll_key() in _QUIT_KEYS:
            raise HudClosed

    def close(self) -> None:
        if self._own_window:
            import cv2

            cv2.destroyAllWindows()
            cv2.waitKey(1)
