"""Debug penceresi (V3). PART2 2.2, 2.4.

macOS'ta `cv2.imshow` ana thread'de çalışmalıdır; `run` döngüyü çağıran thread'de kurar.
Çizim saf fonksiyonlardadır (`draw_overlay`, `draw_tap_prompt`, `render_lift_graph`);
pencere ve klavye yalnızca `run` içindedir. Tuşlar: `c` kalibrasyon, `t` tap kalibrasyonu,
`m` masa modeli, `s` aktif alan (yüzey) kalibrasyonu, `q`/ESC veya pencere X düğmesi çıkış.

cv2.putText yalnızca ASCII çizer; ekran metinleri bu yüzden Türkçe karaktersizdir.

CLI: python -m tkt.vision.debug_view --config config/vision.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from sys import platform
from typing import Any

import cv2
import numpy as np

from tkt.core.types import FINGERS, HandFrame
from tkt.vision import calibration, hand_calib
from tkt.vision.calibration import CORNER_NAMES, SurfaceCalibration
from tkt.vision.hand_calib import STEPS, CalibrationSession, HandCalibration
from tkt.vision.live_source import LiveSource, load_vision_config, open_live_source
from tkt.vision.tracker import RawHand

WINDOW_NAME = "TKT debug"

# MediaPipe el bağlantıları (21 nokta).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # başparmak
    (0, 5), (5, 6), (6, 7), (7, 8),            # işaret
    (5, 9), (9, 10), (10, 11), (11, 12),       # orta
    (9, 13), (13, 14), (14, 15), (15, 16),     # yüzük
    (13, 17), (17, 18), (18, 19), (19, 20),    # serçe
    (0, 17),                                   # avuç kenarı
)

# Ekranda gösterilen adım adları ve yönergeleri (ASCII).
STEP_TITLES: dict[str, str] = {
    "open": "Acik el",
    "fist": "Yumruk",
    "thumb_closed": "Basparmak kapali",
}

# Adım adım kalibrasyon: her adım boşlukla başlar, önce geri sayım (örnek yok), sonra kayıt.
GUIDED_COUNTDOWN_SECONDS = 3.0
GUIDED_RECORD_SECONDS = 3.0
STEP_HINTS: dict[str, str] = {
    "open": "Elini masaya koy, parmaklar duz ve acik",
    "fist": "Yumruk yap, parmaklar tam kivrik",
    "thumb_closed": "Isaret parmagi acik, basparmagi ele yapistir",
}

# Masa modeli kalibrasyonu: geri sayım, sonra elin mousepad'de gezdirildiği süre.
TABLE_COUNTDOWN_SECONDS = 3.0
TABLE_RECORD_SECONDS = 12.0
# Taranan alan: örnekler bu ızgarada sayılır (kapsama yüzdesi ve kareler).
TABLE_GRID = 8

# Aktif alan kalibrasyonu: her köşe boşlukla başlar, geri sayım, sonra işaret ucunun kaydı.
SURFACE_COUNTDOWN_SECONDS = 1.0
SURFACE_RECORD_SECONDS = 1.0

# Kalibrasyon sonuç mesajının ekranda kalma süresi (kare zamanı, sn).
MESSAGE_SECONDS = 4.0

# Tap kalibrasyonu: TAP_COUNT kez, her biri TAP_WINDOW_SECONDS'lik pencerede. Pencerenin
# başında TAP_FLASH_SECONDS boyunca ekran işareti ("TAP!") çıkar; asıl pencereler başlamadan
# önce TAP_LEAD_SECONDS hazırlık süresi vardır (örnek toplanmaz). Süreler kare zamanıyla
# ölçülür.
TAP_COUNT = 10
TAP_WINDOW_SECONDS = 2.0
TAP_FLASH_SECONDS = 0.25
TAP_LEAD_SECONDS = 1.5

# Lift grafiğinin gösterdiği süre (kare zamanı, sn) ve parmakların çizgi renkleri.
LIFT_GRAPH_SECONDS = 3.0
LIFT_GRAPH_FINGERS = ("index", "middle")

KEY_QUIT = (ord("q"), 27)   # q, ESC
KEY_CALIB = ord("c")
KEY_NEXT = ord(" ")          # adım adım kalibrasyonda sonraki adımı başlatır
KEY_TAP = ord("t")
KEY_TABLE = ord("m")         # masa (mousepad) modeli kalibrasyonu
KEY_SURFACE = ord("s")       # aktif alan (yüzey) kalibrasyonu
KEY_REDO = ord("g")          # yüzey kalibrasyonunda köşeyi tekrarla / bir köşe geri

# BGR renkler
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (80, 220, 80)
_RED = (60, 60, 240)
_YELLOW = (60, 220, 240)
_CYAN = (230, 200, 60)
_GRAY = (150, 150, 150)
_BAR_BG = (50, 50, 50)
_GRAPH_COLORS = {"index": _YELLOW, "middle": _CYAN}


@dataclass(frozen=True)
class Stats:
    """Ekranda gösterilen ölçümler."""
    capture_fps: float
    process_fps: float
    process_ms: float


@dataclass(frozen=True)
class CalibState:
    """Kalibrasyon oturumunun ekranda gösterilen durumu. Oturum yoksa `None` verilir."""
    step: str | None         # "open" | "fist" | "thumb_closed" | None (bitti)
    progress: float          # geçerli adımın ilerlemesi, 0–1
    total_progress: float    # tüm oturum, 0–1
    samples: int = 0         # geçerli adımda toplanan örnek sayısı
    message: str = ""        # sonuç / hata mesajı


# ---------------------------------------------------------------------- çizim


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Gri (H,W) / (H,W,1), BGR veya BGRA görüntüyü yeni bir uint8 BGR kopyaya çevirir."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    raise ValueError(f"desteklenmeyen görüntü şekli: {img.shape}")


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float,
          color: tuple[int, int, int] = _WHITE) -> None:
    """Okunabilirlik için siyah gölgeli yazı."""
    s = s.encode("ascii", "replace").decode("ascii")   # putText ASCII dışını "?" çizer
    thick = max(1, round(scale * 1.6))
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float | None,
         color: tuple[int, int, int]) -> None:
    """0–1 dolu çubuk. `value` None ise boş çubuk ve ortada "-" çizilir."""
    cv2.rectangle(img, (x, y), (x + w, y + h), _BAR_BG, -1)
    if value is not None:
        v = float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0
        if v > 0:
            cv2.rectangle(img, (x, y), (x + int(round(w * v)), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _GRAY, 1)


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:4.2f}"


def draw_skeleton(img: np.ndarray, hand: RawHand) -> None:
    """21 noktalı iskeleti `img` üstüne çizer (`image_pts` 0–1 → piksel)."""
    h, w = img.shape[:2]
    pts = np.asarray(hand.image_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 21 or not np.all(np.isfinite(pts[:21, :2])):
        return
    px = [(int(round(x * w)), int(round(y * h))) for x, y in pts[:21, :2]]
    thick = max(1, round(min(h, w) / 300))
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, px[a], px[b], _GREEN, thick, cv2.LINE_AA)
    for i, p in enumerate(px):
        tip = i in (4, 8, 12, 16, 20)
        cv2.circle(img, p, thick + (2 if tip else 1), _YELLOW if tip else _RED, -1, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, frame: HandFrame, hand: RawHand | None,
                 stats: Stats, calib_state: CalibState | None = None) -> np.ndarray:
    """Görüntünün üstüne iskelet, çubuklar, fps ve kalibrasyon durumunu çizer.

    Girdi görüntüyü değiştirmez; aynı boyutta yeni bir BGR görüntü döner. Görüntü renkli
    veya gri olabilir. El yoksa (`frame.present=False` veya `hand=None`) yalnızca fps ve
    durum yazıları çizilir; `lift` None ise çubuk yerine "--" yazılır.
    """
    out = to_bgr(image)
    h, w = out.shape[:2]
    u = max(12, round(h / 36))          # satır yüksekliği
    fs = u / 30.0                       # yazı ölçeği
    pad = u // 2

    if hand is not None and frame.present:
        draw_skeleton(out, hand)

    # --- sol üst: durum + parmak çubukları
    y = pad + u
    present_txt = "present: YES" if frame.present else "present: NO"
    _text(out, present_txt, (pad, y), fs, _GREEN if frame.present else _RED)
    y += u
    _text(out, f"conf {frame.confidence:4.2f}", (pad, y), fs)
    y += u + pad // 2

    label_w = int(u * 4.2)
    bar_w = int(u * 5)
    val_w = int(u * 2.4)
    _text(out, "curl", (pad + label_w, y), fs * 0.8, _CYAN)
    _text(out, "lift", (pad + label_w + bar_w + val_w + pad, y), fs * 0.8, _YELLOW)
    y += u
    show_fingers = frame.present
    for name in FINGERS:
        finger = frame.fingers.get(name) if show_fingers else None
        curl = None if finger is None else finger.curl
        lift = None if finger is None else finger.lift
        _text(out, name, (pad, y), fs * 0.85)
        x = pad + label_w
        _bar(out, x, y - u + 4, bar_w, u - 6, curl, _CYAN)
        _text(out, _fmt(curl), (x + bar_w + 4, y), fs * 0.8)
        x2 = x + bar_w + val_w + pad
        _bar(out, x2, y - u + 4, bar_w, u - 6, lift, _YELLOW)
        _text(out, _fmt(lift), (x2 + bar_w + 4, y), fs * 0.8)
        y += u + 2
    thumb = frame.thumb_out if frame.present else None
    _text(out, "thumb", (pad, y), fs * 0.85)
    _bar(out, pad + label_w, y - u + 4, bar_w, u - 6, thumb, _GREEN)
    _text(out, _fmt(thumb), (pad + label_w + bar_w + 4, y), fs * 0.8)

    # --- sağ üst: fps ve süre
    lines = (
        f"capture {stats.capture_fps:5.1f} fps",
        f"process {stats.process_fps:5.1f} fps",
        f"frame   {stats.process_ms:5.1f} ms",
    )
    (tw, _), _ = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    ry = pad + u
    for line in lines:
        _text(out, line, (max(pad, w - tw - pad), ry), fs)
        ry += u

    # --- alt: kalibrasyon
    if calib_state is not None:
        _draw_calibration(out, calib_state, u, fs, pad)
    else:
        _text(out, "c: kalibrasyon  t: tap  m: masa  s: alan  q/ESC: cikis",
              (pad, h - pad), fs * 0.8, _GRAY)
    return out


def _draw_calibration(out: np.ndarray, cs: CalibState, u: int, fs: float, pad: int) -> None:
    h, w = out.shape[:2]
    y = h - pad
    if cs.message:
        _text(out, cs.message, (pad, y), fs, _YELLOW)
        y -= u
    if cs.step is None:
        return
    n = STEPS.index(cs.step) + 1 if cs.step in STEPS else 0
    title = STEP_TITLES.get(cs.step, cs.step)
    _text(out, f"KALIBRASYON {n}/{len(STEPS)}: {title}", (pad, y), fs * 1.1, _YELLOW)
    y -= u
    _bar(out, pad, y - u + 4, max(10, w - 2 * pad), u - 6, cs.progress, _YELLOW)
    y -= u
    _bar(out, pad, y - u + 4, max(10, w - 2 * pad), u - 6, cs.total_progress, _GREEN)
    if cs.samples == 0:   # -1: kayıt evresi değil (uyarı gösterme)
        y -= u
        _text(out, "el gorunmuyor", (pad, y), fs, _RED)


@dataclass(frozen=True)
class TapState:
    """Tap kalibrasyonunun ekranda gösterilen durumu."""
    total: int              # istenen tap sayısı
    index: int              # geçerli tap (0 tabanlı); hazırlık süresinde 0
    lead: bool              # hazırlık süresi (örnek toplanmıyor)
    flash: bool             # pencere başlangıcı: şimdi tap yap işareti
    progress: float         # tüm oturum, 0–1
    samples: int            # şimdiye dek toplanan örnek sayısı


class TapSession:
    """Tap kalibrasyonu için zamanlı pencereler ve örnek toplama (kamera/GUI yok).

    Zaman ilk `add` çağrısındaki `t` ile başlar. `lead_s` saniye hazırlıktan sonra
    `n` pencere (her biri `window_s` sn) gelir; pencereler boyunca `(t, RawHand)`
    örnekleri `samples` listesine yazılır. `None` (el yok) örnekler atlanır ama zaman
    ilerler. Tüm pencereler bitince `done` olur ve sonraki örnekler yok sayılır.
    Gerçek saate bakmaz; yalnızca verilen `t` değerlerini kullanır.
    """

    def __init__(self, n: int = TAP_COUNT, window_s: float = TAP_WINDOW_SECONDS,
                 lead_s: float = TAP_LEAD_SECONDS,
                 flash_s: float = TAP_FLASH_SECONDS) -> None:
        if n < 1 or not window_s > 0 or lead_s < 0:
            raise ValueError("TapSession: n >= 1, window_s > 0, lead_s >= 0 olmalı")
        self.n = int(n)
        self.window_s = float(window_s)
        self.lead_s = float(lead_s)
        self.flash_s = float(flash_s)
        self.samples: list[tuple[float, RawHand]] = []
        self._t0: float | None = None
        self._elapsed = 0.0

    @property
    def duration(self) -> float:
        return self.lead_s + self.n * self.window_s

    @property
    def windows(self) -> list[tuple[float, float]]:
        """Tap pencerelerinin gerçek sınırları `[başlangıç, bitiş)`; oturum başlamadıysa boş.

        Örnek akışı pencereler arasında kesintisizdir; `fit_taps` pencereleri zaman
        boşluğundan ayıramaz, sınırlar bu yüzden açıkça verilir.
        """
        if self._t0 is None:
            return []
        start = self._t0 + self.lead_s
        return [(start + k * self.window_s, start + (k + 1) * self.window_s)
                for k in range(self.n)]

    @property
    def done(self) -> bool:
        return self._elapsed >= self.duration

    def add(self, hand: RawHand | None, t: float) -> None:
        """`t` anındaki örneği (hazırlık süresi dışındaysa) toplar; zamanı ilerletir."""
        if self.done:
            return
        if self._t0 is None:
            self._t0 = float(t)
        self._elapsed = max(self._elapsed, float(t) - self._t0)   # geri giden zaman yok sayılır
        if self.done or hand is None or self._elapsed < self.lead_s:
            return
        self.samples.append((float(t), hand))

    def state(self) -> TapState:
        lead = self._elapsed < self.lead_s
        w = max(0.0, self._elapsed - self.lead_s)
        index = 0 if lead else min(self.n - 1, int(w // self.window_s))
        phase = w - index * self.window_s
        return TapState(
            total=self.n, index=index, lead=lead,
            flash=(not lead) and not self.done and phase < self.flash_s,
            progress=min(1.0, self._elapsed / self.duration),
            samples=len(self.samples),
        )


class TableSession:
    """Masa modeli kalibrasyonu: geri sayım, sonra el mousepad'de gezinirken örnek toplama.

    Zaman ilk `add` çağrısındaki `t` ile başlar; ilk `countdown_s` saniyede örnek
    alınmaz, sonraki `record_s` saniyede el olan kareler `samples`'a yazılır.
    Gerçek saate bakmaz (kamera/GUI yok).
    """

    def __init__(self, countdown_s: float = TABLE_COUNTDOWN_SECONDS,
                 record_s: float = TABLE_RECORD_SECONDS) -> None:
        if countdown_s < 0 or not record_s > 0:
            raise ValueError("TableSession: countdown_s >= 0, record_s > 0 olmalı")
        self.countdown_s = float(countdown_s)
        self.record_s = float(record_s)
        self.samples: list[RawHand] = []
        self.points: list[tuple[float, float]] = []   # avuç merkezi, 0–1 (çizim için)
        self._t0: float | None = None
        self._elapsed = 0.0

    @property
    def done(self) -> bool:
        return self._elapsed >= self.countdown_s + self.record_s

    @property
    def recording(self) -> bool:
        return not self.done and self._elapsed >= self.countdown_s

    def add(self, hand: RawHand | None, t: float) -> None:
        if self.done:
            return
        if self._t0 is None:
            self._t0 = float(t)
        self._elapsed = max(self._elapsed, float(t) - self._t0)
        if self.done or hand is None or self._elapsed < self.countdown_s:
            return
        self.samples.append(hand)
        img = np.asarray(hand.image_pts, dtype=float)
        x, y = img[[0, 5, 9, 13, 17]].mean(axis=0)
        self.points.append((float(x), float(y)))

    def message(self) -> str:
        """Ekranda gösterilen yönerge ve kalan süre (ASCII)."""
        if self._elapsed < self.countdown_s:
            left = math.ceil(self.countdown_s - self._elapsed)
            return f"MASA: elini mousepad'e koy - {left}"
        left = math.ceil(self.countdown_s + self.record_s - self._elapsed)
        return f"MASA: eli mousepad'in her yerinde gezdir - {left} sn ({len(self.samples)})"


def draw_table_area(out: np.ndarray, points: Sequence[tuple[float, float]],
                    grid: int = TABLE_GRID) -> None:
    """Masa kalibrasyonunda taranan alanı çizer (yerinde): avuç izleri ve dış sınır.

    Görüntü `grid`×`grid` hücreye bölünür; avucun geçtiği hücreler yarı saydam yeşil
    boyanır, izler nokta, örneklerin dış sınırı (dışbükey zarf) çizgi olarak gösterilir.
    """
    if not points:
        return
    h, w = out.shape[:2]
    px = np.array([(min(w - 1, max(0, round(x * w))), min(h - 1, max(0, round(y * h))))
                   for x, y in points], dtype=np.int32)
    overlay = out.copy()
    cw, ch = w / grid, h / grid
    cells = {(int(x // cw), int(y // ch)) for x, y in px}
    for cx, cy in cells:
        cv2.rectangle(overlay, (int(cx * cw), int(cy * ch)),
                      (int((cx + 1) * cw) - 1, int((cy + 1) * ch) - 1), _GREEN, -1)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, dst=out)
    for x, y in px[:: max(1, len(px) // 300)]:
        cv2.circle(out, (int(x), int(y)), 2, _GREEN, -1)
    if len(px) >= 3:
        hull = cv2.convexHull(px)
        cv2.polylines(out, [hull], True, _YELLOW, 2)


class SurfaceSession:
    """Aktif alan kalibrasyonu: 4 köşe, her biri BOŞLUK -> geri sayım -> kayıt (kamera/GUI yok).

    Köşe sırası `CORNER_NAMES` (sol-uzak, sağ-uzak, sağ-yakın, sol-yakın). Kayıtta işaret
    parmağı ucunun görüntü konumu (`Finger.img`) toplanır; köşe değeri medyandır. Kayıtta hiç
    örnek yoksa aynı köşe yeniden BOŞLUK bekler. `redo()` (`g`): geri sayım/kayıt sürerken o
    köşeyi baştan alır; beklemedeyse bir önceki köşeye döner. Zaman yalnızca `add` ile verilen
    kare zamanından gelir.
    """

    def __init__(self, countdown_s: float = SURFACE_COUNTDOWN_SECONDS,
                 record_s: float = SURFACE_RECORD_SECONDS) -> None:
        if not (countdown_s >= 0 and record_s > 0):
            raise ValueError("countdown_s >= 0 ve record_s > 0 olmalı")
        self.countdown_s = float(countdown_s)
        self.record_s = float(record_s)
        self.corners: list[tuple[float, float]] = []
        self.samples: list[tuple[float, float]] = []   # geçerli köşenin kayıtları
        self.note = ""                                  # son uyarı (ör. parmak görünmedi)
        self._pending = False
        self._t0: float | None = None
        self._now: float | None = None

    @property
    def done(self) -> bool:
        return len(self.corners) >= len(CORNER_NAMES)

    @property
    def index(self) -> int:
        return min(len(self.corners), len(CORNER_NAMES) - 1)

    @property
    def phase(self) -> str:
        """"wait" (boşluk bekleniyor), "countdown", "record" veya "done"."""
        if self.done:
            return "done"
        if self._t0 is None or self._now is None:
            return "countdown" if self._pending else "wait"
        return "countdown" if self._now - self._t0 < self.countdown_s else "record"

    @property
    def remaining(self) -> float:
        if self._t0 is None or self._now is None:
            return self.countdown_s if self._pending else 0.0
        el = self._now - self._t0
        if el < self.countdown_s:
            return self.countdown_s - el
        return max(0.0, self.countdown_s + self.record_s - el)

    def start(self) -> None:
        """Beklemedeki köşeyi başlatır; geri sayım/kayıt sürerken etkisizdir."""
        if not self.done and self.phase == "wait":
            self._pending = True
            self.note = ""

    def redo(self) -> None:
        """Sürmekte olan köşeyi baştan alır; beklemedeyse bir önceki köşeye döner."""
        if self.done:
            return
        if self.phase == "wait":
            if self.corners:
                self.corners.pop()
                self.note = f"{CORNER_NAMES[len(self.corners)]} kosesi tekrar"
        else:
            self.note = f"{CORNER_NAMES[self.index]} kosesi tekrar"
        self.samples = []
        self._pending, self._t0 = False, None

    def add(self, tip: tuple[float, float] | None, t: float) -> None:
        """Kare zamanını ilerletir; kayıt evresinde işaret ucunu (`None`: görünmüyor) ekler."""
        if self.done:
            return
        t = float(t)
        if self._pending and self._t0 is None:
            self._t0 = t
        if self._now is None or t > self._now:
            self._now = t
        if self._t0 is None:
            return
        el = self._now - self._t0
        if el >= self.countdown_s + self.record_s:
            if self.samples:
                pts = np.asarray(self.samples, dtype=float)
                med = np.median(pts, axis=0)
                self.corners.append((float(med[0]), float(med[1])))
            else:
                self.note = "isaret parmagi gorunmedi, tekrar BOSLUK"
            self.samples = []
            self._pending, self._t0 = False, None
            return
        if el >= self.countdown_s and tip is not None:
            self.samples.append((float(tip[0]), float(tip[1])))

    def message(self, aspect: float = calibration.DEFAULT_ASPECT) -> str:
        """Ekranda gösterilen yönerge (ASCII)."""
        n, total = self.index + 1, len(CORNER_NAMES)
        name = CORNER_NAMES[self.index]
        phase = self.phase
        if phase == "wait":
            msg = f"{n}/{total} {name} kose: isaret parmagi ucunu koseye koy, BOSLUK"
            if self.note:
                msg = f"{self.note} -- {msg}"
            return msg
        if phase == "countdown":
            return f"{n}/{total} {name}: basliyor {math.ceil(self.remaining)}"
        if phase == "record":
            return (f"{n}/{total} {name}: KAYIT, kipirdama ({self.remaining:.1f} sn, "
                    f"{len(self.samples)})")
        return f"ALAN tamam: {_aspect_text(aspect)}"

    def hint(self, aspect: float) -> str:
        """İkinci satır: tuşlar ve en-boy varsayımı (ASCII)."""
        return f"ALAN KALIBRASYONU  g: kose tekrar/geri  ({_aspect_text(aspect)})"

    def result(self, aspect: float, image_aspect: float) -> SurfaceCalibration:
        if not self.done:
            raise ValueError("yüzey kalibrasyonu bitmedi")
        return SurfaceCalibration(tuple(self.corners), aspect=aspect, image_aspect=image_aspect)


def _aspect_text(aspect: float) -> str:
    """Ekranda gösterilen alan en-boy açıklaması (ASCII)."""
    if math.isclose(aspect, 1.5):
        return "alan 30x20 kabul edildi"
    return f"alan en/boy {aspect:.2f} kabul edildi"


def draw_surface(out: np.ndarray, surface: SurfaceCalibration | None,
                 tip: tuple[float, float] | None = None,
                 partial: Sequence[tuple[float, float]] = ()) -> None:
    """Aktif alanı çizer (yerinde): dörtgen, parmak ucu alan içindeyse yeşil, dışındaysa kırmızı.

    `partial`: kalibrasyon sürerken toplanan köşeler (numaralı nokta ve çizgi).
    """
    h, w = out.shape[:2]
    thick = max(1, round(min(h, w) / 240))

    def px(p: tuple[float, float]) -> tuple[int, int]:
        return int(round(p[0] * w)), int(round(p[1] * h))

    if partial:
        pts = [px(p) for p in partial]
        for a, b in zip(pts, pts[1:], strict=False):
            cv2.line(out, a, b, _YELLOW, thick, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(out, p, thick + 3, _YELLOW, -1, cv2.LINE_AA)
            _text(out, str(i + 1), (p[0] + 6, p[1] - 6), 0.5, _YELLOW)
    if surface is None:
        return
    uv = surface.to_surface(tip) if tip is not None else None
    inside = SurfaceCalibration.contains(uv)
    color = _GREEN if inside else _RED
    poly = np.array([px(c) for c in surface.corners_img], dtype=np.int32)
    cv2.polylines(out, [poly], True, color, thick + 1, cv2.LINE_AA)
    if tip is not None and uv is not None:
        p = px(tip)
        cv2.circle(out, p, thick + 4, color, 2, cv2.LINE_AA)
        _text(out, f"u {uv[0]:4.2f} v {uv[1]:4.2f}", (p[0] + 10, p[1] + 4), 0.5, color)


def _draw_hint(out: np.ndarray, s: str) -> None:
    """Alt mesaj satırının bir üstüne gri yardım satırı."""
    h = out.shape[0]
    u = max(12, round(h / 36))
    _text(out, s, (u // 2, h - u // 2 - u), u / 30.0 * 0.8, _GRAY)


def draw_tap_prompt(out: np.ndarray, ts: TapState) -> None:
    """Tap kalibrasyon yönergesini, ilerlemeyi ve tap işaretini `out` üstüne çizer (yerinde)."""
    h, w = out.shape[:2]
    u = max(12, round(h / 36))
    fs = u / 30.0
    pad = u // 2
    y = h - pad - u          # en alttaki satır yardım yazısına ayrılmıştır
    _text(out, f"isaret parmagiyla {ts.total} kez tap yap", (pad, y), fs * 1.1, _YELLOW)
    y -= u
    if ts.lead:
        _text(out, "hazir ol...", (pad, y), fs, _WHITE)
    else:
        _text(out, f"TAP {ts.index + 1}/{ts.total}", (pad, y), fs, _WHITE)
    y -= u
    _bar(out, pad, y - u + 4, max(10, w - 2 * pad), u - 6, ts.progress, _GREEN)
    if not ts.lead and ts.samples == 0:
        y -= u
        _text(out, "el gorunmuyor", (pad, y), fs, _RED)
    if ts.flash:
        # ekran işareti: kalın çerçeve ve ortada "TAP!"
        t = max(3, round(min(h, w) / 60))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), _GREEN, t)
        label = "TAP!"
        scale = fs * 3.0
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        _text(out, label, (max(pad, (w - tw) // 2), (h + th) // 2), scale, _GREEN)


def render_lift_graph(history: Sequence[tuple[float, float | None, float | None]],
                      width: int, height: int,
                      window_s: float = LIFT_GRAPH_SECONDS) -> np.ndarray:
    """Son `window_s` saniyenin işaret/orta parmak değerlerini (height, width, 3) BGR çizer.

    `history` elemanları `(t, işaret, orta)`; değer lift'tir, lift yoksa curl verilir,
    el yoksa `None` (çizgi kopar). Sağ kenar en son örneğin zamanıdır; eksen 0 altta, 1
    üsttedir. Boş geçmişte yalnızca zemin ve çerçeve çizilir.
    """
    width, height = max(1, int(width)), max(1, int(height))
    out = np.full((height, width, 3), 30, np.uint8)
    cv2.rectangle(out, (0, 0), (width - 1, height - 1), _GRAY, 1)
    cv2.line(out, (0, height // 2), (width - 1, height // 2), _BAR_BG, 1)
    pts = [e for e in history if np.isfinite(e[0])]
    if pts and window_s > 0:
        t_end = max(e[0] for e in pts)
        t_start = t_end - window_s
        for k, name in enumerate(LIFT_GRAPH_FINGERS):
            run: list[tuple[int, int]] = []
            segments: list[list[tuple[int, int]]] = [run]
            for e in sorted(pts, key=lambda e: e[0]):
                v = e[1 + k]
                if e[0] < t_start:
                    continue
                if v is None or not np.isfinite(v):
                    if run:
                        run = []
                        segments.append(run)
                    continue
                x = int(round((e[0] - t_start) / window_s * (width - 1)))
                y = int(round((1.0 - float(np.clip(v, 0.0, 1.0))) * (height - 1)))
                run.append((x, y))
            for seg in segments:
                if len(seg) >= 2:
                    cv2.polylines(out, [np.array(seg, np.int32)], False,
                                  _GRAPH_COLORS[name], 1, cv2.LINE_8)
                elif len(seg) == 1:
                    cv2.circle(out, seg[0], 1, _GRAPH_COLORS[name], -1)
    return out


def draw_lift_graph(out: np.ndarray,
                    history: Sequence[tuple[float, float | None, float | None]]) -> None:
    """Lift grafiğini `out` görüntüsünün sağ üst köşesine (fps yazılarının altına) yerleştirir."""
    h, w = out.shape[:2]
    u = max(12, round(h / 36))
    pad = u // 2
    gw, gh = max(0, min(w // 3, 8 * u * 2)), max(0, min(h // 6, 5 * u))
    x0, y0 = w - gw - pad, pad + 4 * u + pad
    if gw < 24 or gh < 16 or x0 < 0 or y0 + gh + 14 > h:
        return   # görüntü çok küçük
    out[y0:y0 + gh, x0:x0 + gw] = render_lift_graph(history, gw, gh)
    fs = u / 30.0 * 0.7
    _text(out, "lift idx", (x0 + 3, y0 + gh + 12), fs, _GRAPH_COLORS["index"])
    _text(out, "mid", (x0 + int(gw * 0.6), y0 + gh + 12), fs, _GRAPH_COLORS["middle"])


# ---------------------------------------------------------------------- döngü


def _log(msg: str) -> None:
    """Olay günlüğü (stderr); pencere dışından neler olduğunu görmek için."""
    print(f"[debug_view] {msg}", file=sys.stderr, flush=True)


class GuidedCalibration:
    """Adım adım el kalibrasyonu (kamera/GUI yok).

    Her adım `start()` (boşluk tuşu) ile başlar: `countdown_s` saniye geri sayım (örnek
    toplanmaz), sonra `record_s` saniye kayıt. Kayıt bitince sonraki adım yeniden
    `start()` bekler. Zaman yalnızca `add` ile verilen kare zamanından gelir.
    `done`, `samples` ve `result()` `CalibrationSession` ile aynı anlamdadır.
    """

    def __init__(self, countdown_s: float = GUIDED_COUNTDOWN_SECONDS,
                 record_s: float = GUIDED_RECORD_SECONDS) -> None:
        if not (countdown_s >= 0 and record_s > 0):
            raise ValueError("countdown_s >= 0 ve record_s > 0 olmalı")
        self.countdown_s = float(countdown_s)
        self.record_s = float(record_s)
        self.samples: dict[str, list[RawHand]] = {s: [] for s in STEPS}
        self._index = 0
        self._pending = False            # start() çağrıldı, ilk karede zaman başlar
        self._t0: float | None = None    # geçerli adımın başlangıcı (geri sayım dahil)
        self._now: float | None = None

    @property
    def done(self) -> bool:
        return self._index >= len(STEPS)

    @property
    def current_step(self) -> str | None:
        return None if self.done else STEPS[self._index]

    @property
    def phase(self) -> str:
        """"wait" (boşluk bekleniyor), "countdown", "record" veya "done"."""
        if self.done:
            return "done"
        if self._t0 is None or self._now is None:
            return "countdown" if self._pending else "wait"
        return "countdown" if self._now - self._t0 < self.countdown_s else "record"

    @property
    def remaining(self) -> float:
        """Geçerli evrenin kalan süresi (sn); beklemede 0."""
        if self._t0 is None or self._now is None:
            return self.countdown_s if self._pending else 0.0
        el = self._now - self._t0
        if el < self.countdown_s:
            return self.countdown_s - el
        return max(0.0, self.countdown_s + self.record_s - el)

    @property
    def progress(self) -> float:
        """Kayıt evresinin ilerlemesi (0–1)."""
        if self.phase != "record" or self._t0 is None or self._now is None:
            return 0.0
        return min(1.0, (self._now - self._t0 - self.countdown_s) / self.record_s)

    @property
    def total_progress(self) -> float:
        return min(1.0, (self._index + self.progress) / len(STEPS))

    def start(self) -> None:
        """Beklemedeki adımı başlatır; geri sayım ya da kayıt sürerken etkisizdir."""
        if not self.done and self.phase == "wait":
            self._pending = True

    def add(self, hand: RawHand | None, t: float) -> None:
        """Kare zamanını ilerletir; kayıt evresinde eli geçerli adıma ekler."""
        if self.done:
            return
        t = float(t)
        if self._pending and self._t0 is None:
            self._t0 = t
        if self._now is None or t > self._now:
            self._now = t
        if self._t0 is None:
            return
        el = self._now - self._t0
        if el >= self.countdown_s + self.record_s:
            self._index += 1
            self._pending, self._t0 = False, None
            return
        if el >= self.countdown_s and hand is not None:
            self.samples[STEPS[self._index]].append(hand)

    def result(self, aspect: float = 1.0) -> HandCalibration:
        if not self.done:
            raise ValueError("kalibrasyon oturumu bitmedi")
        return hand_calib.fit_hand(self.samples["open"], self.samples["fist"],
                                   self.samples["thumb_closed"], aspect)


def _guided_state(g: GuidedCalibration) -> CalibState:
    step = g.current_step or ""
    hint = STEP_HINTS.get(step, "")
    phase = g.phase
    if phase == "wait":
        msg = f"{hint}. Hazir olunca BOSLUK"
    elif phase == "countdown":
        msg = f"{hint} -- basliyor: {math.ceil(g.remaining)}"
    else:
        msg = f"KAYIT: {hint} -- kipirdama ({g.remaining:.1f} sn)"
    return CalibState(step=g.current_step, progress=g.progress,
                      total_progress=g.total_progress,
                      samples=len(g.samples.get(step, [])) if phase == "record" else -1,
                      message=msg)


def _calib_state(session: CalibrationSession, samples: int, message: str = "") -> CalibState:
    return CalibState(
        step=session.current_step,
        progress=session.progress,
        total_progress=session.total_progress,
        samples=samples,
        message=message,
    )


def _current_calibration(calib_path: Path) -> HandCalibration:
    """Dosyadaki kalibrasyon; dosya yok veya bozuksa `DEFAULT` (LiveSource da böyle başlar)."""
    try:
        return hand_calib.load(calib_path) or hand_calib.DEFAULT
    except ValueError:
        return hand_calib.DEFAULT


def _lift_value(frame: HandFrame, name: str) -> float | None:
    """Grafik değeri: lift; yoksa curl; el yoksa None."""
    finger = frame.fingers.get(name) if frame.present else None
    if finger is None:
        return None
    return finger.curl if finger.lift is None else finger.lift


def _enable_cocoa_close() -> tuple[Any, bool] | None:
    """macOS/Cocoa: pencereye kapatma (X) düğmesini ekler; (pencere, eski ayar) veya None.

    OpenCV 4.11 Cocoa pencerelerini kapatılabilir stil olmadan oluşturur. Pencere kapanınca
    serbest bırakılmasın diye `releasedWhenClosed` kapatılır; HighGUI pencereyi kendi
    kaydından silene dek yaşamalıdır. `_restore_cocoa` eski ayarı geri yükler.
    tools/replay.py'deki aynı fonksiyonun kopyası: vision, control'ü import eden
    tools.replay'i import edemez.
    """
    ui = getattr(cv2, "currentUIFramework", None)
    if platform != "darwin" or ui is None or ui() != "COCOA":
        return None
    try:
        from AppKit import NSApplication, NSWindowStyleMaskClosable
    except ImportError:
        return None
    for window in NSApplication.sharedApplication().windows():
        if window.title() == WINDOW_NAME:
            style = window.styleMask()
            if not style & NSWindowStyleMaskClosable:
                releases_on_close = window.isReleasedWhenClosed()
                window.setReleasedWhenClosed_(False)
                window.setStyleMask_(style | NSWindowStyleMaskClosable)
                return window, releases_on_close
            break
    return None


def _restore_cocoa(cocoa: tuple[Any, bool] | None) -> None:
    if cocoa is not None:
        window, releases_on_close = cocoa
        with suppress(Exception):
            window.setReleasedWhenClosed_(releases_on_close)


def _window_visible() -> bool | None:
    """Pencere görünür mü? Backend desteklemiyorsa None."""
    try:
        v = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return None
    return v >= 1


def run(source: LiveSource, calib_path: Path, surface_path: Path | None = None,
        surface_aspect: float = calibration.DEFAULT_ASPECT) -> None:
    """İskelet, curl/lift çubukları, lift grafiği, fps. Ana thread'de çalışır.

    Tuşlar: `c` kalibrasyon, `t` tap kalibrasyonu, `m` masa, `s` aktif alan, `q`/ESC çıkış;
    pencerenin X düğmesi de çıkartır. Kalibrasyon bitince yeni değerler
    `source.set_calibration` ile hemen geçerli olur ve `calib_path` dosyasına yazılır.
    Aktif alan (`s`) bitince `source.set_surface` ile uygulanır ve `surface_path`
    (verilmezse `~/.tkt/surface.json`) dosyasına yazılır; en-boy `surface_aspect`.
    """
    if surface_path is None:
        surface_path = Path(calibration.DEFAULT_PATH).expanduser()
    surf: SurfaceSession | None = None
    surface: SurfaceCalibration | None = getattr(source, "surface", None)
    session: CalibrationSession | GuidedCalibration | None = None
    tap: TapSession | None = None
    table: TableSession | None = None
    table_points: list[tuple[float, float]] = []   # son masa kalibrasyonunun alanı
    calib: HandCalibration | None = None     # geçerli kalibrasyon; ilk ihtiyaçta dosyadan
    history: deque[tuple[float, float | None, float | None]] = deque()
    message = ""
    message_until = 0.0
    seen_visible = False
    shown = False
    cocoa: tuple[Any, bool] | None = None
    try:
        for frame, hand, image in source.frames_raw():
            if message and frame.t > message_until:
                message = ""

            calib_state: CalibState | None = None
            tap_state: TapState | None = None
            if session is not None:
                session.add(hand if frame.present else None, frame.t)
                if session.done:
                    calib = calib or _current_calibration(calib_path)
                    message, message_until, calib = _finish_calibration(
                        session, source, calib, calib_path, frame.t)
                    counts = {k: len(v) for k, v in session.samples.items()}
                    _log(f"kalibrasyon bitti: ornekler={counts} -> {message} ({calib_path})")
                    session = None
                elif isinstance(session, GuidedCalibration):
                    calib_state = _guided_state(session)
                else:
                    calib_state = _calib_state(
                        session, len(session.samples.get(session.current_step or "", [])),
                        message)
            if tap is not None:
                tap.add(hand if frame.present else None, frame.t)
                if tap.done:
                    calib = calib or _current_calibration(calib_path)
                    message, message_until, calib = _finish_tap(
                        tap, source, calib, calib_path, frame.t)
                    tap = None
                else:
                    tap_state = tap.state()
            if table is not None:
                table.add(hand if frame.present else None, frame.t)
                if table.done:
                    calib = calib or _current_calibration(calib_path)
                    message, message_until, calib = _finish_table(
                        table, source, calib, calib_path, frame.t)
                    _log(f"masa bitti: ornek={len(table.samples)} -> {message}")
                    table_points = table.points
                    table = None
                else:
                    calib_state = CalibState(None, 0.0, 0.0, 0, table.message())
            if surf is not None:
                index = frame.fingers.get("index") if frame.present else None
                surf.add(None if index is None else index.img, frame.t)
                if surf.done:
                    message, message_until, surface = _finish_surface(
                        surf, source, surface, surface_path, surface_aspect, frame.t)
                    _log(f"alan bitti: koseler={surf.corners} -> {message}")
                    surf = None
                else:
                    calib_state = CalibState(None, 0.0, 0.0, -1, surf.message(surface_aspect))
            if calib_state is None and message:
                calib_state = CalibState(None, 0.0, 0.0, 0, message)

            history.append((frame.t, *(_lift_value(frame, n) for n in LIFT_GRAPH_FINGERS)))
            while history and frame.t - history[0][0] > LIFT_GRAPH_SECONDS:
                history.popleft()

            stats = Stats(source.capture_fps(), source.process_fps(), source.last_process_ms)
            out = draw_overlay(image, frame, hand, stats, calib_state)
            draw_lift_graph(out, history)
            if tap_state is not None:
                draw_tap_prompt(out, tap_state)
            if surf is not None or surface is not None:
                index = frame.fingers.get("index") if frame.present else None
                draw_surface(out, surface if surf is None else None,
                             None if index is None else index.img,
                             surf.corners if surf is not None else ())
            if surf is not None:
                _draw_hint(out, surf.hint(surface_aspect))
            if table is not None:
                draw_table_area(out, table.points)
            elif table_points and message:
                draw_table_area(out, table_points)   # sonuç mesajı süresince alan kalır
            cv2.imshow(WINDOW_NAME, out)
            if not shown:
                shown = True
                cocoa = _enable_cocoa_close()
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT:
                _log("cikis tusu")
                break
            visible = _window_visible()
            if visible:
                seen_visible = True
            elif visible is False and seen_visible:
                _log("pencere kapatildi")
                break          # pencere X düğmesiyle kapatıldı
            if key == KEY_CALIB:
                session, tap, table, surf = GuidedCalibration(), None, None, None
                message = ""
                _log("c: kalibrasyon basladi")
            elif key == KEY_NEXT and surf is not None:
                surf.start()
                _log(f"bosluk: kose {surf.index + 1} -> {surf.phase}")
            elif key == KEY_NEXT and isinstance(session, GuidedCalibration):
                session.start()
                _log(f"bosluk: adim {session.current_step} -> {session.phase}")
            elif key == KEY_REDO and surf is not None:
                surf.redo()
                _log(f"g: kose {surf.index + 1} tekrar")
            elif key == KEY_TAP:
                tap, session, table, surf = TapSession(), None, None, None
                message = ""
                _log("t: tap kalibrasyonu basladi")
            elif key == KEY_TABLE:
                table, session, tap, surf = TableSession(), None, None, None
                message = ""
                _log("m: masa kalibrasyonu basladi")
            elif key == KEY_SURFACE:
                surf, session, tap, table = SurfaceSession(), None, None, None
                message = ""
                _log(f"s: alan kalibrasyonu basladi ({_aspect_text(surface_aspect)})")
            elif key not in (255, -1):
                _log(f"tus: {key!r} ({chr(key) if 32 <= key < 127 else '?'})")
    finally:
        source.close()
        _restore_cocoa(cocoa)
        cv2.destroyAllWindows()


def _apply(source: LiveSource, result: HandCalibration, calib_path: Path) -> str | None:
    """Kalibrasyonu hemen uygular, sonra dosyaya yazar. Kayıt hatası varsa mesajı döner."""
    source.set_calibration(result)
    try:
        hand_calib.save(result, calib_path)
    except OSError as e:
        return f"Uygulandi ama kayit hatasi: {e}"[:90]
    return None


def _finish_calibration(session: CalibrationSession | GuidedCalibration, source: LiveSource,
                        calib: HandCalibration, calib_path: Path,
                        t: float) -> tuple[str, float, HandCalibration]:
    """Oturum sonucunu uygular ve kaydeder; (ekran mesajı, bitiş zamanı, geçerli kalibrasyon).

    Daha önce çıkarılmış `tap_ref`/`angle_sign` korunur: eğri kalibrasyonu lift'i etkilemez.
    """
    try:
        result = session.result(aspect=_source_aspect(source))
    except ValueError as e:
        # Örn. örnek yok (el görünmedi) veya yumruk açık elden büyük değil.
        return f"Kalibrasyon basarisiz: {e}"[:90], t + MESSAGE_SECONDS, calib
    result = dataclasses.replace(result, tap_ref=calib.tap_ref, angle_sign=calib.angle_sign,
                                 table=calib.table)
    err = _apply(source, result, calib_path)
    return err or "Kaydedildi ve uygulandi", t + MESSAGE_SECONDS, result


def _finish_surface(surf: SurfaceSession, source: LiveSource,
                    surface: SurfaceCalibration | None, surface_path: Path, aspect: float,
                    t: float) -> tuple[str, float, SurfaceCalibration | None]:
    """4 köşeden yüzey kalibrasyonu kurar, uygular ve kaydeder; (mesaj, bitiş, geçerli alan)."""
    until = t + MESSAGE_SECONDS
    try:
        result = surf.result(aspect=aspect, image_aspect=_source_aspect(source))
    except ValueError as e:
        return f"Alan kalibrasyonu basarisiz: {e}"[:90], until, surface
    set_surface = getattr(source, "set_surface", None)
    if set_surface is not None:
        set_surface(result)
    try:
        calibration.save(result, surface_path)
    except OSError as e:
        return f"Alan uygulandi ama kayit hatasi: {e}"[:90], until, result
    return f"Alan kaydedildi ve uygulandi ({_aspect_text(aspect)})", until, result


def _source_aspect(source: LiveSource) -> float:
    """Kaynak görüntünün genişlik / yüksekliği; bilinmiyorsa 1.0."""
    info = getattr(source, "info", None)
    w, h = getattr(info, "width", 0), getattr(info, "height", 0)
    try:
        a = float(w) / float(h)
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    return a if math.isfinite(a) and a > 0 else 1.0


def _finish_tap(tap: TapSession, source: LiveSource, calib: HandCalibration,
                calib_path: Path, t: float) -> tuple[str, float, HandCalibration]:
    """Tap örneklerinden `fit_taps` ile kalibrasyonu çıkarır, uygular ve kaydeder."""
    until = t + MESSAGE_SECONDS
    if not tap.samples:
        return "Tap kalibrasyonu basarisiz: el gorunmedi", until, calib
    try:
        result = hand_calib.fit_taps(calib, tap.samples, aspect=_source_aspect(source),
                                     windows=tap.windows)
    except NotImplementedError:
        return "fit_taps henuz yok", until, calib
    except ValueError as e:
        return f"Tap kalibrasyonu basarisiz: {e}"[:90], until, calib
    err = _apply(source, result, calib_path)
    return err or "Tap kaydedildi ve uygulandi", until, result


def _finish_table(table: TableSession, source: LiveSource, calib: HandCalibration,
                  calib_path: Path, t: float) -> tuple[str, float, HandCalibration]:
    """Masa örneklerinden `fit_table`; diğer kalibrasyon alanları korunur."""
    until = t + MESSAGE_SECONDS
    try:
        model = hand_calib.fit_table(table.samples, aspect=_source_aspect(source))
    except ValueError as e:
        return f"Masa kalibrasyonu basarisiz: {e}"[:90], until, calib
    result = dataclasses.replace(calib, table=model)
    err = _apply(source, result, calib_path)
    return err or "Masa kaydedildi ve uygulandi", until, result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.debug_view")
    parser.add_argument("--config", default="config/vision.toml")
    args = parser.parse_args(argv)
    cfg = load_vision_config(args.config)
    calib_path = Path(cfg["hand"]["calibration"]).expanduser()
    source = open_live_source(args.config)
    surface_cfg = cfg.get("surface")
    if surface_cfg is None:
        run(source, calib_path)
    else:
        run(source, calib_path,
            surface_path=Path(surface_cfg.get("file", calibration.DEFAULT_PATH)).expanduser(),
            surface_aspect=float(surface_cfg.get("aspect", calibration.DEFAULT_ASPECT)))


if __name__ == "__main__":
    main()
