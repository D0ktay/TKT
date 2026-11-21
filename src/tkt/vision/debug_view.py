"""Debug penceresi (V3). PART2 2.2, 2.4.

macOS'ta `cv2.imshow` ana thread'de çalışmalıdır; `run` döngüyü çağıran thread'de kurar.
Çizim saf fonksiyonlardadır (`draw_overlay`); pencere ve klavye yalnızca `run` içindedir.

cv2.putText yalnızca ASCII çizer; ekran metinleri bu yüzden Türkçe karaktersizdir.

CLI: python -m tkt.vision.debug_view --config config/vision.toml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tkt.core.types import FINGERS, HandFrame
from tkt.vision import hand_calib
from tkt.vision.hand_calib import STEPS, CalibrationSession
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

# Kalibrasyon sonuç mesajının ekranda kalma süresi (kare zamanı, sn).
MESSAGE_SECONDS = 4.0
RESTART_NOTE = "Yeniden baslatinca gecerli"

KEY_QUIT = (ord("q"), 27)   # q, ESC
KEY_CALIB = ord("c")

# BGR renkler
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (80, 220, 80)
_RED = (60, 60, 240)
_YELLOW = (60, 220, 240)
_CYAN = (230, 200, 60)
_GRAY = (150, 150, 150)
_BAR_BG = (50, 50, 50)


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
        _text(out, "c: kalibrasyon   q/ESC: cikis", (pad, h - pad), fs * 0.8, _GRAY)
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
    if cs.samples == 0:
        y -= u
        _text(out, "el gorunmuyor", (pad, y), fs, _RED)


# ---------------------------------------------------------------------- döngü


def _calib_state(session: CalibrationSession, samples: int, message: str = "") -> CalibState:
    return CalibState(
        step=session.current_step,
        progress=session.progress,
        total_progress=session.total_progress,
        samples=samples,
        message=message,
    )


def run(source: LiveSource, calib_path: Path) -> None:
    """İskelet, curl/lift çubukları, fps; `c` kalibrasyon, `q`/ESC çıkış. Ana thread'de çalışır.

    Kalibrasyon bitince sonuç `calib_path` dosyasına yazılır. LiveSource kalibrasyonu
    çalışırken değiştirmek için genel bir yol sunmadığından yeni değerler uygulama
    yeniden başlatılınca geçerli olur; ekranda bu yazılır.
    """
    session: CalibrationSession | None = None
    message = ""
    message_until = 0.0
    try:
        for frame, hand, image in source.frames_raw():
            if message and frame.t > message_until:
                message = ""

            calib_state: CalibState | None = None
            if session is not None:
                session.add(hand if frame.present else None, frame.t)
                if session.done:
                    message, message_until = _finish_calibration(session, calib_path, frame.t)
                    session = None
                else:
                    calib_state = _calib_state(
                        session, len(session.samples.get(session.current_step or "", [])),
                        message)
            if calib_state is None and message:
                calib_state = CalibState(None, 0.0, 0.0, 0, message)

            stats = Stats(source.capture_fps(), source.process_fps(), source.last_process_ms)
            cv2.imshow(WINDOW_NAME, draw_overlay(image, frame, hand, stats, calib_state))
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT:
                break
            if key == KEY_CALIB:
                session = CalibrationSession()
                message = ""
    finally:
        source.close()
        cv2.destroyAllWindows()


def _finish_calibration(session: CalibrationSession, calib_path: Path,
                        t: float) -> tuple[str, float]:
    """Oturum sonucunu kaydeder; (ekran mesajı, mesajın biteceği kare zamanı) döner."""
    try:
        result = session.result()
        hand_calib.save(result, calib_path)
    except ValueError as e:
        # Örn. örnek yok (el görünmedi) veya yumruk açık elden büyük değil.
        return f"Kalibrasyon basarisiz: {e}"[:90], t + MESSAGE_SECONDS
    except OSError as e:
        return f"Kayit hatasi: {e}"[:90], t + MESSAGE_SECONDS
    return f"Kaydedildi. {RESTART_NOTE}", t + MESSAGE_SECONDS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.debug_view")
    parser.add_argument("--config", default="config/vision.toml")
    args = parser.parse_args(argv)
    cfg = load_vision_config(args.config)
    calib_path = Path(cfg["hand"]["calibration"]).expanduser()
    source = open_live_source(args.config)
    run(source, calib_path)


if __name__ == "__main__":
    main()
