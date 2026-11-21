"""Vision ile control arasındaki sözleşme (PART1, 1.3).

Bu dosya kilitlidir. Değişiklik prosedürü PART1 1.3'te: ayrı `core` PR'ı ve
`SCHEMA_VERSION` artışı.

Koordinatlar kullanıcının bakış açısındadır: `x` 0 sol, 1 sağ; `y` 0 uzak kenar
(kamera tarafı), 1 yakın kenar (kullanıcı tarafı).

Şema 2: `HandFrame.pinch` eklendi (isteğe bağlı, varsayılan None). Şema 1 kayıtları
okunurken `pinch=None` ile şema 2'ye dönüştürülür (`core/io.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 2
# Okunabilen kayıt şemaları; eskiler okunurken güncel sürüme dönüştürülür.
READABLE_SCHEMAS = (1, 2)
FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class Finger:
    img: tuple[float, float]             # parmak ucu, 0–1 (kullanıcı bakışı)
    surface: tuple[float, float] | None  # aktif alanda 0–1 (aktif alan yoksa None)
    curl: float                          # 0 düz, 1 yumruktaki kadar kıvrık
    lift: float | None                   # 0 hizada, 1 tap yüksekliğinde (Faz 1'de None)


@dataclass(frozen=True)
class HandFrame:
    t: float                    # karenin yakalandığı an (monotonic, sn)
    frame_id: int               # her kamera karesinde 1 artar
    present: bool
    fingers: dict[str, Finger]  # anahtarlar: FINGERS
    thumb_out: float            # 0 ele yapışık, 1 açık
    palm: tuple[float, float]   # el merkezi, 0–1
    hand_size: float            # bilek–orta parmak kökü mesafesi (görüntü genişliğine oranlı)
    confidence: float           # 0–1
    # Şema 2, isteğe bağlı: başparmak ucu ile her parmak ucu arası mesafe, anahtarlar FINGERS.
    # 0 uçlar değiyor, açıldıkça artar (>= 0; ölçeği vision belirler). Ölçülmüyorsa
    # (Sürüm C açık değil) veya present=False ise None.
    pinch: dict[str, float] | None = None


@dataclass(frozen=True)
class SourceInfo:
    schema: int
    width: int
    height: int
    fps: float
    hand: str                   # "right" | "left"
    note: str = ""
