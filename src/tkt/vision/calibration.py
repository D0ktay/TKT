"""Aktif alan (yüzey) kalibrasyonu (V3-4, deney). PART1 1.1, PART2 2.6.

Masadaki dikdörtgen alanın görüntüdeki 4 köşesinden homografi kurulur; parmak ucunun
görüntü konumu (`Finger.img`) alan koordinatına (`Finger.surface`) çevrilir: (0, 0) sol-uzak,
(1, 1) sağ-yakın köşe. Alan içi 0–1, dışı <0 veya >1. Eğik bakışın perspektif kısalması
böylece düzelir: alanın uzak ve yakın kenarında eşit fiziksel hareket eşit (u, v) farkı verir.

Köşe sırası "kullanıcı bakışı" koordinatlarındadır (LiveSource'un verdiği `img` ile aynı uzay):
sol-uzak, sağ-uzak, sağ-yakın, sol-yakın.

Pad'in en-boy oranı (`aspect`, genişlik/yükseklik) `SourceInfo`'ya değil bu dosyaya yazılır;
`core` değişmez. Control aynı değeri kendi config'inden (`[surface_pointer] aspect`) okur.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CORNER_NAMES = ("SOL-UZAK", "SAG-UZAK", "SAG-YAKIN", "SOL-YAKIN")
DEFAULT_ASPECT = 1.5          # 30 x 20 cm
DEFAULT_PATH = "~/.tkt/surface.json"

# Dörtgenin alanı görüntü alanının bu oranından küçükse kalibrasyon reddedilir.
MIN_AREA_FRACTION = 0.01
# Ufuk çizgisinin ötesi (homografi paydası ~0 veya negatif) ve uzak dışarısı bu sınırlara
# kırpılır: değerler sonlu kalır (kayıt JSON'u NaN/inf kabul etmez), "dışarı" anlamı korunur.
OUTSIDE_MIN = -1.0
OUTSIDE_MAX = 2.0
_EPS = 1e-9

_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _validate(corners: np.ndarray, image_aspect: float) -> None:
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("4 sonlu köşe noktası (x, y) gerekli")
    if not (math.isfinite(image_aspect) and image_aspect > 0):
        raise ValueError(f"image_aspect pozitif olmalı: {image_aspect!r}")
    p = corners * (image_aspect, 1.0)
    # y aşağı bakan uzayda sol-üst -> sağ-üst -> sağ-alt -> sol-alt sırası: tüm çarpımlar > 0.
    crosses = [_cross(p[i], p[(i + 1) % 4], p[(i + 2) % 4]) for i in range(4)]
    if all(c < 0 for c in crosses):
        raise ValueError("köşe sırası ters: sol-uzak, sağ-uzak, sağ-yakın, sol-yakın olmalı")
    if not all(c > _EPS for c in crosses):
        raise ValueError("köşeler dışbükey bir dörtgen oluşturmuyor (sıra karışık veya çakışık)")
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < MIN_AREA_FRACTION * image_aspect:
        raise ValueError(f"alan çok küçük: görüntünün %{100 * area / image_aspect:.2f}'i")


@dataclass(frozen=True)
class SurfaceCalibration:
    """Görüntü (0–1, kullanıcı bakışı) → aktif alan (0–1) dönüşümü.

    `corners_img`: sol-uzak, sağ-uzak, sağ-yakın, sol-yakın. `aspect`: alanın genişlik /
    yüksekliği (yalnız bilgi; dönüşüm birim kareye gider). `image_aspect`: kamera görüntüsünün
    genişlik / yüksekliği; homografi x'i buna göre ölçeklenmiş piksel benzeri uzayda kurulur.
    (Homografi eksen ölçeklemesine göre değişmez, sonuç aynıdır; ölçek sayısal kararlılık ve
    dışbükeylik/alan denetimi için kullanılır.)

    Bozuk dörtgende (ters sıra, dışbükey değil, alan çok küçük) `ValueError` verir.
    """

    corners_img: tuple[tuple[float, float], ...]
    aspect: float = DEFAULT_ASPECT
    image_aspect: float = 1.0
    _h: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_img, dtype=float)
        if corners.shape != (4, 2):
            raise ValueError("4 köşe noktası (x, y) gerekli")
        if not (math.isfinite(self.aspect) and self.aspect > 0):
            raise ValueError(f"aspect pozitif olmalı: {self.aspect!r}")
        image_aspect = float(self.image_aspect)
        _validate(corners, image_aspect)
        object.__setattr__(self, "corners_img",
                           tuple((float(x), float(y)) for x, y in corners))
        object.__setattr__(self, "aspect", float(self.aspect))
        object.__setattr__(self, "image_aspect", image_aspect)
        src = (corners * (image_aspect, 1.0)).astype(np.float32)
        h = cv2.getPerspectiveTransform(src, _UNIT_SQUARE)
        object.__setattr__(self, "_h", np.asarray(h, dtype=float))

    def to_surface(self, img_xy: Sequence[float]) -> tuple[float, float]:
        """Görüntü noktası (0–1) → alan (u, v). Alan içi 0–1; dışı <0 veya >1.

        x, homografinin kurulduğu `image_aspect` ile ölçeklenir (0–1 koordinatlar çözünürlükten
        bağımsız olduğundan kamera çözünürlüğü değişse de geçerlidir; görüş alanı/kırpma
        değişirse yeniden kalibre edin). Sonuç [`OUTSIDE_MIN`, `OUTSIDE_MAX`] aralığına
        kırpılır; ufuk çizgisinin ötesindeki nokta `(OUTSIDE_MIN, OUTSIDE_MIN)` olur.
        """
        x, y = float(img_xy[0]) * self.image_aspect, float(img_xy[1])
        h = self._h
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        if not (math.isfinite(w) and w > _EPS):
            return OUTSIDE_MIN, OUTSIDE_MIN
        u = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
        v = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
        if not (math.isfinite(u) and math.isfinite(v)):
            return OUTSIDE_MIN, OUTSIDE_MIN
        return (min(OUTSIDE_MAX, max(OUTSIDE_MIN, u)),
                min(OUTSIDE_MAX, max(OUTSIDE_MIN, v)))

    def to_image(self, uv: Sequence[float]) -> tuple[float, float]:
        """Alan (u, v) → görüntü noktası (0–1); çizim ve testler için."""
        inv = np.linalg.inv(self._h)
        p = inv @ np.array([float(uv[0]), float(uv[1]), 1.0])
        return float(p[0] / p[2]) / self.image_aspect, float(p[1] / p[2])

    @staticmethod
    def contains(uv: Sequence[float] | None, margin: float = 0.0) -> bool:
        """(u, v) alan içinde mi (kenardan `margin` kadar dışarısı dahil)?"""
        if uv is None:
            return False
        lo, hi = -margin, 1.0 + margin
        return lo <= uv[0] <= hi and lo <= uv[1] <= hi

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "corners_img": [list(c) for c in self.corners_img],
            "aspect": self.aspect,
            "image_aspect": self.image_aspect,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SurfaceCalibration:
        try:
            corners = tuple((float(x), float(y)) for x, y in data["corners_img"])
            return cls(corners, float(data.get("aspect", DEFAULT_ASPECT)),
                       float(data.get("image_aspect", 1.0)))
        except (KeyError, TypeError) as e:
            raise ValueError(f"bozuk yüzey kalibrasyonu: {e}") from e


def load(path: str | Path = DEFAULT_PATH) -> SurfaceCalibration | None:
    """Dosyadan okur; dosya yoksa None, bozuksa `ValueError`."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"bozuk yüzey kalibrasyonu dosyası: {p}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"bozuk yüzey kalibrasyonu dosyası: {p}")
    return SurfaceCalibration.from_dict(data)


def save(calib: SurfaceCalibration, path: str | Path = DEFAULT_PATH) -> None:
    """JSON olarak yazar (klasör yoksa açılır)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(p)
