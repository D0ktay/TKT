"""Kişisel el kalibrasyonu (V2). PART2 2.2, 2.4.

`curl`, `thumb_out` ve `lift` ham değerlerini 0–1 aralığına oturtan referanslar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tkt.vision.tracker import RawHand


@dataclass(frozen=True)
class HandCalibration:
    open_ref: dict[str, float]         # parmak adı → curl_raw (açık el)
    fist_ref: dict[str, float]         # parmak adı → curl_raw (yumruk)
    thumb_closed: float
    thumb_open: float
    tap_ref: dict[str, float] | None   # parmak adı → tap tepe değeri (Faz 2)
    angle_sign: float = 1.0            # kalkış yönü, tap kalibrasyonundan


# Kalibrasyon yokken kullanılır. Kaba değerler; V2-1 kesinleştirir.
DEFAULT: HandCalibration = HandCalibration(
    open_ref={"index": 0.2, "middle": 0.2, "ring": 0.2, "pinky": 0.2},
    fist_ref={"index": 3.0, "middle": 3.0, "ring": 3.0, "pinky": 3.0},
    thumb_closed=0.5,
    thumb_open=1.2,
    tap_ref=None,
)


def load(path: Path) -> HandCalibration | None:
    """Dosyadan okur; dosya yoksa None."""
    raise NotImplementedError


def save(calib: HandCalibration, path: Path) -> None:
    raise NotImplementedError


def fit_hand(open_s: list[RawHand], fist_s: list[RawHand],
             thumb_closed_s: list[RawHand]) -> HandCalibration:
    """Açık el, yumruk ve kapalı başparmak örneklerinin medyanından kalibrasyon."""
    raise NotImplementedError


def fit_taps(calib: HandCalibration, samples: list[tuple[float, RawHand]]) -> HandCalibration:
    """Tap örneklerinden `tap_ref` ve `angle_sign` (Faz 2)."""
    raise NotImplementedError
