"""Kişisel el kalibrasyonu (V2). PART2 2.2, 2.4.

`curl`, `thumb_out` ve `lift` ham değerlerini 0–1 aralığına oturtan referanslar.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
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


# Kalibrasyon yokken kullanılır. Kaba değerlerdir; ilk kayıtlar bununla alınır, `fit_hand`
# kişiye göre yeniler.
# - curl_raw (orta + uç eklem bükülmesi, radyan): açık elde eklemler neredeyse düz (~0,2);
#   yumrukta orta eklem ~100°, uç eklem ~70° (~3,0).
# - thumb_raw (başparmak ucu–işaret kökü / bilek–orta kök, dünya noktaları): ele yapışık
#   başparmağın ucu işaret parmağının ilk boğumu yanındadır (~2–3 cm / ~9 cm ≈ 0,35);
#   serbest başparmakta uç kökten ~7 cm uzaktadır (≈ 0,8).
DEFAULT: HandCalibration = HandCalibration(
    open_ref={"index": 0.2, "middle": 0.2, "ring": 0.2, "pinky": 0.2},
    fist_ref={"index": 3.0, "middle": 3.0, "ring": 3.0, "pinky": 3.0},
    thumb_closed=0.35,
    thumb_open=0.8,
    tap_ref=None,
)


def _floats(d: object, key: str) -> dict[str, float]:
    if not isinstance(d, dict):
        raise ValueError(f"kalibrasyon: {key!r} sözlük olmalı")
    return {str(k): float(v) for k, v in d.items()}


def _from_dict(d: dict) -> HandCalibration:
    tap_ref = d.get("tap_ref")
    return HandCalibration(
        open_ref=_floats(d["open_ref"], "open_ref"),
        fist_ref=_floats(d["fist_ref"], "fist_ref"),
        thumb_closed=float(d["thumb_closed"]),
        thumb_open=float(d["thumb_open"]),
        tap_ref=None if tap_ref is None else _floats(tap_ref, "tap_ref"),
        angle_sign=float(d.get("angle_sign", 1.0)),
    )


def load(path: Path) -> HandCalibration | None:
    """Dosyadan okur; dosya yoksa None. `~` genişletilir.

    Dosya var ama bozuksa `ValueError` yükseltir.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("kök nesne sözlük olmalı")
        return _from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{p}: kalibrasyon okunamadı: {e}") from e


def save(calib: HandCalibration, path: Path) -> None:
    """JSON olarak yazar; dizini oluşturur, `~` genişletilir."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(calib), ensure_ascii=False, indent=2) + "\n"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def fit_hand(open_s: list[RawHand], fist_s: list[RawHand],
             thumb_closed_s: list[RawHand]) -> HandCalibration:
    """Açık el, yumruk ve kapalı başparmak örneklerinin medyanından kalibrasyon."""
    raise NotImplementedError


def fit_taps(calib: HandCalibration, samples: list[tuple[float, RawHand]]) -> HandCalibration:
    """Tap örneklerinden `tap_ref` ve `angle_sign` (Faz 2)."""
    raise NotImplementedError
