"""Canlı kare kaynağı (V1). PART2 2.2, 2.4.

`open_live_source`, `app/main.py`'nin vision'dan çağırdığı tek fonksiyondur.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from tkt.core.types import HandFrame, SourceInfo
from tkt.vision.camera import Camera
from tkt.vision.hand_calib import HandCalibration
from tkt.vision.lift import LiftEstimator
from tkt.vision.tracker import HandTracker, RawHand


class LiveSource:
    """HandFrameSource: kamera + tracker + ölçümler. El yokken de kare üretir."""

    info: SourceInfo

    def __init__(self, camera: Camera, tracker: HandTracker, calib: HandCalibration,
                 hand: str, lift: LiftEstimator | None = None) -> None:
        raise NotImplementedError

    def frames(self) -> Iterator[HandFrame]:
        raise NotImplementedError

    def frames_raw(self) -> Iterator[tuple[HandFrame, RawHand | None, np.ndarray]]:
        """Debug penceresi ve `record --raw` için: kare, ham noktalar, görüntü."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def load_vision_config(path: str | Path) -> dict:
    raise NotImplementedError


def open_live_source(config_path: str | Path = "config/vision.toml") -> LiveSource:
    """Config'e göre kamera, tracker, kalibrasyon ve (varsa) lift ile LiveSource kurar."""
    raise NotImplementedError
