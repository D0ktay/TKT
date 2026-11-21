"""MediaPipe el takibi sarmalayıcısı (V1). PART2 2.2, 2.4.

HandLandmarker, VIDEO modu, num_hands=1. Sağ/sol etiketi MediaPipe'tan alınmaz.

CLI: python -m tkt.vision.tracker --download   (modeli models/ altına indirir)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawHand:
    image_pts: np.ndarray    # (21, 2), 0–1, kullanıcı bakışı
    world_pts: np.ndarray    # (21, 3), metre
    score: float


class HandTracker:
    """Tek el için MediaPipe HandLandmarker."""

    def __init__(self, model_path: str, min_detection: float, min_tracking: float) -> None:
        raise NotImplementedError

    def process(self, image: np.ndarray, t: float) -> RawHand | None:
        """BGR kareyi işler; el yoksa None. `t` saniye, kesin artan olmalı."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.tracker")
    parser.add_argument("--download", action="store_true", help="modeli models/ altına indirir")
    parser.add_argument("--config", default="config/vision.toml")
    parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
