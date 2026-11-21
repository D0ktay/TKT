"""Kamera yakalama (V1). PART2 2.2, 2.4.

Yakalama ayrı thread'de döner ve yalnızca en son kareyi tutar. Görüntü kullanıcının bakış
açısına göre 180° döndürülür.

CLI: python -m tkt.vision.camera --probe
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray        # BGR, 180° döndürülmüş
    t: float                 # time.monotonic(), kare alındığı an
    frame_id: int


class Camera:
    """Arka plan thread'iyle yakalama yapan kamera."""

    def __init__(self, index: int, width: int, height: int, fps: int,
                 rotate_180: bool = True, backend: str = "auto") -> None:
        raise NotImplementedError

    def start(self) -> None:
        """Yakalama thread'ini başlatır."""
        raise NotImplementedError

    def read(self, timeout: float = 1.0) -> CapturedFrame | None:
        """En son kareyi döndürür; `timeout` içinde yeni kare gelmezse None."""
        raise NotImplementedError

    def measured_fps(self) -> float:
        """Son birkaç saniyede sayılan gerçek yakalama hızı."""
        raise NotImplementedError

    def close(self) -> None:
        """Thread'i durdurur, aygıtı bırakır."""
        raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.camera")
    parser.add_argument("--probe", action="store_true", help="5 sn sayıp gerçek fps'i yazar")
    parser.add_argument("--config", default="config/vision.toml")
    parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
