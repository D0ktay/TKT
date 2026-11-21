"""Parmak kalkışı (lift) tahmini (V2, Faz 2). PART2 2.2, 2.5.

CLI: python -m tkt.vision.lift --eval TAP.raw.jsonl NORMAL.raw.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from tkt.vision.hand_calib import HandCalibration
from tkt.vision.tracker import RawHand


class LiftEstimator:
    """Kökten kalkış açısı ve görüntüdeki parmak boyundan `lift` üretir."""

    def __init__(self, calib: HandCalibration, cfg: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def update(self, hand: RawHand, t: float, curls: dict[str, float]) -> dict[str, float]:
        """Parmak adı → lift (0–1)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Taban çizgilerini sıfırlar (el kaybolunca)."""
        raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.lift")
    parser.add_argument("--eval", nargs=2, metavar=("TAP", "NORMAL"),
                        help="ham kayıtlarda ayrışma oranını yazar")
    parser.add_argument("--config", default="config/vision.toml")
    parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
