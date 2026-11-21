"""Ham noktalardan ölçümler (V2). PART2 2.2, 2.4.

Eşik ve zaman filtresi uygulanmaz.
"""

from __future__ import annotations

from tkt.core.types import HandFrame, SourceInfo
from tkt.vision.hand_calib import HandCalibration
from tkt.vision.tracker import RawHand


def curl_raw(hand: RawHand, finger: str) -> float:
    """Orta ve uç eklem bükülmesinin toplamı (radyan, dünya noktalarıyla)."""
    raise NotImplementedError


def thumb_raw(hand: RawHand) -> float:
    """Başparmak ucu ile işaret kökü arası 3D mesafe, el boyuna oranlı."""
    raise NotImplementedError


def build_frame(hand: RawHand | None, t: float, frame_id: int, calib: HandCalibration,
                info: SourceInfo, lift: dict[str, float] | None = None,
                prev_hand_size: float | None = None) -> HandFrame:
    """Tek karenin `HandFrame`'ini üretir; el yoksa `present=False`."""
    raise NotImplementedError
