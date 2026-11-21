"""Parmak konumundan kaydırma hareketi."""

from collections.abc import Mapping
from typing import Any

from tkt.core.types import SourceInfo


class ScrollMotion:
    """Kaydırma hareketini ve ataletini hesaplar."""

    def __init__(self, cfg: Mapping[str, Any], info: SourceInfo) -> None:
        """Scroll ayarlarını ve kaynak bilgisini al."""
        raise NotImplementedError

    def update(self, pos: tuple[float, float], hand_size: float, t: float) -> tuple[float, float]:
        """Yeni konumdan piksel cinsinden kaydırma hesapla."""
        raise NotImplementedError

    def release(self, t: float) -> None:
        """Parmaklar kalkınca ataletli kaydırmayı başlat."""
        raise NotImplementedError

    def tick(self, t: float) -> tuple[float, float]:
        """Ataletin bu karede ürettiği kaydırmayı hesapla."""
        raise NotImplementedError

    def stop(self) -> None:
        """Ataletli kaydırmayı durdur."""
        raise NotImplementedError

    def rebase(self) -> None:
        """Sonraki kareyi kaydırma üretmeden referans al."""
        raise NotImplementedError
