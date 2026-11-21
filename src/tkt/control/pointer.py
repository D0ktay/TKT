"""El konumundan göreli imleç hareketi."""

from collections.abc import Mapping
from typing import Any

from tkt.core.types import SourceInfo


class PointerMotion:
    """Parmak konumunu göreli ekran pikseline çevirir."""

    def __init__(self, cfg: Mapping[str, Any], info: SourceInfo) -> None:
        """Pointer ayarlarını ve kaynak bilgisini al."""
        raise NotImplementedError

    def update(self, pos: tuple[float, float], hand_size: float, t: float) -> tuple[int, int]:
        """Yeni konumdan imleç hareketini hesapla."""
        raise NotImplementedError

    def rebase(self) -> None:
        """Sonraki kareyi hareket üretmeden referans al."""
        raise NotImplementedError

    def rollback(self, ms: float) -> tuple[int, int]:
        """Son verilen süre içinde gönderilen hareketin tersini döndür."""
        raise NotImplementedError

    def reset(self) -> None:
        """Filtreyi ve hareket geçmişini sıfırla."""
        raise NotImplementedError
