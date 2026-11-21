"""Kayıt ve sentetik el karesi kaynakları."""

from collections.abc import Iterator
from pathlib import Path
from typing import Self

from tkt.core.types import HandFrame, SourceInfo


class ReplaySource:
    """JSONL kaydından kareler üretir."""

    info: SourceInfo

    def __init__(self, path: str | Path, fast: bool = False, loop: bool = False) -> None:
        """Kaydı ve oynatma seçeneklerini al."""
        raise NotImplementedError

    def frames(self) -> Iterator[HandFrame]:
        """Kayıttaki kareleri sırayla ver."""
        raise NotImplementedError

    def close(self) -> None:
        """Kayıt kaynağını kapat."""
        raise NotImplementedError


class SyntheticSource:
    """Zincirlenebilen jest adımlarından sentetik kareler üretir."""

    info: SourceInfo

    def __init__(
        self,
        fps: float = 60.0,
        noise_px: float = 0.0,
        drop_rate: float = 0.0,
        seed: int = 0,
        realtime: bool = False,
        with_lift: bool = True,
    ) -> None:
        """Sentetik akışın hızını ve ölçüm seçeneklerini al."""
        raise NotImplementedError

    def rest(self, s: float) -> Self:
        """Dinlenme duruşu ekle."""
        raise NotImplementedError

    def open(self, n: int, s: float = 0.2) -> Self:
        """Belirtilen sayıda parmağı aç."""
        raise NotImplementedError

    def move(self, dx: float, dy: float, s: float) -> Self:
        """Eli normalize görüntü koordinatlarında kaydır."""
        raise NotImplementedError

    def thumb(self, closed: bool, s: float = 0.1) -> Self:
        """Başparmağı kapat veya aç."""
        raise NotImplementedError

    def tap(self, fingers: tuple[str, ...] = ("index",), ms: float = 150) -> Self:
        """Seçili parmaklarla bir tap ekle."""
        raise NotImplementedError

    def curl_all(self, s: float = 0.2) -> Self:
        """Tüm parmakları kıvır."""
        raise NotImplementedError

    def vanish(self, s: float) -> Self:
        """Elin görünmediği bir aralık ekle."""
        raise NotImplementedError

    def frames(self) -> Iterator[HandFrame]:
        """Betikteki adımlardan kareler ver."""
        raise NotImplementedError

    def close(self) -> None:
        """Sentetik kaynağı kapat."""
        raise NotImplementedError
