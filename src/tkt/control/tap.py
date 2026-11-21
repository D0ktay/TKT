"""Kalkıp yüzeye inen parmaklardan tap algılama."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tkt.core.types import HandFrame


@dataclass(frozen=True)
class TapUpdate:
    """Kalkış başlangıcı, meşguliyet ve oluşan tap türü."""

    started: bool
    busy: bool
    kind: str | None


class TapDetector:
    """İzlenen parmaklarda tekli ve ikili tap algılar."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        """Control yapılandırmasının tamamını al."""
        raise NotImplementedError

    def update(self, frame: HandFrame, watched: tuple[str, ...]) -> TapUpdate:
        """İzlenen parmaklar için tap durumunu güncelle."""
        raise NotImplementedError

    def reset(self) -> None:
        """Tap durumunu sıfırla."""
        raise NotImplementedError
