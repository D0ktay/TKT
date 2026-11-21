"""El karelerinden girdi olaylarına geçişi yöneten jest motoru."""

from collections.abc import Mapping
from enum import Enum
from typing import Any

from tkt.control.os_input.base import InputBackend
from tkt.core.types import HandFrame, SourceInfo


class Mode(Enum):
    """Jest motorunun çalışma modları."""

    NONE = 0
    WAIT = 1
    POINTER = 2
    SCROLL = 3
    SWIPE = 4


class GestureEngine:
    """El durumunu izler ve backend'e girdi olayları gönderir."""

    mode: Mode
    left_down: bool

    def __init__(self, cfg: Mapping[str, Any], backend: InputBackend, info: SourceInfo) -> None:
        """Control yapılandırmasını, backend'i ve kaynak bilgisini al."""
        raise NotImplementedError

    def process(self, frame: HandFrame) -> None:
        """Bir el karesini işle."""
        raise NotImplementedError

    def release(self) -> None:
        """Tuşları bırak ve jest durumunu sıfırla."""
        raise NotImplementedError
