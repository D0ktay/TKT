"""Kare kaynağı protokolü (PART1, 1.3).

`LiveSource` (vision), `ReplaySource` ve `SyntheticSource` (control) bunu uygular.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from tkt.core.types import HandFrame, SourceInfo


@runtime_checkable
class HandFrameSource(Protocol):
    info: SourceInfo

    def frames(self) -> Iterator[HandFrame]: ...

    def close(self) -> None: ...
