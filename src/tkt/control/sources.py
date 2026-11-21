"""Kayıt ve sentetik el karesi kaynakları."""

from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from random import Random
from threading import Event
from time import monotonic
from typing import Self

from tkt.core.io import iter_frames, read_header
from tkt.core.types import FINGERS, SCHEMA_VERSION, Finger, HandFrame, SourceInfo


class ReplaySource:
    """JSONL kaydından kareler üretir."""

    info: SourceInfo

    def __init__(self, path: str | Path, fast: bool = False, loop: bool = False) -> None:
        self.info = read_header(path)
        self._path = path
        self._fast = fast
        self._loop = loop
        self._closed = Event()

    def frames(self) -> Iterator[HandFrame]:
        started = monotonic()
        first_t = None
        time_offset = 0.0
        id_offset = 0
        while not self._closed.is_set():
            first = last = None
            with closing(iter_frames(self._path)) as frames:
                for frame in frames:
                    if self._closed.is_set():
                        return
                    if first is None:
                        first = frame
                    if first_t is None:
                        first_t = frame.t
                    t = frame.t + time_offset
                    if not self._fast and self._closed.wait(started + (t - first_t) - monotonic()):
                        return
                    last = frame
                    yield replace(frame, t=t, frame_id=frame.frame_id + id_offset)
            if not self._loop or first is None or last is None:
                return
            time_offset += last.t - first.t + 1 / self.info.fps
            id_offset += last.frame_id - first.frame_id + 1

    def close(self) -> None:
        self._closed.set()


@dataclass(frozen=True)
class _Step:
    count: int
    open_count: int
    thumb_out: float
    move: tuple[float, float] = (0.0, 0.0)
    tap: tuple[str, ...] = ()
    present: bool = True


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


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
        if not isfinite(fps) or fps <= 0:
            raise ValueError("fps pozitif ve sonlu olmalı")
        if not isfinite(noise_px) or noise_px < 0 or not 0 <= drop_rate <= 1:
            raise ValueError("noise_px negatif olamaz ve drop_rate 0–1 arasında olmalı")
        self.info = SourceInfo(SCHEMA_VERSION, 1280, 720, fps, "right")
        self._noise_px = noise_px
        self._drop_rate = drop_rate
        self._seed = seed
        self._realtime = realtime
        self._with_lift = with_lift
        self._closed = Event()
        self._steps: list[_Step] = []
        self._open_count = 0
        self._thumb_out = 1.0

    def _append(self, s: float, **changes) -> Self:
        if not isfinite(s) or s < 0:
            raise ValueError("Adım süresi negatif veya sonsuz olamaz")
        count = max(1, round(s * self.info.fps)) if s else 0
        step = _Step(count, self._open_count, self._thumb_out)
        self._steps.append(replace(step, **changes))
        return self

    def rest(self, s: float) -> Self:
        self._open_count = 0
        self._thumb_out = 1.0
        return self._append(s)

    def open(self, n: int, s: float = 0.2) -> Self:
        if not 0 <= n <= len(FINGERS):
            raise ValueError("Açık parmak sayısı 0–4 arasında olmalı")
        self._open_count = n
        return self._append(s)

    def move(self, dx: float, dy: float, s: float) -> Self:
        return self._append(s, move=(dx, dy))

    def thumb(self, closed: bool, s: float = 0.1) -> Self:
        self._thumb_out = float(not closed)
        return self._append(s)

    def tap(self, fingers: tuple[str, ...] = ("index",), ms: float = 150) -> Self:
        if any(name not in FINGERS for name in fingers):
            raise ValueError("Bilinmeyen tap parmağı")
        return self._append(ms / 1000, tap=fingers)

    def curl_all(self, s: float = 0.2) -> Self:
        return self.open(0, s)

    def vanish(self, s: float) -> Self:
        return self._append(s, present=False)

    def frames(self) -> Iterator[HandFrame]:
        rng = Random(self._seed)
        started = monotonic()
        frame_id = 0
        x, y = 0.5, 0.6
        for step in self._steps:
            dx, dy = step.move
            for i in range(1, step.count + 1):
                offset = frame_id / self.info.fps
                if self._closed.is_set():
                    return
                if self._realtime and self._closed.wait(started + offset - monotonic()):
                    return
                if rng.random() >= self._drop_rate:
                    progress = i / step.count
                    palm = (_unit(x + dx * progress), _unit(y + dy * progress))
                    t = started + offset if self._realtime else offset
                    yield self._frame(step, t, frame_id, palm, progress, rng)
                frame_id += 1
            x, y = _unit(x + dx), _unit(y + dy)

    def _frame(
        self,
        step: _Step,
        t: float,
        frame_id: int,
        palm: tuple[float, float],
        progress: float,
        rng: Random,
    ) -> HandFrame:
        if not step.present:
            return HandFrame(t, frame_id, False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)

        fingers = {}
        for i, name in enumerate(FINGERS):
            tapped = name in step.tap
            curl = float(i >= step.open_count and not tapped)
            lift = 1.0 - abs(2 * progress - 1.0) if tapped else 0.0
            img = (
                _unit(palm[0] + (i - 1.5) * 0.03 + rng.gauss(0, self._noise_px) / self.info.width),
                _unit(
                    palm[1] - 0.15 + 0.08 * curl + rng.gauss(0, self._noise_px) / self.info.height
                ),
            )
            fingers[name] = Finger(img, None, curl, lift if self._with_lift else None)
        return HandFrame(t, frame_id, True, fingers, step.thumb_out, palm, 0.2, 1.0)

    def close(self) -> None:
        self._closed.set()
