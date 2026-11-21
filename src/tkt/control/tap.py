"""Kalkıp yüzeye inen parmaklardan tap algılama."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot
from typing import Any

from tkt.core.types import HandFrame, SourceInfo


@dataclass(frozen=True)
class TapUpdate:
    """Kalkış başlangıcı, meşguliyet ve oluşan tap türü."""

    started: bool
    busy: bool
    kind: str | None


@dataclass
class _FingerTap:
    started_at: float
    reached_up: bool = False
    landed_at: float | None = None


class TapDetector:
    """İzlenen parmaklarda tekli ve ikili tap algılar."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        """Control yapılandırmasının tamamını al."""
        tap = cfg["tap"]
        self._lift_start = tap["lift_start"]
        self._lift_up = tap["lift_up"]
        self._min_s = tap["min_ms"] / 1000
        self._max_s = tap["max_ms"] / 1000
        self._pair_s = tap["pair_ms"] / 1000
        self._max_palm_move = tap["max_palm_move"]
        self._cooldown_s = tap["cooldown_ms"] / 1000
        self._curl_open = cfg["mode"]["curl_open"]
        # Kaynak bilgisi verilmezse normalize koordinatların önceki ölçeğini koru.
        self._width = self._height = 1
        self.reset()

    def set_info(self, info: SourceInfo) -> None:
        """Avuç hareketini ölçmek için kaynak görüntüsünün boyutlarını al."""
        self._width, self._height = info.width, info.height

    def cancel_until_landing(self, watched: tuple[str, ...]) -> None:
        """Tap girişimini iptal et ve izlenen parmaklar inene kadar bekle."""
        self._clear_attempt()
        self._watched = watched
        self._await_landing = True

    def update(self, frame: HandFrame, watched: tuple[str, ...]) -> TapUpdate:
        """İzlenen parmaklar için tap durumunu güncelle."""
        if not frame.present or not watched:
            return TapUpdate(False, bool(self._fingers), None)
        if self._watched != watched:
            self._clear_attempt()
            self._watched = watched
            self._await_landing = False

        lifts: dict[str, float] = {}
        for name in watched:
            lift = frame.fingers[name].lift
            if lift is None:
                self.cancel_until_landing(watched)
                return TapUpdate(False, False, None)
            lifts[name] = lift

        lifting = any(lift > self._lift_start for lift in lifts.values())
        if frame.t < self._cooldown_until or self._await_landing:
            # Kesintiye uğrayan veya cooldown sırasında başlayan kalkışı devralma.
            if lifting:
                self._await_landing = True
            elif all(lift < self._lift_start for lift in lifts.values()):
                self._await_landing = False
            return TapUpdate(False, False, None)

        started = not self._fingers and lifting
        if started:
            self._palm_origin = frame.palm
            self._palm_limit = self._max_palm_move * (frame.hand_size * self._width)
        elif not self._fingers:
            return TapUpdate(False, False, None)

        dx = (frame.palm[0] - self._palm_origin[0]) * self._width
        dy = (frame.palm[1] - self._palm_origin[1]) * self._height
        if hypot(dx, dy) >= self._palm_limit:
            self._cancelled = True
        for name, lift in lifts.items():
            self._update_finger(name, lift, frame.fingers[name].curl, frame.t)

        kind = None
        starts = [tap.started_at for tap in self._fingers.values()]
        if len(starts) < len(watched):
            if frame.t > min(starts) + self._pair_s:
                self._cancelled = True
        elif max(starts) > min(starts) + self._pair_s:
            self._cancelled = True

        landings = [
            tap.landed_at for tap in self._fingers.values() if tap.landed_at is not None
        ]
        if len(landings) < len(starts):
            return TapUpdate(started, True, None)
        if not self._cancelled:
            if len(starts) < len(watched):
                return TapUpdate(started, True, None)
            if max(landings) <= min(landings) + self._pair_s:
                kind = "tap1" if len(watched) == 1 else "tap2"
                self._cooldown_until = frame.t + self._cooldown_s
        self._clear_attempt()
        return TapUpdate(started, False, kind)

    def reset(self) -> None:
        """Tap durumunu sıfırla."""
        self._clear_attempt()
        self._watched: tuple[str, ...] = ()
        self._cooldown_until = float("-inf")
        self._await_landing = False

    def _clear_attempt(self) -> None:
        self._fingers: dict[str, _FingerTap] = {}
        self._palm_origin = (0.0, 0.0)
        self._palm_limit = 0.0
        self._cancelled = False

    def _update_finger(self, name: str, lift: float, curl: float, t: float) -> None:
        tap = self._fingers.get(name)
        if tap is None:
            if lift <= self._lift_start:
                return
            tap = self._fingers[name] = _FingerTap(t)
        elif tap.landed_at is not None:
            if lift <= self._lift_start:
                return
            # Diğer parmak beklenirken aynı parmağın ikinci kalkışı eşleşemez.
            self._cancelled = True
            tap = self._fingers[name] = _FingerTap(t)

        if t > tap.started_at + self._max_s or curl >= self._curl_open:
            self._cancelled = True
        if lift > self._lift_up:
            tap.reached_up = True
        if lift < self._lift_start:
            tap.landed_at = t
            if not tap.reached_up or t < tap.started_at + self._min_s:
                self._cancelled = True
