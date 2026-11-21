"""Parmak konumundan kaydırma hareketi."""

from collections import deque
from collections.abc import Mapping
from math import exp, hypot, isfinite
from statistics import median
from typing import Any

from tkt.control.filters import OneEuro
from tkt.core.types import SourceInfo

# [scroll] ortak filtrenin [pointer] varsayılanlarını kullanır.
_MIN_CUTOFF = 1.0
_BETA = 0.02
_D_CUTOFF = 1.0
_GAP_RESET_MS = 100
_HAND_SIZE_WINDOW_S = 2.0
_VELOCITY_SAMPLES = 3
_ZERO = (0.0, 0.0)


class ScrollMotion:
    """Kaydırma hareketini ve ataletini hesaplar."""

    def __init__(self, cfg: Mapping[str, Any], info: SourceInfo) -> None:
        """Scroll ayarlarını ve kaynak bilgisini al."""
        self._width = info.width
        self._height = info.height
        self._gain = cfg["gain"] if cfg["natural"] else -cfg["gain"]
        self._axis_lock = cfg["axis_lock"]
        self._gap_s = cfg.get("gap_reset_ms", _GAP_RESET_MS) / 1000
        self._inertia_s = cfg["inertia_ms"] / 1000
        self._min_speed = cfg["inertia_min_speed"]
        self._filter = OneEuro(_MIN_CUTOFF, _BETA, _D_CUTOFF)

        self._hand_sizes: deque[tuple[float, float]] = deque()
        self._velocities: deque[tuple[float, float]] = deque(maxlen=_VELOCITY_SAMPLES)
        self.rebase()

    def update(self, pos: tuple[float, float], hand_size: float, t: float) -> tuple[float, float]:
        """Yeni konumdan piksel cinsinden kaydırma hesapla."""
        if self._t is not None:
            if t <= self._t:
                return _ZERO
            if t > self._t + self._gap_s:
                self.rebase()
        self.stop()

        previous_pos, previous_t = self._pos, self._t
        x, y = self._filter((pos[0] * self._width, pos[1] * self._height), t)
        self._pos, self._t = (x, y), t

        while self._hand_sizes and self._hand_sizes[0][0] + _HAND_SIZE_WINDOW_S < t:
            self._hand_sizes.popleft()
        hand_px = hand_size * self._width
        if hand_px > 0 and isfinite(hand_px):
            self._hand_sizes.append((t, hand_px))
        if previous_pos is None or previous_t is None or not self._hand_sizes:
            self._velocities.clear()
            return _ZERO

        scale = self._gain / median(size for _, size in self._hand_sizes)
        dx, dy = (x - previous_pos[0]) * scale, (y - previous_pos[1]) * scale
        if abs(dx) < self._axis_lock * abs(dy):
            dx = 0.0
        elif abs(dy) < self._axis_lock * abs(dx):
            dy = 0.0

        dt = t - previous_t
        self._velocities.append((dx / dt, dy / dt))
        return dx, dy

    def release(self, t: float) -> None:
        """Parmaklar kalkınca ataletli kaydırmayı başlat."""
        if self._t is None or t < self._t:
            return
        recent = t <= self._t + self._gap_s
        vx = vy = 0.0
        if recent and self._velocities:
            vx = sum(dx for dx, _ in self._velocities) / len(self._velocities)
            vy = sum(dy for _, dy in self._velocities) / len(self._velocities)

        self.rebase()
        if self._inertia_s > 0 and hypot(vx, vy) > self._min_speed:
            self._coast_t = t
            self._coast_velocity = vx, vy

    def tick(self, t: float) -> tuple[float, float]:
        """Ataletin bu karede ürettiği kaydırmayı hesapla."""
        if self._coast_t is None or t <= self._coast_t:
            return _ZERO
        if t > self._coast_t + self._gap_s:
            self.stop()
            return _ZERO

        dt = t - self._coast_t
        decay = exp(-dt / self._inertia_s)
        vx, vy = self._coast_velocity
        vx, vy = vx * decay, vy * decay
        if hypot(vx, vy) <= self._min_speed:
            self.stop()
            return _ZERO

        self._coast_velocity = vx, vy
        self._coast_t = t
        return vx * dt, vy * dt

    def stop(self) -> None:
        """Ataletli kaydırmayı durdur."""
        self._coast_t: float | None = None
        self._coast_velocity = _ZERO

    def rebase(self) -> None:
        """Sonraki kareyi kaydırma üretmeden referans al."""
        self.stop()
        self._filter.reset()
        self._pos: tuple[float, float] | None = None
        self._t: float | None = None
        self._velocities.clear()
