"""El konumundan göreli imleç hareketi."""

from collections import deque
from collections.abc import Mapping
from math import hypot, isfinite
from statistics import median
from typing import Any

from tkt.control.filters import OneEuro
from tkt.core.types import SourceInfo

HAND_PX_WINDOW_S = 2.0
HISTORY_MS = 150.0


class PointerMotion:
    """Parmak konumunu göreli ekran pikseline çevirir."""

    def __init__(self, cfg: Mapping[str, Any], info: SourceInfo) -> None:
        """Pointer ayarlarını ve kaynak bilgisini al."""
        self._width = info.width
        self._height = info.height
        filter_options = {"d_cutoff": cfg["d_cutoff"]} if "d_cutoff" in cfg else {}
        self._filter = OneEuro(cfg["min_cutoff"], cfg["beta"], **filter_options)
        self._gain_min = cfg["gain_min"]
        self._gain_range = cfg["gain_max"] - self._gain_min
        self._speed_lo = cfg["speed_lo"]
        self._speed_range = cfg["speed_hi"] - self._speed_lo
        self._gap_reset_s = cfg["gap_reset_ms"] / 1000.0
        self._history_s = max(HISTORY_MS, cfg["rollback_ms"]) / 1000.0
        self._hand_samples: deque[tuple[float, float]] = deque()
        self._history: deque[tuple[float, int, int]] = deque()
        self.reset()

    def update(self, pos: tuple[float, float], hand_size: float, t: float) -> tuple[int, int]:
        """Yeni konumdan imleç hareketini hesapla."""
        previous_t = self._t
        if previous_t is not None:
            if t < previous_t or (t == previous_t and self._pos is not None):
                return 0, 0
            if t > previous_t + self._gap_reset_s:
                self.rebase()

        self._t = t
        while self._history and self._history[0][0] + self._history_s < t:
            self._history.popleft()
        while self._hand_samples and self._hand_samples[0][0] + HAND_PX_WINDOW_S < t:
            self._hand_samples.popleft()
        hand_px = hand_size * self._width
        if hand_px > 0 and isfinite(hand_px):
            self._hand_samples.append((t, hand_px))

        previous_pos = self._pos
        x, y = self._filter((pos[0] * self._width, pos[1] * self._height), t)
        self._pos = x, y
        if previous_t is None or previous_pos is None or not self._hand_samples:
            return 0, 0

        hand_px = median(size for _, size in self._hand_samples)
        dx, dy = (x - previous_pos[0]) / hand_px, (y - previous_pos[1]) / hand_px
        speed = hypot(dx, dy) / (t - previous_t)
        fraction = min(1.0, max(0.0, (speed - self._speed_lo) / self._speed_range))
        gain = self._gain_min + self._gain_range * fraction**2 * (3.0 - 2.0 * fraction)

        out_x = dx * gain + self._remainder[0]
        out_y = dy * gain + self._remainder[1]
        send_x, send_y = round(out_x), round(out_y)
        self._remainder = out_x - send_x, out_y - send_y
        if send_x or send_y:
            self._history.append((t, send_x, send_y))
        return send_x, send_y

    def rebase(self) -> None:
        """Sonraki kareyi hareket üretmeden referans al."""
        self._filter.reset()
        self._pos: tuple[float, float] | None = None
        self._remainder = (0.0, 0.0)

    def rollback(self, ms: float) -> tuple[int, int]:
        """Son kare zamanından geriye doğru verilen aralıktaki hareketi geri al."""
        if self._t is None:
            return 0, 0

        window_s = ms / 1000.0
        undo_x = undo_y = 0
        while self._history and self._history[-1][0] + window_s >= self._t:
            _, dx, dy = self._history.pop()
            undo_x -= dx
            undo_y -= dy
        return undo_x, undo_y

    def reset(self) -> None:
        """Filtreyi ve hareket geçmişini sıfırla."""
        self.rebase()
        self._t: float | None = None
        self._hand_samples.clear()
        self._history.clear()
