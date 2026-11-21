"""Hareket için iki boyutlu One Euro filtresi."""

from math import hypot, tau


def _alpha(cutoff: float, dt: float) -> float:
    scaled = tau * cutoff * dt
    return scaled / (1.0 + scaled)


class OneEuro:
    """Yumuşatmayı hareket hızına göre ayarlar."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        """Filtrenin kesim frekanslarını ve hız katsayısını al."""
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self.reset()

    def __call__(self, x: tuple[float, float], t: float) -> tuple[float, float]:
        """Verilen zamandaki konumu filtrele."""
        if self._t is None:
            self._t = t
            self._raw = self._filtered = x
            return x

        dt = t - self._t
        if dt <= 0.0:
            return self._filtered

        dx = (x[0] - self._raw[0]) / dt
        dy = (x[1] - self._raw[1]) / dt
        derivative_alpha = _alpha(self._d_cutoff, dt)
        vx, vy = self._velocity
        vx += derivative_alpha * (dx - vx)
        vy += derivative_alpha * (dy - vy)
        cutoff = self._min_cutoff + self._beta * hypot(vx, vy)
        alpha = _alpha(cutoff, dt)
        px, py = self._filtered
        self._filtered = (px + alpha * (x[0] - px), py + alpha * (x[1] - py))
        self._velocity = vx, vy
        self._raw = x
        self._t = t
        return self._filtered

    def reset(self) -> None:
        """Filtre geçmişini temizle."""
        self._t: float | None = None
        self._raw = (0.0, 0.0)
        self._filtered = (0.0, 0.0)
        self._velocity = (0.0, 0.0)
