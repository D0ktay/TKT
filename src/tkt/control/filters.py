"""Hareket için iki boyutlu One Euro filtresi."""


class OneEuro:
    """Yumuşatmayı hareket hızına göre ayarlar."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        """Filtrenin kesim frekanslarını ve hız katsayısını al."""
        raise NotImplementedError

    def __call__(self, x: tuple[float, float], t: float) -> tuple[float, float]:
        """Verilen zamandaki konumu filtrele."""
        raise NotImplementedError

    def reset(self) -> None:
        """Filtre geçmişini temizle."""
        raise NotImplementedError
