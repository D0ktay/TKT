"""Girdi göndermeden olayları kaydeden backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """Kare zamanına bağlı hareket, tuş, kaydırma veya sistem olayı."""

    t: float
    kind: str
    args: tuple


class DryBackend:
    """Girdi olaylarını yalnızca bellekte tutar."""

    events: list[Event]

    def set_time(self, t: float) -> None:
        """Sonraki olaylar için karenin zamanını ayarla."""
        raise NotImplementedError

    def move_rel(self, dx: int, dy: int) -> None:
        """Göreli hareket olayını kaydet."""
        raise NotImplementedError

    def button(self, name: str, down: bool) -> None:
        """Tuş olayını kaydet."""
        raise NotImplementedError

    def scroll(self, dx: float, dy: float) -> None:
        """Kaydırma olayını kaydet."""
        raise NotImplementedError

    def system_action(self, name: str) -> None:
        """Sistem eylemini kaydet."""
        raise NotImplementedError

    def release_all(self) -> None:
        """Basılı tüm tuşlar için bırakma olayı kaydet."""
        raise NotImplementedError
