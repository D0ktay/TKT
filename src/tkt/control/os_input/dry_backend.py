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

    def __init__(self) -> None:
        self.events = []
        self._t = 0.0
        self._pressed: dict[str, None] = {}

    @property
    def events_sent(self) -> int:
        """Kaydedilen olay sayısı."""
        return len(self.events)

    def set_time(self, t: float) -> None:
        """Sonraki olaylar için karenin zamanını ayarla."""
        self._t = t

    def move_rel(self, dx: int, dy: int) -> None:
        """Göreli hareket olayını kaydet."""
        self.events.append(Event(self._t, "move", (dx, dy)))

    def button(self, name: str, down: bool) -> None:
        """Tuş olayını kaydet."""
        self.events.append(Event(self._t, "button", (name, down)))
        if down:
            self._pressed[name] = None
        else:
            self._pressed.pop(name, None)

    def scroll(self, dx: float, dy: float) -> None:
        """Kaydırma olayını kaydet."""
        self.events.append(Event(self._t, "scroll", (dx, dy)))

    def system_action(self, name: str) -> None:
        """Sistem eylemini kaydet."""
        self.events.append(Event(self._t, "system", (name,)))

    def release_all(self) -> None:
        """Basılı tüm tuşlar için bırakma olayı kaydet."""
        for name in tuple(self._pressed):
            self.button(name, False)
