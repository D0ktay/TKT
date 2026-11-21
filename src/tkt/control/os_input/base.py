"""İşletim sistemi girdisi protokolü."""

from typing import Protocol


class InputBackend(Protocol):
    """Hareket, tuş, kaydırma ve sistem eylemlerinin ortak arayüzü."""

    def move_rel(self, dx: int, dy: int) -> None:
        """İmleci piksel cinsinden göreli hareket ettir."""
        raise NotImplementedError

    def button(self, name: str, down: bool) -> None:
        """Sol veya sağ tuşu bas ya da bırak."""
        raise NotImplementedError

    def scroll(self, dx: float, dy: float) -> None:
        """Piksel cinsinden kaydır."""
        raise NotImplementedError

    def system_action(self, name: str) -> None:
        """Görev görünümünü aç veya masaüstleri arasında geç."""
        raise NotImplementedError

    def release_all(self) -> None:
        """Basılı tüm tuşları bırak."""
        raise NotImplementedError
