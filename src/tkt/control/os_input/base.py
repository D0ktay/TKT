"""İşletim sistemi girdisi protokolü."""

from typing import Protocol


class InputBackend(Protocol):
    """Hareket, tuş, kaydırma ve sistem eylemlerinin ortak arayüzü.

    İsteğe bağlı events_sent sayacı, gönderilen veya dry backend için
    kaydedilen olaylarda artar. Gecikme ölçümü bu sayacı kullanabilir.
    """

    def move_rel(self, dx: int, dy: int) -> None:
        """İmleci piksel cinsinden göreli hareket ettir."""
        raise NotImplementedError

    def button(self, name: str, down: bool) -> None:
        """Sol veya sağ tuşu bas ya da bırak."""
        raise NotImplementedError

    def scroll(self, dx: float, dy: float) -> None:
        """İçeriği piksel cinsinden hareket ettir. +dx sağa, +dy aşağı."""
        raise NotImplementedError

    def system_action(self, name: str) -> None:
        """Görev görünümünü aç veya masaüstleri arasında geç."""
        raise NotImplementedError

    def release_all(self) -> None:
        """Basılı tüm tuşları bırak."""
        raise NotImplementedError
