"""pynput ile işletim sistemine girdi gönderen backend."""


class PynputBackend:
    """pynput bağımlılığını yalnızca örnek oluşturulurken yükler."""

    def __init__(self) -> None:
        """Girdi denetleyicilerini hazırla."""
        import pynput  # noqa: F401

        raise NotImplementedError

    def move_rel(self, dx: int, dy: int) -> None:
        """İmleci piksel cinsinden göreli hareket ettir."""
        raise NotImplementedError

    def button(self, name: str, down: bool) -> None:
        """Sol veya sağ tuşu bas ya da bırak."""
        raise NotImplementedError

    def scroll(self, dx: float, dy: float) -> None:
        """Piksel cinsinden hareketi kaydırma adımlarına dönüştür."""
        raise NotImplementedError

    def system_action(self, name: str) -> None:
        """Platforma uygun sistem kısayolunu gönder."""
        raise NotImplementedError

    def release_all(self) -> None:
        """Basılı tüm tuşları bırak."""
        raise NotImplementedError
