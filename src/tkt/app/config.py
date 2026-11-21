"""Control yapılandırmasını yükleme arayüzü."""

from pathlib import Path


def load_control_config(path: str | Path = "config/control.toml") -> dict:
    """Verilen TOML dosyasından control ayarlarını yükle."""
    raise NotImplementedError
