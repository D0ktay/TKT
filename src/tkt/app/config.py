"""Control yapılandırmasını yükleme arayüzü."""

import tomllib
from pathlib import Path


def load_control_config(path: str | Path = "config/control.toml") -> dict:
    """Verilen TOML dosyasından control ayarlarını yükle."""
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)
