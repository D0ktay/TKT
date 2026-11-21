"""El karesi kaydını görselleştirmek için komut satırı iskeleti."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Kayıt yolunu ve oynatma seçeneklerini tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.tools.replay")
    parser.add_argument("file", type=Path, help="Görselleştirilecek JSONL kaydı")
    parser.add_argument("--config", type=Path, default=Path("config/control.toml"))
    parser.add_argument("--fast", action="store_true", help="Kareler arası beklemeyi atla")
    parser.add_argument("--loop", action="store_true", help="Kaydı başa sararak yinele")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Komut satırı seçeneklerini ayrıştır."""
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
