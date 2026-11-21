"""Tap kayıtlarını değerlendirmek için komut satırı iskeleti."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Kayıt, bip zamanları ve yapılandırma yollarını tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.tools.eval_taps")
    parser.add_argument("file", type=Path, help="Değerlendirilecek JSONL kaydı")
    parser.add_argument("--prompts", type=Path, help="Bip zamanlarını içeren .prompts.json dosyası")
    parser.add_argument("--config", type=Path, default=Path("config/control.toml"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Komut satırı seçeneklerini ayrıştır."""
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
