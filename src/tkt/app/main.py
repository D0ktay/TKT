"""TKT uygulamasının komut satırı iskeleti."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Uygulamanın kaynak ve backend seçeneklerini tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.app.main")
    parser.add_argument("--source", choices=("live", "replay", "synthetic"), default="live")
    parser.add_argument("--file", type=Path, help="Oynatılacak JSONL kaydı")
    parser.add_argument("--backend", choices=("pynput", "native", "dry"), default="pynput")
    parser.add_argument("--fast", action="store_true", help="Kareler arası beklemeyi atla")
    parser.add_argument("--loop", action="store_true", help="Kaydı başa sararak yinele")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Komut satırı seçeneklerini ayrıştır."""
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
