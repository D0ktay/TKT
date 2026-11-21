"""Kayıt aracı (V1). PART2 2.4.

    python -m tkt.tools.record --out data/samples/ad.jsonl --note "..." --duration 30 --raw
    python -m tkt.tools.record --out ad.jsonl --prompt 50 --kind tap1 --raw --video
    python -m tkt.tools.record --from-raw ad.raw.jsonl --out ad.jsonl
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tkt.tools.record")
    p.add_argument("--out", required=True, help="HandFrame kaydı (.jsonl)")
    p.add_argument("--note", default="", help="başlıktaki not")
    p.add_argument("--duration", type=float, default=None, help="saniye; yoksa q/Ctrl+C")
    p.add_argument("--raw", action="store_true", help="21 noktayı ad.raw.jsonl'e de yaz")
    p.add_argument("--video", action="store_true", help="görüntüyü ad.mp4'e de yaz")
    p.add_argument("--prompt", type=int, default=None, metavar="N",
                   help="2 sn arayla N bip ver, zamanları ad.prompts.json'a yaz")
    p.add_argument("--kind", choices=("tap1", "tap2"), default="tap1", help="bip türü")
    p.add_argument("--from-raw", default=None, metavar="RAW",
                   help="ham kayıttan güncel features ve lift ile HandFrame kaydı üret")
    p.add_argument("--config", default="config/vision.toml")
    return p


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
