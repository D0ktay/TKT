"""Debug penceresi (V3). PART2 2.2, 2.4.

macOS'ta `cv2.imshow` ana thread'de çalışmalıdır.

CLI: python -m tkt.vision.debug_view
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tkt.vision.live_source import LiveSource


def run(source: LiveSource, calib_path: Path) -> None:
    """İskelet, curl/lift çubukları, fps; `c` kalibrasyon, `q` çıkış. Ana thread'de çalışır."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tkt.vision.debug_view")
    parser.add_argument("--config", default="config/vision.toml")
    parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
