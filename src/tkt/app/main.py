"""TKT komut satırı uygulaması."""

import argparse
import sys
from pathlib import Path
from threading import Condition, Thread
from time import monotonic

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.os_input.pynput_backend import PynputBackend
from tkt.control.sources import ReplaySource, SyntheticSource
from tkt.core.source import HandFrameSource


def build_parser() -> argparse.ArgumentParser:
    """Uygulamanın kaynak ve backend seçeneklerini tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.app.main")
    parser.add_argument("--source", choices=("live", "replay", "synthetic"), default="live")
    parser.add_argument("--file", type=Path, help="Oynatılacak JSONL kaydı")
    parser.add_argument("--backend", choices=("pynput", "native", "dry"))
    parser.add_argument("--fast", action="store_true", help="Kayıt oynatma beklemesini atla")
    parser.add_argument("--loop", action="store_true", help="Kaydı başa sararak yinele")
    return parser


def _set_dpi_awareness() -> None:
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if set_context is None or not set_context(ctypes.c_void_p(-4)):
            user32.SetProcessDPIAware()


def _open_source(args: argparse.Namespace) -> HandFrameSource:
    if args.source == "live":
        from tkt.vision.live_source import open_live_source

        return open_live_source()
    if args.source == "replay":
        return ReplaySource(args.file, fast=args.fast, loop=args.loop)
    return SyntheticSource().rest(0.2).open(1).move(0.1, 0, 0.5).curl_all()


def _run(source: HandFrameSource, backend: DryBackend | PynputBackend, cfg: dict) -> None:
    engine = GestureEngine(cfg, backend, source.info)
    timeout = cfg["general"]["lost_release_ms"] / 1000
    condition = Condition()
    deadline = monotonic() + timeout
    paused = stopped = False

    def toggle_pause() -> None:
        nonlocal deadline, paused
        with condition:
            if stopped:
                return
            paused = not paused
            if paused:
                deadline = None
                engine.release()
            condition.notify()

    def watch_frames() -> None:
        nonlocal deadline
        with condition:
            while not stopped:
                if paused or deadline is None:
                    condition.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining > 0:
                    condition.wait(remaining)
                else:
                    deadline = None
                    engine.release()

    listener = None
    watchdog = Thread(target=watch_frames, name="tkt-frame-watchdog")
    watchdog.start()
    try:
        if isinstance(backend, PynputBackend):
            from pynput.keyboard import GlobalHotKeys

            listener = GlobalHotKeys({cfg["general"]["panic_hotkey"]: toggle_pause})
            listener.start()
        for frame in source.frames():
            with condition:
                if isinstance(backend, DryBackend):
                    backend.set_time(frame.t)
                if not paused:
                    deadline = monotonic() + timeout
                    condition.notify()
                    engine.process(frame)
    finally:
        with condition:
            stopped = True
            condition.notify()
        try:
            if listener is not None:
                listener.stop()
        finally:
            watchdog.join()


def main(argv: list[str] | None = None) -> None:
    """Komut satırı seçenekleriyle kaynak ve girdi döngüsünü çalıştır."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "replay" and args.file is None:
        parser.error("--source replay için --file gerekli")
    if args.backend == "native":
        parser.error("native backend henüz desteklenmiyor")

    _set_dpi_awareness()
    cfg = load_control_config()
    backend_name = args.backend or ("pynput" if args.source == "live" else "dry")
    source = _open_source(args)
    try:
        if backend_name == "pynput":
            backend = PynputBackend()
            backend.step_px = cfg["scroll"]["step_px"]
        else:
            backend = DryBackend()
        try:
            _run(source, backend, cfg)
        finally:
            backend.release_all()
    finally:
        source.close()


if __name__ == "__main__":
    main()
