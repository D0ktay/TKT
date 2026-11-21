"""TKT komut satırı uygulaması."""

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import median, quantiles
from threading import Condition, Thread
from time import monotonic, perf_counter
from typing import TextIO

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.base import InputBackend
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.os_input.pynput_backend import PynputBackend
from tkt.control.os_input.windows import WindowsBackend
from tkt.control.sources import ReplaySource, SyntheticSource
from tkt.core.source import HandFrameSource
from tkt.core.types import HandFrame

_REPORT_INTERVAL = 5.0


class _TimedBackend:
    def __init__(self, backend: InputBackend) -> None:
        self._backend = backend
        self.sent_t: float | None = None

    def _mark_event(self) -> None:
        if self.sent_t is None:
            self.sent_t = monotonic()

    def move_rel(self, dx: int, dy: int) -> None:
        self._mark_event()
        self._backend.move_rel(dx, dy)

    def button(self, name: str, down: bool) -> None:
        self._mark_event()
        self._backend.button(name, down)

    def scroll(self, dx: float, dy: float) -> None:
        self._mark_event()
        self._backend.scroll(dx, dy)

    def system_action(self, name: str) -> None:
        self._mark_event()
        self._backend.system_action(name)

    def release_all(self) -> None:
        self._backend.release_all()


def _percentiles(samples: list[float]) -> str:
    if not samples:
        return "p50/p95=—"
    p95 = quantiles(samples, n=20, method="inclusive")[-1] if len(samples) > 1 else samples[0]
    return f"p50={median(samples):.3f} p95={p95:.3f}"


class _LatencyReport:
    def __init__(self, live: bool, output: TextIO | None) -> None:
        self._live = live
        self._output = output
        self._csv = csv.writer(output) if output is not None else None
        if self._csv is not None:
            self._csv.writerow(("frame_id", "t", "sent_t", "latency_ms", "process_ms"))
        self._latencies: list[float] = []
        self._process_times: list[float] = []
        self._event_frames = 0
        self._started = monotonic()

    @property
    def deadline(self) -> float:
        return self._started + _REPORT_INTERVAL

    def record(self, frame: HandFrame, sent_t: float | None, process_ms: float) -> None:
        latency_ms = None
        if sent_t is not None:
            self._event_frames += 1
            if self._live:
                latency_ms = (sent_t - frame.t) * 1000
                self._latencies.append(latency_ms)
        self._process_times.append(process_ms)
        if self._csv is not None:
            self._csv.writerow((frame.frame_id, frame.t, sent_t, latency_ms, process_ms))

    def report(self, now: float, *, final: bool = False) -> None:
        elapsed = now - self._started
        fps = len(self._process_times) / elapsed if elapsed > 0 else 0.0
        if not self._live:
            latency = "geçersiz"
        elif not self._latencies:
            latency = "p50/p95/max=—"
        else:
            latency = f"{_percentiles(self._latencies)} max={max(self._latencies):.3f} ms"
        label = "çıkış" if final else "ara"
        print(
            f"Gecikme ({label}) {latency} (hedef p95 < 60 ms) | "
            f"olaylı kare={self._event_frames} | "
            f"kontrol {_percentiles(self._process_times)} ms (hedef < 2 ms) | {fps:.2f} fps",
            flush=True,
        )
        if self._output is not None:
            self._output.flush()
        self._latencies.clear()
        self._process_times.clear()
        self._event_frames = 0
        self._started = now


def build_parser() -> argparse.ArgumentParser:
    """Uygulamanın kaynak ve backend seçeneklerini tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.app.main")
    parser.add_argument("--source", choices=("live", "replay", "synthetic"), default="live")
    parser.add_argument("--file", type=Path, help="Oynatılacak JSONL kaydı")
    parser.add_argument("--backend", choices=("pynput", "native", "dry"))
    parser.add_argument("--fast", action="store_true", help="Kayıt oynatma beklemesini atla")
    parser.add_argument("--loop", action="store_true", help="Kaydı başa sararak yinele")
    parser.add_argument("--latency", action="store_true", help="Gecikme ve kontrol süresini ölç")
    parser.add_argument("--latency-out", type=Path, help="Ölçümü aç ve kareleri CSV'ye yaz")
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


def _run(
    source: HandFrameSource,
    backend: InputBackend,
    cfg: dict,
    report: _LatencyReport | None = None,
) -> None:
    timed_backend = _TimedBackend(backend) if report is not None else None
    engine = GestureEngine(cfg, timed_backend or backend, source.info)
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
                now = monotonic()
                if deadline is not None and now >= deadline:
                    deadline = None
                    engine.release()
                if report is not None and now >= report.deadline:
                    report.report(now)
                wake_at = deadline
                if report is not None:
                    wake_at = report.deadline if wake_at is None else min(wake_at, report.deadline)
                if wake_at is None:
                    condition.wait()
                else:
                    condition.wait(max(0.0, wake_at - monotonic()))

    listener = None
    watchdog = Thread(target=watch_frames, name="tkt-frame-watchdog")
    watchdog.start()
    try:
        if isinstance(backend, (PynputBackend, WindowsBackend)):
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
                    if timed_backend is None or report is None:
                        engine.process(frame)
                    else:
                        timed_backend.sent_t = None
                        started = perf_counter()
                        engine.process(frame)
                        process_ms = (perf_counter() - started) * 1000
                        report.record(frame, timed_backend.sent_t, process_ms)
                if report is not None:
                    now = monotonic()
                    if now >= report.deadline:
                        report.report(now)
    finally:
        with condition:
            stopped = True
            condition.notify()
        try:
            if listener is not None:
                listener.stop()
        finally:
            watchdog.join()
            if report is not None:
                report.report(monotonic(), final=True)


def main(argv: list[str] | None = None) -> None:
    """Komut satırı seçenekleriyle kaynak ve girdi döngüsünü çalıştır."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "replay" and args.file is None:
        parser.error("--source replay için --file gerekli")
    if args.backend == "native" and sys.platform != "win32":
        if sys.platform == "darwin":
            parser.error("native henüz yok (macos.py, V3 şeridi)")
        parser.error("native backend henüz desteklenmiyor")
    if args.source == "replay" and args.latency_out is not None:
        if args.file.resolve() == args.latency_out.resolve() or (
            args.file.exists()
            and args.latency_out.exists()
            and args.file.samefile(args.latency_out)
        ):
            parser.error("--latency-out kayıt dosyasından farklı olmalı")

    _set_dpi_awareness()
    cfg = load_control_config()
    backend_name = args.backend or ("pynput" if args.source == "live" else "dry")
    source = _open_source(args)
    try:
        if backend_name == "pynput":
            backend = PynputBackend()
            backend.step_px = cfg["scroll"]["step_px"]
        elif backend_name == "native":
            backend = WindowsBackend(step_px=cfg["scroll"]["step_px"], smooth=True)
        else:
            backend = DryBackend()
        try:
            with ExitStack() as files:
                report = None
                if args.latency or args.latency_out is not None:
                    output = None
                    if args.latency_out is not None:
                        output = files.enter_context(args.latency_out.open("w", newline=""))
                    report = _LatencyReport(args.source == "live", output)
                _run(source, backend, cfg, report)
        finally:
            backend.release_all()
    finally:
        source.close()


if __name__ == "__main__":
    main()
