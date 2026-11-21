"""TKT komut satırı uygulaması."""

import argparse
import csv
import logging
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from statistics import median, quantiles
from threading import Condition, Thread
from time import monotonic, perf_counter
from typing import TextIO

from tkt.app.config import load_control_config
from tkt.app.hud import HudClosed
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.base import InputBackend
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.os_input.macos import MacBackend
from tkt.control.os_input.pynput_backend import PynputBackend
from tkt.control.os_input.windows import WindowsBackend
from tkt.control.sources import ReplaySource, SyntheticSource
from tkt.core.source import HandFrameSource
from tkt.core.types import HandFrame

_REPORT_INTERVAL = 5.0
_RELEASE_ATTEMPTS = 3
_LOG = logging.getLogger(__name__)
_HOTKEY_ERROR = (
    "TKT acil durdurma dinleyicisi çalışmıyor. "
    "macOS'ta Erişilebilirlik ve Girdi İzleme izinlerini kontrol edin."
)


class _TimedBackend:
    def __init__(self, backend: InputBackend) -> None:
        self._backend = backend
        self.sent_t: float | None = None

    def _send(self, send: Callable[..., None], *args: object) -> None:
        before = getattr(self._backend, "events_sent", None)
        send(*args)
        if self.sent_t is None and (
            before is None or getattr(self._backend, "events_sent", None) != before
        ):
            self.sent_t = monotonic()

    def move_rel(self, dx: int, dy: int) -> None:
        self._send(self._backend.move_rel, dx, dy)

    def button(self, name: str, down: bool) -> None:
        self._send(self._backend.button, name, down)

    def scroll(self, dx: float, dy: float) -> None:
        self._send(self._backend.scroll, dx, dy)

    def system_action(self, name: str) -> None:
        self._send(self._backend.system_action, name)

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
    parser.add_argument("--hud", action="store_true",
                        help="Kamera görüntüsü ve algılanan hareketi pencerede göster (q kapatır)")
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


def _release_with_retry(release: Callable[[], None]) -> int | None:
    """Başarılı denemenin numarasını döndür. Tüm denemeler başarısızsa None döndür."""
    for attempt in range(1, _RELEASE_ATTEMPTS + 1):
        try:
            release()
        except Exception:
            _LOG.exception("Girdi tuşları bırakılamadı (%d/%d)", attempt, _RELEASE_ATTEMPTS)
        else:
            return attempt
    return None


def _run(
    source: HandFrameSource,
    backend: InputBackend,
    cfg: dict,
    report: _LatencyReport | None = None,
    hud: bool = False,
) -> None:
    hud_view = None
    os_backend = backend          # tür kontrolleri (acil durdurma, DryBackend saati) için
    if hud:
        from tkt.app.hud import ActionTap, Hud, HudSource

        source = HudSource(source)
        backend = ActionTap(backend, monotonic)
        hud_view = Hud(source, backend, cfg["general"]["confidence_min"])
    timed_backend = _TimedBackend(backend) if report is not None else None
    engine = GestureEngine(cfg, timed_backend or backend, source.info)
    general = cfg["general"]
    timeout = general["lost_release_ms"] / 1000
    listener_interval = timeout / 4
    condition = Condition()
    deadline = None
    paused = stopped = release_pending = listener_failed = False
    listener = None

    def release_inputs() -> None:
        nonlocal deadline, paused, release_pending
        deadline = None
        attempt = _release_with_retry(engine.release)
        release_pending = attempt is None
        if attempt != 1:
            paused = True

    def toggle_pause() -> None:
        nonlocal paused
        with condition:
            if stopped or listener_failed:
                return
            if paused:
                if release_pending:
                    release_inputs()
                paused = release_pending
            else:
                paused = True
                release_inputs()
            condition.notify()

    def check_listener() -> None:
        nonlocal listener_failed
        if listener is not None and not listener_failed:
            if not (listener.is_alive() and listener.running):
                listener_failed = True
                _LOG.error(_HOTKEY_ERROR)
                release_inputs()

    def expire_frames(now: float) -> None:
        if deadline is not None and now >= deadline:
            release_inputs()

    def watch_frames() -> None:
        with condition:
            while not stopped:
                check_listener()
                now = monotonic()
                expire_frames(now)
                if report is not None and now >= report.deadline:
                    report.report(now)
                wake_at = deadline
                if listener is not None and not listener_failed:
                    wake_at = now + listener_interval if wake_at is None else min(
                        wake_at, now + listener_interval
                    )
                if report is not None:
                    wake_at = report.deadline if wake_at is None else min(wake_at, report.deadline)
                if wake_at is None:
                    condition.wait()
                else:
                    condition.wait(max(0.0, wake_at - monotonic()))

    watchdog = Thread(target=watch_frames, name="tkt-frame-watchdog")
    try:
        if isinstance(os_backend, (PynputBackend, WindowsBackend, MacBackend)):
            from pynput.keyboard import GlobalHotKeys

            listener = GlobalHotKeys({general["panic_hotkey"]: toggle_pause})
            # Quartz olay dinleyicisi kurulamazsa pynput sessizce çıkabilir.
            try:
                listener.start()
                listener.join(timeout=listener_interval)
            except Exception as exc:
                raise RuntimeError(_HOTKEY_ERROR) from exc
            if not (listener.is_alive() and listener.running):
                raise RuntimeError(_HOTKEY_ERROR)
        watchdog.start()
        for frame in source.frames():
            with condition:
                check_listener()
                if listener_failed:
                    raise RuntimeError(_HOTKEY_ERROR)
                now = monotonic()
                expire_frames(now)
                if isinstance(os_backend, DryBackend):
                    os_backend.set_time(frame.t)
                if not paused:
                    # Kayıt ve sentetik kareler ayrı bir zaman ekseninde olabilir.
                    if frame.present and frame.confidence >= general["confidence_min"]:
                        deadline = now + timeout
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
            if hud_view is not None:
                hud_view.show(frame, engine, monotonic(), paused)
        with condition:
            check_listener()
            if listener_failed:
                raise RuntimeError(_HOTKEY_ERROR)
    finally:
        with condition:
            stopped = True
            condition.notify()
        try:
            if listener is not None:
                listener.stop()
        finally:
            if watchdog.ident is not None:
                watchdog.join()
            if report is not None:
                report.report(monotonic(), final=True)
            if hud_view is not None:
                hud_view.close()


def main(argv: list[str] | None = None) -> None:
    """Komut satırı seçenekleriyle kaynak ve girdi döngüsünü çalıştır."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "replay" and args.file is None:
        parser.error("--source replay için --file gerekli")
    if args.backend == "native" and sys.platform not in ("win32", "darwin"):
        parser.error("native backend yalnızca Windows ve macOS'ta var")
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
        elif backend_name == "native" and sys.platform == "darwin":
            backend = MacBackend()
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
                try:
                    _run(source, backend, cfg, report, hud=args.hud)
                except HudClosed:
                    pass
        finally:
            if _release_with_retry(backend.release_all) is None:
                raise RuntimeError("Çıkışta girdi tuşları bırakılamadı")
    finally:
        source.close()


if __name__ == "__main__":
    main()
