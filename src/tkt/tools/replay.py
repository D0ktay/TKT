"""Kayıt ve sentetik jestler için kamerasız OpenCV oynatıcısı."""

import argparse
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from math import ceil, hypot
from pathlib import Path
from sys import platform
from time import monotonic

import cv2
import numpy as np

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.dry_backend import DryBackend, Event
from tkt.control.sources import ReplaySource, SyntheticSource
from tkt.core.source import HandFrameSource
from tkt.core.types import FINGERS, HandFrame

_WINDOW = "TKT Replay"
_DEMOS = ("pointer", "click", "scroll", "tap", "swipe")
_WIDTH, _HEIGHT = 1280, 720
_PANEL_WIDTH, _PANEL_HEIGHT = 610, 590
_TRAIL_SECONDS = 2.0
_RING_SECONDS = 0.4
_NOTICE_SECONDS = 1.0
_POLL_MS = 30

_BG = (23, 20, 18)
_PANEL = (37, 32, 29)
_MUTED = (165, 150, 139)
_TEXT = (239, 231, 220)
_OPEN = (156, 216, 95)
_CURLED = (116, 139, 240)
_CURSOR = (255, 201, 106)
_BUTTON_COLORS = {"left": _CURLED, "right": (221, 136, 230)}

Point = tuple[int, int]
Color = tuple[int, int, int]


def build_parser() -> argparse.ArgumentParser:
    """Kayıt yolunu ve oynatma seçeneklerini tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.tools.replay")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("file", type=Path, nargs="?", help="Görselleştirilecek JSONL kaydı")
    source.add_argument(
        "--synthetic",
        choices=(*_DEMOS, "all"),
        metavar="DEMO",
        help="Sentetik gösteri: " + ", ".join((*_DEMOS, "all")),
    )
    parser.add_argument("--config", type=Path, default=Path("config/control.toml"))
    parser.add_argument("--fast", action="store_true", help="Kareler arası beklemeyi atla")
    parser.add_argument("--loop", action="store_true", help="Oynatmayı başa sararak yinele")
    return parser


def _synthetic(demo: str, cfg: dict) -> SyntheticSource:
    source = SyntheticSource()
    # Include the first sample and leave time to see the selected posture.
    settle = cfg["mode"]["lock_ms"] / 1000 + 0.15
    thumb = cfg["thumb"]["debounce_ms"] / 1000 + 0.06
    if demo == "pointer":
        return (
            source.rest(settle)
            .open(1, settle)
            .move(0.05, -0.03, 0.45)
            .move(0, 0, settle)
            .curl_all(settle)
            .move(-0.14, 0.12, 0.45)
            .open(1, settle)
            .move(-0.04, 0.03, 0.45)
            .move(0, 0, settle)
        )
    if demo == "click":
        return (
            source.rest(settle)
            .open(1, settle)
            .thumb(True, thumb)
            .thumb(False, 0.65)
            .thumb(True, thumb)
            .thumb(False, thumb)
            .thumb(True, thumb)
            .thumb(False, 0.65)
            .thumb(True, thumb)
            .move(0.055, -0.04, 0.55)
            .thumb(False, settle)
            .curl_all(settle)
            .open(2, settle + cfg["thumb"]["still_ms"] / 1000)
            .thumb(True, thumb)
            .thumb(False, 0.45)
        )
    if demo == "scroll":
        coast = cfg["scroll"]["inertia_ms"] / 1000
        return source.rest(settle).open(2, settle).move(0, -0.23, 0.36).curl_all(coast * 3)
    if demo == "tap":
        tap = cfg["tap"]
        cooldown = tap["cooldown_ms"] / 1000 + settle
        # SyntheticSource uses a triangular lift profile. Keep the interval above
        # lift_start long enough for the minimum and below the maximum by default.
        duration = min(tap["max_ms"], (tap["min_ms"] + tap["max_ms"]) / 2)
        return (
            source.rest(settle)
            .open(1, settle)
            .tap(ms=duration)
            .move(0, 0, cooldown)
            .curl_all(settle)
            .open(2, settle)
            .tap(("index", "middle"), ms=duration)
            .move(0, 0, cooldown)
        )
    if demo == "swipe":
        return (
            source.rest(settle)
            .open(4, settle)
            .move(0, -0.3, cfg["swipe"]["window_ms"] / 2000)
            .move(0, 0, _NOTICE_SECONDS)
        )
    raise ValueError(f"Unknown demo: {demo}")


def _demo_config(config: dict, demo: str) -> dict:
    method = {"tap": "tap", "click": "thumb"}.get(demo)
    if method is None:
        return config
    return {**config, "general": {**config["general"], "click_method": method}}


@dataclass
class ScreenState:
    """Sanal işaretçinin konumu ve kısa süreli olay geçmişi."""

    cursor: Point = (_PANEL_WIDTH // 2, 280)
    pressed: set[str] = field(default_factory=set)
    trail: deque[tuple[float, Point, Point, bool]] = field(default_factory=deque)
    rings: deque[tuple[float, Point, str]] = field(default_factory=deque)
    scroll_total: tuple[float, float] = (0.0, 0.0)
    scroll: Event | None = None
    action: Event | None = None

    def update(self, events: list[Event], t: float) -> None:
        """Olayları sırayla uygula ve geçmişi kare zamanına göre kısalt."""
        for event in events:
            if event.kind == "move":
                x, y = self.cursor
                dx, dy = event.args
                self.cursor = x + dx, y + dy
                self.trail.append((event.t, (x, y), self.cursor, bool(self.pressed)))
            elif event.kind == "button":
                name, down = event.args
                if down:
                    self.pressed.add(name)
                    self.rings.append((event.t, self.cursor, name))
                else:
                    self.pressed.discard(name)
            elif event.kind == "scroll":
                dx, dy = event.args
                sx, sy = self.scroll_total
                self.scroll_total = sx + dx, sy + dy
                self.scroll = event
            elif event.kind == "system":
                self.action = event

        for history, duration in ((self.trail, _TRAIL_SECONDS), (self.rings, _RING_SECONDS)):
            while history and t - history[0][0] > duration:
                history.popleft()


def _text(
    image: np.ndarray, text: str, position: Point, color: Color = _TEXT, scale: float = 0.55
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _panel(title: str) -> np.ndarray:
    image = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), _PANEL, dtype=np.uint8)
    _text(image, title, (20, 35), _TEXT, 0.72)
    return image


def _image_point(xy: tuple[float, float]) -> Point:
    x, y = xy
    return 24 + round(x * 562), 85 + round(y * 335)


def _bar(image: np.ndarray, x: int, y: int, value: float | None, color: Color) -> None:
    cv2.rectangle(image, (x, y), (x + 120, y + 9), _BG, -1)
    if value is not None and value > 0:
        cv2.rectangle(image, (x, y), (x + round(min(1.0, value) * 120), y + 9), color, -1)


def draw_hand(frame: HandFrame, curl_open: float, curl_closed: float) -> np.ndarray:
    """El panelini yeni bir görüntü olarak döndür."""
    image = _panel("HAND")
    _text(image, f"present={int(frame.present)}   confidence={frame.confidence:.2f}", (20, 63))
    if not frame.present:
        _text(image, "NO HAND", (250, 265), _MUTED, 0.8)
        return image

    palette = {}
    palm = _image_point(frame.palm)
    for name, finger in frame.fingers.items():
        color = _MUTED
        if finger.curl < curl_open:
            color = _OPEN
        elif finger.curl > curl_closed:
            color = _CURLED
        palette[name] = color
        tip = _image_point(finger.img)
        cv2.line(image, palm, tip, color, 1, cv2.LINE_AA)
        cv2.circle(image, tip, 7, color, -1, cv2.LINE_AA)
    cv2.circle(image, palm, 9, _TEXT, -1, cv2.LINE_AA)

    _text(image, "LIFT", (20, 455), _MUTED)
    for i, name in enumerate(FINGERS):
        finger = frame.fingers.get(name)
        x = 20 + 145 * i
        lift = None if finger is None else finger.lift
        label = "n/a" if lift is None else f"{lift:.2f}"
        color = palette.get(name, _MUTED)
        _text(image, f"{name} {label}", (x, 483), color)
        _bar(image, x, 495, lift, color)

    _text(image, f"thumb_out {frame.thumb_out:.2f}", (20, 550))
    _bar(image, 205, 539, frame.thumb_out, _OPEN)
    return image


def _screen_point(point: Point) -> Point:
    x, y = point
    return max(18, min(_PANEL_WIDTH - 19, x)), max(78, min(506, y))


def draw_screen(state: ScreenState, t: float) -> np.ndarray:
    """Sanal ekran panelini durumu değiştirmeden çiz."""
    image = _panel("VIRTUAL SCREEN")
    cv2.rectangle(image, (12, 70), (_PANEL_WIDTH - 13, 514), _BG, -1)
    for _, start, end, down in state.trail:
        color = _BUTTON_COLORS["left"] if down else _CURSOR
        cv2.line(image, _screen_point(start), _screen_point(end), color, 2, cv2.LINE_AA)
    for born, point, button in state.rings:
        radius = 10 + round(19 * (t - born) / _RING_SECONDS)
        cv2.circle(
            image, _screen_point(point), radius, _BUTTON_COLORS.get(button, _TEXT), 2, cv2.LINE_AA
        )
    color = _BUTTON_COLORS["left"] if state.pressed else _CURSOR
    cv2.circle(image, _screen_point(state.cursor), 6, color, -1, cv2.LINE_AA)

    if state.action is not None and t - state.action.t < _NOTICE_SECONDS:
        label = state.action.args[0].replace("_", " ").upper()
        _text(image, label, (28, 105), _OPEN, 0.8)
    if state.scroll is not None and t - state.scroll.t < _NOTICE_SECONDS:
        dx, dy = state.scroll.args
        length = hypot(dx, dy)
        if length:
            tip = 550 + round(dx / length * 24), 549 + round(dy / length * 24)
            cv2.arrowedLine(image, (550, 549), tip, _OPEN, 2, cv2.LINE_AA, tipLength=0.35)
    sx, sy = state.scroll_total
    _text(image, f"Scroll X {sx:+.1f}   Y {sy:+.1f}", (20, 553))
    return image


def _enable_cocoa_close():
    if platform != "darwin" or cv2.currentUIFramework() != "COCOA":
        return None

    from AppKit import NSApplication, NSWindowStyleMaskClosable

    # OpenCV 4.11 creates Cocoa windows without the closable style. Keep the
    # native window alive until HighGUI removes it from its window registry.
    for window in NSApplication.sharedApplication().windows():
        if window.title() == _WINDOW:
            style = window.styleMask()
            if not style & NSWindowStyleMaskClosable:
                releases_on_close = window.isReleasedWhenClosed()
                window.setReleasedWhenClosed_(False)
                window.setStyleMask_(style | NSWindowStyleMaskClosable)
                return window, releases_on_close
            break
    return None


@dataclass
class _Playback:
    fast: bool
    paused: bool = False
    speed: float = 1.0
    seen_visible: bool = False
    shown: bool = False
    cocoa: tuple | None = None

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(_WINDOW, image)
        if not self.shown:
            self.shown = True
            self.cocoa = _enable_cocoa_close()

    def key(self, delay: int) -> int:
        key = cv2.waitKey(delay) & 0xFF
        try:
            visible = cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible >= 1:
            self.seen_visible = True
        elif visible == 0 or self.seen_visible:
            return ord("q")
        return key

    def close(self) -> None:
        if self.cocoa is not None:
            window, releases_on_close = self.cocoa
            window.setReleasedWhenClosed_(releases_on_close)
        if self.shown:
            with suppress(cv2.error):
                cv2.destroyWindow(_WINDOW)


def _present_frame(canvas: np.ndarray, status: str, playback: _Playback, seconds: float) -> str:
    """Kareyi göster, sıradaki kareye kadar klavyeyi ve pencereyi dinle."""
    while True:
        speed = "FAST" if playback.fast else f"{playback.speed:g}x"
        paused = "  PAUSED" if playback.paused else ""
        canvas[:82] = _BG
        _text(canvas, "TKT REPLAY", (24, 32), _TEXT, 0.85)
        _text(canvas, f"{status}   Speed {speed}{paused}", (24, 66))
        playback.show(canvas)
        while True:
            timed = not (playback.fast or playback.paused)
            delay = max(1, min(_POLL_MS, ceil(seconds / playback.speed * 1000))) if timed else 1
            if playback.paused:
                delay = _POLL_MS
            started = monotonic()
            key = playback.key(delay)
            if timed:
                seconds -= (monotonic() - started) * playback.speed
            if key in (ord("q"), 27):
                return "quit"
            if key == ord("r"):
                return "rewind"
            if key == ord(" ") or key in (ord("+"), ord("="), ord("-")):
                if key == ord(" "):
                    playback.paused = not playback.paused
                else:
                    playback.fast = False
                    factor = 0.5 if key == ord("-") else 2
                    playback.speed = max(0.125, min(16.0, playback.speed * factor))
                break
            if playback.paused:
                if key == ord("."):
                    return "next"
            elif playback.fast or seconds <= 0:
                return "next"


def _play_source(source: HandFrameSource, label: str, cfg: dict, playback: _Playback) -> str:
    backend = DryBackend()
    engine = GestureEngine(cfg, backend, source.info)
    state = ScreenState()
    frames = source.frames()
    mode = cfg["mode"]
    try:
        frame = next(frames, None)
        while frame is not None:
            backend.set_time(frame.t)
            engine.process(frame)
            state.update(backend.events, frame.t)
            backend.events.clear()

            canvas = np.full((_HEIGHT, _WIDTH, 3), _BG, dtype=np.uint8)
            canvas[84:674, 20:630] = draw_hand(frame, mode["curl_open"], mode["curl_closed"])
            canvas[84:674, 650:1260] = draw_screen(state, frame.t)
            _text(canvas, "Space Pause   . Step   +/- Speed   R Rewind   Q / Esc Exit", (24, 705))
            status = (
                f"{label}   Mode {engine.mode.name}   left_down={int(engine.left_down)}"
                f"   Frame {frame.frame_id}   t={frame.t:.3f}s"
            )
            following = next(frames, None)
            seconds = following.t - frame.t if following is not None else 1 / source.info.fps
            command = _present_frame(canvas, status, playback, seconds)
            if command != "next":
                return command
            frame = following
    finally:
        engine.release()
    return "end"


def main(argv: list[str] | None = None) -> None:
    """Oynatıcıyı çağıran ana thread üzerinde çalıştır."""
    args = build_parser().parse_args(argv)
    cfg = load_control_config(args.config)
    playlist = _DEMOS if args.synthetic == "all" else (args.synthetic,)
    playback = _Playback(args.fast)
    index = 0
    try:
        while index < len(playlist):
            demo = playlist[index]
            if demo is None:
                source = ReplaySource(args.file, fast=True)
                label, source_cfg = "RECORDING", cfg
            else:
                source = _synthetic(demo, cfg)
                label, source_cfg = demo.upper(), _demo_config(cfg, demo)
            try:
                command = _play_source(source, label, source_cfg, playback)
            finally:
                source.close()
            if command == "quit":
                break
            if command == "rewind":
                index = 0
            else:
                index += 1
                if args.loop and playback.shown and index == len(playlist):
                    index = 0
    finally:
        playback.close()


if __name__ == "__main__":
    main()
