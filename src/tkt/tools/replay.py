"""Kayıt ve sentetik jestler için kamerasız OpenCV oynatıcısı."""

import argparse
import json
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from math import ceil, hypot, isfinite
from pathlib import Path
from sys import platform
from time import monotonic

import cv2
import numpy as np

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine, Mode
from tkt.control.os_input.dry_backend import DryBackend, Event
from tkt.control.sources import ReplaySource, SyntheticSource
from tkt.core.io import prompts_path, read_prompts
from tkt.core.source import HandFrameSource
from tkt.core.types import FINGERS, HandFrame

_WINDOW = "TKT Replay"
_DEMOS = ("pointer", "click", "scroll", "tap", "swipe")
_DEMO_DESCRIPTIONS = {
    "pointer": "Slide one finger to move the cursor. Curl it to reposition.",
    "click": "Close/open thumb to click. Repeat quickly to double-click. Hold closed to drag.",
    "scroll": "Slide two fingers to scroll. Curl them to coast.",
    "tap": "Lift and tap one finger for left click or two for right click.",
    "swipe": "Slide four open fingers up for Mission Control.",
}
_WIDTH, _HEIGHT = 1280, 720
_PANEL_WIDTH, _PANEL_HEIGHT = 610, 590
_PANEL_TOP = 76
_HAND_CENTER = (305, 290)
_HAND_PIXELS = 450
_SCALE_BLEND = 0.15
_BAR_WIDTH = 120
_TIMELINE_LEFT, _TIMELINE_RIGHT, _TIMELINE_Y = 24, 920, 683
_HELP_REGION = np.s_[104:560, 670:1240]
_TRAIL_SECONDS = 2.0
_RING_SECONDS = 0.4
_NOTICE_SECONDS = 1.0
_DOUBLE_CLICK_SECONDS = 0.4
_LOG_ROWS = 8
_POLL_MS = 30

_BG = (23, 20, 18)
_PANEL = (37, 32, 29)
_MUTED = (165, 150, 139)
_TEXT = (239, 231, 220)
_OPEN = (156, 216, 95)
_CURLED = (116, 139, 240)
_CURSOR = (255, 201, 106)
_YELLOW = (105, 218, 245)
_PURPLE = (238, 156, 190)
_BUTTON_COLORS = {"left": _CURLED, "right": (221, 136, 230)}
_MODE_COLORS = {
    Mode.NONE: _MUTED,
    Mode.WAIT: _YELLOW,
    Mode.POINTER: _CURSOR,
    Mode.SCROLL: _PURPLE,
    Mode.SWIPE: _CURLED,
}
_PROMPT_COLORS = {"tap1": _YELLOW, "tap2": _PURPLE}

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
    log: deque[Event] = field(default_factory=lambda: deque(maxlen=_LOG_ROWS), init=False)
    notice: tuple[float, str] | None = field(default=None, init=False)
    _last_press: dict[str, float] = field(default_factory=dict, init=False)

    def update(self, events: list[Event], t: float) -> None:
        """Olayları sırayla uygula ve geçmişi kare zamanına göre kısalt."""
        for event in events:
            self.log.append(event)
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
                    previous = self._last_press.get(name)
                    if (
                        previous is not None
                        and previous <= event.t <= previous + _DOUBLE_CLICK_SECONDS
                    ):
                        self.notice = event.t, "DOUBLE CLICK"
                        del self._last_press[name]
                    else:
                        self.notice = event.t, f"{name.upper()} CLICK"
                        self._last_press[name] = event.t
                else:
                    self.pressed.discard(name)
            elif event.kind == "scroll":
                dx, dy = event.args
                sx, sy = self.scroll_total
                self.scroll_total = sx + dx, sy + dy
                self.scroll = event
                self.notice = event.t, "SCROLL"
            elif event.kind == "system":
                self.action = event
                self.notice = event.t, event.args[0].replace("_", " ").upper()

        for history, duration in ((self.trail, _TRAIL_SECONDS), (self.rings, _RING_SECONDS)):
            while history and t - history[0][0] > duration:
                history.popleft()

    def notification(self, t: float) -> str | None:
        """Geçerli bildirimi oynatma zamanına göre döndür."""
        if self.notice is not None:
            born, label = self.notice
            if born <= t < born + _NOTICE_SECONDS:
                return label
        return None


def _text(
    image: np.ndarray, text: str, position: Point, color: Color = _TEXT, scale: float = 0.55
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _panel() -> np.ndarray:
    return np.empty((_PANEL_HEIGHT, _PANEL_WIDTH, 3), dtype=np.uint8)


def _draw_panel(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (_PANEL_WIDTH - 1, _PANEL_HEIGHT - 1), _PANEL, -1)
    _text(image, title, (20, 34), _TEXT, 0.72)


@dataclass
class _HandView:
    aspect: float = 9 / 16
    scale: float | None = None
    thumb_closed: bool | None = None

    def update(self, frame: HandFrame, thumb: dict | None = None) -> None:
        if not frame.present:
            return
        if isfinite(frame.hand_size) and frame.hand_size > 0:
            target = _HAND_PIXELS / frame.hand_size
            self.scale = (
                target if self.scale is None else self.scale + _SCALE_BLEND * (target - self.scale)
            )
        if thumb is not None:
            if frame.thumb_out < thumb["closed"]:
                self.thumb_closed = True
            elif frame.thumb_out > thumb["open"]:
                self.thumb_closed = False

    def point(self, xy: tuple[float, float], palm: tuple[float, float]) -> Point:
        scale = self.scale if self.scale is not None else _HAND_PIXELS
        x = _HAND_CENTER[0] + (xy[0] - palm[0]) * scale
        y = _HAND_CENTER[1] + (xy[1] - palm[1]) * scale * self.aspect
        # Keep the tip and its label inside the drawing area.
        return round(max(36, min(_PANEL_WIDTH - 36, x))), round(max(107, min(376, y)))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _curl_color(value: float, thresholds: tuple[float, float]) -> Color:
    low, high = thresholds
    if value <= low:
        return _OPEN
    if value >= high:
        return _CURLED
    mix = (value - low) / (high - low)
    return tuple(round(a + (b - a) * mix) for a, b in zip(_OPEN, _CURLED, strict=True))


def _bar(
    image: np.ndarray,
    x: int,
    y: int,
    value: float | None,
    color: Color,
    thresholds: tuple[float, float] | None,
) -> None:
    cv2.rectangle(image, (x, y), (x + _BAR_WIDTH, y + 9), _BG, -1)
    if value is not None and value > 0:
        cv2.rectangle(image, (x, y), (x + round(_unit(value) * _BAR_WIDTH), y + 9), color, -1)
    if thresholds is not None:
        for threshold, marker in zip(thresholds, (_TEXT, _YELLOW), strict=True):
            tick = x + round(_unit(threshold) * _BAR_WIDTH)
            cv2.line(image, (tick, y - 3), (tick, y + 12), marker, 1)


def _draw_hand(
    image: np.ndarray,
    frame: HandFrame,
    curl: tuple[float, float],
    lift: tuple[float, float] | None,
    view: _HandView,
) -> None:
    _draw_panel(image, "HAND")
    _text(image, f"Confidence {frame.confidence:.2f}", (430, 34), _MUTED)
    cv2.rectangle(image, (12, 58), (_PANEL_WIDTH - 13, 390), _BG, -1)
    if not frame.present:
        _text(image, "NO HAND", (250, 230), _MUTED, 0.8)
        return

    palm = _HAND_CENTER
    cv2.circle(image, palm, 16, _PANEL, -1, cv2.LINE_AA)
    _text(image, f"CURL   Open {curl[0]:g} / Closed {curl[1]:g}", (20, 418), _MUTED)
    lift_legend = "" if lift is None else f"   Start {lift[0]:g} / Up {lift[1]:g}"
    _text(image, f"LIFT{lift_legend}", (20, 493), _MUTED)
    for i, name in enumerate(FINGERS):
        finger = frame.fingers.get(name)
        x = 20 + 145 * i
        curl_value = None if finger is None else finger.curl
        lift_value = None if finger is None else finger.lift
        color = _MUTED if curl_value is None else _curl_color(curl_value, curl)
        if finger is not None:
            tip = view.point(finger.img, frame.palm)
            cv2.line(image, palm, tip, color, 2, cv2.LINE_AA)
            cv2.circle(image, tip, 7, color, -1, cv2.LINE_AA)
            _text(image, name[0].upper(), (tip[0] - 5, tip[1] - 15), color)
        curl_label = "n/a" if curl_value is None else f"{curl_value:.2f}"
        lift_label = "n/a" if lift_value is None else f"{lift_value:.2f}"
        _text(image, f"{name} {curl_label}", (x, 441), color, 0.48)
        _bar(image, x, 450, curl_value, color, curl)
        _text(image, f"{name} {lift_label}", (x, 516), color, 0.48)
        _bar(image, x, 525, lift_value, _CURSOR, lift)

    cv2.circle(image, palm, 8, _TEXT, -1, cv2.LINE_AA)
    thumb_color = _MUTED
    if view.thumb_closed is not None:
        thumb_label = "THUMB CLOSED" if view.thumb_closed else "THUMB OPEN"
        thumb_color = _CURLED if view.thumb_closed else _OPEN
        thumb_icon = palm[0] + 34, palm[1] + 23
        cv2.circle(image, thumb_icon, 6, thumb_color, -1 if view.thumb_closed else 2, cv2.LINE_AA)
        _text(image, thumb_label, (thumb_icon[0] + 14, thumb_icon[1] + 5), thumb_color, 0.45)
    _text(image, f"Thumb out {frame.thumb_out:.2f}", (20, 570), _MUTED)
    _bar(image, 185, 561, frame.thumb_out, thumb_color, None)


def draw_hand(frame: HandFrame, curl_open: float, curl_closed: float) -> np.ndarray:
    """El panelini yeni bir görüntü olarak döndür."""
    image = _panel()
    view = _HandView()
    view.update(frame)
    _draw_hand(image, frame, (curl_open, curl_closed), None, view)
    return image


def _screen_point(point: Point) -> Point:
    x, y = point
    # Keep the existing virtual cursor bounds while fitting the shorter viewport.
    return max(18, min(_PANEL_WIDTH - 19, x)), 66 + round((max(78, min(506, y)) - 78) * 258 / 428)


def _draw_event_log(image: np.ndarray, state: ScreenState) -> None:
    _text(image, "RECENT EVENTS", (20, 368), _TEXT, 0.63)
    _text(image, "TIME", (24, 393), _MUTED, 0.46)
    _text(image, "EVENT", (175, 393), _MUTED, 0.46)
    _text(image, "ARGUMENTS", (326, 393), _MUTED, 0.46)
    for row, event in enumerate(reversed(state.log)):
        y = 416 + 21 * row
        args = "   ".join(
            f"{arg:+.2f}" if isinstance(arg, float) else ascii(arg) for arg in event.args
        )
        _text(image, f"{event.t:.3f}s", (24, y), _MUTED, 0.48)
        _text(image, event.kind.upper(), (175, y), _TEXT, 0.48)
        _text(image, args, (326, y), _TEXT, 0.48)
    if not state.log:
        _text(image, "No events yet", (24, 416), _MUTED, 0.48)


def _draw_screen(image: np.ndarray, state: ScreenState, t: float) -> None:
    _draw_panel(image, "VIRTUAL SCREEN")
    cv2.rectangle(image, (12, 58), (_PANEL_WIDTH - 13, 332), _BG, -1)
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

    if state.scroll is not None and t - state.scroll.t < _NOTICE_SECONDS:
        dx, dy = state.scroll.args
        length = hypot(dx, dy)
        if length:
            tip = 564 + round(dx / length * 16), 28 + round(dy / length * 16)
            cv2.arrowedLine(image, (564, 28), tip, _OPEN, 2, cv2.LINE_AA, tipLength=0.35)
    sx, sy = state.scroll_total
    _text(image, f"Scroll X {sx:+.1f}   Y {sy:+.1f}", (305, 34), _MUTED, 0.48)
    _draw_event_log(image, state)


def draw_screen(state: ScreenState, t: float) -> np.ndarray:
    """Sanal ekran panelini durumu değiştirmeden çiz."""
    image = _panel()
    _draw_screen(image, state, t)
    return image


@dataclass(frozen=True)
class _Timeline:
    total: int = 0
    start: float = 0.0
    end: float = 0.0
    markers: tuple[tuple[int, Color], ...] = ()


def _timeline_x(fraction: float) -> int:
    return _TIMELINE_LEFT + round(_unit(fraction) * (_TIMELINE_RIGHT - _TIMELINE_LEFT))


def _recording_timeline(path: Path) -> _Timeline:
    # Count nonblank frame rows without decoding every frame before playback.
    with path.open("rb") as stream:
        next(stream, None)
        first = last = None
        total = 0
        for line in stream:
            if line.strip():
                if first is None:
                    first = line
                last = line
                total += 1
    if first is None:
        return _Timeline()
    # Leave playback and its error reporting to ReplaySource.
    try:
        start = float(json.loads(first)["t"])
        end = float(json.loads(last)["t"])
    except (ValueError, TypeError, KeyError):
        return _Timeline(total)

    markers = {}
    with suppress(OSError, ValueError):
        for prompt in read_prompts(prompts_path(path)):
            if start <= prompt.t <= end:
                fraction = (prompt.t - start) / (end - start) if end > start else 0.0
                markers[_timeline_x(fraction)] = _PROMPT_COLORS[prompt.kind]
    return _Timeline(total, start, end, tuple(markers.items()))


def _draw_timeline(image: np.ndarray, t: float, index: int, timeline: _Timeline) -> None:
    if not timeline.total:
        _text(image, f"t={t:.3f}s", (_TIMELINE_LEFT, _TIMELINE_Y + 4), _MUTED, 0.48)
        return

    elapsed = t - timeline.start
    span = timeline.end - timeline.start
    fraction = elapsed / span if span > 0 else index / timeline.total
    left = _TIMELINE_LEFT, _TIMELINE_Y
    position = _timeline_x(fraction)
    cv2.line(image, left, (_TIMELINE_RIGHT, _TIMELINE_Y), _PANEL, 5, cv2.LINE_AA)
    cv2.line(image, left, (position, _TIMELINE_Y), _CURSOR, 5, cv2.LINE_AA)
    for x, color in timeline.markers:
        cv2.line(image, (x, _TIMELINE_Y - 6), (x, _TIMELINE_Y + 6), color, 2)
    _text(image, f"{index} / {timeline.total}   t={t:.3f}s", (944, _TIMELINE_Y + 4), _MUTED, 0.48)


def _badge(image: np.ndarray, label: str, x: int, y: int, color: Color) -> None:
    width, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    cv2.rectangle(image, (x, y - 18), (x + width + 18, y + 6), color, -1)
    _text(image, label, (x + 9, y), _BG, 0.52)


def _draw_frame(
    canvas: np.ndarray,
    frame: HandFrame,
    state: ScreenState,
    mode: Mode,
    label: str,
    view: _HandView,
    cfg: dict,
    index: int,
    timeline: _Timeline,
) -> None:
    """Oynatma durumunu değiştirmeden kareyi çiz."""
    cv2.rectangle(canvas, (0, 0), (_WIDTH - 1, _HEIGHT - 1), _BG, -1)
    _text(canvas, "TKT REPLAY", (24, 34), _TEXT, 0.85)
    _text(canvas, label, (220, 33), _CURSOR, 0.63)
    description = _DEMO_DESCRIPTIONS.get(label.lower())
    if description:
        _text(canvas, description, (342, 33), _TEXT, 0.53)
    _badge(canvas, mode.name, 670, 60, _MODE_COLORS[mode])
    if "left" in state.pressed:
        _badge(canvas, "LEFT DOWN", 790, 60, _CURLED)
    notice = state.notification(frame.t)
    if notice is not None:
        _badge(canvas, notice, 950, 60, _OPEN)

    mode_cfg, tap_cfg = cfg["mode"], cfg["tap"]
    panels = canvas[_PANEL_TOP : _PANEL_TOP + _PANEL_HEIGHT]
    _draw_hand(
        panels[:, 20:630],
        frame,
        (mode_cfg["curl_open"], mode_cfg["curl_closed"]),
        (tap_cfg["lift_start"], tap_cfg["lift_up"]),
        view,
    )
    _draw_screen(panels[:, 650:1260], state, frame.t)
    _draw_timeline(canvas, frame.t, index, timeline)
    _text(canvas, "Space Pause   . Step   +/- Speed   R Rewind   Q / Esc Exit   H Help", (24, 708))


def _draw_help(image: np.ndarray) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), _BG, -1)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), _MUTED, 1)
    _text(image, "REPLAY HELP", (24, 37), _TEXT, 0.77)
    for i, line in enumerate(
        (
            "Space  Pause / resume",
            ".      Step one frame while paused",
            "+ / =  Faster        -  Slower",
            "R      Rewind        Q / Esc  Exit",
            "H      Show / hide help",
        )
    ):
        _text(image, line, (24, 73 + 26 * i))
    _text(image, "MODES", (24, 220), _MUTED)
    for i, (mode, color) in enumerate(_MODE_COLORS.items()):
        _badge(image, mode.name, 24 + 105 * i, 251, color)
    _text(image, "Green = open finger / thumb", (24, 291), _OPEN)
    _text(image, "Orange = curled finger / closed thumb / left down", (24, 317), _CURLED)
    _text(image, "White tick = curl open / lift start", (24, 349), _TEXT)
    _text(image, "Yellow tick = curl closed / lift up", (24, 375), _YELLOW)
    _text(image, "Timeline beeps: tap1", (24, 416), _YELLOW)
    _text(image, "tap2", (225, 416), _PURPLE)


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
    help_visible: bool = False
    help_underlay: np.ndarray | None = field(default=None, init=False, repr=False)

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


def _draw_playback_status(canvas: np.ndarray, status: str, playback: _Playback) -> None:
    speed = "FAST" if playback.fast else f"{playback.speed:g}x"
    paused = "  PAUSED" if playback.paused else ""
    cv2.rectangle(canvas, (0, 42), (649, 73), _BG, -1)
    _text(canvas, f"{status}   Speed {speed}{paused}", (24, 62))


def _present_frame(canvas: np.ndarray, status: str, playback: _Playback, seconds: float) -> str:
    """Kareyi göster, sıradaki kareye kadar klavyeyi ve pencereyi dinle."""
    help_drawn = False
    while True:
        _draw_playback_status(canvas, status, playback)
        if playback.help_visible != help_drawn:
            region = canvas[_HELP_REGION]
            if playback.help_visible:
                if playback.help_underlay is None:
                    playback.help_underlay = np.empty_like(region)
                playback.help_underlay[:] = region
                _draw_help(region)
            else:
                region[:] = playback.help_underlay
            help_drawn = playback.help_visible
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
            if key == ord("h"):
                playback.help_visible = not playback.help_visible
                break
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


def _play_source(
    source: HandFrameSource,
    label: str,
    cfg: dict,
    playback: _Playback,
    timeline: _Timeline,
) -> str:
    backend = DryBackend()
    engine = GestureEngine(cfg, backend, source.info)
    state = ScreenState()
    hand = _HandView(aspect=source.info.height / source.info.width, thumb_closed=False)
    canvas = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    frames = source.frames()
    index = 1
    try:
        frame = next(frames, None)
        while frame is not None:
            backend.set_time(frame.t)
            engine.process(frame)
            state.update(backend.events, frame.t)
            backend.events.clear()

            hand.update(frame, cfg["thumb"])
            _draw_frame(canvas, frame, state, engine.mode, label, hand, cfg, index, timeline)
            following = next(frames, None)
            seconds = following.t - frame.t if following is not None else 1 / source.info.fps
            command = _present_frame(canvas, f"Frame {frame.frame_id}", playback, seconds)
            if command != "next":
                return command
            frame = following
            index += 1
    finally:
        engine.release()
    return "end"


def main(argv: list[str] | None = None) -> None:
    """Oynatıcıyı çağıran ana thread üzerinde çalıştır."""
    args = build_parser().parse_args(argv)
    cfg = load_control_config(args.config)
    playlist = _DEMOS if args.synthetic == "all" else (args.synthetic,)
    playback = _Playback(args.fast)
    timeline = _Timeline()
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
                if demo is None and not timeline.total:
                    timeline = _recording_timeline(args.file)
                command = _play_source(source, label, source_cfg, playback, timeline)
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
