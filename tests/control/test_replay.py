import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine, Mode
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.sources import SyntheticSource
from tkt.core.io import write_recording
from tkt.tools import replay


@pytest.fixture
def window(monkeypatch):
    show = Mock()
    wait = Mock(return_value=-1)
    visible = Mock(return_value=1)
    close = Mock()
    monkeypatch.setattr(replay.cv2, "imshow", show)
    monkeypatch.setattr(replay.cv2, "waitKey", wait)
    monkeypatch.setattr(replay.cv2, "getWindowProperty", visible)
    monkeypatch.setattr(replay.cv2, "destroyWindow", close)
    monkeypatch.setattr(replay, "platform", "win32")
    return show, wait, visible, close


def test_cocoa_enables_close_button_and_restores_window_ownership(window, monkeypatch):
    _, _, visible, close = window
    visible.return_value = 0
    native = Mock()
    native.title.return_value = replay._WINDOW
    native.styleMask.return_value = 1
    native.isReleasedWhenClosed.return_value = True
    app = Mock()
    app.windows.return_value = [native]
    appkit = SimpleNamespace(
        NSApplication=SimpleNamespace(sharedApplication=lambda: app),
        NSWindowStyleMaskClosable=2,
    )
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setattr(replay, "platform", "darwin")
    monkeypatch.setattr(replay.cv2, "currentUIFramework", lambda: "COCOA")

    replay.main(["--synthetic", "pointer", "--fast"])

    native.setStyleMask_.assert_called_once_with(3)
    assert [call.args for call in native.setReleasedWhenClosed_.call_args_list] == [
        (False,),
        (True,),
    ]
    close.assert_called_once()


def test_synthetic_window_exits_on_q(window, monkeypatch):
    show, wait, _, close = window
    wait.side_effect = [-1, ord("q")]
    source = SyntheticSource().open(1, 0.05)
    monkeypatch.setattr(replay, "_synthetic", lambda *_: source)

    replay.main(["--synthetic", "pointer", "--fast"])

    assert show.call_count == 2
    assert show.call_args.args[1].shape == (720, 1280, 3)
    assert [call.args for call in wait.call_args_list] == [(1,), (1,)]
    close.assert_called_once()
    assert list(source.frames()) == []


def test_recording_pause_step_speed_rewind_and_window_close(tmp_path, window, monkeypatch):
    _, wait, visible, close = window
    source = SyntheticSource().open(1, 0.05)
    path = tmp_path / "demo.jsonl"
    write_recording(path, source.info, source.frames())
    wait.side_effect = [ord(" "), ord("."), ord("+"), ord("r"), -1]
    visible.side_effect = [1, 1, 1, 1, -1]
    statuses = []
    draw_text = replay._text

    def capture(image, text, position, *args):
        if "Mode " in text:
            statuses.append(text)
        draw_text(image, text, position, *args)

    monkeypatch.setattr(replay, "_text", capture)
    monkeypatch.setattr(replay, "monotonic", lambda: 0)
    replay.main([str(path), "--fast", "--loop"])
    assert [line.split("Frame ")[1].split()[0] for line in statuses] == ["0", "0", "1", "1", "0"]
    assert statuses[1].endswith("FAST  PAUSED")
    assert statuses[-1].endswith("2x  PAUSED")
    close.assert_called_once()


def test_timed_playback_uses_frame_spacing_without_wall_clock(window, monkeypatch):
    _, wait, _, _ = window
    now = 0.0
    delays = []

    def wait_key(delay):
        nonlocal now
        delays.append(delay)
        now += delay / 1000
        return -1

    wait.side_effect = wait_key
    monkeypatch.setattr(replay, "monotonic", lambda: now)
    playback = replay._Playback(fast=False, speed=2)
    canvas = np.zeros((100, 700, 3), dtype=np.uint8)
    assert replay._present_frame(canvas, "RECORDING", playback, 0.125) == "next"
    assert delays == [30, 30, 3]


def test_draw_panels_for_missing_hand_and_lift():
    source = SyntheticSource(fps=10, with_lift=False).vanish(0.1).open(1, 0.1)
    missing, no_lift = source.frames()
    absent = replay.draw_hand(missing, 0.35, 0.55)
    present = replay.draw_hand(no_lift, 0.35, 0.55)
    screen = replay.draw_screen(replay.ScreenState(), missing.t)

    assert absent.shape == present.shape == screen.shape
    assert not np.array_equal(absent, present)


def test_screen_accumulates_events_in_order_and_expires_visuals():
    backend = DryBackend()
    backend.set_time(1)
    backend.move_rel(12, -5)
    backend.button("left", True)
    backend.move_rel(-2, 7)
    backend.button("left", False)
    backend.button("right", True)
    backend.button("right", False)
    backend.scroll(1.5, -3)
    backend.scroll(0.5, -2)
    backend.system_action("mission_control")

    state = replay.ScreenState(cursor=(100, 100))
    state.update(backend.events, 1)

    assert state.cursor == (110, 102)
    assert [segment[-1] for segment in state.trail] == [False, True]
    assert list(state.rings) == [(1, (112, 95), "left"), (1, (110, 102), "right")]
    assert state.pressed == set()
    assert state.scroll_total == (2, -5)
    assert state.action.args == ("mission_control",)
    previous = replay.draw_screen(state, 1)
    state.update([], 3.1)
    assert not state.trail and not state.rings
    assert state.cursor == (110, 102)
    assert not np.array_equal(previous, replay.draw_screen(state, 3.1))


@pytest.mark.parametrize("demo", ["pointer", "click", "scroll", "tap", "swipe"])
def test_synthetic_demos_exercise_the_engine(demo):
    cfg = load_control_config()
    source = replay._synthetic(demo, cfg)
    backend = DryBackend()
    engine = GestureEngine(replay._demo_config(cfg, demo), backend, source.info)
    moves = []
    coasting = False

    for frame in source.frames():
        backend.set_time(frame.t)
        start = len(backend.events)
        engine.process(frame)
        events = backend.events[start:]
        moves.extend(engine.left_down for event in events if event.kind == "move")
        coasting |= engine.mode is Mode.NONE and any(event.kind == "scroll" for event in events)

    presses = [
        (event.t, event.args[0])
        for event in backend.events
        if event.kind == "button" and event.args[1]
    ]
    if demo == "pointer":
        assert moves and not any(moves)
        assert sum(event.args[0] for event in backend.events if event.kind == "move") != 0
    elif demo == "click":
        assert [button for _, button in presses] == ["left"] * 4 + ["right"]
        assert presses[2][0] - presses[1][0] < presses[1][0] - presses[0][0]
        assert any(moves)
    elif demo == "scroll":
        assert coasting
    elif demo == "tap":
        assert [button for _, button in presses] == ["left", "right"]
    else:
        assert [event.args for event in backend.events] == [("mission_control",)]


def test_all_plays_every_demo_in_order(window, monkeypatch):
    show, _, _, _ = window
    demos = []

    def synthetic(demo, cfg):
        demos.append(demo)
        return SyntheticSource(fps=10).rest(0.1)

    monkeypatch.setattr(replay, "_synthetic", synthetic)
    replay.main(["--synthetic", "all", "--fast"])
    assert demos == ["pointer", "click", "scroll", "tap", "swipe"]
    assert show.call_count == 5


@pytest.mark.parametrize("args", [[], ["file.jsonl", "--synthetic", "tap"], ["--synthetic", "bad"]])
def test_cli_requires_exactly_one_valid_source(args):
    with pytest.raises(SystemExit, match="2"):
        replay.build_parser().parse_args(args)
