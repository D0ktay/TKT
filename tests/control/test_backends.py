import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tkt.app.config import load_control_config
from tkt.control.os_input.dry_backend import DryBackend, Event
from tkt.control.os_input.pynput_backend import PynputBackend
from tkt.control.scroll import ScrollMotion
from tkt.control.sources import SyntheticSource


def test_dry_backend_records_time_and_releases_pressed_buttons():
    backend = DryBackend()
    backend.set_time(10)
    backend.move_rel(1, -2)
    backend.button("left", True)
    backend.button("right", True)
    backend.button("right", False)
    backend.set_time(11)
    backend.scroll(1.5, 0)
    backend.system_action("mission_control")
    backend.release_all()
    backend.release_all()
    assert backend.events == [
        Event(10, "move", (1, -2)),
        Event(10, "button", ("left", True)),
        Event(10, "button", ("right", True)),
        Event(10, "button", ("right", False)),
        Event(11, "scroll", (1.5, 0)),
        Event(11, "system", ("mission_control",)),
        Event(11, "button", ("left", False)),
    ]


@pytest.fixture
def pynput_backend(monkeypatch):
    mouse, keyboard = Mock(), Mock()
    keys = SimpleNamespace(**{name: name for name in ("cmd", "ctrl", "tab", "up", "left", "right")})
    monkeypatch.setitem(
        sys.modules,
        "pynput",
        SimpleNamespace(
            mouse=SimpleNamespace(Controller=lambda: mouse, Button=keys),
            keyboard=SimpleNamespace(Controller=lambda: keyboard, Key=keys),
        ),
    )
    backend = PynputBackend()
    backend.step_px = 40
    return backend, mouse, keyboard


def test_pynput_mouse(pynput_backend, monkeypatch):
    backend, mouse, _ = pynput_backend
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="darwin")
    )
    backend.move_rel(2, -3)
    backend.button("left", True)
    backend.button("right", True)
    backend.scroll(25, -39)
    assert backend.events_sent == 3
    backend.scroll(16, -42)
    backend.scroll(39, 1)
    backend.release_all()
    backend.release_all()
    assert mouse.mock_calls == [
        call.move(2, -3),
        call.press("left"),
        call.press("right"),
        call.scroll(1, -2),
        call.scroll(1, 0),
        call.release("left"),
        call.release("right"),
    ]
    assert backend.events_sent == 7


@pytest.mark.parametrize(
    ("platform", "action", "keys"),
    [
        ("darwin", "mission_control", ("ctrl", "up")),
        ("darwin", "desktop_left", ("ctrl", "left")),
        ("darwin", "desktop_right", ("ctrl", "right")),
        ("win32", "mission_control", ("cmd", "tab")),
        ("win32", "desktop_left", ("cmd", "ctrl", "left")),
        ("win32", "desktop_right", ("cmd", "ctrl", "right")),
    ],
)
def test_pynput_system_shortcuts(pynput_backend, monkeypatch, platform, action, keys):
    backend, _, keyboard = pynput_backend
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform=platform)
    )
    backend.system_action(action)
    assert keyboard.mock_calls == [call.press(key) for key in keys] + [
        call.release(key) for key in reversed(keys)
    ]


@pytest.mark.parametrize(("platform", "horizontal"), [("win32", -1), ("darwin", 1)])
def test_pynput_scroll_content_direction(pynput_backend, monkeypatch, platform, horizontal):
    backend, mouse, _ = pynput_backend
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform=platform)
    )
    backend.scroll(40, -40)
    mouse.scroll.assert_called_once_with(horizontal, -1)


def test_natural_scroll_forward_moves_content_up(pynput_backend, monkeypatch):
    backend, mouse, _ = pynput_backend
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="win32")
    )
    cfg = load_control_config()["scroll"] | {"natural": True}
    motion = ScrollMotion(cfg, SyntheticSource().info)
    motion.update((0.5, 0.5), 0.2, 0.0)
    dx, dy = motion.update((0.5, 0.4), 0.2, 0.03)
    backend.scroll(dx, dy)
    mouse.scroll.assert_called_once()
    horizontal, vertical = mouse.scroll.call_args.args
    assert horizontal == 0
    assert vertical < 0


def test_pynput_failed_mouse_up_can_be_retried(pynput_backend):
    backend, mouse, _ = pynput_backend
    backend.button("left", True)
    backend.button("right", True)
    mouse.release.side_effect = [OSError("mouse-up failed"), None, None]
    with pytest.raises(ExceptionGroup, match="Fare tuşları bırakılamadı"):
        backend.release_all()
    backend.release_all()
    backend.release_all()
    assert mouse.release.call_args_list == [call("left"), call("right"), call("left")]


@pytest.mark.parametrize("action", ["dictation_start", "dictation_stop"])
def test_pynput_dictation_shortcuts(pynput_backend, monkeypatch, action):
    backend, _, keyboard = pynput_backend
    monkeypatch.setattr("tkt.control.os_input.pynput_backend.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="darwin"))
    posted = []
    fake_q = SimpleNamespace(
        kCGEventSourceStateHIDSystemState=1, kCGEventFlagsChanged=12,
        kCGKeyboardEventKeycode=9, kCGEventFlagMaskControl=0x40000, kCGHIDEventTap=0,
        CGEventSourceCreate=lambda state: "src",
        CGEventCreate=lambda src: SimpleNamespace(fields={}),
        CGEventSetType=lambda e, k: setattr(e, "kind", k),
        CGEventSetIntegerValueField=lambda e, f, v: e.fields.__setitem__(f, v),
        CGEventSetFlags=lambda e, fl: setattr(e, "flags", fl),
        CGEventPost=lambda tap, e: posted.append(e),
    )
    monkeypatch.setitem(sys.modules, "Quartz", fake_q)
    backend.system_action(action)
    assert keyboard.mock_calls == []          # macOS'ta pynput klavyesi kullanılmaz
    assert [(e.kind, e.fields[9], e.flags) for e in posted] == [
        (12, 59, 0x40001), (12, 59, 0)] * 2
    assert backend.events_sent == 4
    backend.events_sent = 0
    keyboard.reset_mock()
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="win32"))
    backend.system_action(action)
    assert keyboard.mock_calls == [call.press("cmd"), call.press("h"),
                                   call.release("h"), call.release("cmd")]
