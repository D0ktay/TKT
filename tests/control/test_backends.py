import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tkt.control.os_input.dry_backend import DryBackend, Event
from tkt.control.os_input.pynput_backend import PynputBackend


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


def test_pynput_mouse(pynput_backend):
    backend, mouse, _ = pynput_backend
    backend.move_rel(2, -3)
    backend.button("left", True)
    backend.button("right", True)
    backend.scroll(25, -39)
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
