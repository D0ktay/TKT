import ctypes
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call

import pytest

from tkt.control.os_input import windows


@pytest.fixture
def user32(monkeypatch):
    def cursor(point):
        point.contents.x, point.contents.y = -100, 100
        return 1

    fake = Mock()
    fake.GetCursorPos.side_effect = cursor
    fake.GetSystemMetrics.side_effect = {76: -1920, 77: -200, 78: 3840, 79: 1280}.__getitem__
    fake.SetCursorPos.return_value = 1
    fake.SendInput.return_value = 1
    monkeypatch.setattr(windows, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=fake), raising=False)
    return fake


def _mouse_events(user32):
    events = []
    for send in user32.SendInput.call_args_list:
        count, pointer, size = send.args
        assert count == 1
        assert size == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        packet = pointer.contents
        assert packet.type == 0
        mouse = packet.data.mi
        events.append((mouse.dwFlags, ctypes.c_int32(mouse.mouseData).value))
    return events


def test_unsupported_platform(monkeypatch):
    monkeypatch.setattr(windows, "sys", SimpleNamespace(platform="darwin"))
    with pytest.raises(RuntimeError, match="yalnızca Windows"):
        windows.WindowsBackend(step_px=40)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [((25, -20), (-75, 80)), ((-3000, 2000), (-1920, 1079)), ((3000, -2000), (1919, -200))],
)
def test_move_rel(user32, delta, expected):
    windows.WindowsBackend(step_px=40).move_rel(*delta)
    assert user32.mock_calls == [
        call.GetCursorPos(ANY),
        call.GetSystemMetrics(76),
        call.GetSystemMetrics(77),
        call.GetSystemMetrics(78),
        call.GetSystemMetrics(79),
        call.SetCursorPos(*expected),
    ]


def test_button_and_release_all(user32):
    backend = windows.WindowsBackend(step_px=40)
    backend.button("left", True)
    backend.button("left", False)
    backend.button("right", True)
    backend.button("left", True)
    backend.release_all()
    backend.release_all()
    assert _mouse_events(user32) == [(flag, 0) for flag in (2, 4, 8, 2, 4, 16)]


def test_failed_button_release_can_be_retried(user32):
    backend = windows.WindowsBackend(step_px=40)
    backend.button("left", True)
    backend.button("right", True)
    user32.SendInput.side_effect = [0, 1, 1]
    with pytest.raises(ExceptionGroup, match="tuşları bırakılamadı"):
        backend.release_all()
    backend.release_all()
    assert _mouse_events(user32) == [(flag, 0) for flag in (2, 8, 16, 4, 16)]


def test_smooth_scroll(user32):
    backend = windows.WindowsBackend(step_px=40)
    for _ in range(4):
        backend.scroll(-10, 10)
    # 0.25 piksel = 0.75 birim; iki çağrıdan sonra eksen başına bir birim çıkar.
    backend.scroll(0.25, -0.25)
    assert _mouse_events(user32) == [(0x1000, -30), (0x0800, 30)] * 4
    backend.scroll(0.25, -0.25)
    assert _mouse_events(user32)[8:] == [(0x1000, 1), (0x0800, -1)]


def test_full_step_scroll_and_reset(user32):
    backend = windows.WindowsBackend(step_px=40, smooth=False)
    for _ in range(3):
        backend.scroll(10, -10)
    user32.SendInput.assert_not_called()
    backend.scroll(10, -10)
    backend.scroll(30, -30)
    backend.release_all()
    backend.scroll(10, -10)
    assert _mouse_events(user32) == [(0x1000, 120), (0x0800, -120)]


@pytest.mark.parametrize(
    ("action", "presses"),
    [
        ("mission_control", [(0x5B, 1), (0x09, 0)]),
        ("desktop_left", [(0x5B, 1), (0x11, 0), (0x25, 1)]),
        ("desktop_right", [(0x5B, 1), (0x11, 0), (0x27, 1)]),
    ],
)
def test_system_actions(user32, action, presses):
    backend = windows.WindowsBackend(step_px=40)
    backend.system_action(action)
    backend.release_all()
    packets = [send.args[1].contents for send in user32.SendInput.call_args_list]
    assert all(packet.type == 1 for packet in packets)
    assert [(packet.data.ki.wVk, packet.data.ki.dwFlags) for packet in packets] == presses + [
        (key, flags | 2) for key, flags in reversed(presses)
    ]


def test_system_action_releases_prior_keys_if_press_fails(user32):
    backend = windows.WindowsBackend(step_px=40)
    user32.SendInput.side_effect = [1, 0, 1]
    with pytest.raises(OSError, match="SendInput"):
        backend.system_action("mission_control")
    backend.release_all()
    keys = [send.args[1].contents.data.ki for send in user32.SendInput.call_args_list]
    assert [(key.wVk, key.dwFlags) for key in keys] == [(0x5B, 1), (0x09, 0), (0x5B, 3)]
