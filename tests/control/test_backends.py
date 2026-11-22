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
        ("darwin", "key_left", ("left",)),
        ("win32", "key_right", ("right",)),
        ("win32", "key_tab", ("tab",)),
        ("darwin", "key_c", ("c",)),
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


@pytest.mark.parametrize(
    ("action", "start"), [("dictation_start", True), ("dictation_stop", False)])
def test_pynput_dictation_shortcuts(pynput_backend, monkeypatch, action, start):
    from tkt.control.os_input import mac_dictation

    backend, _, keyboard = pynput_backend
    seen = []
    monkeypatch.setattr(mac_dictation, "toggle", lambda s: seen.append(s))
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="darwin"))
    backend.system_action(action)
    assert seen == [start] and keyboard.mock_calls == []   # macOS: menüden, tuş yok
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="win32"))
    backend.system_action(action)
    assert keyboard.mock_calls == [call.press("cmd"), call.press("h"),
                                   call.release("h"), call.release("cmd")]


def test_mac_dictation_toggle_runs_osascript_without_blocking():
    from tkt.control.os_input import mac_dictation

    calls = []
    mac_dictation.toggle(True, popen=lambda args, **kw: calls.append((args, kw)))
    mac_dictation.toggle(False, popen=lambda args, **kw: calls.append((args, kw)))
    (a1, kw1), (a2, _) = calls
    assert a1[:2] == ["osascript", "-e"] and a1[3] == "start" and a2[3] == "stop"
    assert "Dikte" in a1[2] and "Dictation" in a1[2]
    assert kw1["stdout"] is not None and kw1["stderr"] is not None


def test_mac_dictation_stop_passes_mic_state():
    # Electron uygulamalarında menü adı hep "Start Dictation…" kalır; durdururken betik
    # mikrofon kullanımdaysa Esc gönderir. Açarken mikrofona bakılmaz.
    from tkt.control.os_input import mac_dictation

    calls, asked = [], []

    def mic(state):
        return lambda: asked.append(state) or state

    def popen(args, **kw):
        calls.append(args)

    mac_dictation.toggle(False, popen=popen, mic_active=mic(True))
    mac_dictation.toggle(False, popen=popen, mic_active=mic(False))
    mac_dictation.toggle(True, popen=popen, mic_active=mic(True))
    assert [c[3:] for c in calls] == [["stop", "active"], ["stop", "idle"], ["start", "idle"]]
    assert asked == [True, False]
    script = calls[0][2]
    assert '"Edit"' in script and "Stop" in script and "Cancel" in script
    assert "key code 53" in script


def test_mic_in_use_returns_bool():
    from tkt.control.os_input import mac_dictation

    assert mac_dictation.mic_in_use() in (True, False)


def test_dictation_submit_stops_then_waits_then_enter():
    from tkt.control.os_input import dictation

    log = []

    class Proc:
        def wait(self, timeout):
            log.append(("wait", timeout))

    dictation.submit_later(lambda: log.append("stop") or Proc(), lambda: log.append("enter"),
                           delay=0.6, background=False, sleep=lambda s: log.append(("sleep", s)))
    assert log == ["stop", ("wait", dictation.STOP_WAIT_S), ("sleep", 0.6), "enter"]


def test_pynput_dictation_submit_presses_enter_after_stop(pynput_backend, monkeypatch):
    from tkt.control.os_input import dictation, mac_dictation

    backend, _, keyboard = pynput_backend
    monkeypatch.setattr(backend._keys, "enter", "enter", raising=False)
    seen = []
    monkeypatch.setattr(mac_dictation, "toggle", lambda s: seen.append(s))
    monkeypatch.setattr(dictation, "submit_later",
                        lambda stop, enter: (stop(), enter()))
    monkeypatch.setattr(
        "tkt.control.os_input.pynput_backend.sys", SimpleNamespace(platform="darwin"))
    backend.system_action("dictation_submit")
    assert seen == [False]
    assert keyboard.mock_calls == [call.press("enter"), call.release("enter")]
