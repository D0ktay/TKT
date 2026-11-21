import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.control.conftest import make_frame

from tkt.app import main as app
from tkt.app.config import load_control_config
from tkt.control.os_input.dry_backend import DryBackend


def test_load_control_config(tmp_path):
    config = tmp_path / "control.toml"
    config.write_text("[general]\nlost_release_ms = 300\n", encoding="utf-8")
    assert load_control_config(config) == {"general": {"lost_release_ms": 300}}


@pytest.mark.parametrize(
    "args",
    [
        ["--source", "synthetic"],
        ["--source", "replay", "--file", "hand.jsonl", "--fast", "--loop"],
    ],
)
def test_offline_defaults_to_dry_and_releases_on_error(monkeypatch, args):
    frame = make_frame(t=12.5)
    source = SimpleNamespace(info=object(), frames=lambda: iter([frame]), close=Mock())
    made = []

    def replay(path, *, fast, loop):
        assert path.name == "hand.jsonl"
        assert fast and loop
        return source

    class Engine:
        def __init__(self, cfg, backend, info):
            assert isinstance(backend, DryBackend)
            made.append(backend)
            self.backend = backend

        def process(self, frame):
            self.backend.button("left", True)
            raise RuntimeError("process failed")

    if args[1] == "replay":
        monkeypatch.setattr(app, "ReplaySource", replay)
    else:
        monkeypatch.setattr(app, "_open_source", lambda _: source)
    monkeypatch.setattr(app, "GestureEngine", Engine)
    with pytest.raises(RuntimeError, match="process failed"):
        app.main(args)
    assert [(event.t, event.args) for event in made[0].events] == [
        (12.5, ("left", True)),
        (12.5, ("left", False)),
    ]
    source.close.assert_called_once_with()


def test_watchdog_releases_once_during_a_blocking_read(monkeypatch):
    released = Event()
    engine = Mock()
    engine.release.side_effect = released.set

    def frames():
        yield make_frame()
        assert released.wait(2)

    source = SimpleNamespace(info=None, frames=frames)
    monkeypatch.setattr(app, "GestureEngine", lambda *_: engine)
    app._run(source, DryBackend(), {"general": {"lost_release_ms": 20}})
    engine.process.assert_called_once()
    engine.release.assert_called_once()


def test_live_defaults_to_pynput_and_hotkey_toggles_output(monkeypatch):
    shortcut = "<ctrl>+<alt>+p"
    processed = []
    engine = Mock()
    engine.process.side_effect = lambda frame: processed.append(frame.frame_id)
    backend = Mock()

    class FakeBackend:
        release_all = backend.release_all

    class FakeHotKeys:
        def __init__(self, callbacks):
            self.toggle = callbacks[shortcut]
            listeners.append(self)

        start = Mock()
        stop = Mock()

    listeners = []

    def frames():
        yield make_frame(frame_id=1)
        listeners[0].toggle()
        yield make_frame(frame_id=2)
        listeners[0].toggle()
        yield make_frame(frame_id=3)

    close = Mock()
    live_source = Mock(return_value=SimpleNamespace(info=None, frames=frames, close=close))
    monkeypatch.setitem(
        sys.modules, "tkt.vision.live_source", SimpleNamespace(open_live_source=live_source)
    )
    monkeypatch.setitem(sys.modules, "pynput.keyboard", SimpleNamespace(GlobalHotKeys=FakeHotKeys))
    monkeypatch.setattr(app, "PynputBackend", FakeBackend)
    monkeypatch.setattr(app, "GestureEngine", lambda *_: engine)
    app.main([])
    live_source.assert_called_once_with()
    assert processed == [1, 3]
    engine.release.assert_called_once()
    listeners[0].start.assert_called_once()
    listeners[0].stop.assert_called_once()
    backend.release_all.assert_called_once()
    close.assert_called_once()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--source", "replay"], "--file gerekli"),
        (["--backend", "native"], "native backend henüz desteklenmiyor"),
    ],
)
def test_argument_errors(args, message, capsys):
    with pytest.raises(SystemExit) as error:
        app.main(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_windows_dpi_awareness(monkeypatch):
    user32 = Mock()
    user32.SetProcessDpiAwarenessContext.return_value = False
    monkeypatch.setattr(app, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(
        sys.modules,
        "ctypes",
        SimpleNamespace(
            windll=SimpleNamespace(user32=user32),
            c_void_p=lambda value: value,
        ),
    )
    app._set_dpi_awareness()
    user32.SetProcessDpiAwarenessContext.assert_called_once_with(-4)
    user32.SetProcessDPIAware.assert_called_once_with()
