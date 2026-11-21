import csv
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

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
        (["--backend", "native"], "native backend yalnızca Windows ve macOS'ta var"),
    ],
)
def test_argument_errors(args, message, capsys, monkeypatch):
    monkeypatch.setattr(app, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(SystemExit) as error:
        app.main(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_windows_native_backend_selection(monkeypatch):
    source = SimpleNamespace(info=None, frames=lambda: iter([make_frame()]), close=Mock())
    engine = Mock()
    calls = Mock()
    keyboard = SimpleNamespace(GlobalHotKeys=calls.hotkeys)

    class FakeWindowsBackend:
        def __init__(self, *, step_px, smooth):
            calls.backend_init(step_px, smooth)

        release_all = calls.release_all

    monkeypatch.setattr(app, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(app, "_set_dpi_awareness", calls.dpi)
    monkeypatch.setattr(app, "_open_source", lambda _: source)
    monkeypatch.setattr(app, "WindowsBackend", FakeWindowsBackend)
    monkeypatch.setattr(app, "GestureEngine", lambda *_: engine)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)

    app.main(["--backend", "native", "--source", "synthetic"])

    cfg = load_control_config()
    assert calls.mock_calls[:2] == [call.dpi(), call.backend_init(cfg["scroll"]["step_px"], True)]
    calls.hotkeys.assert_called_once()
    calls.hotkeys.return_value.start.assert_called_once()
    calls.hotkeys.return_value.stop.assert_called_once()
    engine.process.assert_called_once()
    calls.release_all.assert_called_once()
    source.close.assert_called_once()


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


@pytest.fixture
def latency_run(monkeypatch, tmp_path):
    clock = SimpleNamespace(wall=100.0, perf=0.0)
    frames = [make_frame(frame_id=i, t=t) for i, t in enumerate((100.99, 102, 104.95, 105.48))]
    source = SimpleNamespace(info=None, frames=lambda: iter(frames), close=Mock())

    class Engine:
        def __init__(self, cfg, backend, info):
            self.backend = backend

        def process(self, frame):
            clock.wall = (101.0, 102.0, 105.0, 105.5)[frame.frame_id]
            clock.perf += (0.001, 0.003, 0.002, 0.001)[frame.frame_id]
            if frame.frame_id == 0:
                self.backend.move_rel(1, 2)
                clock.wall += 0.001
                self.backend.button("left", True)
            elif frame.frame_id == 2:
                self.backend.scroll(1, 2)
            elif frame.frame_id == 3:
                self.backend.system_action("mission_control")

    monkeypatch.setattr(app, "monotonic", lambda: clock.wall)
    monkeypatch.setattr(app, "perf_counter", lambda: clock.perf)
    monkeypatch.setattr(app, "GestureEngine", Engine)
    monkeypatch.setattr(app, "_open_source", lambda _: source)

    def run(source_name, *, csv_output=True):
        output = tmp_path / "latency.csv"
        args = ["--source", source_name, "--backend", "dry"]
        if source_name == "replay":
            args += ["--file", "test.jsonl"]
        args += ["--latency-out", str(output)] if csv_output else ["--latency"]
        app.main(args)
        if csv_output:
            with output.open(newline="") as stream:
                return list(csv.DictReader(stream))

    return run


def test_live_latency_reports_and_csv(latency_run, capsys):
    rows = latency_run("live")
    assert list(rows[0]) == ["frame_id", "t", "sent_t", "latency_ms", "process_ms"]
    assert [row["frame_id"] for row in rows] == ["0", "1", "2", "3"]
    assert [float(row["t"]) for row in rows] == [100.99, 102, 104.95, 105.48]
    assert [row["sent_t"] for row in rows] == ["101.0", "", "105.0", "105.5"]
    assert rows[1]["latency_ms"] == ""
    assert [float(row["latency_ms"]) for row in rows if row["latency_ms"]] == pytest.approx(
        [10, 50, 20]
    )
    assert [float(row["process_ms"]) for row in rows] == pytest.approx([1, 3, 2, 1])
    periodic, final = capsys.readouterr().out.splitlines()
    assert "(ara) p50=30.000 p95=48.000 max=50.000 ms (hedef p95 < 60 ms)" in periodic
    assert "olaylı kare=2 | kontrol p50=2.000 p95=2.900 ms (hedef < 2 ms) | 0.60 fps" in periodic
    assert "(çıkış) p50=20.000 p95=20.000 max=20.000 ms" in final
    assert "olaylı kare=1 | kontrol p50=1.000 p95=1.000 ms (hedef < 2 ms) | 2.00 fps" in final


@pytest.mark.parametrize("source", ["replay", "synthetic"])
def test_offline_latency_is_invalid(latency_run, capsys, source):
    rows = latency_run(source)
    assert all(row["latency_ms"] == "" for row in rows)
    assert rows[0]["sent_t"] == "101.0"
    assert float(rows[0]["process_ms"]) == pytest.approx(1)
    periodic, final = capsys.readouterr().out.splitlines()
    assert "(ara) geçersiz" in periodic
    assert "(çıkış) geçersiz" in final
    assert "olaylı kare=2 | kontrol p50=2.000 p95=2.900 ms" in periodic


def test_latency_without_csv(latency_run, capsys):
    latency_run("synthetic", csv_output=False)
    assert "Gecikme (çıkış) geçersiz" in capsys.readouterr().out


def test_mac_native_backend_selection(monkeypatch):
    source = SimpleNamespace(info=None, frames=lambda: iter([make_frame()]), close=Mock())
    engine = Mock()
    calls = Mock()
    keyboard = SimpleNamespace(GlobalHotKeys=calls.hotkeys)

    class FakeMacBackend:
        def __init__(self):
            calls.backend_init()

        release_all = calls.release_all

    monkeypatch.setattr(app, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(app, "_set_dpi_awareness", calls.dpi)
    monkeypatch.setattr(app, "_open_source", lambda _: source)
    monkeypatch.setattr(app, "MacBackend", FakeMacBackend)
    monkeypatch.setattr(app, "GestureEngine", lambda *_: engine)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)

    app.main(["--backend", "native", "--source", "synthetic"])

    calls.backend_init.assert_called_once()
    calls.hotkeys.return_value.start.assert_called_once()
    engine.process.assert_called_once()
    calls.release_all.assert_called_once()
