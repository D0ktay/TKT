import importlib
import sys
from types import SimpleNamespace

import pytest

from tkt.control.os_input.macos import MacBackend


class FakeQuartz:
    """Gerçek Quartz'ın kullanılan kısmını taklit eder; olayları kaydeder."""

    kCGMouseButtonLeft, kCGMouseButtonRight = 0, 1
    kCGEventLeftMouseDown, kCGEventLeftMouseUp = 1, 2
    kCGEventRightMouseDown, kCGEventRightMouseUp = 3, 4
    kCGEventMouseMoved = 5
    kCGEventLeftMouseDragged, kCGEventRightMouseDragged = 6, 7
    kCGMouseEventClickState, kCGMouseEventDeltaX, kCGMouseEventDeltaY = 1, 4, 5
    kCGScrollEventUnitPixel = 0
    kCGHIDEventTap = 0
    kCGEventFlagMaskControl = 0x40000

    def __init__(self, pos=(100.0, 100.0), displays=((0, 0, 1000, 800),)):
        self.pos = pos
        self.displays = list(displays)
        self.posted = []

    def CGEventCreate(self, source):
        return SimpleNamespace(kind="probe", pos=self.pos, fields={})

    def CGEventGetLocation(self, event):
        return SimpleNamespace(x=self.pos[0], y=self.pos[1])

    def CGEventCreateMouseEvent(self, source, kind, pos, button):
        return SimpleNamespace(kind=kind, pos=tuple(pos), button=button, fields={})

    def CGEventSetIntegerValueField(self, event, field, value):
        event.fields[field] = value

    def CGEventCreateScrollWheelEvent(self, source, unit, count, *wheels):
        return SimpleNamespace(kind="scroll", unit=unit, count=count, wheels=wheels, fields={})

    def CGEventCreateKeyboardEvent(self, source, keycode, down):
        return SimpleNamespace(kind="key", keycode=keycode, down=down, flags=0, fields={})

    def CGEventSetFlags(self, event, flags):
        event.flags = flags

    def CGEventPost(self, tap, event):
        assert tap == self.kCGHIDEventTap
        self.posted.append(event)
        if hasattr(event, "pos") and event.kind != "probe":
            self.pos = event.pos

    def CGGetActiveDisplayList(self, maximum, ids, count):
        return 0, tuple(range(len(self.displays))), len(self.displays)

    def CGDisplayBounds(self, display):
        x, y, w, h = self.displays[display]
        return SimpleNamespace(
            origin=SimpleNamespace(x=x, y=y), size=SimpleNamespace(width=w, height=h)
        )


@pytest.fixture
def quartz(monkeypatch):
    fake = FakeQuartz()
    monkeypatch.setitem(sys.modules, "Quartz", fake)
    monkeypatch.setitem(sys.modules, "AppKit", None)  # doubleClickInterval yok sayılır
    return fake


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


def test_import_without_quartz_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "Quartz", None)
    with pytest.raises(RuntimeError, match="Quartz"):
        MacBackend()


def test_module_imports_without_quartz(monkeypatch):
    monkeypatch.setitem(sys.modules, "Quartz", None)
    importlib.reload(sys.modules[MacBackend.__module__])


def test_move_is_mouse_moved_with_delta_fields(quartz, clock):
    backend = MacBackend(clock=clock)
    backend.move_rel(5, -3)
    (event,) = quartz.posted
    assert event.kind == quartz.kCGEventMouseMoved
    assert event.pos == (105.0, 97.0)
    assert event.fields[quartz.kCGMouseEventDeltaX] == 5
    assert event.fields[quartz.kCGMouseEventDeltaY] == -3


def test_move_while_pressed_is_dragged(quartz, clock):
    backend = MacBackend(clock=clock)
    backend.button("left", True)
    backend.move_rel(1, 0)
    backend.button("left", False)
    backend.button("right", True)
    backend.move_rel(1, 0)
    kinds = [e.kind for e in quartz.posted]
    assert kinds == [
        quartz.kCGEventLeftMouseDown,
        quartz.kCGEventLeftMouseDragged,
        quartz.kCGEventLeftMouseUp,
        quartz.kCGEventRightMouseDown,
        quartz.kCGEventRightMouseDragged,
    ]
    assert quartz.posted[-1].button == quartz.kCGMouseButtonRight


def test_move_is_clamped_to_display(quartz, clock):
    backend = MacBackend(clock=clock)
    backend.move_rel(-500, -500)
    assert quartz.posted[-1].pos == (0.0, 0.0)
    assert quartz.posted[-1].fields[quartz.kCGMouseEventDeltaX] == -100
    backend.move_rel(5000, 5000)
    assert quartz.posted[-1].pos == (999.0, 799.0)


def test_clamp_uses_nearest_display_across_multiple_displays(quartz, clock):
    quartz.displays = [(0, 0, 1000, 800), (1000, 0, 500, 400)]
    quartz.pos = (990.0, 100.0)
    backend = MacBackend(clock=clock)
    backend.move_rel(300, 0)  # ikinci ekrana geçer
    assert quartz.posted[-1].pos == (1290.0, 100.0)
    backend.move_rel(0, 500)  # ikinci ekran 400 yüksekliğinde; en yakın noktaya kırp
    assert quartz.posted[-1].pos == (1290.0, 399.0)


def test_double_click_sets_click_state(quartz, clock):
    backend = MacBackend(clock=clock, double_click_interval=0.5)
    states = []
    for _ in range(3):
        backend.button("left", True)
        backend.button("left", False)
        clock.t += 0.2
    for event in quartz.posted:
        states.append(event.fields[quartz.kCGMouseEventClickState])
    assert states == [1, 1, 2, 2, 3, 3]
    backend.button("left", True)  # dördüncü basış yeniden 1'den başlar
    assert quartz.posted[-1].fields[quartz.kCGMouseEventClickState] == 1


def test_click_state_resets_after_interval_or_distance_or_other_button(quartz, clock):
    backend = MacBackend(clock=clock, double_click_interval=0.5)
    backend.button("left", True)
    backend.button("left", False)
    clock.t = 0.6  # aralık dışı
    backend.button("left", True)
    assert quartz.posted[-1].fields[quartz.kCGMouseEventClickState] == 1
    backend.button("left", False)
    clock.t = 0.7
    backend.move_rel(50, 0)  # uzaklık dışı
    backend.button("left", True)
    assert quartz.posted[-1].fields[quartz.kCGMouseEventClickState] == 1
    backend.button("left", False)
    clock.t = 0.8
    backend.button("right", True)  # farklı tuş kendi sayacını tutar
    assert quartz.posted[-1].fields[quartz.kCGMouseEventClickState] == 1


def test_default_interval_is_half_second_when_appkit_missing(quartz, clock):
    backend = MacBackend(clock=clock)
    backend.button("left", True)
    backend.button("left", False)
    clock.t = 0.4
    backend.button("left", True)
    assert quartz.posted[-1].fields[quartz.kCGMouseEventClickState] == 2


def test_duplicate_down_and_stray_up_are_ignored(quartz, clock):
    backend = MacBackend(clock=clock)
    backend.button("left", False)
    backend.button("left", True)
    backend.button("left", True)
    assert [e.kind for e in quartz.posted] == [quartz.kCGEventLeftMouseDown]


def test_scroll_accumulates_fractions_in_pixels(quartz):
    backend = MacBackend()
    backend.scroll(0, 0.6)
    assert quartz.posted == []
    assert backend.events_sent == 0
    backend.scroll(0.5, 0.6)  # y: 1.2 -> 1, x: 0.5 -> 0
    (event,) = quartz.posted
    assert event.unit == quartz.kCGScrollEventUnitPixel
    assert event.count == 2
    assert event.wheels == (1, 0)  # İçerik aşağı.
    backend.scroll(0.5, -2.2)  # y: 0.2 - 2.2 = -2.0 ; x: 1.0
    assert quartz.posted[-1].wheels == (-2, 1)  # İçerik yukarı ve sağa.
    assert backend.events_sent == 2


def test_scroll_rejects_non_finite(quartz):
    with pytest.raises(ValueError):
        MacBackend().scroll(float("nan"), 0)


@pytest.mark.parametrize(
    ("name", "keycode"),
    [("mission_control", 126), ("desktop_left", 123), ("desktop_right", 124)],
)
def test_system_action_sends_ctrl_chord(quartz, name, keycode):
    MacBackend().system_action(name)
    down, up = quartz.posted
    assert (down.keycode, down.down, down.flags) == (keycode, True, quartz.kCGEventFlagMaskControl)
    assert (up.keycode, up.down, up.flags) == (keycode, False, quartz.kCGEventFlagMaskControl)


def test_unknown_system_action_raises(quartz):
    with pytest.raises(ValueError):
        MacBackend().system_action("nope")
    assert quartz.posted == []


def test_release_all_releases_held_buttons_and_clears_scroll(quartz, clock):
    backend = MacBackend(clock=clock)
    backend.button("left", True)
    backend.button("right", True)
    backend.scroll(0, 0.7)
    backend.release_all()
    assert [e.kind for e in quartz.posted[2:]] == [
        quartz.kCGEventLeftMouseUp,
        quartz.kCGEventRightMouseUp,
    ]
    backend.release_all()
    assert len(quartz.posted) == 4
    backend.scroll(0, 0.5)  # artık birikmiş kesir yok
    assert len(quartz.posted) == 4
    backend.move_rel(1, 0)
    assert quartz.posted[-1].kind == quartz.kCGEventMouseMoved


def test_failed_mouse_up_stays_pressed_for_release_retry(monkeypatch, quartz):
    backend = MacBackend()
    backend.button("left", True)
    backend.button("right", True)
    post = quartz.CGEventPost

    def fail_left_up(tap, event):
        if event.kind == quartz.kCGEventLeftMouseUp:
            raise OSError("Quartz mouse-up failed")
        post(tap, event)

    monkeypatch.setattr(quartz, "CGEventPost", fail_left_up)
    with pytest.raises(ExceptionGroup, match="Fare tuşları bırakılamadı"):
        backend.release_all()
    monkeypatch.setattr(quartz, "CGEventPost", post)
    backend.release_all()
    backend.release_all()
    assert [e.kind for e in quartz.posted[2:]] == [
        quartz.kCGEventRightMouseUp,
        quartz.kCGEventLeftMouseUp,
    ]


@pytest.mark.parametrize("name", ["dictation_start", "dictation_stop"])
def test_dictation_double_taps_control(quartz, monkeypatch, name):
    monkeypatch.setattr("tkt.control.os_input.macos.time.sleep", lambda s: None)
    MacBackend().system_action(name)
    assert [(e.keycode, e.down) for e in quartz.posted] == [(59, True), (59, False)] * 2
    assert [e.flags for e in quartz.posted] == [quartz.kCGEventFlagMaskControl, 0] * 2
