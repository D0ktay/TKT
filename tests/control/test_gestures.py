from unittest.mock import Mock, call

import pytest
from tests.control.conftest import make_frame

from tkt.app.config import load_control_config
from tkt.control import gestures
from tkt.control.gestures import GestureEngine, Mode
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.sources import SyntheticSource
from tkt.core.types import FINGERS, SourceInfo


def backend_events(backend):
    return [(event.kind, *event.args) for event in backend.events]


@pytest.fixture
def rig(monkeypatch, request):
    cfg = {
        "general": {"confidence_min": 0.75, "lost_release_ms": 250},
        "mode": {"lock_ms": 125, "curl_open": 0.25, "curl_closed": 0.625, "freeze_curl_rate": 4},
        "thumb": {
            "closed": 0.25,
            "open": 0.5,
            "debounce_ms": 31.25,
            "drag_threshold": 0.05,
            "still_ms": 125,
        },
        "pointer": {"rollback_ms": 80},
        "scroll": {},
    }
    if hasattr(request, "param"):
        cfg["general"]["click_method"] = request.param
        cfg["tap"] = load_control_config()["tap"]
    info = SourceInfo(schema=1, width=640, height=512, fps=60, hand="right")
    backend = DryBackend()
    monkeypatch.setattr(backend, "release_all", Mock(wraps=backend.release_all))
    pointer = Mock(spec=gestures.PointerMotion)
    pointer.update.return_value = (0, 0)
    pointer.rollback.return_value = (0, 0)
    pointer_factory = Mock(return_value=pointer)
    scroll = Mock(spec=gestures.ScrollMotion)
    scroll.update.return_value = (0, 0)
    scroll.tick.return_value = (0, 0)
    scroll_factory = Mock(return_value=scroll)
    monkeypatch.setattr(gestures, "PointerMotion", pointer_factory)
    monkeypatch.setattr(gestures, "ScrollMotion", scroll_factory)
    engine = GestureEngine(cfg, backend, info)
    pointer_factory.assert_called_once_with(cfg["pointer"], info)
    scroll_factory.assert_called_once_with(cfg["scroll"], info)
    return engine, backend, pointer, scroll


def pose(t, *opened, **kwargs):
    curls = {name: 0.0 if name in opened else 1.0 for name in FINGERS}
    curls.update(kwargs.pop("curls", {}))
    return make_frame(t=t, curls=curls, **kwargs)


def lock_pointer(engine):
    engine.process(pose(0, "index"))
    engine.process(pose(0.125, "index"))
    assert engine.mode is Mode.POINTER


def test_opening_four_fingers_does_not_lock_intermediate_mode(rig):
    engine, backend, pointer, _ = rig
    engine.process(pose(0))
    assert engine.mode is Mode.NONE

    for count in range(1, 5):
        engine.process(pose(count / 32, *FINGERS[:count]))
        assert engine.mode is Mode.WAIT
    engine.process(pose(0.21875, *FINGERS))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.25, *FINGERS))
    assert engine.mode is Mode.SWIPE
    assert backend_events(backend) == []
    assert pointer.mock_calls == []


def test_locked_mode_changes_only_after_all_fingers_close(rig):
    engine, backend, pointer, _ = rig
    engine.process(pose(0, "index", "middle"))
    engine.process(pose(0.125, "index", "middle"))
    assert engine.mode is Mode.SCROLL

    engine.process(pose(0.25, "index", "middle", "ring"))
    engine.process(pose(0.5, "middle"))
    assert engine.mode is Mode.SCROLL
    engine.process(pose(0.75))
    assert engine.mode is Mode.NONE
    engine.process(pose(1, "index", "middle", "ring"))
    engine.process(pose(1.125, "index", "middle", "ring"))
    assert engine.mode is Mode.SWIPE
    assert backend_events(backend) == []
    assert pointer.mock_calls == []


def test_wait_requires_the_same_fingers_to_stay_open(rig):
    engine, _, _, _ = rig
    engine.process(pose(0, "middle"))
    engine.process(pose(0.125, "middle"))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.25, "index"))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.375, "index"))
    assert engine.mode is Mode.POINTER


def test_curl_hysteresis_preserves_state_at_both_thresholds(rig):
    engine, _, _, _ = rig
    engine.process(pose(0, curls={"index": 0.25}))
    assert engine.mode is Mode.NONE
    engine.process(pose(0.125, curls={"index": 0.125}))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.25, curls={"index": 0.625}))
    assert engine.mode is Mode.POINTER
    engine.process(pose(0.5, curls={"index": 0.25}))
    assert engine.mode is Mode.POINTER
    engine.process(pose(0.75, curls={"index": 0.75}))
    assert engine.mode is Mode.NONE
    engine.process(pose(1, curls={"index": 0.625}))
    assert engine.mode is Mode.NONE


@pytest.mark.parametrize("missing", [{"present": False}, {"confidence": 0.5}])
def test_missing_or_low_confidence_releases_after_timeout(rig, missing):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    engine.process(pose(0.15625, "index", thumb_out=0))
    engine.process(pose(0.1875, "index", thumb_out=0))
    assert engine.left_down
    backend.events.clear()
    pointer.reset_mock()

    engine.process(make_frame(t=0.25, **missing))
    engine.process(make_frame(t=0.4375, **missing))
    assert engine.mode is Mode.POINTER
    assert engine.left_down
    assert backend_events(backend) == []
    assert pointer.mock_calls == []

    engine.process(make_frame(t=0.5, **missing))
    engine.process(make_frame(t=0.75, **missing))
    assert engine.mode is Mode.NONE
    assert not engine.left_down
    backend.release_all.assert_called_once()
    assert backend_events(backend) == [("button", "left", False)]
    assert pointer.mock_calls == [call.reset()]


def test_low_confidence_does_not_count_toward_lock(rig):
    engine, _, pointer, _ = rig
    engine.process(pose(0, "index"))
    engine.process(pose(0.125, "index", confidence=0.5))
    engine.process(pose(0.25, "index", confidence=0.75))
    assert engine.mode is Mode.WAIT
    assert pointer.mock_calls == []
    engine.process(pose(0.375, "index", confidence=0.75))
    assert engine.mode is Mode.POINTER


def test_brief_loss_rebases_pointer_without_releasing(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    pointer.reset_mock()
    engine.process(make_frame(t=0.25, present=False))
    engine.process(pose(0.375, "index", index_pos=(0.8, 0.7)))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls == [call.rebase(), call.update((0.8, 0.7), 0.2, 0.375)]
    assert backend_events(backend) == []


def test_recovery_after_loss_timeout_requires_a_new_mode_lock(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    pointer.reset_mock()
    engine.process(make_frame(t=0.25, present=False))
    engine.process(pose(0.5, "index"))
    assert engine.mode is Mode.WAIT
    assert pointer.mock_calls == [call.reset()]
    backend.release_all.assert_called_once()
    assert backend_events(backend) == []

    engine.process(pose(0.625, "index"))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls[-2:] == [call.rebase(), call.update((0.5, 0.4), 0.2, 0.625)]


def test_pointer_uses_image_tip_and_sends_only_nonzero_moves(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    assert pointer.mock_calls == [call.rebase(), call.update((0.5, 0.4), 0.2, 0.125)]
    assert backend_events(backend) == []

    pointer.update.return_value = (0, -3)
    engine.process(pose(0.25, "index", index_pos=(0.1, 0.2), hand_size=0.3))
    pointer.update.assert_called_with((0.1, 0.2), 0.3, 0.25)
    assert backend_events(backend) == [("move", 0, -3)]


def test_fast_curl_rolls_back_and_freezes_until_index_reopens(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    pointer.reset_mock()
    pointer.rollback.return_value = (-4, 2)

    engine.process(pose(0.15625, "index", "middle", curls={"index": 0.1875}))
    engine.process(pose(0.1875, "middle"))
    engine.process(pose(0.25, "middle", curls={"index": 0.5}))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls == [call.rollback(80)]
    assert backend_events(backend) == [("move", -4, 2)]

    engine.process(pose(0.375, "index", "middle", index_pos=(0.75, 0.5)))
    assert pointer.mock_calls == [
        call.rollback(80),
        call.rebase(),
        call.update((0.75, 0.5), 0.2, 0.375),
    ]
    assert backend_events(backend) == [("move", -4, 2)]


def test_fast_curl_to_fist_rolls_back_before_mode_exit_and_rebases_on_return(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    pointer.reset_mock()
    pointer.rollback.return_value = (0, 2)

    engine.process(pose(0.15625))
    assert engine.mode is Mode.NONE
    assert pointer.mock_calls == [call.rollback(80), call.reset()]
    assert backend_events(backend) == [("move", 0, 2)]

    engine.process(pose(0.25, "index"))
    engine.process(pose(0.375, "index"))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls[-2:] == [call.rebase(), call.update((0.5, 0.4), 0.2, 0.375)]


def test_scroll_rebases_then_updates_with_midpoint_and_sends_only_nonzero_deltas(rig):
    engine, backend, _, scroll = rig
    engine.process(pose(0, "index", "middle"))
    assert scroll.mock_calls == []
    engine.process(
        pose(
            0.125,
            "index",
            "middle",
            index_pos=(0.25, 0.5),
            positions={"middle": (0.5, 0.75)},
            hand_size=0.3,
        )
    )
    assert scroll.mock_calls == [call.rebase(), call.update((0.375, 0.625), 0.3, 0.125)]
    assert backend.events == []

    scroll.update.return_value = (2.5, -4.5)
    engine.process(pose(0.25, "index", "middle"))
    assert backend_events(backend) == [("scroll", 2.5, -4.5)]


def test_scroll_releases_on_fist_and_ticks_on_later_frames(rig):
    engine, backend, _, scroll = rig
    engine.process(pose(0, "index", "middle"))
    engine.process(pose(0.125, "index", "middle"))
    scroll.reset_mock()

    engine.process(pose(0.25))
    assert scroll.mock_calls == [call.release(0.25)]
    assert engine.mode is Mode.NONE
    engine.process(pose(0.375))
    assert backend.events == []
    scroll.tick.return_value = (0, -8.5)
    engine.process(pose(0.5))
    assert scroll.mock_calls == [call.release(0.25), call.tick(0.375), call.tick(0.5)]
    assert backend_events(backend) == [("scroll", 0, -8.5)]


@pytest.mark.parametrize("reopen", [True, False])
def test_scroll_inertia_stops_when_fingers_reopen_or_hand_disappears(rig, reopen):
    engine, backend, _, scroll = rig
    engine.process(pose(0, "index", "middle"))
    engine.process(pose(0.125, "index", "middle"))
    engine.process(pose(0.25))
    scroll.reset_mock()
    scroll.tick.return_value = (0, -8.5)

    engine.process(pose(0.375, "index") if reopen else make_frame(t=0.375, present=False))
    engine.process(pose(0.4375))
    assert scroll.mock_calls == [call.stop()]
    assert backend.events == []


def test_brief_hand_loss_stops_scroll_and_rebases_on_return(rig):
    engine, backend, _, scroll = rig
    engine.process(pose(0, "index", "middle"))
    engine.process(pose(0.125, "index", "middle"))
    scroll.reset_mock()

    engine.process(make_frame(t=0.25, present=False))
    assert scroll.mock_calls == [call.stop()]
    engine.process(pose(0.375, "index", "middle"))
    assert engine.mode is Mode.SCROLL
    assert scroll.mock_calls == [call.stop(), call.rebase(), call.update((0.5, 0.4), 0.2, 0.375)]
    assert backend.events == []


@pytest.mark.parametrize("rig", ["tap"], indirect=True)
def test_tap_takeoff_rolls_back_and_landing_rebases_pointer(rig):
    engine, backend, pointer, scroll = rig
    lock_pointer(engine)
    pointer.reset_mock()
    pointer.rollback.return_value = (-4, 2)

    engine.process(pose(0.25, "index", lift=0.3))
    assert backend_events(backend) == [("move", -4, 2)]
    assert pointer.mock_calls == [call.rollback(80)]
    scroll.stop.assert_called_once()

    engine.process(pose(0.3125, "index", lift=0.8))
    engine.process(pose(0.34375, "index", lift=None))
    assert pointer.mock_calls == [call.rollback(80)]
    assert backend_events(backend) == [("move", -4, 2)]

    engine.process(pose(0.375, "index", lift=0))
    assert pointer.mock_calls == [
        call.rollback(80),
        call.rebase(),
        call.update((0.5, 0.4), 0.2, 0.375),
    ]
    assert backend_events(backend) == [
        ("move", -4, 2),
        ("button", "left", True),
        ("button", "left", False),
    ]


@pytest.mark.parametrize("rig", ["tap"], indirect=True)
@pytest.mark.parametrize("fingers", [("index",), ("index", "middle")])
def test_scroll_requires_two_fingers_for_right_tap(rig, fingers):
    engine, backend, _, scroll = rig
    source = SyntheticSource().open(2).tap(fingers)
    for frame in source.frames():
        engine.process(frame)

    expected = [("button", "right", True), ("button", "right", False)]
    assert backend_events(backend) == (expected if len(fingers) == 2 else [])
    scroll.stop.assert_called_once()
    assert scroll.rebase.call_count == 2


@pytest.mark.parametrize("rig", ["tap"], indirect=True)
def test_leaving_scroll_mid_tap_does_not_restart_inertia(rig):
    engine, backend, _, scroll = rig
    engine.process(pose(0, "index", "middle"))
    engine.process(pose(0.125, "index", "middle"))
    engine.process(pose(0.25, "index", "middle", lift=0.8))
    engine.process(pose(0.3125, lift=0.8))
    engine.process(pose(0.375, lift=0))

    assert engine.mode is Mode.NONE
    scroll.stop.assert_called_once()
    scroll.release.assert_not_called()
    scroll.tick.assert_not_called()
    assert backend_events(backend) == []


@pytest.mark.parametrize(("method", "tap_count"), [("tap", 2), ("both", 2), ("thumb", 0)])
def test_two_quick_taps_follow_click_method(method, tap_count):
    source = SyntheticSource().open(1).tap().open(1, s=0.1).tap()
    cfg = load_control_config()
    cfg["general"]["click_method"] = method
    backend = DryBackend()
    engine = GestureEngine(cfg, backend, source.info)
    for frame in source.frames():
        engine.process(frame)

    click = [("button", "left", True), ("button", "left", False)]
    assert backend_events(backend) == click * tap_count


@pytest.mark.parametrize(("rig", "thumb_click"), [("tap", False), ("both", True)], indirect=["rig"])
@pytest.mark.parametrize("fingers", [("index",), ("index", "middle")])
def test_thumb_click_with_tap_enabled(rig, thumb_click, fingers):
    engine, backend, _, _ = rig
    source = SyntheticSource().open(len(fingers), s=0.3).thumb(True).thumb(False)
    for frame in source.frames():
        engine.process(frame)
    button = "left" if len(fingers) == 1 else "right"
    expected = [("button", button, True), ("button", button, False)]
    assert backend_events(backend) == (expected if thumb_click else [])


@pytest.mark.parametrize("rig", ["tap"], indirect=True)
def test_thumb_in_tap_mode_presses_only_after_drag_threshold(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    engine.process(pose(0.25, "index", thumb_out=0))
    engine.process(pose(0.3125, "index", thumb_out=0, index_pos=(0.504, 0.4)))
    assert not engine.left_down
    assert backend_events(backend) == []

    engine.process(pose(0.375, "index", thumb_out=0, index_pos=(0.53, 0.4)))
    assert engine.left_down
    assert backend_events(backend) == [("button", "left", True)]
    pointer.update.return_value = (4, 0)
    engine.process(pose(0.4375, "index", thumb_out=0, index_pos=(0.54, 0.4)))
    pointer.update.return_value = (0, 0)
    engine.process(pose(0.5, "index"))
    engine.process(pose(0.5625, "index"))
    assert not engine.left_down
    assert backend_events(backend) == [
        ("button", "left", True),
        ("move", 4, 0),
        ("button", "left", False),
    ]
