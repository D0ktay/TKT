from unittest.mock import Mock, call

import pytest
from tests.control.conftest import make_frame

from tkt.control import gestures
from tkt.control.gestures import GestureEngine, Mode
from tkt.core.types import FINGERS, SourceInfo


class RecordingBackend:
    def __init__(self):
        self.events = []

    def move_rel(self, dx: int, dy: int) -> None:
        self.events.append(("move", dx, dy))

    def button(self, name: str, down: bool) -> None:
        self.events.append(("button", name, down))

    def scroll(self, dx: float, dy: float) -> None:
        self.events.append(("scroll", dx, dy))

    def system_action(self, name: str) -> None:
        self.events.append(("system", name))

    def release_all(self) -> None:
        self.events.append(("release",))


@pytest.fixture
def rig(monkeypatch):
    cfg = {
        "general": {"confidence_min": 0.75, "lost_release_ms": 250},
        "mode": {"lock_ms": 125, "curl_open": 0.25, "curl_closed": 0.625, "freeze_curl_rate": 4},
        "pointer": {"rollback_ms": 80},
        "scroll": {},
    }
    info = SourceInfo(schema=1, width=640, height=512, fps=60, hand="right")
    backend = RecordingBackend()
    pointer = Mock(spec=gestures.PointerMotion)
    pointer.update.return_value = (0, 0)
    pointer.rollback.return_value = (0, 0)
    pointer_factory = Mock(return_value=pointer)
    scroll = Mock(spec=gestures.ScrollMotion)
    scroll_factory = Mock(return_value=scroll)
    monkeypatch.setattr(gestures, "PointerMotion", pointer_factory)
    monkeypatch.setattr(gestures, "ScrollMotion", scroll_factory)
    engine = GestureEngine(cfg, backend, info)
    pointer_factory.assert_called_once_with(cfg["pointer"], info)
    scroll_factory.assert_called_once_with(cfg["scroll"], info)
    yield engine, backend, pointer
    assert scroll.mock_calls == []


def pose(t, *opened, **kwargs):
    curls = {name: 0.0 if name in opened else 1.0 for name in FINGERS}
    curls.update(kwargs.pop("curls", {}))
    return make_frame(t=t, curls=curls, **kwargs)


def lock_pointer(engine):
    engine.process(pose(0, "index"))
    engine.process(pose(0.125, "index"))
    assert engine.mode is Mode.POINTER


def test_opening_four_fingers_does_not_lock_intermediate_mode(rig):
    engine, backend, pointer = rig
    engine.process(pose(0))
    assert engine.mode is Mode.NONE

    for count in range(1, 5):
        engine.process(pose(count / 32, *FINGERS[:count]))
        assert engine.mode is Mode.WAIT
    engine.process(pose(0.21875, *FINGERS))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.25, *FINGERS))
    assert engine.mode is Mode.SWIPE
    assert backend.events == []
    assert pointer.mock_calls == []


def test_locked_mode_changes_only_after_all_fingers_close(rig):
    engine, backend, pointer = rig
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
    assert backend.events == []
    assert pointer.mock_calls == []


def test_wait_requires_the_same_fingers_to_stay_open(rig):
    engine, _, _ = rig
    engine.process(pose(0, "middle"))
    engine.process(pose(0.125, "middle"))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.25, "index"))
    assert engine.mode is Mode.WAIT
    engine.process(pose(0.375, "index"))
    assert engine.mode is Mode.POINTER


def test_curl_hysteresis_preserves_state_at_both_thresholds(rig):
    engine, _, _ = rig
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
    engine, backend, pointer = rig
    lock_pointer(engine)
    engine.left_down = True
    pointer.reset_mock()

    engine.process(make_frame(t=0.25, **missing))
    engine.process(make_frame(t=0.4375, **missing))
    assert engine.mode is Mode.POINTER
    assert engine.left_down
    assert backend.events == []
    assert pointer.mock_calls == []

    engine.process(make_frame(t=0.5, **missing))
    engine.process(make_frame(t=0.75, **missing))
    assert engine.mode is Mode.NONE
    assert not engine.left_down
    assert backend.events == [("release",)]
    assert pointer.mock_calls == [call.reset()]


def test_low_confidence_does_not_count_toward_lock(rig):
    engine, _, pointer = rig
    engine.process(pose(0, "index"))
    engine.process(pose(0.125, "index", confidence=0.5))
    engine.process(pose(0.25, "index", confidence=0.75))
    assert engine.mode is Mode.WAIT
    assert pointer.mock_calls == []
    engine.process(pose(0.375, "index", confidence=0.75))
    assert engine.mode is Mode.POINTER


def test_brief_loss_rebases_pointer_without_releasing(rig):
    engine, backend, pointer = rig
    lock_pointer(engine)
    pointer.reset_mock()
    engine.process(make_frame(t=0.25, present=False))
    engine.process(pose(0.375, "index", index_pos=(0.8, 0.7)))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls == [call.rebase(), call.update((0.8, 0.7), 0.2, 0.375)]
    assert backend.events == []


def test_recovery_after_loss_timeout_requires_a_new_mode_lock(rig):
    engine, backend, pointer = rig
    lock_pointer(engine)
    pointer.reset_mock()
    engine.process(make_frame(t=0.25, present=False))
    engine.process(pose(0.5, "index"))
    assert engine.mode is Mode.WAIT
    assert pointer.mock_calls == [call.reset()]
    assert backend.events == [("release",)]

    engine.process(pose(0.625, "index"))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls[-2:] == [call.rebase(), call.update((0.5, 0.4), 0.2, 0.625)]


def test_pointer_uses_image_tip_and_sends_only_nonzero_moves(rig):
    engine, backend, pointer = rig
    lock_pointer(engine)
    assert pointer.mock_calls == [call.rebase(), call.update((0.5, 0.4), 0.2, 0.125)]
    assert backend.events == []

    pointer.update.return_value = (0, -3)
    engine.process(pose(0.25, "index", index_pos=(0.1, 0.2), hand_size=0.3))
    pointer.update.assert_called_with((0.1, 0.2), 0.3, 0.25)
    assert backend.events == [("move", 0, -3)]


def test_fast_curl_rolls_back_and_freezes_until_index_reopens(rig):
    engine, backend, pointer = rig
    lock_pointer(engine)
    pointer.reset_mock()
    pointer.rollback.return_value = (-4, 2)

    engine.process(pose(0.15625, "index", "middle", curls={"index": 0.1875}))
    engine.process(pose(0.1875, "middle"))
    engine.process(pose(0.25, "middle", curls={"index": 0.5}))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls == [call.rollback(80)]
    assert backend.events == [("move", -4, 2)]

    engine.process(pose(0.375, "index", "middle", index_pos=(0.75, 0.5)))
    assert pointer.mock_calls == [
        call.rollback(80),
        call.rebase(),
        call.update((0.75, 0.5), 0.2, 0.375),
    ]
    assert backend.events == [("move", -4, 2)]


def test_fast_curl_to_fist_rolls_back_before_mode_exit_and_rebases_on_return(rig):
    engine, backend, pointer = rig
    lock_pointer(engine)
    pointer.reset_mock()
    pointer.rollback.return_value = (0, 2)

    engine.process(pose(0.15625))
    assert engine.mode is Mode.NONE
    assert pointer.mock_calls == [call.rollback(80), call.reset()]
    assert backend.events == [("move", 0, 2)]

    engine.process(pose(0.25, "index"))
    engine.process(pose(0.375, "index"))
    assert engine.mode is Mode.POINTER
    assert pointer.mock_calls[-2:] == [call.rebase(), call.update((0.5, 0.4), 0.2, 0.375)]
