from unittest.mock import call

import pytest
from tests.control.test_gestures import backend_events, lock_pointer, pose
from tests.control.test_gestures import rig as rig

from tkt.control.gestures import Mode


@pytest.mark.parametrize("click_count", [1, 2])
def test_thumb_close_and_open_produce_separate_left_clicks(rig, click_count):
    engine, backend, _, _ = rig
    lock_pointer(engine)

    for click in range(click_count):
        start = 0.25 + click * 0.125
        engine.process(pose(start, "index", thumb_out=0.125))
        assert not engine.left_down
        engine.process(pose(start + 0.03125, "index", thumb_out=0.125))
        assert engine.left_down
        engine.process(pose(start + 0.0625, "index", thumb_out=0.75))
        assert engine.left_down
        engine.process(pose(start + 0.09375, "index", thumb_out=0.75))
        assert not engine.left_down

    assert backend_events(backend) == [
        ("button", "left", True),
        ("button", "left", False),
    ] * click_count


def test_short_thumb_changes_are_debounced_and_threshold_band_preserves_state(rig):
    engine, backend, _, _ = rig
    lock_pointer(engine)

    engine.process(pose(0.25, "index", thumb_out=0.125))
    engine.process(pose(0.265625, "index", thumb_out=0.75))
    engine.process(pose(0.3125, "index", thumb_out=0.25))
    assert backend.events == []

    engine.process(pose(0.375, "index", thumb_out=0.125))
    engine.process(pose(0.40625, "index", thumb_out=0.5))
    assert engine.left_down
    engine.process(pose(0.4375, "index", thumb_out=0.75))
    engine.process(pose(0.453125, "index", thumb_out=0.125))
    engine.process(pose(0.5, "index", thumb_out=0.5))
    assert backend_events(backend) == [("button", "left", True)]
    assert engine.left_down

    engine.process(pose(0.53125, "index", thumb_out=0.75))
    engine.process(pose(0.5625, "index", thumb_out=0.375))
    assert not engine.left_down
    assert backend_events(backend)[-1] == ("button", "left", False)


def test_pointer_waits_for_drag_threshold_then_rebases_and_moves_with_button_held(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    engine.process(pose(0.25, "index", thumb_out=0))
    pointer.reset_mock()

    engine.process(pose(0.28125, "index", thumb_out=0))
    engine.process(pose(0.3125, "index", thumb_out=0, index_pos=(0.506, 0.407)))
    assert backend_events(backend) == [("button", "left", True)]
    assert pointer.mock_calls == []

    engine.process(pose(0.34375, "index", thumb_out=0, index_pos=(0.52, 0.4)))
    assert pointer.mock_calls == [call.rebase(), call.update((0.52, 0.4), 0.2, 0.34375)]
    pointer.update.return_value = (3, -5)
    engine.process(pose(0.375, "index", thumb_out=0, index_pos=(0.53, 0.39)))
    assert engine.left_down
    assert backend_events(backend) == [("button", "left", True), ("move", 3, -5)]


def test_releasing_without_drag_rebases_to_avoid_a_pointer_jump(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    engine.process(pose(0.25, "index", thumb_out=0))
    engine.process(pose(0.28125, "index", thumb_out=0))
    pointer.reset_mock()

    engine.process(pose(0.3125, "index", thumb_out=1, index_pos=(0.505, 0.4)))
    engine.process(pose(0.34375, "index", thumb_out=1, index_pos=(0.505, 0.4)))

    assert pointer.mock_calls == [call.rebase(), call.update((0.505, 0.4), 0.2, 0.34375)]
    assert backend_events(backend) == [("button", "left", True), ("button", "left", False)]


def test_fast_curl_cannot_move_pointer_while_waiting_for_drag_threshold(rig):
    engine, backend, pointer, _ = rig
    lock_pointer(engine)
    engine.process(pose(0.25, "index", thumb_out=0))
    engine.process(pose(0.28125, "index", thumb_out=0))
    pointer.rollback.return_value = (-8, 0)

    engine.process(pose(0.3125, thumb_out=0))

    pointer.rollback.assert_not_called()
    assert backend_events(backend) == [("button", "left", True), ("button", "left", False)]


@pytest.mark.parametrize("moving", [False, True])
def test_scroll_thumb_right_click_requires_still_palm(rig, moving):
    engine, backend, _, scroll = rig
    engine.process(pose(0, "index", "middle"))
    engine.process(pose(0.125, "index", "middle"))
    palm = (0.525, 0.6) if moving else (0.505, 0.6)
    engine.process(pose(0.15625, "index", "middle", palm=palm))
    engine.process(pose(0.21875, "index", "middle", thumb_out=0, index_pos=(0.7, 0.4)))
    engine.process(pose(0.25, "index", "middle", thumb_out=0, index_pos=(0.7, 0.4)))
    engine.process(pose(0.5, "index", "middle", thumb_out=0))

    if moving:
        assert backend.events == []
        scroll.stop.assert_not_called()
    else:
        assert backend_events(backend) == [("button", "right", True), ("button", "right", False)]
        scroll.stop.assert_called_once()
    assert not engine.left_down


@pytest.mark.parametrize("end", ["mode", "missing", "release"])
def test_held_button_is_released_when_gesture_ends(rig, end):
    engine, backend, _, scroll = rig
    lock_pointer(engine)
    engine.process(pose(0.25, "index", thumb_out=0))
    engine.process(pose(0.28125, "index", thumb_out=0))
    assert engine.left_down

    if end == "missing":
        engine.process(pose(0.3125, present=False))
        engine.process(pose(0.5625, present=False))
    elif end == "release":
        backend.button("right", True)
        engine.release()
    else:
        engine.process(pose(0.3125, thumb_out=0))
        engine.process(pose(0.34375, "index", thumb_out=0))
        engine.process(pose(0.46875, "index", thumb_out=0))
        assert engine.mode is Mode.POINTER

    assert not engine.left_down
    if end == "release":
        assert backend_events(backend) == [
            ("button", "left", True),
            ("button", "right", True),
            ("button", "left", False),
            ("button", "right", False),
        ]
        assert scroll.stop.call_count == 2
    else:
        assert backend_events(backend) == [("button", "left", True), ("button", "left", False)]
