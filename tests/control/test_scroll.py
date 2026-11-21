from math import exp

import pytest

from tkt.control import scroll
from tkt.control.scroll import ScrollMotion
from tkt.core.types import SourceInfo

_INFO = SourceInfo(schema=1, width=800, height=400, fps=50, hand="right")
_CFG = {
    "natural": True,
    "gain": 800,
    "axis_lock": 0.5,
    "inertia_ms": 200,
    "inertia_min_speed": 100,
}


class IdentityFilter:
    def __init__(self, min_cutoff, beta, d_cutoff):
        pass

    def __call__(self, pos, t):
        return pos

    def reset(self):
        pass


@pytest.fixture
def make_scroll(monkeypatch):
    monkeypatch.setattr(scroll, "OneEuro", IdentityFilter)

    def make(**overrides):
        return ScrollMotion(_CFG | overrides, _INFO)

    return make


@pytest.mark.parametrize(
    ("pos", "expected"),
    [
        ((0.55, 0.51), (200, 0)),
        ((0.51, 0.40), (0, -200)),
        ((0.55, 0.42), (200, -160)),
    ],
)
def test_axis_lock_in_camera_pixels(make_scroll, pos, expected):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    assert motion.update(pos, 0.2, 0.02) == pytest.approx(expected)


@pytest.mark.parametrize(("natural", "dy"), [(True, -20), (False, 20)])
def test_natural_direction(make_scroll, natural, dy):
    motion = make_scroll(natural=natural)
    motion.update((0.5, 0.5), 0.2, 0)
    assert motion.update((0.5, 0.49), 0.2, 0.02) == pytest.approx((0, dy))


def test_rebase_discards_previous_position_and_velocity(make_scroll):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    motion.update((0.5, 0.49), 0.2, 0.02)
    motion.rebase()
    assert motion.update((0.9, 0.9), 0.2, 0.04) == (0, 0)
    motion.release(0.06)
    assert motion.tick(0.08) == (0, 0)


def test_inertia_averages_velocity_and_decays_until_threshold(make_scroll):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    motion.update((0.5, 0.49), 0.2, 0.02)  # -1000 px/s
    motion.update((0.5, 0.485), 0.2, 0.04)  # -500 px/s
    motion.release(0.06)
    assert motion.tick(0.08) == pytest.approx((0, -15 * exp(-0.1)))
    assert motion.tick(0.1) == pytest.approx((0, -15 * exp(-0.2)))
    for frame in range(6, 24):
        motion.tick(frame * 0.02)
    assert motion.tick(0.48) == (0, 0)


def test_stop_cuts_inertia_immediately(make_scroll):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    motion.update((0.5, 0.49), 0.2, 0.02)
    motion.release(0.04)
    assert motion.tick(0.06)[1] < 0
    motion.stop()
    assert motion.tick(0.08) == (0, 0)


def test_slow_release_has_no_inertia(make_scroll):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    motion.update((0.5, 0.4995), 0.2, 0.02)
    motion.release(0.04)
    assert motion.tick(0.06) == (0, 0)


def test_tracking_gap_starts_with_a_new_reference(make_scroll):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    motion.update((0.5, 0.49), 0.2, 0.02)
    assert motion.update((0.7, 0.8), 0.2, 0.14) == (0, 0)
    assert motion.update((0.7, 0.79), 0.2, 0.16) == pytest.approx((0, -20))


@pytest.mark.parametrize("release_t", [0.04, 0.14])
def test_gap_before_release_or_tick_cancels_inertia(make_scroll, release_t):
    motion = make_scroll()
    motion.update((0.5, 0.5), 0.2, 0)
    motion.update((0.5, 0.49), 0.2, 0.02)
    motion.release(release_t)
    assert motion.tick(0.16) == (0, 0)
    assert motion.tick(0.18) == (0, 0)


def test_one_euro_smooths_and_resets_on_rebase():
    motion = ScrollMotion(_CFG, _INFO)
    motion.update((0.5, 0.5), 0.2, 0)
    dx, dy = motion.update((0.5, 0.49), 0.2, 0.02)
    assert dx == 0
    assert -20 < dy < 0
    motion.rebase()
    assert motion.update((0.5, 0.1), 0.2, 0.04) == (0, 0)
    assert motion.update((0.5, 0.1), 0.2, 0.06) == (0, 0)
