import pytest
from tests.control.conftest import make_frame

from tkt.app.config import load_control_config
from tkt.control.sources import SyntheticSource
from tkt.control.tap import TapDetector, TapUpdate

INDEX = ("index",)
PAIR = ("index", "middle")


@pytest.fixture
def detector():
    return TapDetector(load_control_config())


@pytest.mark.parametrize(
    ("watched", "fingers", "ms", "expected"),
    [
        (INDEX, INDEX, 150, ["tap1"]),
        (INDEX, INDEX, 50, []),
        (INDEX, INDEX, 500, []),
        (PAIR, PAIR, 150, ["tap2"]),
        (PAIR, INDEX, 150, []),
    ],
)
def test_synthetic_tap(detector, watched, fingers, ms, expected):
    source = SyntheticSource().open(len(watched)).tap(fingers, ms).open(len(watched))
    updates = [detector.update(frame, watched) for frame in source.frames()]
    assert [update.kind for update in updates if update.kind] == expected
    assert sum(update.started for update in updates) == 1
    assert not updates[-1].busy


def test_timeout_stays_busy_until_lift_drops(detector):
    assert detector.update(make_frame(t=0, lift=0.3), INDEX) == TapUpdate(True, True, None)
    assert detector.update(make_frame(t=0.125, lift=0.8), INDEX) == TapUpdate(False, True, None)
    assert detector.update(make_frame(t=0.375, lift=0.8), INDEX) == TapUpdate(False, True, None)
    assert detector.update(make_frame(t=0.5, lift=0), INDEX) == TapUpdate(False, False, None)


@pytest.mark.parametrize(
    "invalid",
    [{"curls": {"index": 0.35}}, {"palm": (0.51, 0.6)}],
)
def test_transient_curl_or_palm_shift_cancels_tap(detector, invalid):
    detector.update(make_frame(t=0, lift=0.3), INDEX)
    assert detector.update(make_frame(t=0.0625, lift=0.8, **invalid), INDEX).busy
    assert detector.update(make_frame(t=0.125, lift=0), INDEX) == TapUpdate(False, False, None)


def test_lift_must_exceed_up_threshold(detector):
    detector.update(make_frame(t=0, lift=0.3), INDEX)
    detector.update(make_frame(t=0.0625, lift=0.5), INDEX)
    assert detector.update(make_frame(t=0.125, lift=0), INDEX) == TapUpdate(False, False, None)


@pytest.mark.parametrize(
    ("second_start", "second_landing", "expected"),
    [(0.0625, 0.1875, "tap2"), (0.09375, 0.1875, None), (0.0625, 0.21875, None)],
)
def test_pair_checks_both_takeoff_and_landing_windows(
    detector, second_start, second_landing, expected
):
    detector.update(make_frame(t=0, lift={"index": 0.8, "middle": 0}), PAIR)
    detector.update(make_frame(t=second_start, lift=0.8), PAIR)
    first_down = make_frame(t=0.125, lift={"index": 0, "middle": 0.8})
    assert detector.update(first_down, PAIR) == TapUpdate(False, True, None)
    assert detector.update(make_frame(t=second_landing, lift=0), PAIR) == TapUpdate(
        False, False, expected
    )


def test_missing_lift_requires_a_landing_before_rearming(detector):
    assert detector.update(make_frame(t=0, lift=0.8), INDEX).started
    assert detector.update(make_frame(t=0.0625, lift=None), INDEX) == TapUpdate(False, False, None)

    for t, lift in [(0.125, 0.8), (0.1875, 0.25), (0.25, 0.8), (0.3125, 0)]:
        assert detector.update(make_frame(t=t, lift=lift), INDEX) == TapUpdate(False, False, None)

    assert detector.update(make_frame(t=0.375, lift=0.8), INDEX).started
    assert detector.update(make_frame(t=0.4375, lift=0), INDEX).kind == "tap1"


def test_cooldown_ignores_entire_lift_started_during_cooldown(detector):
    detector.update(make_frame(t=0, lift=0.8), INDEX)
    assert detector.update(make_frame(t=0.125, lift=0), INDEX).kind == "tap1"

    for t, lift in [(0.1875, 0.3), (0.25, 0.8), (0.375, 0)]:
        assert detector.update(make_frame(t=t, lift=lift), INDEX) == TapUpdate(False, False, None)

    assert detector.update(make_frame(t=0.4375, lift=0.8), INDEX).started
    assert detector.update(make_frame(t=0.5, lift=0), INDEX).kind == "tap1"
