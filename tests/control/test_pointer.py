import tomllib
from pathlib import Path
from time import process_time_ns

import pytest

from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.pointer import PointerMotion
from tkt.control.sources import SyntheticSource
from tkt.core.types import SCHEMA_VERSION, SourceInfo

INFO = SourceInfo(SCHEMA_VERSION, 1280, 720, 60.0, "right")


@pytest.fixture
def pointer_cfg():
    path = Path(__file__).resolve().parents[2] / "config" / "control.toml"
    with path.open("rb") as file:
        cfg = tomllib.load(file)["pointer"]
    # Testler sabit referans kazançlarla yazıldı; config'teki canlı ayar onları bozmasın.
    return dict(cfg, min_cutoff=1.0, gain_min=500, gain_max=2500)


@pytest.fixture
def linear_cfg(pointer_cfg):
    return dict(pointer_cfg, min_cutoff=1e9, beta=0.0, gain_max=pointer_cfg["gain_min"])


def test_fractional_motion_accumulates_over_1000_steps(linear_cfg):
    pointer = PointerMotion(linear_cfg, INFO)
    assert pointer.update((0.25, 0.75), 0.2, 0.0) == (0, 0)
    deltas = [
        pointer.update((0.25 + i / 10_000, 0.75 - i / 10_000), 0.2, i / 60)
        for i in range(1, 1001)
    ]

    assert (sum(dx for dx, _ in deltas), sum(dy for _, dy in deltas)) == (250, -141)


def test_gap_restarts_filter_at_new_position(pointer_cfg):
    pointer = PointerMotion(pointer_cfg, INFO)
    pointer.update((0.1, 0.2), 0.2, 0.0)
    assert pointer.update((0.2, 0.2), 0.2, 0.05)[0] > 0

    assert pointer.update((0.8, 0.7), 0.2, 0.5) == (0, 0)
    assert pointer.update((0.8, 0.7), 0.2, 0.51) == (0, 0)
    assert pointer.rollback(150) == (0, 0)


def test_gap_at_threshold_still_produces_motion(pointer_cfg):
    pointer = PointerMotion(pointer_cfg, INFO)
    pointer.update((0.1, 0.2), 0.2, 0.3)

    assert pointer.update((0.2, 0.2), 0.2, 0.4)[0] > 0


@pytest.mark.parametrize("rebase_t", [0.05, 0.08])
def test_rebase_uses_next_frame_and_preserves_recent_history(pointer_cfg, rebase_t):
    pointer = PointerMotion(pointer_cfg, INFO)
    pointer.update((0.1, 0.2), 0.2, 0.0)
    dx, dy = pointer.update((0.2, 0.3), 0.2, 0.05)
    assert dx > 0 and dy > 0
    pointer.rebase()

    assert pointer.update((0.8, 0.7), 0.2, rebase_t) == (0, 0)
    assert pointer.update((0.8, 0.7), 0.2, 0.1) == (0, 0)
    assert pointer.rollback(100) == (-dx, -dy)


def test_rollback_reverses_only_last_100_ms_and_consumes_moves(pointer_cfg):
    pointer = PointerMotion(pointer_cfg, INFO)
    pointer.update((0.2, 0.8), 0.2, 0.05)
    older = pointer.update((0.3, 0.7), 0.2, 0.1)
    recent = [
        pointer.update((0.3 + i / 10, 0.7 - i / 10), 0.2, t)
        for i, t in enumerate((0.15, 0.2, 0.25), start=1)
    ]

    assert all(dx > 0 and dy < 0 for dx, dy in recent)
    assert pointer.rollback(100) == (-sum(dx for dx, _ in recent), -sum(dy for _, dy in recent))
    assert pointer.rollback(100) == (0, 0)
    assert pointer.rollback(150) == (-older[0], -older[1])


def test_history_covers_longer_configured_rollback(pointer_cfg):
    pointer = PointerMotion(dict(pointer_cfg, rollback_ms=250, gap_reset_ms=500), INFO)
    pointer.update((0.3, 0.2), 0.2, 0.0)
    assert pointer.update((0.2, 0.2), 0.2, 0.05)[0] < 0
    recent_dx, _ = pointer.update((0.3, 0.2), 0.2, 0.1)
    latest_dx, _ = pointer.update((0.4, 0.2), 0.2, 0.35)

    assert recent_dx > 0 and latest_dx > 0
    assert pointer.rollback(500) == (-recent_dx - latest_dx, 0)


def test_hand_scale_uses_two_second_median(linear_cfg):
    pointer = PointerMotion(dict(linear_cfg, gap_reset_ms=3000), INFO)
    for t in (0.0, 0.25, 0.5):
        pointer.update((0.1, 0.1), 0.2, t)

    assert pointer.update((0.12, 0.1), 0.9, 0.75) == (50, 0)
    assert pointer.update((0.14, 0.1), 0.4, 2.8) == (25, 0)


def test_reset_clears_scale_history_and_reference(linear_cfg):
    pointer = PointerMotion(linear_cfg, INFO)
    pointer.update((0.1, 0.1), 0.2, 0.0)
    assert pointer.update((0.2, 0.1), 0.2, 0.05)[0] > 0
    pointer.reset()

    assert pointer.rollback(150) == (0, 0)
    assert pointer.update((0.8, 0.8), 0.4, 0.0) == (0, 0)
    assert pointer.update((0.82, 0.8), 0.4, 0.05) == (25, 0)


def test_zero_hand_size_without_scale_emits_no_motion(linear_cfg):
    pointer = PointerMotion(linear_cfg, INFO)
    assert pointer.update((0.1, 0.1), 0.0, 0.0) == (0, 0)
    assert pointer.update((0.2, 0.1), 0.0, 0.05) == (0, 0)
    assert pointer.update((0.2, 0.1), 0.2, 0.08) == (0, 0)


def test_d_cutoff_can_override_filter_default(pointer_cfg):
    default = PointerMotion(pointer_cfg, INFO)
    explicit_default = PointerMotion(dict(pointer_cfg, d_cutoff=1.0), INFO)
    faster_derivative = PointerMotion(dict(pointer_cfg, d_cutoff=10.0), INFO)
    for pointer in (default, explicit_default, faster_derivative):
        pointer.update((0.1, 0.1), 0.2, 0.0)

    point = (0.12, 0.1)
    baseline = default.update(point, 0.2, 0.01)
    assert explicit_default.update(point, 0.2, 0.01) == baseline
    assert faster_derivative.update(point, 0.2, 0.01)[0] > baseline[0]


def test_equal_distance_at_higher_speed_moves_cursor_further(pointer_cfg):
    def move_for(duration):
        source = SyntheticSource().open(1).move(0.15, 0.0, duration).open(1, s=1.0)
        pointer = PointerMotion(pointer_cfg, source.info)
        backend = DryBackend()
        for frame in source.frames():
            backend.set_time(frame.t)
            delta = pointer.update(frame.fingers["index"].img, frame.hand_size, frame.t)
            if delta != (0, 0):
                backend.move_rel(*delta)
        return sum(event.args[0] for event in backend.events)

    assert 0 < move_for(2.0) < move_for(0.15)


def test_gain_uses_smoothstep_between_speed_thresholds(linear_cfg, pointer_cfg):
    pointer = PointerMotion(dict(linear_cfg, gain_max=pointer_cfg["gain_max"]), INFO)
    pointer.update((0.1, 0.1), 0.2, 0.0)

    # 0,975 el boyu/sn, ayarlı hız aralığının dörtte birine denk gelir.
    # Smoothstep(0,25) ile kazanç 812,5, yer değiştirme ise 0,04875 el boyudur.
    assert pointer.update((0.10975, 0.1), 0.2, 0.05) == (40, 0)


def test_pointer_processing_average_is_under_two_ms(pointer_cfg):
    pointer = PointerMotion(pointer_cfg, INFO)
    source = SyntheticSource().open(1).move(0.25, -0.25, 10.0)
    samples = [
        (frame.fingers["index"].img, frame.hand_size, frame.t) for frame in source.frames()
    ]

    # Sentetik kare zamanları sabit. Toplu CPU süresi ölçümü, tek karedeki saat
    # çözünürlüğünden veya CI'daki zamanlayıcı beklemelerinden etkilenmez.
    started = process_time_ns()
    for sample in samples:
        pointer.update(*sample)
    assert (process_time_ns() - started) / len(samples) < 2_000_000


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"speed_hi": 0.3}, "speed_hi.*speed_lo"),
        ({"speed_hi": 0.1}, "speed_hi.*speed_lo"),
        ({"speed_hi": float("inf")}, "speed_hi.*speed_lo"),
        ({"speed_lo": float("nan")}, "speed_hi.*speed_lo"),
        ({"gain_max": 499}, "gain_max.*gain_min"),
        ({"gain_min": float("-inf")}, "gain_max.*gain_min"),
        ({"gain_max": float("nan")}, "gain_max.*gain_min"),
    ],
)
def test_invalid_acceleration_config_raises_value_error(pointer_cfg, override, message):
    with pytest.raises(ValueError, match=message):
        PointerMotion(pointer_cfg | override, INFO)
