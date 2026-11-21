from math import dist
from random import Random
from statistics import stdev

from tkt.control.filters import OneEuro


def test_constant_input_stays_constant():
    filt = OneEuro(min_cutoff=1.0, beta=0.02)
    point = (31.25, -73.5)

    assert all(filt(point, i / 60) == point for i in range(120))


def test_filter_reduces_noise_on_both_axes():
    rng = Random(42)
    samples = [(rng.gauss(100, 2), rng.gauss(200, 2)) for _ in range(300)]
    filt = OneEuro(min_cutoff=1.0, beta=0.02)
    outputs = [filt(point, i / 60) for i, point in enumerate(samples)]

    for axis in range(2):
        assert stdev(p[axis] for p in outputs[60:]) < stdev(p[axis] for p in samples[60:]) / 2


def test_adaptive_cutoff_limits_ramp_lag():
    adaptive = OneEuro(min_cutoff=1.0, beta=0.02)
    fixed = OneEuro(min_cutoff=1.0, beta=0.0)

    for i in range(120):
        point = (5.0 * i, -4.0 * i)
        adaptive_lag = dist(point, adaptive(point, i / 60))
        fixed_lag = dist(point, fixed(point, i / 60))
        assert adaptive_lag < 15.0

    assert adaptive_lag < fixed_lag / 4


def test_nonpositive_dt_leaves_filter_state_unchanged():
    filt = OneEuro(min_cutoff=1.0, beta=0.02)
    reference = OneEuro(min_cutoff=1.0, beta=0.02)

    first, second = (1.0, 2.0), (5.0, 8.0)
    for one_euro in (filt, reference):
        assert one_euro(first, 0.0) == first
        previous = one_euro(second, 0.02)

    assert filt((100.0, -100.0), 0.02) == previous
    assert filt((100.0, -100.0), 0.01) == previous
    assert filt((9.0, 12.0), 0.05) == reference((9.0, 12.0), 0.05)


def test_reset_restarts_position_and_derivative():
    filt = OneEuro(min_cutoff=1.0, beta=0.02, d_cutoff=3.0)
    filt((0.0, 0.0), 10.0)
    filt((100.0, -100.0), 10.02)
    filt.reset()

    fresh = OneEuro(min_cutoff=1.0, beta=0.02, d_cutoff=3.0)
    assert filt((2.0, 3.0), 0.0) == (2.0, 3.0)
    fresh((2.0, 3.0), 0.0)
    assert filt((4.0, 5.0), 0.02) == fresh((4.0, 5.0), 0.02)
