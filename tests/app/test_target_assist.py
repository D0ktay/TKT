from collections import Counter

from tkt.app.target_assist import candidate, schedule


def test_snap_preserves_inside_and_uses_nearest_edge():
    targets = [(10, 10, 16)]
    assert candidate((11, 13), targets, 4) == (0, (11, 13))
    assert candidate((8, 13), targets, 4) == (0, (10, 13))
    assert candidate((8, 13), targets, 0) == (None, (8, 13))


def test_snap_skips_ambiguous_gap():
    targets = [(0, 0, 16), (22, 0, 16)]
    assert candidate((18, 7), targets, 4) == (None, (18, 7))
    assert candidate((22, 7), targets, 8) == (1, (22, 7))


def test_assisted_and_baseline_trials_match():
    trials = schedule(4, 2, 42)
    assert Counter(t[:3] for t in trials if t[3]) == Counter(t[:3] for t in trials if not t[3])
    assert len(trials) == 72
