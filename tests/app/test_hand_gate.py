"""Barış işaretiyle el faresi kapısı: karar mantığı (kamera olmadan)."""
import math

import pytest

from tkt.app.config import load_control_config
from tkt.app.hand_gate import HandGate, learn_ranges, load_ranges, save_ranges
from tkt.core.types import FINGERS, Finger, HandFrame

OPEN, CURLED = 0.1, 0.9
SIGN = {"index": OPEN, "middle": OPEN, "ring": CURLED, "pinky": CURLED}
POINT = {"index": OPEN, "middle": CURLED, "ring": CURLED, "pinky": CURLED}
FLAT = dict.fromkeys(FINGERS, OPEN)
FIST = dict.fromkeys(FINGERS, CURLED)


@pytest.fixture(autouse=True)
def no_user_file(tmp_path, monkeypatch):
    """Testler kullanıcının öğrettiği işareti (~/.tkt) okumasın / yazmasın."""
    monkeypatch.setitem(CFG, "file", str(tmp_path / "hand-gate.json"))


CFG = dict(load_control_config()["hand_gate"])


def cfg(**over):
    return {**CFG, **over}


def frame(t, curls=None, confidence=0.9):
    if curls is None:
        return HandFrame(t, int(t * 100), False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {n: Finger((0.5, 0.5), None, curls[n], None) for n in FINGERS}
    return HandFrame(t, int(t * 100), True, fingers, 0.5, (0.5, 0.5), 0.2, confidence)


def hold(gate, t0, seconds, curls, fps=30):
    """`curls` pozunu `seconds` tutar; (son zaman, kapının çevrildiği durumlar) döner."""
    flips, t = [], t0
    for i in range(round(seconds * fps)):
        t = t0 + (i + 1) / fps
        got = gate.update(frame(t, curls))
        if got is not None:
            flips.append(got)
    return t, flips


def test_starts_closed_and_sign_opens_then_closes():
    g = HandGate(cfg(), 0.5)
    assert g.active is False
    t, flips = hold(g, 0.0, 1.0, SIGN)
    assert flips == [True] and g.active
    t, flips = hold(g, t, 3.0, SIGN)          # tutmaya devam: ikinci kez saymaz
    assert flips == [] and g.active
    t, _ = hold(g, t, 0.5, FLAT)              # bırak
    t, flips = hold(g, t, 1.0, SIGN)
    assert flips == [False] and not g.active


def test_short_sign_does_not_toggle():
    g = HandGate(cfg(), 0.5)
    t, flips = hold(g, 0.0, cfg()["hold_ms"] / 1000 - 0.15, SIGN)
    assert flips == [] and 0.0 < g.progress < 1.0
    t, flips = hold(g, t, 0.5, FLAT)
    assert flips == [] and g.progress == 0.0 and not g.active


def test_other_poses_do_not_toggle():
    g = HandGate(cfg(), 0.5)
    t = 0.0
    for pose in (POINT, FLAT, FIST, None):
        t, flips = hold(g, t, 2.0, pose)
        assert flips == []
    assert not g.active


def test_brief_tracking_dropout_keeps_the_sign():
    g = HandGate(cfg(), 0.5)
    t, _ = hold(g, 0.0, 0.3, SIGN)
    t, _ = hold(g, t, 0.1, None)              # 100 ms el kayboldu (< grace)
    t, flips = hold(g, t, 0.3, SIGN)
    assert flips == [True]


def test_long_dropout_restarts_the_hold():
    g = HandGate(cfg(), 0.5)
    t, _ = hold(g, 0.0, 0.4, SIGN)
    t, _ = hold(g, t, 0.4, None)
    t, flips = hold(g, t, 0.4, SIGN)
    assert flips == []
    t, flips = hold(g, t, 0.4, SIGN)
    assert flips == [True]


def test_low_confidence_sign_is_ignored():
    g = HandGate(cfg(), 0.5)
    for i in range(60):
        assert g.update(frame(i / 30, SIGN, confidence=0.2)) is None
    assert not g.active


def test_disabled_gate_is_always_active():
    g = HandGate(cfg(enabled=False), 0.5)
    assert g.active
    _, flips = hold(g, 0.0, 2.0, SIGN)
    assert flips == [] and g.active


# Kullanıcının işareti: yüzük parmağı yarım kıvrık (sabit eşik 0,55'i geçmiyor).
MY_SIGN = {"index": 0.02, "middle": 0.01, "ring": 0.45, "pinky": 0.75}


def test_taught_sign_replaces_fixed_thresholds(tmp_path):
    g = HandGate(cfg(), 0.5)
    _, flips = hold(g, 0.0, 2.0, MY_SIGN)
    assert flips == []                        # sabit eşiklerle tanınmıyor
    ranges = learn_ranges([MY_SIGN] * 20, cfg()["teach_margin"])
    save_ranges(tmp_path / "hand-gate.json", ranges)
    g = HandGate(cfg(), 0.5)                  # dosyadan okunur
    for n, (lo, hi) in ranges.items():
        assert g.ranges[n] == (pytest.approx(lo, abs=1e-3), pytest.approx(hi, abs=1e-3))
    t, flips = hold(g, 0.0, 1.0, MY_SIGN)
    assert flips == [True]
    for pose in (POINT, FLAT, FIST):          # öteki pozlar tetiklemez
        t, flips = hold(g, t, 2.0, pose)
        assert flips == []


def test_learn_ranges_open_ended_at_extremes():
    samples = [{"index": 0.02 + i * 0.001, "middle": 0.5, "ring": 0.95, "pinky": 0.4 + i * 0.01}
               for i in range(20)]
    r = learn_ranges(samples, 0.1)
    assert r["index"][0] == -math.inf
    assert r["index"][1] == pytest.approx(0.02 + 0.0179 + 0.1, abs=0.01)
    assert r["middle"] == (pytest.approx(0.4), pytest.approx(0.6))
    assert r["ring"][1] == math.inf
    assert r["pinky"][0] < 0.4 < 0.59 < r["pinky"][1]


def test_corrupt_sign_file_falls_back(tmp_path):
    (tmp_path / "hand-gate.json").write_text("{bozuk")
    assert HandGate(cfg(), 0.5).ranges is None
    assert load_ranges(tmp_path / "yok.json") is None
