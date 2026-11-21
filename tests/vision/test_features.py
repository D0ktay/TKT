import math

import numpy as np
import pytest

from tkt.core.types import FINGERS, SCHEMA_VERSION, SourceInfo
from tkt.vision import features
from tkt.vision.features import build_frame, curl_raw, thumb_raw
from tkt.vision.hand_calib import DEFAULT, HandCalibration
from tkt.vision.tracker import RawHand

INFO = SourceInfo(schema=SCHEMA_VERSION, width=640, height=512, fps=60.0, hand="right")
PORTRAIT = SourceInfo(schema=SCHEMA_VERSION, width=512, height=640, fps=60.0, hand="right")

# Parmak kökleri (dünya, metre): bilekten 9 cm ileride, yan yana; orta kök tam karşıda.
MCP_X = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
SEG = (0.04, 0.025, 0.02)  # proksimal, orta, distal boğum boyları


def _rot_x(v: np.ndarray, a: float) -> np.ndarray:
    """`v`'yi x ekseni etrafında `a` radyan döndürür (bükülme y-z düzleminde)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def make_hand(pip: float | dict[str, float] = 0.0, dip: float | dict[str, float] = 0.0,
              thumb_dist: float = 0.07, score: float = 0.95,
              center: tuple[float, float] = (0.5, 0.6), scale: float = 2.0) -> RawHand:
    """Yapay 21 nokta. `pip`, `dip`: orta ve uç eklem bükülmesi (radyan).

    Görüntü noktaları dünya noktalarının x-y izdüşümüdür: ileri (+y dünya) görüntüde yukarı
    (y azalır). `center` bileğin görüntü konumu, `scale` metre → görüntü birimi.
    """
    w = np.zeros((21, 3))
    w[0] = (0.0, 0.0, 0.0)
    for name, (mcp, p_i, d_i, t_i) in features.FINGER_JOINTS.items():
        a = pip[name] if isinstance(pip, dict) else pip
        b = dip[name] if isinstance(dip, dict) else dip
        w[mcp] = (MCP_X[name], 0.09, 0.0)
        d0 = np.array([0.0, 1.0, 0.0])
        w[p_i] = w[mcp] + SEG[0] * d0
        d1 = _rot_x(d0, -a)
        w[d_i] = w[p_i] + SEG[1] * d1
        d2 = _rot_x(d0, -(a + b))
        w[t_i] = w[d_i] + SEG[2] * d2
    # Başparmak: 1–3 bilekten sola doğru, uç (4) işaret kökünden `thumb_dist` uzakta.
    w[1] = (-0.03, 0.02, 0.0)
    w[2] = (-0.05, 0.04, 0.0)
    w[3] = (-0.06, 0.06, 0.0)
    w[4] = w[5] + np.array([-thumb_dist, 0.0, 0.0])
    img = np.stack([center[0] + w[:, 0] * scale, center[1] - w[:, 1] * scale], axis=1)
    return RawHand(image_pts=img, world_pts=w, score=score)


# --------------------------------------------------------------------------- curl_raw


def test_straight_finger_curl_raw_zero():
    hand = make_hand()
    for name in FINGERS:
        assert curl_raw(hand, name) == pytest.approx(0.0, abs=1e-6)


def test_curl_raw_sums_pip_and_dip():
    hand = make_hand(pip=1.7, dip=1.2)
    for name in FINGERS:
        assert curl_raw(hand, name) == pytest.approx(2.9, abs=1e-6)


def test_curl_raw_ignores_mcp_flexion():
    """Kök eklemden bükülme (lift) curl_raw'u değiştirmez."""
    hand = make_hand(pip=0.5, dip=0.3)
    w = hand.world_pts.copy()
    mcp, p, d, t = features.FINGER_JOINTS["index"]
    # İşaret parmağını kökten 30° kaldır: tüm parmak kök etrafında döner.
    for i in (p, d, t):
        w[i] = w[mcp] + _rot_x(w[i] - w[mcp], 0.5)
    lifted = RawHand(image_pts=hand.image_pts, world_pts=w, score=hand.score)
    assert curl_raw(lifted, "index") == pytest.approx(0.8, abs=1e-6)


def test_curl_raw_unknown_finger():
    with pytest.raises(ValueError):
        curl_raw(make_hand(), "thumb")


# --------------------------------------------------------------------------- curl (kalibre)


def test_curl_open_zero_fist_one_with_default():
    open_f = build_frame(make_hand(pip=0.1, dip=0.1), 0.0, 0, DEFAULT, INFO)
    fist_f = build_frame(make_hand(pip=1.75, dip=1.25), 0.0, 1, DEFAULT, INFO)
    for name in FINGERS:
        assert open_f.fingers[name].curl == pytest.approx(0.0, abs=1e-6)
        assert fist_f.fingers[name].curl == pytest.approx(1.0, abs=1e-6)


def test_calibration_maps_into_unit_range():
    calib = HandCalibration(
        open_ref={"index": 0.4, "middle": 0.2, "ring": 0.3, "pinky": 0.1},
        fist_ref={"index": 2.4, "middle": 2.2, "ring": 2.3, "pinky": 2.1},
        thumb_closed=0.35, thumb_open=0.8, tap_ref=None,
    )
    mid = {n: calib.open_ref[n] + 0.5 * (calib.fist_ref[n] - calib.open_ref[n]) for n in FINGERS}
    f = build_frame(make_hand(pip=mid, dip=0.0), 0.0, 0, calib, INFO)
    for name in FINGERS:
        assert f.fingers[name].curl == pytest.approx(0.5, abs=1e-6)
    # Aralık dışı ham değerler 0–1'e kırpılır.
    over = build_frame(make_hand(pip=2.0, dip=1.0), 0.0, 0, calib, INFO)
    under = build_frame(make_hand(), 0.0, 0, calib, INFO)
    for name in FINGERS:
        assert over.fingers[name].curl == 1.0
        assert under.fingers[name].curl == 0.0


def test_degenerate_calibration_does_not_divide_by_zero():
    calib = HandCalibration(
        open_ref=dict.fromkeys(FINGERS, 1.0), fist_ref=dict.fromkeys(FINGERS, 1.0),
        thumb_closed=0.5, thumb_open=0.5, tap_ref=None,
    )
    f = build_frame(make_hand(pip=2.0), 0.0, 0, calib, INFO)
    assert all(0.0 <= f.fingers[n].curl <= 1.0 for n in FINGERS)
    assert 0.0 <= f.thumb_out <= 1.0


# --------------------------------------------------------------------------- thumb


def test_thumb_raw_is_distance_over_hand_size():
    hand = make_hand(thumb_dist=0.045)
    hand_size = 0.09  # bilek–orta kök
    assert thumb_raw(hand) == pytest.approx(0.045 / hand_size, rel=1e-6)


def test_thumb_raw_scale_invariant():
    hand = make_hand(thumb_dist=0.05)
    big = RawHand(image_pts=hand.image_pts, world_pts=hand.world_pts * 1.3, score=hand.score)
    assert thumb_raw(big) == pytest.approx(thumb_raw(hand), rel=1e-9)


def test_thumb_out_open_and_closed():
    open_f = build_frame(make_hand(thumb_dist=0.075), 0.0, 0, DEFAULT, INFO)
    closed_f = build_frame(make_hand(thumb_dist=0.025), 0.0, 0, DEFAULT, INFO)
    assert open_f.thumb_out > 0.8
    assert closed_f.thumb_out < 0.2
    assert 0.0 <= closed_f.thumb_out <= open_f.thumb_out <= 1.0


# --------------------------------------------------------------------------- konumlar


def test_img_palm_and_fields():
    hand = make_hand(pip=0.3)
    f = build_frame(hand, 1.25, 7, DEFAULT, INFO)
    assert f.present and f.t == 1.25 and f.frame_id == 7
    assert set(f.fingers) == set(FINGERS)
    for name, idx in (("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20)):
        assert f.fingers[name].img == pytest.approx(tuple(hand.image_pts[idx]))
        assert f.fingers[name].surface is None
        assert f.fingers[name].lift is None
    expected_palm = hand.image_pts[[0, 5, 9, 13, 17]].mean(axis=0)
    assert f.palm == pytest.approx(tuple(expected_palm))
    assert f.confidence == pytest.approx(0.95)


def test_lift_written_when_given():
    lift = {"index": 0.7, "middle": 0.0, "ring": 0.1, "pinky": 0.2}
    f = build_frame(make_hand(), 0.0, 0, DEFAULT, INFO, lift=lift)
    for name in FINGERS:
        assert f.fingers[name].lift == pytest.approx(lift[name])


def test_hand_size_uses_aspect_ratio():
    # Bilek (0,5; 0,8), orta kök 0,3 görüntü birimi yukarıda (dikey).
    hand = make_hand(center=(0.5, 0.8), scale=0.3 / 0.09)
    assert hand.image_pts[9, 1] == pytest.approx(0.5)
    # Portre 512x640: 0,3 * 640 px = 192 px; 192 / 512 = 0,375.
    f = build_frame(hand, 0.0, 0, DEFAULT, PORTRAIT)
    assert f.hand_size == pytest.approx(0.3 * 640 / 512)
    # Yatay 640x512: 0,3 * 512 / 640 = 0,24.
    f2 = build_frame(hand, 0.0, 0, DEFAULT, INFO)
    assert f2.hand_size == pytest.approx(0.3 * 512 / 640)


def test_hand_size_horizontal_segment():
    hand = make_hand()
    img = hand.image_pts.copy()
    img[0] = (0.2, 0.5)
    img[9] = (0.5, 0.5)
    f = build_frame(RawHand(img, hand.world_pts, 0.9), 0.0, 0, DEFAULT, PORTRAIT)
    assert f.hand_size == pytest.approx(0.3)


# --------------------------------------------------------------------------- el yok


def test_no_hand_gives_empty_frame():
    f = build_frame(None, 3.5, 42, DEFAULT, INFO, lift={"index": 1.0}, prev_hand_size=0.2)
    assert f.present is False
    assert f.fingers == {}
    assert f.t == 3.5 and f.frame_id == 42
    assert f.thumb_out == 0.0 and f.palm == (0.0, 0.0)
    assert f.hand_size == 0.0 and f.confidence == 0.0


# --------------------------------------------------------------------------- güven


def test_confidence_drops_on_hand_size_jump():
    hand = make_hand()
    base = build_frame(hand, 0.0, 0, DEFAULT, INFO)
    steady = build_frame(hand, 0.0, 1, DEFAULT, INFO, prev_hand_size=base.hand_size * 1.1)
    jump_up = build_frame(hand, 0.0, 1, DEFAULT, INFO, prev_hand_size=base.hand_size / 1.3)
    jump_down = build_frame(hand, 0.0, 1, DEFAULT, INFO, prev_hand_size=base.hand_size * 1.3)
    assert steady.confidence == pytest.approx(base.confidence)
    assert jump_up.confidence < base.confidence
    assert jump_down.confidence < base.confidence


def test_confidence_drops_near_edge():
    centered = build_frame(make_hand(center=(0.5, 0.6)), 0.0, 0, DEFAULT, INFO)
    # Bilek alt kenarda.
    at_edge = build_frame(make_hand(center=(0.5, 1.0)), 0.0, 0, DEFAULT, INFO)
    # Bir kısmı kadraj dışında (sol).
    outside = build_frame(make_hand(center=(0.0, 0.6)), 0.0, 0, DEFAULT, INFO)
    assert centered.confidence == pytest.approx(0.95)
    assert at_edge.confidence < centered.confidence
    assert outside.confidence < centered.confidence
    assert outside.confidence == pytest.approx(0.95 * features.EDGE_MIN_FACTOR)


def test_confidence_stays_in_unit_range():
    f = build_frame(make_hand(score=1.5), 0.0, 0, DEFAULT, INFO)
    assert 0.0 <= f.confidence <= 1.0
