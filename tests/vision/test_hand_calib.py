import json
import math

import numpy as np
import pytest

from tkt.core.types import FINGERS, SCHEMA_VERSION, SourceInfo
from tkt.vision import hand_calib
from tkt.vision.features import FINGER_JOINTS, build_frame, curl_raw, thumb_raw
from tkt.vision.hand_calib import (
    DEFAULT,
    STEPS,
    CalibrationSession,
    HandCalibration,
    fit_hand,
    load,
    save,
)
from tkt.vision.tracker import RawHand

INFO = SourceInfo(schema=SCHEMA_VERSION, width=640, height=512, fps=60.0, hand="right")
MCP_X = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
SEG = (0.04, 0.025, 0.02)


def _rot_x(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def make_hand(pip: float = 0.0, dip: float = 0.0, thumb_dist: float = 0.07) -> RawHand:
    """Yapay 21 nokta: tüm parmaklarda `pip`/`dip` bükülmesi (radyan), başparmak ucu
    işaret kökünden `thumb_dist` metre uzakta. El boyu (bilek–orta kök) 0,09 m."""
    w = np.zeros((21, 3))
    for name, (mcp, p_i, d_i, t_i) in FINGER_JOINTS.items():
        w[mcp] = (MCP_X[name], 0.09, 0.0)
        d0 = np.array([0.0, 1.0, 0.0])
        w[p_i] = w[mcp] + SEG[0] * d0
        w[d_i] = w[p_i] + SEG[1] * _rot_x(d0, -pip)
        w[t_i] = w[d_i] + SEG[2] * _rot_x(d0, -(pip + dip))
    w[1] = (-0.03, 0.02, 0.0)
    w[2] = (-0.05, 0.04, 0.0)
    w[3] = (-0.06, 0.06, 0.0)
    w[4] = w[5] + np.array([-thumb_dist, 0.0, 0.0])
    # Eğik izdüşüm (görüntü x = dünya x + z): bükülme görüntüde görünür.
    img = np.stack([0.5 + (w[:, 0] + w[:, 2]) * 2.0, 0.6 - w[:, 1] * 2.0], axis=1)
    return RawHand(image_pts=img, world_pts=w, score=0.95)


# Kişinin eli: açıkken parmaklar biraz bükülü, yumrukta DEFAULT'tan az bükülü.
def _open(k: float = 0.0) -> RawHand:
    return make_hand(pip=0.3 + k, dip=0.2, thumb_dist=0.065 + k / 10)


def _fist(k: float = 0.0) -> RawHand:
    return make_hand(pip=1.5 + k, dip=0.9, thumb_dist=0.03)


def _thumb_closed(k: float = 0.0) -> RawHand:
    return make_hand(pip=0.3, dip=0.2, thumb_dist=0.02 + k / 10)


def _calib() -> HandCalibration:
    return HandCalibration(
        open_ref={"index": 0.21, "middle": 0.18, "ring": 0.25, "pinky": 0.3},
        fist_ref={"index": 2.9, "middle": 3.1, "ring": 3.05, "pinky": 2.7},
        thumb_closed=0.31,
        thumb_open=0.86,
        tap_ref={"index": 0.4, "middle": 0.35, "ring": 0.3, "pinky": 0.28},
        angle_sign=-1.0,
    )


def test_default_reasonable():
    for name in FINGERS:
        assert DEFAULT.open_ref[name] == pytest.approx(0.2)
        assert DEFAULT.fist_ref[name] == pytest.approx(3.0)
    assert 0.0 < DEFAULT.thumb_closed < DEFAULT.thumb_open < 1.5
    assert DEFAULT.tap_ref is None and DEFAULT.angle_sign == 1.0


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "a" / "b" / "calibration.json"
    c = _calib()
    save(c, path)
    assert path.is_file()
    assert load(path) == c


def test_save_load_roundtrip_without_tap_ref(tmp_path):
    path = tmp_path / "calibration.json"
    save(DEFAULT, path)
    assert load(path) == DEFAULT


def test_save_overwrites(tmp_path):
    path = tmp_path / "calibration.json"
    save(DEFAULT, path)
    save(_calib(), path)
    assert load(path) == _calib()
    assert [p.name for p in tmp_path.iterdir()] == ["calibration.json"]


def test_load_missing_returns_none(tmp_path):
    assert load(tmp_path / "yok.json") is None


def test_tilde_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    rel = hand_calib.Path("~") / ".tkt" / "calibration.json"
    save(_calib(), rel)
    assert (tmp_path / ".tkt" / "calibration.json").is_file()
    assert load(rel) == _calib()


def test_load_without_angle_sign_uses_default(tmp_path):
    path = tmp_path / "c.json"
    d = {
        "open_ref": dict.fromkeys(FINGERS, 0.2),
        "fist_ref": dict.fromkeys(FINGERS, 3.0),
        "thumb_closed": 0.35,
        "thumb_open": 0.8,
        "tap_ref": None,
    }
    path.write_text(json.dumps(d), encoding="utf-8")
    c = load(path)
    assert c is not None and c.angle_sign == 1.0


def test_load_corrupt_raises(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{bozuk", encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)
    path.write_text(json.dumps({"open_ref": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


# --------------------------------------------------------------------------- fit_hand


def test_fit_hand_uses_median():
    jitter = (-0.02, 0.0, 0.01, 0.02)
    # Her listede bir aykırı örnek (ör. takip hatası): medyanı yalnızca bir sıra kaydırır.
    # Açık/kapalı listelerde aykırı yukarıda (medyan = k 0,01), yumrukta aşağıda (k 0,0).
    open_s = [_open(k) for k in jitter] + [make_hand(pip=2.5, dip=1.5, thumb_dist=0.2)]
    fist_s = [_fist(k) for k in jitter] + [make_hand()]
    closed_s = [_thumb_closed(k) for k in jitter] + [make_hand(thumb_dist=0.3)]
    c = fit_hand(open_s, fist_s, closed_s)
    for name in FINGERS:
        assert c.open_ref[name] == pytest.approx(curl_raw(_open(0.01), name), abs=1e-6)
        assert c.fist_ref[name] == pytest.approx(curl_raw(_fist(0.0), name), abs=1e-6)
    assert c.thumb_open == pytest.approx(thumb_raw(_open(0.01)), abs=1e-6)
    assert c.thumb_closed == pytest.approx(thumb_raw(_thumb_closed(0.01)), abs=1e-6)
    assert c.tap_ref is None and c.angle_sign == 1.0


def test_fit_hand_maps_poses_to_unit_range():
    c = fit_hand([_open()] * 3, [_fist()] * 3, [_thumb_closed()] * 3,
                 aspect=INFO.width / INFO.height)
    open_f = build_frame(_open(), 0.0, 0, c, INFO)
    fist_f = build_frame(_fist(), 0.0, 1, c, INFO)
    closed_f = build_frame(_thumb_closed(), 0.0, 2, c, INFO)
    for name in FINGERS:
        assert open_f.fingers[name].curl == pytest.approx(0.0, abs=1e-6)
        assert fist_f.fingers[name].curl == pytest.approx(1.0, abs=1e-6)
    assert open_f.thumb_out == pytest.approx(1.0, abs=1e-6)
    assert closed_f.thumb_out == pytest.approx(0.0, abs=1e-6)
    # DEFAULT ile bu kişinin yumruğu 1'e ulaşmaz; kalibrasyonun farkı budur.
    assert build_frame(_fist(), 0.0, 0, DEFAULT, INFO).fingers["index"].curl < 0.9


def test_fit_hand_result_roundtrips(tmp_path):
    c = fit_hand([_open()], [_fist()], [_thumb_closed()])
    path = tmp_path / "calibration.json"
    save(c, path)
    assert load(path) == c


@pytest.mark.parametrize("which", [0, 1, 2])
def test_fit_hand_empty_raises(which):
    lists = [[_open()], [_fist()], [_thumb_closed()]]
    lists[which] = []
    with pytest.raises(ValueError):
        fit_hand(*lists)


def test_fit_hand_fist_not_above_open_raises():
    with pytest.raises(ValueError, match="yumruk"):
        fit_hand([_fist()], [_open()], [_thumb_closed()])
    with pytest.raises(ValueError):
        fit_hand([_open()], [_open()], [_thumb_closed()])


def test_fit_hand_thumb_not_open_raises():
    with pytest.raises(ValueError, match="başparmak"):
        fit_hand([_open()], [_fist()], [make_hand(thumb_dist=0.09)])


# --------------------------------------------------------------------------- CalibrationSession


def _pose(step: str) -> RawHand:
    return {"open": _open, "fist": _fist, "thumb_closed": _thumb_closed}[step]()


def test_session_steps_advance_with_time():
    s = CalibrationSession(step_s=2.0)
    assert s.current_step == "open" and s.progress == 0.0 and not s.done
    t0 = 100.0
    s.add(_open(), t0)
    s.add(_open(), t0 + 1.0)
    assert s.current_step == "open"
    assert s.progress == pytest.approx(0.5)
    assert s.total_progress == pytest.approx(1 / 6)
    s.add(_fist(), t0 + 2.0)
    assert s.current_step == "fist" and s.progress == pytest.approx(0.0)
    s.add(None, t0 + 4.5)
    assert s.current_step == "thumb_closed" and s.progress == pytest.approx(0.25)
    s.add(None, t0 + 6.0)
    assert s.done and s.current_step is None and s.progress == 1.0
    assert s.total_progress == 1.0


def test_session_collects_and_produces_result():
    s = CalibrationSession(step_s=0.5)
    t = 0.0
    dt = 1 / 30
    while not s.done:
        step = s.current_step
        assert step is not None
        # Her 5. kare el yok: atlanır.
        s.add(None if round(t / dt) % 5 == 0 else _pose(step), t)
        t += dt
    for step in STEPS:
        assert 0 < len(s.samples[step]) < 0.5 / dt + 1
    c = s.result()
    assert c == fit_hand([_open()], [_fist()], [_thumb_closed()])
    # Bittikten sonra gelen örnekler yok sayılır.
    counts = {k: len(v) for k, v in s.samples.items()}
    s.add(_fist(), t + 1.0)
    assert {k: len(v) for k, v in s.samples.items()} == counts


def test_session_none_samples_skipped():
    s = CalibrationSession(step_s=1.0)
    for i in range(30):
        s.add(None, i * 0.1)
    assert all(len(v) == 0 for v in s.samples.values())
    assert s.current_step == "thumb_closed"


def test_session_result_before_done_raises():
    s = CalibrationSession()
    s.add(_open(), 0.0)
    with pytest.raises(ValueError):
        s.result()


def test_session_empty_step_raises():
    """El hiç görünmeyen adım: sonuç açık hata verir."""
    s = CalibrationSession(step_s=1.0)
    s.add(_open(), 0.0)
    s.add(None, 1.5)
    s.add(_thumb_closed(), 2.5)
    s.add(None, 3.0)
    assert s.done
    with pytest.raises(ValueError):
        s.result()


def test_session_default_step_and_bad_step():
    assert CalibrationSession().step_s == hand_calib.STEP_SECONDS == 2.0
    with pytest.raises(ValueError):
        CalibrationSession(step_s=0.0)


def test_session_time_going_back_does_not_rewind():
    s = CalibrationSession(step_s=1.0)
    s.add(_open(), 0.0)
    s.add(_fist(), 1.2)
    s.add(_open(), 0.5)  # geri giden zaman: hâlâ "fist" adımı
    assert s.current_step == "fist"
    assert len(s.samples["fist"]) == 2


# --------------------------------------------------------------------------- fit_taps


def _lifted(lift: dict[str, float], flip_z: bool = False) -> RawHand:
    """Düz el; `lift[f]` radyan kadar parmak kökten +z yönüne kalkık."""
    h = make_hand()
    w = h.world_pts.copy()
    for f, a in lift.items():
        mcp, p, d, t = FINGER_JOINTS[f]
        for i in (p, d, t):
            w[i] = w[mcp] + _rot_x(w[i] - w[mcp], a)
    if flip_z:
        w[:, 2] *= -1
    img = np.stack([0.5 + w[:, 0] * 2.0, 0.6 - w[:, 1] * 2.0], axis=1)
    return RawHand(image_pts=img, world_pts=w, score=0.95)


def _tap_samples(kinds, angle=0.35, flip_z=False, fps=60.0, window=1.5, period=2.0):
    """Bip pencereleri (window sn, period arayla); bipten 0,25 sn sonra 0,2 sn kalkış."""
    out = []
    for k, kind in enumerate(kinds):
        t0 = 10.0 + k * period
        fingers = ("index", "middle") if kind == "tap2" else ("index",)
        for i in range(int(window * fps)):
            x = i / fps
            a = angle if 0.25 <= x < 0.45 else 0.0
            out.append((t0 + x, _lifted({f: a for f in fingers}, flip_z)))
    return out


def _raw_peak(angle=0.35, sign=-1.0, finger="index"):
    from tkt.vision.lift import raw_signals

    a0, l0 = raw_signals(_lifted({}))[finger]
    a1, l1 = raw_signals(_lifted({finger: angle}))[finger]
    return sign * (a1 - a0) + (l0 - l1)


def test_fit_taps_sign_and_ref_tap1():
    calib = fit_hand([_open()], [_fist()], [_thumb_closed()])
    out = hand_calib.fit_taps(calib, _tap_samples(["tap1"] * 6))
    assert out.angle_sign == -1.0
    expected = _raw_peak()
    # yalnızca işaret kalktı: diğer parmaklar aynı değerle doldurulur
    for f in FINGERS:
        assert out.tap_ref[f] == pytest.approx(expected, rel=1e-6)
    # diğer alanlar korunur
    assert out.open_ref == calib.open_ref and out.thumb_open == calib.thumb_open


def test_fit_taps_tap2_and_flipped_direction():
    samples = _tap_samples(["tap1", "tap2", "tap2", "tap1", "tap2"], angle=0.3, flip_z=True)
    out = hand_calib.fit_taps(DEFAULT, samples)
    assert out.angle_sign == 1.0
    # z aynalı elde de tepe aynı büyüklükte
    assert out.tap_ref["index"] == pytest.approx(_raw_peak(0.3), rel=1e-6)
    assert out.tap_ref["middle"] == pytest.approx(_raw_peak(0.3, finger="middle"), rel=1e-6)
    fill = float(np.median([out.tap_ref["index"], out.tap_ref["middle"]]))
    assert out.tap_ref["ring"] == pytest.approx(fill)
    assert out.tap_ref["pinky"] == pytest.approx(fill)


def test_fit_taps_uses_median_of_peaks():
    samples = []
    for k, a in enumerate((0.2, 0.3, 0.5)):
        samples += [(t + k * 2.0, h) for t, h in _tap_samples(["tap1"], angle=a)]
    out = hand_calib.fit_taps(DEFAULT, samples)
    assert out.tap_ref["index"] == pytest.approx(_raw_peak(0.3), rel=1e-6)


def test_fit_taps_majority_sign():
    samples = _tap_samples(["tap1"] * 3)
    samples += [(t + 6.0, h) for t, h in _tap_samples(["tap1"], flip_z=True)]
    out = hand_calib.fit_taps(DEFAULT, samples)
    assert out.angle_sign == -1.0


def test_fit_taps_empty_or_flat_raises():
    with pytest.raises(ValueError):
        hand_calib.fit_taps(DEFAULT, [])
    flat = [(10.0 + i / 60, _lifted({})) for i in range(60)]
    with pytest.raises(ValueError):
        hand_calib.fit_taps(DEFAULT, flat)


def _contiguous_taps(angles, fps=60.0, period=2.0, t0=10.0):
    """Kesintisiz örnek akışı (debug penceresi gibi): her `period` sn'de bir pencere."""
    out = []
    windows = []
    for k, a in enumerate(angles):
        start = t0 + k * period
        windows.append((start, start + period))
        for i in range(int(period * fps)):
            x = i / fps
            out.append((start + x, _lifted({"index": a if 0.25 <= x < 0.45 else 0.0})))
    return out, windows


def test_fit_taps_explicit_windows_on_contiguous_stream():
    samples, windows = _contiguous_taps((0.2, 0.3, 0.5, 0.3, 0.2))
    out = hand_calib.fit_taps(DEFAULT, samples, windows=windows)
    # pencere başına tepe; medyan 0,3'ün tepesi (tek pencerede max 0,5 çıkardı)
    assert out.tap_ref["index"] == pytest.approx(_raw_peak(0.3), rel=1e-6)
    assert out.angle_sign == -1.0


def test_fit_taps_windows_per_window_baseline():
    # her pencere farklı duruşta başlıyor (tüm parmaklar biraz kalkık); taban pencereye göre
    samples, windows = [], []
    for k, base in enumerate((0.0, 0.15, 0.3)):
        start = 10.0 + 2.0 * k
        windows.append((start, start + 2.0))
        for i in range(120):
            x = i / 60
            a = 0.3 if 0.25 <= x < 0.45 else 0.0
            pose = {f: base for f in FINGER_JOINTS}
            pose["index"] = base + a
            samples.append((start + x, _lifted(pose)))
    out = hand_calib.fit_taps(DEFAULT, samples, windows=windows)
    from tkt.vision.lift import raw_signals

    peaks = []
    for base in (0.0, 0.15, 0.3):
        pose = {f: base for f in FINGER_JOINTS}
        a0, l0 = raw_signals(_lifted(pose))["index"]
        pose["index"] = base + 0.3
        a1, l1 = raw_signals(_lifted(pose))["index"]
        peaks.append(-(a1 - a0) + (l0 - l1))
    assert out.tap_ref["index"] == pytest.approx(float(np.median(peaks)), rel=1e-6)


def test_fit_taps_windows_ignore_outside_and_empty():
    samples, windows = _contiguous_taps((0.3, 0.3, 0.3))
    noise = [(5.0 + i / 60, _lifted({"index": 0.6 if i == 10 else 0.0})) for i in range(30)]
    out = hand_calib.fit_taps(DEFAULT, noise + samples,
                              windows=windows + [(100.0, 102.0)])   # boş pencere
    assert out.tap_ref["index"] == pytest.approx(_raw_peak(0.3), rel=1e-6)
    with pytest.raises(ValueError):
        hand_calib.fit_taps(DEFAULT, samples, windows=[(100.0, 102.0)])


def test_fit_taps_aspect_matches_pixel_geometry():
    aspect = 512 / 640

    def rot(h):
        # parmakları görüntüde köklerinin çevresinde 90° çevir; bilek→kök dikey kalır
        img = np.array(h.image_pts, dtype=float)
        for mcp, *rest in FINGER_JOINTS.values():
            for i in rest:
                d = img[i] - img[mcp]
                img[i] = img[mcp] + (-d[1], d[0])
        return RawHand(image_pts=img, world_pts=h.world_pts, score=h.score)

    def scale(h):
        img = np.asarray(h.image_pts, dtype=float).copy()
        img[:, 0] *= aspect
        return RawHand(image_pts=img, world_pts=h.world_pts, score=h.score)

    samples = [(t, rot(h)) for t, h in _tap_samples(["tap1"] * 4)]
    a = hand_calib.fit_taps(DEFAULT, samples, aspect=aspect)
    b = hand_calib.fit_taps(DEFAULT, [(t, scale(h)) for t, h in samples])
    assert a.tap_ref == pytest.approx(b.tap_ref)
    c = hand_calib.fit_taps(DEFAULT, samples)
    assert c.tap_ref["index"] != pytest.approx(a.tap_ref["index"])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_fit_taps_rejects_bad_aspect(bad):
    with pytest.raises(ValueError, match="aspect"):
        hand_calib.fit_taps(DEFAULT, _tap_samples(["tap1"] * 2), aspect=bad)
