import math
from dataclasses import replace

import numpy as np
import pytest

from tkt.core.io import Prompt, prompts_path, write_prompts
from tkt.core.types import FINGERS, SCHEMA_VERSION, SourceInfo
from tkt.tools.record import RawSample, write_raw
from tkt.vision import lift as lift_mod
from tkt.vision.features import FINGER_JOINTS
from tkt.vision.hand_calib import DEFAULT, save
from tkt.vision.lift import LIFT_FREEZE, LiftEstimator, curls_from_hand, raw_signals
from tkt.vision.tracker import RawHand

FPS = 60.0
DT = 1.0 / FPS
CFG = {"w_angle": 1.0, "w_length": 1.0, "baseline_s": 1.0, "curl_rate_suppress": 3.0}
INFO = SourceInfo(schema=SCHEMA_VERSION, width=640, height=512, fps=FPS, hand="right")

MCP_X = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
SEG = (0.04, 0.025, 0.02)
TAP_ANGLE = 0.35  # rad, ~20°: kökten kalkış


def _rot_x(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def make_hand(lift: dict[str, float] | None = None, curl: float | dict[str, float] = 0.0,
              tilt: float = 0.0, flip_z: bool = False) -> RawHand:
    """Yapay 21 nokta (test_features.make_hand gibi).

    `lift[f]`: parmağın kökten kalkış açısı (radyan, +z yönüne). `curl`: orta ve uç eklemin
    her birinde bükülme (radyan, -z yönüne). `tilt`: tüm parmakların kökten ortak açısı
    (yavaş duruş değişimi). `flip_z`: dünya noktalarını z'de aynalar (diğer yön).
    Görüntü noktaları x-y izdüşümüdür; ileri (+y) görüntüde yukarı.
    """
    lift = lift or {}
    w = np.zeros((21, 3))
    for name, (mcp, p_i, d_i, t_i) in FINGER_JOINTS.items():
        c = curl.get(name, 0.0) if isinstance(curl, dict) else curl
        w[mcp] = (MCP_X[name], 0.09, 0.0)
        d0 = np.array([0.0, 1.0, 0.0])
        rel = [SEG[0] * d0]
        rel.append(rel[0] + SEG[1] * _rot_x(d0, -c))
        rel.append(rel[1] + SEG[2] * _rot_x(d0, -2 * c))
        a = lift.get(name, 0.0) + tilt
        for idx, r in zip((p_i, d_i, t_i), rel, strict=True):
            w[idx] = w[mcp] + _rot_x(r, a)
    w[1] = (-0.03, 0.02, 0.0)
    w[2] = (-0.05, 0.04, 0.0)
    w[3] = (-0.06, 0.06, 0.0)
    w[4] = w[5] + np.array([-0.07, 0.0, 0.0])
    if flip_z:
        w[:, 2] *= -1
    img = np.stack([0.5 + w[:, 0] * 2.0, 0.8 - w[:, 1] * 2.0], axis=1)
    return RawHand(image_pts=img, world_pts=w, score=0.95)


def expected_raw(finger: str = "index", angle: float = TAP_ANGLE) -> float:
    """angle_sign=-1 ile tam kalkışın lift_raw değeri (taban çizgisi düz el)."""
    a0, l0 = raw_signals(make_hand())[finger]
    a1, l1 = raw_signals(make_hand({finger: angle}))[finger]
    return -1.0 * (a1 - a0) + (l0 - l1)


def calib_with_tap_ref(value: float | None = None):
    v = expected_raw() if value is None else value
    return replace(DEFAULT, tap_ref={f: v for f in FINGERS}, angle_sign=-1.0)


def tap_profile(t: float, start: float, rise: float = 0.08, hold: float = 0.1) -> float:
    """0→1→0 yamuk kalkış profili."""
    x = t - start
    if x < 0 or x > 2 * rise + hold:
        return 0.0
    if x < rise:
        return x / rise
    if x < rise + hold:
        return 1.0
    return (2 * rise + hold - x) / rise


def run(est: LiftEstimator, hands: list[RawHand], t0: float = 100.0,
        calib=DEFAULT) -> list[dict[str, float]]:
    return [est.update(h, t0 + i * DT, curls_from_hand(h, calib)) for i, h in enumerate(hands)]


# --------------------------------------------------------------------------- ham sinyaller


def test_raw_signals_flat_hand():
    sig = raw_signals(make_hand())
    for f in FINGERS:
        angle, length = sig[f]
        assert angle == pytest.approx(0.0, abs=1e-9)
        # kök→uç 0,085 m, bilek→orta kök 0,09 m; aynı ölçekle izdüşüm.
        assert length == pytest.approx(0.085 / 0.09, rel=1e-6)


def test_raw_signals_lift_changes_angle_and_length():
    a, length = raw_signals(make_hand({"index": TAP_ANGLE}))["index"]
    # avuç normali -z; +z'ye kalkış negatif açı verir.
    assert a == pytest.approx(-TAP_ANGLE, abs=1e-9)
    assert length == pytest.approx(0.085 * math.cos(TAP_ANGLE) / 0.09, rel=1e-6)


# --------------------------------------------------------------------------- kurucu


def test_requires_tap_ref():
    with pytest.raises(ValueError):
        LiftEstimator(DEFAULT, CFG)


def test_rejects_nonpositive_tap_ref():
    with pytest.raises(ValueError):
        LiftEstimator(calib_with_tap_ref(0.0), CFG)


# --------------------------------------------------------------------------- update


def test_first_frame_is_baseline():
    est = LiftEstimator(calib_with_tap_ref(), CFG)
    out = est.update(make_hand({"index": 0.2}), 0.0, {f: 0.0 for f in FINGERS})
    assert set(out) == set(FINGERS)
    assert all(v == 0.0 for v in out.values())


def test_tap_peaks_at_expected_frames():
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    starts = [1.0, 2.0, 3.0]
    n = int(4.0 * FPS)
    times = [i * DT for i in range(n)]
    hands = [make_hand({"index": TAP_ANGLE * max(tap_profile(t, s) for s in starts)})
             for t in times]
    out = run(est, hands, t0=0.0, calib=calib)
    idx = np.array([o["index"] for o in out])
    others = np.array([[o[f] for f in ("middle", "ring", "pinky")] for o in out])

    for s in starts:
        win = [i for i, t in enumerate(times) if s <= t < s + 0.3]
        peak_i = max(win, key=lambda i: idx[i])
        # tepe tutma bölümünde (kalkıştan 80–180 ms sonra)
        assert s + 0.08 - 1e-6 <= times[peak_i] <= s + 0.18 + 1e-6
        assert idx[peak_i] > 0.95
    # tap dışında ve diğer parmaklarda lift sıfıra yakın
    quiet = [i for i, t in enumerate(times) if all(not (s <= t < s + 0.4) for s in starts)]
    assert idx[quiet].max() < 0.05
    assert others.max() < 1e-6


def test_curl_is_not_lift():
    """Kıvırma (curl hızla artarken) ve sonrasında lift düşük kalır."""
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    hands = [make_hand() for _ in range(60)]
    # 1/6 sn'de tam kıvırma (her eklemde 1,45 rad), sonra 1 sn kıvrık bekle.
    ramp = 10
    hands += [make_hand(curl={"index": 1.45 * (k + 1) / ramp}) for k in range(ramp)]
    hands += [make_hand(curl={"index": 1.45}) for _ in range(60)]
    out = run(est, hands, calib=calib)
    assert max(o["index"] for o in out) < 0.15

    # geometri gerçekten kayıyor: bastırma olmasa lift yüksek olurdu
    no_supp = LiftEstimator(calib, {**CFG, "curl_rate_suppress": 1e9})
    out2 = run(no_supp, hands, calib=calib)
    assert max(o["index"] for o in out2) > 0.5


def test_uncurl_is_not_lift():
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    hands = [make_hand(curl={"index": 1.45}) for _ in range(60)]
    hands += [make_hand(curl={"index": 1.45 * (1 - (k + 1) / 10)}) for k in range(10)]
    hands += [make_hand() for _ in range(60)]
    out = run(est, hands, calib=calib)
    assert max(o["index"] for o in out) < 0.15


def test_slow_posture_change_baseline_follows():
    """Yavaş duruş değişiminde taban çizgisi uyar; lift düşük kalır ve sıfıra döner."""
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    drift_n = int(6 * FPS)
    hands = [make_hand(tilt=-0.3 * (k + 1) / drift_n) for k in range(drift_n)]
    hands += [make_hand(tilt=-0.3) for _ in range(int(5 * FPS))]
    out = run(est, hands, calib=calib)
    idx = [o["index"] for o in out]
    assert max(idx) < LIFT_FREEZE
    assert idx[-1] < 0.01

    # ters yönde (kalkış yönüne) yavaş değişim de sönümlenir
    est.reset()
    hands = [make_hand(tilt=0.3 * (k + 1) / drift_n) for k in range(drift_n)]
    hands += [make_hand(tilt=0.3) for _ in range(int(5 * FPS))]
    out = run(est, hands, calib=calib)
    idx = [o["index"] for o in out]
    assert max(idx) < LIFT_FREEZE
    assert idx[-1] < 0.01


def test_baseline_frozen_during_lift_but_not_forever():
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    hands = [make_hand() for _ in range(30)]
    hands += [make_hand({"index": TAP_ANGLE}) for _ in range(int(6 * FPS))]
    out = run(est, hands, calib=calib)
    idx = [o["index"] for o in out]
    # dondurma: kalkıştan 0,5 sn sonra lift hâlâ tam
    assert idx[30 + 30] > 0.95
    # en çok baseline_s dondurulur; sonra taban çizgisi yetişir
    assert idx[-1] < 0.05


def test_reset_restarts_baseline():
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    run(est, [make_hand() for _ in range(10)], calib=calib)
    lifted = make_hand({"index": TAP_ANGLE})
    assert est.update(lifted, 200.0, curls_from_hand(lifted, calib))["index"] > 0.95
    est.reset()
    assert est.update(lifted, 201.0, curls_from_hand(lifted, calib))["index"] == 0.0


def test_non_increasing_time_does_not_break():
    calib = calib_with_tap_ref()
    est = LiftEstimator(calib, CFG)
    h = make_hand()
    est.update(h, 5.0, {f: 0.0 for f in FINGERS})
    out = est.update(h, 4.0, {f: 0.5 for f in FINGERS})  # dt=0: bastırma hesaplanmaz
    assert all(0.0 <= v <= 1.0 for v in out.values())


# --------------------------------------------------------------------------- --eval


def _write_tap_recording(path, n_prompts=8, kind="tap1", start=10.0, period=2.0):
    prompts = [Prompt(start + k * period, kind) for k in range(n_prompts)]
    fingers = ("index", "middle") if kind == "tap2" else ("index",)
    n = int((start + n_prompts * period + 1.0 - (start - 1.0)) * FPS)
    samples = []
    for i in range(n):
        t = start - 1.0 + i * DT
        # bipten 0,25 sn sonra kalkış
        p = max(tap_profile(t, pr.t + 0.25) for pr in prompts)
        samples.append(RawSample(t, i, make_hand({f: TAP_ANGLE * p for f in fingers})))
    write_raw(path, INFO, samples)
    write_prompts(prompts_path(path), prompts)
    return prompts


def _write_normal_recording(path, seconds=20.0, wiggle=0.03, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    tilt = 0.0
    for i in range(int(seconds * FPS)):
        t = 50.0 + i * DT
        tilt = 0.98 * tilt + rng.normal(0, 0.004)
        h = make_hand({"index": wiggle * math.sin(t * 3)}, tilt=tilt)
        samples.append(RawSample(t, i, None if i % 300 == 299 else h))
    write_raw(path, INFO, samples)


def test_evaluate_ratio(tmp_path):
    tap = tmp_path / "tap.raw.jsonl"
    normal = tmp_path / "normal.raw.jsonl"
    _write_tap_recording(tap)
    _write_normal_recording(normal)
    r = lift_mod.evaluate(tap, normal, DEFAULT, CFG)
    assert r.fitted
    assert r.calib.angle_sign == -1.0
    assert r.calib.tap_ref["index"] == pytest.approx(expected_raw(), rel=0.02)
    assert r.n_prompts == 8 and r.n_missing == 0 and len(r.peaks) == 8
    assert r.fingers == ("index",)
    assert r.peak_median > 0.9
    assert 0 < r.normal_p99 < 0.5
    assert r.ratio == pytest.approx(r.peak_median / r.normal_p99)
    assert r.passed


def test_evaluate_tap2_uses_index_and_middle(tmp_path):
    tap = tmp_path / "tap.raw.jsonl"
    normal = tmp_path / "normal.raw.jsonl"
    _write_tap_recording(tap, n_prompts=5, kind="tap2")
    _write_normal_recording(normal, seconds=5)
    r = lift_mod.evaluate(tap, normal, DEFAULT, CFG)
    assert r.fingers == ("index", "middle")
    assert len(r.peaks) == 10


def test_evaluate_fails_when_normal_looks_like_taps(tmp_path):
    tap = tmp_path / "tap.raw.jsonl"
    normal = tmp_path / "normal.raw.jsonl"
    _write_tap_recording(tap)
    _write_tap_recording(normal, start=60.0)  # "normal" kayıtta da tap var
    r = lift_mod.evaluate(tap, normal, DEFAULT, CFG)
    assert not r.passed


def test_eval_cli(tmp_path, capsys):
    tap = tmp_path / "tap.raw.jsonl"
    normal = tmp_path / "normal.raw.jsonl"
    _write_tap_recording(tap)
    _write_normal_recording(normal)
    calib_path = tmp_path / "calib.json"
    save(DEFAULT, calib_path)  # tap_ref yok → fit_taps
    cfg = tmp_path / "vision.toml"
    cfg.write_text(
        "[hand]\n"
        f"calibration = \"{calib_path.as_posix()}\"\n"
        "[lift]\nw_angle = 1.0\nw_length = 1.0\nbaseline_s = 1.0\ncurl_rate_suppress = 3.0\n",
        encoding="utf-8")
    lift_mod.main(["--eval", str(tap), str(normal), "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "ayrışma oranı" in out and "GEÇTİ" in out and "fit_taps" in out


def test_eval_cli_missing_prompts(tmp_path, capsys):
    tap = tmp_path / "tap.raw.jsonl"
    normal = tmp_path / "normal.raw.jsonl"
    _write_tap_recording(tap)
    prompts_path(tap).unlink()
    _write_normal_recording(normal, seconds=2)
    cfg = tmp_path / "vision.toml"
    cfg.write_text(
        "[hand]\ncalibration = \"" + (tmp_path / "none.json").as_posix() + "\"\n"
        "[lift]\nw_angle = 1.0\nw_length = 1.0\nbaseline_s = 1.0\ncurl_rate_suppress = 3.0\n",
        encoding="utf-8")
    with pytest.raises(SystemExit):
        lift_mod.main(["--eval", str(tap), str(normal), "--config", str(cfg)])
    assert "prompts" in capsys.readouterr().err
