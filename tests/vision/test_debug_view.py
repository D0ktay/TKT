import math
import sys
from dataclasses import replace as dataclasses_replace
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from tkt.core.types import FINGERS, SCHEMA_VERSION, Finger, HandFrame, SourceInfo
from tkt.vision import debug_view as dv
from tkt.vision import hand_calib
from tkt.vision.debug_view import (
    CalibState,
    Stats,
    TapSession,
    TapState,
    draw_overlay,
    draw_tap_prompt,
    render_lift_graph,
)
from tkt.vision.features import FINGER_JOINTS
from tkt.vision.hand_calib import STEPS
from tkt.vision.tracker import RawHand

STATS = Stats(capture_fps=59.9, process_fps=55.2, process_ms=12.3)


# ---------------------------------------------------------------- yardımcılar


def make_frame(present=True, lift=None, t=100.0, frame_id=1):
    if not present:
        return HandFrame(t, frame_id, False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {f: Finger((0.5, 0.5), None, 0.3, lift) for f in FINGERS}
    return HandFrame(t, frame_id, True, fingers, 0.9, (0.5, 0.5), 0.2, 0.97)


def _rot_x(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


MCP_X = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}


def make_hand(pip=0.0, dip=0.0, thumb_dist=0.07):
    """Yapay el (test_hand_calib ile aynı geometri)."""
    w = np.zeros((21, 3))
    for name, (mcp, p_i, d_i, t_i) in FINGER_JOINTS.items():
        w[mcp] = (MCP_X[name], 0.09, 0.0)
        d0 = np.array([0.0, 1.0, 0.0])
        w[p_i] = w[mcp] + 0.04 * d0
        w[d_i] = w[p_i] + 0.025 * _rot_x(d0, -pip)
        w[t_i] = w[d_i] + 0.02 * _rot_x(d0, -(pip + dip))
    w[1], w[2], w[3] = (-0.03, 0.02, 0.0), (-0.05, 0.04, 0.0), (-0.06, 0.06, 0.0)
    w[4] = w[5] + np.array([-thumb_dist, 0.0, 0.0])
    img = np.stack([0.5 + w[:, 0] * 2.0, 0.6 - w[:, 1] * 2.0], axis=1)
    return RawHand(image_pts=img, world_pts=w, score=0.95)


def hand_for_step(step):
    return {
        "open": make_hand(pip=0.3, dip=0.2, thumb_dist=0.065),
        "fist": make_hand(pip=1.5, dip=0.9, thumb_dist=0.03),
        "thumb_closed": make_hand(pip=0.3, dip=0.2, thumb_dist=0.02),
    }[step]


class FakeSource:
    """frames_raw: (kare, el, görüntü) üretir; kamera/tracker yok."""

    def __init__(self, items, width=640, height=512):
        self.items = list(items)
        self.info = SourceInfo(schema=SCHEMA_VERSION, width=width, height=height, fps=60.0,
                               hand="right")
        self.closed = False
        self.last_process_ms = 8.0
        self.calibrations = []

    def capture_fps(self):
        return 60.0

    def process_fps(self):
        return 58.0

    def frames_raw(self):
        yield from self.items

    def set_calibration(self, calib):
        self.calibrations.append(calib)

    def close(self):
        self.closed = True


class Recorder:
    """cv2.imshow / waitKey / destroyAllWindows yerine geçer."""

    def __init__(self, keys, visible=None):
        self.keys = list(keys)
        self.shown = []
        self.destroyed = 0
        # visible: kaçıncı imshow çağrısından sonra pencere görünmez olsun (None: hep görünür)
        self.visible_frames = visible

    def imshow(self, name, img):
        self.shown.append((name, img))

    def waitKey(self, delay):
        return self.keys.pop(0) if self.keys else -1

    def destroy(self):
        self.destroyed += 1

    def visible(self, name, prop):
        assert prop == cv2.WND_PROP_VISIBLE
        if self.visible_frames is not None and len(self.shown) > self.visible_frames:
            return 0.0
        return 1.0


@pytest.fixture
def rec(monkeypatch):
    def install(keys, visible=None):
        r = Recorder(keys, visible)
        monkeypatch.setattr(cv2, "imshow", r.imshow)
        monkeypatch.setattr(cv2, "waitKey", r.waitKey)
        monkeypatch.setattr(cv2, "destroyAllWindows", r.destroy)
        monkeypatch.setattr(cv2, "getWindowProperty", r.visible)
        return r
    return install


# ---------------------------------------------------------------- saf çizim


@pytest.mark.parametrize("shape", [(512, 640, 3), (512, 640), (512, 640, 1), (720, 1280, 3)])
def test_overlay_keeps_size_and_returns_bgr(shape):
    img = np.full(shape, 40, np.uint8)
    out = draw_overlay(img, make_frame(), make_hand(), STATS)
    assert out.shape == (shape[0], shape[1], 3) and out.dtype == np.uint8
    assert out is not img and img.shape == shape and np.all(img == 40)  # girdi değişmez
    assert np.any(out != 40)  # bir şey çizildi


def test_overlay_draws_skeleton_on_gray_image():
    img = np.zeros((512, 640), np.uint8)
    hand = make_hand()
    with_hand = draw_overlay(img, make_frame(), hand, STATS)
    without = draw_overlay(img, make_frame(present=False), None, STATS)
    # el noktasının bulunduğu piksel çizimle renklendi
    x, y = (hand.image_pts[8] * (640, 512)).astype(int)
    assert with_hand[y, x].any()
    assert not np.array_equal(with_hand, without)


def test_overlay_no_hand_no_error():
    img = np.zeros((512, 640, 3), np.uint8)
    out = draw_overlay(img, make_frame(present=False), None, STATS)
    assert out.shape == img.shape
    # present True ama ham el yok
    draw_overlay(img, make_frame(), None, STATS)
    # present False ama ham el var: iskelet çizilmez
    a = draw_overlay(img, make_frame(present=False), make_hand(), STATS)
    assert np.array_equal(a, out)


def test_overlay_lift_none_and_values():
    img = np.zeros((512, 640, 3), np.uint8)
    none = draw_overlay(img, make_frame(lift=None), make_hand(), STATS)
    some = draw_overlay(img, make_frame(lift=0.7), make_hand(), STATS)
    assert none.shape == some.shape
    assert not np.array_equal(none, some)  # lift çubuğu doluyor


def test_overlay_odd_values_no_error():
    img = np.zeros((48, 64, 3), np.uint8)   # çok küçük görüntü
    frame = make_frame()
    fingers = {f: Finger((0.5, 0.5), None, float("nan"), 5.0) for f in FINGERS}
    frame = HandFrame(1.0, 1, True, fingers, -1.0, (0, 0), 0.2, 2.0)
    draw_overlay(img, frame, make_hand(), Stats(0.0, 0.0, 0.0))
    nan_hand = RawHand(np.full((21, 2), np.nan), np.zeros((21, 3)), 0.9)
    draw_overlay(img, frame, nan_hand, STATS)


def test_overlay_calibration_state():
    img = np.zeros((512, 640, 3), np.uint8)
    base = draw_overlay(img, make_frame(), make_hand(), STATS)
    for step in STEPS:
        cs = CalibState(step, 0.5, 0.3, samples=4)
        out = draw_overlay(img, make_frame(), make_hand(), STATS, cs)
        assert out.shape == img.shape and not np.array_equal(out, base)
    draw_overlay(img, make_frame(present=False), None, STATS,
                 CalibState("open", 0.1, 0.0, samples=0))
    draw_overlay(img, make_frame(), None, STATS,
                 CalibState(None, 1.0, 1.0, message="Türkçe ğüşıöç mesaj"))


def test_connections_valid():
    assert len(dv.HAND_CONNECTIONS) == 21
    assert all(0 <= a < 21 and 0 <= b < 21 for a, b in dv.HAND_CONNECTIONS)


# ---------------------------------------------------------------- run döngüsü


def test_run_quits_on_q_and_cleans_up(rec):
    r = rec([-1, -1, ord("q")])
    img = np.zeros((512, 640, 3), np.uint8)
    src = FakeSource((make_frame(t=100.0 + i / 60, frame_id=i), make_hand(), img)
                     for i in range(10))
    dv.run(src, "unused")
    assert len(r.shown) == 3          # 'q' üçüncü karede
    assert r.shown[0][0] == dv.WINDOW_NAME
    assert r.shown[0][1].shape == (512, 640, 3)
    assert src.closed and r.destroyed == 1


def test_run_quits_on_escape_with_gray_and_no_hand(rec):
    r = rec([27])
    img = np.zeros((512, 640), np.uint8)
    src = FakeSource([(make_frame(present=False), None, img)] * 5)
    dv.run(src, "unused")
    assert len(r.shown) == 1 and src.closed and r.destroyed == 1


def test_run_ends_when_frames_end_and_closes(rec):
    r = rec([])
    src = FakeSource([(make_frame(present=False), None, np.zeros((64, 64), np.uint8))] * 3)
    dv.run(src, "unused")
    assert len(r.shown) == 3 and src.closed and r.destroyed == 1


def test_run_closes_on_exception(rec):
    r = rec([])

    class Boom(FakeSource):
        def frames_raw(self):
            yield (make_frame(), make_hand(), np.zeros((64, 64, 3), np.uint8))
            raise RuntimeError("kamera koptu")

    src = Boom([])
    with pytest.raises(RuntimeError):
        dv.run(src, "unused")
    assert src.closed and r.destroyed == 1


def _calib_items(dt=0.25, extra=8, present=True):
    """Adım adım kalibrasyon akışı: (kareler, tuşlar).

    Kare 0'da 'c'; her adımda bir bekleme karesinde BOŞLUK, sonra geri sayım + kayıt
    süresi kadar kare; en sonda `extra` kare. Tuş listesi kare sırasıyla hizalıdır.
    """
    img = np.zeros((512, 640, 3), np.uint8)
    items, keys = [], []

    def add(step, key=-1):
        i = len(items)
        frame = make_frame(t=50.0 + i * dt, frame_id=i, present=present)
        items.append((frame, hand_for_step(step) if present else None, img))
        keys.append(key)

    add(STEPS[0], ord("c"))
    per_step = int((dv.GUIDED_COUNTDOWN_SECONDS + dv.GUIDED_RECORD_SECONDS) / dt) + 1
    for step in STEPS:
        add(step, ord(" "))
        for _ in range(per_step):
            add(step)
    for _ in range(extra):
        add(STEPS[-1])
    return items, keys


def test_run_calibration_saves_file(rec, tmp_path):
    items, keys = _calib_items()
    # 'c', her adımda BOŞLUK; akış bitince döngü kendiliğinden çıkar
    r = rec(keys)
    path = tmp_path / "sub" / "calibration.json"
    src = FakeSource(items)
    dv.run(src, path)
    assert path.is_file()
    calib = hand_calib.load(path)
    assert calib is not None
    for f in FINGERS:
        assert calib.fist_ref[f] > calib.open_ref[f]
    assert calib.thumb_open > calib.thumb_closed
    assert src.closed and r.destroyed == 1
    assert len(r.shown) == len(items)
    # yeni değerler hemen uygulandı ve dosyadakiyle aynı
    assert src.calibrations == [calib]


def test_run_calibration_message_text_and_progress(rec, tmp_path, monkeypatch):
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state)
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    items, keys = _calib_items(extra=24)
    rec(keys)
    dv.run(FakeSource(items), tmp_path / "c.json")   # 6 sn > mesaj süresi
    steps = [cs.step for cs in seen if cs is not None and cs.step is not None]
    assert [s for i, s in enumerate(steps) if i == 0 or steps[i - 1] != s] == list(STEPS)
    assert any("BOSLUK" in cs.message for cs in seen if cs is not None and cs.step)
    msgs = [cs.message for cs in seen if cs is not None and cs.step is None and cs.message]
    assert msgs and all("uygulandi" in m for m in msgs)
    # mesaj süresi dolunca kaybolur
    assert seen[-1] is None


def test_run_calibration_without_hand_reports_failure(rec, tmp_path, monkeypatch):
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state)
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    items, keys = _calib_items(present=False)
    rec(keys)
    path = tmp_path / "c.json"
    src = FakeSource(items)
    dv.run(src, path)   # çökmez
    assert not path.exists() and src.calibrations == []
    msgs = [cs.message for cs in seen if cs is not None and cs.message]
    assert any("basarisiz" in m for m in msgs)


def test_run_c_restarts_session(rec, tmp_path):
    # kalibrasyon sırasında tekrar 'c': oturum baştan başlar, çökmez
    # (ikinci 'c' yalnızca 0,25 sn sonra, böylece adım başına örnekler hâlâ doğru adıma düşer)
    items, keys = _calib_items(extra=24)
    r = rec([ord("c")] + keys)
    dv.run(FakeSource([items[0]] + items), tmp_path / "c.json")
    assert (tmp_path / "c.json").is_file()
    assert r.destroyed == 1


def test_run_calibration_keeps_tap_ref(rec, tmp_path):
    old = hand_calib.HandCalibration(
        open_ref={f: 0.1 for f in FINGERS}, fist_ref={f: 2.0 for f in FINGERS},
        thumb_closed=0.3, thumb_open=0.9,
        tap_ref={f: 0.5 for f in FINGERS}, angle_sign=-1.0)
    path = tmp_path / "c.json"
    hand_calib.save(old, path)
    items, keys = _calib_items()
    rec(keys)
    src = FakeSource(items)
    dv.run(src, path)
    (new,) = src.calibrations
    assert new.tap_ref == old.tap_ref and new.angle_sign == -1.0
    assert new.open_ref != old.open_ref


def test_run_save_error_still_applies(rec, tmp_path):
    items, keys = _calib_items()
    rec(keys)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    src = FakeSource(items)
    dv.run(src, blocker / "calibration.json")   # üst dizin dosya: yazılamaz
    assert len(src.calibrations) == 1


# ---------------------------------------------------------------- pencere X


def test_run_ends_when_window_closed(rec):
    r = rec([], visible=4)    # 4 kare görünür, sonra X ile kapatılmış gibi
    img = np.zeros((64, 64, 3), np.uint8)
    src = FakeSource([(make_frame(t=1.0 + i / 60, frame_id=i), make_hand(), img)
                      for i in range(50)])
    dv.run(src, "unused")
    assert len(r.shown) == 5 and src.closed and r.destroyed == 1


def test_run_ignores_invisible_before_first_show(rec, monkeypatch):
    # bazı backend'ler pencere açılmadan 0/-1 verir: hiç görünmediyse çıkılmaz
    r = rec([])
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: -1.0)
    src = FakeSource([(make_frame(present=False), None, np.zeros((64, 64), np.uint8))] * 4)
    dv.run(src, "unused")
    assert len(r.shown) == 4


def test_run_survives_window_property_error(rec, monkeypatch):
    r = rec([])

    def boom(*a):
        raise cv2.error("no window")

    monkeypatch.setattr(cv2, "getWindowProperty", boom)
    src = FakeSource([(make_frame(present=False), None, np.zeros((64, 64), np.uint8))] * 4)
    dv.run(src, "unused")
    assert len(r.shown) == 4


# ---------------------------------------------------------------- tap oturumu


def test_tap_session_windows_and_samples():
    s = TapSession(n=3, window_s=2.0, lead_s=1.0, flash_s=0.25)
    hand = make_hand()
    st = s.state()
    assert st.lead and not st.flash
    s.add(hand, 10.0)                 # t0
    s.add(hand, 10.5)                 # hazırlık: örnek yok
    assert s.samples == [] and s.state().lead
    s.add(hand, 11.0)                 # pencere 1 başlar
    st = s.state()
    assert not st.lead and st.index == 0 and st.flash
    s.add(None, 11.5)                 # el yok: zaman ilerler, örnek eklenmez
    st = s.state()
    assert st.index == 0 and not st.flash and st.samples == 1
    s.add(hand, 13.5)
    assert s.state().index == 1 and s.state().flash is False
    s.add(hand, 15.1)
    assert s.state().index == 2 and not s.done
    s.add(hand, 17.0)                 # 1 + 3*2 = 7 sn doldu
    assert s.done and len(s.samples) == 3
    s.add(hand, 18.0)                 # bittikten sonra yok sayılır
    assert len(s.samples) == 3
    assert [t for t, _ in s.samples] == [11.0, 13.5, 15.1]
    assert s.state().progress == 1.0


def test_tap_session_window_bounds():
    s = TapSession(n=3, window_s=2.0, lead_s=1.0)
    assert s.windows == []
    s.add(make_hand(), 10.0)
    assert s.windows == [(11.0, 13.0), (13.0, 15.0), (15.0, 17.0)]


def test_tap_session_backwards_time_ignored():
    s = TapSession(n=2, window_s=1.0, lead_s=0.0)
    s.add(make_hand(), 5.0)
    s.add(make_hand(), 5.8)
    s.add(make_hand(), 5.2)           # geri giden zaman ilerlemeyi bozmaz
    assert s.state().index == 0 and not s.done


def test_tap_session_validates():
    for kw in ({"n": 0}, {"window_s": 0.0}, {"lead_s": -1.0}):
        with pytest.raises(ValueError):
            TapSession(**kw)


def _tap_items(n_frames, dt=0.25, t0=20.0, present=True):
    img = np.zeros((256, 320, 3), np.uint8)
    n = int(n_frames)
    return [(make_frame(present=present, t=t0 + i * dt, frame_id=i),
             make_hand() if present else None, img) for i in range(n)]


def _tap_total_frames(dt=0.25, extra=8):
    return int((dv.TAP_LEAD_SECONDS + dv.TAP_COUNT * dv.TAP_WINDOW_SECONDS) / dt) + extra


def test_run_tap_calls_fit_taps_and_applies(rec, tmp_path, monkeypatch):
    r = rec([ord("t")])
    path = tmp_path / "cal.json"
    start = hand_calib.DEFAULT
    fitted = hand_calib.HandCalibration(
        open_ref=start.open_ref, fist_ref=start.fist_ref,
        thumb_closed=start.thumb_closed, thumb_open=start.thumb_open,
        tap_ref={f: 0.4 for f in FINGERS}, angle_sign=-1.0)
    calls = []

    def fake_fit(calib, samples, **kw):
        calls.append((calib, list(samples), kw))
        return fitted

    monkeypatch.setattr(hand_calib, "fit_taps", fake_fit)
    src = FakeSource(_tap_items(_tap_total_frames()), width=512, height=640)
    dv.run(src, path)
    (calib_arg, samples, kw), = calls
    assert calib_arg == hand_calib.DEFAULT       # dosya yoksa DEFAULT
    assert samples and all(isinstance(t, float) and isinstance(h, RawHand)
                           for t, h in samples)
    # hazırlık süresindeki kareler (t < t0 + lead) toplanmadı; 't' ilk karede basıldı,
    # oturum 2. kareden başlar
    assert min(t for t, _ in samples) >= 20.25 + dv.TAP_LEAD_SECONDS
    # kesintisiz örnek akışı: pencere sınırları ayrıca verilir (boşluktan ayrılamaz)
    start = 20.25 + dv.TAP_LEAD_SECONDS
    w = dv.TAP_WINDOW_SECONDS
    assert kw["windows"] == [pytest.approx((start + k * w, start + (k + 1) * w))
                             for k in range(dv.TAP_COUNT)]
    assert kw["aspect"] == pytest.approx(512 / 640)
    assert src.calibrations == [fitted]
    assert hand_calib.load(path) == fitted
    assert r.destroyed == 1 and src.closed


def test_run_tap_fit_not_implemented_does_not_crash(rec, tmp_path, monkeypatch):
    def nope(calib, samples, **kw):
        raise NotImplementedError

    monkeypatch.setattr(hand_calib, "fit_taps", nope)
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state)
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    rec([ord("t")])
    path = tmp_path / "cal.json"
    src = FakeSource(_tap_items(_tap_total_frames()))
    dv.run(src, path)
    assert src.calibrations == [] and not path.exists()
    msgs = [cs.message for cs in seen if cs is not None and cs.message]
    assert msgs and all("fit_taps henuz yok" in m for m in msgs)


def test_run_tap_fit_value_error_and_no_hand(rec, tmp_path, monkeypatch):
    def bad(calib, samples, **kw):
        raise ValueError("tepe bulunamadi")

    monkeypatch.setattr(hand_calib, "fit_taps", bad)
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state)
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    rec([ord("t")])
    src = FakeSource(_tap_items(_tap_total_frames()))
    dv.run(src, tmp_path / "cal.json")
    assert src.calibrations == []
    assert any("basarisiz" in cs.message for cs in seen if cs is not None)

    # el hiç görünmezse fit_taps çağrılmaz
    monkeypatch.setattr(hand_calib, "fit_taps",
                        lambda *a: pytest.fail("fit_taps çağrılmamalı"))
    seen.clear()
    rec([ord("t")])
    src = FakeSource(_tap_items(_tap_total_frames(), present=False))
    dv.run(src, tmp_path / "cal2.json")
    assert any("el gorunmedi" in cs.message for cs in seen if cs is not None)


def test_run_c_then_t_switches_session(rec, tmp_path, monkeypatch):
    monkeypatch.setattr(hand_calib, "fit_taps", lambda *a: pytest.fail("çağrılmamalı"))
    items, keys = _calib_items(extra=12)
    rec([ord("c"), ord("t")] + keys)    # son 'c' t'yi iptal eder, c akışı tamamlanır
    src = FakeSource([items[0], items[0]] + items)
    dv.run(src, tmp_path / "c.json")
    assert len(src.calibrations) == 1


# ---------------------------------------------------------------- tap çizimi


def test_draw_tap_prompt_all_states():
    base = np.zeros((256, 320, 3), np.uint8)
    outs = []
    for st in (TapState(10, 0, True, False, 0.0, 0),
               TapState(10, 3, False, False, 0.3, 12),
               TapState(10, 3, False, True, 0.3, 12),
               TapState(10, 3, False, False, 0.3, 0)):
        img = base.copy()
        draw_tap_prompt(img, st)
        assert img.shape == base.shape and np.any(img != 0)
        outs.append(img)
    assert not np.array_equal(outs[1], outs[2])     # flash fark yaratır
    draw_tap_prompt(np.zeros((40, 50, 3), np.uint8), TapState(10, 9, False, True, 1.0, 1))


# ---------------------------------------------------------------- lift grafiği


def test_render_lift_graph_size_and_content():
    hist = [(10.0 + i * 0.1, i / 30, 1.0 - i / 30) for i in range(31)]
    img = render_lift_graph(hist, 120, 60)
    assert img.shape == (60, 120, 3) and img.dtype == np.uint8
    empty = render_lift_graph([], 120, 60)
    assert empty.shape == (60, 120, 3) and not np.array_equal(img, empty)
    # her iki parmağın rengi çizildi
    assert np.any(np.all(img == dv._GRAPH_COLORS["index"], axis=2))
    assert np.any(np.all(img == dv._GRAPH_COLORS["middle"], axis=2))


def test_render_lift_graph_none_values_and_extremes():
    hist = [(1.0, None, None), (1.5, 0.2, None), (2.0, None, 0.9), (2.5, 0.4, 0.5),
            (3.0, 5.0, -2.0), (3.5, float("nan"), 0.5)]
    img = render_lift_graph(hist, 90, 40)
    assert img.shape == (40, 90, 3)
    only_none = render_lift_graph([(1.0, None, None), (2.0, None, None)], 90, 40)
    assert only_none.shape == (40, 90, 3)
    assert render_lift_graph(hist, 0, 0).shape == (1, 1, 3)   # dejenere boyut


def test_render_lift_graph_window_drops_old_samples():
    old_only = [(0.0, 0.5, 0.5), (0.5, 0.6, 0.6)]
    both = old_only + [(10.0, None, None)]
    a = render_lift_graph(old_only, 100, 50, window_s=3.0)
    b = render_lift_graph(both, 100, 50, window_s=3.0)
    assert not np.array_equal(a, b)
    assert np.array_equal(b, render_lift_graph([], 100, 50))   # eskiler pencere dışı


def _graph_rect(h, w):
    """draw_lift_graph'ın yerleşimi (aynı formül)."""
    u = max(12, round(h / 36))
    pad = u // 2
    gw, gh = min(w // 3, 8 * u * 2), min(h // 6, 5 * u)
    return w - gw - pad, pad + 4 * u + pad, gw, gh


def test_run_draws_lift_graph_into_overlay(rec, monkeypatch):
    # lift dolu karelerle grafik köşede çizilir; lift None iken curl kullanılır (çökmez)
    img = np.zeros((512, 640, 3), np.uint8)
    r = rec([])
    calls = []
    real = dv.draw_lift_graph

    def spy(out, history):
        calls.append(list(history))
        real(out, history)

    monkeypatch.setattr(dv, "draw_lift_graph", spy)
    items = [(make_frame(t=1.0 + i * 0.1, frame_id=i, lift=(0.8 if i % 2 else None)),
              make_hand(), img) for i in range(40)]
    dv.run(FakeSource(items), "unused")
    assert len(calls) == 40
    # son çağrıda geçmiş: son LIFT_GRAPH_SECONDS sn; lift varsa lift, yoksa curl (0.3)
    hist = calls[-1]
    assert hist[-1][0] == pytest.approx(1.0 + 39 * 0.1)
    assert sorted({round(v, 6) for _t, v, _m in hist}) == [0.3, 0.8]
    last = r.shown[-1][1]
    assert last.shape == (512, 640, 3)
    x0, y0, gw, gh = _graph_rect(512, 640)
    region = last[y0:y0 + gh, x0:x0 + gw]
    # grafiğe özgü pikseller: zemin (30), çerçeve ve parmak çizgisi (işaret ve orta aynı
    # değerde; orta en son çizildiği için onun rengi görünür)
    assert np.array_equal(region[gh // 4, 2], [30, 30, 30])
    assert np.array_equal(region[0, gw // 2], dv._GRAY)
    assert np.any(np.all(region == dv._GRAPH_COLORS["middle"], axis=2))


def test_overlay_without_graph_has_no_graph_pixels():
    # yukarıdaki denetimin anlamlı olduğunu gösterir: grafik yoksa bu pikseller yok
    img = np.zeros((512, 640, 3), np.uint8)
    out = draw_overlay(img, make_frame(lift=0.8), make_hand(), STATS)
    x0, y0, gw, gh = _graph_rect(512, 640)
    region = out[y0:y0 + gh, x0:x0 + gw]
    assert not np.array_equal(region[gh // 4, 2], [30, 30, 30])
    assert not np.any(np.all(region == dv._GRAPH_COLORS["middle"], axis=2))


# ---------------------------------------------------------------- macOS kapatma düğmesi


def _fake_appkit(monkeypatch, title=dv.WINDOW_NAME, style=1):
    native = Mock()
    native.title.return_value = title
    native.styleMask.return_value = style
    native.isReleasedWhenClosed.return_value = True
    app = Mock()
    app.windows.return_value = [native]
    appkit = SimpleNamespace(
        NSApplication=SimpleNamespace(sharedApplication=lambda: app),
        NSWindowStyleMaskClosable=2,
    )
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    return native


def test_run_enables_cocoa_close_button_and_restores(rec, monkeypatch):
    native = _fake_appkit(monkeypatch)
    monkeypatch.setattr(dv, "platform", "darwin")
    monkeypatch.setattr(cv2, "currentUIFramework", lambda: "COCOA", raising=False)
    r = rec([-1, -1, ord("q")])
    img = np.zeros((256, 320, 3), np.uint8)
    items = [(make_frame(t=1.0 + i * 0.1, frame_id=i), make_hand(), img) for i in range(5)]
    dv.run(FakeSource(items), "unused")
    native.setStyleMask_.assert_called_once_with(3)
    assert [c.args for c in native.setReleasedWhenClosed_.call_args_list] == [(False,), (True,)]
    assert r.destroyed == 1


def test_cocoa_close_skipped_off_mac_or_already_closable(monkeypatch):
    native = _fake_appkit(monkeypatch, style=3)
    monkeypatch.setattr(cv2, "currentUIFramework", lambda: "COCOA", raising=False)
    monkeypatch.setattr(dv, "platform", "win32")
    assert dv._enable_cocoa_close() is None
    monkeypatch.setattr(dv, "platform", "darwin")
    assert dv._enable_cocoa_close() is None          # zaten kapatılabilir
    native.setStyleMask_.assert_not_called()
    monkeypatch.setattr(cv2, "currentUIFramework", lambda: "QT", raising=False)
    assert dv._enable_cocoa_close() is None


# ---------------------------------------------------------------- CLI


def test_main_opens_source_and_runs(monkeypatch, tmp_path):
    cfg = tmp_path / "vision.toml"
    calib = tmp_path / "cal.json"
    cfg.write_text(f'[hand]\ncalibration = "{calib.as_posix()}"\nside = "right"\n')
    calls = {}
    fake = object()

    def fake_open(path):
        calls["open"] = str(path)
        return fake

    def fake_run(source, calib_path):
        calls["run"] = (source, calib_path)

    monkeypatch.setattr(dv, "open_live_source", fake_open)
    monkeypatch.setattr(dv, "run", fake_run)
    dv.main(["--config", str(cfg)])
    assert calls["open"] == str(cfg)
    assert calls["run"] == (fake, calib)


# ---------------------------------------------------------------- adım adım kalibrasyon


def test_guided_waits_for_space_then_counts_down_then_records():
    g = dv.GuidedCalibration(countdown_s=3.0, record_s=3.0)
    h = hand_for_step("open")
    for i in range(20):                      # boşluk yok: bekler, örnek toplamaz
        g.add(h, 10.0 + i * 0.5)
    assert g.phase == "wait" and g.samples["open"] == []
    g.start()
    assert g.phase == "countdown"
    g.add(h, 20.0)                           # geri sayım başlar
    g.add(h, 22.0)
    assert g.phase == "countdown" and g.samples["open"] == []
    assert math.ceil(g.remaining) == 1
    g.add(h, 23.5)
    assert g.phase == "record" and len(g.samples["open"]) == 1
    g.start()                                # kayıt sürerken etkisiz
    g.add(h, 26.0)                           # 6 sn doldu: sonraki adım
    assert g.current_step == "fist" and g.phase == "wait"
    assert len(g.samples["open"]) == 1


def test_guided_result_after_all_steps():
    g = dv.GuidedCalibration(countdown_s=1.0, record_s=1.0)
    t = 0.0
    for step in STEPS:
        g.start()
        for _ in range(9):
            g.add(hand_for_step(step), t)
            t += 0.25
    assert g.done and g.phase == "done"
    calib = g.result()
    for f in FINGERS:
        assert calib.fist_ref[f] > calib.open_ref[f]
    assert calib.thumb_open > calib.thumb_closed


def test_guided_result_before_done_raises():
    with pytest.raises(ValueError):
        dv.GuidedCalibration().result()
    with pytest.raises(ValueError):
        dv.GuidedCalibration(record_s=0)


# ------------------------------------------------------------------ masa kalibrasyonu


def test_table_session_countdown_then_records():
    from tkt.vision.debug_view import TableSession

    ts = TableSession(countdown_s=1.0, record_s=2.0)
    hands = [SimpleNamespace(image_pts=np.full((21, 2), 0.5))] * 40
    for i, h in enumerate(hands):          # 10 Hz, 4 sn
        ts.add(h if i != 15 else None, 10.0 + i / 10)
    # geri sayımda (t < 1 sn) örnek yok; 1–3 sn arası 20 kare, biri el yok
    assert ts.done
    assert len(ts.samples) == 19
    assert "MASA" in ts.message()


def test_table_session_validates():
    from tkt.vision.debug_view import TableSession

    with pytest.raises(ValueError):
        TableSession(record_s=0)


def test_table_session_collects_palm_points():
    from tkt.vision.debug_view import TableSession

    hand = SimpleNamespace(image_pts=np.tile([0.25, 0.75], (21, 1)))
    ts = TableSession(countdown_s=0.0, record_s=1.0)
    ts.add(hand, 0.0)
    ts.add(hand, 0.5)
    assert ts.recording
    assert ts.points == [pytest.approx((0.25, 0.75))] * 2


def test_draw_table_area_marks_visited_cells():
    out = np.zeros((80, 80, 3), np.uint8)
    dv.draw_table_area(out, [])
    assert not out.any()
    dv.draw_table_area(out, [(0.1, 0.1), (0.15, 0.12), (0.12, 0.2)], grid=4)
    assert out[5:15, 5:15].any()          # ziyaret edilen sol üst hücre boyandı
    assert not out[60:80, 60:80].any()    # sağ alt hücreye dokunulmadı


# ---------------------------------------------------------------- aktif alan (yüzey)

SURF_CORNERS = [(0.35, 0.20), (0.65, 0.20), (0.85, 0.80), (0.15, 0.80)]


def tip_frame(tip, t, frame_id=1):
    if tip is None:
        return make_frame(present=False, t=t, frame_id=frame_id)
    fingers = {f: Finger(tip if f == "index" else (0.5, 0.5), None, 0.3, None)
               for f in FINGERS}
    return HandFrame(t, frame_id, True, fingers, 0.9, (0.5, 0.5), 0.2, 0.97)


class SurfaceFakeSource(FakeSource):
    def __init__(self, items, **kw):
        super().__init__(items, **kw)
        self.surface = None
        self.surfaces = []

    def set_surface(self, surface):
        self.surface = surface
        self.surfaces.append(surface)


class SurfaceFlow:
    """`s` akışı: kareler ve tuşlar (kare sırasıyla hizalı)."""

    def __init__(self, dt=0.25):
        self.dt = dt
        self.items, self.keys = [], []
        self.img = np.zeros((512, 640, 3), np.uint8)

    def add(self, tip, key=-1, n=1):
        for _ in range(n):
            i = len(self.items)
            frame = tip_frame(tip, 50.0 + i * self.dt, i)
            self.items.append((frame, make_hand() if tip is not None else None, self.img))
            self.keys.append(key)
            key = -1
        return self

    def corner(self, tip, frames=None):
        per = frames if frames is not None else int(
            (dv.SURFACE_COUNTDOWN_SECONDS + dv.SURFACE_RECORD_SECONDS) / self.dt) + 1
        self.add(tip, ord(" "))
        return self.add(tip, n=per)


def test_surface_session_records_median_per_corner():
    s = dv.SurfaceSession(countdown_s=1.0, record_s=1.0)
    assert s.phase == "wait" and s.index == 0
    s.add((0.9, 0.9), 0.0)              # beklemede örnek yok
    s.start()
    assert s.phase == "countdown"
    s.add((0.9, 0.9), 1.0)              # geri sayım başlar
    s.add((0.9, 0.9), 1.5)              # hâlâ geri sayım: örnek yok
    assert s.samples == []
    for k, t in enumerate((2.0, 2.25, 2.5, 2.75)):
        s.add((0.3 + 0.01 * k, 0.2), t)
    assert s.phase == "record" and len(s.samples) == 4
    s.add((0.3, 0.2), 3.0)              # kayıt biter
    assert s.phase == "wait" and s.index == 1
    assert s.corners[0] == pytest.approx((0.315, 0.2))


def test_surface_session_no_tip_repeats_corner():
    s = dv.SurfaceSession(countdown_s=0.0, record_s=1.0)
    s.start()
    for t in (0.0, 0.5, 1.0):
        s.add(None, t)
    assert s.index == 0 and not s.corners and "gorunmedi" in s.message()


def test_surface_session_redo_during_record_and_back_in_wait():
    s = dv.SurfaceSession(countdown_s=0.0, record_s=1.0)
    s.start()
    s.add((0.1, 0.1), 0.0)
    s.add((0.1, 0.1), 0.5)
    s.redo()                            # kayıt sürerken: aynı köşe baştan
    assert s.phase == "wait" and s.index == 0 and s.samples == []
    s.start()
    s.add((0.2, 0.2), 1.0)
    s.add((0.2, 0.2), 2.0)
    assert s.corners == [(0.2, 0.2)]
    s.redo()                            # beklemede: bir önceki köşeye dön
    assert s.corners == [] and s.index == 0


def test_surface_session_validates_and_result_before_done():
    with pytest.raises(ValueError):
        dv.SurfaceSession(countdown_s=-1.0)
    with pytest.raises(ValueError):
        dv.SurfaceSession().result(1.5, 1.25)


def test_run_surface_flow_saves_and_applies(rec, tmp_path, monkeypatch):
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state.message if calib_state is not None else "")
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    flow = SurfaceFlow()
    flow.add(SURF_CORNERS[0], ord("s"))
    for c in SURF_CORNERS:
        flow.corner(c)
    flow.add((0.5, 0.5), n=2)             # alan içi
    flow.add((0.5, 0.05), n=2)            # alan dışı
    r = rec(flow.keys)
    src = SurfaceFakeSource(flow.items)
    path = tmp_path / "sub" / "surface.json"
    dv.run(src, tmp_path / "cal.json", surface_path=path, surface_aspect=1.5)

    from tkt.vision import calibration
    saved = calibration.load(path)
    assert saved is not None and saved.aspect == 1.5
    assert saved.image_aspect == pytest.approx(640 / 512)
    for got, want in zip(saved.corners_img, SURF_CORNERS, strict=True):
        assert got == pytest.approx(want)
    assert src.surfaces == [saved]
    assert any("1/4 SOL-UZAK kose" in m and "BOSLUK" in m for m in seen)
    assert any("4/4 SOL-YAKIN" in m for m in seen)
    assert any("Alan kaydedildi" in m and "30x20" in m for m in seen)
    assert all(m.isascii() for m in seen)
    assert src.closed and len(r.shown) == len(flow.items)
    # Kalibrasyon sonrası alan çizilir: içerideyken yeşil, dışarıdayken kırmızı dörtgen.
    poly = np.array([(round(x * 640), round(y * 512)) for x, y in SURF_CORNERS], np.int32)
    mask = np.zeros((512, 640), np.uint8)
    cv2.polylines(mask, [poly], True, 255, 1)
    edge = mask > 0
    inside_img, outside_img = r.shown[-3][1], r.shown[-1][1]
    green = np.array(dv._GREEN, np.uint8)
    red = np.array(dv._RED, np.uint8)
    assert np.all(inside_img[edge] == green, axis=1).mean() > 0.5
    assert np.all(outside_img[edge] == red, axis=1).mean() > 0.5


def test_run_surface_redo_with_g(rec, tmp_path):
    flow = SurfaceFlow()
    flow.add(SURF_CORNERS[0], ord("s"))
    flow.corner(SURF_CORNERS[0])
    # 2. köşe yanlış yerde başladı: kayıt sırasında g, sonra doğru yerde tekrar
    flow.add((0.9, 0.5), ord(" "))
    flow.add((0.9, 0.5), n=6)             # geri sayım (1 sn) + kayıdın bir kısmı
    flow.add((0.9, 0.5), ord("g"))
    flow.corner(SURF_CORNERS[1])
    flow.corner(SURF_CORNERS[2])
    flow.corner(SURF_CORNERS[3])
    flow.add((0.5, 0.5), n=2)
    rec(flow.keys)
    src = SurfaceFakeSource(flow.items)
    path = tmp_path / "surface.json"
    dv.run(src, tmp_path / "cal.json", surface_path=path)
    from tkt.vision import calibration
    saved = calibration.load(path)
    assert saved is not None
    assert saved.corners_img[1] == pytest.approx(SURF_CORNERS[1])


def test_run_surface_bad_quad_reports_failure(rec, tmp_path, monkeypatch):
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state.message if calib_state is not None else "")
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    flow = SurfaceFlow()
    flow.add(SURF_CORNERS[0], ord("s"))
    for c in SURF_CORNERS[::-1]:          # ters sıra
        flow.corner(c)
    flow.add((0.5, 0.5), n=2)
    rec(flow.keys)
    src = SurfaceFakeSource(flow.items)
    path = tmp_path / "surface.json"
    dv.run(src, tmp_path / "cal.json", surface_path=path)
    assert not path.exists() and src.surfaces == []
    assert any("Alan kalibrasyonu basarisiz" in m for m in seen)


def test_draw_surface_colors_and_partial():
    from tkt.vision.calibration import SurfaceCalibration
    s = SurfaceCalibration(SURF_CORNERS, image_aspect=1.25)
    img = np.zeros((512, 640, 3), np.uint8)
    dv.draw_surface(img, s, (0.5, 0.5))
    assert np.any(np.all(img == np.array(dv._GREEN, np.uint8), axis=2))
    img = np.zeros((512, 640, 3), np.uint8)
    dv.draw_surface(img, s, (0.5, 0.02))
    assert np.any(np.all(img == np.array(dv._RED, np.uint8), axis=2))
    img = np.zeros((512, 640, 3), np.uint8)
    dv.draw_surface(img, None, None, SURF_CORNERS[:2])
    assert np.any(np.all(img == np.array(dv._YELLOW, np.uint8), axis=2))


def test_main_passes_surface_config(monkeypatch, tmp_path):
    cfg = tmp_path / "vision.toml"
    calib = tmp_path / "cal.json"
    surf = tmp_path / "surf.json"
    cfg.write_text(f'[hand]\ncalibration = "{calib.as_posix()}"\nside = "right"\n'
                   f'[surface]\nenabled = false\nfile = "{surf.as_posix()}"\naspect = 1.4\n')
    calls = {}
    monkeypatch.setattr(dv, "open_live_source", lambda path: "src")
    monkeypatch.setattr(dv, "run", lambda *a, **kw: calls.update(a=a, kw=kw))
    dv.main(["--config", str(cfg)])
    assert calls["a"] == ("src", calib)
    assert calls["kw"] == {"surface_path": surf, "surface_aspect": 1.4}


# ---------------------------------------------------------------- pinch


def touch_hand(finger, gap=0.004):
    """Açık el (hand_for_step("open")) ama başparmak ucu `finger` ucunun yanında."""
    base = hand_for_step("open")
    w = np.array(base.world_pts, dtype=float)
    img = np.array(base.image_pts, dtype=float)
    tip = FINGER_JOINTS[finger][3]
    w[4] = w[tip] + (gap, 0.0, 0.0)
    img[4] = img[tip] + (gap * 2.0, 0.0)
    return RawHand(image_pts=img, world_pts=w, score=base.score)


def _step_hand(step):
    if step in hand_calib.PINCH_STEP_FINGER:
        return touch_hand(hand_calib.PINCH_STEP_FINGER[step])
    return hand_for_step(step)


def _pinch_items(dt=0.25, extra=8, first_key=dv.KEY_PINCH, steps=None, hand_fn=_step_hand):
    """'p' ile başlayan el + pinch kalibrasyonu akışı: (kareler, tuşlar)."""
    steps = steps or STEPS + hand_calib.PINCH_STEPS
    img = np.zeros((512, 640, 3), np.uint8)
    items, keys = [], []

    def add(step, key=-1):
        i = len(items)
        items.append((make_frame(t=50.0 + i * dt, frame_id=i), hand_fn(step), img))
        keys.append(key)

    add(steps[0], first_key)
    per_step = int((dv.GUIDED_COUNTDOWN_SECONDS + dv.GUIDED_RECORD_SECONDS) / dt) + 1
    for step in steps:
        add(step, ord(" "))
        for _ in range(per_step):
            add(step)
    for _ in range(extra):
        add(steps[-1])
    return items, keys


def _pinch_frame(pinch):
    f = make_frame()
    return HandFrame(f.t, f.frame_id, True, f.fingers, f.thumb_out, f.palm, f.hand_size,
                     f.confidence, pinch=pinch)


def test_overlay_draws_four_pinch_bars():
    img = np.zeros((512, 640, 3), np.uint8)
    none = draw_overlay(img, _pinch_frame(None), None, STATS)
    full = draw_overlay(img, _pinch_frame(dict.fromkeys(FINGERS, 1.0)), None, STATS)
    diff = np.any(none != full, axis=2)
    # dört çubuk: farklı pikseller dört ayrı satır bandında
    rows = np.flatnonzero(diff.any(axis=1))
    bands = 1 + int(np.sum(np.diff(rows) > 1))
    assert bands == len(FINGERS)
    # pinch çubuğu curl ve lift sütunlarının sağında
    cols = np.flatnonzero(diff.any(axis=0))
    assert cols.min() > 640 // 4
    # bozuk değerler çökmez; present=False ise pinch gösterilmez
    draw_overlay(img, _pinch_frame({"index": float("nan"), "middle": 3.0}), None, STATS)
    a = draw_overlay(img, make_frame(present=False), None, STATS)
    assert a.shape == img.shape


def test_guided_pinch_session_steps_and_result():
    g = dv.GuidedCalibration(countdown_s=0.0, record_s=1.0, pinch=True)
    assert g.steps == STEPS + hand_calib.PINCH_STEPS
    assert dv.GuidedCalibration().steps == STEPS   # varsayılan: eski üç adım
    t = 0.0
    for step in g.steps:
        g.start()
        for _ in range(6):
            g.add(_step_hand(step), t)
            t += 0.25
    assert g.done
    calib = g.result(aspect=640 / 512)
    assert calib.pinch_ref is not None and set(calib.pinch_ref) == set(FINGERS)
    assert calib.fist_ref["index"] > calib.open_ref["index"]


def test_run_pinch_calibration_one_session_saves_and_keeps_others(rec, tmp_path):
    old = hand_calib.HandCalibration(
        open_ref={f: 0.1 for f in FINGERS}, fist_ref={f: 2.0 for f in FINGERS},
        thumb_closed=0.3, thumb_open=0.9,
        tap_ref={f: 0.5 for f in FINGERS}, angle_sign=-1.0,
        table={"length": [0.0] * 6, "width": [0.0] * 6})
    path = tmp_path / "c.json"
    hand_calib.save(old, path)
    items, keys = _pinch_items()
    rec(keys)
    src = FakeSource(items)
    dv.run(src, path)
    (new,) = src.calibrations
    assert new.pinch_ref is not None
    assert new.tap_ref == old.tap_ref and new.angle_sign == -1.0 and new.table == old.table
    assert hand_calib.load(path) == new


def test_run_pinch_progress_shows_all_steps(rec, tmp_path, monkeypatch):
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state)
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    items, keys = _pinch_items()
    rec(keys)
    dv.run(FakeSource(items), tmp_path / "c.json")
    steps = [cs.step for cs in seen if cs is not None and cs.step is not None]
    assert [s for i, s in enumerate(steps) if i == 0 or steps[i - 1] != s] == \
        list(STEPS + hand_calib.PINCH_STEPS)
    assert all(cs.steps == STEPS + hand_calib.PINCH_STEPS
               for cs in seen if cs is not None and cs.step is not None)


def test_run_c_keeps_existing_pinch_ref(rec, tmp_path):
    ref = {f: {"img_open": 1.0, "img_touch": 0.1, "world_open": 1.0, "world_touch": 0.1}
           for f in FINGERS}
    old = dataclasses_replace(hand_calib.DEFAULT, pinch_ref=ref)
    path = tmp_path / "c.json"
    hand_calib.save(old, path)
    items, keys = _calib_items()
    rec(keys)
    src = FakeSource(items)
    dv.run(src, path)
    (new,) = src.calibrations
    assert new.pinch_ref == ref and new.open_ref != old.open_ref


def test_run_pinch_failure_writes_nothing(rec, tmp_path, monkeypatch):
    # Değme adımlarında başparmak hep açık: fit_pinch başarısız, eski dosya korunur.
    seen = []
    real = dv.draw_overlay

    def spy(image, frame, hand, stats, calib_state=None):
        seen.append(calib_state)
        return real(image, frame, hand, stats, calib_state)

    monkeypatch.setattr(dv, "draw_overlay", spy)
    path = tmp_path / "c.json"
    hand_calib.save(hand_calib.DEFAULT, path)
    before = path.read_text()
    items, keys = _pinch_items(
        hand_fn=lambda s: hand_for_step("open") if s in hand_calib.PINCH_STEPS
        else hand_for_step(s))
    rec(keys)
    src = FakeSource(items)
    dv.run(src, path)
    assert src.calibrations == [] and path.read_text() == before
    assert any("basarisiz" in cs.message for cs in seen if cs is not None and cs.message)
