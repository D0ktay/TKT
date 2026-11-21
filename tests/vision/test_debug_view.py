import math

import cv2
import numpy as np
import pytest

from tkt.core.types import FINGERS, Finger, HandFrame
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
from tkt.vision.hand_calib import STEP_SECONDS, STEPS
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

    def __init__(self, items):
        self.items = list(items)
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


def _calib_items(step_s=STEP_SECONDS, dt=0.25, extra=8):
    """Her adım için `step_s` saniyelik kare, sonra `extra` kare. Zaman kare `t`'sinden."""
    img = np.zeros((512, 640, 3), np.uint8)
    n = int(step_s / dt) * len(STEPS) + extra
    items = []
    for i in range(n):
        t = 50.0 + i * dt
        step = STEPS[min(len(STEPS) - 1, int(i * dt // step_s))]
        items.append((make_frame(t=t, frame_id=i), hand_for_step(step), img))
    return items


def test_run_calibration_saves_file(rec, tmp_path):
    items = _calib_items()
    # ilk karede 'c', sonra hiçbir tuş; akış bitince döngü kendiliğinden çıkar
    r = rec([ord("c")])
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
    rec([ord("c")])
    dv.run(FakeSource(_calib_items(extra=24)), tmp_path / "c.json")   # 6 sn > mesaj süresi
    steps = [cs.step for cs in seen if cs is not None and cs.step is not None]
    assert [s for i, s in enumerate(steps) if i == 0 or steps[i - 1] != s] == list(STEPS)
    msgs = [cs.message for cs in seen if cs is not None and cs.message]
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
    rec([ord("c")])
    img = np.zeros((64, 64, 3), np.uint8)
    items = [(make_frame(present=False, t=10.0 + i * 0.5, frame_id=i), None, img)
             for i in range(20)]
    path = tmp_path / "c.json"
    src = FakeSource(items)
    dv.run(src, path)   # çökmez
    assert not path.exists() and src.calibrations == []
    msgs = [cs.message for cs in seen if cs is not None and cs.message]
    assert any("basarisiz" in m for m in msgs)


def test_run_c_restarts_session(rec, tmp_path):
    # kalibrasyon sırasında tekrar 'c': oturum baştan başlar, çökmez
    # (ikinci 'c' yalnızca 0,25 sn sonra, böylece adım başına örnekler hâlâ doğru adıma düşer)
    r = rec([ord("c"), ord("c")])
    dv.run(FakeSource(_calib_items(extra=24)), tmp_path / "c.json")
    assert (tmp_path / "c.json").is_file()
    assert r.destroyed == 1


def test_run_calibration_keeps_tap_ref(rec, tmp_path):
    old = hand_calib.HandCalibration(
        open_ref={f: 0.1 for f in FINGERS}, fist_ref={f: 2.0 for f in FINGERS},
        thumb_closed=0.3, thumb_open=0.9,
        tap_ref={f: 0.5 for f in FINGERS}, angle_sign=-1.0)
    path = tmp_path / "c.json"
    hand_calib.save(old, path)
    rec([ord("c")])
    src = FakeSource(_calib_items())
    dv.run(src, path)
    (new,) = src.calibrations
    assert new.tap_ref == old.tap_ref and new.angle_sign == -1.0
    assert new.open_ref != old.open_ref


def test_run_save_error_still_applies(rec, tmp_path):
    rec([ord("c")])
    blocker = tmp_path / "file"
    blocker.write_text("x")
    src = FakeSource(_calib_items())
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

    def fake_fit(calib, samples):
        calls.append((calib, list(samples)))
        return fitted

    monkeypatch.setattr(hand_calib, "fit_taps", fake_fit)
    src = FakeSource(_tap_items(_tap_total_frames()))
    dv.run(src, path)
    (calib_arg, samples), = calls
    assert calib_arg == hand_calib.DEFAULT       # dosya yoksa DEFAULT
    assert samples and all(isinstance(t, float) and isinstance(h, RawHand)
                           for t, h in samples)
    # hazırlık süresindeki kareler (t < t0 + lead) toplanmadı; 't' ilk karede basıldı,
    # oturum 2. kareden başlar
    assert min(t for t, _ in samples) >= 20.25 + dv.TAP_LEAD_SECONDS
    assert src.calibrations == [fitted]
    assert hand_calib.load(path) == fitted
    assert r.destroyed == 1 and src.closed


def test_run_tap_fit_not_implemented_does_not_crash(rec, tmp_path, monkeypatch):
    def nope(calib, samples):
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
    def bad(calib, samples):
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
    rec([ord("c"), ord("t"), ord("c")])    # son 'c' t'yi iptal eder, c akışı tamamlanır
    src = FakeSource(_calib_items(extra=12))
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


def test_run_draws_lift_graph_into_overlay(rec):
    # lift dolu karelerle grafik köşede çizilir; lift None iken curl kullanılır (çökmez)
    img = np.zeros((512, 640, 3), np.uint8)
    r = rec([])
    items = [(make_frame(t=1.0 + i * 0.1, frame_id=i, lift=(0.8 if i % 2 else None)),
              make_hand(), img) for i in range(40)]
    dv.run(FakeSource(items), "unused")
    last = r.shown[-1][1]
    x = last.shape[1] - 5
    corner = last[: last.shape[0] // 2, last.shape[1] * 2 // 3:]
    assert corner.shape[0] and np.any(corner != 0) and last.shape == (512, 640, 3)
    assert x > 0


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
