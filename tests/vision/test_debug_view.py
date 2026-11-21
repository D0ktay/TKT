import math

import cv2
import numpy as np
import pytest

from tkt.core.types import FINGERS, Finger, HandFrame
from tkt.vision import debug_view as dv
from tkt.vision import hand_calib
from tkt.vision.debug_view import CalibState, Stats, draw_overlay
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

    def capture_fps(self):
        return 60.0

    def process_fps(self):
        return 58.0

    def frames_raw(self):
        yield from self.items

    def close(self):
        self.closed = True


class Recorder:
    """cv2.imshow / waitKey / destroyAllWindows yerine geçer."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.shown = []
        self.destroyed = 0

    def imshow(self, name, img):
        self.shown.append((name, img))

    def waitKey(self, delay):
        return self.keys.pop(0) if self.keys else -1

    def destroy(self):
        self.destroyed += 1


@pytest.fixture
def rec(monkeypatch):
    def install(keys):
        r = Recorder(keys)
        monkeypatch.setattr(cv2, "imshow", r.imshow)
        monkeypatch.setattr(cv2, "waitKey", r.waitKey)
        monkeypatch.setattr(cv2, "destroyAllWindows", r.destroy)
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
    # ekranda "yeniden başlatınca" mesajı çizildi: kalibrasyon sonrası kareler öncekinden farklı
    assert len(r.shown) == len(items)


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
    assert msgs and all(dv.RESTART_NOTE in m for m in msgs)
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
    dv.run(FakeSource(items), path)   # çökmez
    assert not path.exists()
    msgs = [cs.message for cs in seen if cs is not None and cs.message]
    assert any("basarisiz" in m for m in msgs)


def test_run_c_restarts_session(rec, tmp_path):
    # kalibrasyon sırasında tekrar 'c': oturum baştan başlar, çökmez
    # (ikinci 'c' yalnızca 0,25 sn sonra, böylece adım başına örnekler hâlâ doğru adıma düşer)
    r = rec([ord("c"), ord("c")])
    dv.run(FakeSource(_calib_items(extra=24)), tmp_path / "c.json")
    assert (tmp_path / "c.json").is_file()
    assert r.destroyed == 1


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
