import numpy as np
import pytest

from tkt.core.source import HandFrameSource
from tkt.core.types import FINGERS, Finger, HandFrame
from tkt.vision import live_source as ls
from tkt.vision.camera import CapturedFrame
from tkt.vision.hand_calib import DEFAULT
from tkt.vision.live_source import LiveSource, load_vision_config
from tkt.vision.tracker import RawHand


class FakeCamera:
    fps = 60

    def __init__(self, n=6, shape=(640, 512, 3), skip=1, start_ok=True):
        self.shape = shape
        self.frames = [
            CapturedFrame(np.full(shape, i, np.uint8), 100.0 + i / 60, i * skip + 1)
            for i in range(n)
        ]
        self.started = self.closed = False
        self.start_ok = start_ok

    def start(self):
        self.started = True

    def read(self, timeout=1.0):
        if not self.start_ok or not self.frames:
            return None
        return self.frames.pop(0)

    def measured_fps(self):
        return 59.9

    def close(self):
        self.closed = True


def make_hand(score=0.9):
    return RawHand(np.full((21, 2), 0.5), np.zeros((21, 3)), score)


class FakeTracker:
    def __init__(self, pattern):
        self.pattern = list(pattern)   # her kare için True/False
        self.calls = []
        self.closed = False

    def process(self, image, t):
        self.calls.append((int(image[0, 0, 0]), t))
        return make_hand() if self.pattern.pop(0) else None

    def close(self):
        self.closed = True


def fake_build_frame(hand, t, frame_id, calib, info, lift=None, prev_hand_size=None):
    if hand is None:
        return HandFrame(t, frame_id, False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {f: Finger((0.5, 0.5), None, 0.1, None if lift is None else lift[f])
               for f in FINGERS}
    # hand_size alanı prev_hand_size'ı taşır (testte doğrulamak için)
    size = -1.0 if prev_hand_size is None else prev_hand_size + 1
    return HandFrame(t, frame_id, True, fingers, 0.5, (0.5, 0.5), size, hand.score)


@pytest.fixture(autouse=True)
def _fake_features(monkeypatch):
    monkeypatch.setattr(ls.features, "build_frame", fake_build_frame)


def test_info_from_actual_frame_size():
    cam = FakeCamera(shape=(640, 512, 3))
    src = LiveSource(cam, FakeTracker([]), DEFAULT, "right")
    assert (src.info.width, src.info.height) == (512, 640)
    assert src.info.hand == "right" and src.info.fps == 60
    assert cam.started
    assert isinstance(src, HandFrameSource)


def test_first_frame_failure_closes_camera():
    cam = FakeCamera(start_ok=False)
    with pytest.raises(RuntimeError):
        LiveSource(cam, FakeTracker([]), DEFAULT, "right")
    assert cam.closed


def test_frame_id_increases_and_skips_are_preserved():
    cam = FakeCamera(n=5, skip=2)         # id: 1, 3, 5, 7, 9 (işleme atlıyor)
    src = LiveSource(cam, FakeTracker([True] * 5), DEFAULT, "right")
    frames = list(src.frames())
    assert [f.frame_id for f in frames] == [1, 3, 5, 7, 9]
    assert [f.t for f in frames] == sorted(f.t for f in frames)
    assert frames[0].t == 100.0   # yakalama zamanı, işleme zamanı değil


def test_absent_hand_yields_empty_frames_and_stream_continues():
    cam = FakeCamera(n=4)
    src = LiveSource(cam, FakeTracker([True, False, False, True]), DEFAULT, "right")
    frames = list(src.frames())
    assert [f.present for f in frames] == [True, False, False, True]
    assert [f.frame_id for f in frames] == [1, 2, 3, 4]     # el olmasa da artar
    assert frames[1].fingers == {}


def test_frames_raw_gives_hand_and_image():
    cam = FakeCamera(n=2)
    src = LiveSource(cam, FakeTracker([True, False]), DEFAULT, "right")
    (f0, h0, img0), (f1, h1, img1) = list(src.frames_raw())
    assert isinstance(h0, RawHand) and h1 is None
    assert img0.shape == (640, 512, 3) and int(img0[0, 0, 0]) == 0 and int(img1[0, 0, 0]) == 1
    assert f0.present and not f1.present


def test_tracker_gets_capture_time():
    cam = FakeCamera(n=3)
    trk = FakeTracker([False] * 3)
    list(LiveSource(cam, trk, DEFAULT, "right").frames())
    assert [t for _, t in trk.calls] == [100.0, 100.0 + 1 / 60, 100.0 + 2 / 60]


def test_prev_hand_size_passed_and_reset_when_absent():
    cam = FakeCamera(n=4)
    src = LiveSource(cam, FakeTracker([True, True, False, True]), DEFAULT, "right")
    sizes = [f.hand_size for f in src.frames()]
    # kare 1: önceki yok (-1); kare 2: önceki=-1 → 0; el kaybolunca önceki sıfırlanır
    assert sizes == [-1.0, 0.0, 0.0, -1.0]


def test_processing_time_and_fps_exposed(monkeypatch):
    # Windows'ta monotonic saat ~15 ms çözünürlüklü; hızlı sahte karelerde süre 0 görünür.
    import itertools

    import tkt.vision.live_source as ls

    clock = itertools.count(1000.0, 0.01)
    monkeypatch.setattr(ls.time, "monotonic", lambda: next(clock))
    cam = FakeCamera(n=5)
    src = LiveSource(cam, FakeTracker([True] * 5), DEFAULT, "right")
    assert src.last_process_ms == 0.0 and src.process_fps() == 0.0
    list(src.frames())
    assert src.last_process_ms > 0.0
    assert src.process_fps() > 0.0
    assert src.capture_fps() == 59.9


class FakeLift:
    def __init__(self):
        self.updates, self.resets = [], 0

    def update(self, hand, t, curls):
        self.updates.append((t, dict(curls)))
        return {f: 0.25 for f in FINGERS}

    def reset(self):
        self.resets += 1


def test_lift_is_used_and_reset_on_absence():
    cam = FakeCamera(n=3)
    lift = FakeLift()
    src = LiveSource(cam, FakeTracker([True, False, True]), DEFAULT, "right", lift)
    frames = list(src.frames())
    assert frames[0].fingers["index"].lift == 0.25
    assert frames[1].fingers == {}
    assert len(lift.updates) == 2 and lift.resets == 1
    assert set(lift.updates[0][1]) == set(FINGERS)


def test_no_lift_leaves_none():
    cam = FakeCamera(n=1)
    (frame,) = LiveSource(cam, FakeTracker([True]), DEFAULT, "right").frames()
    assert frame.fingers["index"].lift is None


def test_close_stops_stream_and_releases_both():
    cam, trk = FakeCamera(n=10), FakeTracker([False] * 10)
    src = LiveSource(cam, trk, DEFAULT, "right")
    it = src.frames()
    next(it)
    src.close()
    assert list(it) == []
    assert cam.closed and trk.closed
    src.close()


def test_load_vision_config_reads_repo_config():
    cfg = load_vision_config("config/vision.toml")
    assert cfg["camera"]["width"] == 640 and cfg["tracker"]["min_detection"] == 0.5


def test_open_live_source_builds_from_config(tmp_path, monkeypatch):
    cfg = tmp_path / "v.toml"
    cfg.write_text(
        '[camera]\nindex = 2\nwidth = 640\nheight = 512\nfps = 60\nrotate_180 = false\n'
        'backend = "dshow"\n[tracker]\nmodel = "m.task"\nmin_detection = 0.4\n'
        'min_tracking = 0.3\n[hand]\nside = "left"\ncalibration = "~/yok.json"\n'
        '[lift]\nw_angle = 1.0\n', encoding="utf-8")
    made = {}

    class Cam(FakeCamera):
        def __init__(self, *a):
            super().__init__(n=1)
            made["cam"] = a

    class Trk(FakeTracker):
        def __init__(self, *a):
            super().__init__([])
            made["trk"] = a

    monkeypatch.setattr(ls, "Camera", Cam)
    monkeypatch.setattr(ls, "HandTracker", Trk)
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: None)
    src = ls.open_live_source(cfg)
    assert made["cam"] == (2, 640, 512, 60, False, "dshow")
    assert made["trk"] == ("m.task", 0.4, 0.3)
    assert src.info.hand == "left" and src._lift is None and src._calib is DEFAULT


def test_open_live_source_makes_lift_when_tap_ref(tmp_path, monkeypatch):
    from dataclasses import replace

    cfg = tmp_path / "v.toml"
    cfg.write_text(
        '[camera]\nindex = 0\nwidth = 640\nheight = 512\nfps = 60\n[tracker]\nmodel = "m"\n'
        'min_detection = 0.5\nmin_tracking = 0.5\n[hand]\nside = "right"\ncalibration = "c"\n'
        '[lift]\nw_angle = 1.0\n', encoding="utf-8")
    calib = replace(DEFAULT, tap_ref={f: 1.0 for f in FINGERS})
    monkeypatch.setattr(ls, "Camera", lambda *a: FakeCamera(n=1))
    monkeypatch.setattr(ls, "HandTracker", lambda *a: FakeTracker([]))
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: calib)
    monkeypatch.setattr(ls, "LiftEstimator", lambda c, cfg: ("lift", c, cfg))
    src = ls.open_live_source(cfg)
    assert src._lift == ("lift", calib, {"w_angle": 1.0})


@pytest.mark.camera
def test_real_live_source_opens():
    src = ls.open_live_source("config/vision.toml")
    try:
        assert src.info.width > 0
    finally:
        src.close()
