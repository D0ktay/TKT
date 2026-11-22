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


def fake_build_frame(hand, t, frame_id, calib, info, lift=None, prev_hand_size=None,
                     lift_off=None):
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


def calib_with_tap_ref():
    from dataclasses import replace
    return replace(DEFAULT, tap_ref={f: 1.0 for f in FINGERS})


def test_set_calibration_reaches_build_frame(monkeypatch):
    seen = []

    def spy(hand, t, frame_id, calib, info, lift=None, prev_hand_size=None, lift_off=None):
        seen.append(calib)
        return fake_build_frame(hand, t, frame_id, calib, info, lift, prev_hand_size)

    monkeypatch.setattr(ls.features, "build_frame", spy)
    new = calib_with_tap_ref()
    src = LiveSource(FakeCamera(n=4), FakeTracker([True] * 4), DEFAULT, "right")
    it = src.frames()
    next(it)
    next(it)
    src.set_calibration(new)
    list(it)
    assert seen[:2] == [DEFAULT, DEFAULT]
    assert seen[2:] == [new, new]


def test_set_calibration_rebuilds_lift_when_tap_ref(monkeypatch):
    built = []
    monkeypatch.setattr(
        ls, "LiftEstimator",
        lambda c, cfg, aspect=1.0: built.append((c, cfg, aspect)) or FakeLift())
    src = LiveSource(FakeCamera(n=2), FakeTracker([True] * 2), DEFAULT, "right", FakeLift())
    src.lift_cfg = {"w_angle": 1.0}
    new = calib_with_tap_ref()
    src.set_calibration(new)
    assert built == [(new, {"w_angle": 1.0}, 512 / 640)]   # kare 512x640
    frames = list(src.frames())
    assert frames[0].fingers["index"].lift == 0.25      # yeni lift kullanılıyor


def test_set_calibration_without_tap_ref_drops_lift():
    src = LiveSource(FakeCamera(n=2), FakeTracker([True] * 2), DEFAULT, "right", FakeLift())
    src.set_calibration(DEFAULT)
    assert src._lift is None
    assert list(src.frames())[0].fingers["index"].lift is None


def test_set_calibration_without_lift_stays_without_lift():
    src = LiveSource(FakeCamera(n=1), FakeTracker([True]), DEFAULT, "right")
    src.set_calibration(calib_with_tap_ref())
    assert src._lift is None


def test_set_calibration_creates_lift_for_config_built_source(monkeypatch):
    monkeypatch.setattr(ls, "LiftEstimator", lambda c, cfg, aspect=1.0: FakeLift())
    src = LiveSource(FakeCamera(n=1), FakeTracker([True]), DEFAULT, "right")
    src.lift_cfg = {}
    src.set_calibration(calib_with_tap_ref())
    assert isinstance(src._lift, FakeLift)


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
    monkeypatch.setattr(ls, "LiftEstimator", lambda c, cfg, aspect=1.0: ("lift", c, cfg))
    src = ls.open_live_source(cfg)
    assert src._lift == ("lift", calib, {"w_angle": 1.0})
    assert src.lift_cfg == {"w_angle": 1.0}


def test_set_calibration_keeps_lift_settings_of_given_estimator():
    from dataclasses import replace

    from tkt.vision.lift import LiftEstimator

    cfg = {"w_angle": 2.0, "w_length": 0.5, "baseline_s": 1.5, "curl_rate_suppress": 4.0}
    est = LiftEstimator(calib_with_tap_ref(), cfg)
    src = LiveSource(FakeCamera(n=1), FakeTracker([True]), calib_with_tap_ref(), "right", est)
    assert src.lift_cfg is None                      # config'ten değil, doğrudan kuruldu
    new = replace(DEFAULT, tap_ref={f: 2.0 for f in FINGERS})
    src.set_calibration(new)                         # önceden: KeyError('w_angle')
    lift = src._lift
    assert isinstance(lift, LiftEstimator) and lift is not est
    assert (lift.w_angle, lift.w_length, lift.baseline_s, lift.curl_rate_suppress) == \
        (2.0, 0.5, 1.5, 4.0)
    assert lift.tap_ref == {f: 2.0 for f in FINGERS}
    src.set_calibration(DEFAULT)                     # tap_ref yok → lift kapanır
    assert src._lift is None
    src.set_calibration(new)                         # ayarlar korunur, lift geri gelir
    assert isinstance(src._lift, LiftEstimator) and src._lift.w_angle == 2.0


def _write_cfg(tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text(
        '[camera]\nindex = 0\nwidth = 640\nheight = 512\nfps = 60\n[tracker]\nmodel = "m"\n'
        'min_detection = 0.5\nmin_tracking = 0.5\n[hand]\nside = "right"\ncalibration = "c"\n'
        '[lift]\nw_angle = 1.0\n', encoding="utf-8")
    return cfg


def test_open_live_source_bad_calibration_closes_tracker_and_camera(tmp_path, monkeypatch):
    made = {}

    def cam(*a):
        made["cam"] = FakeCamera(n=1)
        return made["cam"]

    def trk(*a):
        made["trk"] = FakeTracker([])
        return made["trk"]

    def bad_load(p):
        raise ValueError("kalibrasyon okunamadı")

    monkeypatch.setattr(ls, "Camera", cam)
    monkeypatch.setattr(ls, "HandTracker", trk)
    monkeypatch.setattr(ls.hand_calib, "load", bad_load)
    with pytest.raises(ValueError):
        ls.open_live_source(_write_cfg(tmp_path))
    assert made["trk"].closed
    assert made.get("cam") is None or made["cam"].closed


def test_open_live_source_bad_lift_config_closes_tracker_and_camera(tmp_path, monkeypatch):
    made = {}
    monkeypatch.setattr(ls, "Camera", lambda *a: made.setdefault("cam", FakeCamera(n=1)))
    monkeypatch.setattr(ls, "HandTracker", lambda *a: made.setdefault("trk", FakeTracker([])))
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: calib_with_tap_ref())
    with pytest.raises(KeyError):                    # [lift] eksik anahtar (w_length ...)
        ls.open_live_source(_write_cfg(tmp_path))
    assert made["trk"].closed
    assert made.get("cam") is None or made["cam"].closed


def test_given_lift_is_rebuilt_with_real_frame_aspect():
    from tkt.vision.lift import LiftEstimator

    cfg = {"w_angle": 2.0, "w_length": 0.5, "baseline_s": 1.5, "curl_rate_suppress": 4.0}
    est = LiftEstimator(calib_with_tap_ref(), cfg)            # aspect=1.0 ile kurulmuş
    src = LiveSource(FakeCamera(n=1, shape=(640, 512, 3)), FakeTracker([True]),
                     calib_with_tap_ref(), "right", est)
    assert src._lift is not est
    assert src._lift.aspect == pytest.approx(512 / 640)
    assert (src._lift.w_angle, src._lift.w_length, src._lift.baseline_s) == (2.0, 0.5, 1.5)
    src.set_calibration(calib_with_tap_ref())
    assert src._lift.aspect == pytest.approx(512 / 640)


def test_given_lift_with_matching_aspect_is_kept():
    from tkt.vision.lift import LiftEstimator

    cfg = {"w_angle": 1.0, "w_length": 1.0, "baseline_s": 1.0, "curl_rate_suppress": 3.0}
    est = LiftEstimator(calib_with_tap_ref(), cfg, aspect=640 / 512)
    src = LiveSource(FakeCamera(n=1, shape=(512, 640, 3)), FakeTracker([True]),
                     calib_with_tap_ref(), "right", est)
    assert src._lift is est


def test_open_live_source_lift_uses_frame_aspect(tmp_path, monkeypatch):
    cfg = tmp_path / "v.toml"
    cfg.write_text(
        '[camera]\nindex = 0\nwidth = 640\nheight = 512\nfps = 60\n[tracker]\nmodel = "m"\n'
        'min_detection = 0.5\nmin_tracking = 0.5\n[hand]\nside = "right"\ncalibration = "c"\n'
        '[lift]\nw_angle = 1.0\nw_length = 1.0\nbaseline_s = 1.0\ncurl_rate_suppress = 3.0\n',
        encoding="utf-8")
    monkeypatch.setattr(ls, "Camera", lambda *a: FakeCamera(n=1, shape=(640, 512, 3)))
    monkeypatch.setattr(ls, "HandTracker", lambda *a: FakeTracker([]))
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: calib_with_tap_ref())
    src = ls.open_live_source(cfg)
    assert src._lift.aspect == pytest.approx(512 / 640)


def test_open_live_source_resolves_camera_by_name(tmp_path, monkeypatch):
    import tkt.vision.camera as cam_mod

    cfg = tmp_path / "v.toml"
    cfg.write_text(
        '[camera]\nindex = 0\nname = "lvnext"\nwidth = 640\nheight = 512\nfps = 60\n'
        'backend = "dshow"\n[tracker]\nmodel = "m"\nmin_detection = 0.5\nmin_tracking = 0.5\n'
        '[hand]\nside = "right"\ncalibration = "c"\n[lift]\nw_angle = 1.0\n', encoding="utf-8")
    made = {}

    def make_cam(*a):
        made["cam"] = a
        return FakeCamera(n=1)

    monkeypatch.setattr(ls, "Camera", make_cam)
    monkeypatch.setattr(ls, "HandTracker", lambda *a: FakeTracker([]))
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: None)
    monkeypatch.setattr(cam_mod, "list_device_names", lambda backend: ["Laptop", "LVNext Camera"])
    ls.open_live_source(cfg)
    assert made["cam"][0] == 1


@pytest.mark.camera
def test_real_live_source_opens():
    src = ls.open_live_source("config/vision.toml")
    try:
        assert src.info.width > 0
    finally:
        src.close()


def test_lift_off_passed_to_build_frame(monkeypatch):
    seen = []

    def spy(hand, t, frame_id, calib, info, lift=None, prev_hand_size=None, lift_off=None):
        seen.append(lift_off)
        return fake_build_frame(hand, t, frame_id, calib, info, lift, prev_hand_size)

    monkeypatch.setattr(ls.features, "build_frame", spy)
    src = LiveSource(FakeCamera(n=3), FakeTracker([True] * 3), DEFAULT, "right")
    src.lift_off = 0.3
    list(src.frames())
    assert seen and all(v == 0.3 for v in seen)


def test_lift_off_config():
    assert ls._lift_off({}) is None
    assert ls._lift_off({"lift_off": 0.25}) == 0.25
    with pytest.raises(ValueError):
        ls._lift_off({"lift_off": 0})


# ---------------------------------------------------------------- aktif alan (yüzey)

SURFACE_CORNERS = ((0.35, 0.20), (0.65, 0.20), (0.85, 0.80), (0.15, 0.80))


def _surface():
    from tkt.vision.calibration import SurfaceCalibration
    return SurfaceCalibration(SURFACE_CORNERS, image_aspect=512 / 640)


def test_surface_off_leaves_none():
    src = LiveSource(FakeCamera(n=3), FakeTracker([True, False, True]), DEFAULT, "right")
    assert src.surface is None
    frames = list(src.frames())
    assert all(f.surface is None for fr in frames for f in fr.fingers.values())


def test_surface_on_fills_fingers_and_can_be_turned_off():
    surface = _surface()
    src = LiveSource(FakeCamera(n=4), FakeTracker([True, False, True, True]), DEFAULT, "right")
    src.set_surface(surface)
    assert src.surface is surface
    it = src.frames()
    f0, f1, f2 = next(it), next(it), next(it)
    expected = surface.to_surface((0.5, 0.5))
    for f in (f0, f2):
        assert set(f.fingers) == set(FINGERS)
        for finger in f.fingers.values():
            assert finger.surface == pytest.approx(expected)
            assert finger.img == (0.5, 0.5) and finger.curl == 0.1   # diğer alanlar aynı
    assert not f1.present and f1.fingers == {}
    src.set_surface(None)
    f3 = next(it)
    assert all(f.surface is None for f in f3.fingers.values())


def _surface_cfg(tmp_path, enabled, file):
    cfg = tmp_path / "v.toml"
    cfg.write_text(
        '[camera]\nindex = 0\nwidth = 640\nheight = 512\nfps = 60\n[tracker]\nmodel = "m"\n'
        'min_detection = 0.5\nmin_tracking = 0.5\n[hand]\nside = "right"\ncalibration = "c"\n'
        f'[lift]\nw_angle = 1.0\n[surface]\nenabled = {str(enabled).lower()}\n'
        f'file = "{file.as_posix()}"\n', encoding="utf-8")
    return cfg


@pytest.mark.parametrize("enabled, exists, on", [
    (True, True, True), (False, True, False), (True, False, False)])
def test_open_live_source_surface_config(tmp_path, monkeypatch, enabled, exists, on):
    from tkt.vision import calibration

    path = tmp_path / "surface.json"
    if exists:
        calibration.save(_surface(), path)
    monkeypatch.setattr(ls, "Camera", lambda *a: FakeCamera(n=1))
    monkeypatch.setattr(ls, "HandTracker", lambda *a: FakeTracker([True]))
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: None)
    src = ls.open_live_source(_surface_cfg(tmp_path, enabled, path))
    assert (src.surface is not None) == on
    frame = next(src.frames())
    assert (frame.fingers["index"].surface is not None) == on


def test_open_live_source_bad_surface_file_is_off(tmp_path, monkeypatch):
    path = tmp_path / "surface.json"
    path.write_text("{bozuk")
    monkeypatch.setattr(ls, "Camera", lambda *a: FakeCamera(n=1))
    monkeypatch.setattr(ls, "HandTracker", lambda *a: FakeTracker([]))
    monkeypatch.setattr(ls.hand_calib, "load", lambda p: None)
    assert ls.open_live_source(_surface_cfg(tmp_path, True, path)).surface is None


def test_repo_config_surface_is_off():
    cfg = load_vision_config("config/vision.toml")
    assert cfg["surface"]["enabled"] is False
    assert cfg["surface"]["aspect"] == 1.5
