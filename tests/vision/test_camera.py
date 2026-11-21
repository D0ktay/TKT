import sys
import threading
import time

import cv2
import numpy as np
import pytest

from tkt.vision import camera as cam_mod
from tkt.vision.camera import Camera, probe, resolve_backend


class FakeCapture:
    """Gerçek aygıt açmayan VideoCapture; her karede değeri artan 4x6 görüntü verir."""

    instances: list["FakeCapture"] = []
    interval = 0.005
    fail = False

    def __init__(self, index, flag):
        self.index, self.flag = index, flag
        self.sets: list[tuple[int, float]] = []
        self.opened = True
        self.released = False
        self.n = 0
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def read(self):
        time.sleep(self.interval)
        if FakeCapture.fail:
            return False, None
        self.n += 1
        img = np.zeros((4, 6, 3), np.uint8)
        img[0, 0] = self.n % 256      # sol üst piksel kare numarasını taşır
        return True, img

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    FakeCapture.instances = []
    FakeCapture.interval = 0.005
    FakeCapture.fail = False
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCapture)
    return FakeCapture


def test_resolve_backend():
    assert resolve_backend("dshow") == "dshow"
    expected = {"darwin": "avfoundation", "win32": "dshow"}.get(sys.platform, "auto")
    assert resolve_backend("auto") == expected
    with pytest.raises(ValueError):
        resolve_backend("nope")
    with pytest.raises(NotImplementedError):
        Camera(0, 640, 512, 60, backend="pyav")


def test_property_order_size_then_fps_then_fourcc(fake_cv2):
    c = Camera(0, 640, 512, 60, backend="dshow")
    c.start()
    try:
        props = [p for p, _ in fake_cv2.instances[0].sets]
        assert props == [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                         cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC]
        assert fake_cv2.instances[0].flag == cv2.CAP_DSHOW
    finally:
        c.close()


def test_avfoundation_does_not_set_fourcc(fake_cv2):
    c = Camera(0, 640, 512, 60, backend="avfoundation")
    c.start()
    try:
        props = [p for p, _ in fake_cv2.instances[0].sets]
        assert cv2.CAP_PROP_FOURCC not in props
        assert fake_cv2.instances[0].flag == cv2.CAP_AVFOUNDATION
    finally:
        c.close()


def test_open_failure_raises(monkeypatch, fake_cv2):
    monkeypatch.setattr(FakeCapture, "isOpened", lambda self: False)
    with pytest.raises(RuntimeError):
        Camera(0, 640, 512, 60).start()


def test_frames_rotated_monotonic_and_increasing_ids(fake_cv2):
    c = Camera(0, 6, 4, 60, rotate_180=True)
    c.start()
    try:
        before = time.monotonic()
        ids, ts = [], []
        for _ in range(10):
            f = c.read()
            assert f is not None
            ids.append(f.frame_id)
            ts.append(f.t)
            # 180° döndürme: sol üstteki işaret sağ alta gider
            assert f.image[3, 5, 0] != 0 and f.image[0, 0, 0] == 0
        assert ids == sorted(set(ids)) and ids[0] >= 1
        assert ts == sorted(ts) and ts[0] >= before
        assert c.frame_size == (6, 4)
    finally:
        c.close()


def test_no_rotation(fake_cv2):
    c = Camera(0, 6, 4, 60, rotate_180=False)
    c.start()
    try:
        f = c.read()
        assert f.image[0, 0, 0] != 0
    finally:
        c.close()


def test_slow_consumer_skips_ids_and_gets_latest(fake_cv2):
    c = Camera(0, 6, 4, 60)
    c.start()
    try:
        a = c.read()
        time.sleep(0.15)             # işleme yetişmiyor
        b = c.read()
        assert b.frame_id > a.frame_id + 1     # kare biriktirilmez, id atlar
        assert b.t > a.t
    finally:
        c.close()


def test_read_does_not_repeat_frame_and_times_out(fake_cv2):
    fake_cv2.fail = True               # kare gelmiyor
    c = Camera(0, 6, 4, 60)
    c.start()
    try:
        t0 = time.monotonic()
        assert c.read(timeout=0.1) is None
        assert time.monotonic() - t0 < 1.0
    finally:
        c.close()


def test_read_before_start_returns_none():
    assert Camera(0, 6, 4, 60).read(timeout=0.05) is None


def test_measured_fps(fake_cv2):
    fake_cv2.interval = 0.01           # ~100 fps
    c = Camera(0, 6, 4, 60)
    c.start()
    try:
        time.sleep(0.5)
        fps = c.measured_fps()
        assert 5 < fps < 130          # CI makineleri yavaş olabilir; alt sınır gevşek
    finally:
        c.close()


def test_close_stops_thread_and_releases(fake_cv2):
    c = Camera(0, 6, 4, 60)
    c.start()
    c.close()
    assert fake_cv2.instances[0].released
    assert c._thread is None
    assert not any(t.name == "tkt-camera" and t.is_alive() for t in threading.enumerate())
    c.close()  # ikinci kapatma zararsız


def test_probe_reports_real_fps(fake_cv2):
    fake_cv2.interval = 0.01
    c = Camera(0, 6, 4, 60)
    c.start()
    try:
        r = probe(c, seconds=0.5)
    finally:
        c.close()
    assert 5 < r["fps"] < 130
    assert r["size"] == (6, 4)


@pytest.mark.camera
def test_real_camera_gives_frames():
    c = Camera(0, 640, 512, 60)
    c.start()
    try:
        assert c.read(timeout=3.0) is not None
    finally:
        c.close()
