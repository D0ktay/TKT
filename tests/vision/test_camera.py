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
    unopenable_flags: set[int] = set()     # bu backend bayraklarıyla açılamaz
    reported = {                            # sürücünün "gerçekleşen" değerleri
        cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 512, cv2.CAP_PROP_FPS: 60.0,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
    }

    def __init__(self, index, flag):
        self.index, self.flag = index, flag
        self.sets: list[tuple[int, float]] = []
        self.opened = flag not in FakeCapture.unopenable_flags
        self.released = False
        self.n = 0
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def get(self, prop):
        return FakeCapture.reported.get(prop, 0.0)

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
    FakeCapture.unopenable_flags = set()
    monkeypatch.setattr(FakeCapture, "reported", dict(FakeCapture.reported))   # test sonunda geri
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


def test_auto_on_windows_selects_dshow(fake_cv2, monkeypatch):
    monkeypatch.setattr(cam_mod.sys, "platform", "win32")
    c = Camera(0, 640, 512, 60, backend="auto")
    c.start()
    try:
        assert c.backend == "dshow"
        assert fake_cv2.instances[0].flag == cv2.CAP_DSHOW
    finally:
        c.close()


def test_dshow_failure_falls_back_to_msmf_with_warning(fake_cv2, caplog):
    fake_cv2.unopenable_flags = {cv2.CAP_DSHOW}
    c = Camera(0, 640, 512, 60, backend="dshow")
    with caplog.at_level("WARNING", logger=cam_mod.log.name):
        c.start()
    try:
        assert [i.flag for i in fake_cv2.instances] == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        assert fake_cv2.instances[0].released          # başarısız deneme bırakıldı
        assert c.backend == "msmf" and c.negotiated()["backend"] == "msmf"
        assert any("msmf" in r.getMessage() for r in caplog.records)
        props = [p for p, _ in fake_cv2.instances[1].sets]
        assert props == [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                         cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC]
    finally:
        c.close()


def test_dshow_and_msmf_both_fail_raises(fake_cv2):
    fake_cv2.unopenable_flags = {cv2.CAP_DSHOW, cv2.CAP_MSMF}
    with pytest.raises(RuntimeError):
        Camera(0, 640, 512, 60, backend="dshow").start()
    assert all(i.released for i in fake_cv2.instances)


def test_other_backends_do_not_fall_back(fake_cv2):
    fake_cv2.unopenable_flags = {cv2.CAP_AVFOUNDATION}
    with pytest.raises(RuntimeError):
        Camera(0, 640, 512, 60, backend="avfoundation").start()
    assert len(fake_cv2.instances) == 1


def test_set_values_are_requested_ones_in_order(fake_cv2):
    c = Camera(0, 1280, 720, 60, backend="dshow")
    c.start()
    try:
        sets = fake_cv2.instances[0].sets
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        assert sets == [(cv2.CAP_PROP_FRAME_WIDTH, 1280), (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                        (cv2.CAP_PROP_FPS, 60), (cv2.CAP_PROP_FOURCC, mjpg)]
    finally:
        c.close()


def test_negotiated_reads_actual_values(fake_cv2):
    assert Camera(0, 1280, 720, 60, backend="dshow").negotiated() == {}   # açılmadan boş
    fake_cv2.reported = {**fake_cv2.reported, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUY2"),
                         cv2.CAP_PROP_FPS: 30.0}
    c = Camera(0, 1280, 720, 60, backend="dshow")
    c.start()
    try:
        # istenen 1280x720@60 MJPG; sürücü 640x512@30 YUY2 verdi
        assert c.negotiated() == {"backend": "dshow", "width": 640, "height": 512,
                                  "fps": 30.0, "fourcc": "YUY2"}
    finally:
        c.close()


def test_negotiated_fourcc_empty_when_backend_reports_none(fake_cv2):
    fake_cv2.reported = {**fake_cv2.reported, cv2.CAP_PROP_FOURCC: 0.0}
    c = Camera(0, 640, 512, 60, backend="avfoundation")
    c.start()
    try:
        assert c.negotiated()["fourcc"] == ""
        assert c.negotiated()["backend"] == "avfoundation"
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


class ListCam:
    """--list için sahte kamera: index'e göre kare şekli ve zaman damgaları sabit."""

    shapes = {0: (512, 640, 3), 1: (720, 1280, 3), 3: (640, 512, 3)}   # 2 ve 4 açılmaz
    dt = 1 / 30

    def __init__(self, index, width, height, fps, rotate_180=True, backend="auto"):
        self.index, self.closed, self.n = index, False, 0
        self.kwargs = (width, height, fps, rotate_180, backend)

    def start(self):
        if self.index not in self.shapes:
            raise RuntimeError("açılamadı")

    def read(self, timeout=1.0):
        shape = self.shapes[self.index]
        self.n += 1
        img = np.full(shape, 7, np.uint8)
        if self.index == 1:                      # renkli: kanallar farklı
            img[..., 2] = 200
        return cam_mod.CapturedFrame(img, 50.0 + self.n * self.dt, self.n)

    def negotiated(self):
        return {"fourcc": "MJPG" if self.index == 3 else "YUY2"}

    def close(self):
        self.closed = True


def test_list_cameras_reports_each_index():
    rows = cam_mod.list_cameras((0, 1, 2, 3), width=640, height=512, fps=60, backend="dshow",
                                seconds=0.5, camera_factory=ListCam)
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert [r["opened"] for r in rows] == [True, True, False, True]
    r0, r1, _, r3 = rows
    assert (r0["width"], r0["height"], r0["orientation"], r0["gray"]) == (640, 512, "yatay", True)
    assert (r1["width"], r1["height"], r1["gray"]) == (1280, 720, False)
    assert (r3["orientation"], r3["gray"]) == ("dikey", True)
    assert r0["fps"] == pytest.approx(30.0)          # zaman damgalarından, gerçek saatten değil


def test_list_cameras_closes_every_camera_and_passes_config():
    made = []

    def factory(*a, **k):
        made.append(ListCam(*a, **k))
        return made[-1]

    cam_mod.list_cameras((0, 2), width=640, height=512, fps=60, backend="dshow",
                         seconds=0.2, camera_factory=factory)
    assert all(c.closed for c in made)
    assert made[0].kwargs == (640, 512, 60, False, "dshow")


def test_format_camera_table():
    rows = cam_mod.list_cameras((0, 1, 2, 3), width=640, height=512, fps=60,
                                seconds=0.2, camera_factory=ListCam)
    lines = cam_mod.format_camera_table(rows).splitlines()
    assert len(lines) == 5 and "fps" in lines[0]
    assert "640x512" in lines[1] and "gri" in lines[1] and "yatay" in lines[1]
    assert "renkli" in lines[2] and "1280x720" in lines[2]
    assert "açılamadı" in lines[3]
    assert "dikey" in lines[4] and "MJPG" in lines[4]


def test_main_list_prints_table(monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text('[camera]\nindex = 0\nwidth = 640\nheight = 512\nfps = 60\n',
                   encoding="utf-8")
    monkeypatch.setattr(cam_mod, "Camera", ListCam)
    cam_mod.main(["--list", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "index" in out and "640x512" in out and "açılamadı" in out
    assert len(out.strip().splitlines()) == 1 + len(cam_mod.LIST_INDICES)


def test_main_requires_probe_or_list(tmp_path):
    with pytest.raises(SystemExit):
        cam_mod.main([])


def test_main_probe_prints_negotiated(fake_cv2, monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text('[camera]\nindex = 0\nwidth = 640\nheight = 512\nfps = 60\n'
                   'backend = "dshow"\n', encoding="utf-8")
    monkeypatch.setattr(cam_mod, "probe", lambda cam, s: {
        "fps": 59.0, "frames": 10, "seconds": 0.2, "size": (640, 512)})
    cam_mod.main(["--probe", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "fourcc=MJPG" in out and "640x512@60.0" in out and "backend=dshow" in out


@pytest.mark.camera
def test_real_camera_gives_frames():
    c = Camera(0, 640, 512, 60)
    c.start()
    try:
        assert c.read(timeout=3.0) is not None
    finally:
        c.close()
