import ctypes
import itertools
import subprocess
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
    silent_flags: set[int] = set()         # açılır (isOpened=True) ama hiç kare vermez
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
        if FakeCapture.fail or self.flag in FakeCapture.silent_flags:
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
    FakeCapture.silent_flags = set()
    monkeypatch.setattr(FakeCapture, "reported", dict(FakeCapture.reported))   # test sonunda geri
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCapture)
    return FakeCapture


class GatedCapture(FakeCapture):
    """read() `want` olayı set edilene kadar bloklanır, sonra tek kare verir.

    Kare sırası testte olaylarla yönetilir; gerçek saate dayanmaz.
    """

    def __init__(self, index, flag):
        super().__init__(index, flag)
        self.want = threading.Event()
        self.in_read = threading.Event()
        self.reading = False
        self.release_during_read = False
        self.thread: threading.Thread | None = None

    def read(self):
        self.thread = threading.current_thread()
        self.reading = True
        self.in_read.set()
        self.want.wait()
        self.want.clear()
        self.in_read.clear()
        self.n += 1
        img = np.zeros((4, 6, 3), np.uint8)
        img[0, 0] = self.n % 256
        self.reading = False
        return True, img

    def release(self):
        if self.reading:
            self.release_during_read = True
        self.released = True


@pytest.fixture
def gated_cv2(monkeypatch):
    FakeCapture.instances = []
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", GatedCapture)
    # close() read'de bloklu thread'i uzun beklemesin; thread'ler teardown'da serbest kalır.
    monkeypatch.setattr(cam_mod, "_CLOSE_TIMEOUT_S", 0.01)
    yield GatedCapture
    for cap in FakeCapture.instances:    # bloklu thread'leri bırak ve bitmelerini bekle
        cap.want.set()
        if cap.thread is not None:
            cap.thread.join(5.0)


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


def test_frames_rotated_monotonic_and_increasing_ids(gated_cv2):
    before = time.monotonic()          # referans saati thread başlamadan alınır
    c = Camera(0, 6, 4, 60, rotate_180=True, backend="avfoundation")
    c.start()
    cap = gated_cv2.instances[0]
    try:
        ids, ts = [], []
        for expected in range(1, 11):
            cap.want.set()             # tam bir kare üret
            f = c.read(timeout=5.0)
            assert f is not None and f.frame_id == expected
            ids.append(f.frame_id)
            ts.append(f.t)
            # 180° döndürme: sol üstteki işaret sağ alta gider
            assert f.image[3, 5, 0] != 0 and f.image[0, 0, 0] == 0
        assert ids == list(range(1, 11))
        assert ts == sorted(ts) and ts[0] >= before
        assert c.frame_size == (6, 4)
    finally:
        c.close()


def test_t_strictly_increasing_when_clock_repeats_or_goes_back(gated_cv2, monkeypatch):
    # Windows'ta monotonic ~15 ms çözünürlüklü: iki kare aynı t'yi alabilir.
    clock = itertools.chain([5.0, 5.0, 4.9, 5.2], itertools.repeat(6.0))
    monkeypatch.setattr(cam_mod, "_clock", lambda: next(clock))
    c = Camera(0, 6, 4, 60, backend="avfoundation")
    c.start()
    cap = gated_cv2.instances[0]
    try:
        ts = []
        for _ in range(4):
            cap.want.set()
            ts.append(c.read(timeout=5.0).t)
    finally:
        c.close()
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))
    assert ts == pytest.approx([5.0, 5.0 + cam_mod._MIN_DT_S, 5.0 + 2 * cam_mod._MIN_DT_S, 5.2])


def test_close_while_read_blocked_releases_in_thread_and_publishes_nothing(
        gated_cv2, monkeypatch, caplog):
    c = Camera(0, 6, 4, 60, backend="avfoundation")
    c.start()
    cap = gated_cv2.instances[0]
    assert cap.in_read.wait(5.0)       # yakalama thread'i read() içinde bloklu
    thread = c._thread
    with caplog.at_level("WARNING", logger=cam_mod.log.name):
        c.close()
    assert any("bitmedi" in r.getMessage() for r in caplog.records)
    assert not cap.released            # read sürerken aynı nesne bırakılmaz
    cap.want.set()                     # bloklu read kapanıştan sonra bir kare döndürür
    thread.join(5.0)
    assert not thread.is_alive()
    assert cap.released and not cap.release_during_read
    assert c.read(0) is None           # kapanıştan sonra kare yayımlanmaz


def test_read_after_close_returns_none(gated_cv2):
    c = Camera(0, 6, 4, 60, backend="avfoundation")
    c.start()
    cap = gated_cv2.instances[0]
    cap.want.set()
    while c.frame_size is None:        # kare yayımlandı ama okunmadı
        time.sleep(0.001)
    c.close()
    assert c.read(0) is None


def test_auto_dshow_without_frames_falls_back_to_msmf(fake_cv2, monkeypatch, caplog):
    monkeypatch.setattr(cam_mod.sys, "platform", "win32")
    monkeypatch.setattr(cam_mod, "_FIRST_FRAME_TIMEOUT_S", 0.05)
    fake_cv2.silent_flags = {cv2.CAP_DSHOW}      # isOpened=True ama kare yok
    c = Camera(0, 640, 512, 60, backend="auto")
    with caplog.at_level("WARNING", logger=cam_mod.log.name):
        c.start()
    try:
        assert [i.flag for i in fake_cv2.instances] == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        assert fake_cv2.instances[0].released
        assert c.backend == "msmf" and c.negotiated()["backend"] == "msmf"
        assert any("msmf" in r.getMessage() for r in caplog.records)
        assert c.read(timeout=5.0) is not None
    finally:
        c.close()


def test_auto_dshow_with_frames_keeps_dshow(fake_cv2, monkeypatch):
    monkeypatch.setattr(cam_mod.sys, "platform", "win32")
    c = Camera(0, 640, 512, 60, backend="auto")
    c.start()
    try:
        assert [i.flag for i in fake_cv2.instances] == [cv2.CAP_DSHOW]
        assert c.read(timeout=5.0) is not None
    finally:
        c.close()


def test_explicit_dshow_is_not_verified_by_frames(fake_cv2):
    fake_cv2.silent_flags = {cv2.CAP_DSHOW}
    c = Camera(0, 640, 512, 60, backend="dshow")
    c.start()
    try:
        assert c.backend == "dshow" and len(fake_cv2.instances) == 1
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
    monkeypatch.setattr(cam_mod, "list_device_names", lambda backend: ["Laptop", "LVNext Camera"])
    cam_mod.main(["--list", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "index" in out and "640x512" in out and "açılamadı" in out
    assert "LVNext Camera" in out and "ad" in out.splitlines()[0]
    assert "seçilen kamera: index=0" in out and "config index=0" in out
    assert len(out.strip().splitlines()) == 1 + len(cam_mod.LIST_INDICES) + 1


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
    assert "seçilen kamera: index=0" in out


# ---------------------------------------------------------------------- _is_gray


def _gradient_gray(h=120, w=160):
    g = np.tile(np.linspace(20, 235, w, dtype=np.uint8), (h, 1))
    return np.dstack([g, g, g])


def test_is_gray_exact_equal_and_single_channel():
    assert cam_mod._is_gray(_gradient_gray())
    assert cam_mod._is_gray(np.zeros((4, 4), np.uint8))
    assert cam_mod._is_gray(np.zeros((4, 4, 1), np.uint8))


def test_is_gray_tolerates_jpeg_decoded_gray():
    rng = np.random.default_rng(0)
    img = _gradient_gray().astype(np.int16) + rng.integers(-12, 13, (120, 160, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # küçük kanal farkları olabilir; ne olursa olsun gri sayılmalı
    assert cam_mod._is_gray(decoded)


def test_is_gray_small_channel_noise_is_gray_but_old_exact_check_would_not():
    img = _gradient_gray()
    img[10, 10, 0] += 3          # tek pikselde küçük fark
    img[20, 20, 2] += 5
    assert not np.array_equal(img[..., 0], img[..., 1])
    assert cam_mod._is_gray(img)


def test_is_gray_rejects_real_color():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    assert not cam_mod._is_gray(img)
    tint = _gradient_gray().astype(np.int16)
    tint[..., 2] += 40           # sıcak ton: R kanalı ~40 fazla
    assert not cam_mod._is_gray(np.clip(tint, 0, 255).astype(np.uint8))


def test_is_gray_sparse_strong_color_fails_p99():
    img = _gradient_gray()
    img[:3, :] = (0, 0, 255)     # satırların ~%2,5'i kırmızı: ortalama küçük, p99 büyük
    assert not cam_mod._is_gray(img)


# ---------------------------------------------------------------------- ad listesi


FFMPEG_OUT = """\
[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] LVNext CPD Camera
[AVFoundation indev @ 0x7f8] [1] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] [2] Capture screen 0
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x7f8] [1] LVNext Mic
: Input/output error
"""


def test_parse_avfoundation_list_video_only_in_order():
    assert cam_mod.parse_avfoundation_list(FFMPEG_OUT) == [
        "LVNext CPD Camera", "FaceTime HD Camera", "Capture screen 0"]
    assert cam_mod.parse_avfoundation_list("hata\n") == []


def test_avfoundation_device_names_runs_ffmpeg_and_reads_stderr():
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=FFMPEG_OUT)

    names = cam_mod.avfoundation_device_names(run=run, which=lambda n: "/usr/bin/ffmpeg")
    assert names[0] == "LVNext CPD Camera"
    cmd, kw = calls[0]
    assert cmd[0] == "/usr/bin/ffmpeg" and "avfoundation" in cmd and "-list_devices" in cmd
    assert kw["timeout"] == cam_mod._NAME_LIST_TIMEOUT_S


def test_avfoundation_device_names_none_without_ffmpeg_or_on_failure():
    def boom(*a, **k):
        raise AssertionError("ffmpeg yokken çalıştırılmamalı")

    assert cam_mod.avfoundation_device_names(run=boom, which=lambda n: None) is None

    def timeout(cmd, **k):
        raise subprocess.TimeoutExpired(cmd, 1)

    assert cam_mod.avfoundation_device_names(run=timeout, which=lambda n: "ffmpeg") is None

    def oserr(cmd, **k):
        raise OSError("yok")

    assert cam_mod.avfoundation_device_names(run=oserr, which=lambda n: "ffmpeg") is None
    empty = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="x")  # noqa: E731
    assert cam_mod.avfoundation_device_names(run=empty, which=lambda n: "ffmpeg") is None


def test_match_device_name_case_insensitive_substring_first_hit():
    names = ["Integrated Webcam_FHD", "LVNext Camera", "LVNext CPD Camera"]
    assert cam_mod.match_device_name(names, "lvnext") == 1
    assert cam_mod.match_device_name(names, "CPD") == 2
    assert cam_mod.match_device_name(names, "obs") is None
    assert cam_mod.match_device_name(names, "  ") is None
    assert cam_mod.match_device_name([], "x") is None


def test_list_device_names_picks_source_by_backend(monkeypatch):
    monkeypatch.setattr(cam_mod, "dshow_device_names", lambda: ["d"])
    monkeypatch.setattr(cam_mod, "avfoundation_device_names", lambda: ["a"])
    assert cam_mod.list_device_names("dshow") == ["d"]
    assert cam_mod.list_device_names("avfoundation") == ["a"]
    assert cam_mod.list_device_names("msmf") is None     # MSMF sırası DirectShow'dan farklı
    monkeypatch.setattr(sys, "platform", "win32")
    assert cam_mod.list_device_names("auto") == ["d"]
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cam_mod.list_device_names("auto") == ["a"]
    monkeypatch.setattr(sys, "platform", "linux")
    assert cam_mod.list_device_names("auto") is None


def test_dshow_device_names_none_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert cam_mod.dshow_device_names() is None


# --- DirectShow: gerçek vtable bellek düzenine sahip sahte COM nesneleri

_FUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
_HR = ctypes.c_long
_VP = ctypes.c_void_p


class FakeCom:
    """ctypes ile gerçek vtable kuran sahte COM dünyası + sahte ole32/oleaut32."""

    def __init__(self, names, *, init_hr=0, create_hr=0, category_hr=0):
        self.names = names              # str | None (None: FriendlyName okunamaz)
        self.init_hr, self.create_hr, self.category_hr = init_hr, create_hr, category_hr
        self.keep: list = []
        self.created = 0
        self.released = 0
        self.uninit = 0
        self.variant_cleared = 0
        self.next_i = 0

    # -- nesne kurma
    def obj(self, methods):
        table = (ctypes.c_void_p * 24)()
        release = _FUNCTYPE(_HR, _VP)(self._release)
        self.keep.append(release)
        table[2] = ctypes.cast(release, ctypes.c_void_p).value
        for slot, (proto, fn) in methods.items():
            cb = proto(fn)
            self.keep.append(cb)
            table[slot] = ctypes.cast(cb, ctypes.c_void_p).value
        holder = (ctypes.c_void_p * 1)(ctypes.addressof(table))
        self.keep += [table, holder]
        self.created += 1
        return ctypes.addressof(holder)

    def _release(self, this):
        self.released += 1
        return 0

    @staticmethod
    def _put(ptr, value):
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0] = value

    def bag(self, name):
        def read(this, prop, pvar, perr):
            if name is None:
                return -2147467259          # E_FAIL
            v = ctypes.cast(pvar, ctypes.POINTER(cam_mod._VARIANT))[0]
            buf = ctypes.create_unicode_buffer(name)
            self.keep.append(buf)
            v.vt = 8
            v.bstr = ctypes.cast(buf, ctypes.c_wchar_p)
            return 0
        return self.obj({3: (_FUNCTYPE(_HR, _VP, ctypes.c_wchar_p, _VP, _VP), read)})

    def moniker(self, name):
        def bind(this, pbc, pml, riid, ppv):
            self._put(ppv, self.bag(name))
            return 0
        return self.obj({9: (_FUNCTYPE(_HR, _VP, _VP, _VP, _VP, _VP), bind)})

    def enumerator(self):
        def nxt(this, celt, rgelt, fetched):
            if self.next_i >= len(self.names):
                return 1                    # S_FALSE
            self._put(rgelt, self.moniker(self.names[self.next_i]))
            self.next_i += 1
            return 0
        return self.obj({3: (_FUNCTYPE(_HR, _VP, ctypes.c_ulong, _VP, _VP), nxt)})

    def dev_enum(self):
        def create(this, clsid, ppenum, flags):
            if self.category_hr < 0:
                return self.category_hr
            if self.category_hr == 1:
                return 1                    # S_FALSE: kategori boş, enum null kalır
            self._put(ppenum, self.enumerator())
            return 0
        return self.obj({3: (_FUNCTYPE(_HR, _VP, _VP, _VP, ctypes.c_ulong), create)})

    # -- sahte DLL'ler (yöntemler byref nesnelerinin `_obj`'ünden yazar)
    @property
    def ole32(self):
        fake = self

        class Ole32:
            def CoInitializeEx(self, reserved, mode):
                return fake.init_hr

            def CoUninitialize(self):
                fake.uninit += 1

            def CoCreateInstance(self, clsid, outer, ctx, iid, ppv):
                if fake.create_hr < 0:
                    return fake.create_hr
                ppv._obj.value = fake.dev_enum()
                return 0

        return Ole32()

    @property
    def oleaut32(self):
        fake = self

        class Oleaut32:
            def VariantClear(self, pvar):
                fake.variant_cleared += 1
                return 0

        return Oleaut32()


def test_variant_and_guid_sizes():
    assert ctypes.sizeof(cam_mod._VARIANT) == 16
    g = cam_mod._guid("860BB310-5D01-11D0-BD3B-00A0C911CE86")
    assert ctypes.sizeof(g) == 16 and g.Data1 == 0x860BB310 and g.Data3 == 0x11D0
    assert bytes(g.Data4) == bytes.fromhex("BD3B00A0C911CE86")


def test_dshow_device_names_lists_friendly_names_in_order_and_releases_everything():
    names = ["Integrated Webcam_FHD", "LVNext Camera", "LVNext CPD Camera", "OBS Virtual Camera"]
    com = FakeCom(names)
    assert cam_mod.dshow_device_names(com.ole32, com.oleaut32) == names
    assert com.variant_cleared == 4
    assert com.uninit == 1
    assert com.released == com.created           # dev_enum + enum + 4 x (moniker + bag)
    assert com.created == 2 + 8


def test_dshow_unreadable_name_keeps_index_alignment():
    com = FakeCom(["A", None, "C"])
    assert cam_mod.dshow_device_names(com.ole32, com.oleaut32) == ["A", "", "C"]
    assert com.released == com.created


def test_dshow_empty_category_gives_empty_list():
    com = FakeCom([], category_hr=1)
    assert cam_mod.dshow_device_names(com.ole32, com.oleaut32) == []
    assert com.released == com.created and com.uninit == 1


def test_dshow_failures_give_none_and_clean_up():
    com = FakeCom(["A"], create_hr=-2147221164)      # REGDB_E_CLASSNOTREG
    assert cam_mod.dshow_device_names(com.ole32, com.oleaut32) is None
    assert com.uninit == 1
    com = FakeCom(["A"], category_hr=-2147467259)
    assert cam_mod.dshow_device_names(com.ole32, com.oleaut32) is None
    assert com.released == com.created


def test_dshow_changed_mode_does_not_uninitialize_but_still_lists():
    com = FakeCom(["A"], init_hr=-2147417850)         # RPC_E_CHANGED_MODE
    assert cam_mod.dshow_device_names(com.ole32, com.oleaut32) == ["A"]
    assert com.uninit == 0


# ---------------------------------------------------------------------- kamera seçimi


class IrOnlyCam(ListCam):
    """Yalnızca gri ama yatay kare veren kamera (ir için yedek yol)."""

    shapes = {1: (720, 1280, 3), 2: (512, 640, 3)}

    def read(self, timeout=1.0):
        cap = super().read(timeout)
        if self.index == 2:
            return cam_mod.CapturedFrame(np.full(self.shapes[2], 9, np.uint8), cap.t, cap.frame_id)
        return cap


CAM_CFG = {"index": 0, "width": 640, "height": 512, "fps": 60, "backend": "dshow"}
DEV_NAMES = ["Integrated Webcam_FHD", "LVNext Camera", "LVNext CPD Camera", "OBS Virtual Camera"]


def test_resolve_by_name_wins_and_does_not_probe():
    def boom(*a, **k):
        raise AssertionError("ad eşleşince yoklama yapılmamalı")

    ch = cam_mod.resolve_camera({**CAM_CFG, "name": "lvnext cpd", "kind": "rgb"},
                                names=DEV_NAMES, camera_factory=boom)
    assert ch.index == 2 and ch.name == "LVNext CPD Camera"
    assert "ad eşleşti" in ch.reason and "index 2" in ch.reason
    assert "index=2" in cam_mod.format_choice(ch) and "LVNext CPD" in cam_mod.format_choice(ch)


def test_resolve_calls_list_device_names_when_names_not_given(monkeypatch):
    got = []
    monkeypatch.setattr(cam_mod, "list_device_names", lambda b: got.append(b) or DEV_NAMES)
    ch = cam_mod.resolve_camera({**CAM_CFG, "name": "OBS"})
    assert ch.index == 3 and got == ["dshow"]


def test_resolve_without_name_uses_config_index_and_never_lists(monkeypatch):
    monkeypatch.setattr(cam_mod, "list_device_names",
                        lambda b: (_ for _ in ()).throw(AssertionError("listelenmemeli")))
    ch = cam_mod.resolve_camera({**CAM_CFG, "index": 2, "name": ""})
    assert (ch.index, ch.name) == (2, None) and "config index=2" in ch.reason


def test_resolve_name_without_match_falls_back_to_index():
    ch = cam_mod.resolve_camera({**CAM_CFG, "index": 1, "name": "Kinect"}, names=DEV_NAMES)
    assert ch.index == 1 and "ad eşleşmedi" in ch.reason and "config index=1" in ch.reason
    assert ch.name == "LVNext Camera"


def test_resolve_name_list_unavailable_falls_to_kind_probe():
    ch = cam_mod.resolve_camera({**CAM_CFG, "name": "LVNext", "kind": "ir"}, names=None,
                                camera_factory=ListCam)
    assert ch.index == 3 and "ad listesi alınamadı" in ch.reason and "kind='ir'" in ch.reason


def test_resolve_kind_ir_prefers_gray_portrait_and_rgb_picks_color():
    assert cam_mod.resolve_camera({**CAM_CFG, "kind": "ir"}, camera_factory=ListCam).index == 3
    assert cam_mod.resolve_camera({**CAM_CFG, "kind": "rgb"}, camera_factory=ListCam).index == 1


def test_resolve_kind_ir_without_portrait_takes_first_gray():
    ch = cam_mod.resolve_camera({**CAM_CFG, "kind": "ir"}, camera_factory=IrOnlyCam)
    assert ch.index == 2


def test_resolve_kind_probe_stops_at_first_match():
    opened = []

    class Tracking(ListCam):
        def start(self):
            opened.append(self.index)
            super().start()

    cam_mod.resolve_camera({**CAM_CFG, "kind": "rgb"}, camera_factory=Tracking)
    assert opened == [0, 1]                 # 0 gri, 1 renkli; sonrası açılmaz


def test_resolve_kind_without_result_falls_back_to_index():
    class Nothing(ListCam):
        shapes = {}

    ch = cam_mod.resolve_camera({**CAM_CFG, "index": 4, "kind": "ir"}, camera_factory=Nothing)
    assert ch.index == 4 and "yoklaması sonuç vermedi" in ch.reason


def test_resolve_kind_any_never_probes():
    def boom(*a, **k):
        raise AssertionError("any iken yoklama yapılmamalı")

    ch = cam_mod.resolve_camera({**CAM_CFG, "kind": "any"}, camera_factory=boom)
    assert ch.index == 0


def test_resolve_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        cam_mod.resolve_camera({**CAM_CFG, "kind": "thermal"})


def test_resolve_defaults_when_keys_missing():
    ch = cam_mod.resolve_camera({"index": 3, "width": 1, "height": 1, "fps": 1})
    assert ch.index == 3


def test_list_cameras_adds_names_and_table_column():
    rows = cam_mod.list_cameras((0, 1, 2, 3), width=640, height=512, fps=60, seconds=0.2,
                                camera_factory=ListCam, names=DEV_NAMES)
    assert [r["name"] for r in rows] == DEV_NAMES
    lines = cam_mod.format_camera_table(rows).splitlines()
    assert lines[0].rstrip().endswith("ad")
    assert lines[1].endswith("Integrated Webcam_FHD")
    assert "açılamadı" in lines[3] and lines[3].endswith("LVNext CPD Camera")


def test_list_cameras_without_names_shows_dash():
    rows = cam_mod.list_cameras((0, 2), width=640, height=512, fps=60, seconds=0.2,
                                camera_factory=ListCam)
    assert all(r["name"] is None for r in rows)
    assert all(line.endswith("-") for line in cam_mod.format_camera_table(rows).splitlines()[1:])


def test_main_probe_uses_name_resolution(fake_cv2, monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "v.toml"
    cfg.write_text('[camera]\nindex = 0\nname = "lvnext"\nwidth = 640\nheight = 512\nfps = 60\n'
                   'backend = "dshow"\n', encoding="utf-8")
    monkeypatch.setattr(cam_mod, "list_device_names", lambda b: DEV_NAMES)
    monkeypatch.setattr(cam_mod, "probe", lambda cam, s: {
        "fps": 59.0, "frames": 10, "seconds": 0.2, "size": (640, 512)})
    cam_mod.main(["--probe", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "seçilen kamera: index=1 (LVNext Camera)" in out
    assert fake_cv2.instances[0].index == 1


def test_repo_config_has_camera_name_and_kind():
    from tkt.vision.live_source import load_vision_config

    cam = load_vision_config("config/vision.toml")["camera"]
    assert cam["name"] == "LVNext" and cam["kind"] in cam_mod.KINDS and cam["index"] == 0


@pytest.mark.camera
def test_real_camera_gives_frames():
    c = Camera(0, 640, 512, 60)
    c.start()
    try:
        assert c.read(timeout=3.0) is not None
    finally:
        c.close()
