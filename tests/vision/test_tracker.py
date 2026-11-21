import io
from pathlib import Path

import numpy as np
import pytest

from tkt.vision import tracker as tr
from tkt.vision.tracker import HandTracker, RawHand, download_model


class _P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Cat:
    def __init__(self, score):
        self.score = score


class _Result:
    def __init__(self, present, handedness_score=0.9):
        if present:
            self.hand_landmarks = [[_P(i / 40, i / 50) for i in range(21)]]
            self.hand_world_landmarks = [[_P(i / 100, i / 200, -i / 300) for i in range(21)]]
            self.handedness = [[_Cat(handedness_score)]]
        else:
            self.hand_landmarks, self.hand_world_landmarks, self.handedness = [], [], []


class _FakeLandmarker:
    def __init__(self, results):
        self.results = list(results)
        self.stamps = []
        self.closed = False

    def detect_for_video(self, image, ts):
        self.stamps.append(ts)
        return self.results.pop(0)

    def close(self):
        self.closed = True


class _FakeMp:
    class ImageFormat:
        SRGB = 1

    class Image:
        def __init__(self, image_format, data):
            self.data = data


def make_tracker(results):
    """Model yüklemeden HandTracker kurar."""
    t = HandTracker.__new__(HandTracker)
    t._mp = _FakeMp
    t._landmarker = _FakeLandmarker(results)
    t._last_ts_ms = -1
    return t


IMG = np.zeros((8, 6, 3), np.uint8)


def test_process_returns_rawhand_shapes():
    t = make_tracker([_Result(True)])
    h = t.process(IMG, 1.0)
    assert isinstance(h, RawHand)
    assert h.image_pts.shape == (21, 2) and h.world_pts.shape == (21, 3)
    assert h.score == 1.0          # el bulundu; handedness skoru kullanılmaz
    assert h.image_pts[20, 0] == pytest.approx(0.5)


def test_score_does_not_depend_on_handedness():
    # Aynı geometri, MediaPipe sağ/sol skoru 0.51 ve 0.99: el varlığı aynı.
    t = make_tracker([_Result(True, 0.51), _Result(True, 0.99)])
    a, b = t.process(IMG, 1.0), t.process(IMG, 1.1)
    assert a.score == b.score == 1.0


def test_no_hand_returns_none():
    assert make_tracker([_Result(False)]).process(IMG, 1.0) is None


def test_timestamps_ms_strictly_increasing():
    t = make_tracker([_Result(False)] * 4)
    for sec in (1.0, 1.0004, 1.0004, 0.5):    # aynı ms ve geri giden zaman
        t.process(IMG, sec)
    stamps = t._landmarker.stamps
    assert stamps[0] == 1000
    assert all(isinstance(s, int) for s in stamps)
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_close_closes_landmarker():
    t = make_tracker([])
    lm = t._landmarker
    t.close()
    assert lm.closed
    t.close()


def test_missing_model_gives_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        HandTracker(str(tmp_path / "yok.task"), 0.5, 0.5)


def test_download_model_writes_file_atomically(tmp_path, monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return io.BytesIO(b"model-bytes")

    monkeypatch.setattr(tr.urllib.request, "urlopen", fake_urlopen)
    dest = download_model(tmp_path / "models" / "hand_landmarker.task")
    assert dest.read_bytes() == b"model-bytes"
    assert seen["url"] == tr.MODEL_URL
    assert not Path(str(dest) + ".part").exists()


def test_download_failure_leaves_no_partial(tmp_path, monkeypatch):
    def boom(url, timeout=None):
        raise OSError("ağ yok")

    monkeypatch.setattr(tr.urllib.request, "urlopen", boom)
    with pytest.raises(OSError):
        download_model(tmp_path / "m.task")
    assert list(tmp_path.iterdir()) == []


def test_download_cli_uses_config_path(tmp_path, monkeypatch):
    cfg = tmp_path / "v.toml"
    target = tmp_path / "out" / "m.task"
    cfg.write_text(f'[tracker]\nmodel = "{target.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(tr.urllib.request, "urlopen", lambda url, timeout=None: io.BytesIO(b"x"))
    tr.main(["--download", "--config", str(cfg)])
    assert target.read_bytes() == b"x"
