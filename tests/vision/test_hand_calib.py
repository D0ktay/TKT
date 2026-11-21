import json

import pytest

from tkt.core.types import FINGERS
from tkt.vision import hand_calib
from tkt.vision.hand_calib import DEFAULT, HandCalibration, load, save


def _calib() -> HandCalibration:
    return HandCalibration(
        open_ref={"index": 0.21, "middle": 0.18, "ring": 0.25, "pinky": 0.3},
        fist_ref={"index": 2.9, "middle": 3.1, "ring": 3.05, "pinky": 2.7},
        thumb_closed=0.31,
        thumb_open=0.86,
        tap_ref={"index": 0.4, "middle": 0.35, "ring": 0.3, "pinky": 0.28},
        angle_sign=-1.0,
    )


def test_default_reasonable():
    for name in FINGERS:
        assert DEFAULT.open_ref[name] == pytest.approx(0.2)
        assert DEFAULT.fist_ref[name] == pytest.approx(3.0)
    assert 0.0 < DEFAULT.thumb_closed < DEFAULT.thumb_open < 1.5
    assert DEFAULT.tap_ref is None and DEFAULT.angle_sign == 1.0


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "a" / "b" / "calibration.json"
    c = _calib()
    save(c, path)
    assert path.is_file()
    assert load(path) == c


def test_save_load_roundtrip_without_tap_ref(tmp_path):
    path = tmp_path / "calibration.json"
    save(DEFAULT, path)
    assert load(path) == DEFAULT


def test_save_overwrites(tmp_path):
    path = tmp_path / "calibration.json"
    save(DEFAULT, path)
    save(_calib(), path)
    assert load(path) == _calib()
    assert [p.name for p in tmp_path.iterdir()] == ["calibration.json"]


def test_load_missing_returns_none(tmp_path):
    assert load(tmp_path / "yok.json") is None


def test_tilde_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    rel = hand_calib.Path("~") / ".tkt" / "calibration.json"
    save(_calib(), rel)
    assert (tmp_path / ".tkt" / "calibration.json").is_file()
    assert load(rel) == _calib()


def test_load_without_angle_sign_uses_default(tmp_path):
    path = tmp_path / "c.json"
    d = {
        "open_ref": dict.fromkeys(FINGERS, 0.2),
        "fist_ref": dict.fromkeys(FINGERS, 3.0),
        "thumb_closed": 0.35,
        "thumb_open": 0.8,
        "tap_ref": None,
    }
    path.write_text(json.dumps(d), encoding="utf-8")
    c = load(path)
    assert c is not None and c.angle_sign == 1.0


def test_load_corrupt_raises(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{bozuk", encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)
    path.write_text(json.dumps({"open_ref": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)
