"""İşaret kalibrasyonu: kayıt, aralık çıkarma ve akış (kamera ve pencere olmadan)."""
import numpy as np

from tkt.app import sign_calib as sc
from tkt.app.config import load_control_config
from tkt.app.hand_gate import in_ranges, load_ranges
from tkt.core.types import FINGERS, Finger, HandFrame

CFG = load_control_config()["hand_gate"]
SIGN = {"index": 0.03, "middle": 0.02, "ring": 0.45, "pinky": 0.75}
OPEN = {"index": 0.01, "middle": 0.03, "ring": 0.18, "pinky": 0.08}    # canlıda yanlış kaydedilen
FIST = {"index": 0.90, "middle": 0.95, "ring": 0.90, "pinky": 0.90}


def frame(t, curls):
    if curls is None:
        return HandFrame(t, int(t * 100), False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {n: Finger((0.5, 0.5), None, curls[n], None) for n in FINGERS}
    return HandFrame(t, int(t * 100), True, fingers, 0.5, (0.5, 0.5), 0.2, 0.9)


def run(cal, t, seconds, curls, fps=30):
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        cal.update(frame(t, curls))
    return t


def test_recorder_restarts_while_hand_moves():
    rec = sc.PoseRecorder(1.0, 0.10, 0.0)
    t = 0.0
    for i in range(60):                               # 2 sn boyunca parmak sürekli kıvrılıyor
        t += 1 / 30
        rec.update(t, {**SIGN, "ring": i / 60})
    assert not rec.done and rec.progress < 0.3
    for _ in range(40):                               # sabitlenince kayıt biter
        t += 1 / 30
        rec.update(t, SIGN)
    assert rec.done and all(s == SIGN for s in rec.samples)   # geçiş kareleri kayıtta yok
    assert rec.update(t + 0.1, None) == 0.0           # el kaybolursa baştan


def test_fit_accepts_sign_and_rejects_open_hand_and_fist():
    ranges = sc.fit_sign([SIGN] * 30, CFG["teach_margin"], CFG["teach_min_gap"])
    assert in_ranges(ranges, SIGN)
    assert not in_ranges(ranges, OPEN) and not in_ranges(ranges, FIST)
    assert sc.fit_sign([OPEN] * 30, CFG["teach_margin"], CFG["teach_min_gap"]) is None
    assert sc.fit_sign([FIST] * 30, CFG["teach_margin"], CFG["teach_min_gap"]) is None


def test_nothing_is_recorded_until_started(tmp_path):
    cal = sc.SignCalibration(CFG, 0.5)
    t = run(cal, 0.0, 1.0, None)
    assert cal.state == "ready" and "gorunmuyor" in cal.instruction
    assert cal.start() is False                       # el yokken boşluk bir şey yapmaz
    t = run(cal, t, 5.0, OPEN)                        # el açık bekliyor: kayıt yok
    assert cal.state == "ready" and cal.samples == [] and "BOSLUK" in cal.instruction

    t = run(cal, t, 0.5, SIGN)
    assert cal.start() is True and cal.state == "record"
    t = run(cal, t, 1.0, SIGN)
    assert cal.state == "record" and 0.3 < cal.progress < 0.7
    t = run(cal, t, 1.5, SIGN)
    assert cal.state == "test" and in_ranges(cal.ranges, SIGN)

    t = run(cal, t, 1.0, OPEN)                        # işaret hâlâ tutuluyor sayılmasın
    assert not cal.recognized and cal.hits == 0
    t = run(cal, t, 1.5, SIGN)
    assert cal.recognized and cal.hits == 1
    t = run(cal, t, 0.5, OPEN)
    run(cal, t, 1.5, SIGN)
    assert cal.hits == 2
    cal.save(tmp_path / "hand-gate.json")
    assert in_ranges(load_ranges(tmp_path / "hand-gate.json"), SIGN)


def test_open_hand_is_rejected_and_restart_clears():
    cal = sc.SignCalibration(CFG, 0.5)
    t = run(cal, 0.0, 0.5, OPEN)
    cal.start()
    run(cal, t, 3.0, OPEN)
    assert cal.state == "problem" and cal.ranges is None and "benzemiyor" in cal.instruction
    cal.restart()
    assert cal.state == "ready" and cal.samples == []


def test_render_every_state_without_camera():
    cal = sc.SignCalibration(CFG, 0.5)
    img = np.zeros((1080, 1920, 3), np.uint8)
    shape = sc.render(img, None, cal).shape
    assert shape == (sc.VIEW_H, 960 + sc.PANEL_W, 3)
    t = run(cal, 0.0, 0.3, SIGN)
    assert sc.render(img, None, cal).shape == shape           # hazır
    cal.start()
    t = run(cal, t, 0.5, SIGN)
    assert sc.render(img, None, cal).shape == shape           # kayıt
    run(cal, t, 2.5, SIGN)
    assert cal.state == "test"
    assert sc.render(img, None, cal).shape == shape           # deneme, yeşil bantlarla
    assert sc.render(img, None, cal, saved=True).shape == shape
