"""İşaret kalibrasyonu: kayıt, aralık çıkarma ve akış (kamera ve pencere olmadan)."""
import numpy as np
import pytest

from tkt.app import sign_calib as sc
from tkt.app.config import load_control_config
from tkt.app.hand_gate import in_ranges, load_ranges
from tkt.core.types import FINGERS, Finger, HandFrame

CFG = load_control_config()["hand_gate"]
SIGN = {"index": 0.03, "middle": 0.02, "ring": 0.45, "pinky": 0.75}
RELAXED = {"index": 0.10, "middle": 0.12, "ring": 0.15, "pinky": 0.20}
PINCH = {"index": 0.40, "middle": 0.20, "ring": 0.20, "pinky": 0.25}
FIST = {"index": 0.90, "middle": 0.95, "ring": 0.90, "pinky": 0.90}
MOVING = {"index": 0.25, "middle": 0.30, "ring": 0.33, "pinky": 0.34}


def frame(t, curls):
    if curls is None:
        return HandFrame(t, int(t * 100), False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {n: Finger((0.5, 0.5), None, curls[n], None) for n in FINGERS}
    return HandFrame(t, int(t * 100), True, fingers, 0.5, (0.5, 0.5), 0.2, 0.9)


def run(cal, t, seconds, curls, until=None, fps=30):
    """`curls` pozunu verir; `until(cal)` doğru olunca ya da süre bitince durur."""
    for _ in range(round(seconds * fps)):
        t += 1 / fps
        cal.update(frame(t, curls))
        if until is not None and until(cal):
            break
    return t


def do_pose(cal, t, curls):
    step = cal.step
    t = run(cal, t, 10.0, curls, until=lambda c: c.step != step)
    assert cal.step == step + 1
    return run(cal, t, sc.PAUSE_S + 0.2, None)        # araya el değiştirme süresi


def test_recorder_waits_for_a_still_hand():
    rec = sc.PoseRecorder(1.0, 0.10, 0.5)
    t = 0.0
    for i in range(60):                               # 2 sn boyunca parmak sürekli kıvrılıyor
        t += 1 / 30
        rec.update(t, {**SIGN, "ring": i / 60})
    assert not rec.done and rec.progress < 0.2
    for _ in range(50):                               # sabitlenince kayıt başlar ve biter
        t += 1 / 30
        rec.update(t, SIGN)
    assert rec.done and rec.progress == 1.0
    assert all(s == SIGN for s in rec.samples)        # geçiş kareleri kayıtta yok


def test_recorder_restarts_when_hand_disappears():
    rec = sc.PoseRecorder(1.0, 0.10, 0.5)
    for i in range(30):
        rec.update(i / 30, SIGN)
    assert rec.settled and 0 < rec.progress < 1
    assert rec.update(1.0, None) == 0.0 and not rec.settled


def test_fit_keeps_other_poses_out():
    ranges, problem = sc.fit_sign([SIGN] * 30, {"relaxed": [RELAXED] * 20, "pinch": [PINCH] * 20,
                                                "fist": [FIST] * 20}, 0.15, 0.20)
    assert problem is None
    assert in_ranges(ranges, SIGN)
    for other in (RELAXED, PINCH, FIST):
        assert not in_ranges(ranges, other)


def test_fit_tightens_range_when_another_pose_falls_inside():
    near = {**SIGN, "pinky": 0.50}                    # yalnız serçede 0,25 fark; pay 0,15 yetmez
    wide, _ = sc.fit_sign([SIGN] * 30, {}, 0.30, 0.20)
    assert in_ranges(wide, near)
    ranges, problem = sc.fit_sign([SIGN] * 30, {"relaxed": [near] * 20}, 0.30, 0.20)
    assert problem is None and not in_ranges(ranges, near) and in_ranges(ranges, SIGN)
    assert ranges["pinky"][0] == pytest.approx(0.625)


def test_fit_rejects_sign_too_close_to_another_pose():
    close = {n: v + 0.05 for n, v in MOVING.items()}
    assert sc.fit_sign([MOVING] * 30, {"relaxed": [close] * 20}, 0.15, 0.20) == (None, "relaxed")


def test_full_flow_records_tests_and_saves(tmp_path):
    cal = sc.SignCalibration(CFG, 0.5)
    assert cal.state == "record" and "BARIS" in cal.title
    t = run(cal, 0.0, 1.0, None)
    assert "gorunmuyor" in cal.instruction and cal.progress == 0.0
    for pose in (SIGN, RELAXED, PINCH, FIST):
        t = do_pose(cal, t, pose)
    assert cal.state == "test" and cal.problem is None
    assert set(cal.samples) == {"sign", "relaxed", "pinch", "fist"}
    t = run(cal, t, 1.0, RELAXED)
    assert not cal.recognized and cal.hits == 0
    t = run(cal, t, 1.5, SIGN)
    assert cal.recognized and cal.hits == 1
    t = run(cal, t, 0.5, RELAXED)
    t = run(cal, t, 1.5, SIGN)
    assert cal.hits == 2
    cal.save(tmp_path / "hand-gate.json")
    assert in_ranges(load_ranges(tmp_path / "hand-gate.json"), SIGN)


def test_flow_reports_problem_and_restarts():
    cal = sc.SignCalibration(CFG, 0.5)
    t = 0.0
    for pose in (SIGN, SIGN, PINCH, FIST):            # "rahat el" adımında işaret tutulmaya devam
        t = do_pose(cal, t, pose) if cal.step < 3 else run(
            cal, t, 10.0, pose, until=lambda c: c.state != "record")
    assert cal.state == "problem" and cal.problem == "relaxed"
    assert "RAHAT EL" in cal.instruction
    cal.restart()
    assert cal.state == "record" and cal.step == 0 and cal.samples == {}


def test_render_every_state_without_camera():
    cal = sc.SignCalibration(CFG, 0.5)
    img = np.zeros((1080, 1920, 3), np.uint8)
    shape = sc.render(img, None, cal).shape
    assert shape == (sc.VIEW_H, 960 + sc.PANEL_W, 3)
    t = 0.0
    for pose in (SIGN, RELAXED, PINCH, FIST):
        t = run(cal, t, 0.3, pose)
        assert sc.render(img, None, cal).shape == shape       # kayıt sürerken
        t = do_pose(cal, t, pose)
    run(cal, t, 0.5, SIGN)
    assert sc.render(img, None, cal).shape == shape           # deneme, yeşil bantlarla
    assert sc.render(img, None, cal, saved=True).shape == shape
