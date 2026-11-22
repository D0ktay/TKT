"""Otomatik modun platformdan bağımsız çekirdeği: kamera, ekran ve gerçek imleç olmadan.

Gerçek backend yerine DryBackend, kamera yerine sahte kaynak, jest motoru yerine olay üreten
sahte motor kullanılır. macOS arayüzü (`auto_mode_mac`) burada import edilmez.
"""
import dataclasses
import threading

import numpy as np
import pytest

from tkt.app import auto_mode as am
from tkt.control.gestures import Mode
from tkt.control.os_input.dry_backend import DryBackend
from tkt.core.types import FINGERS, Finger, HandFrame, SourceInfo

INFO = SourceInfo(schema=2, width=640, height=480, fps=30.0, hand="right")


def frame(t, present=True):
    if not present:
        return HandFrame(t, int(t * 100), False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {f: Finger((0.5, 0.5), None, 0.3, None) for f in FINGERS}
    return HandFrame(t, int(t * 100), True, fingers, 0.9, (0.5, 0.5), 0.2, 0.97)


class FakeSource:
    info = INFO
    last_process_ms = 5.0

    def __init__(self, n, present=True):
        self.n = n
        self.present = present

    def frames_raw(self):
        img = np.zeros((48, 64, 3), np.uint8)
        for i in range(self.n):
            yield frame(1.0 + i / 30, self.present), None, img

    def process_fps(self):
        return 30.0


class FakeEngine:
    """Her karede sıradaki olay listesini backend'e yazar; release'de dikteyi kapatır."""

    script: list = []

    def __init__(self, cfg, backend, info):
        self.backend = backend
        self.mode = Mode.NONE
        self.left_down = False
        self.i = 0
        self.dictating = False

    def process(self, f):
        events = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        for kind, args in events:
            getattr(self.backend, kind)(*args)
            if kind == "system_action" and args[0] == "dictation_start":
                self.dictating = True

    def release(self):
        if self.dictating:
            self.backend.system_action("dictation_stop")
        self.backend.release_all()


class Recorder(DryBackend):
    def __init__(self):
        super().__init__()
        self.released = 0

    def release_all(self):
        self.released += 1
        super().release_all()


@pytest.fixture
def core(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(am, "backend", rec)
    monkeypatch.setattr(am, "target_click", None)
    monkeypatch.setattr(am, "selected", None)
    monkeypatch.setattr(am, "stop", threading.Event())
    monkeypatch.setattr(am, "hud_img", [None, 0])
    am.history.clear()
    return rec


def no_face(_img):
    return 0.0


def test_profile_a_and_c_use_their_own_settings(core, monkeypatch):
    vcfg, cfg, mirror, assist = am.profile("A")
    assert vcfg.endswith("vision.toml") and cfg["general"]["click_method"] == "thumb"
    assert not mirror and not assist
    vcfg, cfg, mirror, assist = am.profile("C")
    assert vcfg.endswith("vision-front.toml") and cfg["general"]["click_method"] == "index_pinch"
    for k, v in cfg["pointer_front"].items():
        assert cfg["pointer"][k] == v
    assert mirror and not assist                  # hedef bulucu yokken hedef yardımı kapalı
    monkeypatch.setattr(am, "target_click", lambda out, rect: None)
    assert am.profile("C")[3] is True


def test_camera_rotated_reads_config(tmp_path):
    p = tmp_path / "v.toml"
    p.write_text("[camera]\nrotate_180 = false\n")
    assert am.camera_rotated(p) is False
    p.write_text("[camera]\nindex = 0\n")
    assert am.camera_rotated(p) is True


def test_load_surface_missing_file_is_none(tmp_path):
    p = tmp_path / "v.toml"
    p.write_text(f'[surface]\nfile = "{(tmp_path / "yok.json").as_posix()}"\n')
    assert am.load_surface(p) is None


def test_face_votes_switch_mode_and_release(core, monkeypatch):
    monkeypatch.setattr(am, "FACE_EVERY_S", 0.0)
    _, cfg, *_ = am.profile("A")
    got = am.run_mode(FakeSource(10), cfg, False, False, lambda img: 0.95, "A")
    assert got == "C"
    assert core.released >= 1


class FakeZoom:
    period = 0.0
    wants_samples = True
    enabled = True
    level = 0

    def __init__(self):
        self.sizes = []

    def update(self, t, size):
        self.sizes.append(size)
        return "YAKINLASTIRMA seviye +1" if len(self.sizes) == 2 else None

    def text(self):
        return "YAKINLASTIRMA: seviye +0"


def test_zoom_gets_face_size_only_in_c(core, monkeypatch):
    cfg = am.profile("C")[1]
    cfg["face_zoom"]["face_score"] = 0.7
    faces = iter([(0.9, 0.25), (0.9, 0.24), (0.5, None), (0.9, 0.26)])
    z = FakeZoom()
    monkeypatch.setattr(am, "zoom", z)
    monkeypatch.setattr(am, "zoom_txt", [""])
    am.history.clear()
    assert am.run_mode(FakeSource(4), cfg, True, False, lambda img: next(faces), "C") is None
    assert z.sizes == [0.25, 0.24, None, 0.26]
    assert any("YAKINLASTIRMA seviye +1" in msg for _, msg in am.history)
    assert am.zoom_txt[0] == "YAKINLASTIRMA: seviye +0"

    z2 = FakeZoom()
    monkeypatch.setattr(am, "zoom", z2)
    monkeypatch.setattr(am, "FACE_EVERY_S", 99.0)
    am.run_mode(FakeSource(4), am.profile("A")[1], False, False, lambda img: (0.1, None), "A")
    assert z2.sizes == []                    # A'da kamera masaya bakar: mesafe ölçülmez
    assert "A modunda" in am.zoom_txt[0]


def test_face_measure_uses_largest_confident_face():
    best, size = am.face_measure([(0.95, 64, 64), (0.8, 160, 90), (0.4, 600, 400)], 640, 360)
    assert best == 0.95
    assert size == pytest.approx((160 * 90 / (640 * 360)) ** 0.5)
    assert am.face_measure([(0.4, 100, 100)], 640, 360) == (0.4, None)
    assert am.face_measure([], 640, 360) == (0.0, None)


def test_face_finder_keeps_far_face_by_searching_around_it():
    """Uzak (küçük) yüz bütün görüntüde düşük güven alır, çevresi kırpılınca yüksek."""
    face = [300, 100, 60, 60]                 # x, y, genişlik, yükseklik (640x360 görüntüde)
    calls = []

    def detect(img):
        h, w = img.shape[:2]
        calls.append((w, h))
        x0, y0 = (0, 0) if w == 640 else roi_origin[0]
        frac = face[2] / w                    # yüz, aranan görüntünün ne kadarı
        return [(0.95 if frac >= 0.08 else 0.45, face[0] - x0, face[1] - y0, face[2], face[3])]

    img = np.zeros((360, 640, 3), np.uint8)
    roi_origin = [(0, 0)]
    find = am.FaceFinder(detect)
    sc, size = find(img)                      # yakın: bütün görüntüde bulunur
    assert sc == 0.95 and size == pytest.approx((60 * 60 / (640 * 360)) ** 0.5)
    face[:] = [310, 110, 40, 40]              # geriye yaslandı: 40/640 < 0.08
    roi_origin[0] = (240, 40)                 # önceki yüzün çevresi: merkez (330,130) ± 90
    calls.clear()
    sc, size = find(img)
    assert calls == [(180, 180)]              # yalnız çevrede arandı
    assert sc == 0.95 and size == pytest.approx((40 * 40 / (640 * 360)) ** 0.5)
    assert find.last == (sc, size)


def test_face_finder_falls_back_to_full_image_and_forgets_lost_face():
    seen = []

    def detect(img):
        seen.append(img.shape[1])
        return [(0.9, 10, 10, 50, 50)] if len(seen) == 1 else []

    img = np.zeros((360, 640, 3), np.uint8)
    find = am.FaceFinder(detect)
    assert find(img)[0] == 0.9
    assert find(img) == (0.0, None)           # çevrede yok -> bütün görüntü -> yine yok
    assert seen == [640, 110, 640]            # çevre (merkez 35 ± 75) kenarda kırpılır
    find(img)
    assert seen[-1] == 640 and len(seen) == 4  # yüz unutuldu: doğrudan bütün görüntü


class FakeGate:
    """3. karede açılır, 6. karede kapanır."""
    enabled = True

    def __init__(self):
        self.active = False
        self.n = 0

    def update(self, f):
        self.n += 1
        if self.n in (3, 6):
            self.active = not self.active
            return self.active
        return None


def test_hand_gate_blocks_engine_until_opened_and_releases_on_close(core, monkeypatch):
    cfg = am.profile("C")[1]
    monkeypatch.setattr(am, "FACE_EVERY_S", 99.0)
    monkeypatch.setattr(am, "GestureEngine", FakeEngine)
    monkeypatch.setattr(am, "gate", FakeGate())
    monkeypatch.setattr(am, "mode_txt", ["MOD C"])
    # Motor her açılışta baştan kurulur: ilk karesinde tuşa basar, sonra imleci oynatır.
    FakeEngine.script = [[("button", ("left", True))], [("move_rel", (5, 0))],
                         [("move_rel", (5, 0))], [("move_rel", (5, 0))]]
    am.history.clear()
    am.run_mode(FakeSource(8), cfg, True, False, lambda img: 0.9, "C")
    kinds = [(e.kind, e.args) for e in core.events]
    # Kareler 1-2 kapalı (olay yok); 3-5 açık (bas, 2 hareket); 6'da kapanır: tuş bırakılır.
    assert kinds == [("button", ("left", True)), ("move", (-5, 0)), ("move", (-5, 0)),
                     ("button", ("left", False))]
    msgs = [m for _, m in am.history]
    assert msgs.index("EL FARESI ACIK") < msgs.index("EL FARESI KAPALI")
    assert am.mode_txt[0] == f"MOD C - {am.GATE_OFF_TXT}"

    core.events.clear()
    monkeypatch.setattr(am, "gate", FakeGate())      # A'da kapı yok: el hep açık
    FakeEngine.script = [[("move_rel", (1, 0))]] * 4
    am.run_mode(FakeSource(2), am.profile("A")[1], False, False, lambda img: 0.1, "A")
    assert [e.kind for e in core.events] == ["move", "move"]


def test_make_gate_reads_config():
    g = am.make_gate()
    assert g.enabled and not g.active


def test_make_zoom_reads_config(monkeypatch, tmp_path):
    monkeypatch.setattr(am, "zoom_config", lambda: {
        **am.tomllib.loads((am.REPO / "config/control.toml").read_text())["face_zoom"],
        "file": str(tmp_path / "z.json")})
    z = am.make_zoom()
    assert not z.enabled and z.period == pytest.approx(0.2)


def test_face_score_flickers_do_not_switch(core, monkeypatch):
    monkeypatch.setattr(am, "FACE_EVERY_S", 0.0)
    scores = iter([0.95, 0.1, 0.95, 0.1, 0.95, 0.1, 0.95, 0.1])
    _, cfg, *_ = am.profile("A")
    assert am.run_mode(FakeSource(8), cfg, False, False, lambda img: next(scores), "A") is None


def test_events_are_relayed_mirrored_and_logged(core, monkeypatch):
    FakeEngine.script = [
        [("move_rel", (5, 3))],
        [("button", ("left", True)), ("button", ("left", False))],   # aynı karede: tık
        [("scroll", (2.0, -10.0))],
        [("system_action", ("dictation_start",))],
    ]
    monkeypatch.setattr(am, "GestureEngine", FakeEngine)
    _, cfg, *_ = am.profile("C")
    am.run_mode(FakeSource(6), cfg, True, False, no_face, "C")
    kinds = [(e.kind, e.args) for e in core.events]
    assert ("move", (-5, 3)) in kinds                        # C: yatay ayna
    assert ("button", ("left", True)) in kinds and ("button", ("left", False)) in kinds
    assert ("scroll", (-2.0, -10.0)) in kinds
    assert ("system", ("dictation_start",)) in kinds
    assert ("system", ("dictation_stop",)) in kinds          # çıkışta açık dikte kapatıldı
    texts = [m for _, m in am.history]
    assert "SOL TIK" in texts and "DIKTE KAPANDI" in texts
    assert am.hud_img[0] is not None and am.hud_img[0].shape[0] == am.HUD_H


def test_target_click_used_when_same_target_locked(core, monkeypatch):
    clicks = []
    monkeypatch.setattr(am, "target_click", lambda out, rect: clicks.append(rect))
    monkeypatch.setattr(am, "selected", ((10, 20, 30, 40), "el"))
    FakeEngine.script = [[], [("button", ("left", True)), ("button", ("left", False))]]
    monkeypatch.setattr(am, "GestureEngine", FakeEngine)

    class PinchSource(FakeSource):
        def frames_raw(self):
            img = np.zeros((48, 64, 3), np.uint8)
            for i in range(self.n):       # işaret pinch'i hep kapalı (0 = uçlar değiyor)
                f = dataclasses.replace(frame(1.0 + i / 30), pinch=dict.fromkeys(FINGERS, 0.0))
                yield f, None, img

    _, cfg, *_ = am.profile("C")
    am.run_mode(PinchSource(3), cfg, True, True, no_face, "C")
    assert clicks == [(10, 20, 30, 40)]
    assert "SOL TIK (hedef)" in [m for _, m in am.history]


def test_make_backend_picks_platform(monkeypatch):
    from tkt.control.os_input import windows

    made = {}

    class FakeWin:
        def __init__(self, step_px, smooth):
            made.update(step_px=step_px, smooth=smooth)

    monkeypatch.setattr(windows, "WindowsBackend", FakeWin)
    assert isinstance(am.make_backend("win32"), FakeWin)
    assert made["smooth"] is True and made["step_px"] > 0


def test_cv_ui_shows_hud_and_quits_on_esc(core):
    shown = []

    def worker():
        am.hud_img[:] = [np.zeros((4, 4, 3), np.uint8), 1]
        am.stop.wait(2.0)

    keys = iter([255, 255, 27])
    am.run_cv_ui(show=lambda name, img: shown.append(name), poll_key=lambda: next(keys),
                 window_open=lambda: True, worker_target=worker)
    assert shown and shown[0] == am.HUD_WINDOW
    assert am.stop.is_set() and core.released >= 1


def test_cv_ui_quits_when_window_closed(core):
    def worker():
        am.hud_img[:] = [np.zeros((4, 4, 3), np.uint8), 1]
        am.stop.wait(2.0)

    am.run_cv_ui(show=lambda name, img: None, poll_key=lambda: 255,
                 window_open=lambda: False, worker_target=worker)
    assert am.stop.is_set()
