from types import SimpleNamespace

import numpy as np
import pytest

from tkt.app import hud
from tkt.app.hud import ActionTap, Hud, HudClosed, HudSource, status
from tkt.control.gestures import Mode
from tkt.control.os_input.dry_backend import DryBackend
from tkt.core.types import FINGERS, SCHEMA_VERSION, Finger, HandFrame, SourceInfo

INFO = SourceInfo(SCHEMA_VERSION, 64, 48, 30.0, "right", "")


def frame(present=True, confidence=1.0, index_curl=0.0, thumb=1.0):
    if not present:
        return HandFrame(0.0, 0, False, {}, 0.0, (0.0, 0.0), 0.0, 0.0)
    fingers = {f: Finger((0.5, 0.5), None, index_curl if f == "index" else 1.0, None)
               for f in FINGERS}
    return HandFrame(0.0, 0, True, fingers, thumb, (0.5, 0.5), 0.1, confidence)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def engine(mode):
    return SimpleNamespace(mode=mode)


def test_action_tap_forwards_and_labels_clicks_and_drags():
    clock, be = Clock(), DryBackend()
    tap = ActionTap(be, clock)
    tap.button("left", True)
    clock.t = 0.1
    tap.button("left", False)
    assert tap.action == "SOL TIK" and tap.action_t == 0.1
    tap.button("right", True)
    clock.t = 0.2
    tap.button("right", False)
    assert tap.action == "SAG TIK"
    tap.button("left", True)
    clock.t = 1.0
    assert tap.held(clock.t) == "SURUKLEME"
    tap.button("left", False)
    assert tap.action == "SOL TUS BIRAKILDI" and tap.held(clock.t) == ""
    tap.move_rel(1, 2)
    tap.scroll(0.0, 3.0)
    assert [e.kind for e in be.events] == ["button"] * 6 + ["move", "scroll"]
    assert tap.events_sent == len(be.events)


@pytest.mark.parametrize("fr,mode,paused,label", [
    (frame(present=False), Mode.POINTER, False, "EL YOK"),
    (frame(confidence=0.0), Mode.POINTER, False, "EL KALKIK"),
    (frame(confidence=0.3), Mode.POINTER, False, "GUVEN DUSUK"),
    (frame(), Mode.POINTER, False, "IMLEC"),
    (frame(index_curl=0.8), Mode.POINTER, False, "TASIMA (imlec durdu)"),
    (frame(), Mode.SCROLL, False, "KAYDIRMA"),
    (frame(), Mode.NONE, False, "DINLENME"),
    (frame(), Mode.POINTER, True, "DURAKLATILDI (Ctrl+Alt+P)"),
])
def test_status_labels(fr, mode, paused, label):
    tap = ActionTap(DryBackend(), Clock())
    assert status(fr, engine(mode), tap, 0.0, 0.5, paused).label == label


def test_status_shows_recent_action_then_clears_and_finger_summary():
    clock = Clock()
    tap = ActionTap(DryBackend(), clock)
    tap.button("left", True)
    tap.button("left", False)
    st = status(frame(thumb=0.0), engine(Mode.POINTER), tap, 0.5, 0.5, False)
    assert st.action == "SOL TIK"
    assert "isaret:acik" in st.fingers and "basparmak:kapali" in st.fingers
    assert status(frame(), engine(Mode.POINTER), tap, hud.ACTION_SHOW_S + 0.1,
                  0.5, False).action == ""


def test_hud_source_blank_image_without_raw_frames():
    src = SimpleNamespace(info=INFO, frames=lambda: iter([frame()]), close=lambda: None)
    hs = HudSource(src)
    assert len(list(hs.frames())) == 1
    assert hs.image.shape == (48, 64, 3) and hs.hand is None


def test_hud_draws_and_quits_on_q():
    img = np.zeros((48, 64, 3), np.uint8)
    raw = SimpleNamespace(info=INFO, frames_raw=lambda: iter([(frame(), None, img)] * 2),
                          close=lambda: None)
    hs = HudSource(raw)
    shown = []
    keys = iter([255, ord("q")])
    h = Hud(hs, ActionTap(DryBackend(), Clock()), 0.5,
            show=lambda name, im: shown.append((name, im.shape)), poll_key=lambda: next(keys))
    it = hs.frames()
    h.show(next(it), engine(Mode.POINTER), 0.0, False)
    with pytest.raises(HudClosed):
        h.show(next(it), engine(Mode.POINTER), 0.0, False)
    assert shown == [(hud.WINDOW_NAME, (48, 64, 3))] * 2
    assert img.sum() == 0                 # girdi görüntü değişmez


def test_status_dictation_shown_even_when_lifted_and_action_labels():
    clock = Clock()
    tap = ActionTap(DryBackend(), clock)
    tap.system_action("dictation_submit")
    st = status(frame(confidence=0.0), engine(Mode.DICTATE), tap, 0.1, 0.5, False)
    assert st.label.startswith("DIKTE") and st.action == "GONDER (Enter)"
