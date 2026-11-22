from dataclasses import replace

import pytest
from tests.control.conftest import make_frame

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine, Mode
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.sources import SyntheticSource
from tkt.control.surface_pointer import SurfaceMotion
from tkt.core.types import FINGERS, SourceInfo

FPS = 60.0
INFO = SourceInfo(schema=1, width=640, height=512, fps=FPS, hand="right")


def surface_cfg(**over):
    cfg = dict(load_control_config()["surface_pointer"])
    cfg.update(over)
    return cfg


def motion(**over):
    return SurfaceMotion(surface_cfg(**over), load_control_config()["pointer"])


def run_path(m, points, t0=0.0):
    """Noktaları 60 fps ile verir; tüm hareketlerin toplamı ve kare başına hareketler."""
    per_frame = []
    for i, uv in enumerate(points):
        per_frame.append(m.update(uv, t0 + i / FPS))
    total = (sum(d[0] for ms in per_frame for d in ms), sum(d[1] for ms in per_frame for d in ms))
    return total, per_frame


def line(start, end, n):
    return [(start[0] + (end[0] - start[0]) * k / (n - 1),
             start[1] + (end[1] - start[1]) * k / (n - 1)) for k in range(n)]


# ---------------------------------------------------------------- SurfaceMotion


def test_repo_config_is_off_and_matches_vision_aspect():
    cfg = load_control_config()["surface_pointer"]
    assert cfg["enabled"] is False and cfg["absolute"] is False
    assert cfg["aspect"] == 1.5


def test_inside_motion_moves_cursor_in_direction():
    (dx, dy), _ = run_path(motion(), line((0.3, 0.5), (0.7, 0.5), 30) + [(0.7, 0.5)] * 30)
    assert dx > 100 and dy == 0
    (dx, dy), _ = run_path(motion(), line((0.5, 0.8), (0.5, 0.2), 30) + [(0.5, 0.2)] * 30)
    assert dy < -100 and abs(dx) <= 1     # ileri (v azalır) -> imleç yukarı


def test_outside_produces_no_motion():
    (dx, dy), frames = run_path(motion(), line((-0.5, 0.5), (-0.1, 0.5), 30))
    assert (dx, dy) == (0, 0) and all(ms == [] for ms in frames)
    (dx, dy), _ = run_path(motion(), line((0.5, 1.2), (0.5, 1.6), 30))
    assert (dx, dy) == (0, 0)


def test_edge_margin_counts_as_inside():
    m = motion(edge_margin=0.03)
    assert m.inside((1.02, 0.5)) and m.inside((-0.02, -0.02))
    assert not m.inside((1.05, 0.5)) and not m.inside((0.5, -0.04))
    (dx, _), _ = run_path(motion(), line((1.0, 0.5), (1.025, 0.5), 20) + [(1.025, 0.5)] * 20)
    assert dx > 0


def test_entering_does_not_jump():
    m = motion()
    outside = line((-0.6, 0.5), (-0.1, 0.5), 20)
    # Bir karede dışarıdan alan ortasına "ışınlanma" (ör. tespit sıçraması) ve orada durma.
    _, frames = run_path(m, outside + [(0.5, 0.5)] * 20)
    assert all(ms == [] for ms in frames)
    # Girişten sonra yavaş hareket: yalnız giriş sonrası hareket kadar imleç.
    m = motion()
    path = outside + line((0.5, 0.5), (0.55, 0.5), 20) + [(0.55, 0.5)] * 30
    (dx, _), frames = run_path(m, path)
    assert frames[20] == []                       # giriş karesi referans
    ref, _ = run_path(motion(), line((0.5, 0.5), (0.55, 0.5), 20) + [(0.55, 0.5)] * 30)
    assert dx == ref[0]


def test_leaving_stops_and_reentry_rebases():
    m = motion()
    path = (line((0.4, 0.5), (0.6, 0.5), 20) + line((1.2, 0.5), (1.5, 0.5), 10)
            + [(0.3, 0.5)] * 20)
    (dx, _), frames = run_path(m, path)
    assert all(ms == [] for ms in frames[20:])    # dışarıda ve geri girişte (sola) zıplama yok
    ref, _ = run_path(motion(), line((0.4, 0.5), (0.6, 0.5), 20))
    assert dx == ref[0]


def test_equal_physical_motion_far_and_near_gives_equal_cursor_motion():
    far, _ = run_path(motion(), line((0.2, 0.05), (0.6, 0.05), 25) + [(0.6, 0.05)] * 30)
    near, _ = run_path(motion(), line((0.2, 0.95), (0.6, 0.95), 25) + [(0.6, 0.95)] * 30)
    assert far == near and far[0] > 0


def test_aspect_scales_horizontal_units():
    # Aynı u farkı, aspect 1.5 iken 1.5 kat fiziksel genişlik demek: aynı yavaş hızda 1.5 kat.
    slow = dict(gain_min=1000, gain_max=1000)
    path = line((0.4, 0.5), (0.5, 0.5), 60) + [(0.5, 0.5)] * 60
    a1, _ = run_path(motion(aspect=1.0, **slow), path)
    a15, _ = run_path(motion(aspect=1.5, **slow), path)
    assert a15[0] == pytest.approx(1.5 * a1[0], abs=2)


def test_faster_motion_gets_more_gain():
    slow, _ = run_path(motion(), line((0.3, 0.5), (0.5, 0.5), 120) + [(0.5, 0.5)] * 60)
    fast, _ = run_path(motion(), line((0.3, 0.5), (0.5, 0.5), 8) + [(0.5, 0.5)] * 60)
    assert fast[0] > 1.5 * slow[0]


def test_gap_rebases_and_time_going_back_is_ignored():
    m = motion()
    assert m.update((0.5, 0.5), 0.0) == []
    assert m.update((0.6, 0.5), 1.0) == []        # boşluk > gap_reset_ms: referans
    assert m.update((0.7, 0.5), 0.5) == []        # geri giden zaman


def test_rollback_undoes_recent_moves():
    m = motion()
    _, frames = run_path(m, line((0.3, 0.5), (0.6, 0.5), 20))
    t_end = 19 / FPS
    recent = [d for i, ms in enumerate(frames) if i / FPS + 0.1 >= t_end for d in ms]
    assert recent
    undo = m.rollback_at(100, t_end)
    assert undo == (-sum(d[0] for d in recent), -sum(d[1] for d in recent))
    assert m.rollback_at(100, t_end) == (0, 0)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        motion(gain_max=100)
    with pytest.raises(ValueError):
        motion(speed_hi=0.1)
    with pytest.raises(ValueError):
        motion(aspect=0)
    with pytest.raises(ValueError):
        motion(edge_margin=-0.1)


def test_absolute_mode_homes_then_tracks_target():
    m = motion(absolute=True, screen_w=1000, screen_h=500, min_cutoff=1e6, beta=0.0)
    first = m.update((0.5, 0.5), 0.0)
    # İlk karede kenarlara itme (sola, sonra yukarı) ve hedefe gitme.
    assert first[0] == (-4000, 0) and first[1][1] == -2000
    x = sum(d[0] for d in first[1:])
    y = sum(d[1] for d in first[2:])
    assert x == round(0.5 * 999) and y == round(0.5 * 499)
    second = m.update((0.75, 0.25), 1 / FPS)
    assert second == [(round(0.75 * 999) - round(0.5 * 999), round(0.25 * 499) - round(0.5 * 499))]
    # Dışarıda hareket yok; içeri girişte mutlak hedefe gider.
    assert m.update((1.5, 0.5), 2 / FPS) == []
    third = m.update((0.0, 0.0), 3 / FPS)
    assert third == [(-round(0.75 * 999), -round(0.25 * 499))]
    # Geri alma sanal konumu da geri taşır: aynı hedefe yeniden gidilir.
    undo = m.rollback_at(10, 3 / FPS)        # yalnız son hareket
    assert undo == (round(0.75 * 999), round(0.25 * 499))
    assert m.update((0.0, 0.0), 4 / FPS) == [(-undo[0], -undo[1])]


# ---------------------------------------------------------------- GestureEngine


def engine_cfg(**surface):
    cfg = load_control_config()
    cfg["surface_pointer"] = {**cfg["surface_pointer"], **surface}
    return cfg


def events(backend):
    return [(e.kind, *e.args) for e in backend.events]


def pointer_pose(t, uv, img=(0.5, 0.4), thumb_out=1.0):
    curls = {name: 0.0 if name == "index" else 1.0 for name in FINGERS}
    surfaces = {name: uv for name in FINGERS} if uv is not None else {}
    return make_frame(t=t, curls=curls, index_pos=img, surfaces=surfaces, thumb_out=thumb_out)


def drive(engine, frames):
    for f in frames:
        engine.process(f)


def test_engine_moves_from_surface_not_image():
    backend = DryBackend()
    engine = GestureEngine(engine_cfg(enabled=True), backend, INFO)
    path = [(0.4, 0.5)] * 10 + line((0.4, 0.5), (0.6, 0.5), 20) + [(0.6, 0.5)] * 20
    # Görüntü konumu sabit: hareket yalnız alan koordinatından gelmeli.
    drive(engine, [pointer_pose(i / FPS, uv) for i, uv in enumerate(path)])
    assert engine.mode is Mode.POINTER
    moves = [e for e in events(backend) if e[0] == "move"]
    assert moves and sum(m[1] for m in moves) > 50
    assert all(m[2] == 0 for m in moves)


def test_engine_outside_area_does_not_move():
    backend = DryBackend()
    engine = GestureEngine(engine_cfg(enabled=True), backend, INFO)
    path = [(-0.3, 0.5)] * 10 + line((-0.3, 0.5), (-0.6, 0.5), 30)
    drive(engine, [pointer_pose(i / FPS, uv, img=(0.2 + i * 0.01, 0.4))
                   for i, uv in enumerate(path)])
    assert engine.mode is Mode.POINTER
    assert [e for e in events(backend) if e[0] == "move"] == []


def test_engine_falls_back_to_image_pointer_without_surface():
    cfg_on, cfg_off = engine_cfg(enabled=True), engine_cfg(enabled=False)
    frames = [pointer_pose(i / FPS, None, img=(0.3 + 0.005 * i, 0.4)) for i in range(50)]
    a, b = DryBackend(), DryBackend()
    drive(GestureEngine(cfg_on, a, INFO), frames)
    drive(GestureEngine(cfg_off, b, INFO), frames)
    assert events(a) == events(b) and any(e[0] == "move" for e in events(a))


def test_engine_thumb_click_freezes_surface_pointer():
    backend = DryBackend()
    engine = GestureEngine(engine_cfg(enabled=True), backend, INFO)
    frames, t = [], 0.0
    # İmleç modu, sağa hareket.
    for uv in [(0.3, 0.5)] * 10 + line((0.3, 0.5), (0.5, 0.5), 20):
        frames.append(pointer_pose(t, uv))
        t += 1 / FPS
    n_before = len(frames)
    # Başparmak hızla kapanırken parmak ucu kayıyor (tık kayması): imleç donmalı.
    for k, uv in enumerate(line((0.5, 0.5), (0.56, 0.5), 6)):
        frames.append(pointer_pose(t, uv, thumb_out=max(0.0, 0.8 - k * 0.2)))
        t += 1 / FPS
    for uv in [(0.56, 0.5)] * 10:
        frames.append(pointer_pose(t, uv, thumb_out=0.0))
        t += 1 / FPS
    for f in frames[:n_before]:
        engine.process(f)
    before = len(backend.events)
    for f in frames[n_before:]:
        engine.process(f)
    after = events(backend)[before:]
    assert ("button", "left", True) in after
    press = after.index(("button", "left", True))
    # Basıştan önce (donma süresinde) yalnız geri alma hareketi olabilir, ileri kayma yok.
    assert all(e[1] <= 0 for e in after[:press] if e[0] == "move")
    assert not any(e[0] == "move" and e[1] > 0 for e in after[press:])


def _scenario():
    return (SyntheticSource().rest(0.2).open(1, s=0.3).move(0.1, 0.05, s=0.4).thumb(True)
            .move(0.05, 0.0, s=0.3).thumb(False).move(-0.1, 0.0, s=0.3).tap()
            .open(2, s=0.3).move(0.0, 0.1, s=0.3).curl_all().vanish(0.5).open(1, s=0.3)
            .move(0.05, 0.05, s=0.3))


def _with_surface(frame):
    if not frame.present:
        return frame
    fingers = {n: replace(f, surface=(f.img[0], f.img[1])) for n, f in frame.fingers.items()}
    return replace(frame, fingers=fingers)


@pytest.mark.parametrize("method", ["thumb", "both"])
def test_disabled_is_identical_to_old_engine(method):
    src = _scenario()
    frames = list(src.frames())
    old_cfg = load_control_config()
    old_cfg.pop("surface_pointer")
    old_cfg["general"]["click_method"] = method
    off_cfg = engine_cfg(enabled=False)
    off_cfg["general"]["click_method"] = method
    a, b = DryBackend(), DryBackend()
    ea, eb = GestureEngine(old_cfg, a, src.info), GestureEngine(off_cfg, b, src.info)
    for f in frames:
        a.set_time(f.t)
        b.set_time(f.t)
        ea.process(f)
        eb.process(_with_surface(f))     # surface dolu olsa da kapalıyken kullanılmaz
    assert events(a) == events(b)
    assert any(e[0] == "move" for e in events(a))
