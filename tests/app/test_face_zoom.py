"""Yüz mesafesiyle yakınlaştırma: karar mantığı (kamera ve menü olmadan)."""
import json

import pytest

from tkt.app.config import load_control_config
from tkt.app.face_zoom import FaceZoom, parse_hotkey, pick_zoom_items, skip_menu

BASE = 0.20         # normal mesafede yüz boyutu


def cfg(**over):
    return {**load_control_config()["face_zoom"], **over}


def make(tmp_path, baseline=BASE, enabled=True, **over):
    path = tmp_path / "face-zoom.json"
    if baseline is not None or enabled:
        path.write_text(json.dumps({"baseline": baseline, "enabled": enabled}))
    return FaceZoom(cfg(**over), path)


def feed(z, t0, seconds, ratio, period=0.2):
    """`ratio` uzaklık oranında `seconds` boyunca ölçüm verir; (son zaman, mesajlar) döner."""
    msgs, t = [], t0
    for i in range(round(seconds / period)):
        t = t0 + (i + 1) * period
        m = z.update(t, BASE / ratio)
        if m:
            msgs.append(m)
    return t, msgs


def test_defaults_are_sane():
    c = cfg()
    assert all(v > 1 for v in c["far"]) and c["near"] == []     # küçültme varsayılan kapalı
    assert c["far"] == sorted(c["far"])
    # Komşu sınırların ölü bantları üst üste binmemeli.
    bounds = sorted(c["near"]) + c["far"]
    assert all(b - a > c["hysteresis"] for a, b in zip(bounds, bounds[1:], strict=False))


def test_starts_off_without_file(tmp_path):
    z = make(tmp_path, baseline=None, enabled=False)
    assert not z.enabled and z.baseline is None and z.target == 0
    assert z.update(1.0, BASE) is None


def test_stays_at_zero_at_normal_distance(tmp_path):
    z = make(tmp_path)
    _, msgs = feed(z, 0.0, 10.0, 1.0)
    assert z.level == 0 and msgs == []


def test_leaning_back_zooms_in_after_dwell(tmp_path):
    z = make(tmp_path)
    t, msgs = feed(z, 0.0, 0.8, 1.30)       # henüz medyan + bekleme süresi dolmadı
    assert z.level == 0 and msgs == []
    t, msgs = feed(z, t, 3.0, 1.30)
    assert z.level == 1 and len(msgs) == 1 and "+1" in msgs[0]
    t, msgs = feed(z, t, 4.0, 1.60)
    assert z.level == 2 and z.target == 2


def test_leaning_in_does_nothing_by_default(tmp_path):
    z = make(tmp_path)
    _, msgs = feed(z, 0.0, 6.0, 0.60)
    assert z.level == 0 and msgs == []
    t, _ = feed(z, 6.0, 4.0, 1.60)           # geriye yaslanınca yine büyür
    assert z.level == 2
    feed(z, t, 4.0, 0.60)                    # öne eğilince normale döner, altına inmez
    assert z.level == 0


def test_leaning_in_zooms_out_when_near_is_set(tmp_path):
    z = make(tmp_path, near=[0.80])
    feed(z, 0.0, 4.0, 0.70)
    assert z.level == -1


def test_hysteresis_keeps_level_near_boundary(tmp_path):
    z = make(tmp_path)
    t, _ = feed(z, 0.0, 4.0, 1.30)
    assert z.level == 1
    # Sınır 1,20, bant ±0,04: 1,17 hâlâ +1'de kalır; 1,22 de 0'dan +1'e çıkarmaz.
    t, msgs = feed(z, t, 5.0, 1.17)
    assert z.level == 1 and msgs == []
    t, _ = feed(z, t, 4.0, 1.10)
    assert z.level == 0
    t, msgs = feed(z, t, 5.0, 1.22)
    assert z.level == 0 and msgs == []


def test_jitter_around_boundary_does_not_flip(tmp_path):
    z = make(tmp_path)
    msgs, t = [], 0.0
    for i in range(100):                     # 20 sn, sınırın iki yanında ±%3 titreme
        t += 0.2
        m = z.update(t, BASE / (1.20 + (0.03 if i % 2 else -0.03)))
        if m:
            msgs.append(m)
    assert msgs == [] and z.level == 0


def test_short_excursion_is_ignored(tmp_path):
    z = make(tmp_path)
    t, _ = feed(z, 0.0, 3.0, 1.0)
    t, msgs = feed(z, t, 0.6, 1.60)          # kısa süre geri yaslanıp döndü
    t, more = feed(z, t, 3.0, 1.0)
    assert z.level == 0 and msgs + more == []


def test_lost_face_holds_level_and_restarts_dwell(tmp_path):
    z = make(tmp_path)
    t, _ = feed(z, 0.0, 4.0, 1.30)
    assert z.level == 1
    for i in range(20):
        z.update(t + 0.2 * (i + 1), None)
    assert z.level == 1 and z.target == 1


def test_toggle_off_resets_target_and_persists(tmp_path):
    z = make(tmp_path)
    feed(z, 0.0, 4.0, 1.30)
    assert z.toggle() is False
    assert z.target == 0 and z.level == 0
    again = FaceZoom(cfg(), tmp_path / "face-zoom.json")
    assert again.enabled is False and again.baseline == BASE


def test_first_enable_calibrates_current_distance(tmp_path):
    z = make(tmp_path, baseline=None, enabled=False)
    assert z.toggle() is True
    assert z.calibrating and z.target == 0
    msgs, t = [], 0.0
    while z.calibrating:
        t += 0.2
        m = z.update(t, 0.30)
        if m:
            msgs.append(m)
        assert t < 10
    assert z.baseline == pytest.approx(0.30) and len(msgs) == 1
    assert t >= cfg()["calibrate_ms"] / 1000
    saved = json.loads((tmp_path / "face-zoom.json").read_text())
    assert saved == {"baseline": pytest.approx(0.30), "enabled": True}


def test_recalibrate_resets_level_and_waits_for_face(tmp_path):
    z = make(tmp_path)
    t, _ = feed(z, 0.0, 4.0, 1.30)
    assert z.level == 1
    z.calibrate()
    assert z.target == 0 and z.calibrating
    for i in range(30):                      # yüz yokken kalibrasyon bitmez
        z.update(t + 0.2 * (i + 1), None)
    assert z.calibrating
    t += 10.0
    while z.calibrating:
        t += 0.2
        z.update(t, BASE / 1.30)
    assert z.baseline == pytest.approx(BASE / 1.30)
    t, _ = feed(z, t, 4.0, 1.30)             # yeni normal burası: seviye 0
    assert z.level == 0


def test_corrupt_file_starts_off(tmp_path):
    path = tmp_path / "face-zoom.json"
    path.write_text("{bozuk")
    z = FaceZoom(cfg(), path)
    assert not z.enabled and z.baseline is None


def test_bad_thresholds_rejected(tmp_path):
    with pytest.raises(ValueError):
        FaceZoom(cfg(far=[0.9]), tmp_path / "x.json")


def test_parse_hotkey():
    assert parse_hotkey("<ctrl>+<alt>+z") == (frozenset({"ctrl", "alt"}), "z")
    assert parse_hotkey(cfg()["toggle_hotkey"]) == (frozenset({"ctrl"}), "m")
    assert parse_hotkey(cfg()["calibrate_hotkey"]) == (frozenset({"ctrl", "shift"}), "m")
    for bad in ("<ctrl>+", "<hyper>+z", "<ctrl>+zz"):
        with pytest.raises(ValueError):
            parse_hotkey(bad)


def test_pick_zoom_items_prefers_zoom_over_text_size():
    c = cfg()
    menus = [
        [("Bigger", "font+"), ("Smaller", "font-")],
        [("Actual Size", "0"), ("‎Zoom In", "in"), ("zoom out", "out")],
    ]
    assert pick_zoom_items(menus, c["zoom_in_titles"], c["zoom_out_titles"]) == ("in", "out")
    assert pick_zoom_items(menus[:1], c["zoom_in_titles"], c["zoom_out_titles"]) == (
        "font+", "font-")


def test_pick_zoom_items_needs_pair_in_one_menu():
    c = cfg()
    menus = [[("Zoom In", "in")], [("Zoom Out", "out")], [("Zoom", "window")]]
    assert pick_zoom_items(menus, c["zoom_in_titles"], c["zoom_out_titles"]) is None


def test_skip_menu():
    c = cfg()
    assert skip_menu("Format", c["skip_menus"]) and skip_menu(" biçim", c["skip_menus"])
    assert not skip_menu("View", c["skip_menus"]) and not skip_menu(None, c["skip_menus"])
