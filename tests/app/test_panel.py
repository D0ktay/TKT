"""Kontrol penceresinin içeriği ve uygulama tercihleri (pencere olmadan)."""
import json
import types

from tkt.app import auto_mode as am
from tkt.app import panel
from tkt.app.config import load_control_config
from tkt.app.hand_gate import save_ranges


def test_prefs_roundtrip_and_defaults(tmp_path):
    path = tmp_path / "app.json"
    assert panel.load_prefs(path) == panel.DEFAULT_PREFS
    panel.save_prefs({"use_surface": True, "hud": False, "x": 1}, path)
    assert panel.load_prefs(path) == {"use_surface": True, "hud": False}
    assert json.loads(path.read_text()) == {"use_surface": True, "hud": False}
    path.write_text("{bozuk")
    assert panel.load_prefs(path) == panel.DEFAULT_PREFS


def test_file_status(tmp_path):
    assert panel.file_status(tmp_path / "yok.json") == (False, "yok")
    f = tmp_path / "a.json"
    f.write_text("{}")
    done, text = panel.file_status(f)
    assert done and text.startswith("kayıtlı (")
    assert panel.file_status(f, lambda p: False) == (False, "bozuk")


def test_calibrations_list_and_tools(tmp_path):
    cfg = load_control_config()
    cfg["hand_gate"] = {**cfg["hand_gate"], "file": str(tmp_path / "hand-gate.json")}
    zoom = types.SimpleNamespace(baseline=None, calibrating=False)
    items = panel.calibrations(am.REPO, cfg, zoom)
    assert [c.key for c in items] == ["surface", "hand", "pinch", "sign", "distance"]
    by = {c.key: c for c in items}
    assert by["surface"].tool == ("-m", "tkt.vision.debug_view", "--config",
                                  "config/vision.toml", "--start", "surface")
    assert by["pinch"].tool[-3:] == ("config/vision-front.toml", "--start", "pinch")
    assert by["sign"].tool == ("-m", "tkt.app.sign_calib")
    assert by["sign"].status == "yok" and not by["sign"].done
    assert by["distance"].tool is None and by["distance"].status == "yok"
    assert {c.group for c in items} == {"Masa modu", "Kamera sana bakarken"}

    save_ranges(tmp_path / "hand-gate.json", {"index": (0.0, 0.2), "middle": (0.0, 0.2),
                                              "ring": (0.5, 1.0), "pinky": (0.5, 1.0)})
    zoom.baseline = 0.2
    by = {c.key: c for c in panel.calibrations(am.REPO, cfg, zoom)}
    assert by["sign"].done and by["distance"].status == "kayıtlı"
    (tmp_path / "hand-gate.json").write_text("{bozuk")
    by = {c.key: c for c in panel.calibrations(am.REPO, cfg, None)}
    assert by["sign"].status == "bozuk" and "distance" not in by


def test_start_keys_exist_in_debug_view():
    from tkt.vision import debug_view
    starts = {c.tool[-1] for c in panel.calibrations(am.REPO, load_control_config())
              if c.tool and "--start" in c.tool}
    assert starts <= set(debug_view.START_KEYS)


def test_status_line():
    gate = types.SimpleNamespace(enabled=True, active=False)
    zoom = types.SimpleNamespace(enabled=True, level=2)
    assert panel.status_line("", gate, zoom) == "Kamera bekleniyor"
    assert panel.status_line("A", gate, zoom) == "Masa modu"
    assert panel.status_line("C", gate, zoom) == (
        "Kamera sana bakıyor · el faresi kapalı · büyütme +2")
    gate.active = True
    zoom.enabled = False
    assert panel.status_line("C", gate, zoom) == "Kamera sana bakıyor · el faresi açık"
    assert panel.status_line("C", gate, zoom, tool_running=True) == "Kalibrasyon penceresi açık"
    assert "Erişilebilirlik" in panel.status_line("C", gate, zoom, trusted=False)


def test_permissions_line_marks_each_feature():
    line = panel.permissions_line({"camera": True, "pointer": True, "hotkeys": False,
                                   "zoom": True, "dictation": None})
    assert line == "kamera ✓   imleç ve tık ✓   kısayollar ✗   büyütme ✓   dikte ?"
    assert panel.permissions_line({}).count("?") == 5
