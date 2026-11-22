"""Kontrol penceresinin içeriği (platformdan bağımsız): ayarlar ve kalibrasyon listesi.

Pencerenin kendisi platform arayüzündedir (macOS: `panel_mac`). Burada yalnız neyin
gösterileceği ve kalıcı uygulama tercihleri (`~/.tkt/app.json`) vardır.

Kalibrasyonlar seçenektir: her biri bir kez yapılır ve dosyada kalır; pencere hangisinin
kayıtlı olduğunu gösterir ve istenince yalnız o yeniden yapılır.
"""
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tkt.app.hand_gate import load_ranges

PREFS_PATH = "~/.tkt/app.json"
DEFAULT_PREFS = {
    "use_surface": False,   # masa modunda imleç seçilen alandan sürülsün
    "hud": True,            # kamera penceresi görünsün
}


def load_prefs(path=PREFS_PATH):
    prefs = dict(DEFAULT_PREFS)
    try:
        data = json.loads(Path(path).expanduser().read_text())
        prefs.update({k: bool(v) for k, v in data.items() if k in DEFAULT_PREFS})
    except (OSError, ValueError, AttributeError):
        pass
    return prefs


def save_prefs(prefs, path=PREFS_PATH):
    path = Path(path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: bool(prefs[k]) for k in DEFAULT_PREFS}, indent=1))
    except OSError:
        pass


@dataclass(frozen=True)
class Calibration:
    key: str
    group: str              # "Masa modu" | "Kamera sana bakarken"
    title: str
    button: str
    done: bool
    status: str             # "kayitli (22.11 14:05)" | "yok"
    tool: tuple[str, ...] | None    # `python` argümanları; None: uygulama içinde yapılır


def file_status(path, valid=None):
    """(kayıtlı mı, durum metni). `valid(path)` verilirse dosyanın okunabildiği de sınanır."""
    path = Path(path).expanduser()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False, "yok"
    if valid is not None and not valid(path):
        return False, "bozuk"
    return True, "kayıtlı (" + time.strftime("%d.%m %H:%M", time.localtime(mtime)) + ")"


def calibrations(repo, control_cfg, zoom=None):
    """Pencerede gösterilecek kalibrasyonlar, sırasıyla."""
    repo = Path(repo)
    desk = tomllib.loads((repo / "config/vision.toml").read_text(encoding="utf-8"))
    front = tomllib.loads((repo / "config/vision-front.toml").read_text(encoding="utf-8"))
    desk_cfg, front_cfg = "config/vision.toml", "config/vision-front.toml"
    view = ("-m", "tkt.vision.debug_view", "--config")
    a, c = "Masa modu", "Kamera sana bakarken"
    rows = [
        ("surface", a, "Masa alanı", "Seç…",
         file_status(desk.get("surface", {}).get("file", "~/.tkt/surface.json")),
         (*view, desk_cfg, "--start", "surface")),
        ("hand", a, "El ölçüleri", "Kalibre et…", file_status(desk["hand"]["calibration"]),
         (*view, desk_cfg, "--start", "hand")),
        ("pinch", c, "El ve tık (pinch)", "Kalibre et…",
         file_status(front["hand"]["calibration"]), (*view, front_cfg, "--start", "pinch")),
        ("sign", c, "Barış işareti", "Öğret…",
         file_status(control_cfg["hand_gate"]["file"], lambda p: load_ranges(p) is not None),
         ("-m", "tkt.app.sign_calib")),
    ]
    out = [Calibration(k, g, t, b, st[0], st[1], tool) for k, g, t, b, st, tool in rows]
    if zoom is not None:
        done = zoom.baseline is not None
        status = "ölçülüyor…" if zoom.calibrating else "kayıtlı" if done else "yok"
        out.append(Calibration("distance", c, "Normal mesafe", "Şimdi ayarla", done, status, None))
    return out


def status_line(mode, gate, zoom, tool_running=False, trusted=True):
    """Pencerenin üstündeki tek satırlık durum."""
    if not trusted:
        return "Erişilebilirlik izni yok: imleç oynamaz"
    if tool_running:
        return "Kalibrasyon penceresi açık"
    if mode == "A":
        text = "Masa modu"
    elif mode == "C":
        on = gate is None or not gate.enabled or gate.active
        text = "Kamera sana bakıyor · el faresi " + ("açık" if on else "kapalı")
    else:
        return "Kamera bekleniyor"
    if zoom is not None and zoom.enabled and mode == "C":
        text += f" · büyütme {zoom.level:+d}"
    return text
